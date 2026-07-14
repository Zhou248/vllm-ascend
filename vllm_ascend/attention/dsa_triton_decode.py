# SPDX-License-Identifier: Apache-2.0
"""Triton-backed DSA decode sparse attention for wiring back into dsa_v1.

This module wraps the Triton kernel that lives in
``vllm_ascend.ops.triton.dsa.decode_kernel`` so it can replace ascend-c's
`npu_kv_quant_sparse_attn_sharedkv` in `dsa_v1.py._forward_decode` for the
compress_ratio==4 branch.

Pipeline (per decode step):
  1. Dequantize the fp8 paged KV blocks actually referenced by the
     block_tables (ori + cmp) to bf16, with nope(448)+rope(64) in logical
     order. Only referenced blocks are dequantized (cheap for decode).
  2. Run the Triton bf16 sparse decode kernel (SWA + cmp sparse + sink).

Activated via VLLM_ASCEND_ENABLE_DSA_TRITON_DECODE=1; otherwise dsa_v1 keeps
the Ascend C implementation.

NOTE: this experimental version prioritizes correctness for generation
validation. A fused in-kernel dequantization can replace the current torch
implementation later.
"""
from __future__ import annotations

import importlib
import sys
from collections.abc import Callable
from pathlib import Path

import torch
from vllm.logger import logger
from vllm_ascend import envs


def _load_decode_kernel() -> Callable:
    """Resolve the DSA triton decode kernel entry point.

    By default the kernel is imported from the in-tree location
    ``vllm_ascend.ops.triton.dsa.decode_kernel``. The optional env var
    ``VLLM_ASCEND_DSA_TRITON_KERNEL_PATH`` overrides this by pointing at an
    external directory containing ``decode_c4_triton.py`` (kept for iterative
    kernel development outside the tree). Either way the returned callable has
    the same ``decode_dsa_triton`` signature, so callers are agnostic.
    """
    kernel_dir = envs.VLLM_ASCEND_DSA_TRITON_KERNEL_PATH
    if kernel_dir:
        kernel_path = Path(kernel_dir)
        if not (kernel_path / "decode_c4_triton.py").is_file():
            raise RuntimeError(f"decode_c4_triton.py was not found under {kernel_path}")
        path = str(kernel_path)
        if path not in sys.path:
            sys.path.insert(0, path)
        return importlib.import_module("decode_c4_triton").decode_dsa_triton
    # Default: in-tree kernel.
    from vllm_ascend.ops.triton.dsa.decode_kernel import decode_dsa_triton

    return decode_dsa_triton


# Per-token 640-byte pack offsets (mirror decode_c4_ref layout).
_ROPE_BYTES = 128
_NOPE_BYTES = 448
_SCALE_OFF = 576
_NUM_SCALES = 7
_GROUP_SIZE = 64

# DEBUG: track which compress_ratios have actually entered the triton path.
_TRITON_DECODE_CALLED: set = set()


def dequant_blocks_vec(
    kv_pool_fp8: torch.Tensor,
    block_table: torch.Tensor,
    max_logical: int | None = None,
) -> torch.Tensor:
    """Vectorized NPU dequantization without a CPU round trip.

    Same semantics as _dequant_used_blocks_npu but runs entirely on NPU:
      1. gather referenced physical blocks via index_select (one batched op)
      2. rope (64 bf16) at [0:128]   -> view as bfloat16
      3. nope (448 fp8 e4m3) at [128:576] -> native .to(bf16)
      4. 7 e8m0 scales at [576:583]  -> 2^(b-127), broadcast over 64-group
      5. nope *= scale; concat logical nope(448)+rope(64)

    Args/Returns: same as _dequant_used_blocks_npu.
    """
    bt = block_table.long().reshape(-1)
    if max_logical is not None:
        bt = bt[:max_logical]
    dev = kv_pool_fp8.device
    # index_select on NPU doesn't support fp8 dtype, so view the whole pool as
    # uint8 first (1 byte per element, same count), gather, then reinterpret.
    pool_u8 = kv_pool_fp8.view(torch.uint8)
    gathered = pool_u8.index_select(0, bt.to(dev))  # (N, block_size, 1, 640) uint8
    flat = gathered.reshape(-1, 640)  # (N*block_size, 640) uint8
    # The cache stores raw BF16 bytes. Reinterpreting these bytes as FP16
    # corrupts every RoPE value and makes decode diverge after the first token.
    rope = flat[:, :_ROPE_BYTES].contiguous().view(torch.bfloat16)

    # nope: 448 fp8 e4m3 at [128:576] -> view back as fp8, native cast to bf16
    nope_fp8 = flat[:, _ROPE_BYTES : _ROPE_BYTES + _NOPE_BYTES].contiguous().view(
        torch.float8_e4m3fn
    )  # (n_rows, 448)
    nope = nope_fp8.to(torch.bfloat16).to(torch.float32)

    # Seven E8M0 scales cover the 448 NOPE values in 64-element tiles.
    scale_b = flat[:, _SCALE_OFF:_SCALE_OFF + _NUM_SCALES].to(torch.int32)
    scales = torch.pow(2.0, (scale_b - 127).to(torch.float32))
    scale_exp = scales.repeat_interleave(_GROUP_SIZE, dim=1)
    nope = nope * scale_exp

    # logical order: nope(448) then rope(64)
    out = torch.cat([nope.to(torch.bfloat16), rope.to(torch.bfloat16)], dim=1)  # (n_rows, 512)
    block_size = gathered.shape[1]
    return out.reshape(gathered.shape[0], block_size, 512)


def remap_c4_sparse_token_indices(
    sparse_indices: torch.Tensor,
    block_table: torch.Tensor,
    cmp_token_count: int,
    block_size: int,
) -> tuple[list[int], torch.Tensor]:
    """Remap paged compressed-token indices into a compact local KV pool."""
    indices = sparse_indices.long().cpu()
    table = block_table.long().reshape(-1).cpu()
    active = indices[:, : min(cmp_token_count, indices.shape[1])]
    referenced = active[(active >= 0) & (active < cmp_token_count)]
    logical_pages = torch.div(referenced, block_size, rounding_mode="floor").unique()

    physical_pages: list[int] = []
    physical_to_local: dict[int, int] = {}
    for logical_page in logical_pages.tolist():
        physical_page = int(table[logical_page].item())
        if physical_page not in physical_to_local:
            physical_to_local[physical_page] = len(physical_pages)
            physical_pages.append(physical_page)

    valid = (indices >= 0) & (indices < cmp_token_count)
    logical_page = torch.div(indices.clamp_min(0), block_size, rounding_mode="floor")
    page_offset = indices.remainder(block_size)
    physical_page = torch.full_like(indices, -1)
    physical_page[valid] = table[logical_page[valid]]

    remapped = torch.full_like(indices, -1)
    for physical, local in physical_to_local.items():
        page_mask = valid & (physical_page == physical)
        remapped[page_mask] = local * block_size + page_offset[page_mask]
    return physical_pages, remapped


def _dequant_ori_window_batch(
    swa_kv_cache: torch.Tensor,
    ori_block_table: torch.Tensor,   # (num_seqs, max_logical) int32
    seq_lens: list[int],
    ori_block_size: int,
    ori_window_size: int,
    head_dim: int,
    q: torch.Tensor,
) -> torch.Tensor:
    """Batch-dequantize each seq's SWA window blocks into a per-seq local pool.

    Returns (num_seqs, max_ori_blocks, ori_block_size, head_dim) bf16.
    Block 0 of each seq's pool == the first block overlapping its SWA window;
    padding slots (seqs with fewer window blocks) reference physical block 0,
    which is safe because the kernel masks them out via offs_tok >= kv_len.
    """
    num_seqs = ori_block_table.shape[0]
    # Per-seq window block range. swa_start = max(0, kv_len - window).
    first_blocks = []
    n_blocks_list = []
    for s in range(num_seqs):
        kv_len = seq_lens[s]
        swa_start = max(0, kv_len - ori_window_size)
        first_block = swa_start // ori_block_size
        end_block = (kv_len + ori_block_size - 1) // ori_block_size
        first_blocks.append(first_block)
        n_blocks_list.append(max(0, end_block - first_block))
    max_ori_blocks = max(n_blocks_list) if n_blocks_list else 1
    if max_ori_blocks == 0:
        max_ori_blocks = 1

    # Build (num_seqs, max_ori_blocks) physical-block-id matrix on NPU.
    bt_dev = ori_block_table.long()
    phys_matrix = torch.zeros(
        (num_seqs, max_ori_blocks), dtype=torch.long, device=bt_dev.device
    )
    for s in range(num_seqs):
        fb = first_blocks[s]
        nb = n_blocks_list[s]
        if nb > 0:
            phys_matrix[s, :nb] = bt_dev[s, fb : fb + nb]
    # One batched gather + dequant over the flattened matrix.
    flat_phys = phys_matrix.reshape(-1)
    ori_kv_flat = dequant_blocks_vec(swa_kv_cache, flat_phys)  # (N*max_ori, block_size, 512)
    ori_kv_bf16 = ori_kv_flat.view(num_seqs, max_ori_blocks, ori_block_size, head_dim)
    return ori_kv_bf16.contiguous()


def _remap_c4_sparse_per_seq(
    cmp_sparse_indices: torch.Tensor,  # (num_q_tokens, 1, topk) or (num_q_tokens, topk)
    cmp_block_table: torch.Tensor,     # (num_seqs, max_cmp_logical) int32
    seq_lens: list[int],
    cmp_block_size: int,
    compress_ratio: int,
    compress_kv_cache: torch.Tensor,
    head_dim: int,
    q: torch.Tensor,
    tokens_per_seq: int,
) -> tuple[torch.Tensor | None, torch.Tensor]:
    """Per-request remap of c4 sparse indices into per-request local cmp pools.

    The cmp KV pool is per-request (shared by all spec tokens of a request,
    since they share kv_len / cmp_token_count). The sparse indices are per
    query token (each spec token selects its own cmp tokens).

    Args:
      tokens_per_seq: query tokens per request (1 = plain decode, >1 = spec decode).

    Returns:
      cmp_kv_bf16: (num_seqs, max_cmp_blocks, cmp_block_size, head_dim) bf16
      csi: (num_q_tokens, topk) int32 remapped local token indices (-1 = unused)
    """
    num_seqs = cmp_block_table.shape[0]
    num_q_tokens = cmp_sparse_indices.shape[0]
    csi_cpu = cmp_sparse_indices.reshape(num_q_tokens, -1).long().cpu()
    bt_cpu = cmp_block_table.long().reshape(num_seqs, -1).cpu()
    topk = csi_cpu.shape[1]

    # Pass 1: per request, gather all referenced physical cmp pages (across all
    # its query tokens) and build a physical->local map; build the local pool.
    phys_per_seq: list[list[int]] = []
    phys_to_local_per_seq: list[dict[int, int]] = []
    for s in range(num_seqs):
        cmp_token_count = seq_lens[s] // compress_ratio
        # All query tokens of this request share the same cmp_token_count bound.
        tok_slice = csi_cpu[s * tokens_per_seq : (s + 1) * tokens_per_seq].reshape(-1)
        referenced = tok_slice[(tok_slice >= 0) & (tok_slice < cmp_token_count)]
        logical_pages = torch.div(
            referenced, cmp_block_size, rounding_mode="floor"
        ).unique().tolist()

        phys_to_local: dict[int, int] = {}
        physical_pages: list[int] = []
        for lp in logical_pages:
            pp = int(bt_cpu[s, lp].item())
            if pp not in phys_to_local:
                phys_to_local[pp] = len(physical_pages)
                physical_pages.append(pp)
        phys_per_seq.append(physical_pages)
        phys_to_local_per_seq.append(phys_to_local)

    max_cmp_blocks = max((len(p) for p in phys_per_seq), default=0)
    if max_cmp_blocks == 0:
        max_cmp_blocks = 1

    # Build per-request cmp pool: gather each request's physical pages.
    cmp_pool = torch.zeros(
        (num_seqs, max_cmp_blocks, cmp_block_size, head_dim),
        dtype=q.dtype, device=q.device,
    )
    for s in range(num_seqs):
        pages = phys_per_seq[s]
        if pages:
            phys_t = torch.tensor(pages, dtype=torch.long, device=q.device)
            deq = dequant_blocks_vec(compress_kv_cache, phys_t)  # (len(pages), block_size, 512)
            cmp_pool[s, : len(pages)].copy_(deq)

    # Pass 2: per query token, remap its sparse indices into its request's pool.
    csi_out = torch.full((num_q_tokens, topk), -1, dtype=torch.int32)
    for t in range(num_q_tokens):
        s = t // tokens_per_seq
        cmp_token_count = seq_lens[s] // compress_ratio
        phys_to_local = phys_to_local_per_seq[s]
        indices = csi_cpu[t]
        valid = (indices >= 0) & (indices < cmp_token_count)
        logical_page = torch.div(indices.clamp_min(0), cmp_block_size, rounding_mode="floor")
        page_offset = indices.remainder(cmp_block_size)
        physical_page = torch.full_like(indices, -1)
        physical_page[valid] = bt_cpu[s, logical_page[valid]]

        remapped = torch.full_like(indices, -1)
        for physical, local in phys_to_local.items():
            page_mask = valid & (physical_page == physical)
            remapped[page_mask] = local * cmp_block_size + page_offset[page_mask]
        csi_out[t] = remapped.to(torch.int32)

    return cmp_pool.contiguous(), csi_out.to(q.device).contiguous()


def _dequant_c128_batch(
    compress_kv_cache: torch.Tensor,
    cmp_block_table: torch.Tensor,   # (num_seqs, max_cmp_logical) int32
    seq_lens: list[int],
    cmp_block_size: int,
    compress_ratio: int,
    head_dim: int,
    q: torch.Tensor,
) -> torch.Tensor:
    """Batch-dequantize each seq's first N cmp blocks (c128 full scan).

    Returns (num_seqs, max_cmp_blocks, cmp_block_size, head_dim) bf16.
    """
    num_seqs = cmp_block_table.shape[0]
    n_blocks_list = []
    for s in range(num_seqs):
        n_cmp_tokens = seq_lens[s] // compress_ratio
        n_cmp_blocks = (n_cmp_tokens + cmp_block_size - 1) // cmp_block_size
        n_blocks_list.append(n_cmp_blocks)
    max_cmp_blocks = max(n_blocks_list) if n_blocks_list else 1
    if max_cmp_blocks == 0:
        max_cmp_blocks = 1

    bt_dev = cmp_block_table.long()
    phys_matrix = torch.zeros(
        (num_seqs, max_cmp_blocks), dtype=torch.long, device=bt_dev.device
    )
    for s in range(num_seqs):
        nb = n_blocks_list[s]
        if nb > 0:
            phys_matrix[s, :nb] = bt_dev[s, :nb]
    flat_phys = phys_matrix.reshape(-1)
    deq = dequant_blocks_vec(compress_kv_cache, flat_phys)
    return deq.view(num_seqs, max_cmp_blocks, cmp_block_size, head_dim).contiguous()


def triton_decode_dsa(
    q: torch.Tensor,                       # (num_seqs, n_heads, 512) bf16
    swa_kv_cache: torch.Tensor,            # fp8 pack pool (ori/SWA KV)
    compress_kv_cache: torch.Tensor | None,  # fp8 pack pool (cmp KV), None for dense
    cmp_sparse_indices: torch.Tensor | None, # (num_seqs, 1, topk) int32, c4 only
    ori_block_table: torch.Tensor,         # (num_seqs, max_ori) int32
    cmp_block_table: torch.Tensor | None,  # (num_seqs, max_cmp) int32, None for dense
    seqused_kv: torch.Tensor,             # (num_seqs,) int32
    sinks: torch.Tensor,                  # (n_heads,) float32
    softmax_scale: float,
    compress_ratio: int,                  # 0/1=dense, 4=c4, 128=c128
    ori_block_size: int,
    cmp_block_size: int | None,
    ori_window_size: int,
) -> torch.Tensor:
    """Drop-in triton replacement for the decode attn_op call (all ratios).

    Supports multiple decode seqs per call, including speculative decode
    (MTP): a request may carry several query tokens (real + draft). The number
    of requests is taken from ``seqused_kv`` (per-request KV lengths); q's
    first dim is the total query-token count (``num_seqs * tokens_per_seq``).
    All query tokens of a request attend to the same kv_len; c4 sparse indices
    are per query token.

    Returns (num_q_tokens, n_heads, 512) bf16 attention output.
    """
    num_seqs = seqused_kv.numel()
    num_q_tokens = q.shape[0]
    head_dim = q.shape[-1]
    if num_q_tokens % num_seqs != 0:
        raise ValueError(
            f"triton_decode_dsa: q rows ({num_q_tokens}) must be a multiple of "
            f"num_seqs ({num_seqs})"
        )
    tokens_per_seq = num_q_tokens // num_seqs
    # DEBUG: confirm the triton kernel is actually executed (once per ratio).
    global _TRITON_DECODE_CALLED
    if compress_ratio not in _TRITON_DECODE_CALLED:
        logger.info(
            "[DSA-TRITON] triton_decode_dsa ENTERED: compress_ratio=%s "
            "num_seqs=%s tokens_per_seq=%s (kernel is live, not ascend-c)",
            compress_ratio, num_seqs, tokens_per_seq,
        )
        _TRITON_DECODE_CALLED.add(compress_ratio)
    if ori_block_table.shape[0] != num_seqs:
        raise ValueError(
            f"triton_decode_dsa: ori_block_table rows ({ori_block_table.shape[0]}) must match "
            f"num_seqs ({num_seqs})"
        )
    if compress_ratio == 4 and (
        compress_kv_cache is None or cmp_block_table is None or cmp_sparse_indices is None
    ):
        raise ValueError("compress_ratio=4 requires compressed KV, its block table, and sparse indices")
    if compress_ratio == 128 and (compress_kv_cache is None or cmp_block_table is None):
        raise ValueError("compress_ratio=128 requires compressed KV and its block table")

    # Single host sync of all per-seq KV lengths (one D2H transfer).
    seq_lens = seqused_kv.tolist()

    # 1. dequant each seq's SWA window ori blocks into a per-seq local pool.
    ori_kv_bf16 = _dequant_ori_window_batch(
        swa_kv_cache, ori_block_table, seq_lens,
        ori_block_size, ori_window_size, head_dim, q,
    )

    # 2. select cmp_mode + dequant cmp blocks if applicable
    if compress_ratio == 4:
        cmp_mode = 0  # sparse
        assert cmp_block_size is not None
        # cmp_sparse_indices may be (num_q_tokens, 1, topk); flatten to per-token.
        if cmp_sparse_indices.dim() == 3:
            csi_in = cmp_sparse_indices.reshape(num_q_tokens, -1)
        else:
            csi_in = cmp_sparse_indices
        cmp_kv_bf16, csi = _remap_c4_sparse_per_seq(
            csi_in, cmp_block_table, seq_lens,
            cmp_block_size, compress_ratio, compress_kv_cache, head_dim, q,
            tokens_per_seq,
        )
        cmp_ratio = 4
    elif compress_ratio == 128:
        cmp_mode = 1  # full
        assert cmp_block_size is not None
        cmp_kv_bf16 = _dequant_c128_batch(
            compress_kv_cache, cmp_block_table, seq_lens,
            cmp_block_size, compress_ratio, head_dim, q,
        )
        csi = None
        cmp_ratio = 128
    else:  # <= 1, dense
        cmp_mode = 2
        cmp_kv_bf16 = None
        csi = None
        cmp_ratio = 1

    seqused_t = seqused_kv.to(torch.int32).to(q.device).contiguous()

    decode_dsa_triton = _load_decode_kernel()
    out = decode_dsa_triton(
        q.contiguous(),
        ori_kv_bf16,
        cmp_kv_bf16,
        csi,
        seqused_t,
        sinks.to(torch.float32).contiguous(),
        softmax_scale,
        cmp_mode=cmp_mode,
        cmp_ratio=cmp_ratio,
        block_size=ori_block_size,
        cmp_block_size=cmp_block_size or ori_block_size,
        ori_window_size=ori_window_size,
    )
    return out


def triton_decode_c4(**kwargs):
    """Back-compat wrapper for callers that still use the c4-specific name."""
    kwargs.setdefault("compress_ratio", 4)
    return triton_decode_dsa(**kwargs)


def use_triton_decode() -> bool:
    """Whether to route decode through the Triton kernel."""
    return envs.VLLM_ASCEND_ENABLE_DSA_TRITON_DECODE
