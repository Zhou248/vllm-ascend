# SPDX-License-Identifier: Apache-2.0
"""Triton-backed decode_c4 sparse attention for wiring back into dsa_v1.

This module wraps the Triton kernel (originally developed in
triton-ascend-kernels/attention) so that it can replace ascend-c's
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

from vllm_ascend import envs


def _load_decode_kernel() -> Callable:
    kernel_dir = envs.VLLM_ASCEND_DSA_TRITON_KERNEL_PATH
    if not kernel_dir:
        raise RuntimeError(
            "VLLM_ASCEND_DSA_TRITON_KERNEL_PATH must point to the directory "
            "containing decode_c4_triton.py"
        )
    kernel_path = Path(kernel_dir)
    if not (kernel_path / "decode_c4_triton.py").is_file():
        raise RuntimeError(f"decode_c4_triton.py was not found under {kernel_path}")
    path = str(kernel_path)
    if path not in sys.path:
        sys.path.insert(0, path)
    return importlib.import_module("decode_c4_triton").decode_dsa_triton


# Per-token 640-byte pack offsets (mirror decode_c4_ref layout).
_ROPE_BYTES = 128
_NOPE_BYTES = 448
_SCALE_OFF = 576
_NUM_SCALES = 7
_GROUP_SIZE = 64


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


def triton_decode_dsa(
    q: torch.Tensor,                       # (num_seqs, n_heads, 512) bf16
    swa_kv_cache: torch.Tensor,            # fp8 pack pool (ori/SWA KV)
    compress_kv_cache: torch.Tensor | None,  # fp8 pack pool (cmp KV), None for dense
    cmp_sparse_indices: torch.Tensor | None, # (num_seqs, 1, topk) int32, c4 only
    ori_block_table: torch.Tensor,         # (1, max_ori) int32
    cmp_block_table: torch.Tensor | None,  # (1, max_cmp) int32, None for dense
    seqused_kv: torch.Tensor,             # (1,) int32
    sinks: torch.Tensor,                  # (n_heads,) float32
    softmax_scale: float,
    compress_ratio: int,                  # 0/1=dense, 4=c4, 128=c128
    ori_block_size: int,
    cmp_block_size: int | None,
    ori_window_size: int,
) -> torch.Tensor:
    """Drop-in triton replacement for the decode attn_op call (all ratios).

    Returns (num_seqs, n_heads, 512) bf16 attention output.
    """
    if q.shape[0] != 1 or seqused_kv.numel() != 1:
        raise NotImplementedError(
            "The experimental DSA Triton decode wrapper currently supports "
            "exactly one sequence per call"
        )
    if ori_block_table.shape[0] != 1:
        raise NotImplementedError(
            "The experimental DSA Triton decode wrapper requires one block-table row"
        )
    if compress_ratio == 4 and (
        compress_kv_cache is None or cmp_block_table is None or cmp_sparse_indices is None
    ):
        raise ValueError("compress_ratio=4 requires compressed KV, its block table, and sparse indices")
    if compress_ratio == 128 and (compress_kv_cache is None or cmp_block_table is None):
        raise ValueError("compress_ratio=128 requires compressed KV and its block table")
    seqused_kv_val = int(seqused_kv[0].item())
    swa_start = max(0, seqused_kv_val - ori_window_size)
    first_ori_block = swa_start // ori_block_size
    end_ori_block = (seqused_kv_val + ori_block_size - 1) // ori_block_size
    ori_block_table_window = ori_block_table[:, first_ori_block:end_ori_block]

    # 1. dequant referenced ori (SWA) blocks — only the N that hold valid KV,
    #    vectorized on NPU (no CPU roundtrip).
    ori_kv_bf16 = dequant_blocks_vec(swa_kv_cache, ori_block_table_window)
    ori_bt_local = torch.arange(ori_kv_bf16.shape[0], dtype=torch.int32, device=q.device)

    # 2. select cmp_mode + dequant cmp blocks if applicable
    if compress_ratio == 4:
        cmp_mode = 0  # sparse
        # C4 indices address compressed tokens. Dequantize only the physical
        # pages containing selected tokens, preserving each in-page offset.
        assert cmp_block_size is not None
        csi_cpu = cmp_sparse_indices.reshape(q.shape[0], -1).long().cpu()
        cmp_token_count = seqused_kv_val // compress_ratio
        physical_pages, remapped_csi = remap_c4_sparse_token_indices(
            csi_cpu, cmp_block_table, cmp_token_count, cmp_block_size
        )
        if physical_pages:
            phys_t = torch.tensor(physical_pages, dtype=torch.long, device=compress_kv_cache.device)
            cmp_kv_bf16 = dequant_blocks_vec(compress_kv_cache, phys_t.reshape(1, -1))
        else:
            cmp_kv_bf16 = torch.zeros(
                (1, cmp_block_size, q.shape[-1]), dtype=q.dtype, device=q.device
            )
        cmp_bt_local = torch.arange(cmp_kv_bf16.shape[0], dtype=torch.int32, device=q.device)
        csi = remapped_csi.to(torch.int32).to(q.device).contiguous()
        cmp_ratio = 4
    elif compress_ratio == 128:
        cmp_mode = 1  # full
        # c128: cmp tokens = seqused_kv / 128, sequential -> first N cmp blocks.
        assert cmp_block_size is not None
        n_cmp_tokens = seqused_kv_val // compress_ratio
        n_cmp_blocks = (n_cmp_tokens + cmp_block_size - 1) // cmp_block_size
        if n_cmp_blocks:
            cmp_kv_bf16 = dequant_blocks_vec(
                compress_kv_cache, cmp_block_table, max_logical=n_cmp_blocks
            )
        else:
            cmp_kv_bf16 = torch.zeros(
                (1, cmp_block_size, q.shape[-1]), dtype=q.dtype, device=q.device
            )
        cmp_bt_local = torch.arange(cmp_kv_bf16.shape[0], dtype=torch.int32, device=q.device)
        csi = None
        cmp_ratio = 128
    else:  # <= 1, dense
        cmp_mode = 2
        cmp_kv_bf16 = None
        cmp_bt_local = None
        csi = None
        cmp_ratio = 1

    seqused_t = torch.tensor([int(seqused_kv[0].item())], dtype=torch.int32, device=q.device)

    decode_dsa_triton = _load_decode_kernel()
    out = decode_dsa_triton(
        q.contiguous(),
        ori_kv_bf16,
        ori_bt_local,
        cmp_kv_bf16,
        cmp_bt_local,
        csi,
        seqused_t,
        sinks.to(torch.float32).contiguous(),
        softmax_scale,
        cmp_mode=cmp_mode,
        cmp_ratio=cmp_ratio,
        block_size=ori_block_size,
        cmp_block_size=cmp_block_size or ori_block_size,
        ori_window_size=ori_window_size,
        ori_token_base=first_ori_block * ori_block_size,
    )
    return out


def triton_decode_c4(**kwargs):
    """Back-compat wrapper for callers that still use the c4-specific name."""
    kwargs.setdefault("compress_ratio", 4)
    return triton_decode_dsa(**kwargs)


def use_triton_decode() -> bool:
    """Whether to route decode through the Triton kernel."""
    return envs.VLLM_ASCEND_ENABLE_DSA_TRITON_DECODE
