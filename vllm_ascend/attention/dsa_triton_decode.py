# SPDX-License-Identifier: Apache-2.0
"""Triton-backed DSA decode sparse attention for wiring back into dsa_v1.

This module wraps the in-tree Triton kernel so it can replace ascend-c's
`npu_kv_quant_sparse_attn_sharedkv` in `dsa_v1.py._forward_decode` for SWA,
C4 sparse-compressed, and C128 compressed layers.

Pipeline (per decode step):
  1. Build fixed-shape device-side pools and dequantize the referenced fp8
     paged KV rows to bf16, with nope(448)+rope(64) in logical order.
  2. Run the Triton bf16 sparse decode kernel (SWA + cmp sparse + sink).

Activated via VLLM_ASCEND_ENABLE_DSA_TRITON_DECODE=1; otherwise dsa_v1 keeps
the Ascend C implementation.

NOTE: this experimental version prioritizes correctness for generation
validation. A fused in-kernel dequantization can replace the current torch
implementation later.
"""
from __future__ import annotations

import torch
from vllm_ascend import envs


def _load_decode_kernel():
    """Resolve the in-tree graph-capturable DSA Triton kernel."""
    from vllm_ascend.ops.triton.dsa.decode_kernel import decode_dsa_triton

    return decode_dsa_triton


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


def dequant_token_rows_vec(
    kv_pool_fp8: torch.Tensor,
    physical_blocks: torch.Tensor,
    block_offsets: torch.Tensor,
) -> torch.Tensor:
    """Gather and dequantize individual packed KV token rows on device."""
    pool_u8 = kv_pool_fp8.view(torch.uint8)
    flat = pool_u8[physical_blocks.long(), block_offsets.long()].reshape(-1, 640)
    rope = flat[:, :_ROPE_BYTES].contiguous().view(torch.bfloat16)
    nope_fp8 = flat[:, _ROPE_BYTES : _ROPE_BYTES + _NOPE_BYTES].contiguous().view(
        torch.float8_e4m3fn
    )
    nope = nope_fp8.to(torch.bfloat16).to(torch.float32)
    scale_b = flat[:, _SCALE_OFF:_SCALE_OFF + _NUM_SCALES].to(torch.int32)
    scales = torch.pow(2.0, (scale_b - 127).to(torch.float32))
    nope = nope * scales.repeat_interleave(_GROUP_SIZE, dim=1)
    return torch.cat([nope.to(torch.bfloat16), rope], dim=1)


def _build_token_to_seq(query_start_loc: torch.Tensor, num_q_tokens: int) -> torch.Tensor:
    """Build a graph-safe token-to-request map from cumulative query lengths."""
    token_ids = torch.arange(num_q_tokens, dtype=torch.int32, device=query_start_loc.device)
    return (token_ids.unsqueeze(1) >= query_start_loc[1:].unsqueeze(0)).sum(dim=1)


def _build_causal_ends(
    seqused_kv: torch.Tensor,
    query_start_loc: torch.Tensor,
    num_q_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    token_to_seq = _build_token_to_seq(query_start_loc, num_q_tokens).long()
    token_ids = torch.arange(num_q_tokens, dtype=torch.int32, device=query_start_loc.device)
    query_lens = query_start_loc[token_to_seq + 1] - query_start_loc[token_to_seq]
    query_offsets = token_ids - query_start_loc[token_to_seq]
    causal_ends = seqused_kv[token_to_seq] - query_lens + query_offsets + 1
    return token_to_seq, causal_ends


def _prepare_ori_pool_graph(
    swa_kv_cache: torch.Tensor,
    ori_block_table: torch.Tensor,
    seqused_kv: torch.Tensor,
    query_start_loc: torch.Tensor,
    ori_block_size: int,
    ori_window_size: int,
    max_query_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    query_lens = query_start_loc[1:] - query_start_loc[:-1]
    first_query_end = seqused_kv - query_lens + 1
    first_tokens = torch.clamp(first_query_end - ori_window_size, min=0)
    first_blocks = torch.div(first_tokens, ori_block_size, rounding_mode="floor")
    max_blocks = (ori_window_size + max_query_tokens + ori_block_size - 2) // ori_block_size + 1
    offsets = torch.arange(max_blocks, dtype=torch.int64, device=ori_block_table.device)
    logical_blocks = first_blocks.long().unsqueeze(1) + offsets.unsqueeze(0)
    max_table_index = ori_block_table.shape[1] - 1
    logical_blocks = logical_blocks.clamp(max=max_table_index)
    physical_blocks = torch.gather(ori_block_table.long(), 1, logical_blocks)
    dequant = dequant_blocks_vec(swa_kv_cache, physical_blocks.reshape(-1))
    pool = dequant.view(seqused_kv.numel(), max_blocks, ori_block_size, -1)
    return pool.contiguous(), first_blocks.to(torch.int32).contiguous()


def _prepare_c4_pool_graph(
    compress_kv_cache: torch.Tensor,
    cmp_block_table: torch.Tensor,
    cmp_sparse_indices: torch.Tensor,
    token_to_seq: torch.Tensor,
    causal_ends: torch.Tensor,
    cmp_block_size: int,
    compress_ratio: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_q_tokens = cmp_sparse_indices.shape[0]
    indices = cmp_sparse_indices.reshape(num_q_tokens, -1).to(torch.int64)
    cmp_token_count = torch.div(causal_ends, compress_ratio, rounding_mode="floor")
    sparse_slots = torch.arange(indices.shape[1], device=indices.device).unsqueeze(0)
    valid = (
        (indices >= 0)
        & (indices < cmp_token_count.unsqueeze(1))
        & (sparse_slots < cmp_token_count.unsqueeze(1))
    )
    safe_indices = torch.where(valid, indices, torch.zeros_like(indices))
    logical_blocks = torch.div(safe_indices, cmp_block_size, rounding_mode="floor")
    block_offsets = safe_indices.remainder(cmp_block_size)
    physical_blocks = torch.gather(
        cmp_block_table.long().index_select(0, token_to_seq.long()),
        1,
        logical_blocks,
    )
    dequant = dequant_token_rows_vec(
        compress_kv_cache, physical_blocks.reshape(-1), block_offsets.reshape(-1)
    )
    pool = dequant.view(num_q_tokens, indices.shape[1], -1)
    remapped = torch.where(valid, sparse_slots, -1)
    return pool.contiguous(), remapped.to(torch.int32).contiguous()


def _prepare_c128_pool_graph(
    compress_kv_cache: torch.Tensor,
    cmp_block_table: torch.Tensor,
) -> torch.Tensor:
    physical_blocks = cmp_block_table.long().reshape(-1)
    dequant = dequant_blocks_vec(compress_kv_cache, physical_blocks)
    return dequant.view(
        cmp_block_table.shape[0], cmp_block_table.shape[1], compress_kv_cache.shape[1], -1
    ).contiguous()


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
    query_start_loc: torch.Tensor,
    max_query_tokens: int,
) -> torch.Tensor:
    """Drop-in triton replacement for the decode attn_op call (all ratios).

    Supports ragged multi-request decode and MTP. ``query_start_loc`` maps each
    query token to its request and the kernel applies a per-token causal end.

    Returns (num_q_tokens, n_heads, 512) bf16 attention output.
    """
    num_seqs = seqused_kv.numel()
    num_q_tokens = q.shape[0]
    if num_q_tokens == 0 or num_seqs == 0:
        return torch.empty_like(q)
    if query_start_loc.numel() != num_seqs + 1:
        raise ValueError(
            "triton_decode_dsa: query_start_loc must contain num_seqs + 1 entries"
        )
    if ori_block_table.shape[0] != num_seqs:
        raise ValueError(
            f"triton_decode_dsa: ori_block_table rows ({ori_block_table.shape[0]}) must match "
            f"num_seqs ({num_seqs})"
        )
    if max_query_tokens < 1:
        raise ValueError("triton_decode_dsa: max_query_tokens must be positive")
    if compress_ratio == 4 and (
        compress_kv_cache is None or cmp_block_table is None or cmp_sparse_indices is None
    ):
        raise ValueError("compress_ratio=4 requires compressed KV, its block table, and sparse indices")
    if compress_ratio == 128 and (compress_kv_cache is None or cmp_block_table is None):
        raise ValueError("compress_ratio=128 requires compressed KV and its block table")
    if compress_ratio > 1 and cmp_block_table.shape[0] != num_seqs:
        raise ValueError(
            f"triton_decode_dsa: cmp_block_table rows ({cmp_block_table.shape[0]}) "
            f"must match num_seqs ({num_seqs})"
        )

    ori_kv_bf16, ori_first_block = _prepare_ori_pool_graph(
        swa_kv_cache, ori_block_table, seqused_kv, query_start_loc,
        ori_block_size, ori_window_size, max_query_tokens,
    )
    token_to_seq, causal_ends = _build_causal_ends(
        seqused_kv, query_start_loc, num_q_tokens
    )

    # 2. select cmp_mode + dequant cmp blocks if applicable
    if compress_ratio == 4:
        cmp_mode = 0  # sparse
        assert cmp_block_size is not None
        cmp_kv_bf16, csi = _prepare_c4_pool_graph(
            compress_kv_cache, cmp_block_table, cmp_sparse_indices,
            token_to_seq, causal_ends, cmp_block_size, compress_ratio,
        )
        cmp_ratio = 4
    elif compress_ratio == 128:
        cmp_mode = 1  # full
        assert cmp_block_size is not None
        cmp_kv_bf16 = _prepare_c128_pool_graph(compress_kv_cache, cmp_block_table)
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
        token_to_seq.to(torch.int32).contiguous(),
        causal_ends.to(torch.int32).contiguous(),
        ori_first_block,
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
