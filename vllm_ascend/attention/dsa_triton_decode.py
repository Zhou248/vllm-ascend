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

Activated via env var VLLM_DSA_USE_TRITON=1; otherwise dsa_v1 keeps ascend-c.

NOTE: first version prioritizes correctness for generation validation, not
performance. fp8 dequant is done on host via torch ops; a fused in-kernel
dequant can come later.
"""
from __future__ import annotations

import os

import torch

# Reuse the validated dequant + kernel from the triton-ascend-kernels project.
_TRITON_KERNELS_DIR = "/home/z00909726/dsk-quant/triton-ascend-kernels/attention"

import sys

if _TRITON_KERNELS_DIR not in sys.path:
    sys.path.insert(0, _TRITON_KERNELS_DIR)

from decode_c4_ref import dequant_kv_block  # noqa: E402
from decode_c4_triton import decode_dsa_triton  # noqa: E402


def _dequant_used_blocks_npu(
    kv_pool_fp8: torch.Tensor,
    block_table: torch.Tensor,
    max_logical: int | None = None,
) -> torch.Tensor:
    """Dequantize the physical blocks referenced by block_table.

    Args:
        kv_pool_fp8: (num_blocks, block_size, 1, 640) fp8 e4m3 pack.
        block_table: (..., max_logical) int32; logical->physical block ids.
        max_logical: only dequant the first N logical blocks (the ones that
            actually hold valid KV tokens). None = dequant all in the table.
            For decode, N = ceil(seqused_kv / block_size) for ori/c128-cmp
            (sequential layout), giving a large speedup over dequanting the
            whole padded table.

    Returns:
        (N, block_size, 512) bf16, where N = max_logical or table length.
        Row i = dequant of physical block block_table.flatten()[i].
    """
    bt = block_table.long().reshape(-1)
    if max_logical is not None:
        bt = bt[:max_logical]
    pool_u8 = kv_pool_fp8.view(torch.uint8)
    blocks = []
    for i in range(bt.shape[0]):
        phys = int(bt[i].item())
        blk_u8 = pool_u8[phys]  # (block_size, 1, 640)
        blocks.append(dequant_kv_block(blk_u8.cpu().numpy()))
    return torch.stack(blocks, dim=0).to(kv_pool_fp8.device)  # (N, block_size, 512)


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
    block_size: int = 128,
) -> torch.Tensor:
    """Drop-in triton replacement for the decode attn_op call (all ratios).

    Returns (num_seqs, n_heads, 512) bf16 attention output.
    """
    seqused_kv_val = int(seqused_kv[0].item())
    # ori (SWA) blocks holding valid KV tokens: sequential, first N only.
    n_ori_blocks = (seqused_kv_val + block_size - 1) // block_size

    # 1. dequant referenced ori (SWA) blocks — only the N that hold valid KV
    ori_kv_bf16 = _dequant_used_blocks_npu(swa_kv_cache, ori_block_table, max_logical=n_ori_blocks)
    ori_bt_local = torch.arange(ori_kv_bf16.shape[0], dtype=torch.int32, device=q.device)

    # 2. select cmp_mode + dequant cmp blocks if applicable
    if compress_ratio == 4:
        cmp_mode = 0  # sparse
        # c4: cmp_sparse_indices may reference arbitrary logical blocks, so we
        # dequant only the unique referenced blocks (small set). Do all index
        # bookkeeping on CPU (small tensors), dequant referenced physical blocks.
        cmp_bt_orig = cmp_block_table.long().reshape(-1).cpu()
        csi_cpu = cmp_sparse_indices.reshape(q.shape[0], -1).long().cpu()
        referenced = csi_cpu.reshape(-1)
        referenced = referenced[referenced >= 0].unique()
        local_rows = []  # physical block ids, deduplicated
        phys_to_local = {}
        for lb in referenced.tolist():
            phys = int(cmp_bt_orig[lb].item())
            if phys not in phys_to_local:
                phys_to_local[phys] = len(local_rows)
                local_rows.append(phys)
        pool_u8 = compress_kv_cache.view(torch.uint8)
        blocks = [dequant_kv_block(pool_u8[p].cpu().numpy()) for p in local_rows]
        cmp_kv_bf16 = torch.stack(blocks, dim=0).to(q.device)
        cmp_bt_local = torch.arange(cmp_kv_bf16.shape[0], dtype=torch.int32, device=q.device)
        # remap csi: logical -> physical (via cmp_bt_orig) -> local index
        phys_of_logical = cmp_bt_orig[csi_cpu]  # (S, topk) physical ids
        local_map = torch.full(
            (int(cmp_bt_orig.max().item()) + 1,), -1, dtype=torch.long
        )
        for phys, local in phys_to_local.items():
            local_map[phys] = local
        csi = local_map[phys_of_logical].to(torch.int32).to(q.device).contiguous()
        cmp_ratio = 4
    elif compress_ratio == 128:
        cmp_mode = 1  # full
        # c128: cmp tokens = seqused_kv / 128, sequential -> first N cmp blocks.
        n_cmp_tokens = (seqused_kv_val + compress_ratio - 1) // compress_ratio
        n_cmp_blocks = (n_cmp_tokens + block_size - 1) // block_size
        cmp_kv_bf16 = _dequant_used_blocks_npu(
            compress_kv_cache, cmp_block_table, max_logical=n_cmp_blocks
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
        block_size=block_size,
        cmp_block_size=block_size,
    )
    return out


def triton_decode_c4(**kwargs):
    """Back-compat wrapper for callers that still use the c4-specific name."""
    kwargs.setdefault("compress_ratio", 4)
    return triton_decode_dsa(**kwargs)


def use_triton_decode() -> bool:
    """Whether to route decode through the Triton kernel."""
    return os.getenv("VLLM_DSA_USE_TRITON", "0") == "1"
