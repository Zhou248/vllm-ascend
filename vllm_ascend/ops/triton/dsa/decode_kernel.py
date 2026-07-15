# SPDX-License-Identifier: Apache-2.0
"""Triton kernel for DeepSeek-V4 DSA decode sparse attention (bf16 KV path).

Strategy (two-stage, dequant happens in vllm_ascend.attention.dsa_triton_decode):
  Stage A (device, torch): batch-dequantize the fp8 paged KV pools (ori + cmp)
    into graph-capturable bf16 KV tensors, with nope(448)+rope(64) in logical
    order.
  Stage B (device, this kernel): bf16 sparse attention = SWA over the per-seq
    ori KV pool + sparse-selected compressed KV + per-head sink.

This keeps the Triton kernel a plain bf16 attention (no in-kernel fp8 work),
which is far easier to compile/debug on Triton-Ascend. Once the bf16 path is
validated end-to-end (generation in dsa_v1), the fp8 dequant can optionally
be folded into the kernel for performance.

Supports ragged multi-request decode and speculative decode (MTP). Grid is
(num_q_tokens, num_q_heads), one program per (query token, head). The cumulative
query lengths map tokens to requests, and each token gets its own causal KV end.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _decode_c4_bf16_kernel(
    Q_ptr,                 # (num_q_tokens, num_q_heads, HEAD_DIM) bf16
    # OriKV is per request. C4 CmpKV is per token; C128 CmpKV is per request.
    OriKV_ptr,             # (num_seqs, MAX_ORI_BLOCKS, BLOCK_SIZE, HEAD_DIM) bf16
    CmpKV_ptr,             # c4: (T, TOPK, D); c128: (S, MAX_BLOCKS, BS, D)
    CmpSparseIdx_ptr,      # (num_q_tokens, TOPK) int32, -1 = unused (per query token)
    TokenToSeq_ptr,        # (num_q_tokens,) int32
    CausalEnds_ptr,        # (num_q_tokens,) int32, exclusive KV end
    OriFirstBlock_ptr,     # (num_seqs,) int32, first block in local ori pool
    Sinks_ptr,             # (num_q_heads,) float32
    Out_ptr,               # (num_q_tokens, num_q_heads, HEAD_DIM) bf16
    # strides (pools are (num_seqs, max_blocks, block_size, HEAD_DIM) contiguous)
    stride_q_t, stride_q_h, stride_q_d,
    stride_okv_s, stride_okv_b, stride_okv_d,   # seq, block, head ; token-in-block stride == HEAD_DIM
    stride_ckv_s, stride_ckv_b, stride_ckv_d,
    stride_o_t, stride_o_h, stride_o_d,
    softmax_scale_val,
    HEAD_DIM: tl.constexpr,     # 512
    BLOCK_SIZE: tl.constexpr,   # KV page block size (tokens), for addressing
    BLOCK_N: tl.constexpr,      # KV tile loaded per inner iteration (small, fits UB)
    TOPK: tl.constexpr,         # cmp sparse slot count (c4 only)
    CMP_BLOCK_SIZE: tl.constexpr,  # cmp KV tokens per block
    MAX_ORI_BLOCKS: tl.constexpr,  # max ori window blocks dequantized per seq
    CMP_MODE: tl.constexpr,     # 0 = c4 sparse, 1 = c128 full, 2 = dense (no cmp)
    CMP_RATIO: tl.constexpr,    # compress ratio for cmp token count
    MAX_CMP_BLOCKS: tl.constexpr,  # max cmp blocks dequantized per seq (c128 full)
    ORI_WINDOW_SIZE: tl.constexpr,  # sliding-window token count
):
    # One program per (query token, head). query_start_loc supports non-uniform
    # request query lengths, including mixed single-token and MTP requests.
    q_tok_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    seq_idx = tl.load(TokenToSeq_ptr + q_tok_idx)
    causal_end = tl.load(CausalEnds_ptr + q_tok_idx)

    offs_d = tl.arange(0, HEAD_DIM)
    q = tl.load(Q_ptr + q_tok_idx * stride_q_t + head_idx * stride_q_h
                + offs_d * stride_q_d).to(tl.float32)

    # online softmax state (1-element tensors preserve dtype invariance)
    m_i = tl.full([1], -float("inf"), dtype=tl.float32)
    l_i = tl.full([1], 0.0, dtype=tl.float32)
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

    # ---- Phase 1: sliding-window attention over original KV ----
    # SWA window is per-seq: only blocks overlapping [swa_start, causal_end) were
    # dequantized into this seq's local pool (block 0 == first window block).
    swa_start = tl.maximum(0, causal_end - ORI_WINDOW_SIZE)
    first_ori_block = tl.load(OriFirstBlock_ptr + seq_idx)
    ori_token_base = first_ori_block * BLOCK_SIZE
    # Page block holds BLOCK_SIZE tokens but we load BLOCK_N at a time to fit UB.
    n_tiles_per_block = BLOCK_SIZE // BLOCK_N
    offs_n = tl.arange(0, BLOCK_N)
    for lb in range(MAX_ORI_BLOCKS):
        for t in range(n_tiles_per_block):
            offs_tok = ori_token_base + lb * BLOCK_SIZE + t * BLOCK_N + offs_n
            mask_tok = (offs_tok >= swa_start) & (offs_tok < causal_end)
            kv_ptrs = (OriKV_ptr + seq_idx * stride_okv_s + lb * stride_okv_b
                       + (t * BLOCK_N + offs_n)[:, None] * HEAD_DIM  # token-in-block
                       + offs_d[None, :] * stride_okv_d)
            kv = tl.load(kv_ptrs, mask=mask_tok[:, None], other=0.0).to(tl.float32)
            scores = tl.sum(q[None, :] * kv, axis=1) * softmax_scale_val
            scores = tl.where(mask_tok, scores, -float("inf"))
            m_block = tl.max(scores, axis=0)
            m_new = tl.maximum(m_i, m_block)
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(scores - m_new)
            p = tl.where(mask_tok, p, 0.0)
            l_i = l_i * alpha + tl.sum(p, axis=0)
            acc = acc * alpha + tl.sum(p[:, None] * kv, axis=0)
            m_i = m_new

    # ---- Phase 2: compressed KV ----
    n_cmp_tiles = CMP_BLOCK_SIZE // BLOCK_N
    if CMP_MODE == 0:
        # C4 sparse indices and the compact KV rows are both per query token.
        for k in range(TOPK):
            idx = tl.load(CmpSparseIdx_ptr + q_tok_idx * TOPK + k)
            valid_idx = idx >= 0
            if valid_idx:
                kv_ptrs = (CmpKV_ptr + q_tok_idx * stride_ckv_s
                           + k * stride_ckv_b
                           + offs_d * stride_ckv_d)
                kv = tl.load(kv_ptrs).to(tl.float32)
                score = tl.sum(q * kv, axis=0) * softmax_scale_val
                m_new = tl.maximum(m_i, score)
                alpha = tl.exp(m_i - m_new)
                p = tl.exp(score - m_new)
                l_i = l_i * alpha + p
                acc = acc * alpha + p * kv
                m_i = m_new
    elif CMP_MODE == 1:
        # c128: full scan over all valid cmp logical blocks.
        # The compressor emits a token only after a complete ratio-sized group.
        cmp_tokens = causal_end // CMP_RATIO
        n_cmp_blocks = (cmp_tokens + CMP_BLOCK_SIZE - 1) // CMP_BLOCK_SIZE
        for lb in range(MAX_CMP_BLOCKS):
            if lb < n_cmp_blocks:
                for t in range(n_cmp_tiles):
                    offs_tok = lb * CMP_BLOCK_SIZE + t * BLOCK_N + offs_n
                    mask_tok = offs_tok < cmp_tokens
                    kv_ptrs = (CmpKV_ptr + seq_idx * stride_ckv_s + lb * stride_ckv_b
                               + (t * BLOCK_N + offs_n)[:, None] * HEAD_DIM
                               + offs_d[None, :] * stride_ckv_d)
                    kv = tl.load(kv_ptrs, mask=mask_tok[:, None], other=0.0).to(tl.float32)
                    scores = tl.sum(q[None, :] * kv, axis=1) * softmax_scale_val
                    scores = tl.where(mask_tok, scores, -float("inf"))
                    m_block = tl.max(scores, axis=0)
                    m_new = tl.maximum(m_i, m_block)
                    alpha = tl.exp(m_i - m_new)
                    p = tl.exp(scores - m_new)
                    p = tl.where(mask_tok, p, 0.0)
                    l_i = l_i * alpha + tl.sum(p, axis=0)
                    acc = acc * alpha + tl.sum(p[:, None] * kv, axis=0)
                    m_i = m_new
    # CMP_MODE == 2 (dense): no compressed path.

    # ---- sink: extra logit ----
    sink = tl.load(Sinks_ptr + head_idx)
    m_new = tl.maximum(m_i, sink)
    alpha = tl.exp(m_i - m_new)
    p_sink = tl.exp(sink - m_new)
    acc = acc * alpha
    l_i = l_i * alpha + p_sink

    out = acc / l_i
    tl.store(Out_ptr + q_tok_idx * stride_o_t + head_idx * stride_o_h + offs_d * stride_o_d,
             out.to(tl.bfloat16))


def decode_dsa_triton(
    q: torch.Tensor,            # (num_q_tokens, H, 512) bf16 (q tokens incl. spec tokens)
    ori_kv: torch.Tensor,       # (S, MAX_ORI_BLOCKS, BLOCK_SIZE, 512) bf16, dequantized
    cmp_kv: torch.Tensor,       # c4: (T, TOPK, 512); c128: (S, BLOCKS, BS, 512)
    cmp_sparse_idx: torch.Tensor,  # (num_q_tokens, TOPK) int32, only for c4
    seqused_kv: torch.Tensor,   # (S,) int32, per-seq KV length
    token_to_seq: torch.Tensor,  # (num_q_tokens,) int32
    causal_ends: torch.Tensor,   # (num_q_tokens,) int32
    ori_first_block: torch.Tensor,  # (S,) int32
    sinks: torch.Tensor,        # (H,) float32
    softmax_scale: float,
    cmp_mode: int,              # 0=c4 sparse, 1=c128 full, 2=dense
    cmp_ratio: int = 4,
    block_size: int = 128,
    cmp_block_size: int = 128,
    ori_window_size: int = 128,
) -> torch.Tensor:
    num_q_tokens, num_heads, head_dim = q.shape
    num_seqs = seqused_kv.numel()
    if token_to_seq.numel() != num_q_tokens or causal_ends.numel() != num_q_tokens:
        raise ValueError("decode_dsa_triton requires one request id and causal end per query token")
    if head_dim != 512:
        raise ValueError(f"decode_dsa_triton requires head_dim=512, got {head_dim}")
    if block_size % 16 or cmp_block_size % 16:
        raise ValueError("KV block sizes must be multiples of 16")
    if ori_kv.dim() != 4 or ori_kv.shape[0] != num_seqs:
        raise ValueError(
            f"decode_dsa_triton: ori_kv must be (S, MAX_ORI_BLOCKS, BLOCK_SIZE, 512), "
            f"got {tuple(ori_kv.shape)}"
        )
    max_ori_blocks = ori_kv.shape[1]
    out = torch.empty_like(q)
    topk = cmp_sparse_idx.shape[1] if cmp_sparse_idx is not None else 1
    # For dense (cmp_mode==2) cmp tensors are unused but the kernel still needs
    # valid pointers/shapes; pass a dummy per-seq single-block pool.
    if cmp_kv is None:
        cmp_kv = torch.zeros((num_seqs, 1, cmp_block_size, head_dim),
                             dtype=q.dtype, device=q.device)
        max_cmp_blocks = 1
    else:
        expected_dim = 3 if cmp_mode == 0 else 4
        expected_first_dim = num_q_tokens if cmp_mode == 0 else num_seqs
        if cmp_kv.dim() != expected_dim or cmp_kv.shape[0] != expected_first_dim:
            expected_shape = "(T, TOPK, 512)" if cmp_mode == 0 else "(S, BLOCKS, BS, 512)"
            raise ValueError(
                f"decode_dsa_triton: cmp_kv must be {expected_shape}, got {tuple(cmp_kv.shape)}"
            )
        max_cmp_blocks = cmp_kv.shape[1]
    if cmp_sparse_idx is None:
        cmp_sparse_idx = torch.full((num_q_tokens, 1), -1, dtype=torch.int32, device=q.device)
    grid = (num_q_tokens, num_heads)
    # CmpKV layout differs by mode: c4 is (T, TOPK, D) [3D], c128/dummy is
    # (S, BLOCKS, BS, D) [4D]. The kernel strides are (token/seq, block/topk,
    # head); head is always the last dim, so use stride(-1) to stay layout-agnostic.
    _decode_c4_bf16_kernel[grid](
        q, ori_kv, cmp_kv, cmp_sparse_idx, token_to_seq,
        causal_ends, ori_first_block, sinks, out,
        q.stride(0), q.stride(1), q.stride(2),
        ori_kv.stride(0), ori_kv.stride(1), ori_kv.stride(3),
        cmp_kv.stride(0), cmp_kv.stride(1), cmp_kv.stride(-1),
        out.stride(0), out.stride(1), out.stride(2),
        softmax_scale,
        HEAD_DIM=head_dim,
        BLOCK_SIZE=block_size,
        BLOCK_N=16,
        TOPK=topk,
        CMP_BLOCK_SIZE=cmp_block_size,
        MAX_ORI_BLOCKS=max_ori_blocks,
        CMP_MODE=cmp_mode,
        CMP_RATIO=cmp_ratio,
        MAX_CMP_BLOCKS=max_cmp_blocks,
        ORI_WINDOW_SIZE=ori_window_size,
    )
    return out
