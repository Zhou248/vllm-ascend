import torch

from vllm_ascend.attention.dsa_triton_decode import (
    _dequant_c128_batch,
    _dequant_ori_window_batch,
    _remap_c4_sparse_per_seq,
    dequant_blocks_vec,
    remap_c4_sparse_token_indices,
    triton_decode_dsa,
)


def test_dequant_blocks_preserves_bfloat16_rope_bytes():
    block_size = 2
    rope = torch.tensor(
        [[1.0, -0.5] + [0.0] * 62, [3.0, 0.25] + [0.0] * 62],
        dtype=torch.bfloat16,
    )
    packed = torch.zeros((1, block_size, 1, 640), dtype=torch.uint8)
    packed[:, :, :, :128] = rope.view(torch.uint8).reshape(1, block_size, 1, 128)
    # Zero FP8 values with unit E8M0 scales.
    packed[:, :, :, 576:583] = 127

    result = dequant_blocks_vec(packed.view(torch.float8_e4m3fn), torch.tensor([[0]]))

    torch.testing.assert_close(result[0, :, 448:], rope)


def test_dequant_blocks_applies_scale_per_64_values():
    nope = torch.ones((1, 448), dtype=torch.float8_e4m3fn)
    packed = torch.zeros((1, 1, 1, 640), dtype=torch.uint8)
    packed[:, :, :, 128:576] = nope.view(torch.uint8).reshape(1, 1, 1, 448)
    packed[:, :, :, 576:583] = torch.arange(127, 134, dtype=torch.uint8)

    result = dequant_blocks_vec(packed.view(torch.float8_e4m3fn), torch.tensor([[0]]))
    expected = torch.cat(
        [torch.full((64,), float(2**exponent), dtype=torch.bfloat16) for exponent in range(7)]
    )

    torch.testing.assert_close(result[0, 0, :448], expected)


def test_remap_c4_sparse_indices_preserves_page_offset():
    sparse_indices = torch.tensor([[130, 3, -1, 999]], dtype=torch.int32)
    block_table = torch.tensor([[7, 11, 5]], dtype=torch.int32)

    physical_pages, remapped = remap_c4_sparse_token_indices(
        sparse_indices, block_table, cmp_token_count=256, block_size=128
    )

    assert physical_pages == [7, 11]
    torch.testing.assert_close(remapped, torch.tensor([[130, 3, -1, -1]]))


def _fake_dequant_by_block_id():
    """dequant stand-in: returns (N, block_size, head_dim) with the physical
    block id stored at [0,0] so per-seq pool assembly can be checked on CPU."""

    def _impl(pool, bt, max_logical=None):
        ids = bt.long().reshape(-1)
        if max_logical is not None:
            ids = ids[:max_logical]
        out = torch.zeros((ids.shape[0], 128, 512), dtype=torch.float32)
        for i, p in enumerate(ids.tolist()):
            out[i, 0, 0] = float(p)
        return out

    return _impl


def test_dequant_ori_window_batch_assembles_per_seq_pools(monkeypatch):
    import vllm_ascend.attention.dsa_triton_decode as mod

    monkeypatch.setattr(mod, "dequant_blocks_vec", _fake_dequant_by_block_id())
    # seq0 kv_len=300: swa_start=172 -> window blocks logical[1,3) -> phys [1,2]
    # seq1 kv_len=130: swa_start=2   -> window blocks logical[0,2) -> phys [5,6]
    bt = torch.zeros((2, 8), dtype=torch.int32)
    bt[0, 0], bt[0, 1], bt[0, 2] = 0, 1, 2
    bt[1, 0], bt[1, 1] = 5, 6
    pool = None  # fake dequant does not read the pool
    ori = _dequant_ori_window_batch(pool, bt, [300, 130], 128, 128, 512, torch.zeros(1))
    assert tuple(ori.shape) == (2, 2, 128, 512)  # max window blocks = 2
    assert ori[0, 0, 0, 0].item() == 1.0  # seq0 blk0 -> phys 1
    assert ori[0, 1, 0, 0].item() == 2.0  # seq0 blk1 -> phys 2
    assert ori[1, 0, 0, 0].item() == 5.0  # seq1 blk0 -> phys 5
    assert ori[1, 1, 0, 0].item() == 6.0  # seq1 blk1 -> phys 6


def test_dequant_ori_window_batch_pads_short_seq(monkeypatch):
    import vllm_ascend.attention.dsa_triton_decode as mod

    monkeypatch.setattr(mod, "dequant_blocks_vec", _fake_dequant_by_block_id())
    # seq0 kv_len=300 -> 2 window blocks (phys 1,2); seq1 kv_len=64 -> 1 block (phys 5)
    bt = torch.zeros((2, 8), dtype=torch.int32)
    bt[0, 0], bt[0, 1], bt[0, 2] = 0, 1, 2
    bt[1, 0] = 5
    ori = _dequant_ori_window_batch(None, bt, [300, 64], 128, 128, 512, torch.zeros(1))
    assert tuple(ori.shape) == (2, 2, 128, 512)
    assert ori[1, 0, 0, 0].item() == 5.0  # seq1 only window block -> phys 5
    assert ori[1, 1, 0, 0].item() == 0.0  # seq1 padding slot -> phys 0 marker


def test_remap_c4_sparse_per_seq_builds_local_pools(monkeypatch):
    import vllm_ascend.attention.dsa_triton_decode as mod

    monkeypatch.setattr(mod, "dequant_blocks_vec", _fake_dequant_by_block_id())
    cmp_bt = torch.zeros((2, 8), dtype=torch.int32)
    cmp_bt[0, 0], cmp_bt[0, 1] = 10, 11
    cmp_bt[1, 0] = 12
    # seq0 cmp_token_count = 600//4 = 150; idx 130->logical1(phys11), 3->logical0(phys10)
    # seq1 cmp_token_count = 130//4 = 32;  idx 0->logical0(phys12)
    csi = torch.tensor([[[130, 3, -1, 999]], [[0, -1, -1, -1]]], dtype=torch.int32)
    cmp_pool, csi_out = _remap_c4_sparse_per_seq(
        csi, cmp_bt, [600, 130], 128, 4, None, 512, torch.zeros(1), tokens_per_seq=1
    )
    assert tuple(cmp_pool.shape) == (2, 2, 128, 512)
    assert cmp_pool[0, 0, 0, 0].item() == 10.0  # phys 10
    assert cmp_pool[0, 1, 0, 0].item() == 11.0  # phys 11
    assert cmp_pool[1, 0, 0, 0].item() == 12.0  # phys 12
    torch.testing.assert_close(
        csi_out, torch.tensor([[130, 3, -1, -1], [0, -1, -1, -1]], dtype=torch.int32)
    )


def test_remap_c4_sparse_mtp_shares_pool_across_spec_tokens(monkeypatch):
    """MTP: 1 request, 2 query tokens sharing one per-request cmp pool.

    Both spec tokens' sparse indices remap into the SAME local pool; the pool
    is built once from the union of referenced pages across the tokens.
    """
    import vllm_ascend.attention.dsa_triton_decode as mod

    monkeypatch.setattr(mod, "dequant_blocks_vec", _fake_dequant_by_block_id())
    cmp_bt = torch.zeros((1, 8), dtype=torch.int32)
    cmp_bt[0, 0], cmp_bt[0, 1] = 10, 11   # logical0->phys10, logical1->phys11
    # seq0 cmp_token_count = 600//4 = 150
    # token0: idx 130(logical1/phys11), 3(logical0/phys10)
    # token1: idx 4(logical0/phys10), 200(invalid>=150 -> -1)
    csi = torch.tensor([[[130, 3]], [[4, 200]]], dtype=torch.int32)  # (2 q-tokens, 1, topk=2)
    cmp_pool, csi_out = _remap_c4_sparse_per_seq(
        csi, cmp_bt, [600], 128, 4, None, 512, torch.zeros(1), tokens_per_seq=2
    )
    # pool built from union {phys10, phys11} -> max_cmp_blocks=2
    assert tuple(cmp_pool.shape) == (1, 2, 128, 512)
    assert cmp_pool[0, 0, 0, 0].item() == 10.0  # local0 -> phys10
    assert cmp_pool[0, 1, 0, 0].item() == 11.0  # local1 -> phys11
    # token0: 130->local1*128+2=130, 3->local0*128+3=3
    # token1: 4->local0*128+4=4, 200 invalid->-1
    torch.testing.assert_close(
        csi_out, torch.tensor([[130, 3], [4, -1]], dtype=torch.int32)
    )


def test_dequant_c128_batch_assembles_first_n_blocks(monkeypatch):
    import vllm_ascend.attention.dsa_triton_decode as mod

    monkeypatch.setattr(mod, "dequant_blocks_vec", _fake_dequant_by_block_id())
    cmp_bt = torch.zeros((2, 8), dtype=torch.int32)
    cmp_bt[0, 0], cmp_bt[0, 1], cmp_bt[0, 2] = 20, 21, 22
    cmp_bt[1, 0] = 23
    pool = None  # fake dequant does not read the pool
    c128 = _dequant_c128_batch(pool, cmp_bt, [600, 130], 128, 128, 512, torch.zeros(1))
    # cmp_tokens = kv_len // 128 -> seq0: 4, seq1: 1 -> blocks 1, 1 -> max=1
    assert tuple(c128.shape) == (2, 1, 128, 512)
    assert c128[0, 0, 0, 0].item() == 20.0
    assert c128[1, 0, 0, 0].item() == 23.0


def test_triton_decode_dsa_accepts_spec_decode_query_tokens(monkeypatch):
    # MTP: 1 request, 2 query tokens (real + draft). Must NOT be rejected; the
    # dequant chain runs per-request and the kernel is launched over num_q_tokens.
    # Fake the dequant chain + kernel entry so this runs on CPU.
    import vllm_ascend.attention.dsa_triton_decode as mod

    monkeypatch.setattr(mod, "dequant_blocks_vec", _fake_dequant_by_block_id())
    captured = {}

    def fake_kernel_loader():
        # Mimic the decode_dsa_triton entry signature.
        def _kernel(q, ori_kv, cmp_kv, cmp_sparse_idx, seqused_kv, sinks, softmax_scale, **kw):
            captured["num_q_tokens"] = q.shape[0]
            captured["num_seqs"] = seqused_kv.numel()
            return torch.empty_like(q)

        return _kernel

    monkeypatch.setattr(mod, "_load_decode_kernel", fake_kernel_loader)

    q = torch.zeros((2, 64, 512), dtype=torch.bfloat16)           # 2 query tokens
    seqused_kv = torch.tensor([7], dtype=torch.int32)             # 1 request
    ori_bt = torch.zeros((1, 4), dtype=torch.int32)
    triton_decode_dsa(
        q=q,
        swa_kv_cache=None,
        compress_kv_cache=None,
        cmp_sparse_indices=None,
        ori_block_table=ori_bt,
        cmp_block_table=None,
        seqused_kv=seqused_kv,
        sinks=torch.zeros(64, dtype=torch.float32),
        softmax_scale=0.0442,
        compress_ratio=0,
        ori_block_size=128,
        cmp_block_size=None,
        ori_window_size=128,
    )
    # 2 query tokens for 1 request => tokens_per_seq = 2 (MTP), kernel sees both.
    assert captured["num_q_tokens"] == 2
    assert captured["num_seqs"] == 1


def test_triton_decode_dsa_rejects_block_table_row_mismatch():
    # ori_block_table rows must equal num_seqs (from seqused_kv).
    q = torch.zeros((2, 64, 512), dtype=torch.bfloat16)
    seqused_kv = torch.tensor([7, 9], dtype=torch.int32)          # 2 requests
    ori_bt = torch.zeros((1, 4), dtype=torch.int32)              # only 1 row
    import pytest

    with pytest.raises(ValueError, match="ori_block_table rows"):
        triton_decode_dsa(
            q=q,
            swa_kv_cache=None,
            compress_kv_cache=None,
            cmp_sparse_indices=None,
            ori_block_table=ori_bt,
            cmp_block_table=None,
            seqused_kv=seqused_kv,
            sinks=torch.zeros(64, dtype=torch.float32),
            softmax_scale=0.0442,
            compress_ratio=0,
            ori_block_size=128,
            cmp_block_size=None,
            ori_window_size=128,
        )
