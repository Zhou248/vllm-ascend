import pytest
import torch

from vllm_ascend.attention.dsa_triton_decode import (
    _build_causal_ends,
    _prepare_c4_pool_graph,
    dequant_blocks_vec,
    triton_decode_dsa,
)


def test_dequant_blocks_preserves_bfloat16_rope_bytes():
    rope = torch.tensor(
        [[1.0, -0.5] + [0.0] * 62, [3.0, 0.25] + [0.0] * 62],
        dtype=torch.bfloat16,
    )
    packed = torch.zeros((1, 2, 1, 640), dtype=torch.uint8)
    packed[:, :, :, :128] = rope.view(torch.uint8).reshape(1, 2, 1, 128)
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


def test_ragged_multi_request_causal_ends():
    query_start_loc = torch.tensor([0, 1, 4], dtype=torch.int32)
    seqused_kv = torch.tensor([10, 20], dtype=torch.int32)

    token_to_seq, causal_ends = _build_causal_ends(seqused_kv, query_start_loc, 4)

    torch.testing.assert_close(token_to_seq, torch.tensor([0, 1, 1, 1]))
    torch.testing.assert_close(causal_ends, torch.tensor([10, 18, 19, 20]))


def test_prepare_c4_pool_uses_per_token_causal_bound(monkeypatch):
    import vllm_ascend.attention.dsa_triton_decode as mod

    captured = {}

    def fake_dequant(pool, physical_blocks, block_offsets):
        captured["physical_blocks"] = physical_blocks.clone()
        captured["block_offsets"] = block_offsets.clone()
        return torch.zeros((physical_blocks.numel(), 512), dtype=torch.bfloat16)

    monkeypatch.setattr(mod, "dequant_token_rows_vec", fake_dequant)
    query_start_loc = torch.tensor([0, 1, 4], dtype=torch.int32)
    seqused_kv = torch.tensor([10, 20], dtype=torch.int32)
    block_table = torch.tensor([[5, 6], [7, 8]], dtype=torch.int32)
    # Request 1 causal cmp counts are floor([18, 19, 20] / 4) = [4, 4, 5].
    indices = torch.tensor([[[1, -1]], [[3, 4]], [[3, 4]], [[4, 5]]], dtype=torch.int32)

    token_to_seq, causal_ends = _build_causal_ends(seqused_kv, query_start_loc, 4)
    pool, remapped = _prepare_c4_pool_graph(
        torch.empty(1), block_table, indices, token_to_seq, causal_ends, 128, 4
    )

    assert tuple(pool.shape) == (4, 2, 512)
    torch.testing.assert_close(
        remapped,
        torch.tensor([[0, -1], [0, -1], [0, -1], [0, -1]], dtype=torch.int32),
    )
    # All valid compressed tokens are in logical page zero of their request.
    torch.testing.assert_close(captured["physical_blocks"], torch.tensor([5, 5, 7, 7, 7, 7, 7, 7]))
    torch.testing.assert_close(captured["block_offsets"], torch.tensor([1, 0, 3, 0, 3, 0, 4, 0]))


def test_prepare_c4_pool_limits_sparse_slots_to_visible_cmp_tokens(monkeypatch):
    import vllm_ascend.attention.dsa_triton_decode as mod

    def fake_dequant(pool, physical_blocks, block_offsets):
        return torch.zeros((physical_blocks.numel(), 512), dtype=torch.bfloat16)

    monkeypatch.setattr(mod, "dequant_token_rows_vec", fake_dequant)
    pool, remapped = _prepare_c4_pool_graph(
        torch.empty(1),
        torch.tensor([[5]], dtype=torch.int32),
        torch.tensor([[[0, 1, 2, 3]]], dtype=torch.int32),
        torch.tensor([0]),
        torch.tensor([8]),
        128,
        4,
    )

    assert tuple(pool.shape) == (1, 4, 512)
    torch.testing.assert_close(remapped, torch.tensor([[0, 1, -1, -1]], dtype=torch.int32))


def test_triton_decode_passes_ragged_mtp_mapping_to_kernel(monkeypatch):
    import vllm_ascend.attention.dsa_triton_decode as mod

    captured = {}

    monkeypatch.setattr(
        mod,
        "_prepare_ori_pool_graph",
        lambda *args: (
            torch.empty((2, 1, 128, 512), dtype=torch.bfloat16),
            torch.tensor([0, 0], dtype=torch.int32),
        ),
    )
    monkeypatch.setattr(
        mod,
        "_prepare_c4_pool_graph",
        lambda *args: (
            torch.empty((4, 1, 512), dtype=torch.bfloat16),
            torch.zeros((4, 1), dtype=torch.int32),
        ),
    )

    def fake_kernel(q, ori_kv, cmp_kv, sparse_indices, seqused_kv, token_to_seq,
                    causal_ends, ori_first_block, sinks, softmax_scale, **kwargs):
        captured["token_to_seq"] = token_to_seq
        captured["causal_ends"] = causal_ends
        return torch.empty_like(q)

    monkeypatch.setattr(mod, "_load_decode_kernel", lambda: fake_kernel)
    packed_cache = torch.empty((1, 128, 1, 640), dtype=torch.float8_e4m3fn)
    triton_decode_dsa(
        q=torch.empty((4, 2, 512), dtype=torch.bfloat16),
        swa_kv_cache=packed_cache,
        compress_kv_cache=packed_cache,
        cmp_sparse_indices=torch.zeros((4, 1, 1), dtype=torch.int32),
        ori_block_table=torch.zeros((2, 1), dtype=torch.int32),
        cmp_block_table=torch.zeros((2, 1), dtype=torch.int32),
        seqused_kv=torch.tensor([10, 20], dtype=torch.int32),
        sinks=torch.zeros(2),
        softmax_scale=1.0,
        compress_ratio=4,
        ori_block_size=128,
        cmp_block_size=128,
        ori_window_size=128,
        query_start_loc=torch.tensor([0, 1, 4], dtype=torch.int32),
        max_query_tokens=4,
    )

    torch.testing.assert_close(captured["token_to_seq"], torch.tensor([0, 1, 1, 1]))
    torch.testing.assert_close(captured["causal_ends"], torch.tensor([10, 18, 19, 20]))


def test_triton_decode_rejects_non_bfloat16_query():
    packed_cache = torch.empty((1, 128, 1, 640), dtype=torch.float8_e4m3fn)

    with pytest.raises(ValueError, match="q must use bfloat16"):
        triton_decode_dsa(
            q=torch.empty((1, 2, 512), dtype=torch.float16),
            swa_kv_cache=packed_cache,
            compress_kv_cache=None,
            cmp_sparse_indices=None,
            ori_block_table=torch.zeros((1, 1), dtype=torch.int32),
            cmp_block_table=None,
            seqused_kv=torch.tensor([1], dtype=torch.int32),
            sinks=torch.zeros(2),
            softmax_scale=1.0,
            compress_ratio=1,
            ori_block_size=128,
            cmp_block_size=None,
            ori_window_size=128,
            query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
            max_query_tokens=1,
        )


def test_triton_decode_rejects_unpacked_kv_cache():
    with pytest.raises(ValueError, match="swa_kv_cache must use float8_e4m3fn"):
        triton_decode_dsa(
            q=torch.empty((1, 2, 512), dtype=torch.bfloat16),
            swa_kv_cache=torch.empty((1, 128, 1, 512), dtype=torch.bfloat16),
            compress_kv_cache=None,
            cmp_sparse_indices=None,
            ori_block_table=torch.zeros((1, 1), dtype=torch.int32),
            cmp_block_table=None,
            seqused_kv=torch.tensor([1], dtype=torch.int32),
            sinks=torch.zeros(2),
            softmax_scale=1.0,
            compress_ratio=1,
            ori_block_size=128,
            cmp_block_size=None,
            ori_window_size=128,
            query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
            max_query_tokens=1,
        )


@pytest.mark.parametrize(
    ("has_triton", "device_name", "expected"),
    [
        (False, "A5", False),
        (True, "A3", False),
        (True, "A5", True),
    ],
)
def test_use_triton_decode_checks_runtime_support(monkeypatch, has_triton, device_name, expected):
    from vllm_ascend import envs
    from vllm_ascend.attention import dsa_v1
    from vllm_ascend.utils import AscendDeviceType

    monkeypatch.setattr(envs, "VLLM_ASCEND_ENABLE_DSA_TRITON_DECODE", True)
    monkeypatch.setattr(dsa_v1, "HAS_TRITON", has_triton)
    monkeypatch.setattr(
        dsa_v1,
        "get_ascend_device_type",
        lambda: AscendDeviceType[device_name],
    )

    assert dsa_v1._use_triton_decode() is expected
