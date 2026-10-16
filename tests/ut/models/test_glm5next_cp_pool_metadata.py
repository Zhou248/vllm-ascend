# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Causal boundaries for CP slices crossing requests and padding."""

from types import SimpleNamespace

import pytest
import torch

import vllm_ascend.attention.indexer_kpool as module


@pytest.mark.parametrize(
    "rank,expected_q,expected_pool,expected_tail,actual",
    [
        (0, [3, 4], [3, 25], [1, 1], 4),
        (1, [0, 3], [0, 26], [0, 0], 3),
    ],
)
def test_shard_causal_pool_bounds(monkeypatch, rank, expected_q, expected_pool, expected_tail, actual):
    device = "cpu"
    monkeypatch.setattr(module, "get_tp_group", lambda: SimpleNamespace(rank_in_group=rank))
    builder = object.__new__(module.AscendIndexerKPoolMetadataBuilder)
    builder.use_dsa_cp = True
    builder.dsa_cp_size = 2
    builder.compress_ratio = 4
    builder._max_num_metadata_reqs = 8
    builder._dsa_cp_query_buffers = {}
    builder.device = torch.device(device)
    common = SimpleNamespace(
        slot_mapping=torch.zeros(8, dtype=torch.int64, device=device),
        num_reqs=2,
        num_actual_tokens=7,
        query_start_loc=torch.tensor([0, 3, 7], dtype=torch.int32, device=device),
        seq_lens=torch.tensor([13, 104], dtype=torch.int32, device=device),
    )
    positions = torch.tensor([10, 11, 12, 100, 101, 102, 103, 0], device=device)
    metadata = builder._build_query_metadata(common, positions)
    assert metadata.cum_query_lens.tolist() == expected_q
    assert metadata.seq_lens.tolist() == expected_pool
    assert metadata.pool_tail.tolist() == expected_tail
    assert metadata.num_actual_tokens == actual
    assert metadata.positions.tolist() == positions[rank * 4 : (rank + 1) * 4].tolist()
    again = builder._build_query_metadata(common, positions)
    for name in ["cum_query_lens", "seq_lens", "pool_tail"]:
        assert getattr(metadata, name).data_ptr() == getattr(again, name).data_ptr()


def test_padding_only_rank_has_no_visible_pools(monkeypatch):
    device = "cpu"
    monkeypatch.setattr(module, "get_tp_group", lambda: SimpleNamespace(rank_in_group=1))
    builder = object.__new__(module.AscendIndexerKPoolMetadataBuilder)
    builder.use_dsa_cp = True
    builder.dsa_cp_size = 2
    builder.compress_ratio = 4
    builder._max_num_metadata_reqs = 2
    builder._dsa_cp_query_buffers = {}
    builder.device = torch.device(device)
    common = SimpleNamespace(
        slot_mapping=torch.zeros(2, dtype=torch.int64, device=device),
        num_reqs=1,
        num_actual_tokens=1,
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32, device=device),
        seq_lens=torch.tensor([235930], dtype=torch.int32, device=device),
    )
    metadata = builder._build_query_metadata(common, torch.tensor([235929, 0], device=device))
    assert metadata.cum_query_lens.tolist() == [0]
    assert metadata.seq_lens.tolist() == [0]
    assert metadata.pool_tail.tolist() == [0]
    assert metadata.num_actual_tokens == 0
