from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import vllm_ascend.worker.v2.pcp_manager as module


@pytest.mark.parametrize(
    "enabled,groups,has_tables,expected",
    [
        (True, 1, True, 1),
        (False, 1, True, 0),
        (True, 0, True, 0),
        (True, 1, False, 0),
    ],
)
def test_manager_initialization_warms_only_enabled_usable_tables(monkeypatch, enabled, groups, has_tables, expected):
    monkeypatch.setenv("VLLM_ASCEND_FUSED_PCP_BLOCK_TABLES", "1" if enabled else "0")
    warmup = Mock()
    monkeypatch.setattr(module, "warmup_fused_gather", warmup, raising=False)
    tables = (
        SimpleNamespace(slot_mappings=torch.zeros(1, dtype=torch.int32), num_kv_cache_groups=groups)
        if has_tables
        else None
    )

    def base_init(self, **kwargs):
        self._block_tables = kwargs["block_tables"]
        self._local_block_tables = (torch.full((1, 2), 41),) if has_tables and groups else ()
        self._global_batch_slot_mappings = None
        self._gathered_kv_slot_mappings = None
        self._pad_slot_id = torch.tensor(-1)

    monkeypatch.setattr(module.PCPManager, "__init__", base_init)
    manager = module.AscendPCPManager(4, 0, torch.device("cpu"), block_tables=tables)
    assert warmup.call_count == expected
    if expected:
        warmup.assert_called_once_with(torch.device("cpu"))
        assert torch.equal(manager._local_block_tables[0], torch.full((1, 2), 41))
    assert manager._prepared_global_gather is None
