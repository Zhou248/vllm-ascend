# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_ascend.ops.fused_moe import shared_experts


def make_executor(monkeypatch, device="cpu"):
    parameter = SimpleNamespace(device=SimpleNamespace(type=device))
    layer = SimpleNamespace(
        gate_up_proj=SimpleNamespace(input_size=8),
        act_fn=torch.nn.Identity(),
        parameters=lambda: iter([parameter]),
    )
    config = SimpleNamespace(
        hidden_dim=8,
        in_dtype=torch.float32,
        swiglu_limit=None,
        swiglu_alpha=None,
        swiglu_beta=None,
        is_sequence_parallel=False,
    )
    monkeypatch.setattr(
        shared_experts,
        "get_ascend_config",
        lambda: SimpleNamespace(multistream_overlap_shared_expert=True, enable_shared_expert_dp=False),
    )
    quant = SimpleNamespace(process_weights_after_loading=Mock(return_value="processed"))
    executor = shared_experts.AscendSharedExperts(layer, config, None, quant)
    executor.validate_consistency = Mock()
    return executor, quant, parameter


def test_cpu_weights_defer_validation_until_onloaded(monkeypatch):
    executor, quant, parameter = make_executor(monkeypatch)
    assert quant.process_weights_after_loading() == "processed"
    executor.validate_consistency.assert_not_called()
    assert executor._pending_weight_validation
    with pytest.raises(RuntimeError, match="prefetched device weights"):
        executor._validate_prefetched_weights()
    executor.validate_consistency.assert_not_called()
    parameter.device.type = "npu"
    executor._validate_prefetched_weights()
    executor._validate_prefetched_weights()
    executor.validate_consistency.assert_called_once()


def test_resident_weights_keep_immediate_validation(monkeypatch):
    executor, quant, _ = make_executor(monkeypatch, "npu")
    quant.process_weights_after_loading()
    executor.validate_consistency.assert_called_once()
    assert not executor._pending_weight_validation


def test_failed_validation_is_not_marked_complete(monkeypatch):
    executor, quant, parameter = make_executor(monkeypatch)
    quant.process_weights_after_loading()
    parameter.device.type = "npu"
    executor.validate_consistency.side_effect = ValueError("mismatch")
    with pytest.raises(ValueError, match="mismatch"):
        executor._validate_prefetched_weights()
    assert executor._pending_weight_validation
