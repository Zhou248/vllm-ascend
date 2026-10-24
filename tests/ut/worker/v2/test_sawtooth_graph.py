# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.sequence import IntermediateTensors
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor, ModelCudaGraphManager

from vllm_ascend.compilation import acl_graph
from vllm_ascend.compilation.acl_graph import GraphParamsScope, get_graph_params, get_graph_params_descriptor
from vllm_ascend.worker.v2 import aclgraph_utils, sawtooth_graph, sawtooth_runtime


def test_capture_preserves_descriptor_identity_and_fixed_callable(monkeypatch):
    manager = object.__new__(sawtooth_graph.SawtoothAclGraphManager)
    manager.stage_model = object()
    manager.vllm_config = object()
    manager.graph_scope = GraphParamsScope(("plan", 2), object())
    manager._max_full_descs_to_capture = None
    first = BatchExecutionDescriptor(cg_mode=CUDAGraphMode.FULL, num_tokens=8, num_reqs=4)
    second = BatchExecutionDescriptor(cg_mode=CUDAGraphMode.FULL, num_tokens=8, num_reqs=8)
    original = {CUDAGraphMode.FULL: [first, second]}
    manager._capture_descs = original
    seen = []

    def capture(self, model, *args, **kwargs):
        assert model is manager.stage_model
        params = get_graph_params()
        assert params.handles[8] == []
        params.handles[8].append(get_graph_params_descriptor())
        seen.append(params)

    monkeypatch.setattr(sawtooth_graph, "set_current_vllm_config", lambda _: nullcontext())
    monkeypatch.setattr(aclgraph_utils.ModelAclGraphManager, "capture", capture)
    manager.capture(manager.stage_model)
    assert manager._capture_descs is original
    assert seen[0] is not seen[1]
    assert seen[0].handles[8] == [first]
    assert seen[1].handles[8] == [second]
    with pytest.raises(ValueError, match="fixed stage"):
        manager.capture(object())


def test_capture_exception_restores_descriptor_table(monkeypatch):
    manager = object.__new__(sawtooth_graph.SawtoothAclGraphManager)
    manager.stage_model = object()
    manager.vllm_config = object()
    manager.graph_scope = GraphParamsScope(("plan", 0), object())
    manager._max_full_descs_to_capture = None
    desc = BatchExecutionDescriptor(cg_mode=CUDAGraphMode.FULL, num_tokens=8, num_reqs=8)
    manager._capture_descs = original = {CUDAGraphMode.FULL: [desc]}
    monkeypatch.setattr(sawtooth_graph, "set_current_vllm_config", lambda _: nullcontext())
    monkeypatch.setattr(aclgraph_utils.ModelAclGraphManager, "capture", Mock(side_effect=RuntimeError("capture")))
    with pytest.raises(RuntimeError, match="capture"):
        manager.capture(manager.stage_model)
    assert manager._capture_descs is original


def test_dispatch_binds_the_captured_descriptor_until_stage_scope_exits(monkeypatch):
    manager = object.__new__(sawtooth_graph.SawtoothAclGraphManager)
    manager.graph_scope = GraphParamsScope(("plan", 2), object())
    manager.vllm_config = object()
    desc = BatchExecutionDescriptor(cg_mode=CUDAGraphMode.PIECEWISE, num_tokens=8, num_reqs=None)
    monkeypatch.setattr(sawtooth_graph, "set_current_vllm_config", lambda _: nullcontext())
    monkeypatch.setattr(aclgraph_utils.ModelAclGraphManager, "dispatch", lambda *args, **kwargs: desc)
    previous = get_graph_params_descriptor()
    with manager.activate(desc):
        captured_key = (manager.graph_scope.identity, get_graph_params_descriptor())
    with manager.activate():
        assert get_graph_params_descriptor() is None
        assert manager.dispatch(2, 8, None, 0) is desc
        replay_key = (manager.graph_scope.identity, get_graph_params_descriptor())
        assert replay_key == captured_key
    assert get_graph_params_descriptor() is previous


def test_piecewise_dispatch_replays_captured_entry_without_recapture(monkeypatch):
    manager = object.__new__(sawtooth_graph.SawtoothAclGraphManager)
    manager.graph_scope = GraphParamsScope(("plan", 2), object())
    manager.vllm_config = object()
    desc = BatchExecutionDescriptor(cg_mode=CUDAGraphMode.PIECEWISE, num_tokens=8, num_reqs=None)
    context = SimpleNamespace(batch_descriptor="batch8", cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE)
    graph = Mock()
    result = object()
    wrapper = object.__new__(acl_graph.ACLGraphWrapper)
    wrapper.graph_scope = manager.graph_scope
    wrapper.graph_pool = manager.graph_scope.graph_pool
    wrapper.runtime_mode = CUDAGraphMode.PIECEWISE
    wrapper.is_debugging_mode = False
    wrapper.enable_enpu = False
    wrapper.use_eagle = False
    wrapper.concrete_aclgraph_entries = {
        (manager.graph_scope.identity, desc, context.batch_descriptor): acl_graph.ACLGraphEntry(
            batch_descriptor=context.batch_descriptor,
            aclgraph=graph,
            output=result,
        )
    }
    monkeypatch.setattr(sawtooth_graph, "set_current_vllm_config", lambda _: nullcontext())
    monkeypatch.setattr(aclgraph_utils.ModelAclGraphManager, "dispatch", lambda *args, **kwargs: desc)
    monkeypatch.setattr(acl_graph, "get_forward_context", lambda: context)
    monkeypatch.setattr(acl_graph, "_EXTRA_CTX", SimpleNamespace(is_draft_model=False))
    monkeypatch.setattr(
        acl_graph, "validate_cudagraph_capturing_enabled", Mock(side_effect=AssertionError("recapture"))
    )
    with manager.activate():
        manager.dispatch(2, 8, None, 0)
        assert wrapper() is result
    graph.replay.assert_called_once_with()
    assert len(wrapper.concrete_aclgraph_entries) == 1


def test_full_update_is_queued_after_replay_without_waiting_for_that_replay(monkeypatch):
    """A post-replay update-stream wait would deadlock on graph ExternalEvents."""
    calls = []
    manager = object.__new__(aclgraph_utils.ModelAclGraphManager)
    manager.update_stream = SimpleNamespace(wait_stream=lambda stream: calls.append("wait_previous_compute"))
    manager.model_runner = SimpleNamespace(dp_size=1, speculative_config=None)
    manager.vllm_config = object()
    desc = SimpleNamespace(cg_mode=CUDAGraphMode.FULL)
    monkeypatch.setattr(aclgraph_utils.torch.npu, "current_stream", lambda: object())
    monkeypatch.setattr(ModelCudaGraphManager, "run_fullgraph", lambda *args: calls.append("replay"))
    monkeypatch.setattr(aclgraph_utils, "set_current_vllm_config", lambda *args: nullcontext())
    monkeypatch.setattr(aclgraph_utils, "set_forward_context", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(aclgraph_utils, "get_forward_context", lambda: object())
    monkeypatch.setattr(aclgraph_utils, "update_full_graph_params", lambda *args: calls.append("update"))
    manager._graph_relay(object(), desc, 8, {})
    assert calls == ["wait_previous_compute", "replay", "update"]


def test_output_staging_preserves_hidden_and_residual_after_graph_reuse():
    output = IntermediateTensors({"hidden_states": torch.arange(6).reshape(2, 3), "residual": torch.ones(2, 3)})
    lease = IntermediateTensors({key: torch.empty_like(value) for key, value in output.tensors.items()})
    sawtooth_graph.stage_output_to_lease(output, lease)
    output["hidden_states"].zero_()
    output["residual"].zero_()
    assert torch.equal(lease["hidden_states"], torch.arange(6).reshape(2, 3))
    assert torch.equal(lease["residual"], torch.ones(2, 3))
    with pytest.raises(ValueError, match="alias"):
        sawtooth_graph.stage_output_to_lease(lease, lease)
    with pytest.raises(ValueError, match="schema"):
        sawtooth_graph.stage_output_to_lease(output, IntermediateTensors({"hidden_states": torch.empty(2, 3)}))


def test_updatable_graph_resolves_only_this_graph_before_replay(monkeypatch):
    calls = []
    metadata = {"model.layers.16.attn": object()}
    desc = BatchExecutionDescriptor(cg_mode=CUDAGraphMode.FULL, num_tokens=8, num_reqs=8)

    class Graph:
        def resolve_tasks(self, source):
            assert source is metadata
            calls.append("resolve_group_tasks")
            return "tasks16"

        def update(self, stream, tasks):
            assert tasks == "tasks16"
            calls.append("update_group_tasks")

    manager = object.__new__(aclgraph_utils.ModelAclGraphManager)
    manager.graphs = {desc: Graph()}
    manager.update_stream = SimpleNamespace(wait_stream=lambda _: calls.append("wait_previous_compute"))
    monkeypatch.setattr(aclgraph_utils, "UpdatableGraph", Graph)
    monkeypatch.setattr(aclgraph_utils, "ContextSource", lambda value: value)
    monkeypatch.setattr(aclgraph_utils.torch.npu, "current_stream", lambda: object())
    monkeypatch.setattr(ModelCudaGraphManager, "run_fullgraph", lambda *args: calls.append("replay"))
    manager._updatable_graph_replay(desc, metadata)
    assert calls == ["resolve_group_tasks", "wait_previous_compute", "replay", "update_group_tasks"]


def test_stage_attention_builders_are_new_and_preserve_kv_group_ids(monkeypatch):
    class Group:
        def __init__(self, backend, names, spec, group_id):
            self.layer_names = names
            self.kv_cache_group_id = group_id
            self.builder = None

        def create_metadata_builders(self, *args, **kwargs):
            self.builder = object()

    original = SimpleNamespace(
        backend=object(),
        layer_names=["model.layers.0.self_attn", "model.layers.16.self_attn"],
        kv_cache_spec=object(),
        kv_cache_group_id=1,
    )
    runner = SimpleNamespace(vllm_config=None, device=None, attn_groups=[[], [original]], kernel_block_sizes=[64, 128])
    monkeypatch.setattr(sawtooth_graph, "AttentionGroup", Group)
    monkeypatch.setattr(sawtooth_graph, "set_current_vllm_config", lambda _: nullcontext())
    first = sawtooth_graph.make_stage_attention_groups(runner, SimpleNamespace(start_layer=0, end_layer=8))
    second = sawtooth_graph.make_stage_attention_groups(runner, SimpleNamespace(start_layer=16, end_layer=24))
    assert first[0] == second[0] == []
    assert first[1][0].layer_names == ["model.layers.0.self_attn"]
    assert second[1][0].layer_names == ["model.layers.16.self_attn"]
    assert first[1][0].kv_cache_group_id == second[1][0].kv_cache_group_id == 1
    assert first[1][0].builder is not second[1][0].builder


def test_stage_runtime_restores_runner_after_exception_and_rejects_nested_scope():
    runtime = object.__new__(sawtooth_runtime.SawtoothRunnerRuntime)
    fields = {name: object() for name in runtime._RUNNER_FIELDS}
    runner = SimpleNamespace(**fields, kv_cache_config=object(), req_states=SimpleNamespace(req_id_to_index={}))
    stage = SimpleNamespace(is_first=False, is_last=False)
    state = SimpleNamespace()
    manager = SimpleNamespace(activate=lambda: nullcontext())
    resource = sawtooth_runtime.SawtoothStageResources(
        stage,
        object(),
        SimpleNamespace(model_state=state, attn_groups=[]),
        object(),
        object(),
        manager,
    )
    runtime.__dict__.update(
        runner=runner,
        base_model_state=SimpleNamespace(),
        stages={1: resource},
        _active_round=None,
    )
    with pytest.raises(RuntimeError, match="model failure"), runtime.activate(1) as active:
        assert active is resource
        assert runner.model_state is state
        assert runner.cudagraph_manager is manager
        assert runner.is_first_pp_rank is False
        with pytest.raises(RuntimeError, match="cannot be nested"), runtime.activate(1):
            pass
        raise RuntimeError("model failure")
    for name, value in fields.items():
        assert getattr(runner, name) is value
    assert runtime._active_round is None


def test_runtime_late_metadata_binding_allocates_independent_executors():
    class Metadata:
        pass

    runtime = object.__new__(sawtooth_runtime.SawtoothRunnerRuntime)
    states = [SimpleNamespace(), SimpleNamespace()]
    base = SimpleNamespace(device_metadata=None, kvpp_runtime=object())
    runtime.__dict__.update(
        runner=SimpleNamespace(kv_cache_config=object(), req_states=SimpleNamespace(req_id_to_index={})),
        base_model_state=base,
        stages={index: SimpleNamespace(model_state=state) for index, state in enumerate(states)},
    )
    runtime.refresh_shared_state()
    base.device_metadata = Metadata()
    runtime.refresh_shared_state()
    assert states[0].device_metadata is not states[1].device_metadata
    assert states[0].device_metadata is not base.device_metadata
    assert states[0].kvpp_runtime is states[1].kvpp_runtime is base.kvpp_runtime
    first_executor = states[0].device_metadata
    runtime.refresh_shared_state()
    assert states[0].device_metadata is first_executor


def test_profile_scope_needs_no_kv_metadata_and_preserves_dummy_query_flags():
    runtime = object.__new__(sawtooth_runtime.SawtoothProfileRuntime)
    fields = {name: object() for name in runtime._RUNNER_FIELDS if name != "attn_groups"}
    fields["input_buffers"] = SimpleNamespace(dummy_num_tokens=3, profile_cpp=True)
    runner = SimpleNamespace(**fields)
    inputs = SimpleNamespace(dummy_num_tokens=None, profile_cpp=False)
    resource = sawtooth_runtime.SawtoothStageResources(
        SimpleNamespace(is_first=False, is_last=True),
        object(),
        SimpleNamespace(model_state=SimpleNamespace(), attn_groups=[]),
        inputs,
        object(),
        None,
    )
    runtime.__dict__.update(runner=runner, base_model_state=SimpleNamespace(), stages={1: resource}, _active_round=None)
    with runtime.activate(1):
        assert runner.cudagraph_manager is None
        assert runner.attn_groups == []
        assert runner.input_buffers.dummy_num_tokens == 3
        assert runner.input_buffers.profile_cpp is True
    assert not hasattr(runner, "attn_groups")
    assert inputs.dummy_num_tokens is None
    assert inputs.profile_cpp is False


def test_executable_runtime_reuses_profile_state_and_buffer_addresses(monkeypatch):
    stage = SimpleNamespace(group_id=2, round_id=0, is_first=False, is_last=False)
    plan = SimpleNamespace(plan_id="plan", local_stages=lambda _: [stage])
    model = object()
    state = SimpleNamespace()
    profile = sawtooth_runtime.SawtoothStageResources(
        stage,
        model,
        SimpleNamespace(model_state=state),
        object(),
        object(),
        None,
    )
    original = SimpleNamespace(sawtooth_plan=plan, get_sawtooth_stage=lambda _: model)
    runner = SimpleNamespace(
        pcp_manager=None,
        ubatch_runner=None,
        model=original,
        model_state=SimpleNamespace(),
        sawtooth_profile_runtime=SimpleNamespace(plan=plan, stages={0: profile}),
        jit_warmup_registry=SimpleNamespace(activate=lambda: nullcontext()),
        vllm_config=object(),
        device="cpu",
        kv_cache_config=object(),
        req_states=SimpleNamespace(req_id_to_index={}),
    )
    monkeypatch.setattr(sawtooth_runtime, "get_pp_group", lambda: SimpleNamespace(rank_in_group=0))
    monkeypatch.setattr(
        sawtooth_runtime,
        "make_stage_runner_view",
        lambda *args, model_state: SimpleNamespace(model_state=model_state, attn_groups=[]),
    )
    monkeypatch.setattr(sawtooth_runtime, "SawtoothAclGraphManager", lambda *args, **kwargs: object())
    monkeypatch.setattr(sawtooth_runtime, "AscendInputBuffers", Mock(side_effect=AssertionError("reallocation")))
    runtime = sawtooth_runtime.SawtoothRunnerRuntime(runner, CUDAGraphMode.FULL, 1)
    resource = runtime.stages[0]
    assert resource.model_state is state
    assert resource.input_buffers is profile.input_buffers
    assert resource.intermediate_tensors is profile.intermediate_tensors
