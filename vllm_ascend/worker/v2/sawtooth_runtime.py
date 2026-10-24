# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Persistent resources and scoped runner views for fixed sawtooth stages."""

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import torch
from vllm.config import set_current_vllm_config
from vllm.distributed import get_pp_group
from vllm.sequence import IntermediateTensors

from vllm_ascend.worker.v2.input_batch import AscendInputBuffers
from vllm_ascend.worker.v2.sawtooth_graph import SawtoothAclGraphManager, make_stage_runner_view


@dataclass
class SawtoothStageResources:
    stage: Any
    model: Any
    view: Any
    input_buffers: AscendInputBuffers
    intermediate_tensors: IntermediateTensors | None
    manager: SawtoothAclGraphManager | None

    @property
    def model_state(self):
        return self.view.model_state


class SawtoothRunnerRuntime:
    """Graph-manager facade plus one persistent resource set per local group.

    ``activate(round_id)`` changes only runner references. Physical PP request
    admission, token broadcasts and final completion must remain outside this
    scope. The runner is single-threaded and submits all groups to its compute
    stream; transport streams only consume independently leased outputs.
    """

    _RUNNER_FIELDS = (
        "model",
        "model_state",
        "input_buffers",
        "intermediate_tensors",
        "attn_groups",
        "cudagraph_manager",
        "is_first_pp_rank",
        "is_last_pp_rank",
    )

    def __init__(self, runner, cudagraph_mode, decode_query_len, lora_capture_cases=None, varlen_decode=False):
        if runner.pcp_manager is not None or runner.ubatch_runner is not None:
            raise ValueError("Sawtooth stage runtime does not yet support PCP or microbatch graph execution")
        self.runner = runner
        self.original_model = runner.model
        self.base_model_state = runner.model_state
        self.plan = self.original_model.sawtooth_plan
        if self.plan is None:
            raise ValueError("Sawtooth stage runtime requires a model with a fixed stage plan")
        self.pp_rank = get_pp_group().rank_in_group
        self.stages: dict[int, SawtoothStageResources] = {}
        self._active_round: int | None = None
        self._max_full_descs_to_capture = None
        self._capture_mem_samples = None
        profile_runtime = getattr(runner, "sawtooth_profile_runtime", None)
        if profile_runtime is not None and profile_runtime.plan.plan_id != self.plan.plan_id:
            raise ValueError("Profile and executable sawtooth stage plans differ")
        for stage in self.plan.local_stages(self.pp_rank):
            model = self.original_model.get_sawtooth_stage(stage.group_id)
            profile = profile_runtime.stages[stage.round_id] if profile_runtime is not None else None
            # Builders may register kernel warmup providers during allocation.
            with runner.jit_warmup_registry.activate():
                view = make_stage_runner_view(
                    runner,
                    stage,
                    model,
                    model_state=profile.model_state if profile else None,
                )
            input_buffers = (
                profile.input_buffers
                if profile
                else AscendInputBuffers(runner.max_num_reqs, runner.max_num_tokens, runner.device)
            )
            intermediate = profile.intermediate_tensors if profile else None
            if not stage.is_first and profile is None:
                # The incoming schema belongs to the preceding GLOBAL group,
                # including model-specific residual/indexer carry tensors.
                schema = self.original_model.get_sawtooth_carry_schema(stage.group_id - 1, runner.max_num_tokens)
                intermediate = IntermediateTensors(
                    {
                        name: torch.zeros(shape, dtype=dtype, device=runner.device)
                        for name, (shape, dtype) in schema.items()
                    }
                )
                if not intermediate.tensors:
                    raise ValueError(f"Empty incoming carry schema for sawtooth group {stage.group_id}")
            manager = SawtoothAclGraphManager(
                runner.vllm_config,
                runner.device,
                cudagraph_mode,
                decode_query_len,
                view,
                stage=stage,
                plan_version=self.plan.plan_id,
                model=model,
                lora_capture_cases=lora_capture_cases,
                varlen_decode=varlen_decode,
            )
            self.stages[stage.round_id] = SawtoothStageResources(
                stage, model, view, input_buffers, intermediate, manager
            )
        self.refresh_shared_state()

    @property
    def stage_states(self):
        """New request lifecycle updates must reach each of these states once."""
        return tuple(resource.model_state for resource in self.stages.values())

    @property
    def _last_manager(self):
        return self.stages[self.plan.virtual_size - 1].manager

    def __getattr__(self, name):
        # This facade is only used outside active group execution. Inside
        # activate the runner directly holds the exact group's manager.
        return getattr(self._last_manager, name)

    @property
    def _capture_descs(self):
        result = {}
        for resource in self.stages.values():
            for mode, descs in resource.manager._capture_descs.items():
                result.setdefault(mode, []).extend(descs)
        return result

    def needs_capture(self):
        return any(resource.manager.needs_capture() for resource in self.stages.values())

    def captured_token_counts(self):
        sizes = [set(resource.manager.captured_token_counts()) for resource in self.stages.values()]
        return sorted(set.intersection(*sizes)) if sizes else []

    def refresh_shared_state(self):
        """Bind controllers installed after the graph-manager factory returns.

        Attention/build buffers are never copied. In particular the base state
        gets its metadata executor later in Ascend.initialize_kv_cache, so each
        group creates its own executor lazily before capture or first use.
        """
        for resource in self.stages.values():
            state = resource.model_state
            state.kv_cache_config = self.runner.kv_cache_config
            state.kvpp_runtime = getattr(self.base_model_state, "kvpp_runtime", None)
            state.kvpp_is_dummy_run = getattr(self.base_model_state, "kvpp_is_dummy_run", False)
            state._offload_live_req_ids = self.runner.req_states.req_id_to_index
            source_metadata = getattr(self.base_model_state, "device_metadata", None)
            if source_metadata is not None and getattr(state, "device_metadata", None) is None:
                state.device_metadata = type(source_metadata)()

    @contextmanager
    def activate(self, round_id):
        if round_id not in self.stages:
            raise ValueError(f"Invalid sawtooth round {round_id}")
        if self._active_round is not None:
            raise RuntimeError("Sawtooth runner stage scopes cannot be nested")
        self.refresh_shared_state()
        resource = self.stages[round_id]
        runner = self.runner
        missing = object()
        previous = {name: getattr(runner, name, missing) for name in self._RUNNER_FIELDS}
        # _dummy_run sets these on the physical runner's input buffers before
        # execute_dummy selects a group. Preserve the same query semantics.
        flags = ("dummy_num_tokens", "profile_cpp")
        previous_flags = {name: getattr(resource.input_buffers, name, missing) for name in flags}
        selected = {
            "model": resource.model,
            "model_state": resource.model_state,
            "input_buffers": resource.input_buffers,
            "intermediate_tensors": resource.intermediate_tensors,
            "attn_groups": resource.view.attn_groups,
            "cudagraph_manager": resource.manager,
            "is_first_pp_rank": resource.stage.is_first,
            "is_last_pp_rank": resource.stage.is_last,
        }
        self._active_round = round_id
        try:
            for name in flags:
                if hasattr(runner.input_buffers, name):
                    setattr(resource.input_buffers, name, getattr(runner.input_buffers, name))
            for name, value in selected.items():
                setattr(runner, name, value)
            with resource.manager.activate() if resource.manager is not None else nullcontext():
                yield resource
        finally:
            for name, value in previous.items():
                if value is missing:
                    delattr(runner, name)
                else:
                    setattr(runner, name, value)
            for name, value in previous_flags.items():
                if value is not missing:
                    setattr(resource.input_buffers, name, value)
                elif hasattr(resource.input_buffers, name):
                    delattr(resource.input_buffers, name)
            self._active_round = None

    def capture(self, original_model, *args, **kwargs):
        if original_model is not self.original_model:
            raise ValueError("Sawtooth capture facade received a different root model")
        self.refresh_shared_state()
        # Forward only capture options. Buffer/state arguments from the root
        # runner belong to physical PP and must never enter a stage capture.
        capture_options = {
            name: kwargs[name]
            for name in (
                "has_lora",
                "use_aux_hidden_state_outputs",
                "lora_capture_hook",
                "progress_bar_desc",
            )
            if name in kwargs
        }
        for round_id, resource in self.stages.items():
            resource.manager._max_full_descs_to_capture = self._max_full_descs_to_capture
            resource.manager._capture_mem_samples = self._capture_mem_samples
            with self.activate(round_id):
                resource.manager.capture(
                    resource.model,
                    resource.model_state,
                    resource.input_buffers,
                    resource.intermediate_tensors,
                    self.runner.block_tables,
                    resource.view.attn_groups,
                    self.runner.kv_cache_config,
                    **capture_options,
                )


class SawtoothProfileRuntime(SawtoothRunnerRuntime):
    """Pre-KV stage resources, reused by the executable runtime after profiling.

    Only skip-attention profile runs may use this scope. No graph manager,
    metadata builder or KV cache is available until initialize_kv_cache.
    """

    def __init__(self, runner):
        self.runner = runner
        self.original_model = runner.model
        self.base_model_state = runner.model_state
        self.plan = runner.model.sawtooth_plan
        self.pp_rank = get_pp_group().rank_in_group
        self.stages = {}
        self._active_round = None
        for stage in self.plan.local_stages(self.pp_rank):
            model = self.original_model.get_sawtooth_stage(stage.group_id)
            with set_current_vllm_config(runner.vllm_config):
                state = type(self.base_model_state)(runner.vllm_config, model, runner.encoder_cache, runner.device)
            inputs = AscendInputBuffers(runner.max_num_reqs, runner.max_num_tokens, runner.device)
            carry = None
            if not stage.is_first:
                schema = self.original_model.get_sawtooth_carry_schema(stage.group_id - 1, runner.max_num_tokens)
                carry = IntermediateTensors(
                    {
                        name: torch.zeros(shape, dtype=dtype, device=runner.device)
                        for name, (shape, dtype) in schema.items()
                    }
                )
            view = SimpleNamespace(model_state=state, attn_groups=[])
            self.stages[stage.round_id] = SawtoothStageResources(stage, model, view, inputs, carry, None)

    def refresh_shared_state(self):
        for resource in self.stages.values():
            resource.model_state.kvpp_runtime = getattr(self.base_model_state, "kvpp_runtime", None)


def create_profile_runtime(runner):
    """Call after load_model, before profile_run, to reserve all stage buffers."""
    return SawtoothProfileRuntime(runner)
