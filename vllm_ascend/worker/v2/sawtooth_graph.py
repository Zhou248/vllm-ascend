# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fixed-stage ACL graphs; transport slots never form part of graph identity.

The runner owns admission, restores batch inputs, and enters ``activate`` before
preparing attention. All groups must execute on one compute stream. Outputs
must be copied to a batch-owned lease before another graph is submitted.
"""

import re
from contextlib import contextmanager
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import torch
from vllm.config import set_current_vllm_config
from vllm.config.compilation import CUDAGraphMode
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.utils import AttentionGroup

from vllm_ascend.compilation.acl_graph import (
    GraphParamsScope,
    bind_graph_params_descriptor,
    get_graph_params_scope,
    set_graph_params,
    use_graph_params_scope,
)
from vllm_ascend.worker.v2.aclgraph_utils import ModelAclGraphManager


@dataclass(frozen=True)
class SawtoothGraphKey:
    plan_version: Any
    group_id: int
    descriptor: BatchExecutionDescriptor


class SawtoothAclGraphManager(ModelAclGraphManager):
    """One immutable stage, private pool, and descriptor-scoped attention tasks.

    FULL/FULL_DECODE_ONLY use real ACL graphs. PIECEWISE uses the stage's
    independently compiled module or a private breakable runner. The stage
    module must include its group identity in the compilation cache key.
    """

    def __init__(
        self,
        vllm_config,
        device,
        cudagraph_mode,
        decode_query_len,
        model_runner,
        *,
        stage,
        plan_version,
        model,
        lora_capture_cases=None,
        varlen_decode=False,
    ):
        self.stage = stage
        self.stage_model = model
        self.plan_version = plan_version
        scope = GraphParamsScope((plan_version, stage.group_id), current_platform.graph_pool_handle())
        super().__init__(
            vllm_config,
            device,
            cudagraph_mode,
            decode_query_len,
            model_runner,
            lora_capture_cases=lora_capture_cases,
            varlen_decode=varlen_decode,
            graph_scope=scope,
        )
        self.is_first_pp_rank = stage.is_first
        self.is_last_pp_rank = stage.is_last
        self.pool = scope.graph_pool

    def graph_key(self, desc):
        return SawtoothGraphKey(self.plan_version, self.stage.group_id, desc)

    def dispatch(self, *args, **kwargs):
        desc = super().dispatch(*args, **kwargs)
        if get_graph_params_scope() is self.graph_scope:
            # Dispatch happens inside runtime.activate before metadata builds.
            # Keep that exact descriptor through preparation and PW execution.
            bind_graph_params_descriptor(self.graph_scope, desc)
        return desc

    @contextmanager
    def activate(self, desc=None):
        with use_graph_params_scope(self.graph_scope, desc), set_current_vllm_config(self.vllm_config):
            yield

    def init_breakable_cg_runner(self, model):
        super().init_breakable_cg_runner(model)
        self.breakable_cg_runner.graph_pool = self.pool

    def capture(self, model, *args, **kwargs):
        if model is not self.stage_model:
            raise ValueError("A sawtooth graph manager must capture its fixed stage callable")
        original_descs = self._capture_descs
        try:
            # Keep the entire original descriptor, including query/LoRA/ubatch
            # variants that can collide on num_tokens, in registry identity.
            for mode in (CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL):
                descs = original_descs.get(mode, ())
                if mode == CUDAGraphMode.FULL and self._max_full_descs_to_capture is not None:
                    descs = descs[: self._max_full_descs_to_capture]
                for desc in descs:
                    self._capture_descs = {mode: [desc]}
                    with self.activate(desc):
                        set_graph_params([desc.num_tokens])
                        super().capture(model, *args, **kwargs)
        finally:
            self._capture_descs = original_descs

    def run_pw_graph(self, model, model_inputs):
        # Capture wraps this exact callable with ModelWithContext; ordinary
        # replay uses stage_model. Never select layers by changing attributes.
        if model is not self.stage_model and getattr(model, "original_model", None) is not self.stage_model:
            raise ValueError("PIECEWISE replay selected a different stage callable")
        return super().run_pw_graph(model, model_inputs)

    def run_fullgraph_to_lease(self, desc, lease_output, *, ready_event=None):
        """Replay/update, then save outputs before their reusable storage changes."""
        return stage_output_to_lease(self.run_fullgraph(desc), lease_output, ready_event=ready_event)


def make_stage_attention_groups(runner, stage):
    """Reconstruct builders; never copy tensors or events from another group."""
    result = []
    with set_current_vllm_config(runner.vllm_config):
        for groups in runner.attn_groups:
            stage_groups = []
            for group in groups:
                names = []
                for name in group.layer_names:
                    match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", name)
                    if match is None:
                        raise ValueError(f"Cannot assign attention layer to a sawtooth stage: {name}")
                    if stage.start_layer <= int(match.group(1)) < stage.end_layer:
                        names.append(name)
                if not names:
                    continue
                new_group = AttentionGroup(group.backend, names, group.kv_cache_spec, group.kv_cache_group_id)
                new_group.create_metadata_builders(
                    runner.vllm_config,
                    runner.device,
                    kernel_block_size=(
                        runner.kernel_block_sizes[group.kv_cache_group_id]
                        if group.kv_cache_group_id < len(runner.kernel_block_sizes)
                        else None
                    ),
                )
                stage_groups.append(new_group)
            result.append(stage_groups)
    return result


def make_stage_runner_view(runner, stage, model, *, model_state=None):
    """Allocate group-owned ModelState/builders before capture or execution.

    KV allocation and immutable configuration are shared. Metadata buffers,
    producer events and attention-update task sources are newly constructed.
    The caller also allocates stable input/carry buffers per group.
    """
    with set_current_vllm_config(runner.vllm_config):
        state = model_state
        if state is None:
            state = type(runner.model_state)(runner.vllm_config, model, runner.encoder_cache, runner.device)
        metadata = getattr(runner.model_state, "device_metadata", None)
        if metadata is not None:
            state.device_metadata = type(metadata)()
        groups = make_stage_attention_groups(runner, stage)
    return SimpleNamespace(
        model_state=state,
        attn_groups=groups,
        update_stream=runner.update_stream,
        dp_size=runner.dp_size,
        speculative_config=None,
        pcp_manager=None,
    )


def stage_output_to_lease(output, lease_output, *, ready_event=None):
    """D2D save after replay on the same compute stream, including final hidden.

    The caller owns the destination until send/sampling completion. This does
    not wait for network sends, and does not allocate a fresh buffer per step.
    """
    if isinstance(output, IntermediateTensors):
        if not isinstance(lease_output, IntermediateTensors) or output.tensors.keys() != lease_output.tensors.keys():
            raise ValueError("Sawtooth transport lease does not match the stage payload schema")
        for name, value in output.tensors.items():
            _copy_lease_tensor(value, lease_output[name])
    elif isinstance(output, torch.Tensor) and isinstance(lease_output, torch.Tensor):
        _copy_lease_tensor(output, lease_output)
    else:
        raise TypeError("Sawtooth supports Tensor or IntermediateTensors stage outputs")
    if ready_event is not None:
        ready_event.record(torch.npu.current_stream())
    return lease_output


def _copy_lease_tensor(source, destination):
    if source.shape != destination.shape or source.dtype != destination.dtype or source.device != destination.device:
        raise ValueError("Sawtooth lease shape, dtype and device must match its graph output")
    if source.untyped_storage().data_ptr() == destination.untyped_storage().data_ptr():
        raise ValueError("Sawtooth transport lease must not alias a reusable graph output")
    destination.copy_(source, non_blocking=True)
