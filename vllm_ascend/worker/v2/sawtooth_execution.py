# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Runner-side logical admission and fixed virtual-stage execution."""

from vllm.sequence import IntermediateTensors
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT
from vllm.v1.worker.gpu.sawtooth import KeyedPPHandler


def _admit(runner, output, batch_id):
    if not isinstance(runner.pp_handler, KeyedPPHandler):
        runner.pp_handler = KeyedPPHandler(runner.pp_handler)
    cleanup = output.finished_req_ids | set(output.preempted_req_ids or ())
    # A resumed/streaming NewRequestData replaces an old worker allocation even
    # when it is re-added in this same command. Consume its old dependency first.
    cleanup |= {data.req_id for data in output.scheduled_new_reqs if data.req_id in runner.pp_handler.current_steps}
    if cleanup:
        runner.pp_handler.begin_batch(batch_id, cleanup, runner, {})
        runner.pp_handler.forget_requests(cleanup)
    runner.pp_handler.begin_batch(
        batch_id,
        output.num_scheduled_tokens,
        runner,
        getattr(output, "sawtooth_request_steps", None),
    )
    runner.finish_requests(output)
    runner.free_states(output)
    # Only admission produces staged sampler parameters. Calling add_requests
    # with an empty list still flushes every static sampler UVA buffer upstream;
    # on Ascend's fallback this copies unchanged parameters to the device.
    # Sampled-token state and per-stage writes remain independent below.
    if output.scheduled_new_reqs:
        runner.add_requests(output)
    runner.update_requests(output)
    runner.block_tables.apply_staged_writes()

    runtime = getattr(runner, "sawtooth_runtime", None)
    if runtime is not None:
        for state in runtime.stage_states:
            for req_id in output.finished_req_ids | set(output.preempted_req_ids or ()):
                state.remove_request(req_id)
            for data in output.scheduled_new_reqs:
                index = runner.req_states.req_id_to_index[data.req_id]
                state.add_request(index, data)
            state.apply_staged_writes()


def execute_empty(runner, output):
    _admit(runner, output, -1)
    return EMPTY_MODEL_RUNNER_OUTPUT


def execute_round(runner, output, intermediate):
    execution = output.sawtooth_execution
    if execution.is_first_round:
        _admit(runner, output, execution.batch_id)
    steps = output.sawtooth_request_steps
    if steps is None or set(steps) != set(output.num_scheduled_tokens):
        raise RuntimeError("Sawtooth round has no immutable request input versions")
    # A successor can change the CPU view only after its predecessor's final
    # execute + sample, so all previous virtual-stage preparations are already
    # submitted. Fixed graph input/table writes remain compute-stream ordered.
    for rid, step in steps.items():
        index = runner.req_states.req_id_to_index[rid]
        if (
            runner.pp_handler.current_steps.get(rid) != step
            or runner.req_states.num_computed_tokens_np[index] != step.token_start
            or output.num_scheduled_tokens[rid] != step.token_end - step.token_start
        ):
            raise RuntimeError("Sawtooth continuation input version changed")
    runner.pp_handler.batch_id = execution.batch_id
    runtime = runner.sawtooth_runtime
    with runtime.activate(execution.round_id):
        runner._sawtooth_skip_lifecycle = True
        runner._sawtooth_active = True
        try:
            result = runner.execute_model(output, intermediate)
        finally:
            runner._sawtooth_skip_lifecycle = False
            runner._sawtooth_active = False
    if execution.is_last_round:
        runner.pp_handler.sample_steps = steps
    if isinstance(result, IntermediateTensors):
        num_tokens = output.total_num_scheduled_tokens
        return IntermediateTensors({name: value[:num_tokens] for name, value in result.tensors.items()})
    return result


def complete_sample(runner):
    runner.pp_handler.complete_sample()


def execute_dummy(runner, output, **kwargs):
    result = None
    runtime = getattr(runner, "sawtooth_runtime", None)
    if runtime is None:
        runtime = runner.sawtooth_profile_runtime
    for round_id in range(runtime.plan.virtual_size):
        with runtime.activate(round_id):
            runner._sawtooth_active = True
            try:
                result = runner.execute_model(output, runner.intermediate_tensors, **kwargs)
            finally:
                runner._sawtooth_active = False
    return result
