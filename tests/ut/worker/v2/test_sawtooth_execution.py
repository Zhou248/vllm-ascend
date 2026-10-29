# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from collections import deque
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest
import numpy as np
import torch
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT
from vllm.v1.sawtooth import SawtoothExecution, SawtoothRequestStep
from vllm.v1.worker.gpu.sawtooth import KeyedPPHandler

from vllm_ascend.worker.v2.sawtooth_execution import _admit, execute_empty, execute_round


def test_interleaved_prompt_chunks_restore_each_groups_own_positions():
    runner = make_runner()
    runner.pp_handler.allow_prefill_overlap = True
    runner.pp_handler.native.is_last_rank = True
    runner.pp_handler.native.main_stream = SimpleNamespace(
        record_event=lambda: SimpleNamespace(query=lambda: True),
        wait_event=Mock(),
    )
    runner.req_states.req_id_to_index = {"r": 0}
    runner.req_states.num_computed_tokens_np = np.zeros(1, dtype=np.int32)
    runner.req_states.num_computed_prefill_tokens = np.zeros(1, dtype=np.int32)
    runner.req_states.prefill_len = SimpleNamespace(np=np.array([16384]))
    runner.req_states.num_computed_tokens = SimpleNamespace(
        gpu=torch.zeros(1, dtype=torch.int32)
    )
    runner.sawtooth_runtime.activate = lambda _: nullcontext()
    observed = []

    def forward(output, intermediate):
        observed.append((
            output.sawtooth_execution.batch_id,
            output.sawtooth_execution.round_id,
            int(runner.req_states.num_computed_tokens_np[0]),
            int(runner.req_states.num_computed_tokens.gpu[0]),
        ))

    runner.execute_model = forward
    steps = [
        SawtoothRequestStep("r", 0, 0, 0, 10240, False),
        SawtoothRequestStep("r", 0, 1, 10240, 16384, True),
    ]
    for seq, (batch, group) in enumerate([(0, 0), (1, 0), (0, 1), (1, 1)]):
        step = steps[batch]
        output = make_output(scheduled={"r": step.token_end - step.token_start})
        output.sawtooth_request_steps = {"r": step}
        output.sawtooth_execution = SawtoothExecution(batch, group, 2, seq, batch, 1, "p")
        execute_round(runner, output, None)
        if group == 1:
            runner.pp_handler.complete_sample()
    assert observed == [
        (0, 0, 0, 0), (1, 0, 10240, 10240),
        (0, 1, 0, 0), (1, 1, 10240, 10240),
    ]


def make_runner():
    native = SimpleNamespace(queue=deque(), get_prev_sampled_outputs=Mock())
    runner = SimpleNamespace(
        pp_handler=KeyedPPHandler(native),
        finish_requests=Mock(),
        free_states=Mock(),
        update_requests=Mock(),
        postprocess_sampled=Mock(),
        sampler=SimpleNamespace(apply_staged_writes=Mock()),
        block_tables=SimpleNamespace(apply_staged_writes=Mock()),
        req_states=SimpleNamespace(req_id_to_index={}, draft_tokens=object()),
        sawtooth_runtime=SimpleNamespace(stage_states=[Mock(), Mock()]),
    )

    def add_requests(output):
        for data in output.scheduled_new_reqs:
            runner.req_states.req_id_to_index[data.req_id] = 7
        # Upstream add_requests flushes static sampler parameters even when its
        # input list is empty. The admission guard must prevent this call.
        runner.sampler.apply_staged_writes()

    runner.add_requests = Mock(side_effect=add_requests)
    return runner


def make_output(*, scheduled=None, new=(), finished=(), preempted=()):
    return SimpleNamespace(
        num_scheduled_tokens=scheduled or {},
        scheduled_new_reqs=list(new),
        finished_req_ids=set(finished),
        preempted_req_ids=set(preempted),
    )


@pytest.mark.parametrize("kind", ["decode", "empty", "cleanup"])
def test_no_new_requests_preserves_updates_without_static_sampler_flush(kind):
    runner = make_runner()
    output = make_output(
        scheduled={"live": 1} if kind == "decode" else {},
        finished={"finished"} if kind == "cleanup" else (),
        preempted={"preempted"} if kind == "cleanup" else (),
    )
    if kind != "empty":
        needed = "live" if kind == "decode" else "finished"
        pending = SimpleNamespace(event=SimpleNamespace(query=lambda: True))
        runner.pp_handler.pending[3] = ((needed,), pending)
        runner.pp_handler.native.get_prev_sampled_outputs.return_value = {"sampled_tokens": object()}

    if kind == "decode":
        _admit(runner, output, 4)
    else:
        assert execute_empty(runner, output) is EMPTY_MODEL_RUNNER_OUTPUT

    runner.add_requests.assert_not_called()
    runner.sampler.apply_staged_writes.assert_not_called()
    runner.finish_requests.assert_called_once_with(output)
    runner.free_states.assert_called_once_with(output)
    runner.update_requests.assert_called_once_with(output)
    runner.block_tables.apply_staged_writes.assert_called_once_with()
    for state in runner.sawtooth_runtime.stage_states:
        state.add_request.assert_not_called()
        state.apply_staged_writes.assert_called_once_with()
        assert set(c.args[0] for c in state.remove_request.call_args_list) == (
            output.finished_req_ids | output.preempted_req_ids
        )
    if kind != "empty":
        runner.postprocess_sampled.assert_called_once_with(
            **runner.pp_handler.native.get_prev_sampled_outputs.return_value
        )
        assert not runner.pp_handler.pending


def test_new_request_flushes_sampler_and_stage_state_before_later_decode():
    runner = make_runner()
    data = SimpleNamespace(req_id="new")
    admission = make_output(scheduled={"new": 3}, new=[data])
    _admit(runner, admission, 0)

    runner.add_requests.assert_called_once_with(admission)
    runner.sampler.apply_staged_writes.assert_called_once_with()
    for state in runner.sawtooth_runtime.stage_states:
        state.add_request.assert_called_once_with(7, data)
        state.apply_staged_writes.assert_called_once_with()

    decode = make_output(scheduled={"new": 1})
    _admit(runner, decode, 1)
    # The subsequent logical step retains the admission's static parameters;
    # persistent request/block-table updates still happen on both steps.
    runner.add_requests.assert_called_once_with(admission)
    runner.sampler.apply_staged_writes.assert_called_once_with()
    assert runner.update_requests.call_args_list == [call(admission), call(decode)]
    assert runner.block_tables.apply_staged_writes.call_count == 2
    for state in runner.sawtooth_runtime.stage_states:
        assert state.apply_staged_writes.call_count == 2


@pytest.mark.parametrize("changed", [None, "position", "step", "count"])
def test_continuation_checks_immutable_input_version_before_forward(changed):
    runner = make_runner()
    step = SawtoothRequestStep("r", 0, 0, 3, 4, True)
    runner.req_states.req_id_to_index = {"r": 0}
    runner.req_states.num_computed_tokens_np = [4 if changed == "position" else 3]
    runner.pp_handler.current_steps = {"r": step}
    if changed == "step":
        runner.pp_handler.current_steps["r"] = SawtoothRequestStep("r", 0, 1, 4, 5, True)
    runner.sawtooth_runtime.activate = lambda _: nullcontext()
    runner.execute_model = Mock(return_value=None)
    output = make_output(scheduled={"r": 2 if changed == "count" else 1})
    output.sawtooth_request_steps = {"r": step}
    output.sawtooth_execution = SawtoothExecution(0, 1, 2, 1, 0, 1, "p")
    if changed is None:
        assert execute_round(runner, output, None) is None
        runner.execute_model.assert_called_once_with(output, None)
        assert runner.pp_handler.sample_steps == {"r": step}
    else:
        with pytest.raises(RuntimeError, match="input version"):
            execute_round(runner, output, None)
        runner.execute_model.assert_not_called()
