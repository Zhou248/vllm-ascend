# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from collections import defaultdict, deque
from types import SimpleNamespace

import pytest
import torch
from vllm.distributed.sawtooth import SawtoothConfig, StagePlan
from vllm.sequence import IntermediateTensors
from vllm.v1.sawtooth import SawtoothExecution

from vllm_ascend.worker.v2 import sawtooth_transport
from vllm_ascend.worker.v2.sawtooth_transport import RoundLedger, SawtoothTransport


def command(plan, batch, round_id, sequence, generation=0):
    return SawtoothExecution(
        batch_id=batch,
        round_id=round_id,
        num_rounds=plan.virtual_size,
        command_seq=sequence,
        slot_id=batch % 2,
        slot_generation=generation,
        plan_id=plan.plan_id,
    )


def test_round_ledger_rejects_replay_and_live_slot_reuse():
    plan = StagePlan.build(SawtoothConfig(True, 3), 12, 2)
    ledger = RoundLedger(plan.plan_id, 3, 2)
    ledger.accept(command(plan, 0, 0, 0))
    ledger.accept(command(plan, 1, 0, 1))
    with pytest.raises(ValueError, match="Duplicate"):
        ledger.accept(command(plan, 0, 0, 0))
    with pytest.raises(ValueError, match="live"):
        ledger.accept(command(plan, 2, 0, 2, generation=1))
    with pytest.raises(ValueError, match="predecessor"):
        ledger.accept(command(plan, 0, 2, 2))
    ledger.accept(command(plan, 0, 1, 2))
    ledger.accept(command(plan, 0, 2, 3))
    ledger.accept(command(plan, 2, 0, 4, generation=1))
    assert len(ledger.slots) == 2


@pytest.mark.parametrize(
    "preferred,legacy,valid",
    [
        (None, None, True),
        (None, "0", True),
        ("0", "1", True),
        ("1", "0", False),
        (None, "1", False),
        ("yes", None, False),
    ],
)
def test_hccl_wait_guard_respects_torch_npu_environment_precedence(monkeypatch, preferred, legacy, valid):
    monkeypatch.setattr(
        sawtooth_transport,
        "envs",
        SimpleNamespace(TORCH_HCCL_BLOCKING_WAIT=preferred, HCCL_BLOCKING_WAIT=legacy),
    )
    if valid:
        sawtooth_transport._validate_nonblocking_hccl()
    else:
        with pytest.raises(ValueError, match="nonblocking"):
            sawtooth_transport._validate_nonblocking_hccl()


class Event:
    def record(self, stream):
        pass


class Stream:
    def wait_event(self, event):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class Work:
    def __init__(self):
        self.done = False

    def is_completed(self):
        return self.done

    def block_current_stream(self):
        raise RuntimeError("Failed to create StreamBlock")

    def wait(self):
        assert self.done, "Mock attempted to consume an unmatched receive"


class Fabric:
    """FIFO peers without message tags; intentionally catches V>2 misordering."""

    def __init__(self):
        self.rank = 0
        self.send_queues = defaultdict(deque)
        self.recv_queues = defaultdict(deque)
        self.posts = []

    def match(self, key):
        while self.send_queues[key] and self.recv_queues[key]:
            source, send_work = self.send_queues[key].popleft()
            target, recv_work = self.recv_queues[key].popleft()
            target.copy_(source)
            send_work.done = recv_work.done = True

    def isend(self, tensor, dst, group):
        work = Work()
        key = group, self.rank, dst
        self.send_queues[key].append((tensor, work))
        self.posts.append(("send", key))
        self.match(key)
        return work

    def irecv(self, tensor, src, group):
        work = Work()
        key = group, src, self.rank
        self.recv_queues[key].append((tensor, work))
        self.posts.append(("recv", key))
        self.match(key)
        return work


def make_transport(plan, rank):
    transport = SawtoothTransport.__new__(SawtoothTransport)
    transport.device = torch.device("cpu")
    transport.pp = SimpleNamespace(ranks=list(range(plan.pipeline_size)))
    transport.tp = None
    transport.rank = rank
    transport.size = plan.pipeline_size
    transport.plan = plan
    transport.capacity = 2
    transport.max_tokens = 7
    transport.ledger = RoundLedger(plan.plan_id, plan.virtual_size, 2)
    transport.groups = {"ordinary": "ordinary", "wrap": "wrap"}
    transport.send_streams = {name: Stream() for name in transport.groups}
    transport.recv_streams = {name: Stream() for name in transport.groups}
    transport.buffers = {}
    transport.recv_wire_buffers = {}
    transport.transfers = {}
    transport.channel_sequences = {}
    model = SimpleNamespace(get_sawtooth_carry_schema=lambda group_id, count: {"hidden": ((count, 1), torch.float32)})
    transport.initialize_buffers(model)
    return transport


def test_close_releases_full_and_wire_receive_pools(monkeypatch):
    events = []
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(current_stream=lambda device: SimpleNamespace(synchronize=lambda: events.append("sync"))),
        raising=False,
    )
    monkeypatch.setattr(torch.distributed, "destroy_process_group", lambda group: events.append(group))
    plan = StagePlan.build(SawtoothConfig(True, 2), 4, 2)
    transport = make_transport(plan, 1)
    transport.tp = SimpleNamespace(world_size=4, rank_in_group=0)
    transport.initialize_buffers(
        SimpleNamespace(get_sawtooth_carry_schema=lambda group_id, count: {"hidden": ((count, 4), torch.float32)})
    )
    assert transport.recv_wire_buffers and transport.reserved_bytes > 0
    transport.close()
    assert events == ["sync", "ordinary", "wrap"]
    assert not transport.groups and not transport.transfers
    assert not transport.buffers and not transport.recv_wire_buffers
    assert transport.reserved_bytes == 0


@pytest.mark.parametrize("pipeline_size", [2, 3, 4])
@pytest.mark.parametrize("virtual_size", [2, 3, 4])
def test_interleaved_wraps_match_fifo_and_preserve_batch_buffers(monkeypatch, pipeline_size, virtual_size):
    """A.r0/B.r0/A.r1 must progress, including arbitrary further wraps."""
    stream = Stream()
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(Event=Event, stream=lambda value: value, current_stream=lambda device: stream),
        raising=False,
    )
    fabric = Fabric()
    monkeypatch.setattr(torch.distributed, "isend", fabric.isend)
    monkeypatch.setattr(torch.distributed, "irecv", fabric.irecv)
    plan = StagePlan.build(SawtoothConfig(True, virtual_size), pipeline_size * virtual_size, pipeline_size)
    transports = [make_transport(plan, rank) for rank in range(pipeline_size)]
    capacities = [item.reserved_bytes for item in transports]
    results = {}
    sequence = 0
    for generation in range(3):
        for round_id in range(virtual_size):
            for slot in range(2):
                batch = generation * 2 + slot
                cmd = command(plan, batch, round_id, sequence, generation)
                sequence += 1
                count = 7 if slot == 0 else 3
                scheduler_output = SimpleNamespace(sawtooth_execution=cmd, total_num_scheduled_tokens=count)
                for rank, transport in enumerate(transports):
                    fabric.rank = rank

                    def execute(output, incoming, rank=rank, cmd=cmd, count=count, batch=batch):
                        group_id = cmd.round_id * pipeline_size + rank
                        if incoming is None:
                            carry = torch.full((count, 1), batch * 100, dtype=torch.float32)
                        else:
                            carry = incoming.tensors["hidden"]
                            assert torch.all(carry == batch * 100 + group_id)
                        carry = carry + 1
                        if cmd.is_last_round and rank == pipeline_size - 1:
                            results[batch] = carry.clone()
                            return None
                        return IntermediateTensors({"hidden": carry})

                    transport.execute_round(scheduler_output, SimpleNamespace(execute_sawtooth_round=execute))
                if round_id == 0:
                    # One wrap recv per executed command, never V-1 per batch.
                    posts = [p for p in fabric.posts if p[0] == "recv" and p[1][0] == "wrap"]
                    assert len(posts) == generation * 2 * (virtual_size - 1) + slot + 1
    assert all(not values for values in fabric.send_queues.values())
    assert all(not values for values in fabric.recv_queues.values())
    assert [item.reserved_bytes for item in transports] == capacities
    for batch, result in results.items():
        assert torch.all(result == batch * 100 + pipeline_size * virtual_size)
    for transport in transports:
        assert len(transport.transfers) <= 2 * (2 * virtual_size)


def test_sharding_uses_numel_divisibility_and_preserves_unsharded_tensor():
    transport = SawtoothTransport.__new__(SawtoothTransport)
    transport.tp = SimpleNamespace(world_size=4, rank_in_group=2)
    replicated = torch.arange(24).reshape(3, 8)
    assert torch.equal(transport._wire_tensor(replicated), torch.arange(12, 18))
    not_divisible = torch.arange(7)
    assert transport._wire_tensor(not_divisible) is not_divisible
    transport.tp = SimpleNamespace(world_size=1, rank_in_group=0)
    assert transport._wire_tensor(replicated) is replicated


@pytest.mark.parametrize("count", [1, 4, 7])
def test_preallocated_receive_gather_preserves_layout_and_has_no_hot_allocation(monkeypatch, count):
    stream = Stream()
    monkeypatch.setattr(
        torch, "npu", SimpleNamespace(stream=lambda value: value, current_stream=lambda device: stream), raising=False
    )
    plan = StagePlan.build(SawtoothConfig(True, 2), 4, 2)
    transport = make_transport(plan, 1)
    transport.tp = SimpleNamespace(world_size=4, rank_in_group=2, device_group="tp")
    model = SimpleNamespace(
        get_sawtooth_carry_schema=lambda edge, tokens: {
            "hidden": ((tokens, 8), torch.float32),
            "input_ids": ((tokens,), torch.int32),
        }
    )
    transport.initialize_buffers(model)
    reserved = transport.reserved_bytes
    targets = {
        torch.float32: torch.arange(count * 8, dtype=torch.float32).reshape(count, 8),
        torch.int32: torch.arange(count, dtype=torch.int32),
    }
    gathered = []
    gathers = []

    def receive(tensor, src, group):
        expected = targets[tensor.dtype]
        value = expected.reshape(4, -1)[2] if expected.numel() % 4 == 0 else expected
        tensor.copy_(value)
        work = Work()
        work.done = True
        return work

    def gather(output, input_, group, async_op):
        assert group == "tp" and async_op
        assert output.untyped_storage().data_ptr() != input_.untyped_storage().data_ptr()
        expected = targets[output.dtype]
        assert torch.equal(input_, expected.reshape(4, -1)[2])
        output.copy_(expected.reshape(-1))
        work = SimpleNamespace(done=False, wait=lambda: gathered.append(True), is_completed=lambda: work.done)
        gathers.append(work)
        return work

    monkeypatch.setattr(torch.distributed, "irecv", receive)
    monkeypatch.setattr(torch.distributed, "all_gather_into_tensor", gather)
    monkeypatch.setattr(torch, "empty", lambda *args, **kwargs: pytest.fail("Hot receive/gather allocated a tensor"))
    cmd = command(plan, 0, 0, 0)
    transfer = transport._post_receive(cmd, 0, count, "ordinary")
    incoming = transport._consume(transfer)
    assert torch.equal(incoming["hidden"], targets[torch.float32])
    assert torch.equal(incoming["input_ids"], targets[torch.int32])
    assert len(gathered) == (2 if count % 4 == 0 else 1)
    assert not transfer.communication_done(), "Gather input remains leased through asynchronous completion"
    with pytest.raises(RuntimeError, match="terminal communication fence"):
        transport._buffer(command(plan, 2, 0, 1, generation=1), 0, "recv", count)
    for work in gathers:
        work.done = True
    assert transfer.communication_done()
    transport._buffer(command(plan, 2, 0, 1, generation=1), 0, "recv", count)
    assert transport.reserved_bytes == reserved


@pytest.mark.parametrize("virtual_size", [2, 3, 4])
def test_future_wrap_is_posted_after_forward_and_ordinary_send(monkeypatch, virtual_size):
    """A forward/device sync must never depend on its own unsent output."""
    stream = Stream()
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(Event=Event, stream=lambda value: value, current_stream=lambda device: stream),
        raising=False,
    )
    fabric = Fabric()
    monkeypatch.setattr(torch.distributed, "isend", fabric.isend)
    monkeypatch.setattr(torch.distributed, "irecv", fabric.irecv)
    plan = StagePlan.build(SawtoothConfig(True, virtual_size), 2 * virtual_size, 2)
    transports = [make_transport(plan, rank) for rank in range(2)]
    for round_id in range(virtual_size):
        cmd = command(plan, 0, round_id, round_id)
        output = SimpleNamespace(sawtooth_execution=cmd, total_num_scheduled_tokens=3)
        for rank, transport in enumerate(transports):
            fabric.rank = rank
            start = len(fabric.posts)

            def execute(scheduled, incoming, rank=rank, start=start, cmd=cmd):
                # Models/allocators can synchronize the entire device. No
                # receive relying on this forward's future send may exist.
                assert not fabric.posts[start:]
                if cmd.is_last_round and rank == 1:
                    return None
                return IntermediateTensors({"hidden": torch.zeros(3, 1)})

            # Rank1 must enqueue its current incoming receive before compute;
            # only rank0 can own an outgoing future wrap.
            if rank == 0:
                runner = SimpleNamespace(execute_sawtooth_round=execute)
            else:
                runner = SimpleNamespace(
                    execute_sawtooth_round=lambda *_args, cmd=cmd: (
                        None if cmd.is_last_round else IntermediateTensors({"hidden": torch.zeros(3, 1)})
                    )
                )
            transport.execute_round(output, runner)
            if rank == 0:
                actions = [(kind, key[0]) for kind, key in fabric.posts[start:]]
                expected = [("send", "ordinary")]
                if not cmd.is_last_round:
                    expected.append(("recv", "wrap"))
                assert actions == expected


@pytest.mark.parametrize("pipeline_size", [2, 3, 4])
def test_startup_visits_matching_p2p_edges_before_early_receives(monkeypatch, pipeline_size):
    """A collective-only warmup does not create HCCL's per-peer communicator."""
    events = []
    rank = 0

    def post(kind, peer, group):
        events.append((rank, kind, peer, group))
        return SimpleNamespace(wait=lambda: events.append((rank, "dependency")))

    monkeypatch.setattr(torch.distributed, "isend", lambda tensor, dst, group: post("send", dst, group))
    monkeypatch.setattr(torch.distributed, "irecv", lambda tensor, src, group: post("recv", src, group))
    monkeypatch.setattr(torch.distributed, "barrier", lambda group: events.append((rank, "barrier", group)))
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(
            current_stream=lambda device: SimpleNamespace(synchronize=lambda: events.append((rank, "synchronize")))
        ),
        raising=False,
    )
    for rank in range(pipeline_size):
        transport = SawtoothTransport.__new__(SawtoothTransport)
        transport.device = torch.device("cpu")
        transport.rank = rank
        transport.size = pipeline_size
        transport.pp = SimpleNamespace(ranks=list(range(pipeline_size)), cpu_group="control")
        transport.groups = {"ordinary": "ordinary", "wrap": "wrap"}
        transport._initialize_p2p_channels()
        local = [event for event in events if event[0] == rank]
        assert sum(event[1] == "barrier" for event in local) == pipeline_size
        assert local[-1] == (rank, "synchronize")
        for index, event in enumerate(local):
            if event[1] in ("send", "recv"):
                assert local[index + 1] == (rank, "dependency")
                assert local[index + 2] == (rank, "barrier", "control")
    sends = {(event[0], event[2], event[3]) for event in events if event[1] == "send"}
    receives = {(event[2], event[0], event[3]) for event in events if event[1] == "recv"}
    assert (
        sends
        == receives
        == {
            *((rank, rank + 1, "ordinary") for rank in range(pipeline_size - 1)),
            (pipeline_size - 1, 0, "wrap"),
        }
    )
