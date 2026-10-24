# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Bounded, schema-driven transport for virtual pipeline rounds.

The control-plane command is the metadata message. No object receive is needed
to post a wrap receive. Activation process groups are deliberately created
without the GroupCoordinator cache: equal rank lists must not alias either
each other or the sampled-token communicator.
"""

from dataclasses import dataclass, field
from typing import Any

import torch
from vllm.distributed.parallel_state import get_pp_group, get_tp_group
from vllm.distributed.sawtooth import get_sawtooth_plan
from vllm.sequence import IntermediateTensors
from vllm.v1.sawtooth import SawtoothExecution, SawtoothRoundSubmitted

from vllm_ascend import envs
from vllm_ascend.distributed.utils import all_gather_async
from vllm_ascend.utils import create_hccl_pg_options, enable_sp


@dataclass
class RoundLedger:
    """Reject stale/duplicate commands before they can write KV or send data."""

    plan_id: str
    num_rounds: int
    capacity: int
    command_seq: int = -1
    slots: dict[int, SawtoothExecution] = field(default_factory=dict)

    def accept(self, command: SawtoothExecution) -> None:
        if command.plan_id != self.plan_id or command.num_rounds != self.num_rounds:
            raise ValueError("Sawtooth command does not match the worker stage plan")
        if not 0 <= command.slot_id < self.capacity:
            raise ValueError("Sawtooth slot exceeds the reserved transport capacity")
        if not 0 <= command.round_id < self.num_rounds:
            raise ValueError("Invalid sawtooth round")
        if command.command_seq <= self.command_seq:
            raise ValueError("Duplicate or out-of-order sawtooth command")
        previous = self.slots.get(command.slot_id)
        if command.is_first_round:
            if previous is not None and (
                not previous.is_last_round
                or command.slot_generation <= previous.slot_generation
                or command.batch_id == previous.batch_id
            ):
                raise ValueError("Sawtooth slot is live or has a stale generation")
        elif previous is None or (
            previous.batch_id != command.batch_id
            or previous.slot_generation != command.slot_generation
            or previous.round_id + 1 != command.round_id
        ):
            raise ValueError("Sawtooth continuation has no matching predecessor")
        self.command_seq = command.command_seq
        self.slots[command.slot_id] = command


@dataclass
class _Transfer:
    command: SawtoothExecution
    tensors: dict[str, torch.Tensor]
    wires: dict[str, torch.Tensor]
    handles: list[Any]
    # A receive is reusable after its consumer, not merely after HCCL writes it.
    consumer_done: Any = None

    def communication_done(self) -> bool:
        return all(handle.is_completed() for handle in self.handles)


def _stream_dependency(handle: Any) -> None:
    """Insert a dependency on the current device stream.

    WorkHCCL.wait inserts the completing event into the current NPU stream
    with the default nonblocking HCCL setting. Do not use the inherited generic
    Work.block_current_stream: its StreamBlock implementation is unavailable
    for PrivateUse1 in torch 2.10 / torch_npu 2.10, despite hasattr returning true.
    HCCL blocking-wait mode must be rejected by the startup configuration.
    """
    handle.wait()


def _validate_nonblocking_hccl() -> None:
    value = envs.TORCH_HCCL_BLOCKING_WAIT
    if value is None:
        value = envs.HCCL_BLOCKING_WAIT
    if value not in (None, "0"):
        raise ValueError("Sawtooth requires nonblocking HCCL Work.wait; set TORCH_HCCL_BLOCKING_WAIT=0")


class SawtoothTransport:
    def __init__(self, vllm_config: Any, device: torch.device):
        _validate_nonblocking_hccl()
        self.config = vllm_config
        self.device = device
        self.pp = get_pp_group()
        self.tp = None if enable_sp(vllm_config) else get_tp_group()
        self.rank = self.pp.rank_in_group
        self.size = self.pp.world_size
        self.plan = get_sawtooth_plan(vllm_config, vllm_config.model_config.get_total_num_hidden_layers(), self.rank)
        if self.plan is None:
            raise ValueError("SawtoothTransport requires an enabled stage plan")
        self.capacity = self.plan.config.max_live_batches or vllm_config.max_concurrent_batches
        self.max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self.ledger = RoundLedger(self.plan.plan_id, self.plan.virtual_size, self.capacity)
        self.groups: dict[str, Any] = {}
        # Every WORLD rank enters all new_group calls in this same order.
        for channel in ("ordinary", "wrap"):
            for ranks in self.pp.group_ranks:
                group = torch.distributed.new_group(
                    ranks,
                    backend="hccl",
                    pg_options=create_hccl_pg_options(f"sawtooth_{channel}"),
                )
                if self.pp.rank in ranks:
                    self.groups[channel] = group
        self.send_streams = {name: torch.npu.Stream(device=device) for name in self.groups}
        self.recv_streams = {name: torch.npu.Stream(device=device) for name in self.groups}
        self.buffers: dict[tuple[int, int, str], dict[str, torch.Tensor]] = {}
        self.recv_wire_buffers: dict[tuple[int, int, str], dict[str, torch.Tensor]] = {}
        self.transfers: dict[tuple[int, int, str], _Transfer] = {}
        self.channel_sequences: dict[tuple[str, str], int] = {}
        self.schema = None
        self._initialize_p2p_channels()

    def _initialize_p2p_channels(self) -> None:
        """Build the actual peer communicators before any early wrap receive.

        HCCL lazily creates P2P communicators independently of its collective
        communicator. An all_reduce warmup therefore cannot prevent rank zero's
        first wrap recv and rank one's first ordinary recv from waiting on each
        other's store rendezvous. Every PP rank visits each edge in the same
        order, with both endpoints posting a matching real transfer.
        """
        payload = torch.zeros(1, device=self.device)
        handles = []
        edges = [("ordinary", rank, rank + 1) for rank in range(self.size - 1)]
        edges.append(("wrap", self.size - 1, 0))
        for channel, source, destination in edges:
            if self.rank == source:
                handles.append(
                    torch.distributed.isend(payload, dst=self.pp.ranks[destination], group=self.groups[channel])
                )
            elif self.rank == destination:
                handles.append(torch.distributed.irecv(payload, src=self.pp.ranks[source], group=self.groups[channel]))
            if self.rank in (source, destination):
                _stream_dependency(handles[-1])
            # All endpoints have returned from first-use communicator creation
            # before any rank starts creating the next edge. This CPU barrier
            # is startup-only and never shares the activation communication PG.
            torch.distributed.barrier(group=self.pp.cpu_group)
        # Startup only: retain the payload and Works through actual completion.
        # Serving commands use stream events and never synchronize the host.
        torch.npu.current_stream(self.device).synchronize()

    def initialize_buffers(self, model: Any) -> None:
        """Reserve every continuation's buffers before KV memory profiling."""
        self.schema = model.get_sawtooth_carry_schema
        for slot in range(self.capacity):
            for round_id in range(self.plan.virtual_size):
                group_id = round_id * self.size + self.rank
                for direction, edge in (("recv", group_id - 1), ("send", group_id)):
                    if edge < 0 or edge >= self.plan.num_groups - 1:
                        continue
                    specs = self.schema(edge, self.max_tokens)
                    buffers = {}
                    for name, (shape, dtype) in specs.items():
                        if not shape or shape[0] != self.max_tokens:
                            raise ValueError(f"Carry {name} must have a static token-leading shape")
                        buffers[name] = torch.empty(shape, dtype=dtype, device=self.device)
                    self.buffers[slot, round_id, direction] = buffers
                    if direction == "recv" and self.tp is not None and self.tp.world_size > 1:
                        # Keep the receive shard disjoint from the all-gather
                        # output. HCCL must not rely on aliased input/output.
                        # Reserve before KV profiling, including the worst-case
                        # divisible prefix of token-count-dependent ID carries.
                        self.recv_wire_buffers[slot, round_id, direction] = {
                            name: torch.empty(
                                tensor.numel() // self.tp.world_size, dtype=tensor.dtype, device=self.device
                            )
                            for name, tensor in buffers.items()
                        }

    @property
    def reserved_bytes(self) -> int:
        return sum(
            t.numel() * t.element_size()
            for pool in (self.buffers, self.recv_wire_buffers)
            for values in pool.values()
            for t in values.values()
        )

    def _sequence(self, channel: str, direction: str, command: SawtoothExecution) -> None:
        key = channel, direction
        if command.command_seq <= self.channel_sequences.get(key, -1):
            raise ValueError("Sawtooth channel order differs from command order")
        self.channel_sequences[key] = command.command_seq

    def _buffer(
        self, command: SawtoothExecution, round_id: int, direction: str, num_tokens: int
    ) -> tuple[tuple[int, int, str], dict[str, torch.Tensor]]:
        key = command.slot_id, round_id, direction
        previous = self.transfers.get(key)
        if previous is not None:
            # Slot generations are reused only after the engine terminal fence.
            # Never wait for an unmatched old send at RPC entry.
            if not previous.communication_done():
                raise RuntimeError("Transport slot reused before its terminal communication fence")
            if previous.consumer_done is not None:
                torch.npu.current_stream(self.device).wait_event(previous.consumer_done)
        values = {name: tensor[:num_tokens] for name, tensor in self.buffers[key].items()}
        return key, values

    def _wire_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.tp is not None and self.tp.world_size > 1 and tensor.numel() % self.tp.world_size == 0:
            return tensor.reshape(self.tp.world_size, -1)[self.tp.rank_in_group]
        return tensor

    def _post_receive(self, command: SawtoothExecution, round_id: int, num_tokens: int, channel: str) -> _Transfer:
        self._sequence(channel, "recv", command)
        stream = self.recv_streams[channel]
        with torch.npu.stream(stream):
            key, tensors = self._buffer(command, round_id, "recv", num_tokens)
            wires = {}
            for name, tensor in tensors.items():
                if self.tp is not None and self.tp.world_size > 1 and tensor.numel() % self.tp.world_size == 0:
                    wires[name] = self.recv_wire_buffers[key][name][: tensor.numel() // self.tp.world_size]
                else:
                    wires[name] = tensor
            source = self.size - 1 if channel == "wrap" else self.rank - 1
            handles = [
                torch.distributed.irecv(tensor, src=self.pp.ranks[source], group=self.groups[channel])
                for tensor in wires.values()
                if tensor.numel()
            ]
        transfer = _Transfer(command, tensors, wires, handles)
        self.transfers[key] = transfer
        # Do not wait or all-gather here: an early wrap recv must not gate B.r0.
        return transfer

    def _consume(self, transfer: _Transfer) -> IntermediateTensors:
        for handle in transfer.handles:
            _stream_dependency(handle)
        for name, wire in transfer.wires.items():
            tensor = transfer.tensors[name]
            if self.tp is not None and wire is not tensor and tensor.numel():
                # Gather only when this round executes, on the compute stream,
                # keeping TP collectives in the model's command order.
                # The receive shard and full output have fixed leased storage.
                # Avoid allocating a full temporary and copying it each round.
                _, handle = all_gather_async(wire, self.tp, output=tensor.reshape(-1))
                transfer.handles.append(handle)
                _stream_dependency(handle)
        return IntermediateTensors(transfer.tensors)

    def _send(self, command: SawtoothExecution, output: IntermediateTensors, num_tokens: int) -> None:
        channel = "wrap" if self.rank == self.size - 1 else "ordinary"
        self._sequence(channel, "send", command)
        key, tensors = self._buffer(command, command.round_id, "send", num_tokens)
        if tensors.keys() != output.tensors.keys():
            raise ValueError(f"Stage output keys differ from its carry schema: {tuple(output.tensors)}")
        for name, target in tensors.items():
            source = output.tensors[name]
            if source.shape != target.shape or source.dtype != target.dtype:
                raise ValueError(f"Stage output {name} differs from its carry schema")
            target.copy_(source)
        ready = torch.npu.Event()
        ready.record(torch.npu.current_stream(self.device))
        stream = self.send_streams[channel]
        with torch.npu.stream(stream):
            stream.wait_event(ready)
            wires = {name: self._wire_tensor(tensor) for name, tensor in tensors.items()}
            destination = 0 if channel == "wrap" else self.rank + 1
            handles = [
                torch.distributed.isend(tensor, dst=self.pp.ranks[destination], group=self.groups[channel])
                for tensor in wires.values()
                if tensor.numel()
            ]
        self.transfers[key] = _Transfer(command, tensors, wires, handles, ready)

    def execute_round(self, scheduler_output: Any, runner: Any) -> SawtoothRoundSubmitted | None:
        command = scheduler_output.sawtooth_execution
        if not isinstance(command, SawtoothExecution):
            raise TypeError("Sawtooth execution requires a typed round command")
        num_tokens = scheduler_output.total_num_scheduled_tokens
        if not 0 <= num_tokens <= self.max_tokens:
            raise ValueError("Sawtooth token count exceeds its reserved buffers")
        self.ledger.accept(command)
        incoming = None
        if num_tokens:
            if self.rank == 0:
                if not command.is_first_round:
                    incoming = self.transfers[command.slot_id, command.round_id, "recv"]
                    if (
                        incoming.command.batch_id != command.batch_id
                        or incoming.command.slot_generation != command.slot_generation
                        or incoming.command.round_id + 1 != command.round_id
                    ):
                        raise ValueError("Wrap receive does not match the continuation")
            else:
                incoming = self._post_receive(command, command.round_id, num_tokens, "ordinary")
        intermediate = None if incoming is None else self._consume(incoming)
        output = runner.execute_sawtooth_round(scheduler_output, intermediate)
        if incoming is not None:
            incoming.consumer_done = torch.npu.Event()
            incoming.consumer_done.record(torch.npu.current_stream(self.device))
        is_model_exit = command.is_last_round and self.rank == self.size - 1
        if num_tokens and not is_model_exit:
            if not isinstance(output, IntermediateTensors):
                raise TypeError("Non-final sawtooth stage must return IntermediateTensors")
            self._send(command, output, num_tokens)
        elif output is not None:
            raise TypeError("Final sawtooth stage must retain its output for sampling")
        if num_tokens and self.rank == 0 and not command.is_last_round:
            # The local forward (including allocator/operator device syncs)
            # and ordinary send must be submitted before this future receive.
            # Otherwise a device sync waits for the wrap, whose producer is
            # still waiting for the ordinary send we have not submitted yet.
            # Only this command's outgoing edge is posted: posting all V-1
            # wraps at once would corrupt FIFO matching across batches.
            self._post_receive(command, command.round_id + 1, num_tokens, "wrap")
        return None if command.is_last_round else SawtoothRoundSubmitted(command)

    def close(self) -> None:
        # Shutdown is outside the execution hot path. All admitted batches must
        # be drained by EngineCore before workers are shut down.
        for transfer in self.transfers.values():
            for handle in transfer.handles:
                handle.wait()
        torch.npu.current_stream(self.device).synchronize()
        for group in self.groups.values():
            torch.distributed.destroy_process_group(group)
        self.groups.clear()
        self.transfers.clear()
        self.buffers.clear()
        self.recv_wire_buffers.clear()
