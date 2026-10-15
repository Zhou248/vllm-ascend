# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pipeline-parallel contract of the GLM-5.3-Flash text model.

An mHC decoder layer defers its ``hc_post`` state (``post`` / ``comb``) to the
next layer. Across a PP boundary that state must travel with the intermediate
tensors: dropping it would make multi-stage execution differ from the
single-stage math. These tests pin the payload key set (which must match
``make_empty_intermediate_tensors``) and the hand-off between two stages.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from vllm.sequence import IntermediateTensors

from vllm_ascend.models.glm5next import model as glm5next_model
from vllm_ascend.models.glm5next.model import Glm5NextModel

HIDDEN_SIZE = 8
N_STREAMS = 4


def _stub_model(*, mhc: bool) -> Glm5NextModel:
    # The PP plumbing only reads ``config`` plus the active layer slice, so a
    # stub avoids building the NPU-bound attention / MoE stack.
    model = Glm5NextModel.__new__(Glm5NextModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        hidden_size=HIDDEN_SIZE,
        mhc=mhc,
        mhc_num_residual_streams=N_STREAMS,
    )
    model.is_sequence_parallel = False
    return model


def _pp_group(*, is_first_rank: bool, is_last_rank: bool):
    return lambda: SimpleNamespace(is_first_rank=is_first_rank, is_last_rank=is_last_rank, world_size=2)


class _DeferredStateLayer(nn.Module):
    """Emulates an mHC decoder layer.

    Records the deferred state it received and hands a fresh one to the next
    layer, mirroring ``Glm5NextDecoderLayer``'s 4-tuple contract.
    """

    def __init__(self) -> None:
        super().__init__()
        self.incoming: tuple[torch.Tensor | None, torch.Tensor | None] | None = None

    def forward(self, positions, hidden_states, residual, post, comb):
        del positions
        self.incoming = (post, comb)
        return (
            hidden_states + 1,
            residual,
            torch.full((hidden_states.shape[0], N_STREAMS, 1), 2.0),
            torch.full((hidden_states.shape[0], N_STREAMS, N_STREAMS), 3.0),
        )


@pytest.mark.parametrize("mhc", [True, False])
def test_make_empty_intermediate_tensors_matches_pp_payload(mhc):
    model = _stub_model(mhc=mhc)

    tensors = model.make_empty_intermediate_tensors(batch_size=5, dtype=torch.bfloat16, device=torch.device("cpu"))

    assert tensors["hidden_states"].shape == (5, HIDDEN_SIZE)
    assert tensors["hidden_states"].dtype == torch.bfloat16
    if mhc:
        # The per-token stream stays 2D; the hyper-connection streams live in
        # ``residual`` and the deferred state is FP32 (hc_pre kernel output).
        assert tensors["residual"].shape == (5, N_STREAMS, HIDDEN_SIZE)
        assert tensors["post"].shape == (5, N_STREAMS, 1)
        assert tensors["comb"].shape == (5, N_STREAMS, N_STREAMS)
        assert tensors["post"].dtype == torch.float32
        assert tensors["comb"].dtype == torch.float32
    else:
        assert set(tensors.tensors) == {"hidden_states", "residual"}
        assert tensors["residual"].shape == (5, HIDDEN_SIZE)


@pytest.mark.parametrize("mhc", [True, False])
def test_non_last_rank_ships_state_the_next_rank_loads(monkeypatch, mhc):
    model = _stub_model(mhc=mhc)
    model._active_layers = [_DeferredStateLayer()]
    monkeypatch.setattr(glm5next_model, "get_pp_group", _pp_group(is_first_rank=True, is_last_rank=False))
    inputs_embeds = torch.zeros(3, HIDDEN_SIZE, dtype=torch.bfloat16)

    out = model(input_ids=None, positions=torch.arange(3), intermediate_tensors=None, inputs_embeds=inputs_embeds)

    # Next rank indexes these keys directly, so the shipped set must match the
    # placeholder factory key-for-key.
    expected_keys = {"hidden_states", "residual"} | ({"post", "comb"} if mhc else set())
    assert set(out.tensors) == expected_keys
    torch.testing.assert_close(out["hidden_states"], inputs_embeds + 1)
    if mhc:
        assert out["post"].dtype == torch.float32
        assert out["comb"].dtype == torch.float32


def test_non_first_rank_continues_deferred_mhc_state(monkeypatch):
    model = _stub_model(mhc=True)
    layer = _DeferredStateLayer()
    model._active_layers = [layer]
    model.norm = nn.Identity()
    monkeypatch.setattr(glm5next_model, "get_pp_group", _pp_group(is_first_rank=False, is_last_rank=True))
    hidden_states = torch.zeros(3, HIDDEN_SIZE, dtype=torch.bfloat16)
    residual = torch.zeros(3, N_STREAMS, HIDDEN_SIZE, dtype=torch.bfloat16)
    post = torch.ones(3, N_STREAMS, 1)
    comb = torch.ones(3, N_STREAMS, N_STREAMS)
    payload = IntermediateTensors({"hidden_states": hidden_states, "residual": residual, "post": post, "comb": comb})

    out = model(input_ids=None, positions=torch.arange(3), intermediate_tensors=payload)

    # The receiving first layer must run hc_post_pre on the previous rank's
    # state instead of falling back to a standalone pre.
    assert layer.incoming is not None
    assert layer.incoming[0] is post
    assert layer.incoming[1] is comb
    torch.testing.assert_close(out, hidden_states + 1)


def test_non_first_rank_without_mhc_reads_plain_payload(monkeypatch):
    model = _stub_model(mhc=False)
    layer = _DeferredStateLayer()
    model._active_layers = [layer]
    model.norm = nn.Identity()
    monkeypatch.setattr(glm5next_model, "get_pp_group", _pp_group(is_first_rank=False, is_last_rank=True))
    hidden_states = torch.zeros(3, HIDDEN_SIZE, dtype=torch.bfloat16)
    payload = IntermediateTensors(
        {"hidden_states": hidden_states, "residual": torch.zeros(3, HIDDEN_SIZE, dtype=torch.bfloat16)}
    )

    out = model(input_ids=None, positions=torch.arange(3), intermediate_tensors=payload)

    assert layer.incoming == (None, None)
    torch.testing.assert_close(out, hidden_states + 1)


@pytest.mark.parametrize("mhc", [True, False])
@pytest.mark.parametrize("last_stage", [True, False])
def test_sp_receiving_stage_shards_all_states_and_restores_full_token_layout(monkeypatch, mhc, last_stage):
    model = _stub_model(mhc=mhc)
    model.is_sequence_parallel = True
    model.norm = nn.Identity()
    monkeypatch.setattr(glm5next_model, "get_pp_group", _pp_group(is_first_rank=False, is_last_rank=last_stage))
    sharded = []

    def shard(tensor):
        sharded.append(tensor)
        return tensor[:2]  # TP0 of three tokens, padded to four across TP2.

    monkeypatch.setattr(glm5next_model, "sp_shard", shard)
    hidden = torch.arange(3 * HIDDEN_SIZE, dtype=torch.bfloat16).reshape(3, HIDDEN_SIZE)
    residual = hidden[:, None, :].expand(-1, N_STREAMS, -1).clone() if mhc else hidden.clone()
    tensors = {"hidden_states": hidden, "residual": residual}
    if mhc:
        tensors.update(
            post=torch.arange(3 * N_STREAMS, dtype=torch.float32).reshape(3, N_STREAMS, 1),
            comb=torch.arange(3 * N_STREAMS * N_STREAMS, dtype=torch.float32).reshape(3, N_STREAMS, N_STREAMS),
        )
    expected_gathers = iter(tensors.values())

    def gather(tensor):
        original = next(expected_gathers)
        # TP1 carries the third token plus padding. Only padding may be dropped.
        remote = torch.cat((original[2:], torch.zeros_like(original[:1])), dim=0)
        return torch.cat((tensor, remote), dim=0)

    monkeypatch.setattr(glm5next_model, "sp_all_gather", gather)

    class CheckLocalState(nn.Module):
        def forward(self, positions, hidden_states, residual, post, comb):
            assert positions.shape[0] == 3
            assert hidden_states.shape[0] == residual.shape[0] == 2
            if mhc:
                assert post.shape[0] == comb.shape[0] == 2
                assert post.dtype == comb.dtype == torch.float32
            return hidden_states, residual, post, comb

    model._active_layers = [CheckLocalState()]
    output = model(input_ids=None, positions=torch.arange(3), intermediate_tensors=IntermediateTensors(tensors))
    assert len(sharded) == len(tensors)
    if last_stage:
        torch.testing.assert_close(output, hidden)
        return
    for name, original in tensors.items():
        assert output[name].shape == original.shape
        assert output[name].dtype == original.dtype
        torch.testing.assert_close(output[name], original)
