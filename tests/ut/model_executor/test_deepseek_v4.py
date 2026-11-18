from types import SimpleNamespace

import pytest
import torch
from torch import nn
from vllm.distributed.sawtooth import SawtoothConfig, StagePlan

from vllm_ascend.models.deepseek_v4 import model as deepseek_v4


class StubModule(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()


def test_routed_moe_receives_configured_swiglu_limit(monkeypatch):
    fused_moe_kwargs = {}

    class StubFusedMoEFactory(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            fused_moe_kwargs.update(kwargs)

    monkeypatch.setattr(deepseek_v4, "FusedMoEFactory", StubFusedMoEFactory)
    monkeypatch.setattr(deepseek_v4, "ReplicatedLinear", StubModule)
    monkeypatch.setattr(deepseek_v4, "DeepseekV2MLP", StubModule)
    monkeypatch.setattr(deepseek_v4, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(deepseek_v4, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        deepseek_v4,
        "get_ep_group",
        lambda: SimpleNamespace(device_group=SimpleNamespace(size=lambda: 1), rank_in_group=0),
    )
    monkeypatch.setattr(deepseek_v4, "get_ascend_config", lambda: SimpleNamespace(mix_placement=False))
    monkeypatch.setattr(deepseek_v4.rocm_aiter_ops, "is_fused_moe_enabled", lambda: False)
    monkeypatch.setattr(deepseek_v4.rocm_aiter_ops, "is_fusion_moe_shared_experts_enabled", lambda: False)

    config = SimpleNamespace(
        hidden_act="silu",
        hidden_size=16,
        moe_intermediate_size=8,
        n_group=1,
        n_routed_experts=8,
        n_shared_experts=1,
        norm_topk_prob=True,
        num_experts_per_tok=2,
        num_hash_layers=0,
        routed_scaling_factor=2.5,
        scoring_func="sigmoid",
        swiglu_limit=10.0,
        topk_group=1,
    )
    parallel_config = SimpleNamespace(
        enable_eplb=False,
        eplb_config=SimpleNamespace(num_redundant_experts=0),
        use_sequence_parallel_moe=False,
    )

    deepseek_v4.DeepseekV4MoE(config, parallel_config, prefix="model.layers.1.mlp")

    assert fused_moe_kwargs["swiglu_limit"] == config.swiglu_limit


def test_sawtooth_preserves_hash_tokens_and_topk_across_non_c4_group():
    """A middle group without C4 must relay the previous indexer's carry."""
    plan = StagePlan.build(SawtoothConfig(True, 2), 4, 2)
    token_ids = torch.tensor([3, 7], dtype=torch.int32)
    calls = []
    buffers = {g: torch.zeros((2, 1), dtype=torch.int32) for g in range(4)}

    def embed(ids):
        calls.append("embed")
        return ids.float().unsqueeze(-1).repeat(1, 2)

    def head(hidden, *args):
        calls.append("head")
        return hidden.sum(dim=1)

    def norm(hidden):
        calls.append("norm")
        return hidden * 3

    class Layer:
        def __init__(self, layer_id):
            self.layer_id = layer_id

        def __call__(self, positions, hidden, residual, input_ids):
            calls.append(self.layer_id)
            torch.testing.assert_close(input_ids, token_ids)
            if self.layer_id == 0:
                buffers[0].copy_(input_ids.unsqueeze(-1))
            elif self.layer_id == 2:
                torch.testing.assert_close(buffers[2], token_ids.unsqueeze(-1))
            return hidden + self.layer_id + 1, hidden

    model = SimpleNamespace(
        config=SimpleNamespace(num_hidden_layers=4, num_hash_layers=4),
        sawtooth_topk_buffers=buffers,
        aux_hidden_state_layers=(),
        _mtp_hidden_buffer=None,
        use_sequence_parallel_moe=False,
        hc_mult=2,
        embed_input_ids=embed,
        layers=[Layer(i) for i in range(4)],
        hc_head=head,
        hc_head_fn=None,
        hc_head_scale=None,
        hc_head_base=None,
        norm=norm,
    )
    carry = None
    for stage in plan.stages:
        carry = deepseek_v4.DeepseekV4Model.forward_sawtooth_stage(
            model,
            stage,
            token_ids if stage.is_first else None,
            torch.arange(2),
            carry,
        )
    assert calls == ["embed", 0, 1, 2, 3, "head", "norm"]
    expected = (token_ids.float() + 10).unsqueeze(-1).repeat(1, 2) * 6
    torch.testing.assert_close(carry, expected)


@pytest.mark.parametrize("late_hash,vision", [(False, False), (True, False), (False, True)])
def test_sawtooth_carries_ids_only_while_a_future_layer_needs_them(late_hash, vision):
    """Removing IDs must preserve later hash/vision routing and numerical output."""
    plan = StagePlan.build(SawtoothConfig(True, 2), 4, 2)
    types = ["hash_moe", "moe", "moe", "hash_moe" if late_hash else "moe"]
    config = SimpleNamespace(hidden_size=2, num_hidden_layers=4, mlp_layer_types=types, vision_n_layers=int(vision))
    ids = torch.tensor([3, 7], dtype=torch.int32)
    seen = []

    class Layer:
        def __init__(self, index):
            self.index = index

        def __call__(self, positions, hidden, residual, input_ids):
            needs_ids = types[self.index] == "hash_moe" or vision
            if needs_ids:
                torch.testing.assert_close(input_ids, ids)
            seen.append(input_ids is not None)
            return hidden + self.index + 1, hidden

    model = SimpleNamespace(
        aux_hidden_state_layers=(),
        _mtp_hidden_buffer=None,
        config=config,
        sawtooth_plan=plan,
        sawtooth_dtype=torch.float32,
        sawtooth_index_cache=False,
        sawtooth_topk_buffers={},
        hc_mult=1,
        use_sequence_parallel_moe=False,
        embed_input_ids=lambda x: x.float().unsqueeze(-1).repeat(1, 2),
        layers=[Layer(i) for i in range(4)],
        hc_head=lambda x, *_: x.sum(dim=1),
        hc_head_fn=None,
        hc_head_scale=None,
        hc_head_base=None,
        norm=lambda x: x,
    )
    carry = None
    for stage in plan.stages:
        carry = deepseek_v4.DeepseekV4Model.forward_sawtooth_stage(
            model, stage, ids if stage.is_first else None, torch.arange(2), carry
        )
        if not stage.is_last:
            schema = deepseek_v4.DeepseekV4Model.get_sawtooth_carry_schema(model, stage.group_id, 2)
            assert set(carry.tensors) == set(schema)
            assert ("input_ids" in schema) == (late_hash or vision)
    assert seen == ([True] * 4 if late_hash or vision else [True, False, False, False])
    torch.testing.assert_close(carry, (ids.float() + 10).unsqueeze(-1).repeat(1, 2))


def test_legacy_three_hash_layers_fit_first_stage_without_any_id_carry():
    plan = StagePlan.build(SawtoothConfig(True, 2), 43, 2)
    model = SimpleNamespace(
        aux_hidden_state_layers=(),
        config=SimpleNamespace(hidden_size=4096, num_hidden_layers=43, num_hash_layers=3),
        sawtooth_plan=plan,
        sawtooth_dtype=torch.bfloat16,
        hc_mult=4,
        sawtooth_index_cache=False,
    )
    for stage in plan.stages[:-1]:
        schema = deepseek_v4.DeepseekV4Model.get_sawtooth_carry_schema(model, stage.group_id, 32)
        assert set(schema) == {"hidden_states"}


def test_sawtooth_dspark_carries_aux_states_across_wrap_and_stashes_residual():
    plan = StagePlan.build(SawtoothConfig(True, 2), 4, 2)
    tokens = torch.tensor([1, 2], dtype=torch.int32)

    class Layer:
        def __init__(self, index):
            self.index = index

        def __call__(self, positions, hidden, residual, input_ids):
            return hidden + self.index + 1, None

    model = SimpleNamespace(
        sawtooth_plan=plan,
        sawtooth_dtype=torch.float32,
        sawtooth_index_cache=False,
        sawtooth_topk_buffers={},
        config=SimpleNamespace(hidden_size=2, num_hidden_layers=4, num_hash_layers=0),
        use_sequence_parallel_moe=False,
        aux_hidden_state_layers=(1, 2, 3, 4),
        _mtp_hidden_buffer=torch.zeros(2, 4),
        hc_mult=2,
        embed_input_ids=lambda ids: ids.float().unsqueeze(-1).repeat(1, 2),
        layers=[Layer(i) for i in range(4)],
        hc_head=lambda hidden, *args: hidden.sum(dim=1),
        hc_head_fn=None,
        hc_head_scale=None,
        hc_head_base=None,
        norm=lambda hidden: hidden,
    )
    carry = None
    for stage in plan.stages:
        carry = deepseek_v4.DeepseekV4Model.forward_sawtooth_stage(
            model, stage, tokens if stage.is_first else None, torch.arange(2), carry
        )
        if not stage.is_last:
            schema = deepseek_v4.DeepseekV4Model.get_sawtooth_carry_schema(model, stage.group_id, 2)
            assert set(carry.tensors) == set(schema)
            for name, (shape, dtype) in schema.items():
                assert carry[name].shape == shape and carry[name].dtype == dtype
    hidden, aux = carry
    assert len(aux) == 4
    for layer, actual in enumerate(aux, 1):
        expected = (tokens.float() + layer * (layer + 1) / 2).unsqueeze(-1).repeat(1, 2)
        torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(model._mtp_hidden_buffer, (tokens.float() + 10).unsqueeze(-1).repeat(1, 4))
    torch.testing.assert_close(hidden, aux[-1] * 2)
