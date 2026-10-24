from types import SimpleNamespace

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
        sawtooth_topk_buffers=buffers,
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
