# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise the real EngineArgs/platform boundary without loading weights."""

import json
from types import SimpleNamespace

import pytest
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed.sawtooth import get_sawtooth_plan, validate_sawtooth_capabilities
from vllm.engine.arg_utils import EngineArgs

from vllm_ascend.ascend_config import KVPPConfig, clear_ascend_config, init_ascend_config


@pytest.fixture
def engine_args(tmp_path, monkeypatch):
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    # This is architecture metadata only; there are no weights or tokenizer.
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3MoeForCausalLM"],
                "model_type": "qwen3_moe",
                "vocab_size": 256,
                "hidden_size": 128,
                "intermediate_size": 256,
                "moe_intermediate_size": 128,
                "num_hidden_layers": 8,
                "num_attention_heads": 8,
                "num_key_value_heads": 4,
                "head_dim": 16,
                "num_experts": 8,
                "num_experts_per_tok": 2,
                "max_position_embeddings": 16384,
                "hidden_act": "silu",
                "rms_norm_eps": 1e-6,
                "tie_word_embeddings": False,
            }
        ),
        encoding="utf-8",
    )

    def build(**kwargs):
        options = dict(
            model=str(tmp_path),
            skip_tokenizer_init=True,
            tensor_parallel_size=4,
            pipeline_parallel_size=2,
            enable_expert_parallel=True,
            distributed_executor_backend="mp",
            max_num_seqs=64,
            max_model_len=16384,
            max_num_batched_tokens=10240,
            enable_chunked_prefill=True,
            enable_prefix_caching=True,
            async_scheduling=True,
            compilation_config={"cudagraph_mode": "FULL_DECODE_ONLY"},
            additional_config={
                "sawtooth_pipeline": {"enabled": True, "virtual_size": 2, "max_live_batches": 4},
                "sawtooth_stage_identity": "sawtooth_config_test_g0",
                "enable_cpu_binding": True,
                "multistream_overlap_shared_expert": True,
                "ascend_compilation_config": {"enable_npugraph_ex": True},
            },
        )
        options.update(kwargs)
        return EngineArgs(**options)

    clear_ascend_config()
    yield build
    clear_ascend_config()


@pytest.mark.parametrize("async_scheduling", [False, True])
def test_engine_config_preserves_pipeline_keys_and_target_features(engine_args, async_scheduling):
    args = engine_args(async_scheduling=async_scheduling)
    config = args.create_engine_config()
    # Exercise JSON serialization used by launchers and per-stage identities.
    saved = json.loads(json.dumps(config.additional_config))
    restored = engine_args(additional_config=saved, async_scheduling=async_scheduling).create_engine_config()
    for result in (config, restored):
        plan = get_sawtooth_plan(result, 8)
        assert plan is not None and plan.num_groups == 4
        assert result.use_v2_model_runner
        assert result.parallel_config.tensor_parallel_size == 4
        assert result.parallel_config.enable_expert_parallel
        assert result.scheduler_config.async_scheduling == async_scheduling
        assert result.scheduler_config.enable_chunked_prefill
        assert result.scheduler_config.max_num_batched_tokens == 10240
        assert result.cache_config.enable_prefix_caching
        assert result.compilation_config.cudagraph_mode == CUDAGraphMode.FULL_DECODE_ONLY
        ascend = init_ascend_config(result)
        assert ascend.enable_cpu_binding
        assert ascend.multistream_overlap_shared_expert
        assert ascend.ascend_compilation_config.enable_npugraph_ex
        assert not hasattr(ascend, "sawtooth_pipeline")
        assert not hasattr(ascend, "sawtooth_stage_identity")
        assert result.additional_config["sawtooth_stage_identity"] == "sawtooth_config_test_g0"
    assert get_sawtooth_plan(config, 8).plan_id == get_sawtooth_plan(restored, 8).plan_id


@pytest.mark.parametrize(
    "bad_option, expected",
    [
        ({"pipeline_parallel_size": 1}, "Sawtooth requires PP"),
        ({"additional_config": {"sawtooth_pipeline": {"enabled": True, "virtual_size": 1}}}, "virtual_size"),
        (
            {
                "additional_config": {
                    "sawtooth_pipeline": {
                        "enabled": True,
                        "virtual_size": 2,
                        "layer_boundaries": [0, 3, 2, 6, 8],
                    }
                }
            },
            "layer_boundaries",
        ),
    ],
)
def test_engine_config_rejects_invalid_plan_before_worker_start(engine_args, bad_option, expected):
    with pytest.raises(ValueError, match=expected):
        engine_args(**bad_option).create_engine_config()


def test_engine_config_rejects_v1_explicitly(engine_args, monkeypatch):
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "0")
    with pytest.raises(ValueError, match="Sawtooth does not support Model Runner V1"):
        engine_args().create_engine_config()


def _capability_config():
    return SimpleNamespace(
        additional_config={"sawtooth_pipeline": {"enabled": True, "virtual_size": 2}},
        use_v2_model_runner=True,
        parallel_config=SimpleNamespace(pipeline_parallel_size=2),
        model_config=SimpleNamespace(hf_config=SimpleNamespace(architectures=["DeepseekV4ForCausalLM"])),
    )


@pytest.mark.parametrize(
    "scope, field, value, expected",
    [
        ("parallel_config", "data_parallel_size", 2, "data parallelism"),
        ("parallel_config", "enable_dbo", True, "DBO"),
        ("parallel_config", "prefill_context_parallel_size", 2, "PCP"),
        ("parallel_config", "decode_context_parallel_size", 2, "DCP"),
        (None, "speculative_config", object(), "speculative decoding"),
        (None, "lora_config", object(), "LoRA"),
        (None, "kv_transfer_config", object(), "KV connectors"),
        (None, "ec_transfer_config", object(), "encoder connectors"),
        ("model_config", "runner_type", "pooling", "pooling"),
        ("model_config", "model_impl", "transformers", "non-vLLM"),
        (None, "cache_config", SimpleNamespace(kv_offloading_size=1), "KV offloading"),
        (None, "offload_config", SimpleNamespace(uva=SimpleNamespace(cpu_offload_gb=1)), "weight offloading"),
    ],
)
def test_unsupported_capabilities_are_feature_gated(scope, field, value, expected):
    config = _capability_config()
    setattr(getattr(config, scope) if scope else config, field, value)
    with pytest.raises(ValueError, match=expected):
        validate_sawtooth_capabilities(config)
    config.additional_config["sawtooth_pipeline"]["enabled"] = False
    validate_sawtooth_capabilities(config)


def test_unknown_architecture_does_not_silently_run_physical_pp():
    config = _capability_config()
    config.model_config.hf_config.architectures = ["LlamaForCausalLM"]
    with pytest.raises(ValueError, match="model adapter"):
        validate_sawtooth_capabilities(config)


def test_default_structured_output_config_and_deepseek_sp_are_not_rejected():
    config = _capability_config()
    config.structured_outputs_config = object()
    config.parallel_config.use_sequence_parallel_moe = True
    validate_sawtooth_capabilities(config)


@pytest.mark.parametrize(
    "config, expected",
    [
        (SimpleNamespace(kvpp_config=SimpleNamespace(size=4)), "KVPP"),
        (SimpleNamespace(sparse_kv_offload_config=SimpleNamespace(enabled=True)), "sparse KV offloading"),
        (SimpleNamespace(enable_dsa_cp=True), "DSA context parallelism"),
    ],
)
def test_platform_cache_ownership_extensions_require_support(config, expected):
    with pytest.raises(ValueError, match=expected):
        validate_sawtooth_capabilities(_capability_config(), config)


@pytest.mark.parametrize(
    "enabled, tp_size, expected_size",
    [
        (True, 4, 4),
        ("true", 4, 4),
        (False, 4, 1),
        ("false", 4, 1),
        (True, 1, 1),
    ],
)
def test_kvpp_guard_uses_the_same_derived_size_as_platform(enabled, tp_size, expected_size):
    config = _capability_config()
    config.additional_config["enable_kvpp"] = enabled
    config.parallel_config.tensor_parallel_size = tp_size
    config.parallel_config.prefill_context_parallel_size = 1
    kvpp = KVPPConfig.from_vllm_config(config)
    assert kvpp.size == expected_size
    platform = SimpleNamespace(kvpp_config=kvpp)
    if expected_size > 1:
        with pytest.raises(ValueError, match="KVPP"):
            validate_sawtooth_capabilities(config, platform)
    else:
        validate_sawtooth_capabilities(config, platform)
