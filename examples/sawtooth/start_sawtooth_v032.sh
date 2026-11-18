#!/usr/bin/env bash
# Sawtooth v0.32: PP2 / PCP4 / TP1 / EP / MC2=2; run in an Ascend container.
# Requires paired vLLM/vLLM-Ascend sources with round-slot scheduling support.
set -euo pipefail

# ===== 使用前修改这里；ROOT 下应有 vllm 和 vllm-ascend 两个目录 =====
export ROOT=/mnt/share/d00893600/LPP_VLLM_32.0
export MODEL_PATH=/mnt/weight/DeepSeek-V4-Flash-DSpark-w4a8
export PYTHON_BIN="$ROOT/.venv/bin/python"
export CANN_ENV=/usr/local/Ascend/cann/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MBT=16384
export HOST=0.0.0.0
export PORT=8000

# ===== 启动配置：固定两批配对，按物理轮次占用并回收三个执行槽 =====
cd "$ROOT"
RUN_ROOT="$ROOT/runtime/sawtooth_v032"
SOURCE_ROOT="$ROOT"
ASCEND_SOURCE="$SOURCE_ROOT/vllm-ascend"
# shellcheck disable=SC1090
source "$CANN_ENV"
export TZ=Asia/Shanghai
unset VLLM_VERSION VLLM_SAWTOOTH_TRACE_DIR VLLM_SAWTOOTH_SCHEDULE_TRACE_DIR VLLM_SAWTOOTH_TRACE_MAX_EVENTS

export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_SERVER_DEV_MODE=1
export VLLM_USE_FASTOKENS=1
export TORCH_HCCL_BLOCKING_WAIT=0
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=1024
export TASK_QUEUE_ENABLE=1
export HCCL_OP_EXPANSION_MODE="AIV"
export VLLM_PREFIX_CACHE_RETENTION_INTERVAL=4096

# Explicit source roots prevent stale image editable finders.
export PYTHONPATH="$SOURCE_ROOT/vllm:$ASCEND_SOURCE:${ASCEND_HOME_PATH}/python/site-packages:${ASCEND_HOME_PATH}/opp/built-in/op_impl/ai_core/tbe"
unset FUSED_GATHER_AUDIT_DIR PACKED_PP_AUDIT_DIR STAGE_CARRY_AUDIT_DIR QLI_METADATA_AUDIT_DIR ROPE_GATHER_AUDIT_DIR VLLM_ASCEND_PCP_GATHER_PARTITIONS
export VLLM_ASCEND_FUSED_PCP_BLOCK_TABLES=1
export VLLM_ASCEND_PACKED_PP_SAMPLES=1
export VLLM_ASCEND_PACKED_STAGE_CARRY=0
export VLLM_ASCEND_DECODE_QLI_METADATA_REUSE=0
export VLLM_ASCEND_FUSED_ROPE_GATHER=1

mkdir -p "$RUN_ROOT"
export XDG_CACHE_HOME="$RUN_ROOT/cache_exec"
export VLLM_CACHE_ROOT="$RUN_ROOT/cache_exec/vllm"
export TORCH_EXTENSIONS_DIR="$RUN_ROOT/cache_exec/torch_extensions"
export TRITON_CACHE_DIR="$RUN_ROOT/cache_exec/triton_rope_model"

exec "$PYTHON_BIN" -P -m vllm.entrypoints.cli.main serve "$MODEL_PATH" \
    --max-model-len 1048576 \
    --max-num-batched-tokens "$MBT" \
    --served-model-name dsv4 \
    --gpu-memory-utilization 0.9 \
    --api-server-count 1 \
    --max-num-seqs 64 \
    --data-parallel-size 1 \
    --pipeline-parallel-size 2 \
    --tensor-parallel-size 1 \
    --prefill-context-parallel-size 4 \
    --enable-prefix-caching \
    --no-scheduler-reserve-full-isl \
    --async-scheduling \
    --enable-expert-parallel \
    --tokenizer-mode deepseek_v4 \
    --tool-call-parser deepseek_v4 \
    --enable-auto-tool-choice \
    --reasoning-parser deepseek_v4 \
    --model-loader-extra-config='{"enable_multithread_load": true, "num_threads": 128}' \
    --quantization ascend \
    --host "$HOST" \
    --port "$PORT" \
    --num-gpu-blocks-override 56000 \
    --block-size 32 \
    --attention_config.indexer_kv_dtype int8 \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
    --additional-config '{
        "ascend_compilation_config": {
            "enable_npugraph_ex": true,
            "enable_static_kernel": false
        },
        "enable_fused_mc2": 2,
        "enable_force_eplb": false,
        "enable_cpu_binding": true,
        "multistream_overlap_shared_expert": false,
        "sawtooth_pipeline": {
            "enabled": true,
            "virtual_size": 2,
            "max_live_batches": 3,
            "enable_sas_metadata_reuse": true,
            "enable_decode_sas_metadata_reuse": true,
            "enable_prefill_pair_balance": true,
            "enable_early_admission": true,
            "enable_prefill_chunk_overlap": true,
            "enable_round_slot_scheduling": true
        }
    }'
