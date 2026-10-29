#!/usr/bin/env bash
# Experimental TP4/PP2 Sawtooth launch template; see ../../SAWTOOTH.md.
set -e
ASCEND_ENV_SH="${ASCEND_ENV_SH:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"
if [[ -f "$ASCEND_ENV_SH" ]]; then
    source "$ASCEND_ENV_SH"
fi
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VLLM_SOURCE_DIR="${VLLM_SOURCE_DIR:-$REPO_ROOT/../vllm}"
PYTHON_BIN="${PYTHON_BIN:-${VIRTUAL_ENV:+$VIRTUAL_ENV/bin/python}}"
: "${PYTHON_BIN:?Activate a virtual environment or set PYTHON_BIN}"
: "${MODEL_PATH:?Set MODEL_PATH to the local model directory}"
unset VLLM_SAWTOOTH_TRACE_DIR

export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export VLLM_USE_V2_MODEL_RUNNER=1
export TORCH_HCCL_BLOCKING_WAIT=0
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=1024
export TASK_QUEUE_ENABLE=1
export HCCL_OP_EXPANSION_MODE="AIV"
export VLLM_PREFIX_CACHE_RETENTION_INTERVAL=4096
export PYTHONPATH="$VLLM_SOURCE_DIR:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# This template uses DP1/PP2/TP4; speculative decoding is not covered.
exec "$PYTHON_BIN" -m vllm.entrypoints.cli.main serve "$MODEL_PATH" \
    --max-model-len 1048576 \
    --max-num-batched-tokens 10240 \
    --served-model-name dsv4 \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.9}" \
    --api-server-count 1 \
    --max-num-seqs 64 \
    --data-parallel-size 1 \
    --pipeline-parallel-size 2 \
    --tensor-parallel-size 4 \
    --no-enable-prefix-caching \
    --async-scheduling \
    --enable-expert-parallel \
    --tokenizer-mode deepseek_v4 \
    --tool-call-parser deepseek_v4 \
    --enable-auto-tool-choice \
    --reasoning-parser deepseek_v4 \
    --model-loader-extra-config='{"enable_multithread_load": true, "num_threads": 128}' \
    --quantization ascend \
    --host 127.0.0.1 \
    --port "${PORT:-14752}" \
    --block-size 32 \
    --attention_config.indexer_kv_dtype int8 \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
    --additional-config '{
        "ascend_compilation_config": {
            "enable_npugraph_ex": true,
            "enable_static_kernel": false
        },
        "enable_cpu_binding": true,
        "multistream_overlap_shared_expert": true,
        "sawtooth_pipeline": {
            "enabled": true,
            "virtual_size": 2,
            "max_live_batches": 3,
            "enable_early_admission": true, "enable_prefill_chunk_overlap": true
        }
    }'
