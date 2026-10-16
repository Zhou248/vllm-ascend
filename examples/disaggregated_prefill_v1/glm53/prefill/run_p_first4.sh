#!/usr/bin/env bash
# Run inside glm53_zy on 141.61.33.17: one four-card P, TP2/PP2/DP1/EP2.
# Foreground launch; keep this terminal open.
set -euo pipefail
ulimit -l unlimited
unset CPLUS_INCLUDE_PATH CPATH C_INCLUDE_PATH
set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
set -u

P_DEVICES="0,1,2,3"
P_PORT="9081"
P_KV_PORT="30100"
P_ENGINE_ID="101"

MODEL_PATH="/mnt/share/w00936111/GLM-5.3-Flash-W8A8-MXFP8-0918"
local_ip=172.27.8.17
nic_name="data0.3001"

export ASCEND_RT_VISIBLE_DEVICES="$P_DEVICES"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_BUFFSIZE=2048
export VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export ASCEND_LOCAL_COMM_RES='{"version":"1.3"}'
export HCCL_IF_IP="$local_ip"
export GLOO_SOCKET_IFNAME="$nic_name"
export TP_SOCKET_IFNAME="$nic_name"
export HCCL_SOCKET_IFNAME="$nic_name"
export HCCL_ALGO=level0:fullmesh

exec vllm serve "$MODEL_PATH" \
  --host 0.0.0.0 \
  --port "$P_PORT" \
  --data-parallel-size 1 \
  --tensor-parallel-size 2 \
  --pipeline-parallel-size 2 \
  --offload-backend prefetch \
  --offload-group-size 4 \
  --offload-num-in-group 1 \
  --offload-prefetch-step 1 \
  --offload-params mlp \
  --enable-expert-parallel \
  --seed 1024 \
  --served-model-name glm \
  --enable-prefix-caching \
  --safetensors-load-strategy prefetch \
  --kv-cache-dtype fp8 \
  --attention_config.indexer_kv_dtype fp8 \
  --max-num-seqs 32 \
  --max-model-len 528000 \
  --max-num-batched-tokens 16384 \
  --trust-remote-code \
  --enforce-eager \
  --quantization ascend \
  --tool-call-parser glm47 \
  --enable-auto-tool-choice \
  --reasoning-parser glm47 \
  --limit-mm-per-prompt '{"image":1,"video":0}' \
  --gpu-memory-utilization 0.9 \
  --speculative-config '{"num_speculative_tokens":5,"method":"deepseek_mtp","enforce_eager":true}' \
  --additional-config '{"multistream_overlap_shared_expert":true,"enable_fused_mc2":0,"enable_flashcomm1":true,"enable_dsa_cp":true,"c8_enable_reshape_optim":false}' \
  --kv-transfer-config "{\"kv_connector\":\"MooncakeConnectorV1\",\"kv_role\":\"kv_producer\",\"kv_port\":\"${P_KV_PORT}\",\"engine_id\":\"${P_ENGINE_ID}\",\"kv_connector_extra_config\":{\"prefill\":{\"dp_size\":1,\"tp_size\":2,\"pp_size\":2},\"decode\":{\"dp_size\":8,\"tp_size\":1},\"ascend_local_comm_res_path\":\"/etc/hixlep\"}}" \
  --profiler-config '{"profiler": "torch", "torch_profiler_dir": "/mnt/share/z00909726/glm53/profiling/2p1d", "torch_profiler_with_stack": false, "torch_profiler_record_shapes": false}'
