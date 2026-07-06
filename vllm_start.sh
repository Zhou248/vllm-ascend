export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7
# export VLLM_ASCEND_ENABLE_FLASHCOMM1=1


export ASCEND_SLOG_PRINT_TO_STDOUT=0 # 1/0 是否打屏
export ASCEND_GLOBAL_LOG_LEVEL=2
export ASCEND_HOST_LOG_FILE_NUM=1000
# export ASCEND_LAUNCH_BLOCKING=1 # 强制同步日志

export ASCEND_PROCESS_LOG_PATH=/home/w00608002/plog

# source /usr/local/Ascend/driver/bin/setenv.bash
# source /usr/local/Ascend/cann-9.1.T560/set_env.sh
# source /usr/local/Ascend/cann/set_env.sh

# vllm serve /mnt/weight/Qwen3-235B-mxw4a4-pack-full-0423 \
# vllm serve /mnt/share/weight/Qwen3-235B-W4A4-622 \
vllm serve /mnt/share/weight/Qwen3-235B-W4A4C4-622 \
       --host "localhost" \
       --served-model-name Qwen3 \
       --trust-remote-code \
       --port 10037 \
       --gpu-memory-utilization 0.9 \
       --block-size 128 \
       --distributed-executor-backend mp \
       --no-enable-prefix-caching \
       --async-scheduling \
       --max-model-len 40960 \
       --max-num-batched-tokens 40960 \
       --max-num-seqs 400 \
       --additional-config '{"enable_cpu_binding":true,"ascend_compilation_config":{"fuse_qknorm_rope":false}}' \
       --quantization ascend \
       --tensor-parallel-size 4 \
       --enable-expert-parallel \
       --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
       # --enforce-eager \
       # --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
       
