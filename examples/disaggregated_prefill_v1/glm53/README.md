# GLM5.3 two-prefill performance configuration

These scripts preserve the configuration validated on 141.61.33.17 in
`glm53_zy` on 2026-10-16. Model, network and profiler paths are specific to
that deployment. Run each script in a separate terminal inside the container.

```bash
bash prefill/run_p_first4.sh
bash prefill/run_p_last4.sh
```

Each service uses four NPUs, TP2/PP2/DP1/EP2, prefetch MLP weight offload,
SP, DSA-CP, shared-expert multistream overlap, FP8 KV/indexer cache and MTP5.
The validated settings are `--max-num-batched-tokens 16384` and
`--gpu-memory-utilization 0.9`, with maximum model length 528000.
Ports are 9081 and 9082. The scripts remain in the foreground.

The deployment copies are in `/mnt/share/z00909726/glm53/prefill/`.
After both services are healthy, start the existing prefill-only proxy:

```bash
bash /mnt/share/z00909726/glm53/start_2p_proxy.sh
cd /mnt/share/z00909726/aisbench_auto_tools_prefix
python3 aisbench_test.py --input_len 262144 --output_len 1 --data_num 128 \
  --concurrency 32 --request_rate 0 --npu_num 8 --dataset_type prefix_cache \
  --repeat_rate 0.9 --prefix_num 1 --prefix_test --dp 2
```

Two complete runs with the same saved dataset achieved 1.2235 and
1.2508 requests/s, with 128 successful requests in each run and 89.64%
measured prefix-cache hit rate. Each run restarted both services and the
proxy, then warmed the prefix before timing. Profiling was performed
separately through port 9082. Ordinary benchmark runs regenerate the prefix;
the experiment runner preserves the dataset for strict comparisons.

The CANN CP indexer optimization and the existing KDA fused-normalization
change were both present in this configuration. Performance validation is
not an end-to-end text accuracy acceptance test.

Full evidence, including slower runs and the excluded UB link failure, is at:

```text
/mnt/share/z00909726/glm53/experiments/perf_compare_20261016/
/mnt/share/z00909726/glm53/GLM53_prefill_performance_report_20261016.html
```
