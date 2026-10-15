# GLM-5.3 parallel execution with prefetch offload

## Shared-expert multistream overlap

`multistream_overlap_shared_expert` previously validated the split shared MLP
immediately after quantization. The prefetch loading context can already have
restored those parameters to CPU, making the validation's NPU matmul fail with
CPU/NPU weight mismatch.

Resident shared experts still validate immediately. Offloaded experts defer the
same validation until their first forward, after the enclosing decoder layer's
prefetch wait and device-buffer assignment. Validation remains mandatory and
fails explicitly if weights are still on CPU. Its completion flag is set only
on success. Existing shared-stream output joins continue to protect buffer reuse.

The following combination was exercised on 141.61.33.17 with four Ascend950DT
cards, CANN 9.2, vLLM 0.30.0, GLM-5.3-Flash-W8A8-MXFP8-0918:

```bash
--tensor-parallel-size 2 --pipeline-parallel-size 2 --enable-expert-parallel \
--offload-backend prefetch --offload-group-size 4 --offload-num-in-group 1 \
--offload-prefetch-step 1 --offload-params mlp --enforce-eager \
--kv-cache-dtype fp8 --attention_config.indexer_kv_dtype fp8 \
--additional-config '{"multistream_overlap_shared_expert":true,"enable_fused_mc2":0,"enable_flashcomm1":false,"enable_dsa_cp":false,"c8_enable_reshape_optim":false}'
```

Keep the existing model, MTP5, network and Mooncake producer arguments. The raw
NZ prefetch integration in `glm53_prefetch_nz_offload.md` is a prerequisite.
FusedMC2 is explicitly disabled because enabling it overrides shared overlap.

The final code passed 148 CPU regressions covering the existing fused-MoE
schedules, deferred validation and NZ loader behavior. Full-model initialization
completed the integrated/split shared-expert consistency check. Requests with
17, 129, 513, 4097 and 8201 input tokens, a 16-token continuation, four concurrent
requests and SSE completion all succeeded. The sampled greedy token sequences
matched the overlap-disabled baseline; chosen-token log probabilities were not
identical (maximum observed absolute difference 1.0473). This smoke comparison
is not an accuracy evaluation or performance benchmark.

Experiment launchers, request results and logs are retained under
`/mnt/share/z00909726/glm53/experiments/feature_adaptation/`. The repository's
`format.sh ci` entry could not run because `pre-commit` is unavailable; targeted
Ruff lint/format, syntax and diff checks are run separately. This validation is
for eager MLP prefetch, not graph capture or weight reload.
