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

## DSA-CP with PP and FP8 indexer cache

DSA-CP enables the sequence-parallel MoE path. The old PP guard prevented this
combination because a stage could shard an already-sharded intermediate tensor.
PP boundaries now retain the runner's full-token contract: each stage shards
hidden states, residuals and deferred mHC post/comb state together, then gathers
all outgoing tensors and trims padding before the next PP stage. Post/comb stay
FP32. Positions and attention metadata continue to describe the full sequence.
This adds TP all-gathers at a non-final PP boundary; measure that communication
cost when assessing the benefit of CP.

The FP8 indexer cache constructor also contained a stale DSA-CP rejection. Its
removal exposes the existing CP path introduced by commit `0391134d7`: global
K/gates are gathered before pooling, queries remain local and FP8 keys retain
separate FP32 scales compatible with the CANN cache layout.

Real-device validation found an additional compiler problem that CPU shape
checks cannot detect: `tl.div_rn` with a scalar reduction result aborts the
Ascend Triton compiler with a TypeRange/null-types assertion. The KPool
compressor uses standard division for its scale and normalized key instead.
Standalone vector division/casts worked; reduction-based precise division
failed both with and without a dynamic branch. INT8/FP8 packed/planar cache
compression, scale values, scoring and untouched page padding passed all 19
existing single-NPU kernel cases after the change.

Enable CP alongside shared overlap with:

```json
{
  "multistream_overlap_shared_expert": true,
  "enable_fused_mc2": 0,
  "enable_flashcomm1": true,
  "enable_dsa_cp": true,
  "c8_enable_reshape_optim": false
}
```

The model, TP2/PP2/EP2, eager MLP prefetch group4/num1/step1, FP8 KV/indexer and
MTP5 settings stay as in the preceding example. Do not substitute PCP/DCP for
this GLM KPool implementation without a separate adaptation.

The final related CPU suite passed 417 tests, including last/intermediate PP
stages, odd token padding, FP32 mHC state, FP8 cache construction, DSA, shared
streams and raw NZ offload. DSA-only full-model tests completed requests from
1 through 32769 input tokens, both sides of the 8192-token chunk boundary,
32 concurrent requests, MTP execution and SSE without a worker failure.
All six one-token outputs in the seven-case comparison matched the baseline.

Multi-token output quality remains unvalidated. The CP path changes some
log probabilities and the 16-token continuation differed from the baseline.
A separate four-prompt, 32-token chat-template comparison found malformed
continuations in both the original overlap/CP-disabled baseline and CP mode.
These results do not establish generation accuracy; the validated deployment
purpose here is one-token, pure-Prefill performance testing through the P-only
proxy. A complete P/D generation/accuracy evaluation is still required before
using this configuration for answer generation. The complete UT collection is
also blocked by missing optional `modelscope`; no full-suite pass is claimed.

## Combined two-P acceptance

Both features were then enabled together on cards 0-3 (port 9081) and 4-7
(port 9082), with a separate TP2/PP2/EP2 and the same MLP prefetch configuration
per P. Start the two P services sequentially to avoid simultaneous pinned-memory
loading peaks. Each instance completed all 11 sequential input-length cases
(1, 2, 3, 127, 128, 129, 8191, 8192, 8193, 16385, 32769), then both handled
32 concurrent one-token requests at the same time: 86 requests in that final
acceptance run. Both services remained healthy afterward.

The P-only proxy on port 8082 then completed 32 concurrent requests and one SSE
request, forcing `max_tokens=1` even when callers requested 32. Its backend
request increments were 17/16, with zero failures and no outstanding requests.
The original 2P1D decode/KV-transfer path was not exercised in this acceptance.
The first-four-card, last-four-card and generic PP2/offload launchers under
`/mnt/share/z00909726/glm53/` were synchronized to the combined settings above;
original scripts are retained with the experiment evidence for comparison.
