# GLM-5.3 MXFP8 prefetch weight offload on A5

## Source and scope

The raw padded-NZ host storage and pointer-level prefetch transport are adapted
from [upstream PR #13950](https://github.com/vllm-project/vllm-ascend/pull/13950),
source head `7e357dd67af6aa7f33033c54d1ba12e69c197bef`, authored by KINGFIOX.
That PR is not merged at the time of this integration.

The local adaptation additionally covers the installed vLLM loader's
`empty_like(..., device="cpu").copy_` restoration. A5 rejects that operation
for internal-format weights. CPU storage instead retains the padded NZ bytes;
its logical tensor view is metadata for buffer allocation, not CPU arithmetic.
The prefetch buffer uses the captured format and physical storage size.

The loader adapter is installed lazily when the prefetch backend is created.
CPU-target loading and non-prefetch offloaders retain upstream behavior. NZ
restoration fails explicitly when its owner or transport op is unavailable.
Quantization may replace Parameter objects, so ownership is resolved after
processing. Old pinned storage is released incrementally. MXFP8 reload-buffer
entries referring to offloaded device weights are removed, preventing stale
references from keeping the weights resident; unselected entries are preserved.

## Validated experiment configuration

GLM-5.3-Flash-W8A8-MXFP8-0918, four Ascend950DT devices, CANN 9.2.0,
vLLM 0.30.0, TP2/PP2/DP1/EP2, MTP5 and eager execution:

```bash
--offload-backend prefetch \
--offload-group-size 2 \
--offload-num-in-group 1 \
--offload-prefetch-step 1 \
--offload-params mlp \
--enforce-eager \
--additional-config '{"enable_flashcomm1":false,"enable_dsa_cp":false,"multistream_overlap_shared_expert":false}'
```

Keep the existing model, network and Mooncake PP topology arguments. SP+PP is
still rejected by this branch. Shared-expert overlap's initialization check can
read CPU-resident weights before prefetch buffers are assigned, so it remains
disabled for this deployment. Attention-wide offload can reconstruct non-pinned
`kv_b_proj` storage; this integration validates MLP-only offload.

## Evidence and limits

The isolated full-model prototype completed initialization and returned HTTP
200 for three short, one-token completion requests. Repeated prompt output and
logprob matched. Loader-reported device weight allocation fell from roughly
74/85 GB to 39.3/46.7 GB for the two PP stages after retained MXFP8 buffers were
released. These are loader allocation figures, not total process peak memory.
The first request included kernel compilation; its latency is not a steady-state
benchmark. The subsequent short-request wall times were about 1.93 and 1.94 s.

CPU regressions cover replaced parameters, raw-view bounds, selective device
reference release and unavailable transport. A device regression compares three
raw offload/onload cycles against a non-offloaded NZ quantized GEMM, with exact
output equality.

This does not establish full-model accuracy, long-context throughput, 2P1D KV
transfer correctness, graph capture or runtime RL weight reload support. NZ
offload retains an eager-only guard in this branch. Two simultaneous high-ratio
offload initializations hit driver pinned-memory allocation failures despite
ample pageable RAM; memlock alone did not resolve that limit. Bring up one P at
a time and validate host/NPU capacity before starting the second.

Only adding `weight_nz_mode=0` did not make the unadapted four-device configuration
usable: the experiments also exposed shared-expert initialization, non-pinned
Attention weights and retained MXFP8 device storage. This patch addresses the
loader/transport/storage lifecycle, not those separate configuration constraints.

## Integration checks

The final integration passed nine targeted CPU tests, one real-NPU three-cycle
quantized-GEMM regression, Ruff lint/format and Python syntax/diff checks.
The full UT collection stopped because pytest-mock is missing; the repository
format.sh ci entry could not run because pre-commit is missing. These are not
reported as passing. A fresh final-code full-model launch was blocked before
weight loading by a different TP8 service occupying all eight devices; the
full-model smoke evidence above refers to the isolated prototype.
