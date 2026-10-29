# Experimental Sawtooth prefill snapshot

This branch shares the paired vLLM / vLLM-Ascend code used in the
2026-10-29 Ascend experiments. It includes the original opt-in virtual
pipeline implementation, tail splitting, prefill overlap, request-ledger
handling, and associated control-plane tests.

This is a reproducible source snapshot, not a release or a claim of a
proven performance win over native pipeline parallelism. AI assistance
was used to develop and package the changes.

## Get both repositories

```bash
git clone -b codex/sawtooth-prefill https://github.com/HangqiDuan/vllm.git
git clone --recurse-submodules -b codex/sawtooth-prefill \
    https://github.com/HangqiDuan/vllm-ascend.git
```

Use this branch from both repositories. Do not mix either side with an
arbitrary current main branch. Record both published HEAD SHAs with
`git rev-parse HEAD` when reproducing an experiment.

## Original source provenance

| Repository | Official base | Original local feature commit |
| --- | --- | --- |
| vLLM | `ced6857afa0ea7b2e3f0846a62e1394e90f15607` | `e9570dc38743d0bc732b5855bd578331f7a30c7a` |
| vLLM-Ascend | `af2b751aaf677255a6597b56261ee03f1d28c12f` | `c89841adfcbc2a1f9aa0416b67f231bdae851585` |

The original feature IDs above identify the initial implementation, before
the subsequent tested working-tree changes. Publication preserves the original
commits and adds those tested changes in the published branch HEAD. A source manifest records file hashes and
the original dirty-patch hashes. Use the published commits for checkout.

## Tested scope

The tested Ascend configuration uses eight NPUs: TP4, physical PP2,
DP1, virtual_size=2, max_live_batches=3, early admission and prefill
overlap enabled. DP>1 and speculative decoding are not covered here.
The feature is opt-in through `additional_config.sawtooth_pipeline`.

Recent hardware measurements used two requests, input setting 65536
(65619 actual input tokens per request), output length 1, concurrency 2,
request rate 0, max_num_batched_tokens=10240, prefix caching disabled,
and FULL_DECODE_ONLY graphs. They exercise this short-output workload;
they do not constitute a full accuracy evaluation or long-decode test.

Cross-startup performance comparisons are sensitive to automatic KV
block counts. In the measured native PP setup, fixing 83192 versus
83193 blocks changed Attention performance substantially. The Sawtooth
configuration cannot run those capacities on the tested runtime: both
encountered driver OOM during the first warmup. Do not force those values
from the native baseline into Sawtooth.

On the observed Sawtooth runs, automatic KV with memory utilization 0.9
averaged 14865.2585 input tokens/s, versus 14102.4111 at 0.92, each after
three warmups and over three formal runs. Actual KV counts were 71654
and 75101. These are observations from separate startups, not an isolated
proof of a memory-utilization or odd/even-block mechanism.

## Installation and launch

See the companion vLLM-Ascend branch's `SAWTOOTH.md` for the environment
manifest and portable launch template. Model weights, CANN, drivers,
compiled operators, credentials, and profiler traces are not included.

## Ascend environment

`examples/sawtooth/validated_environment.json` records installed package
versions and source provenance. Package metadata can predate a source
override, so use the paired source revisions rather than relying on the
installed vLLM version string alone.

Start with a supported CANN/driver/PyTorch/torch-npu environment and an
activated Python virtual environment. Following the repository's source
installation flow, an editable installation can be prepared with:

```bash
VLLM_TARGET_DEVICE=empty uv pip install --no-build-isolation -e ./vllm
uv pip install --no-build-isolation -e ./vllm-ascend
```

Use the main repository installation guide for the required build tools
and Ascend index settings. A fresh-environment build of this publication
has not been validated; the hardware experiments used an existing
compatible Ascend runtime with source overrides.

The tested DeepSeek-V4 environment also had external Ascend custom
operators and site-package compatibility patches, including support for
`indexer_kv_dtype=int8`. These external components are not bundled in
this branch. The launch template assumes they are already available;
cloning these two repositories alone does not install them.

```bash
source /path/to/venv/bin/activate
export MODEL_PATH=/path/to/DeepSeek-V4-Flash-DSpark-w4a8
export PYTHON_BIN="$VIRTUAL_ENV/bin/python"
bash vllm-ascend/examples/sawtooth/server_dsv4.sh
```

The template defaults to memory utilization 0.9 and automatic KV sizing.
Override MODEL_PATH, PYTHON_BIN, VLLM_SOURCE_DIR, ASCEND_RT_VISIBLE_DEVICES,
PORT, or GPU_MEMORY_UTILIZATION as needed. Keep both source checkouts
paired and record the actual runtime `num_gpu_blocks` when benchmarking.

## Publication checks

All changed Python files passed syntax compilation. Selected CPU tests passed:
119 vLLM ledger/engine/distributed tests and 26 Ascend memory/communication
tests. Package initialization was bypassed in the CPU harness; these checks
do not cover NPU execution or a fresh build. The launcher passed `bash -n`
and the staged patch passed `git diff --check`. Full pre-commit lint was
not run because pre-commit is not installed in the publication environment.
