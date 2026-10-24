# Sawtooth Pipeline Parallelism

Sawtooth Pipeline Parallelism is an experimental, opt-in execution mode for
interleaving multiple fixed layer groups on each Pipeline Parallel (PP) rank.
It can reduce pipeline bubbles by admitting bounded logical batches and moving
each batch through multiple virtual rounds.

The feature is disabled by default. It requires the companion vLLM Sawtooth
implementation at `e9570dc38743d0bc732b5855bd578331f7a30c7a`. Do not enable it with a vLLM build
that does not contain that exact scheduler, model, and worker contract.

!!! warning

    This feature is intended for development evaluation. Numerical accuracy
    and full masking coverage have not been accepted, and no performance claim
    is made by this guide. Development validation was stopped at the user's
    direction, so no additional tests were run for this documentation change.
    Do not treat the current implementation as production-qualified.

## How interleaving works

With PP size `2` and virtual size `2`, the model is divided into four fixed,
ordered layer groups. Group ownership alternates between the two PP ranks:

```text
round 0: group 0 on PP rank 0 -> group 1 on PP rank 1
round 1: group 2 on PP rank 0 -> group 3 on PP rank 1 -> sampling
```

The activation after group 1 wraps from PP rank 1 to PP rank 0. A logical batch
is admitted once, then carries an immutable request-step identity through both
rounds. `max_live_batches` bounds the number of batches that may own transport
slots concurrently.

By default, layer groups are balanced and contiguous in execution order. An
optional `layer_boundaries` list can set explicit group boundaries. For PP size
`P` and virtual size `V`, the list must contain `P * V + 1` strictly increasing
offsets, start at `0`, and end at the model's hidden-layer count.

## Enable the feature

The following example enables a PP2, virtual-size-2 deployment. Replace
`/path/to/model` with a supported model checkpoint and select a TP size that
matches the visible NPUs.

```bash
export VLLM_USE_V2_MODEL_RUNNER=1
export TORCH_HCCL_BLOCKING_WAIT=0

vllm serve /path/to/model \
    --pipeline-parallel-size 2 \
    --tensor-parallel-size 4 \
    --async-scheduling \
    --compilation-config '{"cudagraph_mode": "FULL"}' \
    --additional-config '{
        "sawtooth_pipeline": {
            "enabled": true,
            "virtual_size": 2,
            "max_live_batches": 2,
            "enable_early_admission": true
        }
    }'
```

The Sawtooth options are:

| Option | Default | Description |
| --- | --- | --- |
| `enabled` | `false` | Enables Sawtooth execution. |
| `virtual_size` | `1` | Number of virtual rounds. An enabled deployment requires at least `2`. |
| `max_live_batches` | Derived by vLLM | Fixed number of live logical-batch transport slots. The value must be positive. |
| `enable_early_admission` | `true` | Allows safe early admission when no older continuation would be reordered. |
| `layer_boundaries` | Balanced groups | Explicit boundaries for all `PP * virtual_size` groups. |

## HCCL requirement

Sawtooth depends on `WorkHCCL.wait()` adding a dependency to the current NPU
stream without blocking the host. Use:

```bash
export TORCH_HCCL_BLOCKING_WAIT=0
```

If `TORCH_HCCL_BLOCKING_WAIT` is unset, the implementation also reads the
legacy `HCCL_BLOCKING_WAIT` variable. Its only accepted explicit value is `0`.
Startup fails when either effective value enables blocking wait. The
`TORCH_HCCL_BLOCKING_WAIT` variable takes precedence when both are set.

## Runtime components

The Ascend implementation is split into five modules:

| Module | Responsibility |
| --- | --- |
| `sawtooth_transport.py` | Creates ordinary and wrap PP channels, reserves bounded slot buffers before KV profiling, enforces command and slot generations, and reconstructs TP-sharded carries into preallocated receive buffers. |
| `sawtooth_runtime.py` | Owns persistent per-stage model state, input/carry buffers, attention metadata, and graph managers, then activates exactly one immutable stage on the physical runner. |
| `sawtooth_graph.py` | Isolates ACL graph pools and graph parameters by plan, stage, and complete batch descriptor, and copies reusable graph outputs into slot-owned leases. |
| `sawtooth_execution.py` | Applies request admission once, validates immutable continuation inputs, executes each virtual round, and binds final sampling to the admitted request step. |
| `sawtooth_memory.py` | Accounts for persistent stage and transport resources before KV allocation and adds graph memory once after warmup. |

The companion vLLM implementation owns configuration parsing, the immutable
stage plan, scheduler admission, logical-batch retirement, all-rank RPC, and
the common model-stage interface. vLLM Ascend validates that contract before
allocating weights or initializing collectives.

## Supported boundary

The current configuration boundary is deliberately narrow and fails closed.

- Model Runner V2 is required.
- PP size must be greater than `1`, and virtual size must be at least `2`.
- The accepted architecture identifiers are `Qwen2ForCausalLM`,
  `Qwen3ForCausalLM`, `Qwen3MoeForCausalLM`, and
  `DeepseekV4ForCausalLM`. This list indicates adapter availability, not
  numerical qualification of every checkpoint or quantization.
- Tensor Parallel execution is supported by the transport design. Async
  scheduling and ACL graph execution are accepted by the configuration shown
  above.
- Data Parallelism, DBO, Prefill Context Parallelism, Decode Context
  Parallelism, speculative decoding, LoRA, KV connectors, encoder connectors,
  KV offload, weight offload, pooling runners, and non-vLLM model
  implementations are rejected during configuration.
- Structured-output requests are rejected before execution.
- Sawtooth commands carry a plan identity, command sequence, slot generation,
  and immutable request-step versions. Missing, stale, reordered, or
  incompatible state raises an error instead of falling back to ordinary PP.
- A nonterminal stage must return the declared tensor schema. Shape, dtype,
  device, or carry-key mismatches fail instead of allocating an unbounded
  replacement.

Keep the ordinary PP configuration as the reference deployment. Evaluate
Sawtooth only with frozen model, source, topology, request inputs, KV budget,
and graph settings, and report functional or performance observations
separately from numerical acceptance.
