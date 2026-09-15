# GPU-memory reclamation break-even study

## Decision

**INT8 compression does not remain in the runtime memory-pressure action space.**
The request-local INT8 path was upgraded from the earlier materialize-to-FP16
reference to a format-specialized segmented Triton path, and conversion was
changed to a batched GPU data-path operation. It is numerically correct and
reclaims real logical KV bytes, but it is slower than both the dense FP16 path
and the lossless choices over the matched study. No tested horizon made INT8
the lowest-cost action. The subsequent runtime direction is limited to
queueing/not admitting and the existing lossless CPU/GPU KV swap mechanism.

This is a scoped implementation decision for the pinned SwiftLLM/RTX 3090
path and the tested symmetric per-group INT8 format. It is not a claim that
all quantized-attention kernels are impossible. The optimized path is retained
in the repository as auditable enabling evidence, but is not approved for a
scheduler or serving policy.

## Objective and boundaries

The experiment compares three responses to the same KV deficit:

1. **Queue/not admit:** retain FP16 and avoid allocating the incoming request's
   KV state.
2. **CPU offload:** move an existing FP16 request's KV state with the pinned
   `swiftllm_c.swap_blocks` implementation, then restore it.
3. **INT8 compression:** batch-demote selected live FP16 pages, decode them with
   a format-specialized Triton kernel, and batch-promote them when the episode
   ends.

The experiment does not implement a scheduler, controller, router, policy,
MorphServe weight swapping, INT4 runtime action, or a new quantizer. Queue and
compression costs are compared offline from matched measurements; no scheduler
choice is inferred from this artifact.

The cache representation remains the existing symmetric max-absolute,
per-group-128 quantizer with FP16 scales. A logical page has shape
`[block_size, num_kv_heads, head_dim]`, with `block_size=16`. The existing
`page_attention_for_layer` implementation remains an explicit correctness
oracle.

## Optimized INT8 enabling path

`vendor/swiftLLM/swiftllm/worker/kernels/segmented_paged_attn.py` launches
format-specialized phase-1 kernels for homogeneous FP16 and INT8 page segments,
then performs an online-softmax phase-2 reduction. INT8 values are read as
INT8 and dequantized in registers using the page's FP16 scales; mixed pages do
not first become a dense FP16 tensor. GQA head mapping and arbitrary physical
block tables are supported. `transformer_layer.py` routes FP16/INT8 page sets
through this path and retains the old reference for INT4 or unsupported
mixed-component pages.

`PagedKVCache.demote_pages_batch` stacks selected pages and runs reduction,
scaling, and quantization as one device operation. `promote_pages_batch` gives
an equivalent batched reversal measurement. Python still publishes per-page
metadata after the data-path operation; the timing is the CUDA event around
batched conversion, not a sum of hundreds of page calls.

The optimized smoke was run on RTX 3090 for both Llama shapes with batch 4,
256 tokens, 50% INT8 pages. Across the final 54-cell study, optimized output
was compared with the dequantized PyTorch oracle:

- maximum absolute error: `0.00048828125`;
- no optimized output contained NaN;
- all cells used format-specialized FP16/INT8 segments;
- the smoke and repository tests cover batched conversion, restoration,
  metadata, and CPU correctness.

The implementation is an enabling kernel, not the research contribution. Its
host segment construction and compact one-tensor-per-page storage are included
in the measured serving overhead rather than hidden.

## Matched measurement design

The final raw artifact is
`results/sensitivity/kv_break_even_study.json`. It contains 54 cells:

- model shapes: Llama 3.2 1B (`8x64`, 16 layers) and Llama 3.1 8B (`8x128`,
  32 layers);
- batch sizes: `1, 4, 8`;
- contexts: `128, 512, 2048` tokens;
- selected old-page fractions: `25%, 50%, 75%`;
- pressure horizons: `1, 4, 8, 16, 32, 64` decode iterations.

The attention datapath is measured for one layer to keep the kernel unit
isolated. Exact all-layer KV capacity is not estimated from a guessed dtype:
per-page bytes are calculated from the actual K/V payload and FP16 scale
format, then multiplied by the known layer count. CPU offload uses dense
all-layer tensors and the unchanged C++ swap API. Its reported physical HBM
delta is zero because SwiftLLM preallocates the dense GPU cache; the reclaimed
quantity is reusable GPU block capacity, not a false physical allocator free.

For each cell, all actions face the same target deficit. Compression uses the
selected old pages; CPU offload uses the smallest number of dense pages that
meets that target, exposing any sub-page granularity overshoot. The reported
action cost is deliberately decomposed as:

```text
queue cost       = horizon * dense FP16 per-token time
CPU-offload cost = swap-out + swap-in transition time
INT8 cost        = batch demotion + batch restoration
                   + horizon * (mixed INT8 per-token time - dense FP16 time)
```

The queue term is the user-visible delay avoided for a new request while the
pressure episode lasts. The offload term is the extra transition cost when an
existing request is moved out and restored; its reusable-capacity and physical
HBM observations are reported separately. Compression is credited with
remaining live, so only its incremental per-token attention cost is charged.
This explicit decomposition prevents a one-time transition from being
mistaken for a persistent decode penalty.

## Results

### Memory and transition example: batch 8, context 2048, 50% selected

| shape | all-layer capacity reclaimed/avoided | batch demote | batch restore | CPU swap out+in | optimized/dense attention time |
|---|---:|---:|---:|---:|---:|
| Llama 3.2 1B | 126 MiB | 3.16 ms | 1.82 ms | 50.71 ms | 5.74 / 0.22 ms per layer-token |
| Llama 3.1 8B | 504 MiB | 2.10 ms | 2.05 ms | 203.54 ms | 6.28 / 0.38 ms per layer-token |

The exact raw values, including allocator readings and repeated timings, are
in the JSON artifact. At this stress cell, the measured compression allocator
deltas were 8,257,536 B and 16,515,072 B, equal to the logical reclaimed
payload; CPU offload's physical HBM delta was 0 B because its dense cache is
preallocated. INT8 page sizes are 16,640 B for the 1B-shaped page and
33,280 B for the 8B-shaped page, versus 32,768 B and 65,536 B for FP16 K+V.
Thus INT8 reclaims 49.2% of a page including scales.

Across the study, all-layer INT8 reclaim ranged from 0.492--189 MiB for the
1B shape and 1.969--756 MiB for the 8B shape. CPU offload transitions ranged
from 0.74--76.39 ms and 1.35--305.39 ms respectively. The dense GPU cache is
preallocated, so the offload physical HBM delta was recorded rather than
silently reported as reclaimed memory.

### Persistent cost and break-even map

The optimized INT8 attention path remained slower than dense FP16 in every
cell. The fast/dense wall-time ratio ranged from 6.41--27.26 for the 1B shape
and 4.17--22.19 for the 8B shape. This includes the measured host segment
construction and packed-page view; the raw artifact also records GPU-event
measurements. The earlier PyTorch reference was not used as the optimized
path or as evidence against efficient kernels.

For every model shape, batch, context, and compression fraction, and all six
horizons:

- queue was the global minimum in 132/162 1B action rows and 152/162 8B rows;
- CPU offload was the global minimum in the remaining 30/162 1B and 10/162
  8B rows;
- INT8 compression was the global minimum in **0/324 rows**;
- INT8 was below CPU-offload transition cost in 34/162 1B rows and 86/162
  8B rows, but queueing was still lower in every one of those rows;
- INT8 was below queue cost in **0/324 rows**.

At the most favorable long-context stress cell (8B, batch 8, context 2048,
50% selected), INT8 approaches CPU offload only when the horizon is long:

| horizon | queue | CPU offload | INT8 compression | winner |
|---:|---:|---:|---:|---|
| 1 | 0.38 ms | 203.54 ms | 10.05 ms | queue |
| 8 | 3.03 ms | 203.54 ms | 51.35 ms | queue |
| 32 | 12.14 ms | 203.54 ms | 192.95 ms | queue |
| 64 | 24.27 ms | 203.54 ms | 381.75 ms | queue |

The analogous 1B stress cell is even less favorable: at horizon 64, queue,
offload, and INT8 costs are 13.81 ms, 50.71 ms, and 358.31 ms. These values
are action costs, not a scheduler throughput claim.

### Checkpoint quality

The paired forced-prefix probes use actual pinned Llama checkpoints. Queueing
and FP16 CPU offload are lossless (`0` quality change by construction). Old-page
INT8 demotion is near-lossless in the limited probe:

| checkpoint | old pages | paired NLL delta | top-1 agreement | KL mean |
|---|---:|---:|---:|---:|
| Llama 3.2 1B | 25/50/75% | -0.000036 / -0.000198 / -0.000423 | 100% / 100% / 100% | 1.09e-5 / 6.11e-6 / 1.84e-5 |
| Llama 3.1 8B | 25/50/75% | -0.001077 / -0.000980 / -0.002048 | 100% / 100% / 100% | 5.89e-6 / 7.74e-6 / 1.07e-5 |

These are one short forced-prefix probe per checkpoint, not a broad quality
certification. They show that quality is not the reason for the no-go; the
persistent optimized attention cost is.

The actual checkpoint batch-conversion probes also confirm small transition
work: at 50% old pages, batched demotion took 0.59 ms for 1B and 0.67 ms for
8B in the short probes, with no retained FP16 shadow. This satisfies the
conversion-amortization part of the gate in a small regime, but the steady
state and break-even requirements fail.

## Gate outcome

The predeclared gate for opening a later memory-pressure scheduler required:

1. acceptable steady-state INT8 decode overhead;
2. conversion amortized within a small number of decode iterations in at least
   one meaningful long-context regime; and
3. a compression break-even region not dominated by queueing or CPU offload.

Criterion 2 is supported by the batched conversion measurements. Criteria 1
and 3 fail: the optimized path is several times slower than dense FP16 in the
attention datapath and INT8 is never the global minimum in the matched map.
Therefore **no scheduler phase is opened** and INT8 is removed from the
runtime action space. Lossless queueing and existing FP16 CPU/GPU swapping are
the only retained responses for the next phase.

## Findings by certainty

### Confirmed findings

- SwiftLLM remains pinned at
  `682cf9a28f97f7490409981a2f181528f377eb5d` and all final runs used RTX
  3090 hardware.
- The optimized INT8 path reads native INT8 pages, dequantizes in the
  format-specialized Triton kernel, supports mixed FP16/INT8 segments and GQA,
  and matches the explicit oracle within the recorded error bound.
- Batched demotion and restoration use one GPU quantization/dequantization
  operation per transition, with exact payload/scale byte accounting and no
  steady-state FP16 shadow.
- The existing C++ `swap_blocks` path was measured with all model layers and
  reclaimed reusable GPU block capacity while leaving physical allocation
  unchanged because SwiftLLM preallocates it.
- Queue cost is zero-transition, zero-quality-loss and globally dominates the
  tested action-cost map; CPU offload is the next-best lossless action in a
  minority of long/high-deficit cells.
- INT8 compression was never the lowest-cost action in the final matched map.

### Supported but uncertain

- Old-page INT8 appears near-lossless for the two short paired checkpoint
  probes, but this does not establish task quality, long-horizon drift, or
  robustness to outliers.
- A more mature fused kernel or packed arena could reduce the measured
  overhead. The amount needed to create a non-dominated region is not known;
  no such kernel was available locally and kernel engineering is not the
  contribution.
- Queue cost is represented as an episode-duration model rather than a new
  online workload trace. The earlier pinned SwiftLLM KV-admission evidence
  establishes that queueing occurs under pressure, but it is not mixed into
  these microbenchmark timings.

### Blocked questions

- No scheduler or online action policy was implemented, by design.
- No large continuous-arrival trace was rerun with all three actions because
  that would turn this phase into controller work after the strict gate
  failed.
- No third-party local INT8 paged-attention kernel was available. FlashInfer
  was inspected in the adjacent environment and exposes direct FP8, not this
  symmetric integer INT8 page format; vLLM FlashAttention in this checkout is
  the dense FP16 prefill baseline.
- Large arena fragmentation and concurrent allocator pressure remain open for
  lossless actions, not reasons to retain INT8.

### Remaining uncertainty

The result is not evidence that every future native INT8 attention design is
impossible. It is evidence that this smallest credible reused-layout Triton
path does not earn a runtime action under the measured RTX 3090 tradeoff. A
future project would need to start with a new published/native kernel and a
fresh end-to-end gate; it should not reopen a scheduler on this artifact alone.

## Reproduction and verification

```bash
# Unit and syntax checks
PYTHONPATH="$PWD/vendor/swiftLLM:$PWD/vendor/swiftLLM/csrc" \
  .venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m py_compile \
  scripts/kv_precision_experiment.py scripts/kv_break_even_experiment.py \
  scripts/verify_artifacts.py \
  vendor/swiftLLM/swiftllm/worker/kv_cache.py \
  vendor/swiftLLM/swiftllm/worker/kernels/segmented_paged_attn.py \
  vendor/swiftLLM/swiftllm/worker/layers/transformer_layer.py
.venv/bin/python scripts/verify_artifacts.py
```

Final break-even command:

```bash
CUDA_VISIBLE_DEVICES=3 \
PYTHONPATH="$PWD/vendor/swiftLLM:$PWD/vendor/swiftLLM/csrc" \
  .venv/bin/python scripts/kv_break_even_experiment.py \
  --output results/sensitivity/kv_break_even_study.json \
  --device cuda:0 --model-family both --batches 1 4 8 \
  --contexts 128 512 2048 --attention-iterations 3 \
  --attention-repetitions 3 --seed 20261012
```

The optimized checkpoint probes are
`results/sensitivity/kv_precision_quality_batched_1b.json` and
`results/sensitivity/kv_precision_quality_batched_8b.json`; the optimized
kernel smoke is `results/sensitivity/kv_precision_optimized_smoke_final.json`.
Legacy fallback measurements remain in the original `kv_precision_*` artifacts
and are not used as the optimized-path result.
