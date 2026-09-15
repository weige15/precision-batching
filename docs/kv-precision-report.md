# Live KV-page precision feasibility report

## Decision

**No-go for using live page precision as a serving control in the current
SwiftLLM path.** The conversion mechanism is feasible: it changes already
resident pages, really releases payload storage, supports mixed formats, and
INT8 demotion of old pages is nearly lossless in the paired checkpoint tests.
The blocking result is mixed-format attention. The transparent page attention
implementation is about 2.1--3.9x slower for INT8 and 3.3--6.9x slower for
INT4 at 50% compressed pages across the measured grid (3.8--6.9x at the
batch-8/context-1024 stress point), and the CUDA overlap test did not hide
conversion work. A scheduler must not be opened on this evidence.

This is a scoped mechanism decision, not a claim that a fused INT8/INT4 KV
kernel is impossible. The current mixed path is intentionally a correctness
reference, not a production kernel. A future native mixed-format kernel would
need a fresh end-to-end benchmark before this decision can change.

The prior structured **weight** precision result remains closed historical
evidence. No Q/K/V/O/FFN weight allocation, weight swapping, scheduler policy,
router, or automatic precision choice was reopened here.

## What was implemented

`vendor/swiftLLM/swiftllm/worker/kv_cache.py` adds an explicit page store:

- page shape `[block_size, num_kv_heads, head_dim]`, one logical
  `(physical_block, layer)` record;
- independent K and V metadata at page granularity;
- FP16, INT8, and packed INT4 payloads;
- synchronous live demotion and event-ordered asynchronous demotion;
- exact tensor-byte accounting, pending-source accounting, and no-shadow checks;
- a layer-aware mixed-format decode-attention reference.

`--kv-page-format dense_fp16` is the unchanged upstream allocator. `fp16`,
`int8`, and `int4` select the research page store. Page-store mode explicitly
does not pretend to support SwiftLLM's existing CPU/GPU FP16 swap extension.

The quantizer is a standard symmetric per-group max-absolute rule with
FP16 scales and group size 128; it is not presented as a contribution. K-only
and V-only demotion are supported. Full format details are in
[`kv-page-format.md`](kv-page-format.md).

## Evidence and procedure

The raw artifacts are:

- `results/sensitivity/kv_precision_mechanism.json` and `.log`: 180 conversion
  trials, a 3x3 batch/context grid for both model shapes, homogeneous and mixed
  attention, exact allocator observations, and stream-overlap traces;
- `results/sensitivity/kv_precision_quality_1b.json` and `.log`: actual Llama
  3.2 1B checkpoint, 512-token context, 16 forced generation positions;
- `results/sensitivity/kv_precision_quality_8b.json` and `.log`: actual Llama
  3.1 8B checkpoint, 256-token context, 12 forced generation positions.

Quality uses paired forced prefixes: each compressed variant receives the
FP16 baseline's generated token at every step. An all-FP16 page-store control
is also run; compressed deltas are reported against that control as
`page_fp16_reference`, so dense Triton versus page-reference attention is not
confounded with quantization. It records per-step token NLL delta, top-1
agreement, KL, logit RMSE, and full token traces. It is a controlled proxy, not
a broad downstream benchmark.

## Confirmed mechanism findings

### Exact representation and memory

For one FP16 K+V page:

| model shape | FP16 page | INT8 page | INT4 page |
|---|---:|---:|---:|
| Llama 3.2 1B (`8x64`) | 32,768 B | 16,640 B | 8,448 B |
| Llama 3.1 8B (`8x128`) | 65,536 B | 33,280 B | 16,896 B |

Scales are included. Therefore demoting both components reclaims 49.2% for
INT8 and 74.2% for INT4, rather than claiming an ideal 50%/75% ratio.

At batch 8/context 1024, converting all 512 measured pages produced:

| shape | target | logical before → after | logical reclaimed | measured GPU allocation delta |
|---|---|---:|---:|---:|
| 1B | INT8 | 16,777,216 → 8,519,680 B | 8,257,536 B (7.875 MiB) | 7,864,320 B |
| 1B | INT4 | 16,777,216 → 4,325,376 B | 12,451,840 B (11.875 MiB) | 12,058,624 B |
| 8B | INT8 | 33,554,432 → 17,039,360 B | 16,515,072 B (15.75 MiB) | 16,252,928 B |
| 8B | INT4 | 33,554,432 → 8,650,752 B | 24,903,680 B (23.75 MiB) | 24,641,536 B |

The logical counts are recomputed from payload and scale tensors; the GPU
allocation columns are independent `torch.cuda.memory_allocated()` readings.
All measured conversion rows report no pending source/shadow after
synchronization. The small difference between logical and allocator deltas is
allocator granularity, not an uncounted FP16 copy.

### Conversion cost

Across the full batch/context grid, median per-page conversion costs at 50%
selected pages were:

| shape | INT8 | INT4 | median reclamation rate |
|---|---:|---:|---:|
| 1B | 0.320 ms/page | 0.483 ms/page | about 48.0--48.1 MiB/s |
| 8B | 0.318 ms/page | 0.487 ms/page | about 95.2--96.8 MiB/s |

At the 512-page stress point, the corresponding totals were 165.6 ms/241.6
ms for 1B and 166.0 ms/246.6 ms for 8B (INT8/INT4). The measured cost per
reclaimed MiB was about 20.3--21.0 ms/MiB for 1B and 10.4--10.5 ms/MiB for
8B. The raw traces show run-to-run GPU variability, so these are medians from
this final run rather than universal throughput claims.

The first-page peak allocation increase during conversion was 107,520 B for
the 1B page and 214,016 B for the 8B page. This includes quantization
workspace and target allocation; it is a transient conversion cost, not the
steady-state page size.

The actual model quality artifacts converted 256 pages for 50% old-page
INT8: the 1B trace reclaimed 3.938 MiB and the 8B trace reclaimed 7.875 MiB.
Per-page traces are preserved because host/GPU allocator interference makes a
single end-to-end total noisier than the repeated synthetic mechanism grid.

### Attention cost and overlap

The reference attention runs three iterations and three repetitions per point.
At batch 8/context 1024:

| shape | FP16 page reference | 50% INT8 mixed | 50% INT4 mixed | full INT8 static | full INT4 static |
|---|---:|---:|---:|---:|---:|
| 1B shape | 15.00 ms | 56.81 ms (3.8x) | 103.81 ms (6.9x) | 97.37 ms (6.5x) | 187.55 ms (12.5x) |
| 8B shape | 14.87 ms | 56.08 ms (3.8x) | 101.16 ms (6.8x) | 96.20 ms (6.5x) | 185.56 ms (12.5x) |

At batch 1/context 128, 50% mixed INT8 was about 2.0--2.9x and mixed INT4
about 3.2--4.5x in this final run; the overhead grew with both context and
batch. The raw grid contains every batch in `{1,4,8}`, context in
`{128,512,1024}`, format in `{INT8,INT4}`, fraction in `{0,.25,.5,.75,1}`.
The JSON reports *stored payload bytes processed per iteration* as a bandwidth
proxy. It is not actual GPU DRAM traffic: the reference first dequantizes
pages to FP16, so no hardware-bandwidth claim is made.

The overlap experiment scheduled conversion on a CUDA stream while running
eight FP16 2048x2048 matmuls on the default stream, then consumed the page
before any global synchronization. The consumer-side event wait produced zero
max error for both formats, observed a pending source before and after the
consumer read (until explicit synchronization), and left no shadow after
synchronization. However, the
measured overlap wall time exceeded conversion-plus-work serial estimates in
all four shape/format cases; the recorded hidden fraction was zero. Thus
ordering is **correct but overlap is not yet useful** as a control mechanism.

These attention timings are not evidence of a low-bit speedup. They measure
the deliberately simple Python/PyTorch mixed-format reference and establish
that this implementation has no acceptable fast path today.

## Quality findings

The quality runs are checkpoint-backed but small and single-prompt. Results
below report paired NLL delta versus the same FP16 logits, not an absolute
claim about language-model perplexity.

### Llama 3.2 1B, 512 context, 16 positions

| policy | reclaim | top-1 agreement | paired NLL delta | KL |
|---|---:|---:|---:|---:|
| static INT8 | 8.121 MiB | 100% | -0.00161 | 0.000180 |
| dynamic old INT8, 25/50/75% | 1.969/3.938/5.906 MiB | 100/100/100% | +0.00029/+0.00165/+0.00018 | 0.000021/0.000041/0.000050 |
| dynamic recent INT8, 25/50/75% | 1.969/3.938/5.906 MiB | 100/100/100% | -0.00303/-0.00230/-0.00217 | 0.000040/0.000055/0.000068 |
| dynamic old INT4, 50% | 5.938 MiB | 100% | +0.02884 | 0.010012 |
| dynamic recent INT4, 50% | 5.938 MiB | 93.8% | +0.03119 | 0.041327 |
| old INT8, early/late quarter layers | 0.984/0.984 MiB | 100/100% | +0.00054/+0.00030 | 0.000018/0.000009 |
| old INT8, K-only/V-only | 1.969/1.969 MiB | 100/100% | +0.00217/-0.00050 | 0.000031/0.000016 |
| static INT4 | 12.246 MiB | 87.5% | +0.04622 | 0.086078 |

### Llama 3.1 8B, 256 context, 12 positions

| policy | reclaim | top-1 agreement | paired NLL delta | KL |
|---|---:|---:|---:|---:|
| static INT8 | 16.734 MiB | 91.7% | +0.00055 | 0.000062 |
| dynamic old INT8, 25/50/75% | 3.938/7.875/11.812 MiB | 100/100/100% | +0.00249/+0.00299/+0.00147 | 0.000008/0.000008/0.000010 |
| dynamic recent INT8, 25/50/75% | 3.938/7.875/11.812 MiB | 100/100/91.7% | -0.00012/+0.00044/+0.00052 | 0.000010/0.000015/0.000016 |
| dynamic old INT4, 50% | 11.875 MiB | 100% | +0.03329 | 0.001760 |
| dynamic recent INT4, 50% | 11.875 MiB | 91.7% | +0.08289 | 0.009165 |
| old INT8, early/late quarter layers | 1.969/1.969 MiB | 100/100% | +0.00206/-0.00081 | 0.000005/0.000003 |
| old INT8, K-only/V-only | 3.938/3.938 MiB | 100/100% | +0.00226/+0.00192 | 0.000006/0.000005 |
| static INT4 | 25.234 MiB | 83.3% | +0.09802 | 0.035052 |

The all-FP16 page-store control differed from dense FP16 by only about
`+0.00001` (1B) and `-0.00004` (8B) paired NLL in these traces. Thus the
reported compressed-vs-page-FP16 deltas are not materially explained by the
attention implementation swap. INT8 old-page demotion is supported as a
near-lossless quality region in this small paired test. INT4 has clear quality
risk, especially static/recent pages. The result is not enough to certify a
policy: there is one prompt per checkpoint and no broad task suite.

### Small-iteration capacity criterion

At the actual checkpoint probe, 50% old-page INT8 demotion took 74.0 ms for
1B and 78.7 ms for 8B, reclaiming 3.938 MiB and 7.875 MiB respectively. The
corresponding dense FP16 decode medians were 8.9 ms and 21.1 ms, so a full
transition costs about 8.3 and 3.7 dense decode iterations. The current
page-reference decode after demotion was itself about 60.8 ms and 86.1 ms,
and the overlap test could not hide the transition. INT4 costs and quality
loss are worse. The mechanism can therefore admit capacity only when the
workload has enough long-lived page bytes to amortize a transition; it does
not meet a generally safe small-iteration control threshold in this path.

## Comparisons and interpretation

- **Unchanged FP16 KV:** the upstream dense cache and Triton attention remain
the default and are regression-tested against the prior no-op run. It has no
conversion or mixed-page overhead.
- **Static compressed KV:** uses the same page quantizer but compresses on
write. It establishes the quality/memory ceiling without live transitions;
its current reference attention is slower than FP16 because every page is
materialized and decoded in the fallback path.
- **Queue/refuse capacity:** retains FP16 and admits no additional request. It
reclaims 0 bytes and changes quality by 0, but avoids all conversion and mixed
attention cost. This is the correct conservative fallback while no efficient
mixed path exists.
- **MorphServe:** its published design uses runtime quantized layer swapping
and KVResizer capacity resizing; its KVResizer does not quantize existing KV
pages. This phase measures a different tradeoff—preserving a request while
changing its page representation. No equivalent MorphServe serving experiment
was run, so no superiority claim is made.
- **Weights versus KV state:** the previous structured weight experiment used
W8/W4/FP16 weight profiles and is closed. The current memory numbers are only
K/V state payloads and scales; no weight precision is included.

## Confirmed, supported, blocked, and uncertain

### Confirmed

- SwiftLLM remains pinned at `682cf9a28f97f7490409981a2f181528f377eb5d`.
- The default dense FP16 path preserves prior output behavior.
- FP16/INT8/INT4 page metadata and payloads coexist in one decode batch.
- Conversion arithmetic, format metadata, mixed attention, exact bytes, and
  asynchronous event ordering pass the repository tests and artifact verifier.
- Live demotion releases real allocated payload memory after synchronization;
  no dense FP16 shadow remains in the reported steady state.
- Conversion cost, reclaimed bytes, workspace, attention timing, batch/context
  variation, compressed fraction, recent/old pages, layer ranges, K-only, and
  V-only traces are preserved.

### Supported but uncertain

- Old-page INT8 is a plausible quality region for a later scheduler, based on
  two actual checkpoints and forced-prefix metrics. It is not a production
  quality guarantee.
- Conversion bandwidth scales favorably with page width: the 8B-shaped page
  reclaims about twice the MiB/s of the 1B-shaped page in this GPU run.
- A production fused kernel might reduce the measured mixed-attention penalty,
  but the amount is unknown and must not be inferred from this reference.

### Blocked questions

- No packed, format-aware Triton/CUDA attention kernel was implemented. The
  current reference is therefore insufficient evidence to approve a scheduler
  even though it is sufficient to reject the current fallback path.
- No arena allocator was built; one-tensor-per-page overhead and fragmentation
  under a large continuous workload are not characterized.
- Appending into an already compressed active page decodes and re-encodes the
  whole page; the append correctness path is tested, but its separate serving
  cost is not isolated.
- No CPU/GPU compressed swap format, priority policy, or automatic admission
  policy was tested by design.

### Remaining uncertainty

Longer contexts and real multi-request arrival traces could increase the value
of reclaimed bytes, while a fused kernel could reduce attention cost. Conversely,
attention-aware/outlier-preserving published methods such as KIVI, KVQuant, and
QAQ could improve INT4 quality but add metadata and conversion complexity. The
next experiment, if pursued, must measure those choices end to end rather than
reopen scheduler work on this result.

## Reproduction and audit

```bash
.venv/bin/python scripts/structured_precision_experiment.py --storage-self-test
PYTHONPATH="$PWD/vendor/swiftLLM:$PWD/vendor/swiftLLM/csrc" \
  .venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/verify_artifacts.py
```

The recorded mechanism command was:

```bash
PYTHONPATH="$PWD/vendor/swiftLLM:$PWD/vendor/swiftLLM/csrc" \
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/kv_precision_experiment.py \
  --output results/sensitivity/kv_precision_mechanism.json --device cuda:0 \
  --model-family both --batches 1 4 8 --contexts 128 512 1024 \
  --seed 20260915 --attention-iterations 3 --attention-repetitions 3
```

The quality commands used the same script with `--run-quality`, the local 1B
checkpoint on GPU 1 (`--quality-context-tokens 512 --quality-generation-tokens
16`), and the local 8B checkpoint on GPU 2 (`--quality-context-tokens 256
--quality-generation-tokens 12`). The full argv is stored in each JSON's
`invocation` field. The matching `.log` files preserve successful completion
markers and the JSON files retain raw conversion/page/attention/quality traces.
The requirement-by-requirement audit is [`kv-completion-audit.md`](kv-completion-audit.md).
