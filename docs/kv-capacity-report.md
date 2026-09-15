# Mixed-precision KV capacity and decode feasibility report

**Decision: GO, narrowly scoped to the tested KIVI K4V4 path.**

On the RTX 3090, native KIVI K4V4 (`group_size=32`, FP16 residual window
`residual_length=32`) produced a replicated, fixed-budget end-to-end gain at
8B/8,192 context/256 decode steps: batch 4 completed versus corrected
Transformers FP16 batch 3, while decode p95 was 0.474x and prefill median was
1.296x the FP16 control. Its held-out quality interval had upper NLL delta
+0.0156 nats/token. The same path also produced an independently rerun
same-batch throughput gain at 16,384 context/512 decode steps (1.324x).

This is **not** a claim that all workloads or a production scheduler benefit.
The larger-batch KIVI gains at 2,048, 4,096, and 16,384 contexts fail the
predeclared 1.50x prefill limit; KIVI K2V2 quality fails or is uncertain on the
primary long-context probes. GO therefore authorizes only the smallest next
online admission/precision experiment around the K4V4 8K/256 cell. No
scheduler or new quantizer was built here.

## Frozen question, scope, and gates

The question was whether a complete mixed-precision KV execution path can
increase sustainable batch capacity or decode throughput under one fixed GPU
memory budget, fixed FP16 weights/activation precision, quality limits, and
latency limits. “Sustainable” means prefill and every requested decode step
completed in a fresh process while both allocated and reserved peak memory
were within budget. It does not mean that an arrival trace or scheduler was
implemented.

The frozen plan is [`kv-capacity-plan-v2.md`](kv-capacity-plan-v2.md), whose
SHA-256 is recorded in `results/kv-capacity-v2/summary.json`. The budget is
0.90 of the RTX 3090's 25,296,044,032-byte physical memory:
**22,766,439,628 bytes**. The fixed gates were:

* decode median and p95 <= 1.25x the FP16 control;
* prefill median <= 1.50x the FP16 control;
* held-out paired NLL delta 95% upper bound <= +0.02 nats/token;
* no non-finite/safety failure or unhandled termination; and
* either a strictly larger feasible batch or higher output-token throughput at
the same batch.

The primary baseline is corrected Transformers FP16 (`hf_fp16`, SDPA). The
secondary baseline is post-Llama-3.1-fix unchanged SwiftLLM dense FP16. All
weights and activations remain FP16. Mixed candidates are the pinned public
KIVI native K4V4 and K2V2 paths; no KIVI/SwiftLLM adapter or scheduler was
fabricated.

The accepted grid covers Llama 3.1 8B at `(context, decode)` =
`(2048,256)`, `(8192,256)`, `(4096,512)`, and `(16384,512)`, with ordered
boundary searches. Each selected success has one warmup and five timed
repetitions; failures retain exact OOM/error records without invented timing.
Separate synchronized memory replays cover the selected capacity points.
Llama 3.2 1B is a smoke/debug screen and has eight-sample quality probes.

## Confirmed findings

### Capacity and complete-model timing

The table reports the largest accepted completed batch, decode milliseconds
per token (median; p95 across the five timed repeats), prefill median, and peak
as a fraction of the common budget. Each timing call executes all decoder
layers and the actual KV update; only the final-token projection is returned
so full-vocabulary logits do not dominate the capacity measurement. The KIVI and HF runs use the same
`expandable_segments:True` allocator setting in the accepted primary set;
post-fix SwiftLLM is shown as a secondary reference.

| Model/context/decode | Method | Max batch | Decode median / p95 (ms/token) | Prefill median (ms) | Peak/budget |
|---|---:|---:|---:|---:|---:|
| 8B / 2K / 256 | HF FP16 | 12 | 99.51 / 99.64 | 6,469 | 0.983 |
| 8B / 2K / 256 | KIVI K4V4 | 18 | 51.76 / 52.30 | 10,767 | 0.996 |
| 8B / 2K / 256 | KIVI K2V2 | 21 | 51.93 / 52.57 | 14,040 | 0.993 |
| 8B / 8K / 256 | HF FP16 | 3 | 108.48 / 108.50 | 7,079 | 0.989 |
| 8B / 8K / 256 | KIVI K4V4 | **4** | 51.23 / 51.43 | 9,174 | 0.960 |
| 8B / 8K / 256 | KIVI K2V2 | 5 | 52.96 / 53.43 | 13,212 | 0.971 |
| 8B / 4K / 512 | HF FP16 | 6 | 103.06 / 103.20 | 6,587 | 0.983 |
| 8B / 4K / 512 | KIVI K4V4 | 9 | 50.86 / 51.31 | 10,901 | 0.977 |
| 8B / 4K / 512 | KIVI K2V2 | 11 | 52.35 / 52.48 | 15,541 | 1.000 |
| 8B / 16K / 512 | HF FP16 | 1 | 67.07 / 67.16 | 5,303 | 0.899 |
| 8B / 16K / 512 | KIVI K4V4 | 2 | 50.49 / 62.24 | 10,600 | 0.958 |
| 8B / 16K / 512 | KIVI K2V2 | 2 | 50.20 / 50.71 | 10,586 | 0.945 |

At the K4V4 8K capacity point, the ratios versus HF batch 3 are decode
median **0.472x**, decode p95 **0.474x**, prefill **1.296x**, and batch
**4/3**. Its 95% bootstrap interval for decode median is retained in the
summary and raw JSON. The K4V4 capacity point is therefore a qualifying
capacity win after applying every declared gate and its matching 8K quality
probe.

At K4V4 2K, 4K, and 16K capacity points, decode is favorable but prefill
ratios are 1.665x, 1.655x, and 1.999x; these are not qualifying end-to-end
capacity wins. K2V2 has still larger observed batches but fails the prefill
limit and has weaker quality evidence.

### Equal-batch throughput

Equal-batch runs use batch 1 and the same prompt/decode workload. The
independent K4V4 16K rerun is selected in the summary; the original equal-batch
run remains beside it.

| Context/decode | Candidate | Decode median ratio / p95 ratio vs HF | Prefill ratio | Throughput ratio |
|---|---|---:|---:|---:|
| 2K/256 | K4V4 | 1.581 / 1.573 | 0.997 | 0.633 |
| 4K/512 | K4V4 | 1.526 / 1.564 | 0.999 | 0.655 |
| 8K/256 | K4V4 | 1.206 / 1.248 | 0.993 | 0.829 |
| 16K/512 | K4V4 | **0.755 / 0.792** | **0.974** | **1.324** |

The 16K K4V4 rerun measured 50.71 ms/token [bootstrap 95% CI
50.54--52.74] versus HF 67.17 ms/token [67.13--67.22], with a complete 512
step trajectory. This is the qualifying same-batch decode-throughput result.
K2V2 also has a raw 16K same-batch throughput ratio of 1.324, but its matching
16K quality interval has upper NLL delta +0.0372 and therefore does not pass.

### Held-out quality

Quality was run separately from timing, after the benchmark workload was
chosen. It used eight Wikitext-2 validation windows (identified continuation
boundary query) and eight naturally occurring HellaSwag validation prompts
with at least 64 tokens. No prompt was repeated or padded; the threshold is a
recorded KIVI support limitation. The quality path records paired NLL, KL,
logit MSE, per-token/per-step divergence, EOS handling, termination reason,
non-finite logits, safety failure, and repetition pathology.

| Model/context | Method | NLL delta mean [95% bootstrap upper] | Long-context top-1 | Free prefix agreement | Quality gate |
|---|---|---:|---:|---:|---|
| 8B/2K | K4V4 | +0.0030 [+0.0081] | 99.2% | 78.1% | pass |
| 8B/8K | K4V4 | +0.0042 [+0.0156] | 97.3% | 78.1% | **pass** |
| 8B/16K | K4V4 | +0.0003 [+0.0033] | 98.0% | 78.1% | **pass** |
| 8B/2K | K2V2 | +0.0581 [+0.1034] | 92.6% | 40.2% | fail |
| 8B/8K | K2V2 | +0.0038 [+0.0278] | 89.5% | 40.2% | uncertain/fail |
| 8B/16K | K2V2 | +0.0185 [+0.0372] | 89.8% | 40.2% | uncertain/fail |

For the qualifying K4V4 cells, mean long-context KL(reference||candidate)
was 0.00213 at 8K and 0.00130 at 16K; corresponding logit MSE was 0.0162
and 0.00930. No accepted 8B free-running sample produced non-finite logits, an identified
termination failure, or a repetition pathology. They reached the declared
64-token maximum rather than EOS; this is recorded as `termination_reason:
max_tokens`, not silently treated as EOS. The 1B smoke quality is retained in
the raw artifacts; K2V2 has one detected repetition pathology and both 1B
candidates are weaker than the primary 8B evidence.

### Memory, ownership, and allocation evidence

The corrected accounting audit is in
[`kv-measurement-audit-v2.md`](kv-measurement-audit-v2.md). It fixes the old
V-only DynamicCache accounting, deduplicates aliased storage, fixes the
head/token permutation, follows request-local block tables, and separates
one-layer experiments from all-layer conversion. KIVI capacity records
inventory the returned legacy tuple's K and V tensors, packed payloads,
scales, residuals, token positions, allocator allocated/reserved bytes, reset
window peaks, and post-release observations.

Representative prefill logical K+V cache payloads at selected capacity points
were 3,221,225,472 B for HF 8B/8K/batch 3, 1,347,944,448 B for K4V4
8B/8K/batch 4, and 1,015,152,640 B for K2V2 8B/8K/batch 5. These are logical
cache payloads, not free HBM claims. Full-process peak memory, including model
workspace and outputs required by the measured path, is the capacity gate.
The allocator does not expose a reliable per-allocation workspace owner, so
workspace attribution remains a limitation even though the peak is measured.

The direct additional-request probe used the unchanged SwiftLLM block manager:
with an exact 13-request 2K pool, a 14th request failed with zero free blocks;
with one extra page group it completed and consumed 128 additional blocks.
Peaks were 22,336,765,952 B after the base batch and 22,630,367,232 B after
the additional request, both below budget. This verifies real block allocation
and release, without implementing admission policy. The analogous 8K probe
was blocked by a concurrent device-side memory condition and is retained as a
blocked raw attempt.

### Declared sensitivity checks

Sensitivity was computed from the recorded full-process peaks and repeat
summaries, not by selecting a favorable margin afterward. At physical-memory
margins of 5%, 10%, and 15%, the number of the four 8B workloads fitting was
respectively: HF FP16 **4/4, 4/4, 1/4**; KIVI K2V2 **4/4, 4/4, 0/4**; and KIVI
K4V4 **4/4, 4/4, 0/4**. The 15% result is a stricter stress test than the
frozen 10% margin and does not change the declared decision.

For method-capacity comparisons, the number of 8B KIVI workload cells passing
both decode median/p95 and the fixed 1.50x prefill gate was K2V2 **0/4** and
K4V4 **1/4** at decode multipliers 1.10x, 1.25x, and 1.50x. For equal-batch
K4V4, the corresponding counts were **1/4, 2/4, 2/4** at those multipliers.
The NLL upper-bound sensitivity over the three tested KIVI long-context
windows was K2V2 **0/3, 0/3, 2/3** and K4V4 **2/3, 3/3, 3/3** at limits
+0.01, +0.02, and +0.05 nats/token. Full machine-readable values are in
`summary.json:sensitivity_results`.

### Provenance and verification

* Hardware: NVIDIA RTX 3090, compute capability 8.6, 24,576 MiB; environment
  details are in `results/kv-measurement-v2/environment_snapshot.json` and the
  v2 provenance manifest.
* Software: Python 3.12.3, PyTorch 2.4.0+cu121, Transformers 4.51.3, Triton
  3.0.0, vllm-flash-attn 2.6.2, driver 580.159.03.
* Models are local Llama 3.1 8B and Llama 3.2 1B snapshots. Full model-file
  hashes are in `results/kv-capacity-v2/provenance/manifest.json`.
* Public KIVI is pinned at `876b4d2d08e3b1d5f70d0969c299d8c7c42ddfb6`;
  SwiftLLM upstream is pinned at
  `682cf9a28f97f7490409981a2f181528f377eb5d`.
* Accepted raw timing records, failures, memory replays, quality JSON, exact
  invocations, and source snapshots are under `results/kv-capacity-v2/`.
* Final checks passed: `verify_kv_capacity_v2.py`, historical
  `verify_artifacts.py`, `verify_public_kv.py`, 29 repository unit tests,
  Python compilation, and `git diff --check`. The combined output is
  `results/kv-capacity-v2/verification-final.log`.

## Supported but uncertain findings

* K4V4's 8K result is a strong bounded feasibility signal, but it compares
  corrected Transformers SDPA with KIVI's native FlashAttention-compatible
  implementation. Some decode benefit may be backend behavior rather than
  low-bit KV alone. A matched attention backend or kernel-isolated ablation is
  needed before claiming a quantization-only speedup.
* The 8K K4V4 quality result has the planned minimum of eight independent
  windows/prompts, not a broad task benchmark. HellaSwag prompts were selected
  naturally long enough for the public KIVI GQA/residual implementation.
* The fixed batch cells are complete full-model trajectories, but not a
  continuous-arrival, preemption, or concurrent request trace. The GO is for
  bounded follow-up work, not production capacity.
* Allocator peaks include workspace and runtime allocations, while owner-level
  attribution and fragmentation under a long-lived serving process remain
  uncertain. KIVI's native packed storage is accounted for, but it is not a
  SwiftLLM reusable block-table representation.
* The 1B result is diagnostic only. It exposes much worse K2V2 quality and is
  not used to extrapolate 8B behavior.

## Blocked questions

* No online scheduler, admission controller, precision router, or KIVI-to-
  SwiftLLM page-table adapter was implemented, by design. Thus additional
  request behavior for compressed K/V is not established; only the direct
  dense SwiftLLM allocator probe is confirmed.
* The 8K additional-request allocator replay was blocked by a concurrent
  device-side memory condition. Re-run it on an isolated GPU before making a
  long-lived allocator claim.
* The exact quality of a production K4V4 adapter, including EOS behavior under
  the actual serving tokenizer/generation policy, remains unmeasured.
* KVQuant local Llama GQA integration is blocked by its MHA-only seam; Kitty
  lacks the pinned tree's Llama wrapper; QAQ's evaluator dequantizes before
  ordinary attention; Minima-KV's canonical repository remains unresolved.
  These implementations were inspected and not combined into a fabricated
  result. Details are in [`public-kv-study.md`](public-kv-study.md).
* Capacity under concurrent arrivals, fragmentation, preemption, and request
  cancellation is not answered by this fixed-batch study.

## Remaining uncertainty and next action

The next smallest experiment authorized by this GO is a two-method, fixed
8B/8K/256 trace around the observed boundary: corrected HF FP16 batch 3 and a
K4V4 path at batch 4, with one real additional-request arrival, allocator
free-block/release observations, and the same eight-window quality gate. It
must use a format-aware reusable arena or page adapter and keep the measured
K/V payload, residual, metadata, workspace, and allocator ledger explicit.
Do not add a scheduler until that adapter reproduces the current K4V4 result
without backend or duplicate-storage ambiguity.

A future failure of that matched adapter, the 8K allocator replay, or the
quality gate should downgrade the online decision to **BLOCKED/NO_GO**. The
present **GO** is limited to the reproducible K4V4 8K capacity point and K4V4
16K same-batch throughput point under this declared research budget and gates.
