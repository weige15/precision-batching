# Fixed-budget mixed-precision KV capacity report v2

**Decision date:** 2026-09-15
**Plan:** [`docs/kv-capacity-plan-v2.md`](kv-capacity-plan-v2.md)
**Raw artifacts:** `results/kv-capacity-v2/`
**Summary manifest:** `results/kv-capacity-v2/summary.json`

## Decision

**STOP this tested GPU-resident compression path; do not invest in a
continuous-batching precision scheduler on these KIVI layouts.**

KIVI's native FP16-residual/older-4-bit and FP16-residual/older-2-bit paths
really do complete larger full-model batches under the same declared budget.
That is a confirmed capacity fact, not a failure of accounting. However, on
the primary Llama 3.1 8B confirmation model, the larger batch did not produce
a verified useful serving win after the declared decode-latency and quality
constraints were applied:

* at 2K/256, FP16 SwiftLLM completed batch 13, K4 batch 18, and K2 batch 21;
  mixed decode latency per token was about 1.79x/1.76x FP16 and aggregate
  output throughput was 0.77x/0.92x FP16;
* at 8K/256, FP16 completed batch 3, K4 batch 4, and K2 batch 5; mixed
  latency was about 1.85x/1.89x and throughput was 0.72x/0.88x;
* at 16K/512, FP16 completed batch 1 and both K4/K2 completed batch 2, but
  repeated timing still gave about 2.22x/2.18x latency and 0.90x/0.92x
  throughput;
* K2's held-out long-context NLL delta was +0.058 nats/token with a 95%
  bootstrap interval of [+0.016,+0.100], above the provisional +0.02 gate.
  K4's +0.003 interval [-0.003,+0.008] passed that provisional NLL gate, but
  did not pass the latency/throughput gate.

This is a negative decision about the tested RTX 3090, Llama 3.1/3.2,
KIVI-native implementation and this budget—not a claim that every future
fused low-bit KV engine is impossible. The narrow alternative is to retain
unchanged SwiftLLM FP16 serving plus existing admission/lossless memory
handling, rather than adding an online precision controller.

Capacity-only gains were confirmed, but none passed the declared combined
latency/throughput/quality gate, so no online experiment is opened. If this
path is deliberately reopened after a new backend or quality fix, the smallest
bounded admission experiment would be an 8B 16K/512 trace with SwiftLLM FP16
batch 1 as the best feasible FP16 baseline and fixed K2V2 batch 2 (plus K4V4
as a control), using the same budget, repeated victim-request and arrival
traces, and the held-out quality gate. It must measure waiting and victim
latency rather than infer them from the fixed-batch numbers above.

## Scope and fixed protocol

* Hardware was an authorized RTX 3090 with 24,576 MiB physical memory. Every
  cell declared `0.90 * total_memory = 22,766,439,628` bytes and used the same
  10% safety margin plus the PyTorch per-process memory guard.
* All model weights remained FP16. Local snapshots were the available Llama
  3.2 1B Instruct and Llama 3.1 8B checkpoints. KIVI used its pinned native
  K2V2 or K4V4 packed old-cache region, `group_size=32`, and an FP16
  `residual_length=32` recent-token window. K and V were both included.
* The primary FP16 serving comparison was the unchanged SwiftLLM dense FP16
  allocator/block-table engine. KIVI was never ported into SwiftLLM. A
  corrected Transformers FP16 path was retained for paired quality reference;
  its ordinary SDPA path was not used to claim a serving-engine win.
* A cell was feasible only if prefill and all decode steps completed and both
  allocator peak allocated and peak reserved memory stayed under the budget.
  OOM/unsupported cells retain the exact error and no invented timing.
* Clean timing, synchronized memory replay, and held-out quality were separate
  invocations. Timing inputs were prepared before the timed region, complete
  model forward and KV growth were inside it, and quality/host extraction were
  outside it. Warmed runs used five repeats for the primary confirmations and
  independent three-repeat reruns; boundary screens used one repeat while
  locating cells and are marked weaker evidence.

## Confirmed findings

### Fixed-budget full-model capacities

The following are largest observed feasible batches, not payload estimates.
Each selected primary 8B cell completed every decode step and has a separate
memory trace. Throughput is aggregate output tokens/second at that method's
own largest feasible batch.

| model | prompt + decode | SwiftLLM FP16 | KIVI K4V4 | KIVI K2V2 |
|---|---:|---:|---:|---:|
| Llama 3.1 8B | 2,048 + 256 | 13 / 415.7 tok/s | 18 / 321.3 | 21 / 380.5 |
| Llama 3.1 8B | 4,096 + 512 | 6 / 198.0 | 9 / 124.1 | 10 / 177.7 |
| Llama 3.1 8B | 8,192 + 256 | 3 / 101.0 | 4 / 72.9 | 5 / 88.9 |
| Llama 3.1 8B | 16,384 + 512 | 1 / 39.2 | 2 / 35.4 | 2 / 36.0 |

The 2K and 8K rows have five-repeat cold/warm confirmations plus independent
reruns. The 4K row is a 512-step boundary confirmation with fewer repeated
cells. The 16K row has five-repeat confirmation and independent cold-start
records. Attempted higher batches are recorded as OOM; for example 8B K2
failed at 22/24 for 2K, at 6 for 8K, and at 3 for 16K, while SwiftLLM failed
at 14 for 2K, 4 for 8K, and 2 for 16K.

The 1B screening model reached larger capacities (for example 2K: SwiftLLM
64 versus K4/K2 128; 8K: SwiftLLM 28 versus K4/K2 32), but those are debugging
screen results and are not used to generalize to 8B.

### Same-batch overhead

At batch 1 on the native deployed paths, five-repeat warmed decode latency
per token was:

| 8B prompt | SwiftLLM FP16 | K4V4 | K2V2 | K4/K2 ratio to FP16 |
|---|---:|---:|---:|---:|
| 2K | 21.75 ms | 55.36 ms | 55.19 ms | 2.55x / 2.54x |
| 8K | 23.28 ms | 56.36 ms | 55.98 ms | 2.42x / 2.40x |

These are complete model decode measurements, not attention-only estimates.
Warm prefill medians at the 8B capacity points were 8.4 s/9.9 s/12.2 s for
Swift/K4/K2 at 2K and 6.3 s/9.1 s/12.3 s at 8K; cold prefill and decode
values are retained in each `cold_start` record. The corrected KIVI 1B/8B probes in `kivi_*_v2.json` independently measured
about 2.02x/2.07x KIVI-over-ordinary-Transformers decode slowdown at their
short forced-prefix scope. The numbers are consistent in direction but are
not substituted for the full-model SwiftLLM comparison.

### Memory and accounting

The memory replays contain 257 positions for each 256-step capacity cell and
513 positions for the 512-step 16K cells. They record model/runtime base
allocation, K and V cache tensors, residual/packed KIVI records, allocator
allocated/reserved values, reset-window peaks, actual token positions, and
release state.

Examples from the primary 8B memory traces:

* at 2K/256 batch 21, K2's final logical K+V cache was 1,224,867,840 bytes;
  the corresponding FP16 K+V payload for the completed trajectory is about
  6,341,787,648 bytes, an observed logical reduction of 80.7%;
* at 2K/256 batch 18, K4's final logical cache was 1,724,645,376 bytes,
  about 68.3% below the corresponding FP16 payload;
* at 16K/512 batch 2, K2's final logical cache was 833,880,064 bytes versus
  about 4,429,185,024 FP16 payload bytes, an 81.2% logical reduction;
* the 8B peak reserved values were approximately 21.1 GiB (K2 2K), 21.1 GiB
  (K4 2K), and 20.5 GiB (K2 16K), all below the same declared budget recorded
  by the artifacts.

Logical reduction is not silently relabeled as physical free HBM. The KIVI
legacy inventory includes both K and V and unique storage; SwiftLLM dense
capacity includes the full preallocated K/V pool. The prior v2 audit remains
binding for packed-arena and page-store accounting; this capacity run does
not claim a SwiftLLM KIVI adapter.

### Quality

Held-out quality was run on distinct Wikitext-2 validation windows (8 for 8B,
4 for 1B) with 32-token continuations and eight distinct HellaSwag validation
prompts for 8B (four for 1B) with 64-token free-running generation. Quality
runs were not used to choose screening cells.

* 8B K2: long-context NLL delta +0.0581, 95% interval [+0.0157,+0.1000],
  mean top-1 match 92.6%, free-running prefix agreement 75.4% — provisional
  quality **fail/uncertain**, not acceptable under the +0.02 gate.
* 8B K4: NLL delta +0.0030, interval [-0.0031,+0.0081], top-1 match 99.2%,
  free-running prefix agreement 84.6% — provisional NLL **pass**, but this
  does not rescue its latency/throughput failure.
* 1B K2: NLL delta +0.1942, interval [+0.0985,+0.2900], top-1 match 78.9%.
  1B K4's interval was [-0.0290,+0.0012]. These are screening evidence only.
* SwiftLLM quality was evaluated on the same native SwiftLLM FP16 path against
  a same-path FP16 control; its paired delta is identically zero by design.
  It is not used as a cross-engine quality equivalence claim. The raw artifact
  retains absolute NLL and generated text, including visible degeneration in
  some native outputs.

### Sensitivity to declared tolerances

The plan's latency sensitivity range does not change the primary decision: the
8B mixed-at-capacity decode ratios are roughly 1.76--2.39x at 2K/4K/8K and
2.18--2.22x at 16K, so none passes even the relaxed 1.50x decode limit. The
8B K4 long-context NLL upper bound (+0.0081) passes the +0.01/+0.02/+0.05
NLL sensitivity thresholds, while K2's +0.1000 upper bound fails all three.
Throughput remains below FP16 at every repeated 8B capacity comparison.

Memory-margin sensitivity was recomputed from recorded peaks, not rerun with a
favorable budget. At a 5% margin (95% of device memory), the primary 8B
capacity rows were unchanged. At a 15% margin (85% budget), observed
capacities changed to: 2K/256 Swift 8, K4 12, K2 16; 4K/512 Swift 5, K4 4,
K2 8; 8K/256 Swift 2, K4 3, K2 3; and 16K/512 all methods 1. This range is
reported as a sensitivity analysis, not as a retuned acceptance choice.

## Supported but uncertain findings

* The KIVI capacity advantage is reproducible at the tested 8B 2K/8K/16K
  boundaries, but the 4K boundary has fewer repeated runs and should not be
  treated as a precise production capacity curve.
* The 1B capacities are useful for debugging and show much larger apparent
  throughput gains, but they do not generalize to the 8B primary model. The
  1B extended 4K/16K scans were not exhaustively expanded beyond their recorded
  boundaries.
* The provisional K4 quality result is based on eight Wikitext windows and
  eight free-running prompts, not a broad task benchmark or human evaluation.
  The free-running prompt is padded to cross KIVI's 32-token residual/GQA
  seam; source sample IDs and prompts remain recorded. This is adequate for a
  directional quality gate, not a product-quality certification.
* The corrected Transformers long-context baseline was useful for paired KIVI
  NLL, but the ordinary SDPA implementation is not the optimized SwiftLLM
  serving path. The system-level comparison is therefore explicitly made to
  the runnable native SwiftLLM FP16 engine for performance.

## Blocked questions

* No arrival trace, queue waiting time, victim-request delay, preemption,
  admission controller, or continuous-batching goodput was implemented. Fixed
  batch throughput is not labeled online goodput.
* CPU offload/swap was not rerun inside this fixed-budget v2 matrix. Existing
  lossless SwiftLLM swap evidence remains historical and is not combined with
  KIVI into an action ranking.
* KIVI has no SwiftLLM page-table adapter in this scope. No claim is made that
  KIVI's contiguous packed layout can be turned into reusable SwiftLLM blocks
  without a new storage/allocator design.
* Full 16K held-out task-quality comparison for every backend was not needed
  after the repeated 16K latency/throughput gate failed; it remains a blocked
  generalization question. No inference from the 2K quality sample is made.
* The ordinary Transformers FP16 long-context capacity path was not used as
  the serving winner because its SDPA memory behavior produced OOMs in the
  long screen. Those failures are preserved, not assigned latency.

## Remaining uncertainty

The experiments do not establish anything about arrival distributions,
waiting, queue fairness, victim-request delay, CPU offload overlap, dynamic
precision changes, residual-window admission, request-specific precision, or
end-to-end continuous-batching performance. They also do not test compressed
CPU swap, a zero-copy KIVI arena, a new quantizer, or a new low-bit kernel.
Those questions would require a new implementation and a fresh matched gate;
starting an online scheduler here would hide, rather than resolve, the
measured native overhead and K2 quality loss.

## Completion audit: requirement-to-artifact checklist

| Objective requirement | Evidence | Disposition |
|---|---|---|
| Begin from audit prerequisites/fixes | `docs/kv-measurement-audit-v2.md`, historical and v2 verifiers, source pins | confirmed |
| Freeze plan before held-out evaluation | `docs/kv-capacity-plan-v2.md`, plan hash in `summary.json` | confirmed |
| Same declared memory budget/safety margin | every successful cell budget; summary has one value `22,766,439,628` | confirmed |
| Preserve FP16 weights | model files, `weights_dtype=torch.float16`, local snapshots | confirmed |
| Native KIVI K2/K4 with FP16 residuals | cell `method_config`: bits 2/4, group 32, residual 32 | confirmed |
| 2K/8K/256 screening | 1B and 8B timing cells plus OOM neighbors | confirmed |
| 4K/16K/512 expansion | 8B extended cells and repeated 16K boundary | confirmed; 4K weaker replication |
| Multiple batch/page/residual boundaries | batch sweeps, residual crossing through 256/512 growth traces | confirmed for tested cells |
| Full prefill and entire decode required | `completed_decode_steps`, memory traces, OOM cells | confirmed |
| Persistent/transient memory accounting | allocator peaks, base runtime, K/V inventories, logical/unique ledgers | confirmed for measured paths |
| Separate cold/warm timing | `cold_start` in nine primary confirmation artifacts; warmed repeats | confirmed for primary confirmation |
| Same-batch overhead | `same-batch/*.json`, five repeats at 2K/8K | confirmed, native cross-backend scope |
| Same-budget capacity comparison | `summary.json` capacities/comparisons | confirmed |
| Explicit latency and quality tolerances | frozen plan: 1.25x decode, 1.50x prefill, +0.02 NLL | confirmed as research assumptions |
| Distinct held-out task families | Wikitext validation and HellaSwag validation quality artifacts | confirmed, limited sample |
| Same deployed path for quality/performance | native KIVI candidate and native SwiftLLM FP16 control | confirmed with reference caveat |
| SwiftLLM FP16 comparison | full-model native cells on RTX 3090, including 8B | confirmed |
| All attempted cells and rejected/OOM cases | `summary.json` 214 timing attempts, 48 failures; raw JSON retained | confirmed |
| Regression/artifact verification | 28 historical tests, v2 verifier, public verifier, capacity verifier, pycompile, diff checks | confirmed below |
| Final categories and one decision | this report | confirmed |

## Reproduction and verification

```bash
.venv/bin/python scripts/aggregate_kv_capacity_v2.py
.venv/bin/python scripts/verify_kv_capacity_v2.py
PYTHONPATH="$PWD/vendor/swiftLLM:$PWD/vendor/swiftLLM/csrc" \
  .venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/verify_artifacts.py
.venv/bin/python scripts/verify_public_kv.py
.venv/bin/python -m py_compile \
  scripts/kv_capacity_experiment_v2.py \
  scripts/kv_capacity_quality_v2.py \
  scripts/aggregate_kv_capacity_v2.py \
  scripts/verify_kv_capacity_v2.py
```

The capacity worker invocation shape is:

```bash
CUDA_VISIBLE_DEVICES=2 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONPATH="$PWD/vendor/swiftLLM:$PWD/vendor/swiftLLM/csrc:$PWD" \
.venv/bin/python scripts/kv_capacity_experiment_v2.py \
  --output results/kv-capacity-v2/<cell>.json \
  --method kivi2|kivi4|swiftllm_fp16 \
  --model-family llama31_8b \
  --context-tokens 2048 --decode-tokens 256 --batch-size 21 \
  --instrumentation timing --warmups 1 --repeats 5
```

Every raw JSON records the complete argv, model snapshot files, source
fingerprint, public KIVI commit, SwiftLLM pin, budget, allocator observations,
status, and release state. The old development/contaminated-device attempts
remain in `results/kv-capacity-v2/` but are excluded from the decision summary
and are explicitly identified by the aggregator.
