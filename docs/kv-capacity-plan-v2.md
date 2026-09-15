# Fixed-budget mixed-precision KV capacity plan v2

**Status:** frozen before the v2 held-out evaluation and capacity runs.
**Scope:** RTX 3090, local Llama 3.2 1B Instruct screening and Llama 3.1 8B
primary confirmation; no online scheduler or new kernel.

## Decision question

Can recent-FP16/older-low-bit KV storage (KIVI's native FP16 residual window
plus native 4-bit or 2-bit older cache) increase the largest full-model batch
that completes prefill and the entire decode trajectory under one declared
GPU-memory budget, without violating latency or task-quality constraints?

The primary comparison is corrected Transformers FP16 versus native KIVI
K4V4 and K2V2 at the same model weights, prompt/decode workload, and device
budget. The unchanged SwiftLLM FP16 engine is a secondary serving-system
comparison under comparable full-model conditions. KIVI and SwiftLLM are not
combined into a fabricated adapter.

## Supported configurations

| Item | Frozen choice |
|---|---|
| Models | Local Llama 3.2 1B Instruct (screen/debug) and Llama 3.1 8B (primary) |
| Weights | FP16 for every method; same local checkpoint hash per model; no weight quantization |
| Mixed methods | Pinned KIVI native K4V4 and K2V2, `group_size=32`, `residual_length=32` FP16 recent tokens |
| FP16 controls | Corrected Transformers `use_cache=True`; unchanged SwiftLLM dense FP16 cache where runnable |
| KV scope | Both K and V; KIVI's native packed older region and FP16 residual region |
| Page/residual probes | Prompt lengths 2,048 and 8,192 with 256 decode steps first; then 4,096/16,384 and 512 steps only where feasible and decision-relevant; lengths around residual/page boundaries are retained as explicit mechanism checks |
| Batch search | Ordered candidates `1, 2, 4, 8, 16, 32` for 1B and `1, 2, 4, 8` for 8B, stopping only after a recorded OOM/unsupported boundary and a successful neighboring cell; expand around an ambiguous boundary |
| Repeats | Screening: 1 cold start plus 3 warmed runs per feasible cell. Confirmation: 1 cold start plus 5 warmed runs for the best FP16 and each promising mixed cell, with independent process rerun. Memory trace: one separate synchronized replay per cell. |
| Timing | Prefill starts immediately before model forward after input preparation; decode wraps complete model forwards and actual KV growth, with one end synchronization; prefill and decode reported separately. Quality and host scalar extraction are outside timing. |
| Memory | Every process records model/runtime allocations, K and V, residuals, packed metadata, workspaces, allocator allocated/reserved, reset-window peaks, actual token growth, and release observations. OOM has no invented latency. |

## Fixed memory budget and safety margin

The declared budget is **0.90 × physical device memory**, computed once from
`torch.cuda.get_device_properties(device).total_memory`, with the exact byte
value recorded in every cell. On a 24,576 MiB RTX 3090 this is approximately
22,118 MiB. Each method is run in a fresh process with the same
`torch.cuda.set_per_process_memory_fraction(0.90)` guard when supported. No
method receives a larger budget. The remaining 10% is the common safety
margin for driver/runtime variability. A cell is feasible only when its
prefill and every decode step complete and its measured reset-window peak
allocated bytes remain at or below the declared budget. Cache payload fitting
before decode is insufficient.

The research assumption is that 10% is a reasonable reproducibility margin,
not a user requirement. Final analysis reports a sensitivity range for 5%,
10%, and 15% margins using the recorded peaks; no favorable margin is selected
afterward.

## Cell protocol and adaptive stopping

1. Verify audit prerequisites, source pins, model files, KIVI extension, free
   authorized GPUs, and baseline fixes before launching cells.
2. Screen 1B first at 2K/8K × 256. Run FP16, K4V4, and K2V2 at increasing
   batches. Record success, OOM, unsupported, or runtime error with the exact
   exception and no latency values for failures.
3. Repeat the same screen for 8B as the primary confirmation. Do not infer 8B
   capacity from 1B scaling.
4. For each method, run 4K/16K × 512 only when the screen establishes enough
   headroom or the boundary is needed to distinguish capacity from overhead.
   Freeze the choice before looking at held-out quality.
5. At each boundary run one lower and one higher batch, then independently
   rerun the largest promising feasible cell. Expand only if the two runs
   disagree, the peak is within 2% of budget, or FP16 and mixed rankings are
   indistinguishable within timing uncertainty.
6. Keep clean timing, memory-instrumented replay, paired forced-prefix
   diagnostics, and free-running/held-out task quality as separate run modes.

Repeated warmed runs use controlled method order (FP16, K4V4, K2V2) and fresh
processes. Raw repeats, medians, p95, bootstrap or t-based 95% intervals, and
all attempted cells are retained.

## Latency and quality constraints

No project requirement supplied numeric SLOs, so these are declared research
assumptions and are not retuned after held-out results:

* decode per-token median and p95 must not exceed **1.25×** the best feasible
  FP16 point at the same workload for a capacity win;
* prefill median must not exceed **1.50×** that FP16 point;
* paired held-out NLL delta must have its 95% upper bound ≤ **+0.02 nats/token**
  for a provisional quality pass; if the interval is wider or sample support
  is weak, quality is **uncertain**, not passed;
* free-running generation must have no safety/termination failure and must be
  qualitatively inspected; a short sample is not sufficient to certify quality;
* a capacity win requires a strictly larger feasible batch than the best FP16
  point at the same prompt/decode workload, or higher aggregate output-token
  throughput at the same batch, while satisfying the latency and quality gates.

Sensitivity is reported for latency limits of 1.10/1.25/1.50× and NLL limits
of +0.01/+0.02/+0.05 where sample size permits. These are assumptions, not
claims about a production SLO.

## Held-out quality sampling

Quality is never used to select screening cells. The held-out set is loaded
only after screening choices are frozen and uses distinct prompts from timing.
It contains two families:

* **Long-context information use:** held-out Wikitext-2 validation windows
  with a planted/identified information query at the end, at least 8 distinct
  windows per model where memory permits. Report next-token NLL, KL/logit MSE
  against the same-path FP16 reference, top-1 agreement, and per-window
  values. This tests use of older context rather than only the residual.
* **Free-running generation:** at least 8 distinct instruction/topic prompts
  (and the available HellaSwag validation examples when runnable), 64 greedy
  decode tokens per prompt. Record token IDs/text, termination, repeated-token
  pathologies, and per-step divergence against FP16. A short or blocked sample
  is explicitly uncertain.

The primary 8B quality result takes precedence. The 1B result is a debugging
screen. Bootstrap 95% intervals are computed over independent prompt/window
units, not tokens alone; if the planned count is not reached, the report
retains an uncertain quality classification.

## SwiftLLM comparison boundary

For every configuration in which the unchanged SwiftLLM dense FP16 engine is
runnable, report its full-model prefill/decode timing, actual cache growth,
peak allocated/reserved bytes, and largest completed batch under the same
budget and prompt/decode trajectory. A KIVI win over corrected Transformers
is preliminary backend evidence only. If SwiftLLM cannot run at the matching
length/batch, the system-level comparison is marked blocked rather than
extrapolated.

## Decision rule and outputs

The final report must separate confirmed findings, supported-but-uncertain
findings, blocked questions, and remaining uncertainty. The single decision
is one of:

1. **Proceed** to a bounded online admission/precision experiment, naming the
   smallest next experiment and baselines, only if a mixed method has a
   replicated fixed-budget capacity/throughput win under the declared gates;
2. **Obtain specifically missing evidence first**, if the result is promising
   but replication, quality, or SwiftLLM comparison is incomplete; or
3. **Stop this tested GPU-resident compression path**, if no mixed method wins
   the tested boundary under the gates.

Required artifacts are under `results/kv-capacity-v2/`, including an immutable
plan snapshot/hash, environment/source fingerprints, one raw record for every
attempted cell (including failures), repeat timing and memory traces, paired
quality records, summaries with uncertainty, and verification logs. The final
report is `docs/kv-capacity-report-v2.md`.
