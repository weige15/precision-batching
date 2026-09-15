# Completion audit: mixed-precision KV capacity objective

This audit was performed against the actual current checkout, raw JSON, command
output, and verifier after the final report was written. It does not treat a
green verifier as sufficient evidence for requirements the verifier does not
cover.

## Objective restatement

Produce a reproducible RTX 3090 feasibility verdict for whether native mixed
K/V precision (KIVI K4V4 and K2V2 with FP16 residual windows) increases
full-model sustainable batch capacity or decode throughput under one fixed
memory budget, fixed FP16 weights/activations, declared latency gates, and
held-out quality gates. Llama 3.1 8B is primary; Llama 3.2 1B is smoke/debug.
Audit K/V accounting, duplicate storage, layouts, incremental updates, full
model timing, actual allocation, peak memory, repeated uncertainty, and
quality. Do not add a scheduler or quantizer. Preserve history and finish
with `docs/kv-capacity-report.md` separating certainty categories.

## Prompt-to-artifact checklist

| Explicit requirement | Evidence inspected | Status |
|---|---|---|
| RTX 3090 and Llama 3.1 8B primary | `results/kv-measurement-v2/environment_snapshot.json`; 8B raw cells under `results/kv-capacity-v2/` | Confirmed: RTX 3090, SM86, 24,576 MiB; 8B is the gate-bearing model |
| Llama 3.2 1B smoke tests | `results/kv-capacity-v2/quality-final/llama32_1b_*`; existing 1B timing/memory cells | Confirmed as diagnostic, not extrapolated |
| Existing code, KIVI, primary-source boundaries | `docs/kv-measurement-audit-v2.md`, `docs/public-kv-study.md`, `docs/papers.md`, pinned `vendor/public/KIVI` | Confirmed; KVQuant/Kitty/QAQ/Minima-KV were inspected and not combined |
| Frozen benchmark grid before evaluation | `docs/kv-capacity-plan-v2.md`; `summary.json:plan_sha256` | Confirmed: 2K/8K/4K/16K and 256/512 decode workloads, boundary search, 0.90 budget |
| KIVI 4-bit and 2-bit with FP16 residual | KIVI raw `method_config` fields; KIVI pinned commit | Confirmed: K4V4/K2V2, group 32, residual 32, FP16 weights |
| Corrected FP16 control | `primary-expandable/*hf_fp16*`, `equal-batch-final/*hf_fp16*` | Confirmed: Transformers SDPA, final-token projection only, complete decoder calls |
| Equal GPU-memory budget | every selected success `budget.declared_budget_bytes`; verifier | Confirmed: 22,766,439,628 B; allocated and reserved peaks checked |
| Equal batch size | `equal-batch-final/` and `equal-batch-rerun/`; `summary.equal_batch_comparisons` | Confirmed: batch 1 comparisons at all four 8B workloads; K4V4 16K independently rerun |
| Full-model timing | cell `run.timing_boundary`; `kv_capacity_experiment_v2.py` | Confirmed: complete model calls and cache growth are timed; quality and per-token host scoring are outside timing |
| Sustained decode | `completed_decode_steps`; 256/512-step raw traces | Confirmed: selected successes complete every requested decode step |
| Repeated timings and uncertainty | five repetitions in selected capacity/equal-batch cells; bootstrap fields in `summary.json` | Confirmed for selected cells; raw one-repeat historical cells are excluded from final aggregation |
| Actual additional-request allocation | `additional-request/llama31_8b_c2048_base13.json` | Confirmed for dense SwiftLLM: exact pool rejects request with zero free blocks; one-extra pool accepts and consumes 128 blocks |
| Peak memory including runtime/workspace | per-repeat `peak_allocated_bytes`/`peak_reserved_bytes`; memory-final traces | Confirmed as full-process peak; owner-level workspace attribution remains uncertain |
| K+V accounting and duplicate storage | `docs/kv-measurement-audit-v2.md`; `reproduce_kivi.py`; v2 tests | Confirmed: recursive K/V inventory and deduplicated storage; old V-only result withdrawn |
| Layout/permutation and request-local selection | `tests/test_kv_measurement_v2.py`; native audit artifacts | Confirmed by non-symmetric layout and non-identity block-table tests |
| Incremental cache updates and boundaries | memory traces at every decode position; prior native page boundary smoke | Confirmed for measured KIVI/Swift paths; no compressed Swift adapter is claimed |
| Held-out multi-prompt quality | `quality-final/*.json`, including 8B K4V4 at 8K and 16K | Confirmed for the declared minimum: 8 Wikitext windows and 8 natural HellaSwag prompts; NLL/KL/MSE/per-step divergence/termination/pathology fields recorded |
| Declared quality/latency limits | plan; aggregate comparisons; quality intervals | Confirmed: K4V4 8K capacity row passes all gates; K4V4 16K equal-batch throughput row passes all gates |
| No scheduler/new quantizer | `git diff`, source scope, report blocked section | Confirmed; only direct block-manager probe and native KIVI were used |
| Preserve history and unrelated work | old reports/artifacts remain; current dirty files inspected with `git status` | Confirmed; historical files are retained and explicitly excluded when stale |
| Runnable tests, commands, versions, seeds, raw results | `results/kv-capacity-v2/commands.txt`, `provenance/manifest.json`, `provenance/run-environment.json`, raw JSON | Confirmed; commands and exact invocations are retained |
| Required final report | `docs/kv-capacity-report.md` | Confirmed; includes confirmed, supported-but-uncertain, blocked, and remaining-uncertainty sections |
| Final verdict rule | report and `summary.json:verdict`; strict verifier | Confirmed: GO is narrow, tied to K4V4 8K capacity and 16K equal-batch throughput, not a production scheduler claim |

## Verified decision evidence

The final summary contains 52 selected timing cells (34 completed, 18 exact
failure/OOM records), 19 selected memory replays, and 10 final quality artifacts.
The strict verifier reports one qualifying 8B K4V4 capacity gain and one
qualifying equal-batch gain. The 8K capacity result is batch 4 versus HF batch
3 with decode p95 ratio 0.474, prefill ratio 1.296, and quality NLL upper
bound +0.0156. The 16K equal-batch rerun has decode ratio 0.755 median / 0.792
p95, prefill ratio 0.974, throughput ratio 1.324, and quality NLL upper bound
+0.0033.

Final command results, in `results/kv-capacity-v2/verification-final.log`:

* capacity aggregator: pass;
* strict v2 capacity verifier: pass;
* preserved historical artifact verifier: pass;
* public low-bit-KV verifier: pass;
* 29 repository unit tests: pass;
* Python compilation: pass; and
* `git diff --check`: pass.

## Uncovered or bounded requirements

These are not silently treated as complete:

1. There is no KIVI-to-SwiftLLM page-table adapter, so a compressed additional
   request was not tested in a live serving allocator. The direct dense
   additional-request behavior is confirmed; compressed online admission is
   blocked by the explicit no-scheduler boundary.
2. The 8K additional-request replay was blocked by concurrent device memory
   pressure. The exact failure is retained as raw evidence.
3. Peak memory includes workspace/runtime allocations, but PyTorch does not
   provide a reliable owner map for every temporary allocation. Logical K/V,
   unique backing storage, allocator allocated/reserved, and release are still
   separately recorded.
4. KIVI and Transformers use different attention implementations (native
   KIVI FlashAttention-compatible path versus HF SDPA). The GO is an
   execution-path feasibility result, not a kernel-attribution claim.
5. The held-out quality set is a minimum eight-window/eight-prompt study and
   uses an identified continuation-boundary query rather than a planted
   semantic fact. HellaSwag prompts are natural but selected at >=64 tokens
   because the public KIVI GQA/residual path otherwise errors. No EOS occurred
   within 64 tokens; max-token termination and pathology checks are recorded.
6. Continuous arrivals, queueing, preemption, cancellation, and long-lived
   allocator fragmentation are not measured. They are the next experiment,
   not evidence for production readiness.

These bounds are reflected in `docs/kv-capacity-report.md`; they do not negate
the narrow GO because the explicit rule permits either a gate-compliant
capacity gain or a gate-compliant same-batch throughput gain, and both are
observed for K4V4 in predeclared 8B workloads.
