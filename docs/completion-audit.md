# Completion audit

Audit performed against the user-provided objective after the current baseline and experiments. This is an evidence checklist, not a claim that the larger native-systems research program is complete.

## Concrete deliverables

The objective decomposes into: (1) pinned and reproducible SwiftLLM baseline; (2) exact execution map; (3) minimal per-request Q/K/V/O/FFN metadata seam with unchanged FP16 default; (4) offline Q/K/V sensitivity matrix and layer/query/phase analysis; (5) fixed-vs-oracle comparison; (6) raw artifacts and tests; (7) claim/evidence matrix and final findings/recommendation that clearly marks blockers; (8) no premature full scheduler/KV-swapping implementation.

## Prompt-to-artifact checklist

| Requirement | Concrete evidence inspected | Audit result |
|---|---|---|
| Use interestingLSY/swiftLLM at commit `682cf9a28f97f7490409981a2f181528f377eb5d` | `vendor/swiftLLM-upstream/UPSTREAM_COMMIT`, `vendor/swiftLLM/UPSTREAM_COMMIT`, `results/baseline/upstream_git_verification.txt` | Pass |
| Preserve an untouched upstream baseline | Clean archived source in `vendor/swiftLLM-upstream/`; upstream run log in `results/baseline/swiftllm_unmodified_1b.log` | Pass |
| Reproduce baseline with exact environment/model/commands/hardware | `docs/baseline.md`, three environment snapshots, `pip-freeze.txt`, `requirements-lock.txt`, `toolchain.txt`, `invocations.json`, model file hashes, baseline shell command | Pass, subject to machine-local weight paths |
| Attribute artifacts to both upstream and research source | `results/baseline/research-provenance.json`, sensitivity JSON `swiftllm_upstream_commit` and `swiftllm_research_diff_sha256`, verifier check | Pass |
| Baseline correctness before research | Clean upstream run and research no-op run; `results/baseline/noop-output-comparison.txt` | Pass: exact output lines after excluding creation time |
| Include QAQ paper and code context | `references/qaq-2403.04643.pdf`, `vendor/qaq/`, `docs/papers.md` with arXiv/code IDs and hash | Pass |
| Include MorphServe paper context | `references/morphserve-2506.02006-v2.pdf`, `docs/papers.md` with exact title/authors/version/hash | Pass; MorphServe implementation itself is not reproduced |
| Map request → scheduler → model → Q/K/V → KV cache with exact locations | `docs/source-map.md` and current `nl -ba` source inspection | Pass |
| Represent separate Q/K/V/O/FFN precision per request | `vendor/swiftLLM/swiftllm/precision.py`, request structs, infer state, model and transformer call sites | Pass |
| Keep default FP16 path behaviorally unchanged | All-FP16 branch retains original `F.linear`; exact baseline/no-op logs; unit test | Pass for tested offline path |
| Keep scheduling semantics unchanged | `Scheduler` has no precision branch; `SchedulerSemanticsTests`; claim matrix | Pass for admission semantics; no online mixed-profile trace |
| Metadata reaches intended Q/K/V call sites | Transformer layer passes profile metadata to q/k/v/o/ffn; `swiftllm_precision_metadata_smoke.json` changes only opted-in proxy result | Pass |
| Invalid precision input is rejected at the API boundary | `_raw_request_from_dict` returns HTTP 422; `test_invalid_api_profile_is_a_422` | Pass |
| Add offline `{4,8,16}^3` controlled matrix | 27 all-layer configs in each matrix JSON; `scripts/qkv_sensitivity.py`; verifier checks exact set | Pass |
| Clearly label quantization proxy and avoid speedup claims | Script/docstrings, JSON `numerical_proxy`, report and README warnings | Pass |
| Measure Q/K/V differences | All-layer single-projection comparisons in both model matrix JSONs and report table | Pass, sample-limited |
| Measure layer differences | 16-layer 1B and 32-layer 8B raw layer-sweep JSONs; report ranges | Pass, sample-limited |
| Measure query/prompt type differences | Per-query, per-position, and `by_prompt_type` records; exact-budget oracle choices | Pass, sample-limited |
| Measure prefill/decode positions | Each per-query record has prefill positions and one-token decode positions | Pass, only 4 decode tokens |
| Record per-query and aggregate quality/error metrics | JSON `per_query`, `prefill`, `decode`, logit MSE/RMSE/max/cosine/KL/top-1 | Pass for proxy metrics; task accuracy/perplexity not run |
| Preserve raw experiment outputs | `results/sensitivity/*.json` and baseline logs | Pass |
| Verify FP16/no-op against an explicit tolerance | JSON `correctness.fp16_control` defines exact max-abs 0 and zero top-1 mismatches; log comparison exact | Pass for regression surface |
| Compare fixed profiles with query-specific oracle at equal/comparable budgets | `unweighted_fixed_oracle_comparison`, non-comparable `unweighted_sum_oracle_comparison`, and parameter-count/scale-aware `weighted_budget_oracle_comparison` in matrix JSONs; report | Pass; weighted oracle is offline and coarse |
| Treat non-native timings as overhead only | Per-config timing and explicit caveats; no throughput claim | Pass |
| Do not implement full precision-aware scheduler, CPU/GPU bit swapping, KV-cache quantization, per-element precision, or production kernels prematurely | Source diff contains only metadata/eager proxy; report lists these as blocked/out of scope | Pass |
| Update claim/evidence matrix between iterations | `docs/claim-evidence.md` records status, artifacts, limitations, next action | Pass |
| Final report separates confirmed/uncertain/blocked and gives go/no-go | `docs/feasibility-report.md` sections and conditional recommendation | Pass |
| Tests and artifact gate pass | `results/baseline/unit-tests.log`, `results/baseline/artifact-verification.log`, and `scripts/verify_artifacts.py` | Pass, verifier itself explicitly says it does not resolve native blockers |

## Uncovered or weak requirements

- The sensitivity harness uses Hugging Face Llama modules plus the shared proxy, not SwiftLLM's Triton/PagedAttention execution. SwiftLLM is covered as the untouched serving baseline and integration target; a SwiftLLM-native logits/attention sensitivity harness is not present.
- QAQ's LLaMA 2 downstream task results and MorphServe's online serving results are documented from primary sources but not independently reproduced.
- The local AWQ MorphServe weights are available but blocked by the pinned ordinary loader.
- Native packed low-bit kernels, true KV-cache quantization, online mixed-profile batching, and CPU/GPU swapping remain intentionally unimplemented.
- Query-specific oracle gains are modest (up to 2.65% for the 1B sample and 0.013% for the 8B sample at exactly equal weighted projection-storage bits), and prompt/decode samples are small. The larger unweighted-sum diagnostic is not a memory comparison. This is evidence for a next experiment, not evidence for a production scheduler.

## Audit conclusion

The evidence-backed **baseline deliverable** is complete, including explicit blockers and a conditional next-step recommendation. The broader objective is not treated as proving native systems feasibility: the next defensible action is a native W8/W4 projection microkernel/overhead experiment and a larger task-calibrated query-dependence study. Full dynamic scheduling should remain gated on those results.
