# Completion audit against the active objective

This audit distinguishes completed evidence from requirements that remain blocked. It is not a claim that native serving implementation is complete.

## Concrete deliverables / success criteria

1. Audit and correct the existing Q/K/V claims and storage formulas.
2. Measure Q, K, V, O, and a per-layer FFN block at 4/8/16 numerical proxy precision.
3. Use separate calibration and held-out data, uncertainty estimates, and a task-level metric where feasible.
4. Construct uniform, projection-only, layer-only, and layer-by-projection policies from reproducible single-unit measurements under explicit storage accounting.
5. Run combined policies, test additive predictions against measured quality, and report global versus favorable per-query oracle results.
6. Keep the SwiftLLM baseline/no-op path and scheduler/KV-cache scope unchanged.
7. Update evidence artifacts and give separate decisions for static structured precision, native kernels, and query-aware batching.

## Prompt-to-artifact checklist

| Explicit requirement | Evidence inspected | Audit result |
|---|---|---|
| Pin and preserve SwiftLLM baseline | `vendor/swiftLLM-upstream/UPSTREAM_COMMIT`, `docs/baseline.md`, baseline logs | Pass |
| Verify all-FP16/no-op equivalence and regression tests | `results/baseline/noop-output-comparison.txt`, `tests/test_precision.py`, `tests/test_structured_precision.py` | Pass: 9 tests after building the checked-in extension |
| Do not change scheduler or implement KV-cache quantization | `vendor/swiftLLM/swiftllm/server/scheduler.py`, precision metadata path, structured JSON `scope` | Pass: no precision scheduler branch; KV cache explicitly false |
| Correct unsupported Q/K ordering | Corrected table in `docs/feasibility-report.md`; claim matrix | Pass: only V>Q/K is retained; Q-vs-K flip is stated |
| FP16 has no scale overhead | `Unit.storage`, manual cases, verifier recomputation, JSON ledgers | Pass |
| Count applicable quantized payload padding/scales/zero points/metadata | `Unit.storage`, `storage_accounting`, verifier | Pass for the declared proxy representation; native packed headers remain unknown and are not silently counted |
| Include fixed model weights in cost | `fixed_fp16_bits` and `uniform_profiles` in structured JSONs; verifier | Pass: embeddings/norms/untied LM head are included as constant FP16 cost |
| Measure Q/K/V/O/FFN per layer at 4/8/16 | `single_unit_sensitivity`; 80 units for 1B and 160 for 8B; verifier | Pass |
| Preserve weight-projection vs KV-cache distinction | Script scope/proxy fields, report and claim matrix | Pass |
| Use calibration and held-out text | Wikitext train/validation IDs and data metadata in both JSONs | Pass |
| Avoid only eight handcrafted prompts | 1B: 16 calibration + 32 held-out sequences; 8B: 8 + 16; HellaSwag task subsets | Pass, with explicit 8B sample limitation |
| Estimate uncertainty | Per-policy bootstrap 95% intervals for NLL/logit/KL/top-1 and task accuracy | Pass |
| Do not claim prompt-type effects from one example | New headline analysis has no prompt-type claim; old prompt-type records remain sample-limited | Pass |
| Use 1B exploration and 8B confirmation | `llama32_1b_structured.json`, `llama31_8b_structured.json` | Pass |
| Evaluate uniform W4/W8/FP16 | `policies` rows and report tables | Pass |
| Evaluate projection-only, layer-only, layer-by-projection | `policies` rows; verifier requires all kinds | Pass |
| Use reproducible optimizer | Exhaustive projection enumeration and deterministic greedy upgrade records in profile `optimizer` fields | Pass; greedy is explicitly not presented as globally optimal |
| Test additive sensitivity assumption | `interaction_calibration` with measured/predicted residual and bootstrap intervals | Pass: additive predictor is not trusted |
| Compare actual combined profiles on held-out outputs | `heldout` per-sample NLL, paired logits, KL, top-1, and policy/task outputs | Pass |
| Include task/generation quality where feasible | HellaSwag teacher-forced multiple-choice accuracy and bootstrap CIs | Pass as task metric; free generation remains blocked |
| Compare global profile and favorable per-query oracle | `query_oracle` in each JSON | Pass: exact W8 candidates include an integer-budget structured profile; oracle NLL headroom is small (~0.16–0.17%) and logit-MSE worsens |
| Do not train router without headroom | Report and claim matrix decision | Pass |
| Do not implement scheduler, swapping, KV quantization, native kernels, or optimize grouping path | Source diff/scope and report blocked-work section | Pass |
| Reproduce selected profiles from configs/raw results | `unit_bits`, storage ledger, per-sample results, verifier recomputation | Pass |
| Verify headline comparisons use held-out data | Verifier checks calibration/held-out counts and disjoint IDs; report uses held-out table | Pass |
| Update claim/evidence matrix | `docs/claim-evidence.md` | Pass |
| Give separate go/no-go decisions | `docs/feasibility-report.md` decision summary | Pass: all three are no-go under the requested gates |

## Remaining uncovered or weak requirements

- The experiment is a symmetric fake-quantization proxy, not a native packed W4/W8 representation; packed metadata, alignment, and kernel overhead remain blocked.
- HellaSwag subsets are small, especially 16 items for 8B, and measure teacher-forced multiple-choice rather than free generation.
- The per-query oracle ranges over executed candidate profiles at the closest W8 cost, not every possible profile; it is favorable but not an exhaustive ILP oracle.
- The structured runs use Wikitext validation for headline language quality and do not establish broad task-generalization or long-generation quality.
- Published QAQ and MorphServe results remain documented but not independently reproduced.

## Audit conclusion

The model-level evidence phase is complete enough to make the requested decisions: no defensible structured Pareto advantage was found, the query oracle has only small metric-dependent headroom with worse NLL-selected logit MSE, and the current objective's native-kernel and scheduler gates therefore remain closed. The remaining questions are explicitly blocked rather than treated as achieved.
