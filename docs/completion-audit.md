# Completion audit against the active structured-precision objective

This audit is against the user-provided objective, not against the previous
commit's completion claim. The branch was `structured-precision-evidence` at
base commit `b5fc47744d6c4aa1a46206439cdb6b70c29a88ad`; the new raw artifacts
record that provenance.

## Concrete deliverables / success criteria

1. Correct the prior audit: additive-FP16-background selections are not
   interaction-aware, while preserving their negative result with a narrowed
   claim.
2. Measure W8-centered W4 and FP16 marginals for every Llama 3.2 1B Q/K/V/O/
   FFN layer unit, with paired quality metrics and exact storage deltas.
3. Use at least three dispersed, non-overlapping random Wikitext train
   calibration shards and keep validation unseen during search.
4. Run a deterministic, budgeted interaction-aware combined-profile search
   from uniform W8; margins may propose moves but cannot be the quality
   objective.
5. Execute the full 3^5 = 243 projection-only assignment sanity check on 1B,
   correcting the former reversed-exponent typo.
6. Evaluate every surviving frontier profile against uniform W8 on a larger,
   dispersed held-out validation set with paired NLL bootstrap intervals.
7. Apply the staged 1B → 8B gate, without initially running 8B, and make one of
   the required final decisions.
8. Verify actual procedure, storage, separation, execution, provenance, and
   gate evidence rather than only declarative flags.
9. Keep native kernels, scheduler/router, KV-cache quantization, swapping,
   serving optimization, and query-specific oracle work out of scope.

## Prompt-to-artifact checklist

| Requirement | Concrete evidence | Audit result |
|---|---|---|
| Work from requested branch/base | `source_provenance` in both `*_interaction_aware.json`; current branch/log | Pass: branch and required base commit recorded |
| Correct prior completion audit | This file; `docs/feasibility-report.md`; legacy artifacts explicitly labeled additive-selected | Pass: old negative claim narrowed |
| Preserve existing negative results | `results/sensitivity/llama32_1b_structured.json` and `llama31_8b_structured.json` retained | Pass; only optimizer typo text was corrected |
| W8 background | `w8_centered_marginals.background_profile`; uniform W8 reference logits | Pass |
| Every 1B Q/K/V/O/FFN layer unit | 80 units in `w8_centered_marginals.records`; verifier checks coverage | Pass |
| W8→W4 and W8→FP16 actual combined calibration effects | Every 1B unit has measured `4` and `16` records with per-shard NLL/logit/KL metrics | Pass |
| Exact representation-aware storage deltas | Marginal/profile `storage_delta_vs_uniform_w8`; verifier recomputes padding/scales/fixed FP16 cost | Pass: integer bits and bytes |
| No additive assumption | Search ranking and `final_calibration_frontier` use actual combined shard execution; margins are proposal-only | Pass |
| Deterministic interaction-aware search | `interaction_aware_search.rounds`, deterministic seeds/order, profile signatures, beam/coordinate moves | Pass |
| Feasible 8→4 and compensating 8→16 moves | `neighbor_profiles` paired move generation; search history records `paired:*->4,*->16` | Pass |
| Recompute moves around accepted profiles | Multiple search rounds and accepted anchor signatures in raw JSON | Pass |
| Explicit evaluation budget | 1B: 96 new evaluations under budget 96; verifier checks budget | Pass |
| At least three dispersed calibration shards | 1B: 3×4 train windows; metadata has randomized bucket selector, starts, intervals, and non-overlap | Pass |
| Validation unseen during search | Validation loaded after search in code and recorded; search IDs equal train IDs and disjoint from validation | Pass |
| Stable calibration selection | Shard NLL deltas, worst shard, std, improved-shard count, and declared tolerance; gate requires at least 2/3 | Pass |
| Full 3^5 projection sanity check on 1B | `projection_enumeration`: 243 total, 243 executed, 243 unique; actual combined calibration per row | Pass |
| Projection budget coverage | 112 assignments at/below W8 and 7 in the declared 1% close band; all 243 were still executed | Pass |
| Correct `3^5` enumeration description | Script, verifier, legacy optimizer strings, and report use `3^5` | Pass |
| Larger dispersed held-out set | 64 non-overlapping validation windows on 1B, 32 on gated 8B confirmation | Pass relative to prior 32/16 runs |
| Direct paired comparison to uniform W8 | Every held-out frontier row has `heldout_vs_uniform_w8` and paired metric arrays | Pass |
| NLL bootstrap intervals | `paired_nll_difference_vs_uniform_w8` for every frontier row; verifier checks count and CI | Pass |
| HellaSwag not used as small decision criterion | New artifacts contain no HellaSwag decision metric; report says it is not a gate | Pass |
| Do not run 8B initially | 1B artifact gate is `OPEN_8B_CONFIRMATION`; 8B artifact is separate `run_mode=confirmation` | Pass |
| Open 8B only after 1B gate | 1B has five eligible profiles with negative CI upper bounds and stable calibration before 8B artifact | Pass |
| 8B confirmation and final decision | 8B raw artifact: 48 combined searches, 11 held-out frontier profiles; best CIs cross zero; final gate is `NO_GO_CLOSE_STRUCTURED_WEIGHT_PRECISION` | Pass: required B decision |
| No forbidden implementation work | Both artifacts set KV/native/scheduler/router/query-oracle scope false; source diff contains no such work | Pass |
| Verifier checks procedure, not flags only | `scripts/verify_artifacts.py` recomputes storage, profile structure, shard IDs/intervals, execution counts, dedup, history, paired CIs, provenance, and gate | Pass |
| Tests | 13 repository unit tests pass; storage self-test passes; verifier passes for baseline matrices and both new artifacts | Pass |

## Actual final gate evidence

- 1B provisional gate: `OPEN_8B_CONFIRMATION`; five candidates met the declared
  negative held-out NLL CI plus stable-calibration condition.
- 8B confirmation gate: `NO_GO_CLOSE_STRUCTURED_WEIGHT_PRECISION`; no candidate
  had a held-out NLL CI upper bound below zero, despite a few negative point
  estimates.
- 8B confirmation did not run the 1B-only 243 projection sanity check or repeat
  all 160 8B margins; it used a bounded actual combined search because the
  objective scopes the exhaustive projection requirement to Llama 3.2 1B and
  the confirmation gate only requires testing the interaction-aware direction.

## Verification commands and current evidence

```bash
.venv/bin/python scripts/structured_precision_experiment.py --storage-self-test
PYTHONPATH="$PWD/vendor/swiftLLM:$PWD/vendor/swiftLLM/csrc" \
  .venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/verify_artifacts.py
```

The commands pass after the recorded experiments. The current decision is
complete under the declared proxy and gate. Native-kernel feasibility is **not**
reopened; the next direction is KV-cache precision/serving behavior.
