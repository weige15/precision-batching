# Claim/evidence matrix — final interaction-aware phase

A claim is supported only when the raw artifact and verifier cover the actual
procedure. The old additive-selected artifacts remain historical evidence with
their claim narrowed accordingly.

| Claim / uncertainty | Status | Evidence | Limitation / boundary |
|---|---|---|---|
| SwiftLLM source and baseline are pinned | Confirmed | `vendor/swiftLLM*/UPSTREAM_COMMIT`, baseline logs, `docs/baseline.md` | Clean-machine rebuild remains a host prerequisite |
| Default FP16/no-op and scheduler semantics remain unchanged | Confirmed for tested path | Existing tests, baseline/no-op comparison, unchanged scheduler scope | No online mixed-profile serving trace |
| Prior structured negative result was interaction-aware | **Corrected / refuted** | `docs/completion-audit.md`; old policies were selected from FP16-background additive risks | Old negative claim is only about those additive-selected candidates |
| Every 1B Q/K/V/O/FFN unit has W8-centered W4/FP16 marginals | Confirmed | `results/sensitivity/llama32_1b_interaction_aware.json:w8_centered_marginals` | Fake-quant proxy only |
| W8-centered margins include actual paired NLL/logit/KL effects | Confirmed | Per-shard and per-sample `calibration` records for 80×(W4,W16) profiles | Margins propose moves; they are not final objective |
| Exact modeled weight storage is represented | Confirmed | Unit shape ledger, profile deltas, verifier recomputation | Native packed headers/alignment and KV cache are not modeled |
| Calibration uses dispersed, non-overlapping random windows | Confirmed | Three train shards with deterministic seeds, starts, intervals, and verifier overlap check | Wikitext language proxy, not broad task evaluation |
| Validation is unseen during search | Confirmed | Validation loaded after search; train/validation IDs disjoint; verifier checks search IDs | Held-out sample size remains finite |
| Additive single-unit risk is a valid final objective | **Refuted for this proxy/search** | Actual combined-profile calibration ranking and previous interaction evidence | No universal statement about other quantizers |
| Interaction-aware search was executed | Confirmed | `interaction_aware_search`: beam/coordinate rounds, paired moves, accepted anchors, 96-evaluation budget; verifier checks history and actual combined records | Bounded, not global search |
| Marginals are only proposal inputs | Confirmed | `marginals_only_propose_moves=true`, actual-shard ranking, no additive objective field used for final selection | Proposal width bounds reachable neighborhoods |
| All 243 projection assignments were actually executed on 1B | Confirmed | `projection_enumeration`: 243/243, 243 unique, each with combined calibration metrics | Requirement scoped to 1B exploration; 8B confirmation skips this sanity check |
| Projection-only optimizer text is correct | Confirmed | Script/report/JSON/verifier say the corrected `3^5` expression |
| A 1B structured Pareto signal exists in the proxy | Provisional only | Five 1B profiles have negative held-out paired NLL CIs and stable calibration | It required 8B confirmation; not a final claim |
| 1B provisional gate opened 8B only after positive signal | Confirmed | 1B `final_gate=OPEN_8B_CONFIRMATION`; separate 8B `run_mode=confirmation` artifact | Confirmation uses smaller calibration set |
| 8B confirms a repeatable advantage | **Not demonstrated** | `llama31_8b_interaction_aware.json`: 11 frontier profiles; best NLL CIs cross zero | 8B held-out set is 32 windows; no positive-confidence candidate |
| Structured weight precision has a repeatable Pareto advantage | **No-go / closed** | 1B provisional signal failed 8B repeatability gate; final decision `NO_GO_CLOSE_STRUCTURED_WEIGHT_PRECISION` | Only this symmetric fake-quant proxy and bounded search are closed |
| HellaSwag supports the decision | Not used | New artifacts intentionally omit it as a criterion | Avoid small-subset decision making |
| Native W4/W8 kernels should be implemented now | **No-go** | Final gate did not confirm a structured advantage | Reopen only after a future positive evidence phase |
| Query routing or precision-aware batching should be implemented | **No-go / paused** | No multiple globally competitive profiles after 8B confirmation; scope flags false | Scheduler, KV quantization, swapping remain untouched |
| Next direction should be KV-cache precision/serving behavior | Recommended | Final report and final gate next-direction field | Separate experiment required |
