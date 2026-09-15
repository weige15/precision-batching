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
| Next direction should be KV-cache precision/serving behavior | Historical recommendation | Earlier final report | Superseded by the matched action study below |

## Optimized INT8 and memory-pressure action phase

| Claim / uncertainty | Status | Evidence | Limitation / boundary |
|---|---|---|---|
| Optimized mixed FP16/INT8 attention exists without materializing pages to FP16 | Confirmed for the enabling path | `vendor/swiftLLM/swiftllm/worker/kernels/segmented_paged_attn.py`, `results/sensitivity/kv_break_even_study.json` | Compact copied arena and host segment construction; not a production scheduler |
| Optimized path matches the explicit page oracle | Confirmed for measured cells | 54-cell artifact, max absolute error `0.00048828125`; `page_attention_for_layer` remains oracle | RTX 3090 and tested Llama shapes only |
| Conversion is a batched GPU data-path operation | Confirmed | `PagedKVCache.demote_pages_batch`, `promote_pages_batch`, optimized quality artifacts, verifier | Python metadata publication and transient workspace remain |
| Existing CPU/GPU swap was measured under matched all-layer bytes | Confirmed | `swiftllm_c.swap_blocks`, final artifact `cpu_offload` records | Preallocated dense cache reports reusable capacity, not physical free |
| Queue, offload, and INT8 action costs are explicitly separated into transition and persistent terms | Confirmed | `scripts/kv_break_even_experiment.py`, verifier recomputes horizon costs | Queue is a predeclared duration model, not an online controller |
| INT8 has a non-dominated break-even region | **Refuted for this study** | 0/324 global wins; INT8 below queue in 0/324 rows; `kv-break-even-report.md` | Scoped to this optimized path, format, GPU, and grid |
| Queue is globally lowest cost | Confirmed for measured map | 284/324 horizon rows; remaining 40 are CPU-offload wins | Not a universal workload policy |
| Old-page INT8 quality is acceptable | Supported but uncertain | Two short paired checkpoint probes, 100% top-1 in tested rows | One short probe per checkpoint; quality is not a positive runtime gate |
| Later memory-pressure scheduler should open | **No-go** | Strict gate fails steady-state overhead and non-dominated-region criteria | Lossless actions only for next direction |
| INT8 belongs in runtime action space | **No-go / removed** | Final report, completion audit, unchanged scheduler, no policy integration | Research enabling code is retained as auditable evidence; no serving policy selects it |

## Live KV-page precision phase

| Claim / uncertainty | Status | Evidence | Limitation / boundary |
|---|---|---|---|
| SwiftLLM page abstraction supports FP16/INT8/INT4 | Confirmed | `vendor/swiftLLM/swiftllm/worker/kv_cache.py`, `docs/kv-page-format.md`, `tests/test_kv_cache.py` | One-tensor-per-page research allocator |
| K/V format metadata is explicit at page granularity | Confirmed | `PagedKVCache.metadata_codes`, `KVPage.k_format/v_format`, verifier | K/V share a logical page but can differ in format |
| Live demotion reclaims real memory without a steady-state FP16 shadow | Confirmed after synchronization | `kv_precision_mechanism.json` exact logical/`torch.cuda.memory_allocated` fields; verifier | Transient source is retained during async conversion until event completion |
| Mixed-format attention is numerically correct | Confirmed (historical oracle) | Explicit reconstruction test and model-backed quality traces | Earlier phase used a PyTorch reference; optimized follow-up is covered below |
| Conversion is fast enough by itself | Historical supported | 0.326--0.495 ms/page medians and 47--94 MiB/s synthetic rates | Later batched measurements supersede per-page transition timing |
| Conversion overlaps useful GPU work | Not demonstrated | CUDA event overlap traces are zero-error but hidden fraction is zero | Test uses concurrent matmuls, not a full serving workload |
| Old-page INT8 quality is near-lossless | Supported but uncertain | 1B/8B paired forced-prefix traces, 100% top-1 for tested old-page fractions | One prompt/checkpoint probe per model and short generation |
| INT4 quality is safe | No-go in tested proxy | Static/recent variants show NLL increases and top-1 mismatches | Other published outlier/attention-aware quantizers could differ |
| Live demotion should drive a scheduler now | Historical no-go | Earlier reference path was about 3.7--6.9x slower at 50% compression; follow-up re-tested optimized path | Final matched action gate is covered below |
| MorphServe was outperformed | Not claimed | Report records MorphServe KVResizer boundary and no equivalent run | MorphServe resizes capacity rather than quantizing live KV pages |
