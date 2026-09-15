# Completion audit: GPU-memory reclamation break-even objective

This audit checks the active user objective against the current repository,
raw artifacts, commands, and verifier. It does not treat the older live-KV
no-go report or a green test suite as proof of the new optimized-path study.

## Objective restatement

Deliver an evidence-backed comparison of queueing, existing lossless FP16
CPU/GPU KV offload, and request-local INT8 KV demotion under matched GPU-memory
deficits. First replace the old materialize-to-FP16 mixed-attention reference
with the smallest credible optimized INT8 path, batch page conversion, measure
RTX 3090 behavior across serving dimensions, model one-time versus persistent
cost, and gate any later scheduler. If INT8 remains dominated, remove it from
the runtime action space and retain only lossless actions. Do not implement a
controller, router, MorphServe weight swapping, or INT4 policy.

## Prompt-to-artifact checklist

| Requirement / deliverable | Concrete evidence inspected | Result |
|---|---|---|
| Preserve completed weight/KV context | `docs/feasibility-report.md`, `docs/kv-precision-report.md`, original sensitivity JSONs | Pass; weight result remains closed and old KV reference evidence is preserved as historical context |
| Use pinned SwiftLLM | `vendor/swiftLLM/UPSTREAM_COMMIT`, `vendor/swiftLLM-upstream/UPSTREAM_COMMIT`, final JSON provenance | Pass: `682cf9a28f97f7490409981a2f181528f377eb5d` |
| Use RTX 3090 evidence | final JSON `provenance.device_name`, captured `docs/baseline.md` hardware/toolchain | Pass: final run on `CUDA_VISIBLE_DEVICES=3`, RTX 3090 |
| Inspect/reuse available low-bit implementations | adjacent FlashInfer symbol inspection; `vendor/qaq`; existing SwiftLLM Triton kernel | Pass: no local integer-INT8 paged attention kernel was available; existing dense Triton path and local page quantizer were reused; FlashInfer FP8 is not counted as INT8 |
| Replace materialize-to-FP16 mixed reference | `vendor/swiftLLM/swiftllm/worker/kernels/segmented_paged_attn.py`; `PagedKVCache.optimized_attention`; `transformer_layer.py` routing | Pass: format-specialized FP16/INT8 segment kernels read native payloads and reduce online softmax; old `page_attention_for_layer` remains oracle |
| INT8 only for compressed decision | optimized kernel, break-even scope, final report | Pass: symmetric integer INT8 with group-128 FP16 scales; INT4 is only retained in the old oracle/control path |
| Do not claim kernel engineering as contribution | optimized kernel docstring and final report | Pass: explicitly enabling infrastructure and scoped no-go |
| Mixed FP16/INT8 pages | segmented kernel format-specialized launches and segment construction | Pass; K/V must match for optimized path, unsupported formats deliberately use the oracle |
| GQA and block tables | segmented kernel `num_my_heads`, block-table indexing; 54-cell smoke/study | Pass |
| Preserve correctness oracle | `page_attention_for_layer`; `tests/test_kv_cache.py`; per-cell oracle errors | Pass |
| Batched conversion, not hundreds of calls | `PagedKVCache.demote_pages_batch`, `BatchConversionResult`, quality JSON `conversion.batched`, break-even `batched_gpu_operation` | Pass; device reduction/scale/quantization is one batched operation per transition |
| Measure conversion as GPU data path | CUDA event timings, `temporary_bytes`, raw conversion fields in final JSON and quality JSONs | Pass; Python metadata publication is separate; allocator/workspace limitations are recorded |
| Measure reversal/restoration | `promote_pages_batch`, unit test, final break-even `restoration` records | Pass |
| Match memory deficits | each final cell uses one exact target deficit; compression reclaims it and offload uses the smallest dense-page count meeting it, with any granularity overshoot recorded | Pass |
| Queue/not-admit action | final cell `quality_delta.queue=0`, `capacity_bytes_avoided_or_reclaimed`, horizon queue model | Pass as an explicit no-transition analytical action; no scheduler rerun was needed after the gate |
| Existing CPU/GPU swap action | `swiftllm_c.swap_blocks`, `vendor/swiftLLM/csrc/src/block_swapping.cpp`, final `cpu_offload` rows | Pass; actual all-layer FP16 transfers and both directions measured |
| Actual HBM versus reusable capacity distinguished | `cpu_offload.physical_hbm_delta_bytes`, `allocator_is_preallocated`, report | Pass; physical delta is reported as zero for preallocated SwiftLLM cache, reusable capacity is separately exact |
| INT8 demotion action | final `conversion`, `optimized_mixed_int8`, `restoration` rows and checkpoint probes | Pass |
| Every action's one-time transition latency | queue zero, offload swap-out/in, compression demote/restore fields | Pass |
| Every action's subsequent per-token latency | dense baseline and optimized mixed per-token fields plus explicit cost decomposition | Pass |
| Every action's quality change | queue/offload exact zero; paired old-page INT8 NLL/KL/top-1 in `quality_probes` and two checkpoint artifacts | Pass as a controlled, short quality probe; broad quality remains uncertain |
| Context variation | final grid contexts `128,512,2048` | Pass |
| Batch variation | final grid batches `1,4,8` | Pass |
| Reclaimed amount variation | selected old pages `25%,50%,75%`, exact all-layer MiB | Pass |
| Pressure duration variation | horizons `1,4,8,16,32,64` decode iterations | Pass |
| Transition/persistent distinction | final report formula and verifier recomputation of all horizon costs | Pass |
| Empirical break-even map | 54-cell × 6-horizon raw rows, report winner counts and stress table | Pass |
| Request length/decode horizon effect | context grid and horizon map; report notes no INT8 winner | Pass; request arrival waiting is modeled rather than rerun online |
| Strict scheduler gate | report gate section; no scheduler files changed; verifier scope flags | Pass: scheduler not opened |
| INT8 dominated decision | final artifact: compression global winner `0/324`; compression below queue `0/324`; report | Pass: no runtime action approval |
| Remove KV precision from runtime action space | `vendor/swiftLLM/swiftllm/engine_config.py` exposes only `dense_fp16` on the runtime CLI; final report decision and unchanged scheduler; no precision-aware policy/router introduced | Pass for runtime policy/action space; direct research constructors remain archived for audit/reproduction |
| No full controller/router | `git diff`, `scheduler.py` unchanged, scope fields | Pass |
| No MorphServe weight swapping | source diff and report scope | Pass |
| No INT4 policy | no new INT4 optimized path or scheduler; transformer uses oracle fallback only | Pass |
| Source/checkpoint provenance | final JSON source hash, upstream pin, quality artifact checkpoint paths/hashes inherited from existing quality procedure | Pass |
| Verification is procedure-aware | `verify_break_even` checks dimensions, exact bytes, batched flags, oracle error, swap mechanism, and recomputed costs; `verify_kv_quality` checks optimized provenance and batching | Pass |
| Required report categories | `docs/kv-break-even-report.md`: confirmed, supported but uncertain, blocked questions, remaining uncertainty | Pass |
| Required final state statement | report Decision and Gate outcome explicitly state INT8 is not retained in runtime action space | Pass |
| Tests | 23 repository unit tests; syntax checks; optimized smoke and two checkpoint probes | Pass |
| Final verifier | `.venv/bin/python scripts/verify_artifacts.py` | Pass |

## Final evidence summary

- Raw matched study: `results/sensitivity/kv_break_even_study.json`;
  54 cells, 324 horizon rows per model family, current source fingerprint,
  exact all-layer bytes, conversion/restoration, swap, attention, quality, and
  winner records.
- Optimized kernel smoke: `results/sensitivity/kv_precision_optimized_smoke_final.json`;
  both model shapes, max observed oracle error `0.00048828125`.
- Actual checkpoint batched quality probes:
  `results/sensitivity/kv_precision_quality_batched_1b.json` and
  `results/sensitivity/kv_precision_quality_batched_8b.json`.
- Final verifier: passed, including preserved legacy artifact provenance,
  current optimized quality provenance, and all break-even arithmetic; output
  is preserved in `results/baseline/artifact-verification-break-even.log`.
- Final unit-test and syntax outputs are preserved in
  `results/baseline/unit-tests-break-even.log` and
  `results/baseline/pycompile-break-even.log`.
- Final study outcome: queue is globally best in 284/324 horizon rows, CPU
  offload in 40/324, and INT8 in 0/324. (The split is 132/30 for 1B and
  152/10 for 8B.) INT8 is below offload in some high-deficit cells but never
  below queue in the same cells.

## Known limits that do not support a positive INT8 decision

The optimized kernel is a smallest credible enabling path, not a production
kernel: it uses compact copied arenas and host segment construction, and the
one-layer attention datapath is paired with exact all-layer byte accounting.
The queue term is a predeclared horizon model, not a new three-action online
arrival trace. Quality is two short forced-prefix checkpoint probes. These
limits are recorded rather than hidden; they make the result a scoped no-go,
not a universal impossibility claim.
