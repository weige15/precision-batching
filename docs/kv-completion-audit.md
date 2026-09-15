# Completion audit: live KV-page precision objective

This audit checks the user-provided live-KV objective against the repository's
current files, raw artifacts, commands, and verifier. It does not treat a green
test or a manifest as proof for requirements that the test does not cover.

## Concrete deliverables / success criteria

1. Preserve the closed structured-weight-precision result and do not reopen
   weight allocation.
2. Establish an explicit SwiftLLM page-format abstraction with FP16 plus
   conservative/aggressive compressed formats, standard documented quantization,
   and page-granular metadata.
3. Convert already-populated live pages from higher to lower precision, execute
   mixed-format attention, and avoid scheduler automation.
4. Measure conversion latency/reclaimed MiB, reclamation rate, workspace/peak
   memory, homogeneous/mixed attention latency and bandwidth, overlap safety,
   and batch/context/compressed-fraction variation.
5. Evaluate recent/old pages, layer ranges, K-only/V-only, and multiple page
   fractions on actual 1B and 8B checkpoints with generation-relevant quality.
6. Compare unchanged FP16, static compressed pages with the same quantizer,
   queue/refuse capacity, and MorphServe without claiming an unrun advantage.
7. Verify numerical correctness against explicit reference, exact allocated and
   reclaimed bytes, no dense shadow, raw traces, and source/checkpoint provenance.
8. End with confirmed, uncertain, blocked, and remaining-uncertainty findings,
   plus a scoped go/no-go decision.

## Prompt-to-artifact checklist

| Requirement | Concrete evidence inspected | Result |
|---|---|---|
| Closed structured weight result preserved | `docs/feasibility-report.md`, `docs/completion-audit.md`, old `results/sensitivity/llama*_structured.json` and `*_interaction_aware.json` | Pass; current report explicitly closes weights and does not modify those artifacts |
| SwiftLLM pinned | `vendor/swiftLLM/UPSTREAM_COMMIT`, `vendor/swiftLLM-upstream/UPSTREAM_COMMIT`, `results/sensitivity/kv_precision_mechanism.json:provenance` | Pass: `682cf9a28f97f7490409981a2f181528f377eb5d`; current source manifest is hash-bound |
| FP16 + conservative/aggressive formats | `vendor/swiftLLM/swiftllm/worker/kv_cache.py`, `docs/kv-page-format.md` | Pass: FP16, INT8, packed INT4 |
| Standard documented quantizer, not novelty | `kv_cache.py:_encode_tensor`, `docs/kv-page-format.md`, `docs/papers.md` | Pass: symmetric max-abs per-group, group 128, FP16 scales; no novelty claim |
| Explicit page metadata | `PagedKVCache.metadata_codes`, `KVPage.k_format/v_format`, unit tests | Pass: `[block,layer,K/V]` codes; K/V independent |
| Live higher→lower conversion | `PagedKVCache.demote_page`, model hook `LlamaModel.demote_kv_pages`, raw `quality_*.json:demotions` | Pass: actual populated pages converted; 256-page traces on each checkpoint |
| Mixed-format attention | `page_attention_for_layer`, `tests/test_kv_cache.py:test_mixed_format_attention_matches_explicit_page_reference`, mechanism grid | Pass; intentionally reference/PyTorch fallback, not production kernel |
| No automatic scheduler choice | `EngineConfig.kv_page_format`, model path, unchanged `scheduler.py`, scope flags in raw JSON | Pass; no scheduler/router/policy implementation |
| (1) Conversion latency per page/MiB | `kv_precision_mechanism.json:conversion`, raw page conversion arrays; report tables | Pass; 180 trials and exact per-page records |
| (2) Memory reclamation rate | `reclaimed_bytes`, `reclaimed_mib`, `reclamation_mib_per_s`, actual allocator fields | Pass; logical and `torch.cuda.memory_allocated` readings both present |
| (3) Workspace and peak memory | `peak_conversion_overhead_bytes`, `temporary_bytes_logical`, pending-source fields | Pass for measured first-page GPU peak and exact pending state; allocator fragmentation remains uncertain |
| (4) Homogeneous/mixed attention latency and bandwidth | `families[*].trials[*].fp16/static/mixed.attention`, 3 repetitions × 3 iterations | Pass; batch `{1,4,8}`, context `{128,512,1024}`, fractions `{0,.25,.5,.75,1}`; stored-byte bandwidth is explicitly a proxy |
| (5) Overlap without state corruption | `overlap` rows, CUDA event path, pending-before/after consumer read, post-sync no-shadow, `max_error_after_event_ordering=0` | Pass for controlled concurrent matmul and consumer-side stream wait; useful overlap not demonstrated |
| (6) Cost variation dimensions | Full mechanism grid in raw JSON | Pass for synthetic page mechanism; end-to-end quality timing is noisier and only one context per checkpoint |
| Recent vs old | `quality_1b/8b.json` names and per-step traces at 25/50/75% INT8; recent/old INT4 at 50% | Pass |
| Layer ranges | `dynamic_old_int8_50_early`, `..._late` on both checkpoints | Pass |
| K vs V | `..._k_only`, `..._v_only` on both checkpoints | Pass |
| Several quality fractions | Actual checkpoint variants at INT8 25/50/75%, static at 100%; mechanism grid includes 0/25/50/75/100 | Pass, with controlled single-prompt limitation |
| Long-context/generation quality | 1B 512 context/16 forced positions; 8B 256/12; per-step NLL delta/KL/RMSE/top-1 and tokens; `page_fp16_control` | Pass as a quantization-isolated controlled proxy; broad task quality is not claimed |
| Unchanged FP16 comparison | `quality_*.json:baseline.kind=unchanged_fp16_dense`; current no-op log vs upstream log; verifier | Pass |
| Static compressed same quantizer | `static_int8`, `static_int4` variants use `PagedKVCache(default_format=target)` | Pass |
| Queue/refuse comparison | Raw `comparisons.queue_or_refuse_capacity`, report | Pass analytically: zero reclaim/zero quality change; no scheduler run by scope |
| MorphServe positioning | `docs/papers.md`, raw `comparisons.morphserve`, report | Pass; boundary stated and no superiority claim |
| Weight vs KV separation | report, `scope`, unchanged structured artifacts | Pass |
| Explicit reference correctness | unit mixed-attention test independently reconstructs dequantized K/V and attention; all-FP16 page control versus dense baseline | Pass; verifier checks test/artifact coverage |
| Exact bytes and no FP16 shadow | page byte formulas, actual allocator readings, `has_unreclaimed_shadow`, `compressed_payloads_are_not_fp16`, verifier | Pass after sync; source is intentionally retained only while async conversion is pending |
| Raw traces preserved | six JSON/log files under `results/sensitivity/`, including per-page conversions, repeated attention timings, overlap state, and per-step quality | Pass |
| Relevant published designs | `docs/papers.md`: KIVI, KVQuant, QAQ, MorphServe with primary URLs and boundaries | Pass as reference, not reproduced claims |
| Forbidden work absent | `references/swiftllm-kv-page.diff`, `scheduler.py` unchanged, scope flags, report blocked section | Pass for scheduler/router/weight swapping/CPU hierarchy; no native mixed kernel was attempted by design |
| Final decision categories | `docs/kv-precision-report.md` sections Confirmed, Supported but uncertain, Blocked questions, Remaining uncertainty | Pass: no-go for current mechanism |

## Verification performed

The following commands were run after the current implementation and artifacts
were present:

```bash
.venv/bin/python scripts/verify_artifacts.py
PYTHONPATH="$PWD/vendor/swiftLLM:$PWD/vendor/swiftLLM/csrc" \
  .venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m py_compile \
  scripts/kv_precision_experiment.py scripts/verify_artifacts.py \
  vendor/swiftLLM/swiftllm/worker/kv_cache.py \
  vendor/swiftLLM/swiftllm/worker/model.py
```

The live-KV artifacts also record the exact source-file manifest and SHA-256
fingerprint used by the experiment; the verifier recomputes that fingerprint
before accepting the artifacts. The current dense no-op comparison is retained
in `results/baseline/noop-output-comparison-kv-phase.txt` and
`results/baseline/swiftllm_precision_noop_kv_phase.log`.

Observed results:

- artifact verifier passed, including historical structured-gate provenance,
  current KV source-manifest binding, exact page storage arithmetic, conversion
  coverage, 3×3 attention repeats, consumer-side overlap correctness, page-FP16
  quality control, quality policy coverage, and checkpoint paths;
- 19 repository tests passed;
- Python compilation passed;
- current dense FP16 SwiftLLM output lines matched the preserved upstream
  baseline after ignoring model-creation timing lines.

## Gaps that are deliberately not completion blockers for this phase

The report's no-go is explicitly scoped because these questions remain open:

- a packed arena and fused format-aware Triton/CUDA attention kernel;
- large continuous-batched allocator fragmentation and concurrent request
  arrival traces;
- broad task/perplexity evaluation beyond the two paired checkpoint probes;
- scheduler, priority, CPU/GPU swap, router, and automatic policy work.

Those gaps prevent a claim of production readiness or a positive go decision;
they do not invalidate the requested feasibility result. The current evidence
supports stopping before scheduler work: conversion and INT8 quality are
plausible, but the implemented mixed attention path is too expensive and
cannot demonstrate useful overlap.
