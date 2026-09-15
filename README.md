# Precision-batching feasibility baseline

Evidence-first research baseline for query- and Q/K/V-aware mixed-precision continuous batching.

## Start here

- [Feasibility report](docs/feasibility-report.md) — findings, numbers, blocked questions, and go/no-go.
- [Reproducible baseline](docs/baseline.md) — environment, model paths, setup, and exact commands.
- [SwiftLLM execution map](docs/source-map.md) — request → scheduler → model → Q/K/V → KV-cache locations.
- [Claim/evidence matrix](docs/claim-evidence.md) — what is confirmed versus still uncertain.
- [QAQ and MorphServe notes](docs/papers.md) — primary sources and scope boundaries.
- [Available local assets](docs/available-assets.md) — model checkpoints and blocked AWQ asset.
- [Completion audit](docs/completion-audit.md) — structured-precision historical evidence.
- [KV page format](docs/kv-page-format.md) — page ownership, formats, quantizer, and mixed attention seam.
- [KV feasibility report](docs/kv-precision-report.md) — live-demotion measurements, quality, blockers, and decision.
- [KV completion audit](docs/kv-completion-audit.md) — requirement-by-requirement audit for this phase.

## What is included

- `vendor/swiftLLM-upstream/`: clean snapshot of `interestingLSY/swiftLLM` at commit `682cf9a28f97f7490409981a2f181528f377eb5d`.
- `vendor/swiftLLM/`: modular research fork with request-level `PrecisionProfile(q_bits, k_bits, v_bits, o_bits, ffn_bits)` metadata, an explicitly eager weight proxy, and an optional FP16/INT8/INT4 live page-store mechanism.
- `scripts/qkv_sensitivity.py`: reproducible `{4,8,16}^3` Q/K/V projection sensitivity matrix, prefill/decode paired metrics, layer sweep, and offline oracle comparison.
- `scripts/structured_precision_experiment.py`: W8-centered Q/K/V/O/FFN marginals, dispersed-shard calibration, actual combined-profile beam/coordinate search, exhaustive 1B `3^5 = 243` projection assignments, held-out paired evaluation, and exact storage ledgers.
- `scripts/swiftllm_precision_smoke.py`: verifies metadata propagation without changing default behavior.
- `scripts/kv_precision_experiment.py`: measures live FP16→INT8/INT4 page demotion, exact storage, mixed attention, CUDA overlap, and paired checkpoint quality.
- `tests/test_precision.py` and `tests/test_structured_precision.py`: metadata, eager-proxy, scheduler-semantics, and storage-accounting checks.
- `references/swiftllm-kv-page.diff`: source diff for the live-page phase.
- `results/`: raw baseline logs, environment snapshots, structured-precision results, and live-KV conversion/attention/quality evidence artifacts. Model weights are not checked in.

The structured-weight proxy returns FP16 after quantize/dequantize and is not evidence of low-bit acceleration. The current KV phase uses actual INT8/INT4 page payloads but a transparent PyTorch mixed-attention reference. It found real reclamation and near-lossless old-page INT8 quality in small probes, but no useful mixed-attention fast path; the scoped decision is no-go for opening a scheduler. Native mixed kernels, routers, schedulers, and swapping remain deferred.

## Quick check

```bash
uv venv --python python3.12 .venv
# Follow docs/baseline.md for the pinned GPU dependencies and C++ extension build (requirements-lock.txt).
PYTHONPATH="$PWD/vendor/swiftLLM:$PWD/vendor/swiftLLM/csrc" \
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/verify_artifacts.py
```

The verifier checks actual W8-centered records, combined executions, dispersed
shards, exact storage, the 243-assignment 1B enumeration, held-out pairing,
and the staged final gate. It also checks the live-KV mechanism grid, exact
page accounting, no-shadow state, overlap correctness, and checkpoint quality
traces.
