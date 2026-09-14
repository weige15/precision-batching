# Precision-batching feasibility baseline

Evidence-first research baseline for query- and Q/K/V-aware mixed-precision continuous batching.

## Start here

- [Feasibility report](docs/feasibility-report.md) — findings, numbers, blocked questions, and go/no-go.
- [Reproducible baseline](docs/baseline.md) — environment, model paths, setup, and exact commands.
- [SwiftLLM execution map](docs/source-map.md) — request → scheduler → model → Q/K/V → KV-cache locations.
- [Claim/evidence matrix](docs/claim-evidence.md) — what is confirmed versus still uncertain.
- [QAQ and MorphServe notes](docs/papers.md) — primary sources and scope boundaries.
- [Available local assets](docs/available-assets.md) — model checkpoints and blocked AWQ asset.
- [Completion audit](docs/completion-audit.md) — requirement-by-requirement evidence and uncovered work.

## What is included

- `vendor/swiftLLM-upstream/`: clean snapshot of `interestingLSY/swiftLLM` at commit `682cf9a28f97f7490409981a2f181528f377eb5d`.
- `vendor/swiftLLM/`: modular research fork with request-level `PrecisionProfile(q_bits, k_bits, v_bits, o_bits, ffn_bits)` metadata and an explicitly eager fake-quantization proxy.
- `scripts/qkv_sensitivity.py`: reproducible `{4,8,16}^3` Q/K/V projection sensitivity matrix, prefill/decode paired metrics, layer sweep, and offline oracle comparison.
- `scripts/swiftllm_precision_smoke.py`: verifies metadata propagation without changing default behavior.
- `tests/test_precision.py`: metadata, eager-proxy, and scheduler-semantics checks.
- `results/`: raw baseline logs, environment snapshot, and sensitivity JSON outputs. Model weights are not checked in.

The proxy returns FP16 after quantize/dequantize and uses ordinary PyTorch operations. Its timings are overhead measurements only; they are not evidence of low-bit acceleration. A full precision-aware scheduler, KV-cache quantization, CPU/GPU bit-plane swapping, and native low-bit kernels are intentionally deferred.

## Quick check

```bash
uv venv --python python3.12 .venv
# Follow docs/baseline.md for the pinned GPU dependencies and C++ extension build (requirements-lock.txt).
PYTHONPATH="$PWD/vendor/swiftLLM:$PWD/vendor/swiftLLM/csrc" \
.venv/bin/python -m unittest discover -s tests -v
```
