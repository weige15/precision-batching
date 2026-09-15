# Reproducible baseline

## Scope and current status

This repository contains a pinned SwiftLLM checkout plus a small research fork. The baseline target is Llama 3.2 1B Instruct because its complete FP16/BF16-compatible checkpoint is already available locally and fits on one 24-GB GPU. A second, independent sensitivity run uses the available Llama 3.1 8B checkpoint.

The baseline does **not** use the local MorphServe AWQ W4 checkpoint: SwiftLLM at the pinned commit expects ordinary Llama projection tensors, while the AWQ checkpoint contains quantized packed weight tensors and cannot be loaded by the unmodified loader without an additional AWQ implementation. The AWQ checkpoint is recorded as available but blocked for this baseline.

## Pinned sources and artifacts

To reconstruct the clean upstream source independently:

```bash
git clone https://github.com/interestingLSY/swiftLLM /tmp/swiftLLM
git -C /tmp/swiftLLM checkout --detach 682cf9a28f97f7490409981a2f181528f377eb5d
```

- SwiftLLM upstream: `https://github.com/interestingLSY/swiftLLM`
- SwiftLLM commit: `682cf9a28f97f7490409981a2f181528f377eb5d`
- Clean snapshot: `vendor/swiftLLM-upstream/`
- Research fork: `vendor/swiftLLM/`
- Research diff: `references/swiftllm-research.diff`
- QAQ source/code notes: [`docs/papers.md`](papers.md), `vendor/qaq/`
- Paper PDFs: `references/qaq-2403.04643.pdf` and `references/morphserve-2506.02006-v2.pdf`

## Environment captured

`results/baseline/environment_snapshot.json`, `results/baseline/pip-freeze.txt`, `results/baseline/toolchain.txt`, and `results/baseline/invocations.json` are generated artifacts from the run. The important captured values are:

- Linux x86_64, Python 3.12.3
- 7 × NVIDIA GeForce RTX 3090, 24,576 MiB each, driver 580.159.03, compute capability 8.6
- PyTorch 2.4.0+cu121, CUDA runtime 12.1, Triton 3.0.0
- `transformers` 4.51.3, `safetensors` 0.8.0, `datasets` 3.6.0, `pyarrow` 25.0.1, `vllm-flash-attn` 2.6.2, Ray 2.58.0
- CUDA toolkit selected by the C++ extension build: `/usr/local/cuda-12.4`
- Primary model: local Llama 3.2 1B Instruct snapshot `9213176726f574b556790deb65791e0c5aa438b6`, with per-file SHA-256 hashes in the snapshot and sensitivity JSON.
- Secondary model: local Llama 3.1 8B snapshot `d04e592bb4f6aa9cfee91e2e20afa771667e1d4b`.

GPU processes from other work occupied approximately 8–11 GB on several devices during discovery; the recorded sensitivity runs used otherwise free GPU 5 (1B) and GPU 6 (8B). Exact device visibility is set by each command. The clean baseline logs below were rerun on free GPU 5 so the profiler's available-memory calculation is not polluted by another process.

## Setup from a fresh checkout

```bash
uv venv --python python3.12 .venv
uv pip install --python .venv/bin/python -r requirements-lock.txt
uv pip install --python .venv/bin/python -e vendor/swiftLLM
(cd vendor/swiftLLM-upstream/csrc && ../../../.venv/bin/python setup.py build_ext --inplace)
(cd vendor/swiftLLM/csrc && ../../../.venv/bin/python setup.py build_ext --inplace)
```

The exact installed environment used here is in `pip-freeze.txt` and the checked-in `requirements-lock.txt` (the latter omits only the editable local SwiftLLM install). The CUDA compiler/toolkit and driver observed in `results/baseline/toolchain.txt` are CUDA 12.4.99 and driver 580.159.03. The C++ extension is required by SwiftLLM imports and is not a low-bit kernel. The lock file pins Python packages, but CUDA toolkit availability remains a host prerequisite.

## Untouched SwiftLLM baseline

The upstream command, run with the original `vendor/swiftLLM-upstream/examples/offline.py`, was:

```bash
MODEL=/nfs/home/s314511048/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6
PYTHONPATH="$PWD/vendor/swiftLLM-upstream:$PWD/vendor/swiftLLM-upstream/csrc" \
CUDA_VISIBLE_DEVICES=5 \
.venv/bin/python vendor/swiftLLM-upstream/examples/offline.py --model-path "$MODEL"
```

The checked-in output is `results/baseline/swiftllm_unmodified_1b.log`. On a free RTX 3090 it reports 37,975 GPU KV blocks, a 4.78-GB profiled runtime peak, and the four upstream prompts' greedy token text. The research-fork no-op run is `results/baseline/swiftllm_precision_noop_1b.log`; after removing the non-deterministic model creation-time line, the complete outputs are byte-for-byte identical. Earlier discovery runs with another process on the GPU reported a polluted 12.99-GB value and are not used as the clean baseline.

The convenience wrapper `scripts/run_swiftllm_baseline.sh` runs the same upstream example against the research checkout and writes a log. To reproduce the truly untouched copy, replace `vendor/swiftLLM` with `vendor/swiftLLM-upstream` in the command as shown above. All matrix and layer-sweep invocations, including their physical GPU masks and prompt counts, are recorded in `results/baseline/invocations.json`.

## Research-fork checks

```bash
PYTHONPATH="$PWD/vendor/swiftLLM:$PWD/vendor/swiftLLM/csrc" \
.venv/bin/python -m unittest discover -s tests -v

MODEL=...
PYTHONPATH="$PWD/vendor/swiftLLM:$PWD/vendor/swiftLLM/csrc" \
CUDA_VISIBLE_DEVICES=5 \
.venv/bin/python scripts/swiftllm_precision_smoke.py \
  --model-path "$MODEL" \
  --output results/baseline/swiftllm_precision_metadata_smoke.json
```

The smoke output verifies exact repeated all-FP16 output, successful mixed-profile propagation through Q/K/V/O/FFN call sites, and a changed output for the opted-in eager proxy. It is not a serving-performance result. The final artifact-gate output is preserved in `results/baseline/artifact-verification.log`.

## Structured model-level experiment

The model-level experiment is intentionally separate from the scheduler. It uses Wikitext-2 train as calibration and validation as held-out text, plus a deterministic HellaSwag validation subset for teacher-forced multiple-choice accuracy. It measures 5 units per layer (Q/K/V/O/FFN), includes fixed FP16 parameters in its representation-aware storage ledger, and records exact profile assignments and bootstrap intervals:

```bash
MODEL=/path/to/local/llama-checkpoint
CUDA_VISIBLE_DEVICES=5 .venv/bin/python scripts/structured_precision_experiment.py \
  --model-path "$MODEL" \
  --output results/sensitivity/structured.json \
  --device cuda:0 --calibration-samples 16 --heldout-samples 32 \
  --task-samples 32 --seq-len 128 --batch-size 2 --bootstrap-iterations 1000
```

Use the checked-in JSONs and `docs/feasibility-report.md` for the exact 1B/8B counts. `scripts/verify_artifacts.py` independently recomputes every profile's bit ledger and checks calibration/held-out separation. The experiment is a FP16-after-dequantization numerical proxy; it does not measure native kernel speed or KV-cache compression.
