#!/usr/bin/env bash
# Reproduce the pinned SwiftLLM offline example. This intentionally keeps the
# upstream example and no precision profile argument.
set -euo pipefail

MODEL_PATH=${1:?"usage: $0 MODEL_PATH [GPU] [OUTPUT]"}
GPU=${2:-0}
OUTPUT=${3:-results/baseline/swiftllm_precision_noop_1b.log}

mkdir -p "$(dirname "$OUTPUT")"
PYTHONPATH="$PWD/vendor/swiftLLM:$PWD/vendor/swiftLLM/csrc${PYTHONPATH:+:$PYTHONPATH}" \
CUDA_VISIBLE_DEVICES="$GPU" \
  .venv/bin/python vendor/swiftLLM/examples/offline.py --model-path "$MODEL_PATH" 2>&1 | tee "$OUTPUT"
