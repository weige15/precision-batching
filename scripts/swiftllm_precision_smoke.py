#!/usr/bin/env python3
"""Exercise the SwiftLLM request-profile plumbing without a KV-cache run."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SWIFT_ROOT = ROOT / "vendor" / "swiftLLM"
C_SRC = SWIFT_ROOT / "csrc"
for path in (C_SRC, SWIFT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from swiftllm import EngineConfig, LlamaModel, PrecisionProfile  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--use-dummy", action="store_true")
    args = parser.parse_args()

    config = EngineConfig(
        model_path=args.model_path,
        use_dummy=args.use_dummy,
        block_size=16,
        gpu_mem_utilization=0.5,
        num_cpu_blocks=0,
        max_seqs_in_block_table=4,
        max_blocks_per_seq=16,
        max_batch_size=2,
        max_tokens_in_batch=32,
    )
    model = LlamaModel(config)
    model.load_weights()
    input_ids = [[128000, 128001, 123, 456, 789], [111, 222, 333]]
    profiles = [
        PrecisionProfile(),
        PrecisionProfile(q_bits=4, k_bits=8, v_bits=4, o_bits=8, ffn_bits=4),
    ]

    legacy = model.forward(input_ids, [0, 1], [], ignore_kvcache=True)
    legacy_repeat = model.forward(input_ids, [0, 1], [], ignore_kvcache=True)
    proxy = model.forward(
        input_ids,
        [0, 1],
        [],
        ignore_kvcache=True,
        precision_profiles=profiles,
    )
    result = {
        "model_path": str(Path(args.model_path).resolve()),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "legacy_output": legacy,
        "legacy_repeat_output": legacy_repeat,
        "legacy_exact_repeat": legacy == legacy_repeat,
        "mixed_profile_output": proxy,
        "mixed_profile_differs": proxy != legacy,
        "profiles": [profile.as_dict() for profile in profiles],
        "note": "The mixed result uses SwiftLLM's eager quantize/dequantize proxy; it is not a native low-bit benchmark.",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
