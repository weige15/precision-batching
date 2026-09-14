#!/usr/bin/env python3
"""Capture reproducibility metadata for a local experiment."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path


def command_output(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    packages = {}
    for name in ("torch", "transformers", "safetensors", "triton", "vllm-flash-attn", "ray", "fastapi", "uvicorn"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None

    model = None
    if args.model_path:
        files = []
        for path in sorted(args.model_path.iterdir()):
            if path.is_file() and (
                path.name.endswith((".safetensors", ".bin"))
                or path.name in {"config.json", "tokenizer.json", "tokenizer_config.json"}
            ):
                files.append({"name": path.name, "bytes": path.stat().st_size, "sha256": digest(path)})
        model = {"path": str(args.model_path.resolve()), "files": files}
        config = args.model_path / "config.json"
        if config.exists():
            model["config"] = json.loads(config.read_text())

    research_diff = Path("references/swiftllm-research.diff")
    snapshot = {
        "python": sys.version,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "cwd": os.getcwd(),
        "packages": packages,
        "torch_cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_smi": command_output([
            "nvidia-smi", "--query-gpu=index,name,memory.total,driver_version,compute_cap",
            "--format=csv,noheader",
        ]),
        "torch_details": command_output([
            sys.executable, "-c",
            "import torch; print({'version':torch.__version__,'cuda':torch.version.cuda,'available':torch.cuda.is_available(),'count':torch.cuda.device_count()}); print([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])",
        ]),
        "swiftllm_commit": (
            Path("vendor/swiftLLM/UPSTREAM_COMMIT").read_text().strip()
            if Path("vendor/swiftLLM/UPSTREAM_COMMIT").exists()
            else command_output(["git", "-C", "vendor/swiftLLM", "rev-parse", "HEAD"])
        ),
        "swiftllm_research_diff_sha256": digest(research_diff) if research_diff.exists() else None,
        "swiftllm_status": (
            "research fork; clean pinned source is vendor/swiftLLM-upstream; "
            "see references/swiftllm-research.diff"
            if Path("vendor/swiftLLM/UPSTREAM_COMMIT").exists()
            else command_output(["git", "-C", "vendor/swiftLLM", "status", "--short"])
        ),
        "model": model,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(snapshot, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
