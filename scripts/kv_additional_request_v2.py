#!/usr/bin/env python3
"""Measure a real request allocation against the unchanged SwiftLLM block pool.

This is a direct block-manager probe, not a scheduler or admission policy.  It
prefills ``base_batch`` live sequences, then attempts one additional sequence
with an exactly-full pool and with one additional physical page group.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.kv_capacity_experiment_v2 import MODEL_PATHS, environment, model_files, prompts, snapshot, sync  # noqa: E402

BLOCK_SIZE = 16


def source_hash() -> str:
    digest = hashlib.sha256()
    for name in ("scripts/kv_additional_request_v2.py", "scripts/kv_capacity_experiment_v2.py", "vendor/swiftLLM/swiftllm/worker/model.py", "vendor/swiftLLM/swiftllm/worker/weight.py", "vendor/swiftLLM/swiftllm/engine_config.py"):
        path = ROOT / name
        digest.update(name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def block_state(model) -> dict[str, object]:
    manager = model.gpu_block_manager
    counts = manager.num_seq_allocated_blocks.detach().cpu().tolist()
    return {"free_blocks": int(manager.num_free_blocks), "allocated_blocks": int(sum(counts)), "sequence_block_counts": [int(x) for x in counts], "block_table_shape": list(manager.block_table.shape)}


def one_pool(path: str, context: int, decode: int, base_batch: int, pool_groups: int, device: torch.device, budget_fraction: float) -> dict[str, object]:
    from swiftllm.engine_config import EngineConfig
    from swiftllm.worker.model import LlamaModel
    pages_per_request = (context + BLOCK_SIZE - 1) // BLOCK_SIZE
    pool_blocks = pool_groups * pages_per_request
    config = EngineConfig(model_path=path, use_dummy=False, block_size=BLOCK_SIZE, gpu_mem_utilization=budget_fraction, num_cpu_blocks=0, max_seqs_in_block_table=base_batch + 2, max_blocks_per_seq=pages_per_request + 2, max_batch_size=base_batch + 1, max_tokens_in_batch=base_batch * context + base_batch, kv_page_format="dense_fp16")
    model = LlamaModel(config)
    model.load_weights()
    model.init_kvcache_and_swap(pool_blocks)
    _, prompt_lists = prompts(__import__("transformers").AutoTokenizer.from_pretrained(path, local_files_only=True), context, base_batch + 1)
    before = {"allocator": snapshot(device), "blocks": block_state(model), "cache_capacity_bytes": int(model.k_cache.numel() * model.k_cache.element_size() * 2), "pages_per_request": pages_per_request, "pool_blocks": pool_blocks}
    model.forward(prompt_lists[:base_batch], list(range(base_batch)), [], return_logits=False)
    sync(device)
    after_base = {"allocator": snapshot(device), "blocks": block_state(model)}
    additional = {"status": "started"}
    try:
        model.forward([prompt_lists[base_batch]], [base_batch], [], return_logits=False)
        sync(device)
        additional = {"status": "completed", "allocator": snapshot(device), "blocks": block_state(model), "allocated_additional_blocks": int(block_state(model)["sequence_block_counts"][base_batch])}
    except Exception as exc:
        additional = {"status": "failed", "error": {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}, "allocator": snapshot(device), "blocks": block_state(model)}
    model.free_seqs_resources(list(range(base_batch + 1)))
    sync(device)
    released = {"allocator": snapshot(device), "blocks": block_state(model)}
    del model
    torch.cuda.empty_cache()
    sync(device)
    return {"pool_groups": pool_groups, "pool_blocks": pool_blocks, "before": before, "after_base": after_base, "additional_request": additional, "released": released}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-family", choices=tuple(MODEL_PATHS), default="llama31_8b")
    parser.add_argument("--model-path")
    parser.add_argument("--context-tokens", type=int, default=2048)
    parser.add_argument("--decode-tokens", type=int, default=256)
    parser.add_argument("--base-batch", type=int, default=13)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--budget-fraction", type=float, default=.90)
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(args.budget_fraction, device)
    path = str(Path(args.model_path or MODEL_PATHS[args.model_family]).resolve())
    try:
        import sys as _sys
        _sys.path.insert(0, str(ROOT / "vendor/swiftLLM"))
        _sys.path.insert(0, str(ROOT / "vendor/swiftLLM/csrc"))
        pages = (args.context_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE
        result = {"schema": "kv-additional-request-v2", "status": "completed", "model_family": args.model_family, "model_path": path, "model_files": model_files(path), "context_tokens": args.context_tokens, "decode_tokens": args.decode_tokens, "base_batch": args.base_batch, "budget": environment(device, args.budget_fraction), "provenance": {"git_head": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(), "source_sha256": source_hash(), "swiftllm_upstream_commit": (ROOT / "vendor/swiftLLM/UPSTREAM_COMMIT").read_text().strip()}, "protocol": "base batch is prefetched, then one additional request is allocated in a fresh exact-pool and one-extra-pool replay; no scheduler", "exact_pool": one_pool(path, args.context_tokens, args.decode_tokens, args.base_batch, args.base_batch, device, args.budget_fraction), "one_extra_pool": one_pool(path, args.context_tokens, args.decode_tokens, args.base_batch, args.base_batch + 1, device, args.budget_fraction), "invocation": {"argv": sys.argv, "cwd": os.getcwd()}}
    except Exception as exc:
        result = {"schema": "kv-additional-request-v2", "status": "blocked", "error": {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}, "invocation": {"argv": sys.argv, "cwd": os.getcwd()}}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(output), "status": result["status"], "exact_additional": result.get("exact_pool", {}).get("additional_request", {}).get("status"), "one_extra_additional": result.get("one_extra_pool", {}).get("additional_request", {}).get("status")}))


if __name__ == "__main__":
    main()
