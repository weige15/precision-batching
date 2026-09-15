#!/usr/bin/env python3
"""One isolated full-model KV capacity cell for the v2 study.

Each invocation runs one method/model/batch/length in a fresh process.  The
caller can therefore record OOMs without poisoning later cells.  ``timing``
uses one synchronization at each phase boundary; ``memory`` is a separate
per-position synchronized replay.  No scheduler or request-specific policy is
implemented.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MODEL_PATHS = {
    "llama32_1b": "/nfs/home/s314511048/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6",
    "llama31_8b": "/nfs/home/s314511048/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b",
}
BLOCK_SIZE = 16
SOURCE_FILES = (
    "scripts/kv_capacity_experiment_v2.py",
    "scripts/reproduce_kivi.py",
    "vendor/swiftLLM/swiftllm/worker/model.py",
    "vendor/swiftLLM/swiftllm/worker/kv_cache.py",
    "vendor/public/KIVI/models/llama_kivi.py",
    "vendor/public/KIVI/quant/new_pack.py",
    "vendor/public/KIVI/quant/matmul.py",
)


def source_sha256() -> str:
    digest = hashlib.sha256()
    for name in SOURCE_FILES:
        digest.update(name.encode())
        path = ROOT / name
        if name.startswith("vendor/public/KIVI/"):
            digest.update(subprocess.check_output(["git", "-C", str(ROOT / "vendor/public/KIVI"), "rev-parse", "HEAD"], text=True).encode())
        else:
            digest.update(path.read_bytes())
    return digest.hexdigest()


def model_files(path: str) -> list[dict[str, object]]:
    root = Path(path)
    result = []
    for item in sorted(root.iterdir()):
        if item.is_file() and (item.name.endswith(".safetensors") or item.name in {"config.json", "model.safetensors.index.json"}):
            result.append({"name": item.name, "bytes": item.stat().st_size, "sha256": hashlib.sha256(item.read_bytes()).hexdigest() if item.name in {"config.json", "model.safetensors.index.json"} else None})
    return result


def environment(device: torch.device, budget_fraction: float) -> dict[str, object]:
    props = torch.cuda.get_device_properties(device)
    total = int(props.total_memory)
    return {
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
        "device_name": props.name,
        "capability": list(torch.cuda.get_device_capability(device)),
        "total_device_memory_bytes": total,
        "declared_budget_fraction": budget_fraction,
        "declared_budget_bytes": int(total * budget_fraction),
        "declared_budget_mib": total * budget_fraction / (1024 ** 2),
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def snapshot(device: torch.device) -> dict[str, int]:
    return {"allocated_bytes": int(torch.cuda.memory_allocated(device)), "reserved_bytes": int(torch.cuda.memory_reserved(device)), "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)), "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device))}


def sync(device: torch.device) -> None:
    torch.cuda.synchronize(device)


def cleanup() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def prompts(tokenizer, context: int, batch: int) -> tuple[torch.Tensor, list[list[int]]]:
    # Repeated text is explicitly a mechanism workload, not held-out quality.
    seeds = [
        "A red kite crossed the northern valley while the archivist measured each page. ",
        "A blue lantern marked the eastern trail and the engineer recorded the boundary. ",
        "A green vessel entered the quiet harbor as the researcher checked the ledger. ",
        "A gold bridge joined two districts while the operator watched memory pressure. ",
        "A silver compass pointed toward the old library where the analyst stored notes. ",
        "A violet train left at dawn and the observer tracked every token position. ",
        "An orange signal crossed the desert and the technician verified the cache. ",
        "A white tower stood beside the river while the tester compared two paths. ",
    ]
    ids = []
    for index in range(batch):
        one = tokenizer.encode(seeds[index % len(seeds)], add_special_tokens=True)
        repeats = (context + len(one) - 1) // len(one)
        ids.append((one * repeats)[:context])
    return torch.tensor(ids, dtype=torch.long), ids


def fixed_targets(tokenizer, decode: int, batch: int) -> list[torch.Tensor]:
    base = tokenizer.encode(" The next measured token is part of the fixed decode trajectory.", add_special_tokens=False)
    if not base:
        base = [1]
    return [torch.tensor([base[i % len(base)]] * batch, dtype=torch.long) for i in range(decode)]


def cache_state(model, cache, device: torch.device, position: int, phase: str) -> dict[str, object]:
    from scripts.reproduce_kivi import model_cache_ledger
    return {"phase": phase, "token_position": position, "cache": model_cache_ledger(model, cache), "allocator": snapshot(device)}


def load_hf(method: str, path: str, device: torch.device, bits: int, group_size: int, residual_length: int):
    if method == "hf_fp16":
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16, low_cpu_mem_usage=True, attn_implementation="sdpa", local_files_only=True).to(device).eval()
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        return model, tokenizer, {"weights_dtype": str(next(model.parameters()).dtype), "kv_backend": "Transformers DynamicCache FP16", "attention_implementation": "sdpa"}
    from scripts.reproduce_kivi import install_compat_shims
    install_compat_shims()
    sys.path.insert(0, str(ROOT / "vendor/public/KIVI/quant"))
    sys.path.insert(0, str(ROOT / "vendor/public/KIVI"))
    from models.llama_kivi import LlamaForCausalLM_KIVI
    from transformers import AutoConfig, AutoTokenizer
    config = AutoConfig.from_pretrained(path, local_files_only=True)
    config.k_bits = bits
    config.v_bits = bits
    config.group_size = group_size
    config.residual_length = residual_length
    config.use_flash = True
    config._flash_attn_2_enabled = True
    model = LlamaForCausalLM_KIVI.from_pretrained(path, config=config, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map={"": device.index or 0}, local_files_only=True).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    return model, tokenizer, {"weights_dtype": str(next(model.parameters()).dtype), "kv_backend": f"KIVI native K{bits}V{bits}", "k_bits": bits, "v_bits": bits, "group_size": group_size, "residual_length": residual_length, "wrapper_logits_avoided": True, "upstream_commit": subprocess.check_output(["git", "-C", str(ROOT / "vendor/public/KIVI"), "rev-parse", "HEAD"], text=True).strip()}


def hf_call(model, input_ids: torch.Tensor, past=None):
    # Run the complete decoder but project only the final token.  The public
    # KIVI wrapper materializes [batch, context, vocab] FP32 logits, which is
    # not a serving requirement and can swamp a long-context capacity cell.
    kwargs = {"input_ids": input_ids, "use_cache": True, "return_dict": True}
    if past is not None:
        kwargs["past_key_values"] = past
    outputs = model.model(**kwargs)
    hidden = outputs.last_hidden_state[:, -1:, :]
    logits = model.lm_head(hidden).float()
    return SimpleNamespace(past_key_values=outputs.past_key_values, logits=logits)


@torch.inference_mode()
def hf_run(model, tokenizer, context: int, decode: int, batch: int, device: torch.device, warmups: int, repeats: int, memory_only: bool) -> dict[str, object]:
    ids_cpu, prompt_lists = prompts(tokenizer, context, batch)
    ids = ids_cpu.to(device)
    targets = [x.to(device) for x in fixed_targets(tokenizer, decode, batch)]

    def trajectory(record_memory: bool = False, timed: bool = False):
        prefill_start = time.perf_counter() if timed else None
        output = hf_call(model, ids)
        if timed:
            sync(device)
        prefill_ms = (time.perf_counter() - prefill_start) * 1000 if timed else None
        past = output.past_key_values
        trace = [cache_state(model, past, device, context, "prefill")] if record_memory else []
        decode_start = time.perf_counter() if timed else None
        for step, token in enumerate(targets, start=1):
            output = hf_call(model, token[:, None], past)
            past = output.past_key_values
            if record_memory:
                sync(device)
                trace.append(cache_state(model, past, device, context + step, "decode"))
        if timed:
            sync(device)
        decode_ms = (time.perf_counter() - decode_start) * 1000 if timed else None
        final = cache_state(model, past, device, context + decode, "decode_end")
        del output, past
        return {"prefill_ms": prefill_ms, "decode_ms": decode_ms, "trace": trace, "final": final}

    cold_start = None
    if not memory_only:
        torch.cuda.reset_peak_memory_stats(device)
        cold_start = trajectory(timed=True)
        cold_start["peak"] = snapshot(device)
        cleanup()
        sync(device)
    for _ in range(warmups):
        trajectory()
        cleanup()
        sync(device)
    repetitions = []
    for _ in range(repeats if not memory_only else 1):
        cleanup()
        sync(device)
        torch.cuda.reset_peak_memory_stats(device)
        base = snapshot(device)
        row = trajectory(record_memory=memory_only, timed=not memory_only)
        row["base_model_runtime"] = base
        row["peak"] = snapshot(device)
        row["budget_check"] = budget_check(row["peak"], device)
        repetitions.append(row)
    return {"prompt_tokens": context, "batch_size": batch, "decode_tokens": decode, "prompt_kind": "distinct-seed-repeated-token-mechanism-workload", "target_inputs_prepared_before_timing": True, "cold_start": cold_start, "repetitions": repetitions, "memory_instrumented": memory_only, "timing_boundary": "synchronized wall-clock around complete prefill/decode model calls; no quality scoring or per-token host synchronization" if not memory_only else "not a latency run; synchronized after every position for memory evidence"}


def load_swift(path: str, context: int, decode: int, batch: int, device: torch.device, budget_fraction: float):
    sys.path.insert(0, str(ROOT / "vendor/swiftLLM"))
    sys.path.insert(0, str(ROOT / "vendor/swiftLLM/csrc"))
    from swiftllm.engine_config import EngineConfig
    from swiftllm.worker.model import LlamaModel
    pages = math.ceil((context + decode) / BLOCK_SIZE)
    config = EngineConfig(model_path=path, use_dummy=False, block_size=BLOCK_SIZE, gpu_mem_utilization=budget_fraction, num_cpu_blocks=0, max_seqs_in_block_table=max(8, batch + 1), max_blocks_per_seq=pages + 2, max_batch_size=batch, max_tokens_in_batch=context * batch + batch, kv_page_format="dense_fp16")
    model = LlamaModel(config)
    model.load_weights()
    model.init_kvcache_and_swap(batch * pages)
    tokenizer = __import__("transformers").AutoTokenizer.from_pretrained(path, local_files_only=True)
    return model, tokenizer, {"weights_dtype": "torch.float16", "kv_backend": "unchanged SwiftLLM dense FP16 allocator", "allocated_cache_capacity_bytes": int(model.k_cache.numel() * model.k_cache.element_size() * 2), "pages_reserved": batch * pages}


def swift_state(model, device: torch.device, context: int, decode: int, phase: str) -> dict[str, object]:
    manager = model.gpu_block_manager
    counts = manager.num_seq_allocated_blocks.detach().cpu().tolist()
    active = counts[: int(model.engine_config.max_batch_size)]
    page_bytes = model.model_config.get_kvslot_size() * BLOCK_SIZE
    used_pages = sum(int(x) for x in active)
    return {"phase": phase, "token_position": context if phase == "prefill" else context + decode, "cache": {"mode": "dense_fp16", "logical_payload_bytes": used_pages * page_bytes, "unique_live_storage_bytes": int(model.k_cache.numel() * model.k_cache.element_size() * 2), "allocated_cache_capacity_bytes": int(model.k_cache.numel() * model.k_cache.element_size() * 2), "used_pages": used_pages, "per_sequence_pages": [int(x) for x in active], "both_k_and_v": True}, "allocator": snapshot(device)}


def swift_run(model, tokenizer, context: int, decode: int, batch: int, device: torch.device, warmups: int, repeats: int, memory_only: bool) -> dict[str, object]:
    _, prompt_lists = prompts(tokenizer, context, batch)
    targets = [int(x[0].item()) for x in fixed_targets(tokenizer, decode, batch)]
    seq_ids = list(range(batch))

    def trajectory(record_memory=False, timed=False):
        # return_logits keeps all work on-device; the fixed next inputs avoid
        # host synchronization from greedy selection inside the timing scope.
        start = time.perf_counter() if timed else None
        output = model.forward(prompt_lists, seq_ids, [], return_logits=True)
        if timed:
            sync(device)
        prefill_ms = (time.perf_counter() - start) * 1000 if timed else None
        trace = [swift_state(model, device, context, 0, "prefill")] if record_memory else []
        start = time.perf_counter() if timed else None
        for step, token in enumerate(targets, start=1):
            model.forward([[token] for _ in range(batch)], seq_ids, [context + step] * batch, return_logits=True)
            if record_memory:
                sync(device)
                trace.append(swift_state(model, device, context, step, "decode"))
        if timed:
            sync(device)
        decode_ms = (time.perf_counter() - start) * 1000 if timed else None
        final = swift_state(model, device, context, decode, "decode_end")
        model.free_seqs_resources(seq_ids)
        sync(device)
        return {"prefill_ms": prefill_ms, "decode_ms": decode_ms, "trace": trace, "final": final}

    cold_start = None
    if not memory_only:
        torch.cuda.reset_peak_memory_stats(device)
        cold_start = trajectory(timed=True)
        cold_start["peak"] = snapshot(device)
        sync(device)
    for _ in range(warmups):
        trajectory()
        sync(device)
    repetitions = []
    for _ in range(repeats if not memory_only else 1):
        sync(device)
        torch.cuda.reset_peak_memory_stats(device)
        base = snapshot(device)
        row = trajectory(record_memory=memory_only, timed=not memory_only)
        row["base_model_runtime"] = base
        row["peak"] = snapshot(device)
        row["budget_check"] = budget_check(row["peak"], device)
        repetitions.append(row)
    return {"prompt_tokens": context, "batch_size": batch, "decode_tokens": decode, "prompt_kind": "distinct-seed-repeated-token-mechanism-workload", "target_inputs_prepared_before_timing": True, "cold_start": cold_start, "repetitions": repetitions, "memory_instrumented": memory_only, "timing_boundary": "synchronized wall-clock around complete unchanged SwiftLLM model.forward calls; no quality scoring or per-token host synchronization" if not memory_only else "not a latency run; synchronized after every position for memory evidence"}


def budget_check(peak: dict[str, int], device: torch.device) -> dict[str, object]:
    budget = int(torch.cuda.get_device_properties(device).total_memory * CURRENT_BUDGET_FRACTION)
    return {"declared_budget_bytes": budget, "peak_allocated_bytes": peak["peak_allocated_bytes"], "peak_reserved_bytes": peak["peak_reserved_bytes"], "within_budget_allocated": peak["peak_allocated_bytes"] <= budget, "within_budget_reserved": peak["peak_reserved_bytes"] <= budget, "within_budget": max(peak["peak_allocated_bytes"], peak["peak_reserved_bytes"]) <= budget}


CURRENT_BUDGET_FRACTION = 0.90


def run(args: argparse.Namespace) -> dict[str, object]:
    global CURRENT_BUDGET_FRACTION
    CURRENT_BUDGET_FRACTION = args.budget_fraction
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    env = environment(device, args.budget_fraction)
    torch.cuda.set_per_process_memory_fraction(args.budget_fraction, device)
    path = str(Path(args.model_path or MODEL_PATHS[args.model_family]).resolve())
    base = {"schema": "kv-capacity-cell-v2", "status": "started", "method": args.method, "model_family": args.model_family, "model_path": path, "model_files": model_files(path), "context_tokens": args.context_tokens, "decode_tokens": args.decode_tokens, "batch_size": args.batch_size, "instrumentation": args.instrumentation, "budget": env, "provenance": {"branch": subprocess.check_output(["git", "-C", str(ROOT), "branch", "--show-current"], text=True).strip(), "git_head": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(), "source_files": list(SOURCE_FILES), "source_sha256": source_sha256(), "kivi_commit": subprocess.check_output(["git", "-C", str(ROOT / "vendor/public/KIVI"), "rev-parse", "HEAD"], text=True).strip(), "swiftllm_upstream_commit": (ROOT / "vendor/swiftLLM/UPSTREAM_COMMIT").read_text().strip()}}
    if args.method == "swiftllm_fp16":
        model, tokenizer, method_config = load_swift(path, args.context_tokens, args.decode_tokens, args.batch_size, device, args.budget_fraction)
        loaded = snapshot(device)
        result = swift_run(model, tokenizer, args.context_tokens, args.decode_tokens, args.batch_size, device, args.warmups, args.repeats, args.instrumentation == "memory")
    else:
        bits = args.bits or (2 if args.method == "kivi2" else 4)
        residual_length = 65536 if args.method == "kivi_fp16" and args.residual_length == 32 else args.residual_length
        model, tokenizer, method_config = load_hf(args.method, path, device, bits, args.group_size, residual_length)
        loaded = snapshot(device)
        result = hf_run(model, tokenizer, args.context_tokens, args.decode_tokens, args.batch_size, device, args.warmups, args.repeats, args.instrumentation == "memory")
    base.update({"status": "completed", "method_config": method_config, "model_loaded_allocator": loaded, "run": result})
    # The trajectory is complete only if all decode positions returned and the
    # peak includes both allocated and reserved allocator state.
    rows = result["repetitions"]
    base["completed_decode_steps"] = [args.decode_tokens if (row.get("final") and row["final"]["token_position"] == args.context_tokens + args.decode_tokens) else None for row in rows]
    base["feasible"] = bool(rows and all(row["budget_check"]["within_budget"] for row in rows) and all(x == args.decode_tokens for x in base["completed_decode_steps"]))
    del model
    cleanup()
    sync(device)
    base["post_release_allocator"] = snapshot(device)
    return base


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--method", choices=("hf_fp16", "kivi_fp16", "kivi2", "kivi4", "swiftllm_fp16"), required=True)
    parser.add_argument("--model-family", choices=tuple(MODEL_PATHS), required=True)
    parser.add_argument("--model-path")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--context-tokens", type=int, required=True)
    parser.add_argument("--decode-tokens", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--instrumentation", choices=("timing", "memory"), default="timing")
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--bits", type=int, choices=(2, 4))
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--residual-length", type=int, default=32)
    parser.add_argument("--budget-fraction", type=float, default=0.90)
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = run(args)
    except Exception as exc:  # One cell must retain OOM/unsupported evidence.
        observed = {}
        try:
            device = torch.device(args.device)
            if torch.cuda.is_available():
                torch.cuda.synchronize(device)
                observed = {"environment": environment(device, args.budget_fraction), "allocator_at_failure": snapshot(device)}
        except Exception:
            observed = {"allocator_observation": "unavailable after failure"}
        result = {"schema": "kv-capacity-cell-v2", "status": "oom" if "out of memory" in str(exc).lower() else "error", "method": args.method, "model_family": args.model_family, "context_tokens": args.context_tokens, "decode_tokens": args.decode_tokens, "batch_size": args.batch_size, "instrumentation": args.instrumentation, **observed, "error": {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}, "invocation": {"argv": sys.argv, "cwd": os.getcwd()}}
    result["invocation"] = {"argv": sys.argv, "cwd": os.getcwd()}
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(output), "status": result["status"], "feasible": result.get("feasible"), "method": args.method, "model_family": args.model_family, "batch": args.batch_size, "context": args.context_tokens, "decode": args.decode_tokens}))


if __name__ == "__main__":
    main()
