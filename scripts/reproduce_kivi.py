"""Reproduce the public KIVI decode path with auditable v2 measurements.

The KIVI source under ``vendor/public/KIVI`` is not modified.  This adapter
only supplies compatibility shims and a measurement boundary.  Performance
runs contain model prefill/decode calls only; teacher-forced quality scoring is
run separately.  Cache accounting records both K and V for DynamicCache,
legacy tuple payloads, aliases, token position, allocator state, and cache
release state.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.machinery
import json
import os
import subprocess
import sys
import time
import types
import typing
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
KIVI_ROOT = ROOT / "vendor/public/KIVI"
KIVI_QUANT = KIVI_ROOT / "quant"


def install_compat_shims() -> dict[str, str]:
    """Make the pinned KIVI source import/run on the local package versions."""
    import vllm_flash_attn

    flash_attn = types.ModuleType("flash_attn")
    flash_attn.__spec__ = importlib.machinery.ModuleSpec("flash_attn", loader=None)
    flash_attn.__version__ = "2.6.2-compat-vllm_flash_attn"
    flash_attn.flash_attn_func = vllm_flash_attn.flash_attn_func
    flash_attn.flash_attn_varlen_func = vllm_flash_attn.flash_attn_varlen_func

    padding = types.ModuleType("flash_attn.bert_padding")
    padding.__spec__ = importlib.machinery.ModuleSpec("flash_attn.bert_padding", loader=None)

    def index_first_axis(x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        return x.reshape(x.shape[0], *x.shape[2:])[indices]

    def unpad_input(x: torch.Tensor, mask: torch.Tensor):
        indices = mask.flatten().nonzero(as_tuple=False).flatten()
        values = x.reshape(-1, *x.shape[2:])[indices]
        lengths = mask.sum(1, dtype=torch.int32)
        cumulative = torch.nn.functional.pad(lengths.cumsum(0), (1, 0))
        return values, indices, cumulative, int(lengths.max().item())

    def pad_input(x: torch.Tensor, indices: torch.Tensor, batch: int, length: int):
        result = torch.zeros(
            (batch * length, *x.shape[1:]), device=x.device, dtype=x.dtype
        )
        result[indices] = x
        return result.view(batch, length, *x.shape[1:])

    padding.index_first_axis = index_first_axis
    padding.unpad_input = unpad_input
    padding.pad_input = pad_input
    sys.modules["flash_attn"] = flash_attn
    sys.modules["flash_attn.bert_padding"] = padding

    # Transformers >=4.51 uses an explicit __all__, while KIVI's source was
    # written against a module whose star import exported these names.
    import transformers.models.llama.modeling_llama as llama_module
    from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

    llama_module._prepare_4d_causal_attention_mask = _prepare_4d_causal_attention_mask
    llama_module.Union = typing.Union
    llama_module.__all__ = [
        name for name in llama_module.__dict__ if not name.startswith("_")
    ] + ["_prepare_4d_causal_attention_mask", "Union"]
    return {
        "transformers": __import__("transformers").__version__,
        "flash_attn_provider": "vllm_flash_attn.flash_attn_func/varlen_func",
        "compatibility": "KIVI star-import exports + flash_attn.bert_padding shim",
        "upstream_kivi_source_modified": "false",
    }


def tensor_logical_bytes(value) -> int:
    return int(value.numel() * value.element_size()) if isinstance(value, torch.Tensor) else 0


def _iter_cache_tensors(cache, prefix="cache"):
    """Yield every tensor in a KIVI cache, including both K and V.

    KIVI returns a legacy tuple while the ordinary Transformers control uses a
    DynamicCache.  A recursive walk is intentional: it also catches future
    nested metadata records without treating scalar sequence lengths as bytes.
    """
    if isinstance(cache, torch.Tensor):
        yield prefix, cache
    elif isinstance(cache, (tuple, list)):
        for index, value in enumerate(cache):
            yield from _iter_cache_tensors(value, f"{prefix}[{index}]")
    elif isinstance(cache, dict):
        for key, value in cache.items():
            yield from _iter_cache_tensors(value, f"{prefix}[{key!r}]")
    elif hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        for index, value in enumerate(cache.key_cache):
            yield from _iter_cache_tensors(value, f"key_cache[{index}]")
        for index, value in enumerate(cache.value_cache):
            yield from _iter_cache_tensors(value, f"value_cache[{index}]")


def _storage_info(tensor: torch.Tensor) -> tuple[tuple[object, ...], int]:
    try:
        storage = tensor.untyped_storage()
        size = int(storage.nbytes())
        pointer = int(storage.data_ptr())
    except (AttributeError, RuntimeError):
        size = tensor_logical_bytes(tensor)
        pointer = int(tensor.data_ptr())
    if size == 0:
        # Empty tensors can share a null pointer without sharing storage.
        pointer = id(tensor)
    key = (tensor.device.type, tensor.device.index, pointer, size)
    return key, size


def cache_inventory(cache) -> dict[str, object]:
    records = []
    seen: dict[tuple[object, ...], int] = {}
    logical = 0
    unique = 0
    for name, tensor in _iter_cache_tensors(cache):
        key, storage_bytes = _storage_info(tensor)
        logical_bytes = tensor_logical_bytes(tensor)
        logical += logical_bytes
        if key not in seen:
            seen[key] = storage_bytes
            unique += storage_bytes
        records.append({
            "name": name,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "logical_bytes": logical_bytes,
            "storage_bytes": storage_bytes,
            "storage_alias_id": f"{key[0]}:{key[1]}:{key[2]}:{key[3]}",
        })
    return {
        "tensor_count": len(records),
        "logical_payload_bytes": logical,
        "unique_live_storage_bytes": unique,
        "tensors": records,
    }


def cache_bytes(cache) -> int:
    """Return logical bytes for both K and V, never just the zipped V side."""
    return int(cache_inventory(cache)["logical_payload_bytes"])


def cache_shape_summary(cache) -> dict[str, object]:
    if isinstance(cache, tuple):
        return {
            "kind": "kivi_legacy_tuple",
            "layers": len(cache),
            "layer0": [list(x.shape) if isinstance(x, torch.Tensor) else x for x in cache[0]],
        }
    return {
        "kind": type(cache).__name__,
        "layers": len(cache.key_cache),
        "key0": list(cache.key_cache[0].shape),
        "value0": list(cache.value_cache[0].shape),
    }


def _owned_model_cache_tensors(model):
    """Inspect model-owned cache-like tensor attributes without counting weights."""
    found = []
    for module_name, module in model.named_modules():
        for name, value in vars(module).items():
            lowered = name.lower()
            if any(marker in lowered for marker in ("kcache", "vcache", "k_cache", "v_cache", "kv_cache")) and isinstance(value, torch.Tensor):
                found.append((f"{module_name}.{name}", value))
    return found


def model_cache_ledger(model, cache) -> dict[str, object]:
    past = cache_inventory(cache)
    owned_records = []
    owned_seen = set()
    owned_logical = 0
    owned_unique = 0
    for name, tensor in _owned_model_cache_tensors(model):
        key, storage_bytes = _storage_info(tensor)
        logical_bytes = tensor_logical_bytes(tensor)
        owned_logical += logical_bytes
        if key not in owned_seen:
            owned_seen.add(key)
            owned_unique += storage_bytes
        owned_records.append({"name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype), "logical_bytes": logical_bytes, "storage_bytes": storage_bytes})
    past["model_owned_cache_tensors"] = owned_records
    past["model_owned_cache_logical_bytes"] = owned_logical
    past["model_owned_cache_unique_live_storage_bytes"] = owned_unique
    return past


def allocator_snapshot(device: torch.device) -> dict[str, int]:
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def cache_state(model, cache, device: torch.device, position: int, phase: str) -> dict[str, object]:
    return {
        "phase": phase,
        "token_position": int(position),
        "cache": model_cache_ledger(model, cache),
        "allocator": allocator_snapshot(device),
    }


def release_observation(cache_holder: list, device: torch.device, base_allocated: int) -> dict[str, int]:
    """Measure a live past, then clear its last reference before the snapshot."""
    torch.cuda.synchronize(device)
    before = allocator_snapshot(device)
    cache_holder.clear()
    gc.collect()
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    after = allocator_snapshot(device)
    return {
        "base_model_allocated_bytes": int(base_allocated),
        "allocated_before_empty_cache_bytes": before["allocated_bytes"],
        "reserved_before_empty_cache_bytes": before["reserved_bytes"],
        "allocated_after_empty_cache_bytes": after["allocated_bytes"],
        "reserved_after_empty_cache_bytes": after["reserved_bytes"],
        "allocator_allocated_released_bytes": before["allocated_bytes"] - after["allocated_bytes"],
        "allocator_reserved_released_bytes": before["reserved_bytes"] - after["reserved_bytes"],
    }


def _model_inputs(targets: list[int], device: torch.device, batch: int) -> list[torch.Tensor]:
    return [torch.full((batch, 1), token, device=device, dtype=torch.long) for token in targets]


def _full_model_pass(model, ids: torch.Tensor, inputs: list[torch.Tensor]) -> None:
    with torch.no_grad():
        output = model(ids, use_cache=True)
        past = output.past_key_values
        for next_input in inputs:
            output = model(next_input, past_key_values=past, use_cache=True)
            past = output.past_key_values
    del output, past


def run_performance(model, ids: torch.Tensor, targets: list[int], warmups: int, repeats: int) -> dict[str, object]:
    """Measure only prefill and model-only decode; quality is not in this path."""
    device = ids.device
    inputs = _model_inputs(targets, device, ids.shape[0])
    for _ in range(warmups):
        _full_model_pass(model, ids, inputs)
        torch.cuda.synchronize(device)
    prefill_ms = []
    decode_ms = []
    prefill_states = []
    decode_states = []
    releases = []
    for _ in range(repeats):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        base_allocated = int(torch.cuda.memory_allocated(device))
        start = time.perf_counter()
        with torch.no_grad():
            output = model(ids, use_cache=True)
        torch.cuda.synchronize(device)
        prefill_ms.append((time.perf_counter() - start) * 1000.0)
        past = output.past_key_values
        prefill_states.append(cache_state(model, past, device, ids.shape[1], "prefill"))
        output = None
        gc.collect()
        torch.cuda.synchronize(device)

        # Inputs are prepared before timing. There is one synchronization after
        # the complete loop, not a .item()/synchronize at every token.
        start = time.perf_counter()
        with torch.no_grad():
            for next_input in inputs:
                output = model(next_input, past_key_values=past, use_cache=True)
                past = output.past_key_values
        torch.cuda.synchronize(device)
        decode_ms.append((time.perf_counter() - start) * 1000.0)
        decode_states.append(cache_state(model, past, device, ids.shape[1] + len(targets), "decode_end"))
        del output
        # Hold the live past in a mutable holder, clear the caller reference,
        # then measure before/after clearing the holder. This captures actual
        # cache release rather than measuring after an already-deleted object.
        holder = [past]
        past = None
        releases.append(release_observation(holder, device, base_allocated))
    return {
        "timing_unit": "milliseconds",
        "timing_boundary": "synchronized wall-clock around model forward calls; input construction and quality scoring excluded",
        "prefill_ms": prefill_ms,
        "decode_ms_for_tokens": decode_ms,
        "decode_ms_per_token": [x / len(targets) for x in decode_ms],
        "decode_token_count": len(targets),
        "repetitions": repeats,
        "warmups": warmups,
        "cache_at_prefill": prefill_states,
        "cache_at_decode_end": decode_states,
        "cache_release": releases,
    }


def run_quality(model, ids: torch.Tensor, targets: list[int]) -> dict[str, object]:
    """Run paired teacher-forced scoring separately from the timed pass."""
    device = ids.device
    inputs = _model_inputs(targets, device, ids.shape[0])
    with torch.no_grad():
        output = model(ids, use_cache=True)
        past = output.past_key_values
        logits = output.logits[:, -1].detach().clone()
        nlls = []
        top1 = []
        for next_input, token in zip(inputs, targets):
            target = torch.full((ids.shape[0],), token, device=device, dtype=torch.long)
            nlls.append(float(torch.nn.functional.cross_entropy(logits.float(), target).item()))
            top1.append(float((logits.argmax(-1) == target).float().mean().item()))
            output = model(next_input, past_key_values=past, use_cache=True)
            past = output.past_key_values
            logits = output.logits[:, -1].detach().clone()
    result = {
        "scoring_boundary": "un-timed teacher-forced logits and host-side NLL/top-1 scoring",
        "token_count": len(targets),
        "token_nll": nlls,
        "token_top1_agreement": top1,
        "mean_nll": sum(nlls) / len(nlls),
        "top1_agreement": sum(top1) / len(top1),
        "cache_position_after_scoring": ids.shape[1] + len(targets),
        "cache_ledger_after_scoring": model_cache_ledger(model, past),
        "cache_shape_after_scoring": cache_shape_summary(past),
    }
    del output, past, logits
    return result


def run_memory_trace(model, ids: torch.Tensor, targets: list[int]) -> dict[str, object]:
    """Replay one run with per-position synchronized memory snapshots."""
    device = ids.device
    inputs = _model_inputs(targets, device, ids.shape[0])
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    base_allocated = int(torch.cuda.memory_allocated(device))
    trace = []
    with torch.no_grad():
        torch.cuda.reset_peak_memory_stats(device)
        output = model(ids, use_cache=True)
        past = output.past_key_values
        torch.cuda.synchronize(device)
        trace.append(cache_state(model, past, device, ids.shape[1], "prefill"))
        for position, next_input in enumerate(inputs, start=ids.shape[1] + 1):
            torch.cuda.reset_peak_memory_stats(device)
            output = model(next_input, past_key_values=past, use_cache=True)
            past = output.past_key_values
            torch.cuda.synchronize(device)
            trace.append(cache_state(model, past, device, position, "decode"))
    del output
    holder = [past]
    past = None
    release = release_observation(holder, device, base_allocated)
    return {
        "trace_synchronization": "synchronize after each snapshot; this run is memory evidence, not latency evidence",
        "trace": trace,
        "release": release,
    }


def greedy_targets(model, ids: torch.Tensor, count: int) -> list[int]:
    tokens = ids.clone()
    result = []
    with torch.no_grad():
        output = model(tokens, use_cache=True)
        past = output.past_key_values
        next_token = output.logits[:, -1].argmax(-1)
        for _ in range(count):
            result.append(int(next_token[0].item()))
            output = model(next_token[:, None], past_key_values=past, use_cache=True)
            past = output.past_key_values
            next_token = output.logits[:, -1].argmax(-1)
    del output, past
    gc.collect()
    torch.cuda.synchronize(ids.device)
    return result


def hf_load(model_path: str, device: torch.device):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer


def kivi_load(model_path: str, device: torch.device, bits: int, group_size: int, residual_length: int):
    sys.path.insert(0, str(KIVI_QUANT))
    sys.path.insert(0, str(KIVI_ROOT))
    from models.llama_kivi import LlamaForCausalLM_KIVI
    from transformers import AutoConfig, AutoTokenizer

    config = AutoConfig.from_pretrained(model_path)
    config.k_bits = bits
    config.v_bits = bits
    config.group_size = group_size
    config.residual_length = residual_length
    config.use_flash = True
    config._flash_attn_2_enabled = True
    model = LlamaForCausalLM_KIVI.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map={"": device.index if device.index is not None else 0},
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer, config


def prompt_ids(tokenizer, prompt: str, prompt_tokens: int, device: torch.device) -> torch.Tensor:
    ids = tokenizer(prompt, return_tensors="pt").input_ids
    if ids.shape[1] < prompt_tokens:
        repeats = (prompt_tokens + ids.shape[1] - 1) // ids.shape[1]
        ids = ids.repeat(1, repeats)
    return ids[:, :prompt_tokens].to(device)


def release_model(model, device: torch.device) -> None:
    del model
    gc.collect()
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)


def run_method(model, ids, targets, warmups, repeats):
    performance = run_performance(model, ids, targets, warmups, repeats)
    quality = run_quality(model, ids, targets)
    memory = run_memory_trace(model, ids, targets)
    return performance, quality, memory


def run_one(model, ids, targets, warmups, repeats):
    performance, quality, memory = run_method(model, ids, targets, warmups, repeats)
    prefill_cache = performance["cache_at_prefill"]
    return {
        "performance": performance,
        "quality": quality,
        "memory": memory,
        # Compatibility fields retain the old public artifact shape while now
        # referring to the same-position, K+V-inclusive corrected snapshots.
        "prefill_ms": performance["prefill_ms"],
        "decode_ms_for_tokens": performance["decode_ms_for_tokens"],
        "decode_ms_per_token": performance["decode_ms_per_token"],
        "cache_bytes": [row["cache"]["logical_payload_bytes"] for row in prefill_cache],
        "allocated_delta_bytes": [row["allocator"]["allocated_bytes"] - memory["release"]["base_model_allocated_bytes"] for row in prefill_cache],
        "peak_delta_bytes": [row["allocator"]["peak_allocated_bytes"] - memory["release"]["base_model_allocated_bytes"] for row in prefill_cache],
        "paired_nll": [quality["mean_nll"]] * repeats,
        "top1_agreement": [quality["top1_agreement"]] * repeats,
        "cache_summary": quality["cache_shape_after_scoring"],
    }


def source_fingerprint() -> str:
    digest = hashlib.sha256()
    for path in ("scripts/reproduce_kivi.py", "vendor/public/KIVI"):
        digest.update(path.encode())
        if path.endswith(".py"):
            digest.update((ROOT / path).read_bytes())
        else:
            digest.update(subprocess.check_output(["git", "-C", str(ROOT / path), "rev-parse", "HEAD"], text=True).encode())
    return digest.hexdigest()


def dirty_provenance() -> dict[str, object]:
    status = subprocess.check_output(["git", "-C", str(ROOT), "status", "--short"], text=True)
    diff = subprocess.check_output(["git", "-C", str(ROOT), "diff", "--binary", "HEAD"], text=False)
    return {
        "reviewed_remote_commit": "8c0a0bfcc6bf87461c104603718ed2dc507df390",
        "local_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "branch": subprocess.check_output(["git", "-C", str(ROOT), "branch", "--show-current"], text=True).strip(),
        "dirty_status": status.splitlines(),
        "tracked_dirty_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "submodules": subprocess.check_output(["git", "-C", str(ROOT), "submodule", "status", "--recursive"], text=True).splitlines(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prompt-tokens", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--decode-tokens", type=int, default=16)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--bits", type=int, choices=[2, 4], default=2)
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--residual-length", type=int, default=32)
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("KIVI reproduction requires CUDA")
    torch.cuda.set_device(device)
    model_path = str(Path(args.model_path).resolve())
    prompt = "The quick brown fox jumps over the lazy dog. This is a reproducible KV cache benchmark."
    compat = install_compat_shims()

    # Baseline supplies deterministic teacher-forced targets and the matched
    # ordinary Transformers control. Its cache is released before KIVI loads.
    baseline, tokenizer = hf_load(model_path, device)
    ids = prompt_ids(tokenizer, prompt, args.prompt_tokens, device)
    targets = greedy_targets(baseline, ids, args.decode_tokens)
    ids = ids.repeat(args.batch_size, 1)
    baseline_result = run_one(baseline, ids, targets, args.warmups, args.repeats)
    release_model(baseline, device)
    del baseline

    kivi, kivi_tokenizer, config = kivi_load(
        model_path, device, args.bits, args.group_size, args.residual_length
    )
    kivi_ids = prompt_ids(kivi_tokenizer, prompt, args.prompt_tokens, device).repeat(args.batch_size, 1)
    kivi_result = run_one(kivi, kivi_ids, targets, args.warmups, args.repeats)
    release_model(kivi, device)
    del kivi

    baseline_cache = median(baseline_result["cache_bytes"])
    kivi_cache = median(kivi_result["cache_bytes"])
    baseline_decode = median(baseline_result["decode_ms_per_token"])
    kivi_decode = median(kivi_result["decode_ms_per_token"])
    result = {
        "schema_version": 2,
        "schema": "public-kv-reproduction-v2",
        "method": "KIVI",
        "upstream_commit": subprocess.check_output(["git", "-C", str(KIVI_ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "model_path": model_path,
        "model_config": {
            "num_attention_heads": int(config.num_attention_heads),
            "num_key_value_heads": int(config.num_key_value_heads),
            "head_dim": int(config.hidden_size // config.num_attention_heads),
            "gqa_groups": int(config.num_attention_heads // config.num_key_value_heads),
        },
        "kivi_config": {"k_bits": args.bits, "v_bits": args.bits, "group_size": args.group_size, "residual_length": args.residual_length},
        "device": torch.cuda.get_device_name(device),
        "cuda_capability": list(torch.cuda.get_device_capability(device)),
        "compatibility": compat,
        "provenance": dirty_provenance() | {"source_sha256": source_fingerprint()},
        "prompt_tokens": args.prompt_tokens,
        "batch_size": args.batch_size,
        "decode_tokens": args.decode_tokens,
        "targets": targets,
        "baseline_fp16": baseline_result,
        "kivi": kivi_result,
        "summary": {
            "fp16_cache_bytes_median": baseline_cache,
            "kivi_cache_bytes_median": kivi_cache,
            "cache_reduction_ratio": 1 - kivi_cache / baseline_cache,
            "decode_slowdown": kivi_decode / baseline_decode,
            "paired_nll_delta": kivi_result["quality"]["mean_nll"] - baseline_result["quality"]["mean_nll"],
            "top1_agreement_vs_targets": kivi_result["quality"]["top1_agreement"],
            "accounting_basis": "logical K+V cache tensors at prefill token position",
            "capacity_claim": "not a SwiftLLM reusable block-capacity claim; KIVI has a distinct contiguous layout and residual state",
        },
        "measurement_notes": {
            "performance_excludes_quality": True,
            "quality_scored_separately": True,
            "cache_position_is_explicit": True,
            "decode_peak_is_recorded": True,
            "release_before_next_method": True,
            "raw_source_artifacts": ["vendor/public/KIVI", "scripts/reproduce_kivi.py"],
        },
        "invocation": {"argv": sys.argv, "cwd": os.getcwd()},
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2))


def median(values):
    values = sorted(values)
    return values[len(values) // 2]


if __name__ == "__main__":
    main()
