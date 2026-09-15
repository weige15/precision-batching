"""Reproduce the public KIVI decode path on a local Llama checkpoint.

This adapter leaves vendor/public/KIVI untouched.  It only supplies compatibility
shims for the installed Transformers/flash-attention packages and records those
shims in the output artifact.  The KIVI model class, Triton packer, and CUDA GEMV
kernel remain the upstream implementation.
"""
from __future__ import annotations

import argparse
import gc
import importlib.machinery
import json
import os
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


def tensor_bytes(value) -> int:
    return int(value.numel() * value.element_size()) if isinstance(value, torch.Tensor) else 0


def cache_bytes(cache) -> int:
    if isinstance(cache, tuple):
        return sum(tensor_bytes(value) for layer in cache for value in layer)
    return sum(tensor_bytes(value) for key, value in zip(cache.key_cache, cache.value_cache))


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


def model_cache_bytes(cache) -> int:
    return cache_bytes(cache)


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
    # Import only after the compatibility shim has populated the old KIVI import surface.
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


def greedy_targets(model, ids: torch.Tensor, count: int) -> list[int]:
    tokens = ids.clone()
    result = []
    with torch.no_grad():
        output = model(tokens, use_cache=True)
        past = output.past_key_values
        next_token = output.logits[:, -1].argmax(-1)
        for _ in range(count):
            result.append(int(next_token.item()))
            output = model(next_token[:, None], past_key_values=past, use_cache=True)
            past = output.past_key_values
            next_token = output.logits[:, -1].argmax(-1)
    del output, past
    gc.collect()
    return result


def run_model(model, ids: torch.Tensor, targets: list[int], warmups: int, repeats: int):
    device = ids.device
    for _ in range(warmups):
        with torch.no_grad():
            output = model(ids, use_cache=True)
            past = output.past_key_values
            for token in targets:
                next_input = torch.full((ids.shape[0], 1), token, device=device, dtype=torch.long)
                output = model(next_input, past_key_values=past, use_cache=True)
                past = output.past_key_values
        del output, past
        gc.collect()
        torch.cuda.synchronize(device)

    prefill_ms = []
    decode_ms = []
    quality_nll = []
    quality_top1 = []
    cache_bytes_values = []
    allocated_delta = []
    peak_delta = []
    for _ in range(repeats):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        base_alloc = torch.cuda.memory_allocated(device)
        start = time.perf_counter()
        with torch.no_grad():
            output = model(ids, use_cache=True)
        torch.cuda.synchronize(device)
        prefill_ms.append((time.perf_counter() - start) * 1000)
        past = output.past_key_values
        cache_bytes_values.append(model_cache_bytes(past))
        # Keep only one token's logits for the teacher-forced comparison.  A
        # full [prompt, vocab] logits tensor would obscure KV-cache residency.
        logits = output.logits[:, -1].detach().clone()
        output = None
        gc.collect()
        torch.cuda.synchronize(device)
        after_prefill = torch.cuda.memory_allocated(device)
        prefill_peak = torch.cuda.max_memory_allocated(device)
        allocated_delta.append(after_prefill - base_alloc)
        peak_delta.append(prefill_peak - base_alloc)

        nlls = []
        top1 = 0
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        with torch.no_grad():
            for token in targets:
                target = torch.full((ids.shape[0],), token, device=device, dtype=torch.long)
                nlls.append(float(torch.nn.functional.cross_entropy(logits.float(), target).item()))
                top1 += float((logits.argmax(-1) == target).float().mean().item())
                next_input = torch.full((ids.shape[0], 1), token, device=device, dtype=torch.long)
                output = model(next_input, past_key_values=past, use_cache=True)
                past = output.past_key_values
                logits = output.logits[:, -1].detach().clone()
                output = None
        torch.cuda.synchronize(device)
        decode_ms.append((time.perf_counter() - start) * 1000)
        quality_nll.append(sum(nlls) / len(nlls))
        quality_top1.append(top1 / len(targets))

    return {
        "prefill_ms": prefill_ms,
        "decode_ms_for_tokens": decode_ms,
        "decode_ms_per_token": [x / len(targets) for x in decode_ms],
        "cache_bytes": cache_bytes_values,
        "allocated_delta_bytes": allocated_delta,
        "peak_delta_bytes": peak_delta,
        "paired_nll": quality_nll,
        "top1_agreement": quality_top1,
        "cache_summary": cache_shape_summary(past),
    }


def median(values):
    values = sorted(values)
    return values[len(values) // 2]


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
    torch.cuda.set_device(device)
    model_path = str(Path(args.model_path).resolve())
    prompt = "The quick brown fox jumps over the lazy dog. This is a reproducible KV cache benchmark."
    compat = install_compat_shims()

    # Baseline run supplies deterministic teacher-forced targets and the matched FP16 reference.
    baseline, tokenizer = hf_load(model_path, device)
    ids = prompt_ids(tokenizer, prompt, args.prompt_tokens, device)
    targets = greedy_targets(baseline, ids, args.decode_tokens)
    ids = ids.repeat(args.batch_size, 1)
    baseline_result = run_model(baseline, ids, targets, args.warmups, args.repeats)
    del baseline
    torch.cuda.empty_cache()

    kivi, kivi_tokenizer, config = kivi_load(
        model_path, device, args.bits, args.group_size, args.residual_length
    )
    kivi_ids = prompt_ids(kivi_tokenizer, prompt, args.prompt_tokens, device).repeat(args.batch_size, 1)
    kivi_result = run_model(kivi, kivi_ids, targets, args.warmups, args.repeats)
    result = {
        "method": "KIVI",
        "upstream_commit": None,
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
        "batch_size": args.batch_size,
        "prompt_tokens": args.prompt_tokens,
        "decode_tokens": args.decode_tokens,
        "targets": targets,
        "baseline_fp16": baseline_result,
        "kivi": kivi_result,
        "summary": {
            "fp16_cache_bytes_median": median(baseline_result["cache_bytes"]),
            "kivi_cache_bytes_median": median(kivi_result["cache_bytes"]),
            "cache_reduction_ratio": 1 - median(kivi_result["cache_bytes"]) / median(baseline_result["cache_bytes"]),
            "decode_slowdown": median(kivi_result["decode_ms_per_token"]) / median(baseline_result["decode_ms_per_token"]),
            "paired_nll_delta": median(kivi_result["paired_nll"]) - median(baseline_result["paired_nll"]),
            "top1_agreement_vs_targets": median(kivi_result["top1_agreement"]),
        },
        "invocation": " ".join(sys.argv),
    }
    # Replace the intentionally non-portable source expression with the exact clone commit.
    result["upstream_commit"] = os.popen(f"git -C {KIVI_ROOT} rev-parse HEAD").read().strip()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
