"""Probe Kitty's public paged quantization and decode kernels on Llama GQA shapes.

Kitty HEAD currently ships a Qwen3 model integration, not a Llama integration.
This probe therefore exercises the upstream KittyCache/Triton attention kernels
with the local Llama 3.2 1B GQA dimensions (32 query heads, 8 KV heads, D=64).
It is explicitly a portability/synthetic-kernel reproduction, not an end-to-end
Llama result.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor/public/Kitty/src"))
from kitty.kvcache import get_kvcache_kitty  # noqa: E402
from kitty.kvcache.kernels.kitty_attention import kitty_attention_forward  # noqa: E402


def tensor_bytes(value) -> int:
    return int(value.numel() * value.element_size()) if isinstance(value, torch.Tensor) else 0


def kitty_storage_bytes(layer) -> int:
    names = (
        "KeyCache", "KeyCache_metadata", "ValueCache", "ValueCache_metadata",
        "PageTable_K", "PageTable_V", "Sink_Buffer_K", "Sink_Buffer_V",
        "Q_Buffer_K", "Q_Buffer_V", "Local_Buffer_V",
    )
    return sum(tensor_bytes(getattr(layer, name)) for name in names)


def median(values):
    values = sorted(values)
    return values[len(values) // 2]


def timed(fn, warmups: int, repeats: int, device: torch.device):
    for _ in range(warmups):
        fn()
    torch.cuda.synchronize(device)
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record(torch.cuda.current_stream(device))
        fn()
        stop.record(torch.cuda.current_stream(device))
        stop.synchronize()
        samples.append(float(start.elapsed_time(stop)))
    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--prompt-tokens", type=int, default=288)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.manual_seed(20260915)

    # Exact Llama 3.2 1B GQA dimensions. Kitty's fixed policy is K2/V2 with
    # 25% K channels promoted to 4 bit and 32 FP16 sink tokens.
    B, H_Q, H_KV, D = 1, 32, 8, 64
    config = SimpleNamespace(
        num_hidden_layers=1,
        head_dim=D,
        num_key_value_heads=H_KV,
        num_attention_heads=H_Q,
    )
    assert args.prompt_tokens > 32 + 128
    key = torch.randn(B, H_KV, args.prompt_tokens, D, device=device, dtype=torch.float16)
    value = torch.randn_like(key)
    new_key = torch.randn(B, H_KV, 1, D, device=device, dtype=torch.float16)
    new_value = torch.randn_like(new_key)
    query = torch.randn(B, H_Q, 1, D, device=device, dtype=torch.float16).contiguous()

    cache = get_kvcache_kitty(config, B, args.max_length)
    layer = cache.kv_cache[0]
    cache.update(key, value, 0)
    cache.quantize_prefill(0)
    cache.update(new_key, new_value, 0)
    assert layer.get_total_length() == args.prompt_tokens + 1
    kitty_output, _ = kitty_attention_forward(
        SimpleNamespace(num_attention_heads=H_Q, num_key_value_heads=H_KV),
        query,
        layer,
        D ** -0.5,
    )

    full_key = torch.cat([key, new_key], dim=2)
    full_value = torch.cat([value, new_value], dim=2)
    full_key = full_key.repeat_interleave(H_Q // H_KV, dim=1)
    full_value = full_value.repeat_interleave(H_Q // H_KV, dim=1)
    reference = torch.nn.functional.scaled_dot_product_attention(
        query, full_key, full_value, scale=D ** -0.5
    ).transpose(1, 2)
    error = (kitty_output.float() - reference.float()).abs()
    kitty_ms = timed(
        lambda: kitty_attention_forward(
            SimpleNamespace(num_attention_heads=H_Q, num_key_value_heads=H_KV),
            query, layer, D ** -0.5
        )[0], args.warmups, args.repeats, device
    )
    reference_ms = timed(
        lambda: torch.nn.functional.scaled_dot_product_attention(
            query, full_key, full_value, scale=D ** -0.5
        ), args.warmups, args.repeats, device
    )

    dense_cache_bytes = B * H_KV * args.max_length * D * 2 * 2
    result = {
        "method": "Kitty",
        "upstream_commit": __import__("os").popen(f"git -C {ROOT / 'vendor/public/Kitty'} rev-parse HEAD").read().strip(),
        "upstream_url": "https://github.com/Summer-Summer/Kitty",
        "status": "synthetic_gqa_kernel_reproduction_portability_adaptation",
        "device": torch.cuda.get_device_name(device),
        "cuda_capability": list(torch.cuda.get_device_capability(device)),
        "shape": {"batch": B, "query_heads": H_Q, "kv_heads": H_KV, "head_dim": D, "gqa_groups": H_Q // H_KV},
        "kitty_policy": {"low_bits": 2, "boosted_key_bits": 4, "boosted_key_fraction": 0.25, "value_bits": 2, "page_size": layer.PAGE_SIZE, "sink_tokens": layer.S},
        "max_length": args.max_length,
        "prompt_tokens_plus_decode": args.prompt_tokens + 1,
        "used_pages": {"key": layer.PageCount_K, "value": layer.PageCount_V},
        "storage": {
            "dense_fp16_capacity_bytes": dense_cache_bytes,
            "kitty_allocated_tensor_bytes": kitty_storage_bytes(layer),
            "capacity_reduction_ratio": 1 - kitty_storage_bytes(layer) / dense_cache_bytes,
            "key_payload_bytes_per_page": layer.bytes_per_page_K,
            "value_payload_bytes_per_page": layer.bytes_per_page_V,
            "key_metadata_bytes_per_page": tensor_bytes(layer.KeyCache_metadata[0]),
            "value_metadata_bytes_per_page": tensor_bytes(layer.ValueCache_metadata[0]),
            "includes_static_fp16_buffers": True,
        },
        "latency_ms": {"kitty": kitty_ms, "fp16_sdpa_reference": reference_ms},
        "summary": {
            "kitty_ms_median": median(kitty_ms),
            "fp16_sdpa_ms_median": median(reference_ms),
            "kitty_slowdown_vs_sdpa": median(kitty_ms) / median(reference_ms),
            "max_abs_error_vs_unquantized": float(error.max().item()),
            "mean_abs_error_vs_unquantized": float(error.mean().item()),
            "finite_output": bool(torch.isfinite(kitty_output).all()),
        },
        "invocation": " ".join(sys.argv),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
