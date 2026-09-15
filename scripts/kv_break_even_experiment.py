#!/usr/bin/env python3
"""Matched queue/offload/INT8 pressure-action break-even study.

This is an offline action-cost experiment, not a scheduler.  It uses the
existing SwiftLLM dense FP16 swap extension for offload, the batched INT8 page
conversion and segmented Triton attention path for compression, and the
unchanged dense Triton paged-attention kernel as the FP16 datapath baseline.
The page microbenchmark uses one layer because it is the attention kernel's
unit of work; exact all-layer KV bytes are reported by multiplying the known
Llama layer count.  End-to-end checkpoint quality remains a separate paired
probe in kv_precision_experiment.py.
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
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor" / "swiftLLM"))
sys.path.insert(0, str(ROOT / "vendor" / "swiftLLM" / "csrc"))

import swiftllm_c  # noqa: E402
from swiftllm.worker.kv_cache import PagedKVCache, page_attention_for_layer  # noqa: E402
from swiftllm.worker.kernels.paged_attn import paged_attention  # noqa: E402


SHAPES = {
    "llama32_1b": {"num_layers": 16, "num_q_heads": 32, "num_kv_heads": 8, "head_dim": 64},
    "llama31_8b": {"num_layers": 32, "num_q_heads": 32, "num_kv_heads": 8, "head_dim": 128},
}
BLOCK_SIZE = 16
GROUP_SIZE = 128
FRACTIONS = (0.25, 0.5, 0.75)
HORIZONS = (1, 4, 8, 16, 32, 64)
SOURCE_FILES = (
    "scripts/kv_break_even_experiment.py",
    "scripts/kv_precision_experiment.py",
    "vendor/swiftLLM/swiftllm/worker/kv_cache.py",
    "vendor/swiftLLM/swiftllm/worker/kernels/segmented_paged_attn.py",
    "vendor/swiftLLM/swiftllm/worker/kernels/paged_attn.py",
    "vendor/swiftLLM/swiftllm/worker/model.py",
    "vendor/swiftLLM/csrc/src/block_swapping.cpp",
)
QUALITY_ARTIFACTS = {
    "llama32_1b": ROOT / "results/sensitivity/kv_precision_quality_batched_1b.json",
    "llama31_8b": ROOT / "results/sensitivity/kv_precision_quality_batched_8b.json",
}


def source_fingerprint() -> str:
    digest = hashlib.sha256()
    for relative in SOURCE_FILES:
        digest.update(relative.encode())
        digest.update((ROOT / relative).read_bytes())
    return digest.hexdigest()


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def make_shape_model(shape: dict[str, int]):
    return SimpleNamespace(
        num_layers=1,
        num_q_heads=shape["num_q_heads"],
        num_kv_heads=shape["num_kv_heads"],
        head_dim=shape["head_dim"],
    )


def make_engine(pages_per_seq: int):
    return SimpleNamespace(
        block_size=BLOCK_SIZE,
        max_blocks_per_seq=pages_per_seq,
    )


def page_bytes(shape: dict[str, int], fmt: str = "fp16") -> int:
    elements = BLOCK_SIZE * shape["num_kv_heads"] * shape["head_dim"]
    if fmt == "fp16":
        return 2 * elements * 2  # K and V, each FP16
    if fmt == "int8":
        scales = math.ceil(elements / GROUP_SIZE)
        return 2 * (elements + 2 * scales)  # K/V int8 payloads + FP16 scales
    raise ValueError(fmt)


def selected_blocks(total_pages: int, fraction: float) -> list[int]:
    return list(range(round(total_pages * fraction)))


def make_case(shape: dict[str, int], batch: int, context: int, device: torch.device, seed: int):
    pages_per_seq = math.ceil(context / BLOCK_SIZE)
    total_pages = batch * pages_per_seq
    cache = PagedKVCache(
        total_pages, 1, shape["num_kv_heads"], BLOCK_SIZE, shape["head_dim"],
        device, default_format="fp16", group_size=GROUP_SIZE,
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    # Keep the same FP16 source for dense attention, the oracle, conversion,
    # and the optimized kernel.  Each logical physical block is request-local.
    k_pages, v_pages = [], []
    for block_id in range(total_pages):
        k = torch.randn(cache.page_shape, generator=generator, device=device, dtype=torch.float16)
        v = torch.randn(cache.page_shape, generator=generator, device=device, dtype=torch.float16)
        k_pages.append(k)
        v_pages.append(v)
        cache.write(block_id, 0, k, v)
    block_table = torch.arange(total_pages, dtype=torch.int32, device=device).reshape(batch, pages_per_seq)
    seq_ids = torch.arange(batch, dtype=torch.int32, device=device)
    lengths = torch.full((batch,), context, dtype=torch.int32, device=device)
    q = torch.randn(
        (batch, shape["num_q_heads"], shape["head_dim"]),
        generator=generator, device=device, dtype=torch.float16,
    )
    out = torch.empty((batch, shape["num_q_heads"] * shape["head_dim"]), device=device, dtype=torch.float16)
    dense_k = torch.stack(k_pages).reshape(total_pages, 1, shape["num_kv_heads"], BLOCK_SIZE, shape["head_dim"])
    dense_v = torch.stack(v_pages).reshape_as(dense_k)
    return {
        "cache": cache,
        "shape": shape,
        "batch": batch,
        "context": context,
        "pages_per_seq": pages_per_seq,
        "total_pages": total_pages,
        "block_table": block_table,
        "seq_ids": seq_ids,
        "lengths": lengths,
        "q": q,
        "out": out,
        "dense_k": dense_k,
        "dense_v": dense_v,
        "model": make_shape_model(shape),
        "engine": make_engine(pages_per_seq),
    }


def dense_attention(case: dict[str, object]) -> None:
    infer = SimpleNamespace(
        seq_block_size=2048,
        num_seq_blocks=math.ceil(int(case["context"]) / 2048),
        num_decoding_seqs=int(case["batch"]),
        num_prefill_seqs=0,
        decoding_seq_lens=case["lengths"],
        seq_ids=case["seq_ids"],
        softmax_scale=case["shape"]["head_dim"] ** -0.5,
    )
    paged_attention(
        case["q"], case["dense_k"], case["dense_v"], case["block_table"],
        case["model"], case["engine"], infer, 0, case["out"],
    )


def time_attention(case: dict[str, object], kind: str, iterations: int, repetitions: int) -> dict[str, object]:
    fn = dense_attention if kind == "dense_fp16" else lambda c: c["cache"].optimized_attention(
        c["q"], c["block_table"], c["seq_ids"], c["lengths"], c["model"], c["engine"], 0, c["out"]
    )
    fn(case)
    sync(case["q"].device)
    wall_repeats, gpu_repeats = [], []
    for _ in range(repetitions):
        start_gpu = torch.cuda.Event(enable_timing=True)
        stop_gpu = torch.cuda.Event(enable_timing=True)
        start_wall = time.perf_counter()
        start_gpu.record()
        for _ in range(iterations):
            fn(case)
        stop_gpu.record()
        stop_gpu.synchronize()
        wall_repeats.append((time.perf_counter() - start_wall) * 1000)
        gpu_repeats.append(start_gpu.elapsed_time(stop_gpu))
    return {
        "iterations": iterations,
        "repetitions": repetitions,
        "wall_ms_repeats": wall_repeats,
        "gpu_ms_repeats": gpu_repeats,
        "wall_ms_median": statistics.median(wall_repeats),
        "gpu_ms_median": statistics.median(gpu_repeats),
        "per_token_wall_ms": statistics.median(wall_repeats) / iterations,
        "per_token_gpu_ms": statistics.median(gpu_repeats) / iterations,
        "output_checksum": float(case["out"].float().sum().item()),
    }


def time_swap(shape: dict[str, int], total_pages: int, selected: list[int], device: torch.device, seed: int) -> dict[str, object]:
    # This is the exact pinned C++ swap_blocks API and dense FP16 layout, with
    # all model layers present.  The preallocated GPU tensor means offload
    # frees reusable KV capacity, not a physical CUDA allocation.
    layers = shape["num_layers"]
    tensor_shape = (total_pages, layers, shape["num_kv_heads"], BLOCK_SIZE, shape["head_dim"])
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    k_cache = torch.randn(tensor_shape, generator=generator, device=device, dtype=torch.float16)
    v_cache = torch.randn(tensor_shape, generator=generator, device=device, dtype=torch.float16)
    k_swap = torch.zeros(tensor_shape, dtype=torch.float16, device="cpu")
    v_swap = torch.zeros_like(k_swap)
    src_gpu = selected
    dst_cpu = list(range(len(selected)))
    bytes_moved = len(selected) * page_bytes(shape, "fp16") * layers
    before = torch.cuda.memory_allocated(device)
    sync(device)
    start = time.perf_counter()
    start_event = torch.cuda.Event(enable_timing=True)
    stop_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    swiftllm_c.swap_blocks(src_gpu, dst_cpu, False, k_cache, v_cache, k_swap, v_swap)
    stop_event.record()
    sync(device)
    out_ms = (time.perf_counter() - start) * 1000
    out_gpu_ms = start_event.elapsed_time(stop_event)
    start = time.perf_counter()
    start_event.record()
    swiftllm_c.swap_blocks(dst_cpu, src_gpu, True, k_cache, v_cache, k_swap, v_swap)
    stop_event.record()
    sync(device)
    in_ms = (time.perf_counter() - start) * 1000
    in_gpu_ms = start_event.elapsed_time(stop_event)
    after = torch.cuda.memory_allocated(device)
    result = {
        "selected_pages": len(selected),
        "all_layer_page_bytes": page_bytes(shape, "fp16") * layers,
        "bytes_reclaimed_as_reusable_capacity": bytes_moved,
        "swap_out_wall_ms": out_ms,
        "swap_in_wall_ms": in_ms,
        "swap_out_gpu_ms": out_gpu_ms,
        "swap_in_gpu_ms": in_gpu_ms,
        "transition_wall_ms": out_ms + in_ms,
        "transition_gpu_ms": out_gpu_ms + in_gpu_ms,
        "physical_hbm_delta_bytes": before - after,
        "allocator_is_preallocated": True,
        "mechanism": "swiftllm_c.swap_blocks",
    }
    del k_cache, v_cache, k_swap, v_swap
    gc.collect()
    torch.cuda.empty_cache()
    return result


def action_rows(case: dict[str, object], fraction: float, iterations: int, repetitions: int) -> dict[str, object]:
    shape = case["shape"]
    batch = int(case["batch"])
    context = int(case["context"])
    total_pages = int(case["total_pages"])
    selected = selected_blocks(total_pages, fraction)
    cache: PagedKVCache = case["cache"]
    keys = [(block_id, 0) for block_id in selected]
    baseline = time_attention(case, "dense_fp16", iterations, repetitions)
    dense_n = baseline["per_token_wall_ms"]
    # The optimized path is evaluated on a fresh FP16 cache so each fraction
    # has the same source state and conversion transition.
    if keys:
        conversion_allocated_before = torch.cuda.memory_allocated(case["q"].device)
        converted = cache.demote_pages_batch(keys, "int8")
        sync(case["q"].device)
        conversion_allocated_after = torch.cuda.memory_allocated(case["q"].device)
    else:
        converted = None
        conversion_allocated_before = conversion_allocated_after = torch.cuda.memory_allocated(case["q"].device)
    fast = time_attention(case, "optimized_int8_mixed", iterations, repetitions)
    oracle_out = torch.empty_like(case["out"])
    page_attention_for_layer(
        case["q"], cache, case["block_table"], case["seq_ids"], case["lengths"],
        case["model"], case["engine"], 0, oracle_out,
    )
    sync(case["q"].device)
    max_error = float((oracle_out - case["out"]).abs().max().item())
    conversion = {
        "pages": len(keys),
        "elapsed_ms": converted.elapsed_ms if converted else 0.0,
        "before_bytes": converted.before_bytes if converted else 0,
        "after_bytes": converted.after_bytes if converted else 0,
        "reclaimed_bytes": converted.reclaimed_bytes if converted else 0,
        "temporary_bytes": converted.temporary_bytes if converted else 0,
        "batched_gpu_operation": True,
        "has_unreclaimed_shadow": cache.has_unreclaimed_shadow(),
        "gpu_memory_allocated_before_bytes": conversion_allocated_before,
        "gpu_memory_allocated_after_bytes": conversion_allocated_after,
        "gpu_memory_allocated_delta_bytes": conversion_allocated_before - conversion_allocated_after,
    }
    if converted:
        restoration_allocated_before = torch.cuda.memory_allocated(case["q"].device)
        restored = cache.promote_pages_batch(keys, "fp16")
        sync(case["q"].device)
        restoration_allocated_after = torch.cuda.memory_allocated(case["q"].device)
        restoration = {
            "pages": len(keys),
            "elapsed_ms": restored.elapsed_ms,
            "before_bytes": restored.before_bytes,
            "after_bytes": restored.after_bytes,
            "reclaimed_bytes": restored.reclaimed_bytes,
            "temporary_bytes": restored.temporary_bytes,
            "batched_gpu_operation": True,
            "gpu_memory_allocated_before_bytes": restoration_allocated_before,
            "gpu_memory_allocated_after_bytes": restoration_allocated_after,
            "gpu_memory_allocated_delta_bytes": restoration_allocated_after - restoration_allocated_before,
        }
    else:
        restoration = {"pages": 0, "elapsed_ms": 0.0, "before_bytes": 0, "after_bytes": 0, "reclaimed_bytes": 0, "temporary_bytes": 0, "batched_gpu_operation": True, "gpu_memory_allocated_before_bytes": conversion_allocated_after, "gpu_memory_allocated_after_bytes": conversion_allocated_after, "gpu_memory_allocated_delta_bytes": 0}
    layer_count = shape["num_layers"]
    dense_page = page_bytes(shape, "fp16")
    int8_page = page_bytes(shape, "int8")
    all_layer_reclaim = conversion["reclaimed_bytes"] * layer_count
    dense_all_layer_page = dense_page * layer_count
    # Offload is page-granular. Use the smallest number of dense pages that
    # meets the same target deficit, and expose the unavoidable <= one-page
    # granularity overshoot in the raw record.
    offload_page_count = max(1, math.ceil(all_layer_reclaim / dense_all_layer_page)) if selected else 0
    offload_selected = selected[:offload_page_count]
    swap = time_swap(shape, total_pages, offload_selected, case["q"].device, 77 + total_pages)
    rows = []
    for horizon in HORIZONS:
        # Queue means the pressure episode leaves a new request unadmitted for
        # H decode iterations. Offload admits it after moving an existing KV
        # state; compression keeps it live but pays persistent mixed attention.
        queue_cost = horizon * dense_n
        offload_cost = swap["transition_wall_ms"]
        compression_cost = conversion["elapsed_ms"] + restoration["elapsed_ms"] + horizon * (fast["per_token_wall_ms"] - dense_n)
        costs = {"queue": queue_cost, "cpu_offload": offload_cost, "int8_compression": compression_cost}
        winner = min(costs, key=costs.get)
        rows.append({
            "horizon_decode_iterations": horizon,
            "queue_cost_ms": queue_cost,
            "offload_cost_ms": offload_cost,
            "compression_cost_ms": compression_cost,
            "winner": winner,
            "queue_wait_ms_if_not_admitted": queue_cost,
            "offload_transition_ms": offload_cost,
            "compression_transition_ms": conversion["elapsed_ms"] + restoration["elapsed_ms"],
            "compression_persistent_delta_ms_per_token": fast["per_token_wall_ms"] - dense_n,
        })
    return {
        "model_family": "llama32_1b" if shape["head_dim"] == 64 else "llama31_8b",
        "shape": shape,
        "batch": batch,
        "context_tokens": context,
        "compressed_fraction": fraction,
        "resident_pages": total_pages,
        "selected_pages": len(selected),
        "offload_selected_pages": len(offload_selected),
        "memory": {
            "dense_fp16_page_bytes_one_layer": dense_page,
            "int8_page_bytes_one_layer": int8_page,
            "reclaimed_bytes_one_layer": conversion["reclaimed_bytes"],
            "reclaimed_bytes_all_layers": all_layer_reclaim,
            "reclaimed_mib_all_layers": all_layer_reclaim / (1024 ** 2),
            "exact_all_layer_scaling": True,
            "target_deficit_bytes": all_layer_reclaim,
            "capacity_bytes_avoided_or_reclaimed": all_layer_reclaim,
            "physical_hbm_delta_offload_bytes": swap["physical_hbm_delta_bytes"],
            "offload_page_granularity_overshoot_bytes": swap["bytes_reclaimed_as_reusable_capacity"] - all_layer_reclaim,
        },
        "baseline_dense_fp16": baseline,
        "optimized_mixed_int8": fast,
        "oracle": {"max_abs_error_vs_optimized": max_error, "oracle": "page_attention_for_layer"},
        "conversion": conversion,
        "restoration": restoration,
        "cpu_offload": swap,
        "actions": {
            "queue": {
                "capacity_bytes_avoided": all_layer_reclaim,
                "transition_latency_ms": 0.0,
                "persistent_per_token_ms": dense_n,
                "restoration_latency_ms": 0.0,
                "quality_change": 0.0,
            },
            "cpu_offload": {
                "capacity_bytes_reclaimed": swap["bytes_reclaimed_as_reusable_capacity"],
                "target_deficit_bytes": all_layer_reclaim,
                "transition_latency_ms": swap["transition_wall_ms"],
                "persistent_per_token_ms": dense_n,
                "restoration_latency_ms": swap["swap_in_wall_ms"],
                "quality_change": 0.0,
            },
            "int8_compression": {
                "capacity_bytes_reclaimed": all_layer_reclaim,
                "transition_latency_ms": conversion["elapsed_ms"],
                "persistent_per_token_ms": fast["per_token_wall_ms"],
                "restoration_latency_ms": restoration["elapsed_ms"],
                "quality_change": None,
            },
        },
        "quality_delta": {"queue": 0.0, "cpu_offload": 0.0, "int8_compression": None, "quality_source": "paired checkpoint probe, not inferred from attention microbenchmark"},
        "break_even": rows,
    }


def load_quality_probes() -> dict[str, dict[float, dict[str, object]]]:
    probes: dict[str, dict[float, dict[str, object]]] = {}
    for family, path in QUALITY_ARTIFACTS.items():
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        variants = data.get("quality", [{}])[0].get("variants", [])
        family_probes = {}
        for variant in variants:
            name = variant.get("name", "")
            if name.startswith("dynamic_old_int8_") and name.endswith("_both"):
                percent = int(name.split("_")[3])
                family_probes[percent / 100] = {
                    "paired_nll_delta_mean": variant["quality"]["paired_nll_delta_mean"],
                    "top1_agreement": variant["quality"]["forced_prefix_top1_agreement"],
                    "logit_kl_mean": variant["quality"]["logit_kl_mean"],
                    "artifact": str(path),
                    "variant": name,
                }
        if family_probes:
            probes[family] = family_probes
    return probes


def run(args: argparse.Namespace) -> dict[str, object]:
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("the break-even study requires a CUDA RTX 3090")
    result = {
        "schema": "kv-memory-pressure-break-even-v1",
        "provenance": {
            "branch": subprocess.check_output(["git", "-C", str(ROOT), "branch", "--show-current"], text=True).strip(),
            "git_head": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
            "swiftllm_upstream_commit": (ROOT / "vendor/swiftLLM/UPSTREAM_COMMIT").read_text().strip(),
            "source_sha256": source_fingerprint(),
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device),
        },
        "scope": {
            "actions": ["queue_not_admit", "existing_swiftllm_fp16_cpu_offload", "live_int8_compression"],
            "scheduler_implemented": False,
            "scheduler_changed": False,
            "compression_format": "INT8 symmetric per-group max-abs, FP16 scales, group_size=128",
            "attention_path": "format-specialized segmented Triton; page_attention_for_layer retained as oracle",
            "one_layer_attention_with_exact_all_layer_byte_scaling": True,
            "cost_model": "transition cost + horizon * persistent per-token delta",
        },
        "quality_probes": load_quality_probes(),
        "cells": [],
    }
    families = (args.model_family,) if args.model_family != "both" else tuple(SHAPES)
    for family in families:
        shape = SHAPES[family]
        for batch in args.batches:
            for context in args.contexts:
                for fraction in FRACTIONS:
                    case = make_case(shape, batch, context, device, args.seed + batch * 10000 + context)
                    cell = action_rows(case, fraction, args.attention_iterations, args.attention_repetitions)
                    probe = result["quality_probes"].get(family, {}).get(fraction)
                    if probe is not None:
                        cell["quality_delta"]["int8_compression"] = probe["paired_nll_delta_mean"]
                        cell["actions"]["int8_compression"]["quality_change"] = probe["paired_nll_delta_mean"]
                        cell["quality_delta"]["top1_agreement"] = probe["top1_agreement"]
                        cell["quality_delta"]["logit_kl_mean"] = probe["logit_kl_mean"]
                        cell["quality_delta"]["quality_source"] = probe
                    result["cells"].append(cell)
                    del case
                    gc.collect()
                    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-family", choices=("llama32_1b", "llama31_8b", "both"), default="both")
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--contexts", type=int, nargs="+", default=[128, 512, 2048])
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--attention-iterations", type=int, default=3)
    parser.add_argument("--attention-repetitions", type=int, default=3)
    args = parser.parse_args()
    output = run(args)
    output["invocation"] = {"argv": sys.argv, "cwd": os.getcwd()}
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"output": str(path), "cells": len(output["cells"])}))


if __name__ == "__main__":
    main()
