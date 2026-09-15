#!/usr/bin/env python3
"""Measure live page demotion and mixed-format decode attention.

The microbenchmark uses the actual SwiftLLM research page store and records
raw per-trial timings, tensor-byte accounting, and stream-overlap checks.  The
optional quality run uses the pinned SwiftLLM model path and compares logits
under paired forced-generation prefixes.  No scheduler or policy is changed.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import gc
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

from swiftllm.engine_config import EngineConfig  # noqa: E402
from swiftllm.worker.kv_cache import (  # noqa: E402
    PagedKVCache,
    page_attention_for_layer,
)
from swiftllm.worker.model import LlamaModel  # noqa: E402


SHAPES = {
    "llama32_1b": {"num_layers": 16, "num_q_heads": 32, "num_kv_heads": 8, "head_dim": 64},
    "llama31_8b": {"num_layers": 32, "num_q_heads": 32, "num_kv_heads": 8, "head_dim": 128},
}
FRACTIONS = (0.0, 0.25, 0.5, 0.75, 1.0)
TARGETS = ("int8", "int4")
SOURCE_FILES = (
    "vendor/swiftLLM/swiftllm/worker/kv_cache.py",
    "vendor/swiftLLM/swiftllm/worker/kernels/kvcache_mgmt.py",
    "vendor/swiftLLM/swiftllm/worker/kernels/segmented_paged_attn.py",
    "vendor/swiftLLM/swiftllm/worker/layers/transformer_layer.py",
    "vendor/swiftLLM/swiftllm/worker/layers/post_layer.py",
    "vendor/swiftLLM/swiftllm/worker/model.py",
    "vendor/swiftLLM/swiftllm/engine_config.py",
    "scripts/kv_precision_experiment.py",
)


def source_fingerprint() -> str:
    digest = hashlib.sha256()
    for relative in SOURCE_FILES:
        digest.update(relative.encode("utf-8"))
        digest.update((ROOT / relative).read_bytes())
    return digest.hexdigest()


def device_arg(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the requested device")
    return device


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def make_cache(shape: dict[str, int], batch: int, context: int, device: torch.device, default_format="fp16"):
    block_size = 16
    pages_per_seq = math.ceil(context / block_size)
    total_pages = batch * pages_per_seq
    cache = PagedKVCache(
        num_blocks=total_pages,
        num_layers=shape["num_layers"],
        num_kv_heads=shape["num_kv_heads"],
        block_size=block_size,
        head_dim=shape["head_dim"],
        device=device,
        default_format=default_format,
        group_size=128,
    )
    return cache, pages_per_seq, total_pages


def populate(cache: PagedKVCache, batch: int, pages_per_seq: int, seed: int) -> None:
    generator = torch.Generator(device=cache.device)
    generator.manual_seed(seed)
    for physical_id in range(batch * pages_per_seq):
        k = torch.randn(cache.page_shape, generator=generator, device=cache.device, dtype=torch.float16)
        v = torch.randn(cache.page_shape, generator=generator, device=cache.device, dtype=torch.float16)
        cache.write(physical_id, 0, k, v)
    sync(cache.device)


def attention_args(shape: dict[str, int], batch: int, context: int, pages_per_seq: int, device: torch.device):
    model = SimpleNamespace(
        num_q_heads=shape["num_q_heads"],
        num_kv_heads=shape["num_kv_heads"],
        head_dim=shape["head_dim"],
    )
    engine = SimpleNamespace(block_size=16)
    block_table = torch.arange(batch * pages_per_seq, dtype=torch.int32, device=device).reshape(batch, pages_per_seq)
    seq_ids = torch.arange(batch, dtype=torch.int32, device=device)
    lengths = torch.full((batch,), context, dtype=torch.int32, device=device)
    generator = torch.Generator(device=device)
    generator.manual_seed(991 + batch + context + shape["head_dim"])
    q = torch.randn((batch, shape["num_q_heads"], shape["head_dim"]), generator=generator, device=device, dtype=torch.float16)
    out = torch.empty((batch, shape["num_q_heads"] * shape["head_dim"]), device=device, dtype=torch.float16)
    return model, engine, block_table, seq_ids, lengths, q, out


def attention_time(
    cache: PagedKVCache,
    shape: dict[str, int],
    batch: int,
    context: int,
    pages_per_seq: int,
    device: torch.device,
    iterations: int = 3,
    repetitions: int = 3,
) -> dict[str, object]:
    args = attention_args(shape, batch, context, pages_per_seq, device)
    model, engine, block_table, seq_ids, lengths, q, out = args
    for _ in range(2):
        page_attention_for_layer(q, cache, block_table, seq_ids, lengths, model, engine, 0, out)
    sync(device)
    replicate_ms = []
    for _ in range(repetitions):
        if device.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            stop = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iterations):
                page_attention_for_layer(q, cache, block_table, seq_ids, lengths, model, engine, 0, out)
            stop.record()
            stop.synchronize()
            replicate_ms.append(start.elapsed_time(stop))
        else:
            start_time = time.perf_counter()
            for _ in range(iterations):
                page_attention_for_layer(q, cache, block_table, seq_ids, lengths, model, engine, 0, out)
            replicate_ms.append((time.perf_counter() - start_time) * 1000)
    elapsed_ms = statistics.median(replicate_ms)
    pages_read = batch * pages_per_seq
    bytes_read = sum(cache.page(physical_id, 0).allocated_bytes for physical_id in range(pages_read))
    seconds = elapsed_ms / 1000
    return {
        "iterations": iterations,
        "repetitions": repetitions,
        "replicate_elapsed_ms": replicate_ms,
        "elapsed_ms_median": elapsed_ms,
        "per_decode_ms_median": elapsed_ms / iterations,
        "stored_payload_bytes_processed_per_iteration": bytes_read,
        "stored_payload_gib_per_s_proxy": (bytes_read * iterations / seconds / (1024 ** 3)) if seconds else None,
        "output_checksum": float(out.float().sum().item()),
    }


def selected_blocks(batch: int, pages_per_seq: int, fraction: float, recency: str = "old") -> list[int]:
    count = round(batch * pages_per_seq * fraction)
    blocks = list(range(batch * pages_per_seq))
    return blocks[:count] if recency == "old" else blocks[-count:] if count else []


def conversion_trial(shape: dict[str, int], batch: int, context: int, target: str, fraction: float, device: torch.device, seed: int) -> dict[str, object]:
    cache, pages_per_seq, total_pages = make_cache(shape, batch, context, device)
    populate(cache, batch, pages_per_seq, seed)
    before_logical = cache.logical_bytes()
    before_allocated = torch.cuda.memory_allocated(device) if device.type == "cuda" else before_logical
    keys = [(block_id, 0) for block_id in selected_blocks(batch, pages_per_seq, fraction)]
    raw_results = []
    peak_overhead = 0
    for index, (block_id, layer_id) in enumerate(keys):
        if device.type == "cuda" and index == 0:
            torch.cuda.reset_peak_memory_stats(device)
            before_one = torch.cuda.memory_allocated(device)
        result = cache.demote_page(block_id, layer_id, target)
        raw_results.append({
            "block_id": block_id,
            "layer_id": layer_id,
            "before_bytes": result.before_bytes,
            "after_bytes": result.after_bytes,
            "reclaimed_bytes": result.before_bytes - result.after_bytes,
            "elapsed_ms": result.elapsed_ms,
            "temporary_bytes_logical": result.temporary_bytes,
        })
        if device.type == "cuda" and index == 0:
            sync(device)
            peak_overhead = torch.cuda.max_memory_allocated(device) - before_one
    sync(device)
    after_allocated = torch.cuda.memory_allocated(device) if device.type == "cuda" else cache.logical_bytes()
    after_logical = cache.logical_bytes()
    reclaimed = before_logical - after_logical
    elapsed_ms = sum(row["elapsed_ms"] for row in raw_results)
    return {
        "target_format": target,
        "fraction": fraction,
        "batch": batch,
        "context_tokens": context,
        "resident_pages": total_pages,
        "converted_pages": len(keys),
        "before_logical_bytes": before_logical,
        "after_logical_bytes": after_logical,
        "reclaimed_bytes": reclaimed,
        "reclaimed_mib": reclaimed / (1024 ** 2),
        "before_torch_memory_allocated": before_allocated,
        "after_torch_memory_allocated": after_allocated,
        "torch_allocated_delta": before_allocated - after_allocated,
        "peak_conversion_overhead_bytes": peak_overhead,
        "total_conversion_ms": elapsed_ms,
        "latency_ms_per_page": elapsed_ms / len(keys) if keys else 0.0,
        "latency_ms_per_reclaimed_mib": elapsed_ms / (reclaimed / (1024 ** 2)) if reclaimed else None,
        "reclamation_mib_per_s": (reclaimed / (1024 ** 2)) / (elapsed_ms / 1000) if elapsed_ms else None,
        "has_unreclaimed_shadow": cache.has_unreclaimed_shadow(),
        "metadata_counts": cache.metadata_counts(),
        "raw_page_conversions": raw_results,
    }


def optimized_attention_time(
    cache: PagedKVCache,
    shape: dict[str, int],
    batch: int,
    context: int,
    pages_per_seq: int,
    device: torch.device,
    iterations: int = 10,
    repetitions: int = 3,
) -> dict[str, object]:
    model, engine, block_table, seq_ids, lengths, q, out = attention_args(shape, batch, context, pages_per_seq, device)
    # Compile and populate the packed arenas before recording steady-state
    # timings.  The first call is reported separately as warm-up overhead.
    cache.optimized_attention(q, block_table, seq_ids, lengths, model, engine, 0, out)
    sync(device)
    replicates = []
    for _ in range(repetitions):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            cache.optimized_attention(q, block_table, seq_ids, lengths, model, engine, 0, out)
        stop.record()
        stop.synchronize()
        replicates.append(start.elapsed_time(stop))
    return {
        "iterations": iterations,
        "repetitions": repetitions,
        "replicate_elapsed_ms": replicates,
        "elapsed_ms_median": statistics.median(replicates),
        "per_decode_ms_median": statistics.median(replicates) / iterations,
        "output_checksum": float(out.float().sum().item()),
    }


def optimized_smoke(shape: dict[str, int], device: torch.device, seed: int) -> dict[str, object]:
    batch, context = 4, 256
    cache, pages_per_seq, total_pages = make_cache(shape, batch, context, device)
    populate(cache, batch, pages_per_seq, seed)
    keys = [(block_id, 0) for block_id in selected_blocks(batch, pages_per_seq, 0.5)]
    batch_result = cache.demote_pages_batch(keys, "int8")
    model, engine, block_table, seq_ids, lengths, q, oracle_out = attention_args(shape, batch, context, pages_per_seq, device)
    optimized_out = torch.empty_like(oracle_out)
    page_attention_for_layer(q, cache, block_table, seq_ids, lengths, model, engine, 0, oracle_out)
    sync(device)
    kernel = optimized_attention_time(cache, shape, batch, context, pages_per_seq, device)
    cache.optimized_attention(q, block_table, seq_ids, lengths, model, engine, 0, optimized_out)
    sync(device)
    return {
        "model_shape": shape,
        "batch": batch,
        "context_tokens": context,
        "resident_pages": total_pages,
        "converted_pages": len(keys),
        "batch_conversion": {
            "target_format": batch_result.target_format,
            "pages": len(batch_result.page_keys),
            "before_bytes": batch_result.before_bytes,
            "after_bytes": batch_result.after_bytes,
            "reclaimed_bytes": batch_result.reclaimed_bytes,
            "elapsed_ms": batch_result.elapsed_ms,
            "latency_ms_per_page": batch_result.elapsed_ms / len(keys),
            "temporary_bytes": batch_result.temporary_bytes,
            "has_unreclaimed_shadow": cache.has_unreclaimed_shadow(),
            "per_page": [dataclasses.asdict(item) for item in batch_result.per_page],
        },
        "attention": kernel,
        "correctness": {
            "max_abs_error_vs_reference": float((oracle_out - optimized_out).abs().max().item()),
            "mean_abs_error_vs_reference": float((oracle_out - optimized_out).abs().mean().item()),
            "optimized_has_nan": bool(torch.isnan(optimized_out).any().item()),
        },
    }


def mixed_attention_trial(shape: dict[str, int], batch: int, context: int, target: str, fraction: float, device: torch.device, seed: int, iterations: int, repetitions: int) -> dict[str, object]:
    cache, pages_per_seq, total_pages = make_cache(shape, batch, context, device)
    populate(cache, batch, pages_per_seq, seed)
    keys = selected_blocks(batch, pages_per_seq, fraction)
    batch_conversion = None
    if target == "int8" and keys:
        batch_result = cache.demote_pages_batch([(block_id, 0) for block_id in keys], "int8")
        batch_conversion = {
            "pages": len(batch_result.page_keys),
            "elapsed_ms": batch_result.elapsed_ms,
            "before_bytes": batch_result.before_bytes,
            "after_bytes": batch_result.after_bytes,
            "reclaimed_bytes": batch_result.reclaimed_bytes,
            "has_unreclaimed_shadow": cache.has_unreclaimed_shadow(),
        }
    else:
        for block_id in keys:
            cache.demote_page(block_id, 0, target)
    result = attention_time(cache, shape, batch, context, pages_per_seq, device, iterations, repetitions)
    return {
        "kind": "mixed_dynamic",
        "target_format": target,
        "compressed_fraction": fraction,
        "batch": batch,
        "context_tokens": context,
        "resident_pages": total_pages,
        "storage": cache.storage_summary(),
        "batch_conversion": batch_conversion,
        "attention": result,
    }


def static_attention_trial(shape: dict[str, int], batch: int, context: int, target: str, device: torch.device, seed: int, iterations: int, repetitions: int) -> dict[str, object]:
    cache, pages_per_seq, total_pages = make_cache(shape, batch, context, device, default_format=target)
    populate(cache, batch, pages_per_seq, seed)
    return {
        "kind": "static_compressed",
        "target_format": target,
        "compressed_fraction": 1.0,
        "batch": batch,
        "context_tokens": context,
        "resident_pages": total_pages,
        "storage": cache.storage_summary(),
        "attention": attention_time(cache, shape, batch, context, pages_per_seq, device, iterations, repetitions),
    }


def fp16_attention_trial(shape: dict[str, int], batch: int, context: int, device: torch.device, seed: int, iterations: int, repetitions: int) -> dict[str, object]:
    cache, pages_per_seq, total_pages = make_cache(shape, batch, context, device)
    populate(cache, batch, pages_per_seq, seed)
    return {
        "kind": "unchanged_fp16_page_reference",
        "target_format": "fp16",
        "compressed_fraction": 0.0,
        "batch": batch,
        "context_tokens": context,
        "resident_pages": total_pages,
        "storage": cache.storage_summary(),
        "attention": attention_time(cache, shape, batch, context, pages_per_seq, device, iterations, repetitions),
    }


def overlap_trial(shape: dict[str, int], target: str, device: torch.device, seed: int) -> dict[str, object]:
    cache, _, _ = make_cache(shape, 1, 16, device)
    populate(cache, 1, 1, seed)
    serial = cache.demote_page(0, 0, target)
    serial_k, serial_v = cache.read(0, 0)
    del cache
    if device.type != "cuda":
        return {"target_format": target, "supported": False, "reason": "CUDA stream overlap requires CUDA"}

    cache, _, _ = make_cache(shape, 1, 16, device)
    populate(cache, 1, 1, seed)
    conversion_stream = torch.cuda.Stream(device=device)
    work_a = torch.randn((2048, 2048), device=device, dtype=torch.float16)
    work_b = torch.randn((2048, 2048), device=device, dtype=torch.float16)
    work_repeats = 8
    _ = torch.mm(work_a, work_b)
    sync(device)
    work_start = time.perf_counter()
    for _ in range(work_repeats):
        _ = torch.mm(work_a, work_b)
    sync(device)
    work_alone_ms = (time.perf_counter() - work_start) * 1000
    start = time.perf_counter()
    cache.demote_page_async(0, 0, target, stream=conversion_stream)
    pending_before_consumer = cache.has_unreclaimed_shadow()
    for _ in range(work_repeats):
        _ = torch.mm(work_a, work_b)
    # This is intentionally before any global synchronization.  read() must
    # wait for the conversion event before decoding the published payload.
    mixed_k, mixed_v = cache.read(0, 0)
    pending_after_consumer = cache.has_unreclaimed_shadow()
    sync(device)
    cache.synchronize()
    overlap_ms = (time.perf_counter() - start) * 1000
    sync(device)
    max_error = max((mixed_k - serial_k).abs().max().item(), (mixed_v - serial_v).abs().max().item())
    return {
        "target_format": target,
        "supported": True,
        "serial_conversion_ms": serial.elapsed_ms,
        "work_alone_ms": work_alone_ms,
        "serial_conversion_plus_work_ms": serial.elapsed_ms + work_alone_ms,
        "overlap_wall_ms": overlap_ms,
        "overlap_hidden_fraction": max(0.0, (serial.elapsed_ms + work_alone_ms - overlap_ms) / (serial.elapsed_ms + work_alone_ms)),
        "concurrent_work": f"2048x2048 FP16 matmul x {work_repeats} on default stream",
        "max_error_after_event_ordering": max_error,
        "pending_before_consumer_read": pending_before_consumer,
        "pending_after_consumer_read": pending_after_consumer,
        "has_unreclaimed_shadow_after_sync": cache.has_unreclaimed_shadow(),
        "overlap_not_a_speedup_claim": True,
    }


def run_microbench(args: argparse.Namespace) -> dict[str, object]:
    device = device_arg(args.device)
    result: dict[str, object] = {
        "schema": "live-kv-precision-v1",
        "provenance": {
            "branch": subprocess.check_output(["git", "-C", str(ROOT), "branch", "--show-current"], text=True).strip(),
            "git_head": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
            "swiftllm_upstream_commit": (ROOT / "vendor/swiftLLM/UPSTREAM_COMMIT").read_text().strip(),
            "research_checkout": str(ROOT / "vendor/swiftLLM"),
            "source_files": list(SOURCE_FILES),
            "source_sha256": source_fingerprint(),
        },
        "scope": {
            "mechanism": "explicit page records with FP16/INT8/INT4 symmetric per-group quantization",
            "group_size": 128,
            "block_size": 16,
            "device": str(device),
            "seed": args.seed,
            "quantizer_is_new_contribution": False,
            "scheduler_changed": False,
            "cpu_gpu_hierarchy_changed": False,
        },
        "conversion": [],
        "attention": [],
        "overlap": [],
        "comparisons": {
            "queue_or_refuse_capacity": {
                "quality_change": 0.0,
                "reclaimed_bytes": 0,
                "interpretation": "analytical baseline: preserve FP16 pages and admit no extra request",
            },
            "morphserve": {
                "local_experiment": False,
                "paper_boundary": "MorphServe KVResizer reallocates capacity; it does not quantize already-live KV pages",
                "superiority_claim": False,
            },
        },
    }
    shapes = [args.model_family] if args.model_family != "both" else list(SHAPES)
    for family in shapes:
        shape = SHAPES[family]
        family_result = {"model_family": family, "shape": shape, "trials": {}}
        for batch in args.batches:
            for context in args.contexts:
                key = f"batch{batch}_context{context}"
                seed = args.seed + batch * 1000 + context
                family_result["trials"][key] = {
                    "fp16": fp16_attention_trial(shape, batch, context, device, seed, args.attention_iterations, args.attention_repetitions),
                    "static": [static_attention_trial(shape, batch, context, target, device, seed, args.attention_iterations, args.attention_repetitions) for target in TARGETS],
                    "mixed": [mixed_attention_trial(shape, batch, context, target, fraction, device, seed, args.attention_iterations, args.attention_repetitions) for target in TARGETS for fraction in FRACTIONS],
                }
                for target in TARGETS:
                    for fraction in FRACTIONS:
                        result["conversion"].append({
                            "model_family": family,
                            **conversion_trial(shape, batch, context, target, fraction, device, seed),
                        })
        for target in TARGETS:
            result["overlap"].append({"model_family": family, **overlap_trial(shape, target, device, args.seed + 77)})
        result.setdefault("families", []).append(family_result)
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return result


def tokenizer_prompt(model_path: str, context_tokens: int):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    seed_text = (
        "A request remains live while a continuous batching engine appends one token at a time. "
        "The cache stores keys and values in pages so that memory pressure can be measured directly. "
    )
    ids = tokenizer.encode(seed_text, add_special_tokens=True)
    while len(ids) < context_tokens:
        ids.extend(ids[: max(1, min(len(ids), context_tokens - len(ids)))])
    return ids[:context_tokens]


def build_model(model_path: str, page_format: str, context_tokens: int, generation_tokens: int, device: torch.device) -> LlamaModel:
    if device.type != "cuda":
        raise RuntimeError("quality run requires CUDA")
    max_blocks = math.ceil((context_tokens + generation_tokens + 2) / 16) + 4
    config = EngineConfig(
        model_path=model_path,
        use_dummy=False,
        block_size=16,
        gpu_mem_utilization=0.80,
        num_cpu_blocks=0,
        max_seqs_in_block_table=8,
        max_blocks_per_seq=max_blocks,
        max_batch_size=1,
        max_tokens_in_batch=context_tokens + 2,
        kv_page_format=page_format,
    )
    model = LlamaModel(config)
    model.load_weights()
    model.init_kvcache_and_swap(max_blocks)
    return model


def timed_forward(model: LlamaModel, input_ids, seq_ids, decoding_lens, device: torch.device):
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    output = model.forward(input_ids, seq_ids, decoding_lens, return_logits=True)
    stop.record()
    stop.synchronize()
    return output, start.elapsed_time(stop)


def capture_baseline(model_path: str, prompt_ids: list[int], generation_tokens: int, device: torch.device):
    model = build_model(model_path, "dense_fp16", len(prompt_ids), generation_tokens, device)
    with torch.inference_mode():
        output, prefill_ms = timed_forward(model, [prompt_ids], [0], [], device)
        logits = output[0].float().cpu()
        tokens = [int(torch.argmax(logits).item())]
        logits_trace = [logits]
        decode_ms = []
        for step in range(1, generation_tokens):
            output, elapsed_ms = timed_forward(model, [[tokens[-1]]], [0], [len(prompt_ids) + step], device)
            decode_ms.append(elapsed_ms)
            logits = output[0].float().cpu()
            logits_trace.append(logits)
            tokens.append(int(torch.argmax(logits).item()))
    sync(device)
    return model, logits_trace, tokens, {"prefill_ms": prefill_ms, "decode_ms": decode_ms}


def dynamic_page_keys(model: LlamaModel, fraction: float, recency: str, layer_range: str):
    assert model.page_kv_cache is not None
    keys = model.page_kv_cache.resident_page_keys()
    blocks = sorted({block_id for block_id, _ in keys})
    count = round(len(blocks) * fraction)
    selected_blocks = set(blocks[:count] if recency == "old" else blocks[-count:] if count else [])
    num_layers = model.model_config.num_layers
    if layer_range == "early":
        layer_ids = set(range(max(1, num_layers // 4)))
    elif layer_range == "late":
        layer_ids = set(range(num_layers - max(1, num_layers // 4), num_layers))
    else:
        layer_ids = set(range(num_layers))
    return [(block_id, layer_id) for block_id, layer_id in keys if block_id in selected_blocks and layer_id in layer_ids]


def quality_variant(
    model_path: str,
    prompt_ids: list[int],
    generation_tokens: int,
    baseline_logits: list[torch.Tensor],
    baseline_tokens: list[int],
    device: torch.device,
    variant: dict[str, object],
    page_reference_logits: list[torch.Tensor] | None = None,
    return_trace: bool = False,
) -> dict[str, object]:
    page_format = str(variant.get("initial_format", "fp16"))
    model = build_model(model_path, page_format, len(prompt_ids), generation_tokens, device)
    with torch.inference_mode():
        first_output, prefill_ms = timed_forward(model, [prompt_ids], [0], [], device)
        first = first_output[0].float()
        demoted = []
        conversion = None
        if variant.get("demote"):
            keys = dynamic_page_keys(model, float(variant["fraction"]), str(variant["recency"]), str(variant["layer_range"]))
            if str(variant["target"]) == "int8" and tuple(variant["components"]) == ("k", "v"):
                batch_result = model.page_kv_cache.demote_pages_batch(keys, "int8", variant["components"])
                demoted = [
                    {
                        "block_id": item.block_id,
                        "layer_id": item.layer_id,
                        "components": list(item.components),
                        "result": dataclasses.asdict(item),
                    }
                    for item in batch_result.per_page
                ]
                conversion = {
                    "batched": True,
                    "pages": len(batch_result.page_keys),
                    "elapsed_ms": batch_result.elapsed_ms,
                    "before_bytes": batch_result.before_bytes,
                    "after_bytes": batch_result.after_bytes,
                    "reclaimed_bytes": batch_result.reclaimed_bytes,
                    "temporary_bytes": batch_result.temporary_bytes,
                }
            else:
                demoted = [
                    {
                        "block_id": block_id,
                        "layer_id": layer_id,
                        "components": list(variant["components"]),
                        "result": dataclasses.asdict(model.page_kv_cache.demote_page(block_id, layer_id, str(variant["target"]), variant["components"])),
                    }
                    for block_id, layer_id in keys
                ]
                conversion = {
                    "batched": False,
                    "pages": len(demoted),
                    "elapsed_ms": sum(item["result"]["elapsed_ms"] for item in demoted),
                    "before_bytes": sum(item["result"]["before_bytes"] for item in demoted),
                    "after_bytes": sum(item["result"]["after_bytes"] for item in demoted),
                    "reclaimed_bytes": sum(item["result"]["before_bytes"] - item["result"]["after_bytes"] for item in demoted),
                    "temporary_bytes": max((item["result"]["temporary_bytes"] for item in demoted), default=0),
                }
            sync(device)
        comp_logits = [first.cpu()]
        decode_ms = []
        for step in range(1, generation_tokens):
            output, elapsed_ms = timed_forward(model, [[baseline_tokens[step - 1]]], [0], [len(prompt_ids) + step], device)
            decode_ms.append(elapsed_ms)
            comp_logits.append(output[0].float().cpu())
    sync(device)

    nll = []
    baseline_nll = []
    nll_delta = []
    kl = []
    rmse = []
    agreement = []
    page_nll_delta = []
    page_kl = []
    page_rmse = []
    page_agreement = []
    for index, (base, comp, target) in enumerate(zip(baseline_logits, comp_logits, baseline_tokens)):
        base_logp = torch.log_softmax(base, dim=-1)
        comp_logp = torch.log_softmax(comp, dim=-1)
        base_prob = base_logp.exp()
        reference_nll = float(-base_logp[target])
        candidate_nll = float(-comp_logp[target])
        baseline_nll.append(reference_nll)
        nll.append(candidate_nll)
        nll_delta.append(candidate_nll - reference_nll)
        kl.append(float((base_prob * (base_logp - comp_logp)).sum()))
        rmse.append(float(torch.sqrt(torch.mean((base - comp) ** 2))))
        agreement.append(int(torch.argmax(comp).item()) == target)
        if page_reference_logits is not None:
            page = page_reference_logits[index]
            page_logp = torch.log_softmax(page, dim=-1)
            page_nll_delta.append(candidate_nll - float(-page_logp[target]))
            page_kl.append(float((page_logp.exp() * (page_logp - comp_logp)).sum()))
            page_rmse.append(float(torch.sqrt(torch.mean((page - comp) ** 2))))
            page_agreement.append(int(torch.argmax(comp).item()) == int(torch.argmax(page).item()))
    summary = model.kv_storage_summary()
    if model.page_kv_cache is not None:
        full_fp16 = len(model.page_kv_cache.resident_page_keys()) * math.prod(model.page_kv_cache.page_shape) * 2 * 2
        logical = model.page_kv_cache.logical_bytes()
        summary.update({
            "full_fp16_resident_page_bytes": full_fp16,
            "reclaimed_bytes_vs_fp16_pages": full_fp16 - logical,
            "reclaimed_mib_vs_fp16_pages": (full_fp16 - logical) / (1024 ** 2),
            "no_dense_fp16_shadow": not model.page_kv_cache.has_unreclaimed_shadow(),
            "compressed_payloads_are_not_fp16": all(
                (page.k_format == "fp16" or page.k_payload.dtype != torch.float16)
                and (page.v_format == "fp16" or page.v_payload.dtype != torch.float16)
                for key in model.page_kv_cache.resident_page_keys()
                for page in [model.page_kv_cache.page(*key)]
            ),
        })
    quality_result = {
        "name": variant["name"],
        "target_format": variant.get("target", "fp16"),
        "fraction": variant.get("fraction", 0.0),
        "recency": variant.get("recency"),
        "layer_range": variant.get("layer_range"),
        "components": list(variant.get("components", ("k", "v"))),
        "demoted_pages": len(demoted),
        "demotions": demoted,
        "conversion": conversion,
        "storage": summary,
        "latency": {
            "prefill_ms": prefill_ms,
            "decode_ms": decode_ms,
            "decode_ms_median": statistics.median(decode_ms) if decode_ms else None,
        },
        "quality": {
            "generation_tokens": generation_tokens,
            "forced_prefix_top1_agreement": sum(agreement) / len(agreement),
            "first_mismatch_step": next((i for i, same in enumerate(agreement) if not same), None),
            "baseline_token_nll_mean": sum(nll) / len(nll),
            "baseline_token_nll_max": max(nll),
            "reference_fp16_token_nll_mean": sum(baseline_nll) / len(baseline_nll),
            "paired_nll_delta_mean": sum(nll_delta) / len(nll_delta),
            "paired_nll_delta_max": max(nll_delta),
            "logit_kl_mean": sum(kl) / len(kl),
            "logit_kl_max": max(kl),
            "logit_rmse_mean": sum(rmse) / len(rmse),
            "per_step": [{"nll": a, "reference_nll": e, "nll_delta": f, "kl": b, "rmse": c, "top1_agrees": d} for a, e, f, b, c, d in zip(nll, baseline_nll, nll_delta, kl, rmse, agreement)],
            "forced_baseline_tokens": baseline_tokens,
        },
    }
    if page_reference_logits is not None:
        quality_result["quality"]["page_fp16_reference"] = {
            "top1_agreement": sum(page_agreement) / len(page_agreement),
            "paired_nll_delta_mean": sum(page_nll_delta) / len(page_nll_delta),
            "logit_kl_mean": sum(page_kl) / len(page_kl),
            "logit_rmse_mean": sum(page_rmse) / len(page_rmse),
        }
    if return_trace:
        quality_result["_logits_trace"] = comp_logits
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return quality_result


def run_quality(args: argparse.Namespace) -> dict[str, object]:
    device = device_arg(args.device)
    paths = args.quality_model_path
    if not paths:
        raise ValueError("--quality-model-path is required with --run-quality")
    results = []
    for model_path in paths:
        name = "llama32_1b" if "3.2-1B" in model_path or "1b" in model_path.lower() else "llama31_8b"
        prompt_ids = tokenizer_prompt(model_path, args.quality_context_tokens)
        baseline_model, baseline_logits, baseline_tokens, baseline_latency = capture_baseline(model_path, prompt_ids, args.quality_generation_tokens, device)
        results.append({
            "model_family": name,
            "model_path": model_path,
            "prompt_token_count": len(prompt_ids),
            "baseline": {
                "kind": "unchanged_fp16_dense",
                "generation_tokens": len(baseline_tokens),
                "tokens": baseline_tokens,
                "storage_bytes": baseline_model.k_cache.numel() * baseline_model.k_cache.element_size() * 2,
                "latency": baseline_latency,
            },
            "variants": [],
        })
        del baseline_model
        gc.collect()
        torch.cuda.empty_cache()
        page_control = quality_variant(
            model_path,
            prompt_ids,
            args.quality_generation_tokens,
            baseline_logits,
            baseline_tokens,
            device,
            {"name": "page_fp16_control", "initial_format": "fp16", "target": "fp16", "components": ("k", "v")},
            return_trace=True,
        )
        page_reference_logits = page_control.pop("_logits_trace")
        results[-1]["variants"].append(page_control)
        variants = [
            {"name": "static_int8", "initial_format": "int8", "target": "int8", "fraction": 1.0, "components": ("k", "v")},
            {"name": "static_int4", "initial_format": "int4", "target": "int4", "fraction": 1.0, "components": ("k", "v")},
        ]
        for target in ("int8", "int4"):
            fractions = (0.5,) if target == "int4" else (0.25, 0.5, 0.75)
            for fraction in fractions:
                label = int(fraction * 100)
                variants.extend([
                    {"name": f"dynamic_old_{target}_{label}_both", "initial_format": "fp16", "target": target, "fraction": fraction, "recency": "old", "layer_range": "all", "components": ("k", "v"), "demote": True},
                    {"name": f"dynamic_recent_{target}_{label}_both", "initial_format": "fp16", "target": target, "fraction": fraction, "recency": "recent", "layer_range": "all", "components": ("k", "v"), "demote": True},
                ])
        variants.extend([
            {"name": "dynamic_old_int8_50_early", "initial_format": "fp16", "target": "int8", "fraction": 0.5, "recency": "old", "layer_range": "early", "components": ("k", "v"), "demote": True},
            {"name": "dynamic_old_int8_50_late", "initial_format": "fp16", "target": "int8", "fraction": 0.5, "recency": "old", "layer_range": "late", "components": ("k", "v"), "demote": True},
            {"name": "dynamic_old_int8_50_k_only", "initial_format": "fp16", "target": "int8", "fraction": 0.5, "recency": "old", "layer_range": "all", "components": ("k",), "demote": True},
            {"name": "dynamic_old_int8_50_v_only", "initial_format": "fp16", "target": "int8", "fraction": 0.5, "recency": "old", "layer_range": "all", "components": ("v",), "demote": True},
        ])
        for variant in variants:
            results[-1]["variants"].append(quality_variant(model_path, prompt_ids, args.quality_generation_tokens, baseline_logits, baseline_tokens, device, variant, page_reference_logits=page_reference_logits))
    return {"quality": results}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--model-family", choices=("llama32_1b", "llama31_8b", "both"), default="both")
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--contexts", type=int, nargs="+", default=[128, 512, 1024])
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--attention-iterations", type=int, default=3)
    parser.add_argument("--attention-repetitions", type=int, default=3)
    parser.add_argument("--run-quality", action="store_true")
    parser.add_argument("--quality-model-path", action="append", default=[])
    parser.add_argument("--quality-context-tokens", type=int, default=256)
    parser.add_argument("--quality-generation-tokens", type=int, default=16)
    parser.add_argument("--optimized-smoke", action="store_true", help="Run the small Triton INT8 segmented-path smoke instead of the full grid")
    args = parser.parse_args()

    if args.optimized_smoke:
        device = device_arg(args.device)
        output = {
            "schema": "live-kv-optimized-int8-v1",
            "provenance": {
                "branch": subprocess.check_output(["git", "-C", str(ROOT), "branch", "--show-current"], text=True).strip(),
                "git_head": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
                "swiftllm_upstream_commit": (ROOT / "vendor/swiftLLM/UPSTREAM_COMMIT").read_text().strip(),
                "source_files": list(SOURCE_FILES),
                "source_sha256": source_fingerprint(),
                "device": str(device),
                "seed": args.seed,
                "oracle": "page_attention_for_layer",
                "kernel": "segmented_paged_attention",
            },
            "smoke": [optimized_smoke(SHAPES[family], device, args.seed + index) for index, family in enumerate(("llama32_1b", "llama31_8b"))],
        }
    else:
        output = run_microbench(args)
    if args.run_quality:
        output.update(run_quality(args))
    output["invocation"] = {"argv": sys.argv, "cwd": os.getcwd()}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
        handle.write("\n")
    print(json.dumps({
        "output": args.output,
        "conversion_trials": len(output.get("conversion", [])),
        "attention_families": len(output.get("families", [])),
        "quality_runs": len(output.get("quality", [])),
        "optimized_smoke_runs": len(output.get("smoke", [])),
    }))


if __name__ == "__main__":
    main()
