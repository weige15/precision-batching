#!/usr/bin/env python3
"""Corrected SwiftLLM KV measurement harness (v2).

This is deliberately a small evidence harness, not a scheduler or allocator
rewrite.  It exercises the native page store with an explicit non-identity
block table and GQA, and reports logical payload, unique tensor storage,
allocator allocated/reserved/peak state, copied packed arenas, temporary
workspace, and release observations separately.  Timing never performs
quality scoring or a per-token host synchronization.
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
sys.path.insert(0, str(ROOT / "vendor/swiftLLM"))
sys.path.insert(0, str(ROOT / "vendor/swiftLLM/csrc"))

from swiftllm.engine_config import EngineConfig  # noqa: E402
from swiftllm.worker.kv_cache import PagedKVCache, page_attention_for_layer  # noqa: E402
from swiftllm.worker.kernels.paged_attn import paged_attention  # noqa: E402
from swiftllm.worker.model import LlamaModel  # noqa: E402


SHAPES = {
    "llama32_1b": {"num_layers": 16, "num_q_heads": 32, "num_kv_heads": 8, "head_dim": 64},
    "llama31_8b": {"num_layers": 32, "num_q_heads": 32, "num_kv_heads": 8, "head_dim": 128},
}
BLOCK_SIZE = 16
GROUP_SIZE = 128
SOURCE_FILES = (
    "scripts/kv_measurement_v2.py",
    "vendor/swiftLLM/swiftllm/worker/kv_cache.py",
    "vendor/swiftLLM/swiftllm/worker/kernels/segmented_paged_attn.py",
    "vendor/swiftLLM/swiftllm/worker/kernels/paged_attn.py",
    "vendor/swiftLLM/swiftllm/worker/kernels/kvcache_mgmt.py",
)


def sync(device: torch.device) -> None:
    torch.cuda.synchronize(device)


def allocator_snapshot(device: torch.device) -> dict[str, int]:
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def tensor_logical_bytes(value) -> int:
    return int(value.numel() * value.element_size()) if isinstance(value, torch.Tensor) else 0


def tensor_storage(value: torch.Tensor) -> tuple[tuple[object, ...], int]:
    storage = value.untyped_storage()
    size = int(storage.nbytes())
    pointer = int(storage.data_ptr())
    if size == 0:
        pointer = id(value)
    return (value.device.type, value.device.index, pointer, size), size


def storage_ledger(records: list[tuple[str, torch.Tensor]]) -> dict[str, object]:
    seen: dict[tuple[object, ...], int] = {}
    logical = 0
    unique = 0
    inventory = []
    for name, value in records:
        key, storage_bytes = tensor_storage(value)
        logical_bytes = tensor_logical_bytes(value)
        logical += logical_bytes
        if key not in seen:
            seen[key] = storage_bytes
            unique += storage_bytes
        inventory.append({
            "name": name,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "logical_bytes": logical_bytes,
            "storage_bytes": storage_bytes,
            "storage_alias_id": f"{key[0]}:{key[1]}:{key[2]}:{key[3]}",
        })
    return {"logical_bytes": logical, "unique_live_storage_bytes": unique, "tensor_count": len(inventory), "tensors": inventory}


def cache_records(cache: PagedKVCache, include_packed: bool = True, workspaces: list[tuple[str, torch.Tensor]] | None = None):
    page_records: list[tuple[str, torch.Tensor]] = []
    for (block_id, layer_id), page in sorted(cache._pages.items()):  # measured, explicit research store
        for component in ("k_payload", "v_payload", "k_scales", "v_scales"):
            value = getattr(page, component)
            if value is not None:
                page_records.append((f"page[{block_id},{layer_id}].{component}", value))
    metadata = [("metadata_codes", cache.metadata_codes)]
    packed_records: list[tuple[str, torch.Tensor]] = []
    if include_packed and cache._packed_attention_storage is not None:
        packed = cache._packed_attention_storage
        for field in packed.__dataclass_fields__:
            value = getattr(packed, field)
            if isinstance(value, torch.Tensor):
                packed_records.append((f"packed.{field}", value))
    workspace_records = workspaces or []
    page_ledger = storage_ledger(page_records)
    metadata_ledger = storage_ledger(metadata)
    packed_ledger = storage_ledger(packed_records)
    all_ledger = storage_ledger(page_records + metadata + packed_records + workspace_records)
    return {
        "logical_payload_bytes": int(sum(cache._pages[key].allocated_bytes for key in cache._pages)),
        "page_payload_unique_live_storage_bytes": page_ledger["unique_live_storage_bytes"],
        "metadata_bytes": metadata_ledger["logical_bytes"],
        "packed_arena_and_map_bytes": packed_ledger["logical_bytes"],
        "packed_arena_unique_live_storage_bytes": packed_ledger["unique_live_storage_bytes"],
        "workspace_logical_bytes": sum(tensor_logical_bytes(value) for _, value in workspace_records),
        "unique_live_storage_bytes": all_ledger["unique_live_storage_bytes"],
        "tensor_count": all_ledger["tensor_count"],
        "tensor_inventory": all_ledger["tensors"],
        "pending_old_page_bytes": int(cache.storage_summary()["pending_old_page_bytes"]),
        "has_unreclaimed_shadow": cache.has_unreclaimed_shadow(),
        "packed_storage_materialized": cache._packed_attention_storage is not None,
        "accounting_basis": "logical payload excludes metadata/packed copies; unique storage deduplicates underlying tensor storage; packed/workspace are reported separately",
    }


def select_request_local_blocks(block_table: torch.Tensor, fraction: float, recency: str = "old") -> list[int]:
    """Select old/recent logical pages within every request, then return physical IDs."""
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be in [0, 1]")
    table = block_table.detach().cpu()
    selected = []
    for row in table.tolist():
        count = round(len(row) * fraction)
        positions = list(range(count)) if recency == "old" else list(range(len(row) - count, len(row))) if count else []
        selected.extend(int(row[position]) for position in positions)
    return selected


def dense_page_layout(k_pages: list[torch.Tensor], v_pages: list[torch.Tensor], layers: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    """Build SwiftLLM's [block, layer, KV-head, token, dim] layout correctly.

    A plain reshape from [block, token, KV-head, dim] to [block, KV-head,
    token, dim] silently changes token/head interpretation.  The permutation
    is intentional and is covered by the non-identity GQA check below.
    """
    k = torch.stack(k_pages)
    v = torch.stack(v_pages)
    if layers != 1:
        raise ValueError("dense_page_layout takes one layer of pages")
    return k.unsqueeze(1).permute(0, 1, 3, 2, 4).contiguous(), v.unsqueeze(1).permute(0, 1, 3, 2, 4).contiguous()


def source_fingerprint() -> str:
    digest = hashlib.sha256()
    for relative in SOURCE_FILES:
        digest.update(relative.encode())
        digest.update((ROOT / relative).read_bytes())
    return digest.hexdigest()


def dirty_provenance() -> dict[str, object]:
    diff = subprocess.check_output(["git", "-C", str(ROOT), "diff", "--binary", "HEAD"], text=False)
    return {
        "reviewed_remote_commit": "8c0a0bfcc6bf87461c104603718ed2dc507df390",
        "local_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "branch": subprocess.check_output(["git", "-C", str(ROOT), "branch", "--show-current"], text=True).strip(),
        "dirty_status": subprocess.check_output(["git", "-C", str(ROOT), "status", "--short"], text=True).splitlines(),
        "tracked_dirty_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "submodule_status": subprocess.check_output(["git", "-C", str(ROOT), "submodule", "status", "--recursive"], text=True).splitlines(),
    }


def shape_model(shape, num_layers: int | None = None):
    return SimpleNamespace(
        num_layers=num_layers or shape["num_layers"],
        num_q_heads=shape["num_q_heads"],
        num_kv_heads=shape["num_kv_heads"],
        head_dim=shape["head_dim"],
    )


def make_cache(shape, batch: int, pages_per_seq: int, device: torch.device, layers: int | None = None):
    layers = layers or shape["num_layers"]
    return PagedKVCache(
        num_blocks=batch * pages_per_seq, num_layers=layers,
        num_kv_heads=shape["num_kv_heads"], block_size=BLOCK_SIZE,
        head_dim=shape["head_dim"], device=device, default_format="fp16",
        group_size=GROUP_SIZE,
    )


def populate_prefill(cache: PagedKVCache, block_table: torch.Tensor, context: int, seed: int):
    generator = torch.Generator(device=cache.device)
    generator.manual_seed(seed)
    k_by_page: dict[tuple[int, int], torch.Tensor] = {}
    v_by_page: dict[tuple[int, int], torch.Tensor] = {}
    for request_id, row in enumerate(block_table.detach().cpu().tolist()):
        for logical_page, physical_id in enumerate(row):
            start = logical_page * BLOCK_SIZE
            if start >= context:
                continue
            tokens = min(BLOCK_SIZE, context - start)
            for layer_id in range(cache.num_layers):
                k = torch.randn((tokens, cache.num_kv_heads, cache.head_dim), generator=generator, device=cache.device, dtype=torch.float16)
                v = torch.randn(k.shape, generator=generator, device=cache.device, dtype=torch.float16)
                cache.write(int(physical_id), layer_id, k, v)
                k_by_page[(int(physical_id), layer_id)] = cache.page(int(physical_id), layer_id).decode("k")
                v_by_page[(int(physical_id), layer_id)] = cache.page(int(physical_id), layer_id).decode("v")
    sync(cache.device)
    return k_by_page, v_by_page


def attention_args(shape, batch, pages_per_seq, lengths, device):
    block_table = torch.arange(batch * pages_per_seq, dtype=torch.int32, device=device).reshape(batch, pages_per_seq)
    # Deliberately remap physical pages: request 0 sees the second chunk and
    # request 1 sees the first chunk.  Selection must follow this table.
    if batch == 2:
        block_table = torch.flip(block_table, dims=(1,))
    seq_ids = torch.arange(batch, dtype=torch.int32, device=device)
    lengths_tensor = torch.tensor(lengths, dtype=torch.int32, device=device)
    generator = torch.Generator(device=device)
    generator.manual_seed(991 + shape["head_dim"] + sum(lengths))
    q = torch.randn((batch, shape["num_q_heads"], shape["head_dim"]), generator=generator, device=device, dtype=torch.float16)
    out = torch.empty((batch, shape["num_q_heads"] * shape["head_dim"]), device=device, dtype=torch.float16)
    return block_table, seq_ids, lengths_tensor, q, out


def infer_args(shape, batch, pages_per_seq, lengths, device):
    block_table, seq_ids, lengths_tensor, q, out = attention_args(shape, batch, pages_per_seq, lengths, device)
    infer = SimpleNamespace(
        seq_block_size=2048,
        num_seq_blocks=math.ceil(max(lengths) / 2048),
        num_decoding_seqs=batch,
        num_prefill_seqs=0,
        decoding_seq_lens=lengths_tensor,
        seq_ids=seq_ids,
        softmax_scale=shape["head_dim"] ** -0.5,
    )
    return block_table, seq_ids, lengths_tensor, q, out, infer


def timed(fn, device: torch.device, iterations: int, repetitions: int):
    fn()
    sync(device)
    rows = []
    for _ in range(repetitions):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            fn()
        stop.record()
        stop.synchronize()
        rows.append(float(start.elapsed_time(stop)))
    return {
        "timing_unit": "milliseconds",
        "iterations": iterations,
        "repetitions": repetitions,
        "elapsed_ms_repeats": rows,
        "elapsed_ms_median": statistics.median(rows),
        "per_decode_ms_median": statistics.median(rows) / iterations,
        "timing_boundary": "CUDA event around attention only; no quality scoring or host synchronization in the loop",
    }


def dense_reference(cache, shape, block_table, seq_ids, lengths, q, out, layer_id=0):
    max_blocks = block_table.shape[1]
    dense_k = torch.zeros((cache.num_blocks, cache.num_layers, cache.num_kv_heads, BLOCK_SIZE, cache.head_dim), device=cache.device, dtype=torch.float16)
    dense_v = torch.zeros_like(dense_k)
    for physical_id in range(cache.num_blocks):
        if (physical_id, layer_id) not in cache._pages:
            continue
        dense_k[physical_id, layer_id] = cache.page(physical_id, layer_id).decode("k").permute(1, 0, 2)
        dense_v[physical_id, layer_id] = cache.page(physical_id, layer_id).decode("v").permute(1, 0, 2)
    model = shape_model(shape)
    engine = SimpleNamespace(block_size=BLOCK_SIZE, max_blocks_per_seq=max_blocks)
    infer = SimpleNamespace(
        seq_block_size=2048, num_seq_blocks=math.ceil(max(lengths) / 2048),
        num_decoding_seqs=len(lengths), num_prefill_seqs=0,
        decoding_seq_lens=lengths, seq_ids=seq_ids, softmax_scale=shape["head_dim"] ** -0.5,
    )
    paged_attention(q, dense_k, dense_v, block_table, model, engine, infer, layer_id, out)


def run_family(family: str, device: torch.device, seed: int, iterations: int, repetitions: int) -> dict[str, object]:
    shape = SHAPES[family]
    batch = 2
    pages_per_seq = 3
    prefill_tokens = 31
    decode_positions = [31, 32, 33]  # existing-page append, page boundary, repeated new-page append
    lengths = [34, 34]
    cache = make_cache(shape, batch, pages_per_seq, device)
    block_table, seq_ids, lengths_tensor, q, out = attention_args(shape, batch, pages_per_seq, lengths, device)
    populate_prefill(cache, block_table, prefill_tokens, seed)
    prefill_ledger = cache_records(cache, include_packed=False)
    prefill_allocator = allocator_snapshot(device)

    selected_blocks = select_request_local_blocks(block_table, 1.0 / 3.0, "old")
    expected_selected = [int(block_table[request, 0].item()) for request in range(batch)]
    selected_keys = [(physical_id, layer_id) for physical_id in selected_blocks for layer_id in range(cache.num_layers)]
    conversion_before = allocator_snapshot(device)
    conversion = cache.demote_pages_batch(selected_keys, "int8")
    sync(device)
    conversion_after = allocator_snapshot(device)

    # Append through the residual/page boundary after conversion.  Newly used
    # pages remain FP16; no global physical-ID assumption is made.
    append_trace = []
    generator = torch.Generator(device=device)
    generator.manual_seed(seed + 17)
    for position in decode_positions:
        block_index, token_offset = divmod(position, BLOCK_SIZE)
        for request in range(batch):
            physical_id = int(block_table[request, block_index].item())
            for layer_id in range(cache.num_layers):
                k = torch.randn((1, cache.num_kv_heads, cache.head_dim), generator=generator, device=device, dtype=torch.float16)
                v = torch.randn(k.shape, generator=generator, device=device, dtype=torch.float16)
                cache.write(physical_id, layer_id, k, v, token_offset=token_offset)
        sync(device)
        append_trace.append({"token_position": position + 1, "storage": cache_records(cache, include_packed=False), "allocator": allocator_snapshot(device)})

    # FP16 logical inputs/outputs: the dense and page paths see the same
    # decoded pages, but use different physical page orderings and GQA.
    fp16_cache = make_cache(shape, batch, pages_per_seq, device)
    populate_prefill(fp16_cache, block_table, prefill_tokens, seed + 99)
    fp_block_table, fp_seq_ids, fp_lengths, fp_q, fp_page_out = attention_args(shape, batch, pages_per_seq, [prefill_tokens, prefill_tokens], device)
    # Reuse the same non-identity table and logical values for the comparison.
    fp_block_table.copy_(block_table)
    fp_seq_ids.copy_(seq_ids)
    fp_lengths.fill_(prefill_tokens)
    fp_q.copy_(q)
    dense_out = torch.empty_like(fp_page_out)
    dense_reference(fp16_cache, shape, fp_block_table, fp_seq_ids, fp_lengths, fp_q, dense_out)
    page_attention_for_layer(fp_q, fp16_cache, fp_block_table, fp_seq_ids, fp_lengths, shape_model(shape), SimpleNamespace(block_size=BLOCK_SIZE), 0, fp_page_out)
    sync(device)
    fp16_correctness = {
        "max_abs_error_dense_vs_page_oracle": float((dense_out - fp_page_out).abs().max().item()),
        "mean_abs_error_dense_vs_page_oracle": float((dense_out - fp_page_out).abs().mean().item()),
        "gqa_groups": shape["num_q_heads"] // shape["num_kv_heads"],
        "non_identity_block_table": True,
        "matched_logical_fp16_inputs": True,
    }
    del fp16_cache, dense_out, fp_page_out
    gc.collect()
    torch.cuda.empty_cache()
    sync(device)

    conversion_allocator_peak_before = allocator_snapshot(device)
    torch.cuda.reset_peak_memory_stats(device)
    attention_allocator_before = allocator_snapshot(device)
    restored_before = attention_allocator_before
    # Build packed arenas and run a warm/cold path.  Both copied arenas and
    # temporary mid-o/descriptor tensors are included in the ledger/peak.
    oracle_out = torch.empty_like(out)
    page_attention_for_layer(q, cache, block_table, seq_ids, lengths_tensor, shape_model(shape), SimpleNamespace(block_size=BLOCK_SIZE), 0, oracle_out)
    optimized_cold = cache.optimized_attention(q, block_table, seq_ids, lengths_tensor, shape_model(shape), SimpleNamespace(block_size=BLOCK_SIZE), 0, out)
    sync(device)
    cold_peak = allocator_snapshot(device)
    total_segments = int(optimized_cold["total_segments"])
    descriptor_segments = int(optimized_cold["fp16_segments"]) + int(optimized_cold["int8_segments"])
    transient_workspace = {
        "segment_valid_bytes": batch * total_segments * 4,
        "mid_o_bytes": batch * shape["num_q_heads"] * total_segments * shape["head_dim"] * 4,
        "mid_log_bytes": batch * shape["num_q_heads"] * total_segments * 4,
        "segment_descriptor_bytes": descriptor_segments * 4 * 4,
    }
    transient_workspace["logical_bytes"] = sum(transient_workspace.values())
    transient_workspace["after_return_bytes"] = 0
    steady = timed(lambda: cache.optimized_attention(q, block_table, seq_ids, lengths_tensor, shape_model(shape), SimpleNamespace(block_size=BLOCK_SIZE), 0, out), device, iterations, repetitions)
    optimized_ledger = cache_records(cache, include_packed=True)
    max_error = float((oracle_out - out).abs().max().item())
    finite = bool(torch.isfinite(out).all().item())

    restored = cache.promote_pages_batch(selected_keys, "fp16")
    sync(device)
    restored_after = allocator_snapshot(device)
    restored_ledger = cache_records(cache, include_packed=False)

    # Release every page and the packed arena before a future repetition. This
    # is a direct reclamation observation, not a claim about global HBM free.
    before_release = allocator_snapshot(device)
    logical_before_release = restored_ledger["logical_payload_bytes"]
    cache.free_blocks(range(cache.num_blocks))
    del cache
    gc.collect()
    sync(device)
    torch.cuda.empty_cache()
    sync(device)
    after_release = allocator_snapshot(device)
    release = {
        "logical_payload_bytes_before_release": logical_before_release,
        "allocator_allocated_before_release": before_release["allocated_bytes"],
        "allocator_reserved_before_release": before_release["reserved_bytes"],
        "allocator_allocated_after_release": after_release["allocated_bytes"],
        "allocator_reserved_after_release": after_release["reserved_bytes"],
        "allocator_allocated_released_bytes": before_release["allocated_bytes"] - after_release["allocated_bytes"],
        "allocator_reserved_released_bytes": before_release["reserved_bytes"] - after_release["reserved_bytes"],
        "capacity_reusable_basis": "freed page-store payload plus metadata/packed copies after explicit cache release",
    }
    return {
        "model_family": family,
        "shape": shape,
        "batch": batch,
        "prefill_tokens": prefill_tokens,
        "decode_append_positions": decode_positions,
        "pages_per_sequence_capacity": pages_per_seq,
        "block_table": block_table.cpu().tolist(),
        "selected_old_physical_blocks": selected_blocks,
        "expected_request_local_old_blocks": expected_selected,
        "selection_is_request_local": selected_blocks == expected_selected,
        "prefill": {"storage": prefill_ledger, "allocator": prefill_allocator},
        "conversion": {
            "target_format": conversion.target_format,
            "pages": len(conversion.page_keys),
            "all_layer_page_keys": [list(key) for key in conversion.page_keys],
            "before_bytes": conversion.before_bytes,
            "after_bytes": conversion.after_bytes,
            "reclaimed_bytes": conversion.reclaimed_bytes,
            "temporary_bytes_reported": conversion.temporary_bytes,
            "elapsed_ms": conversion.elapsed_ms,
            "allocator_before": conversion_before,
            "allocator_after": conversion_after,
            "allocator_peak_before_attention": conversion_allocator_peak_before,
        },
        "append_trace": append_trace,
        "fp16_layout_correctness": fp16_correctness,
        "optimized_attention": {
            "cold_result": optimized_cold,
            "cold_allocator_peak": cold_peak,
            "cold_peak_additional_allocated_bytes": max(0, cold_peak["peak_allocated_bytes"] - attention_allocator_before["allocated_bytes"]),
            "cold_peak_additional_reserved_bytes": max(0, cold_peak["peak_reserved_bytes"] - attention_allocator_before["reserved_bytes"]),
            "steady_state": steady,
            "storage": optimized_ledger,
            "packed_copies_included_in_ledger": True,
            "transient_workspace": transient_workspace,
            "workspace_peak_includes_mid_o_and_descriptors": True,
            "max_abs_error_vs_page_oracle": max_error,
            "finite_output": finite,
            "output_checksum": float(out.float().sum().item()),
        },
        "restoration": {
            "pages": len(restored.page_keys),
            "elapsed_ms": restored.elapsed_ms,
            "before_bytes": restored.before_bytes,
            "after_bytes": restored.after_bytes,
            "allocator_before": restored_before,
            "allocator_after": restored_after,
        },
        "release": release,
        "capacity_claim": "narrow native page-store evidence only; copied packed arenas are charged, and this is not a request-level scheduler capacity claim",
    }


def _checkpoint_prompt(model_path: str, tokens: int) -> list[int]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    seed = "A native SwiftLLM request appends one token at a time while its paged KV cache crosses a block boundary. "
    ids = tokenizer.encode(seed, add_special_tokens=True)
    while len(ids) < tokens:
        ids.extend(ids[: max(1, min(len(ids), tokens - len(ids)))])
    return ids[:tokens]


def _build_checkpoint_model(model_path: str, page_format: str, prompt_tokens: int, decode_tokens: int, device: torch.device):
    config = EngineConfig(
        model_path=model_path, use_dummy=False, block_size=BLOCK_SIZE,
        gpu_mem_utilization=0.80, num_cpu_blocks=0, max_seqs_in_block_table=8,
        max_blocks_per_seq=math.ceil((prompt_tokens + decode_tokens + 2) / BLOCK_SIZE) + 4,
        max_batch_size=1, max_tokens_in_batch=prompt_tokens + 2,
        kv_page_format=page_format,
    )
    model = LlamaModel(config)
    model.load_weights()
    model.init_kvcache_and_swap(config.max_blocks_per_seq)
    return model


def _timed_native_forward(model: LlamaModel, input_ids, decode_lengths, device: torch.device):
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    output = model.forward(input_ids, [0], decode_lengths, return_logits=True)
    stop.record()
    stop.synchronize()
    return output, float(start.elapsed_time(stop))


def _native_used_page_bytes(model: LlamaModel, prompt_tokens: int, current_length: int) -> int:
    shape = model.model_config
    pages = math.ceil(current_length / BLOCK_SIZE)
    page_bytes = BLOCK_SIZE * shape.num_kv_heads * shape.head_dim * 4
    return pages * model.model_config.num_layers * page_bytes


def _native_state(model: LlamaModel, prompt_tokens: int, current_length: int, phase: str, device: torch.device):
    if model.page_kv_cache is None:
        logical = _native_used_page_bytes(model, prompt_tokens, current_length)
        storage = {"mode": "dense_fp16", "logical_payload_bytes": logical, "unique_live_storage_bytes": logical, "allocated_cache_capacity_bytes": int(model.k_cache.numel() * model.k_cache.element_size() * 2)}
    else:
        storage = cache_records(model.page_kv_cache, include_packed=True)
    return {"phase": phase, "token_position": current_length, "storage": storage, "allocator": allocator_snapshot(device)}


def _run_native_checkpoint_method(model_path: str, page_format: str, prompt: list[int], decode_tokens: int, device: torch.device, demote_old: bool):
    model = _build_checkpoint_model(model_path, page_format, len(prompt), decode_tokens, device)
    token_trace = []
    timings = []
    with torch.inference_mode():
        output, elapsed = _timed_native_forward(model, [prompt], [], device)
        timings.append({"phase": "prefill", "elapsed_ms": elapsed})
        logits = output[0].float()
        next_token = int(torch.argmax(logits).item())
        token_trace.append(_native_state(model, len(prompt), len(prompt), "prefill", device))
        conversion = None
        if demote_old:
            table = model.gpu_block_manager.block_table
            old_physical = int(table[0, 0].item())
            keys = [(old_physical, layer_id) for layer_id in range(model.model_config.num_layers)]
            converted = model.page_kv_cache.demote_pages_batch(keys, "int8")
            sync(device)
            conversion = {"pages": len(keys), "elapsed_ms": converted.elapsed_ms, "before_bytes": converted.before_bytes, "after_bytes": converted.after_bytes, "reclaimed_bytes": converted.reclaimed_bytes, "selected_old_physical_block": old_physical, "all_layers": True}
        for offset in range(decode_tokens):
            output, elapsed = _timed_native_forward(model, [[next_token]], [len(prompt) + offset + 1], device)
            timings.append({"phase": "decode", "token_position": len(prompt) + offset + 1, "elapsed_ms": elapsed})
            sync(device)
            logits = output[0].float()
            next_token = int(torch.argmax(logits).item())
            token_trace.append(_native_state(model, len(prompt), len(prompt) + offset + 1, "decode", device))
    sync(device)
    before_release = allocator_snapshot(device)
    if model.page_kv_cache is not None:
        model.free_seqs_resources([0])
    del model
    gc.collect()
    torch.cuda.empty_cache()
    sync(device)
    after_release = allocator_snapshot(device)
    return {
        "method": "swiftllm_dense_fp16" if not demote_old else "swiftllm_dynamic_old_int8",
        "page_format": page_format,
        "demote_old": demote_old,
        "prompt_tokens": len(prompt),
        "decode_tokens": decode_tokens,
        "timing_unit": "milliseconds",
        "timing_boundary": "CUDA event around native SwiftLLM model.forward; no quality scoring in timing",
        "timings": timings,
        "memory_trace": token_trace,
        "conversion": conversion,
        "release": {"allocated_before": before_release["allocated_bytes"], "reserved_before": before_release["reserved_bytes"], "allocated_after": after_release["allocated_bytes"], "reserved_after": after_release["reserved_bytes"]},
        "quality": "not scored in this checkpoint smoke; greedy token trace only",
    }


def checkpoint_probe(model_path: str, device: torch.device, decode_tokens: int):
    prompt = _checkpoint_prompt(model_path, 31)
    return {
        "model_path": str(Path(model_path).resolve()),
        "model_family": "llama32_1b",
        "prompt": {"token_count": len(prompt), "tokens": prompt},
        "methods": [
            _run_native_checkpoint_method(model_path, "dense_fp16", prompt, decode_tokens, device, False),
            _run_native_checkpoint_method(model_path, "fp16", prompt, decode_tokens, device, True),
        ],
        "scope": "checkpoint-backed native SwiftLLM prefill plus repeated decode append, including a page-boundary crossing; quality is intentionally separate and not scored here",
    }


def run(args: argparse.Namespace):
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("v2 native measurement requires CUDA")
    torch.cuda.set_device(device)
    result = {
        "schema_version": 2,
        "schema": "swiftllm-kv-measurement-v2",
        "provenance": dirty_provenance() | {"source_files": list(SOURCE_FILES), "source_sha256": source_fingerprint()},
        "hardware": {
            "device": str(device),
            "name": torch.cuda.get_device_name(device),
            "capability": list(torch.cuda.get_device_capability(device)),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
        },
        "scope": {
            "scheduler_implemented": False,
            "scheduler_changed": False,
            "queue_or_swap_ranking": False,
            "quality_scoring_in_timing": False,
            "formats": ["fp16", "int8"],
            "block_size": BLOCK_SIZE,
            "group_size": GROUP_SIZE,
            "methods": ["swiftllm_dense_fp16_layout_control", "swiftllm_page_fp16", "swiftllm_page_int8_segmented"],
        },
        "measurement_contract": {
            "logical_payload": "sum of live page K/V payload and scales",
            "unique_live_storage": "deduplicated underlying tensor storages including metadata, packed copies, and listed workspaces",
            "allocator": "torch.cuda.memory_allocated/reserved plus reset per-operation peaks",
            "reusable_capacity": "observed after explicit cache free; never inferred from logical bytes or HBM free",
            "timing": "CUDA-event attention/conversion boundaries; separate synchronized memory replay",
        },
        "cases": [run_family(family, device, args.seed + index * 1000, args.attention_iterations, args.attention_repetitions) for index, family in enumerate(args.model_families)],
        "checkpoint_probe": checkpoint_probe(args.checkpoint_model_path, device, args.checkpoint_decode_tokens) if args.checkpoint_model_path else None,
        "verdict": {
            "swiftllm_dense_fp16_layout_control": "measured",
            "swiftllm_page_fp16": "measured",
            "swiftllm_page_int8_segmented": "measured_narrow_capacity_evidence",
            "full_model_request_level_capacity": "not_run",
            "quality": "not_run_in_this_harness",
        },
        "invocation": {"argv": sys.argv, "cwd": os.getcwd()},
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-families", nargs="+", choices=tuple(SHAPES), default=list(SHAPES))
    parser.add_argument("--seed", type=int, default=20261012)
    parser.add_argument("--attention-iterations", type=int, default=3)
    parser.add_argument("--attention-repetitions", type=int, default=3)
    parser.add_argument("--checkpoint-model-path", default=None)
    parser.add_argument("--checkpoint-decode-tokens", type=int, default=3)
    args = parser.parse_args()
    output = run(args)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"output": str(path), "cases": len(output["cases"])}))


if __name__ == "__main__":
    main()
