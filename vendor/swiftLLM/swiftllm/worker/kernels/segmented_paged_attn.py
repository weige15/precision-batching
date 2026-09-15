"""Format-specialized segmented paged attention for FP16/INT8 KV pages.

The kernels keep the existing SwiftLLM online-softmax structure but split a
request's page table into homogeneous FP16 and INT8 segments.  INT8 values are
scaled in registers from the page's existing per-group FP16 scales; no dense
FP16 cache is materialized.  This is an enabling path and intentionally keeps
the control-plane segment construction in Python.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _segment_attention_phase1(
    mid_o,
    mid_logexpsum,
    q,
    k_cache,
    v_cache,
    k_scales,
    v_scales,
    block_table,
    seq_ids,
    decoding_seq_lens,
    segment_batch,
    segment_start,
    segment_blocks,
    segment_slot,
    page_slots,
    num_layers,
    num_q_heads,
    num_kv_heads,
    max_blocks_per_seq,
    max_segments,
    page_numel,
    scale_count,
    softmax_scale,
    num_segments,
    max_blocks_per_segment: tl.constexpr,
    block_size: tl.constexpr,
    head_dim: tl.constexpr,
    num_my_heads: tl.constexpr,
    group_size: tl.constexpr,
    INT8: tl.constexpr,
):
    segment_id = tl.program_id(0)
    q_head_id = tl.program_id(1)
    valid_segment = segment_id < num_segments
    batch_id = tl.load(segment_batch + segment_id, mask=valid_segment, other=0).to(tl.int64)
    start_block = tl.load(segment_start + segment_id, mask=valid_segment, other=0).to(tl.int64)
    num_blocks = tl.load(segment_blocks + segment_id, mask=valid_segment, other=0).to(tl.int64)
    output_slot = tl.load(segment_slot + segment_id, mask=valid_segment, other=0).to(tl.int64)
    seq_id = tl.load(seq_ids + batch_id).to(tl.int64)
    seq_len = tl.load(decoding_seq_lens + batch_id).to(tl.int64)
    kv_head_id = q_head_id // num_my_heads

    q_offsets = batch_id * num_q_heads * head_dim + q_head_id * head_dim + tl.arange(0, head_dim)
    my_q = tl.load(q + q_offsets, mask=valid_segment).to(tl.float32)
    max_score = float("-1e20")
    sum_exp = 0.0
    acc = tl.zeros([head_dim], dtype=tl.float32)

    for block_i in tl.static_range(0, max_blocks_per_segment):
        block_valid = valid_segment & (block_i < num_blocks)
        physical_id = tl.load(
            block_table + seq_id * max_blocks_per_seq + start_block + block_i,
            mask=block_valid,
            other=0,
        ).to(tl.int64)
        local_token_ids = tl.arange(0, block_size)
        token_ids = (start_block + block_i) * block_size + local_token_ids
        token_valid = block_valid & (token_ids < seq_len)
        page_slot = tl.load(
            page_slots + physical_id,
            mask=block_valid,
            other=0,
        ).to(tl.int64)
        element_offsets = (
            page_slot * page_numel
            + local_token_ids[:, None] * num_kv_heads * head_dim
            + kv_head_id * head_dim
            + tl.arange(0, head_dim)[None, :]
        )
        if INT8:
            k_values = tl.load(k_cache + element_offsets, mask=token_valid[:, None], other=0).to(tl.float32)
            v_values = tl.load(v_cache + element_offsets, mask=token_valid[:, None], other=0).to(tl.float32)
            scale_offsets = page_slot * scale_count + (
                local_token_ids[:, None] * num_kv_heads * head_dim
                + kv_head_id * head_dim
                + tl.arange(0, head_dim)[None, :]
            ) // group_size
            k_values *= tl.load(k_scales + scale_offsets, mask=token_valid[:, None], other=1.0).to(tl.float32)
            v_values *= tl.load(v_scales + scale_offsets, mask=token_valid[:, None], other=1.0).to(tl.float32)
        else:
            k_values = tl.load(k_cache + element_offsets, mask=token_valid[:, None], other=0).to(tl.float32)
            v_values = tl.load(v_cache + element_offsets, mask=token_valid[:, None], other=0).to(tl.float32)

        scores = tl.sum(my_q[None, :] * k_values, axis=1) * softmax_scale
        scores = tl.where(token_valid, scores, float("-1e20"))
        cur_max = tl.max(scores, axis=0)
        new_max = tl.maximum(max_score, cur_max)
        exp_scores = tl.exp(scores - new_max)
        old_scale = tl.exp(max_score - new_max)
        acc = acc * old_scale + tl.sum(exp_scores[:, None] * v_values, axis=0)
        sum_exp = sum_exp * old_scale + tl.sum(exp_scores, axis=0)
        max_score = new_max

    output_offsets = (
        (batch_id * num_q_heads + q_head_id) * max_segments + output_slot
    ) * head_dim + tl.arange(0, head_dim)
    log_offsets = (batch_id * num_q_heads + q_head_id) * max_segments + output_slot
    tl.store(mid_o + output_offsets, acc / sum_exp, mask=valid_segment)
    tl.store(mid_logexpsum + log_offsets, tl.log(sum_exp) + max_score, mask=valid_segment)


@triton.jit
def _segment_attention_phase2(
    mid_o,
    mid_logexpsum,
    segment_valid,
    out,
    num_q_heads,
    max_segments,
    head_dim: tl.constexpr,
    total_segments: tl.constexpr,
):
    batch_id = tl.program_id(0)
    q_head_id = tl.program_id(1)
    max_score = float("-1e20")
    sum_exp = 0.0
    acc = tl.zeros([head_dim], dtype=tl.float32)
    for segment_id in tl.static_range(0, total_segments):
        valid = tl.load(segment_valid + batch_id * max_segments + segment_id)
        log_offset = (batch_id * num_q_heads + q_head_id) * max_segments + segment_id
        value_offset = log_offset * head_dim + tl.arange(0, head_dim)
        loaded_log = tl.load(mid_logexpsum + log_offset)
        loaded_value = tl.load(mid_o + value_offset)
        current_log = tl.where(valid, loaded_log, max_score)
        current_value = tl.where(valid, loaded_value, 0.0)
        new_max = tl.maximum(max_score, current_log)
        old_scale = tl.exp(max_score - new_max)
        current_scale = tl.where(valid, tl.exp(loaded_log - new_max), 0.0)
        acc = acc * old_scale + current_scale * current_value
        sum_exp = sum_exp * old_scale + current_scale
        max_score = new_max
    output_offset = (batch_id * num_q_heads + q_head_id) * head_dim + tl.arange(0, head_dim)
    tl.store(out + output_offset, (acc / sum_exp).to(tl.float16))


def segmented_paged_attention(
    q: torch.Tensor,
    storage,
    block_table: torch.Tensor,
    seq_ids: torch.Tensor,
    decoding_seq_lens: torch.Tensor,
    out: torch.Tensor,
    segments: dict[str, list[tuple[int, int, int, int]]],
    model_config,
    engine_config,
    num_layers: int,
    layer_id: int,
    max_blocks_per_segment: int = 32,
) -> dict[str, int]:
    """Run format-specialized segment kernels and reduce their partials.

    ``segments[format]`` entries are ``(batch_id, logical_start, block_count,
    output_slot)``.  The storage arenas are indexed by physical block and
    layer through compact slot maps, so only the selected format's arena is
    read by each phase-1 launch.
    """
    if q.device.type != "cuda":
        raise RuntimeError("segmented INT8 attention requires CUDA")
    if not q.is_contiguous() or not out.is_contiguous():
        raise ValueError("q and out must be contiguous")
    batch = q.shape[0]
    q_heads = model_config.num_q_heads
    total_segments = max((row[3] for rows in segments.values() for row in rows), default=-1) + 1
    if total_segments == 0:
        out.zero_()
        return {"fp16_segments": 0, "int8_segments": 0, "total_segments": 0}
    # Entries are already assigned global output slots by the caller.  The
    # compact descriptors below are stored on GPU once per attention call.
    segment_valid = torch.zeros((batch, total_segments), dtype=torch.int32, device=q.device)
    mid_o = torch.zeros((batch, q_heads, total_segments, model_config.head_dim), dtype=torch.float32, device=q.device)
    mid_log = torch.full((batch, q_heads, total_segments), -1.0e20, dtype=torch.float32, device=q.device)

    def launch(format_name: str, cache_format: int) -> int:
        rows = segments.get(format_name, [])
        if not rows:
            return 0
        batch_ids = torch.tensor([row[0] for row in rows], dtype=torch.int32, device=q.device)
        starts = torch.tensor([row[1] for row in rows], dtype=torch.int32, device=q.device)
        counts = torch.tensor([row[2] for row in rows], dtype=torch.int32, device=q.device)
        slots = torch.tensor([row[3] for row in rows], dtype=torch.int32, device=q.device)
        segment_valid[batch_ids, slots] = 1
        grid = (len(rows), q_heads)
        if cache_format == 0:
            k_cache, v_cache = storage.fp16_k, storage.fp16_v
            k_scales = v_scales = storage.empty_scales
        else:
            k_cache, v_cache = storage.int8_k, storage.int8_v
            k_scales, v_scales = storage.int8_k_scales, storage.int8_v_scales
        slot_map_k = storage.fp16_slots if cache_format == 0 else storage.int8_slots
        # Segment physical IDs are represented by the original block table;
        # slots maps are indexed inside the kernel by physical block/layer.
        # We remap block IDs to compact arena slots in temporary table views.
        k_slot = slot_map_k
        v_slot = slot_map_k
        # The compact arenas use a physical-page slot map.  A gather table is
        # materialized once per call so the Triton kernel can use one pointer.
        physical_slots = k_slot
        _segment_attention_phase1[grid](
            mid_o, mid_log,
            q, k_cache, v_cache, k_scales, v_scales,
            block_table, seq_ids, decoding_seq_lens,
            batch_ids, starts, counts, slots,
            (storage.int8_slots if cache_format == 1 else storage.fp16_slots)[layer_id],
            num_layers, q_heads, model_config.num_kv_heads,
            block_table.stride(0), total_segments,
            storage.page_numel, storage.scale_count,
            model_config.head_dim ** -0.5,
            len(rows), max(1, max(row[2] for row in rows)),
            engine_config.block_size, model_config.head_dim,
            q_heads // model_config.num_kv_heads, storage.group_size,
            INT8=(cache_format == 1), num_warps=2, num_stages=2,
        )
        return len(rows)

    fp16_count = launch("fp16", 0)
    int8_count = launch("int8", 1)
    if total_segments == 1:
        out.copy_(mid_o[:, :, 0, :].reshape_as(out))
    else:
        _segment_attention_phase2[(batch, q_heads)](
            mid_o, mid_log, segment_valid, out,
            q_heads, total_segments, model_config.head_dim, total_segments,
            num_warps=2,
        )
    return {"fp16_segments": fp16_count, "int8_segments": int8_count, "total_segments": total_segments}
