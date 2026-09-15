#!/usr/bin/env python3
"""Check matched FP16 prefill/decode logits and SwiftLLM KV layout."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def stats(a: torch.Tensor, b: torch.Tensor) -> dict:
    a, b = a.float(), b.float()
    d = (a - b).abs()
    return {
        "shape_reference": list(a.shape),
        "shape_candidate": list(b.shape),
        "max_abs": float(d.max()),
        "mean_abs": float(d.mean()),
        "relative_l2": float(d.norm() / max(a.norm(), torch.tensor(1e-12))),
    }


def cache_layer(cache, layer_id: int):
    if hasattr(cache, "layers"):
        layer = cache.layers[layer_id]
        return layer.keys, layer.values
    layer = cache[layer_id]
    return layer[0], layer[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--prompt", default="Life blooms like a flower, far away")
    args = ap.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    ids = tokenizer(args.prompt, return_tensors="pt").input_ids[0]
    n = int(ids.numel())

    hf = AutoModelForCausalLM.from_pretrained(
        args.model_path, local_files_only=True, torch_dtype=torch.float16,
        attn_implementation="eager",
    ).cuda().eval()
    with torch.inference_mode():
        hf_prefill = hf(ids[None].cuda(), use_cache=True, return_dict=True)
        next_id = int(hf_prefill.logits[0, -1].argmax())
        hf_kv = [cache_layer(hf_prefill.past_key_values, i) for i in range(hf.config.num_hidden_layers)]
        hf_kv = [(k[0].detach().cpu().clone(), v[0].detach().cpu().clone()) for k, v in hf_kv]
        hf_decode = hf(torch.tensor([[next_id]], device="cuda"),
                       past_key_values=hf_prefill.past_key_values,
                       use_cache=True, return_dict=True)
    hf_prefill_logits = hf_prefill.logits[0, -1].cpu()
    hf_decode_logits = hf_decode.logits[0, -1].cpu()
    del hf, hf_prefill, hf_decode
    gc.collect()
    torch.cuda.empty_cache()

    from swiftllm.engine_config import EngineConfig
    from swiftllm.worker.model import LlamaModel

    config = EngineConfig(
        model_path=args.model_path, use_dummy=False, block_size=16,
        gpu_mem_utilization=0.97, num_cpu_blocks=0, max_seqs_in_block_table=8,
        max_blocks_per_seq=2048, max_batch_size=8, max_tokens_in_batch=max(2048, n + 16),
    )
    model = LlamaModel(config)
    model.load_weights()
    num_blocks = (n + 1 + config.block_size - 1) // config.block_size + 2
    model.init_kvcache_and_swap(num_blocks)
    with torch.inference_mode():
        sw_prefill = model.forward([ids.tolist()], [0], [], return_logits=True)
        sw_decode = model.forward([[next_id]], [0], [n + 1], return_logits=True)
    sw_prefill_logits = sw_prefill[0].cpu()
    sw_decode_logits = sw_decode[0].cpu()

    used_blocks = int(model.gpu_block_manager.num_seq_allocated_blocks[0].item())
    physical_ids = model.gpu_block_manager.block_table[0, :used_blocks].tolist()
    k_parts, v_parts = [], []
    for block_id in physical_ids:
        k_parts.append(model.k_cache[block_id, :, :, :, :])
        v_parts.append(model.v_cache[block_id, :, :, :, :])
    # Physical layout is [block, layer, kv-head, offset, head-dim].
    sw_k = torch.cat(k_parts, dim=2)[:, :, :n, :].cpu()
    sw_v = torch.cat(v_parts, dim=2)[:, :, :n, :].cpu()
    kv_rows = []
    for layer_id, (ref_k, ref_v) in enumerate(hf_kv):
        kv_rows.append({
            "layer": layer_id,
            "k": stats(ref_k, sw_k[layer_id]),
            "v": stats(ref_v, sw_v[layer_id]),
        })

    result = {
        "schema": "swiftllm-fp16-decode-check-v1",
        "model_path": args.model_path,
        "prompt": args.prompt,
        "token_ids": [int(x) for x in ids.tolist()],
        "decode_input_token_id": next_id,
        "sequence_length_after_decode": n + 1,
        "weights_dtype": str(model.weight.wte.dtype),
        "gqa": {"q_heads": model.model_config.num_q_heads, "kv_heads": model.model_config.num_kv_heads, "head_dim": model.model_config.head_dim},
        "kv_layout": {
            "swift_shape": list(sw_k.shape),
            "expected_shape": [model.model_config.num_layers, model.model_config.num_kv_heads, n, model.model_config.head_dim],
            "physical_block_ids": [int(x) for x in physical_ids],
            "block_size": config.block_size,
            "rows": kv_rows,
        },
        "prefill_logits": {
            "stats": stats(hf_prefill_logits, sw_prefill_logits),
            "reference_top1": int(hf_prefill_logits.argmax()),
            "candidate_top1": int(sw_prefill_logits.argmax()),
            "top1_match": bool(hf_prefill_logits.argmax() == sw_prefill_logits.argmax()),
        },
        "decode_logits": {
            "stats": stats(hf_decode_logits, sw_decode_logits),
            "reference_top1": int(hf_decode_logits.argmax()),
            "candidate_top1": int(sw_decode_logits.argmax()),
            "top1_match": bool(hf_decode_logits.argmax() == sw_decode_logits.argmax()),
        },
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "output": args.output,
        "prefill": result["prefill_logits"],
        "decode": result["decode_logits"],
        "kv_max_k": max(row["k"]["max_abs"] for row in kv_rows),
        "kv_max_v": max(row["v"]["max_abs"] for row in kv_rows),
    }, indent=2))


if __name__ == "__main__":
    main()
