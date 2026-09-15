#!/usr/bin/env python3
"""Compare SwiftLLM FP16 with Transformers FP16, stage by stage.

This is an evidence tool, not a serving path.  It runs a matched single-request
prefill with the same token IDs, records the intermediate tensors from both
implementations, and reports the first material divergence.  The optional
intervention replaces SwiftLLM's post-RoPE Q/K tensors with the corresponding
Transformers tensors; this is deliberately kept as a causal diagnostic.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama import modeling_llama as hf_llama

ROOT = Path(__file__).resolve().parents[1]


def cpu_copy(value: torch.Tensor) -> torch.Tensor:
    return value.detach().to("cpu").clone()


def flat(value: torch.Tensor) -> torch.Tensor:
    """Normalize the one-batch HF layout to SwiftLLM's token-major layout."""
    value = value.detach()
    if value.ndim > 0 and value.shape[0] == 1:
        return value[0]
    return value


def record(store: dict[str, torch.Tensor], name: str, value: torch.Tensor) -> None:
    store[name] = cpu_copy(flat(value))


def tensor_stats(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    reference = reference.float()
    candidate = candidate.float()
    if tuple(reference.shape) != tuple(candidate.shape):
        return {
            "shape_reference": list(reference.shape),
            "shape_candidate": list(candidate.shape),
            "shape_match": False,
        }
    delta = (candidate - reference).abs()
    ref_norm = reference.norm().item()
    return {
        "shape_reference": list(reference.shape),
        "shape_candidate": list(candidate.shape),
        "shape_match": True,
        "max_abs": float(delta.max().item()),
        "mean_abs": float(delta.mean().item()),
        "relative_l2": float(delta.norm().item() / max(ref_norm, 1e-12)),
        "finite_reference": bool(torch.isfinite(reference).all()),
        "finite_candidate": bool(torch.isfinite(candidate).all()),
    }


def source_sha256(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def hf_trace(model_path: str, token_ids: torch.Tensor) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Capture Transformers' eager Llama path."""
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        torch_dtype=torch.float16,
        attn_implementation="eager",
    ).cuda().eval()
    traces: dict[str, torch.Tensor] = {}
    hooks = []

    hooks.append(model.model.embed_tokens.register_forward_hook(
        lambda _m, _i, out: record(traces, "embedding", out)
    ))

    for layer_id, layer in enumerate(model.model.layers):
        prefix = f"layers.{layer_id}"
        hooks.append(layer.register_forward_pre_hook(
            lambda _m, args, p=prefix: record(traces, f"{p}.layer_input", args[0])
        ))
        hooks.append(layer.input_layernorm.register_forward_hook(
            lambda _m, _i, out, p=prefix: record(traces, f"{p}.attn_norm", out)
        ))
        for projection in ("q", "k", "v"):
            module = getattr(layer.self_attn, f"{projection}_proj")
            hooks.append(module.register_forward_hook(
                lambda _m, _i, out, p=prefix, proj=projection: record(traces, f"{p}.{proj}_raw", out)
            ))
        hooks.append(layer.self_attn.o_proj.register_forward_pre_hook(
            lambda _m, args, p=prefix: record(traces, f"{p}.attn_context", args[0])
        ))
        hooks.append(layer.self_attn.o_proj.register_forward_hook(
            lambda _m, _i, out, p=prefix: record(traces, f"{p}.o_proj", out)
        ))
        hooks.append(layer.post_attention_layernorm.register_forward_hook(
            lambda _m, _i, out, p=prefix: record(traces, f"{p}.ffn_norm", out)
        ))
        hooks.append(layer.mlp.gate_proj.register_forward_hook(
            lambda _m, _i, out, p=prefix: record(traces, f"{p}.ffn_gate", out)
        ))
        hooks.append(layer.mlp.up_proj.register_forward_hook(
            lambda _m, _i, out, p=prefix: record(traces, f"{p}.ffn_up", out)
        ))
        hooks.append(layer.mlp.down_proj.register_forward_pre_hook(
            lambda _m, args, p=prefix: record(traces, f"{p}.ffn_activation", args[0])
        ))
        hooks.append(layer.mlp.down_proj.register_forward_hook(
            lambda _m, _i, out, p=prefix: record(traces, f"{p}.ffn_output", out)
        ))
        hooks.append(layer.register_forward_hook(
            lambda _m, _inputs, out, p=prefix: record(traces, f"{p}.layer_output", out[0])
        ))

    hooks.append(model.model.norm.register_forward_hook(
        lambda _m, _i, out: record(traces, "final_norm_all", out)
    ))

    original_rope = hf_llama.apply_rotary_pos_emb
    original_eager = hf_llama.eager_attention_forward
    original_rope_embedding = hf_llama.LlamaRotaryEmbedding.forward
    rope_index = {"value": 0}

    def capture_rope(q, k, cos, sin, *args, **kwargs):
        idx = rope_index["value"]
        rope_index["value"] += 1
        rotated_q, rotated_k = original_rope(q, k, cos, sin, *args, **kwargs)
        record(traces, f"layers.{idx}.q_rope", rotated_q.transpose(1, 2))
        record(traces, f"layers.{idx}.k_rope", rotated_k.transpose(1, 2))
        return rotated_q, rotated_k

    def capture_eager(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
        output, weights = original_eager(
            module, query, key, value, attention_mask, scaling, dropout, **kwargs
        )
        idx = module.layer_idx
        record(traces, f"layers.{idx}.attn_context", output)
        record(traces, f"layers.{idx}.attn_weights", weights)
        return output, weights

    def capture_rope_embedding(module, x, position_ids):
        cos, sin = original_rope_embedding(module, x, position_ids)
        record(traces, "rope.cos_half", cos[..., : cos.shape[-1] // 2])
        record(traces, "rope.sin_half", sin[..., : sin.shape[-1] // 2])
        record(traces, "positions", position_ids)
        return cos, sin

    hf_llama.apply_rotary_pos_emb = capture_rope
    hf_llama.eager_attention_forward = capture_eager
    hf_llama.LlamaRotaryEmbedding.forward = capture_rope_embedding
    try:
        with torch.inference_mode():
            output = model(token_ids.cuda().unsqueeze(0), use_cache=False, return_dict=True)
        record(traces, "logits", output.logits[:, -1, :])
    finally:
        hf_llama.apply_rotary_pos_emb = original_rope
        hf_llama.eager_attention_forward = original_eager
        hf_llama.LlamaRotaryEmbedding.forward = original_rope_embedding
        for hook in hooks:
            hook.remove()

    for layer_id in range(len(model.model.layers)):
        prefix = f"layers.{layer_id}"
        traces[f"{prefix}.ffn_up_gate"] = torch.cat(
            (traces[f"{prefix}.ffn_up"], traces[f"{prefix}.ffn_gate"]), dim=-1
        )
    traces["final_norm"] = traces["final_norm_all"][-1]

    metadata = {
        "weights_dtype": str(next(model.parameters()).dtype),
        "logits_top1": int(output.logits[0, -1].argmax().item()),
        "logits_top5": [int(x) for x in output.logits[0, -1].topk(5).indices.tolist()],
        "num_layers": len(model.model.layers),
        "num_q_heads": model.config.num_attention_heads,
        "num_kv_heads": model.config.num_key_value_heads,
        "head_dim": model.config.head_dim,
        "attention_implementation": model.config._attn_implementation,
    }
    del model, output
    gc.collect()
    torch.cuda.empty_cache()
    return traces, metadata


def swift_weight_audit(model: Any, model_path: str) -> dict[str, Any]:
    """Compare every loaded FP16 tensor with its safetensors source tensor."""
    files = sorted(Path(model_path).glob("*.safetensors"))
    index_path = Path(model_path) / "model.safetensors.index.json"
    if index_path.exists():
        weight_map = json.loads(index_path.read_text())["weight_map"]
    else:
        weight_map = {}
    handles: dict[Path, Any] = {}

    def source_tensor(key: str) -> torch.Tensor:
        filename = weight_map.get(key, files[0].name)
        path = Path(model_path) / filename
        if path not in handles:
            handles[path] = safe_open(str(path), framework="pt", device="cpu")
        return handles[path].get_tensor(key)

    items = [("wte", "model.embed_tokens.weight"), ("lm_head", "lm_head.weight"), ("final_norm", "model.norm.weight")]
    for layer_id, layer in enumerate(model.weight.layers):
        for attr, key_suffix in (
            ("attn_norm", "input_layernorm.weight"),
            ("q_proj", "self_attn.q_proj.weight"),
            ("k_proj", "self_attn.k_proj.weight"),
            ("v_proj", "self_attn.v_proj.weight"),
            ("o_proj", "self_attn.o_proj.weight"),
            ("ffn_norm", "post_attention_layernorm.weight"),
            ("up_proj", "mlp.up_proj.weight"),
            ("gate_proj", "mlp.gate_proj.weight"),
            ("down_proj", "mlp.down_proj.weight"),
        ):
            # up_proj and gate_proj are deleted after concatenation.  They are
            # checked through the two slices of up_gate_proj below.
            if attr in {"up_proj", "gate_proj"}:
                continue
            items.append((f"layers.{layer_id}.{attr}", f"model.layers.{layer_id}.{key_suffix}"))

    checks = []
    # Check the normal registered weights and the split concatenated FFN weight.
    for attr, key in items:
        if attr.startswith("layers."):
            layer_id, layer_attr = attr.split(".")[1:]
            loaded = getattr(model.weight.layers[int(layer_id)], layer_attr)
        else:
            loaded = getattr(model.weight, attr)
        if key == "lm_head.weight" and not any(path.name.startswith("model") for path in files):
            pass
        try:
            source = source_tensor(key).to(dtype=torch.float16)
        except Exception:
            # Llama 3.2 ties lm_head to embed_tokens; its source key is the
            # embedding tensor and the Swift loader intentionally uses it.
            if key == "lm_head.weight":
                source = source_tensor("model.embed_tokens.weight").to(dtype=torch.float16)
            else:
                raise
        loaded_cpu = loaded.detach().cpu()
        checks.append({
            "name": attr,
            "source_key": key,
            "exact": bool(torch.equal(loaded_cpu, source)),
            "shape": list(loaded_cpu.shape),
        })
    # Concatenated up/gate storage is ordered [up, gate], matching the kernel.
    for layer_id, layer in enumerate(model.weight.layers):
        up = source_tensor(f"model.layers.{layer_id}.mlp.up_proj.weight").to(dtype=torch.float16)
        gate = source_tensor(f"model.layers.{layer_id}.mlp.gate_proj.weight").to(dtype=torch.float16)
        expected = torch.cat((up, gate), dim=0)
        actual = layer.up_gate_proj.detach().cpu()
        checks.append({
            "name": f"layers.{layer_id}.up_gate_proj_concat",
            "source_key": "up_proj + gate_proj",
            "exact": bool(torch.equal(actual, expected)),
            "shape": list(actual.shape),
        })
    for handle in handles.values():
        handle.__exit__(None, None, None)
    mismatches = [item for item in checks if not item["exact"]]
    return {"checked": len(checks), "mismatches": len(mismatches), "examples": mismatches[:5]}


def swift_trace(
    model_path: str,
    token_ids: torch.Tensor,
    intervention: dict[str, torch.Tensor] | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Capture SwiftLLM's eager FP16 model path without a KV cache."""
    # Imports are intentionally local so the script can still inspect HF in a
    # clean environment without importing SwiftLLM's CUDA extension first.
    from swiftllm.engine_config import EngineConfig
    from swiftllm.worker import model as swift_model_module
    from swiftllm.worker.layers import post_layer as swift_post
    from swiftllm.worker.layers import transformer_layer as swift_layer

    config = EngineConfig(
        model_path=model_path,
        use_dummy=False,
        block_size=16,
        gpu_mem_utilization=0.97,
        num_cpu_blocks=0,
        max_seqs_in_block_table=8,
        max_blocks_per_seq=2048,
        max_batch_size=8,
        max_tokens_in_batch=max(2048, token_ids.numel() + 16),
    )
    model = swift_model_module.LlamaModel(config)
    model.load_weights()
    traces: dict[str, torch.Tensor] = {}
    record(traces, "rope.cos_half", model._cos_cached[: token_ids.numel()])
    record(traces, "rope.sin_half", model._sin_cached[: token_ids.numel()])
    record(traces, "embedding", model.pre_layer.forward(token_ids.cuda()))
    state = {"layer": -1, "norm_count": 0, "ffn_count": 0}

    old_fused = swift_layer.fused_add_rmsnorm_inplace
    old_linear = swift_layer.linear
    old_rope = swift_layer.rotary_embedding_inplace
    old_silu = swift_layer.silu_and_mul_inplace
    old_flash = swift_layer.vllm_flash_attn.flash_attn_varlen_func
    old_post_norm = swift_post.rmsnorm_inplace
    old_post_linear = swift_post.linear

    def fused(x, residual, weight, eps):
        if state["norm_count"] == 0:
            state["layer"] += 1
            state["ffn_count"] = 0
            layer = state["layer"]
            record(traces, f"layers.{layer}.layer_input", x + residual)
            state["norm_count"] = 1
            old_fused(x, residual, weight, eps)
            record(traces, f"layers.{layer}.attn_norm", x)
        else:
            layer = state["layer"]
            state["norm_count"] = 0
            old_fused(x, residual, weight, eps)
            state["residual_after_attention"] = residual.detach().clone()
            record(traces, f"layers.{layer}.ffn_norm", x)
        return None

    def linear(x, weight, *args, **kwargs):
        output = old_linear(x, weight, *args, **kwargs)
        projection = kwargs.get("projection")
        layer = state["layer"]
        if projection in {"q", "k", "v"}:
            record(traces, f"layers.{layer}.{projection}_raw", output)
        elif projection == "o":
            record(traces, f"layers.{layer}.o_proj", output)
        elif projection == "ffn":
            if state["ffn_count"] == 0:
                record(traces, f"layers.{layer}.ffn_up_gate", output)
            else:
                record(traces, f"layers.{layer}.ffn_output", output)
                record(
                    traces,
                    f"layers.{layer}.layer_output",
                    output + state["residual_after_attention"],
                )
            state["ffn_count"] += 1
        return output

    def rope(q, k, infer_state):
        layer = state["layer"]
        q_before, k_before = q.clone(), k.clone()
        old_rope(q, k, infer_state)
        record(traces, f"layers.{layer}.q_rope", q)
        record(traces, f"layers.{layer}.k_rope", k)
        if intervention is not None:
            q.copy_(intervention[f"layers.{layer}.q_rope"].to(device=q.device, dtype=q.dtype))
            k.copy_(intervention[f"layers.{layer}.k_rope"].to(device=k.device, dtype=k.dtype))
        return None

    def silu(x):
        layer = state["layer"]
        old_silu(x)
        record(traces, f"layers.{layer}.ffn_activation", x[:, : x.shape[1] // 2])
        return None

    def flash(q, k, v, *args, **kwargs):
        layer = state["layer"]
        output = old_flash(q, k, v, *args, **kwargs)
        record(traces, f"layers.{layer}.attn_context", output.reshape(output.shape[0], -1))
        return output

    def final_norm(x, weight, eps):
        record(traces, "final_norm_input", x)
        old_post_norm(x, weight, eps)
        record(traces, "final_norm", x)
        return None

    def final_linear(x, weight, *args, **kwargs):
        output = old_post_linear(x, weight, *args, **kwargs)
        record(traces, "logits", output)
        return output

    swift_layer.fused_add_rmsnorm_inplace = fused
    swift_layer.linear = linear
    swift_layer.rotary_embedding_inplace = rope
    swift_layer.silu_and_mul_inplace = silu
    swift_layer.vllm_flash_attn.flash_attn_varlen_func = flash
    swift_post.rmsnorm_inplace = final_norm
    swift_post.linear = final_linear
    try:
        with torch.inference_mode():
            output = model.forward(
                [token_ids.tolist()], [0], [], ignore_kvcache=True, return_logits=True
            )
        metadata = {
            "weights_dtype": "torch.float16",
            "logits_top1": int(output[0].argmax().item()),
            "logits_top5": [int(x) for x in output[0].topk(5).indices.tolist()],
            "num_layers": model.model_config.num_layers,
            "num_q_heads": model.model_config.num_q_heads,
            "num_kv_heads": model.model_config.num_kv_heads,
            "head_dim": model.model_config.head_dim,
            "attention_implementation": "vllm_flash_attn.flash_attn_varlen_func",
            "weight_audit": swift_weight_audit(model, model_path),
        }
    finally:
        swift_layer.fused_add_rmsnorm_inplace = old_fused
        swift_layer.linear = old_linear
        swift_layer.rotary_embedding_inplace = old_rope
        swift_layer.silu_and_mul_inplace = old_silu
        swift_layer.vllm_flash_attn.flash_attn_varlen_func = old_flash
        swift_post.rmsnorm_inplace = old_post_norm
        swift_post.linear = old_post_linear
        del model
        gc.collect()
        torch.cuda.empty_cache()
    return traces, metadata


def compare_traces(reference: dict[str, torch.Tensor], candidate: dict[str, torch.Tensor]) -> dict[str, Any]:
    rows = []
    for name in sorted(set(reference) | set(candidate)):
        if name not in reference or name not in candidate:
            rows.append({"name": name, "present_reference": name in reference, "present_candidate": name in candidate})
            continue
        rows.append({"name": name, **tensor_stats(reference[name], candidate[name])})
    return {"rows": rows}


def stage_order(num_layers: int) -> list[str]:
    stages = ["embedding"]
    for layer_id in range(num_layers):
        p = f"layers.{layer_id}"
        stages.extend([
            f"{p}.layer_input", f"{p}.attn_norm", f"{p}.q_raw", f"{p}.k_raw", f"{p}.v_raw",
            f"{p}.q_rope", f"{p}.k_rope", f"{p}.attn_context", f"{p}.o_proj", f"{p}.ffn_norm",
            f"{p}.ffn_activation", f"{p}.ffn_output", f"{p}.layer_output",
        ])
    stages.extend(["final_norm", "logits"])
    return stages


def first_divergence(comparison: dict[str, Any], threshold: float = 0.01) -> dict[str, Any] | None:
    for row in comparison["rows"]:
        if row.get("name") not in comparison.get("ordered_names", []):
            continue
        if row.get("shape_match") and row.get("max_abs", 0.0) > threshold:
            return row
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompt", default="Life blooms like a flower, far away")
    parser.add_argument("--intervention", action="store_true")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    token_ids = tokenizer(args.prompt, return_tensors="pt").input_ids[0]
    source_files = [
        ROOT / "scripts/swiftllm_fp16_diagnosis.py",
        ROOT / "vendor/swiftLLM/swiftllm/worker/model.py",
        ROOT / "vendor/swiftLLM/swiftllm/worker/layers/transformer_layer.py",
        ROOT / "vendor/swiftLLM/swiftllm/worker/layers/post_layer.py",
        ROOT / "vendor/swiftLLM/swiftllm/worker/kernels/rotary_emb.py",
        ROOT / "vendor/swiftLLM/swiftllm/worker/kernels/rmsnorm.py",
        ROOT / "vendor/swiftLLM/swiftllm/worker/weight.py",
    ]

    hf, hf_meta = hf_trace(args.model_path, token_ids)
    intervention = {
        key: value for key, value in hf.items()
        if key.endswith(".q_rope") or key.endswith(".k_rope")
    }
    sw, sw_meta = swift_trace(args.model_path, token_ids, intervention if args.intervention else None)
    comparison = compare_traces(hf, sw)
    ordered = stage_order(hf_meta["num_layers"])
    comparison["ordered_names"] = ordered
    ordered_rows = [row for name in ordered for row in comparison["rows"] if row.get("name") == name]
    first = next((row for row in ordered_rows if row.get("shape_match") and row.get("max_abs", 0.0) > 0.01), None)

    logits_reference = hf["logits"].flatten()
    logits_candidate = sw["logits"].flatten()
    output = {
        "schema": "swiftllm-fp16-diagnosis-v1",
        "model_path": args.model_path,
        "prompt": args.prompt,
        "token_ids": [int(x) for x in token_ids.tolist()],
        "token_text": tokenizer.convert_ids_to_tokens(token_ids.tolist()),
        "sequence_length": int(token_ids.numel()),
        "positions": list(range(int(token_ids.numel()))),
        "masking": {
            "single_sequence_causal": True,
            "swift_flash_attention_causal": True,
            "future_attention_weights_not_compared_in_swift_kernel": True,
            "transformers_max_future_attention_weight": max(
                float(value.triu(1).abs().max())
                for name, value in hf.items() if name.endswith(".attn_weights")
            ),
        },
        "gqa": {"q_heads": sw_meta["num_q_heads"], "kv_heads": sw_meta["num_kv_heads"], "head_dim": sw_meta["head_dim"], "group_size": sw_meta["num_q_heads"] // sw_meta["num_kv_heads"]},
        "environment": {
            "git_head": git_head(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(),
            "swift_source_sha256": source_sha256(source_files),
        },
        "transformers": hf_meta,
        "swiftllm": sw_meta,
        "intervention": {"enabled": bool(args.intervention), "replaced": sorted(intervention.keys()) if args.intervention else []},
        "comparison": comparison,
        "first_divergence_over_0.01_abs": first,
        "logits": {
            "stats": tensor_stats(logits_reference, logits_candidate),
            "reference_top1": hf_meta["logits_top1"],
            "candidate_top1": sw_meta["logits_top1"],
            "top1_match": hf_meta["logits_top1"] == sw_meta["logits_top1"],
            "reference_top5": hf_meta["logits_top5"],
            "candidate_top5": sw_meta["logits_top5"],
        },
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({
        "output": args.output,
        "model": Path(args.model_path).name,
        "length": len(output["token_ids"]),
        "first_divergence": first,
        "logits": output["logits"],
        "weight_audit": sw_meta["weight_audit"],
        "intervention": output["intervention"]["enabled"],
    }, indent=2))


if __name__ == "__main__":
    main()
