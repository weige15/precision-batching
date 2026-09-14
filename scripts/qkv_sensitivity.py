#!/usr/bin/env python3
"""Offline Q/K/V projection sensitivity harness for Llama checkpoints.

This intentionally uses an eager weight-only fake-quantization proxy: each
weight is quantized per 128-input-channel group and immediately dequantized
back to FP16 before Hugging Face executes the model. It is useful for numerical
sensitivity only. It is not a packed low-bit kernel, a serving benchmark, or
evidence of low-bit acceleration.

The harness compares paired logits on the same prompts and reference decode
prefix. It reports prefill and one-token-at-a-time decode errors, prompt-level
records, and optional one-projection/one-layer sweeps.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import platform
import importlib.util
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Allow running this file directly from a checkout without installing the repo.
ROOT = Path(__file__).resolve().parents[1]
SWIFT_ROOT = ROOT / "vendor" / "swiftLLM"
C_SRC = SWIFT_ROOT / "csrc"
if str(C_SRC) not in sys.path:
    sys.path.insert(0, str(C_SRC))
if str(SWIFT_ROOT) not in sys.path:
    sys.path.insert(0, str(SWIFT_ROOT))
try:
    from swiftllm.precision import quantize_dequantize  # noqa: E402
except ModuleNotFoundError as error:
    # The numerical harness does not need SwiftLLM's CUDA swap extension. Load
    # the shared helper directly when a checkout has not built ``swiftllm_c``.
    if error.name != "swiftllm_c":
        raise
    spec = importlib.util.spec_from_file_location("research_precision", SWIFT_ROOT / "swiftllm" / "precision.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the shared precision helper") from error
    precision_module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = precision_module
    spec.loader.exec_module(precision_module)
    quantize_dequantize = precision_module.quantize_dequantize


PROMPTS: list[dict[str, str]] = [
    {"id": "short_factual", "type": "short", "text": "The capital of France is"},
    {
        "id": "long_factual",
        "type": "factual",
        "text": "Explain in two sentences why water boils at a lower temperature at high altitude.",
    },
    {
        "id": "code",
        "type": "code",
        "text": "Write a Python function that returns the factorial of a nonnegative integer.",
    },
    {
        "id": "math",
        "type": "math",
        "text": "A train travels 120 kilometers in 2 hours. What is its average speed? Show the calculation.",
    },
    {
        "id": "reasoning",
        "type": "reasoning",
        "text": "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly? Explain.",
    },
    {
        "id": "summarization",
        "type": "summarization",
        "text": "Summarize this sentence in one phrase: A small team measured latency before changing a production scheduler.",
    },
    {
        "id": "long_context",
        "type": "long",
        "text": "The experiment has four stages: collect traces, validate the reference, measure sensitivity, and decide whether native kernels are justified. "
        * 8,
    },
    {
        "id": "list",
        "type": "list",
        "text": "List three practical ways to reduce GPU memory use during autoregressive language-model inference.",
    },
]

PROJECTIONS = ("q", "k", "v")


def git_revision() -> str | None:
    marker = SWIFT_ROOT / "UPSTREAM_COMMIT"
    if marker.exists():
        return marker.read_text().strip()
    try:
        return subprocess.check_output(
            ["git", "-C", str(SWIFT_ROOT), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def research_diff_sha256() -> str | None:
    path = ROOT / "references" / "swiftllm-research.diff"
    return sha256(path) if path.exists() else None


def model_files(model_path: Path) -> list[dict[str, Any]]:
    result = []
    for path in sorted(model_path.iterdir()):
        if path.is_file() and (
            path.name.endswith(".safetensors")
            or path.name in {"config.json", "tokenizer.json", "tokenizer_config.json"}
        ):
            result.append({"name": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)})
    return result


def jsonable_config(q: int, k: int, v: int, scope: str = "all_layers", layer: int | None = None) -> dict[str, Any]:
    return {
        "q_bits": q,
        "k_bits": k,
        "v_bits": v,
        "unweighted_projection_bits": (q + k + v) / 3,
        "scope": scope,
        "layer": layer,
    }


def annotate_projection_budget(
    config: dict[str, Any],
    num_layers: int,
    hidden_size: int,
    num_kv_heads: int,
    head_dim: int,
    group_size: int = 128,
    scale_bits: int = 16,
) -> None:
    layer_count = num_layers if config["scope"] == "all_layers" else 1
    q_numel = layer_count * hidden_size * hidden_size
    kv_numel = layer_count * num_kv_heads * head_dim * hidden_size
    total_numel = q_numel + 2 * kv_numel
    weighted_bits = (
        config["q_bits"] * q_numel
        + config["k_bits"] * kv_numel
        + config["v_bits"] * kv_numel
    ) / total_numel
    # Each group has one FP16 scale in this proxy's storage convention. This
    # accounts for scale overhead but intentionally excludes packed metadata.
    config["weighted_projection_bits"] = weighted_bits
    config["weighted_storage_bits"] = weighted_bits + scale_bits / group_size
    config["projection_numel"] = total_numel


def prepare_configs(
    bits: list[int],
    num_layers: int,
    hidden_size: int,
    num_kv_heads: int,
    head_dim: int,
    layer_sweep: bool,
) -> list[dict[str, Any]]:
    configs = [jsonable_config(q, k, v) for q, k, v in itertools.product(bits, repeat=3)]
    if layer_sweep:
        for layer in range(num_layers):
            for projection in PROJECTIONS:
                for bit in bits:
                    if bit == 16:
                        continue
                    values = {"q": 16, "k": 16, "v": 16}
                    values[projection] = bit
                    configs.append(jsonable_config(**values, scope="single_layer", layer=layer))
    for config in configs:
        annotate_projection_budget(config, num_layers, hidden_size, num_kv_heads, head_dim)
    return configs


def metric(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float | int]:
    """Compute paired next-token metrics without sampling."""
    ref = reference.float().flatten()
    cand = candidate.float().flatten()
    delta = cand - ref
    ref_centered = ref - ref.mean()
    cand_centered = cand - cand.mean()
    cosine = torch.nn.functional.cosine_similarity(ref[None], cand[None]).item()
    centered_cosine = torch.nn.functional.cosine_similarity(ref_centered[None], cand_centered[None]).item()
    ref_logp = torch.log_softmax(ref, dim=-1)
    cand_logp = torch.log_softmax(cand, dim=-1)
    kl = torch.sum(torch.exp(ref_logp) * (ref_logp - cand_logp)).item()
    return {
        "logit_mse": torch.mean(delta * delta).item(),
        "logit_rmse": torch.sqrt(torch.mean(delta * delta)).item(),
        "logit_max_abs": torch.max(torch.abs(delta)).item(),
        "logit_cosine": cosine,
        "centered_logit_cosine": centered_cosine,
        "kl_reference_to_candidate": kl,
        "reference_top1": int(torch.argmax(ref).item()),
        "candidate_top1": int(torch.argmax(cand).item()),
        "top1_match": int(torch.argmax(ref).item() == torch.argmax(cand).item()),
    }


def mean_metrics(records: list[dict[str, Any]]) -> dict[str, float]:
    if not records:
        return {}
    numeric = [
        key
        for key, value in records[0].items()
        if isinstance(value, (int, float)) and key not in {"reference_top1", "candidate_top1"}
    ]
    return {key: float(sum(float(record[key]) for record in records) / len(records)) for key in numeric}


class ProjectionState:
    """Restore original weights and apply one experiment config in-place."""

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.layers = list(model.model.layers)
        self.original: dict[tuple[int, str], torch.Tensor] = {}
        for layer_id, layer in enumerate(self.layers):
            attention = layer.self_attn
            for projection in PROJECTIONS:
                self.original[(layer_id, projection)] = getattr(attention, f"{projection}_proj").weight.detach().clone()

    @torch.inference_mode()
    def apply(self, config: dict[str, Any]) -> None:
        scope = config["scope"]
        target_layer = config["layer"]
        for layer_id, layer in enumerate(self.layers):
            for projection in PROJECTIONS:
                parameter = getattr(layer.self_attn, f"{projection}_proj").weight
                source = self.original[(layer_id, projection)]
                if scope == "single_layer" and layer_id != target_layer:
                    parameter.copy_(source)
                    continue
                bits = config[f"{projection}_bits"]
                parameter.copy_(quantize_dequantize(source, bits))

    @torch.inference_mode()
    def restore(self) -> None:
        for (layer_id, projection), source in self.original.items():
            getattr(self.layers[layer_id].self_attn, f"{projection}_proj").weight.copy_(source)


def run_prefill_and_decode(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    reference: dict[str, Any],
    decode_tokens: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    with torch.inference_mode():
        output = model(input_ids=input_ids, use_cache=True, return_dict=True)
    prefill = [
        metric(reference["prefill_logits"][0, position], output.logits[0, position])
        | {"position": position, "phase": "prefill"}
        for position in range(input_ids.shape[1])
    ]

    decode: list[dict[str, Any]] = []
    past = output.past_key_values
    for position, token_id in enumerate(reference["continuation"][:decode_tokens]):
        token = torch.tensor([[token_id]], device=input_ids.device, dtype=torch.long)
        with torch.inference_mode():
            output = model(input_ids=token, past_key_values=past, use_cache=True, return_dict=True)
        decode.append(
            metric(reference["decode_logits"][position], output.logits[0, 0])
            | {"position": position, "phase": "decode"}
        )
        past = output.past_key_values
    return prefill, decode


def build_reference(model: torch.nn.Module, input_ids: torch.Tensor, decode_tokens: int) -> dict[str, Any]:
    with torch.inference_mode():
        output = model(input_ids=input_ids, use_cache=True, return_dict=True)
    continuation: list[int] = []
    decode_logits: list[torch.Tensor] = []
    past = output.past_key_values
    next_token = int(torch.argmax(output.logits[0, -1]).item())
    for _ in range(decode_tokens):
        continuation.append(next_token)
        token = torch.tensor([[next_token]], device=input_ids.device, dtype=torch.long)
        with torch.inference_mode():
            output = model(input_ids=token, past_key_values=past, use_cache=True, return_dict=True)
        decode_logits.append(output.logits[0, 0].detach().clone())
        past = output.past_key_values
        next_token = int(torch.argmax(output.logits[0, 0]).item())
    return {
        "continuation": continuation,
        "decode_logits": decode_logits,
    }


def summarize_config(
    model: torch.nn.Module,
    prompts: list[dict[str, str]],
    encoded: list[torch.Tensor],
    references: list[dict[str, Any]],
    state: ProjectionState,
    config: dict[str, Any],
    decode_tokens: int,
) -> dict[str, Any]:
    state.apply(config)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    started = time.perf_counter()
    per_query: list[dict[str, Any]] = []
    all_prefill: list[dict[str, Any]] = []
    all_decode: list[dict[str, Any]] = []
    for prompt, input_ids, reference in zip(prompts, encoded, references):
        prefill, decode = run_prefill_and_decode(model, input_ids, reference, decode_tokens)
        prefill_summary = mean_metrics(prefill)
        decode_summary = mean_metrics(decode)
        per_query.append(
            {
                "prompt_id": prompt["id"],
                "prompt_type": prompt["type"],
                "prompt_tokens": int(input_ids.shape[1]),
                "reference_continuation": reference["continuation"],
                "prefill": prefill_summary,
                "decode": decode_summary,
                "prefill_positions": prefill,
                "decode_positions": decode,
            }
        )
        all_prefill.extend(prefill)
        all_decode.extend(decode)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return {
        "config": config,
        "elapsed_seconds": time.perf_counter() - started,
        "prefill": mean_metrics(all_prefill),
        "decode": mean_metrics(all_decode),
        "per_query": per_query,
    }


def aggregate_by_type(config_result: dict[str, Any], phase: str) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in config_result["per_query"]:
        grouped.setdefault(record["prompt_type"], []).append(record[phase])
    return {prompt_type: mean_metrics(metrics) for prompt_type, metrics in sorted(grouped.items())}


def unweighted_sum_oracle_comparison(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Diagnostic comparison at an equal *unweighted* Q+K+V bit sum.

    Q/K/V matrices have different sizes for GQA, so this is not a memory
    budget. It is retained only to expose how much the raw profile sum hides.
    """
    fixed = [result for result in results if result["config"]["scope"] == "all_layers"]
    comparisons = []
    for bit_sum in sorted({
        sum(result["config"][projection + "_bits"] for projection in PROJECTIONS)
        for result in fixed
    }):
        candidates = [
            result for result in fixed
            if sum(result["config"][projection + "_bits"] for projection in PROJECTIONS) == bit_sum
        ]
        if len(candidates) < 2:
            continue
        global_best = min(candidates, key=lambda result: result["decode"]["logit_mse"])
        oracle_rows = []
        global_rows = []
        choices = []
        for query_index, global_query in enumerate(global_best["per_query"]):
            oracle = min(
                candidates,
                key=lambda result: result["per_query"][query_index]["decode"].get("logit_mse", float("inf")),
            )
            oracle_query = oracle["per_query"][query_index]
            oracle_rows.append(oracle_query["decode"])
            global_rows.append(global_query["decode"])
            choices.append({
                "prompt_id": global_query["prompt_id"],
                "chosen_profile": oracle["config"],
            })
        global_metrics = mean_metrics(global_rows)
        oracle_metrics = mean_metrics(oracle_rows)
        comparisons.append({
            "unweighted_projection_bit_sum": bit_sum,
            "unweighted_projection_bits": bit_sum / 3,
            "candidate_profiles": [candidate["config"] for candidate in candidates],
            "global_profile": global_best["config"],
            "oracle_choices": choices,
            "global_decode": global_metrics,
            "oracle_decode": oracle_metrics,
            "oracle_mse_reduction_fraction": (
                1 - oracle_metrics["logit_mse"] / global_metrics["logit_mse"]
                if global_metrics.get("logit_mse", 0) > 0 else 0.0
            ),
        })
    return comparisons


def weighted_budget_oracle_comparison(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compare a global profile with a per-query oracle at equal storage bits."""
    fixed = [result for result in results if result["config"]["scope"] == "all_layers"]
    groups: dict[float, list[dict[str, Any]]] = {}
    for result in fixed:
        key = round(result["config"]["weighted_storage_bits"], 6)
        groups.setdefault(key, []).append(result)

    comparisons = []
    for storage_bits, candidates in sorted(groups.items()):
        if len(candidates) < 2:
            continue
        global_best = min(candidates, key=lambda result: result["decode"]["logit_mse"])
        oracle_rows = []
        global_rows = []
        choices = []
        for query_index, global_query in enumerate(global_best["per_query"]):
            oracle = min(
                candidates,
                key=lambda result: result["per_query"][query_index]["decode"].get("logit_mse", float("inf")),
            )
            oracle_query = oracle["per_query"][query_index]
            oracle_rows.append(oracle_query["decode"])
            global_rows.append(global_query["decode"])
            choices.append({
                "prompt_id": global_query["prompt_id"],
                "chosen_profile": oracle["config"],
            })
        global_metrics = mean_metrics(global_rows)
        oracle_metrics = mean_metrics(oracle_rows)
        comparisons.append({
            "weighted_storage_bits": storage_bits,
            "candidate_profiles": [candidate["config"] for candidate in candidates],
            "global_profile": global_best["config"],
            "oracle_choices": choices,
            "global_decode": global_metrics,
            "oracle_decode": oracle_metrics,
            "oracle_mse_reduction_fraction": (
                1 - oracle_metrics["logit_mse"] / global_metrics["logit_mse"]
                if global_metrics.get("logit_mse", 0) > 0 else 0.0
            ),
        })
    return comparisons


def oracle_comparison(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compare fixed all-equal profiles with a per-query oracle.

    For a fixed N-bit profile, the oracle may choose any tested Q/K/V tuple
    whose unweighted average is within 2/3 bit of N. This is explicitly an offline oracle,
    not a realizable scheduler policy.
    """
    fixed = [result for result in results if result["config"]["scope"] == "all_layers"]
    comparisons = []
    for fixed_bits in (4, 8, 16):
        fixed_results = [
            result for result in fixed
            if result["config"]["q_bits"] == fixed_bits
            and result["config"]["k_bits"] == fixed_bits
            and result["config"]["v_bits"] == fixed_bits
        ]
        if not fixed_results:
            continue
        fixed_result = fixed_results[0]
        oracle_rows = []
        fixed_rows = []
        oracle_choices = []
        for query_index, fixed_query in enumerate(fixed_result["per_query"]):
            candidates = [
                result for result in fixed
                if abs(result["config"]["unweighted_projection_bits"] - fixed_bits) <= (2 / 3)
            ]
            if not candidates:
                continue
            oracle = min(candidates, key=lambda result: result["per_query"][query_index]["decode"].get("logit_mse", float("inf")))
            oracle_query = oracle["per_query"][query_index]
            oracle_rows.append(oracle_query["decode"])
            fixed_rows.append(fixed_query["decode"])
            oracle_choices.append({
                "prompt_id": fixed_query["prompt_id"],
                "chosen_profile": oracle["config"],
            })
        comparisons.append(
            {
                "fixed_profile": {"q_bits": fixed_bits, "k_bits": fixed_bits, "v_bits": fixed_bits},
                "unweighted_projection_bits": fixed_bits,
                "unweighted_tolerance_bits": 2 / 3,
                "query_count": len(oracle_rows),
                "candidate_profiles": [candidate["config"] for candidate in candidates],
                "oracle_choices": oracle_choices,
                "fixed_decode": mean_metrics(fixed_rows),
                "oracle_decode": mean_metrics(oracle_rows),
                "oracle_mse_reduction_fraction": (
                    1 - mean_metrics(oracle_rows)["logit_mse"] / mean_metrics(fixed_rows)["logit_mse"]
                    if mean_metrics(fixed_rows).get("logit_mse", 0) > 0 else 0.0
                ),
            }
        )
    return comparisons


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-prompts", type=int, default=len(PROMPTS))
    parser.add_argument("--decode-tokens", type=int, default=4)
    parser.add_argument("--bits", type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument("--layer-sweep", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model_path).expanduser().resolve()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA is required for this harness; use --device cpu only for a tiny smoke test")
    device = torch.device(args.device)
    prompts = PROMPTS[: args.max_prompts]
    if not prompts:
        raise ValueError("at least one prompt is required")
    if any(bit not in (4, 8, 16) for bit in args.bits):
        raise ValueError("--bits must contain only 4, 8, and/or 16")

    started = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map={"": device},
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    encoded = [
        tokenizer(prompt["text"], return_tensors="pt")["input_ids"].to(device)
        for prompt in prompts
    ]

    # Keep references on the GPU only while measuring; each reference has a
    # small continuation cache and full-vocabulary logits for paired metrics.
    references = []
    for input_ids in encoded:
        with torch.inference_mode():
            prefill = model(input_ids=input_ids, use_cache=True, return_dict=True)
        # ``build_reference`` starts with a fresh prefill so it can retain the
        # exact logits independently of the subsequent decode calls.
        output_prefill_logits = prefill.logits.detach().clone()
        reference = build_reference(model, input_ids, args.decode_tokens)
        reference["prefill_logits"] = output_prefill_logits
        references.append(reference)

    state = ProjectionState(model)
    configs = prepare_configs(
        args.bits,
        len(model.model.layers),
        model.config.hidden_size,
        model.config.num_key_value_heads,
        model.config.head_dim,
        args.layer_sweep,
    )
    results = []
    for index, config in enumerate(configs, start=1):
        print(f"[{index}/{len(configs)}] {config}", flush=True)
        results.append(summarize_config(model, prompts, encoded, references, state, config, args.decode_tokens))
    state.restore()
    fp16_controls = [
        result for result in results
        if result["config"]["scope"] == "all_layers"
        and result["config"]["q_bits"] == result["config"]["k_bits"] == result["config"]["v_bits"] == 16
    ]
    if len(fp16_controls) != 1:
        raise AssertionError("the matrix must contain exactly one all-FP16 control")
    fp16_control = fp16_controls[0]
    correctness = {
        "fp16_control": {
            "prefill_max_abs_logit_error": fp16_control["prefill"]["logit_max_abs"],
            "decode_max_abs_logit_error": fp16_control["decode"]["logit_max_abs"],
            "prefill_top1_match": fp16_control["prefill"]["top1_match"],
            "decode_top1_match": fp16_control["decode"]["top1_match"],
            "defined_tolerance": {
                "max_abs_logit_error": 0.0,
                "top1_mismatch_count": 0,
            },
        },
        "note": "The control compares the same FP16 model state before and after profile restoration; tolerance is exact equality for this regression check.",
    }

    result = {
        "schema_version": 1,
        "experiment": "qkv_projection_sensitivity",
        "created_unix": time.time(),
        "elapsed_seconds": time.time() - started,
        "numerical_proxy": {
            "kind": "symmetric_weight_only_fake_quantization",
            "bits": args.bits,
            "group_size": 128,
            "representation": "FP16 after quantize/dequantize",
            "native_low_bit_speedup_claim": False,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
            "swiftllm_commit": git_revision(),
            "swiftllm_upstream_commit": git_revision(),
            "swiftllm_research_diff_sha256": research_diff_sha256(),
        },
        "model": {
            "path": str(model_path),
            "config": json.loads((model_path / "config.json").read_text()),
            "files": model_files(model_path),
        },
        "prompts": prompts,
        "decode_tokens": args.decode_tokens,
        "correctness": correctness,
        "results": results,
        "by_prompt_type": {
            str(result["config"]): {
                "prefill": aggregate_by_type(result, "prefill"),
                "decode": aggregate_by_type(result, "decode"),
            }
            for result in results
        },
        "unweighted_fixed_oracle_comparison": oracle_comparison(results),
        "unweighted_sum_oracle_comparison": unweighted_sum_oracle_comparison(results),
        "weighted_budget_oracle_comparison": weighted_budget_oracle_comparison(results),
    }
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
