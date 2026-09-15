#!/usr/bin/env python3
"""Held-out task-quality probes for one native KV method.

Timing is intentionally absent here.  The script compares long-context
Wikitext validation continuations and free-running HellaSwag validation
prompts against an unchanged FP16 Transformers reference, using the same
candidate path that is used for the capacity cells.
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
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.kv_capacity_experiment_v2 import (  # noqa: E402
    MODEL_PATHS,
    SOURCE_FILES,
    environment,
    fixed_targets,
    hf_call,
    load_hf,
    load_swift,
    model_files,
    cleanup,
    source_sha256,
    sync,
)

CACHE_DIR = "/nfs/home/s314511048/.cache/huggingface/datasets"


def quality_source_sha256() -> str:
    digest = hashlib.sha256()
    digest.update(source_sha256().encode())
    digest.update(Path(__file__).read_bytes())
    return digest.hexdigest()


def bootstrap(values: list[float], seed: int, iterations: int = 2000) -> dict[str, object]:
    array = torch.tensor(values, dtype=torch.float64).numpy()
    import numpy as np
    rng = np.random.default_rng(seed)
    samples = array[rng.integers(0, len(array), size=(iterations, len(array)))]
    means = samples.mean(axis=1)
    return {"mean": float(array.mean()), "ci95_low": float(np.quantile(means, .025)), "ci95_high": float(np.quantile(means, .975)), "count": len(values), "bootstrap_iterations": iterations}


def heldout_inputs(tokenizer, context: int, count: int) -> list[dict[str, object]]:
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation", cache_dir=CACHE_DIR, download_mode="reuse_dataset_if_exists")
    text = "\n".join(row["text"] for row in ds if row["text"].strip())
    ids = tokenizer.encode(text, add_special_tokens=False)
    continuation = 32
    max_start = len(ids) - context - continuation
    if max_start <= 0:
        raise RuntimeError(f"Wikitext validation has {len(ids)} tokens; need {context + continuation}")
    starts = [int(round((i + 1) * max_start / (count + 1))) for i in range(count)]
    return [{"sample_id": f"wikitext_validation_window_{i:02d}_start_{start}", "window_start": start, "input_ids": ids[start:start + context], "target_ids": ids[start + context:start + context + continuation], "query": {"kind": "identified_continuation_boundary", "description": "predict the held-out validation continuation immediately after this long context window", "context_token_start": start, "context_token_end_exclusive": start + context, "target_token_start": start + context, "target_token_end_exclusive": start + context + continuation}} for i, start in enumerate(starts)]


def hellaswag_inputs(tokenizer, count: int) -> list[dict[str, object]]:
    from datasets import load_dataset
    ds = load_dataset("Rowan/hellaswag", split="validation", cache_dir=CACHE_DIR, download_mode="reuse_dataset_if_exists")
    rows = []
    stride = max(1, len(ds) // (count * 8))
    # KIVI's native GQA path requires a sufficiently long real older region.
    # Select naturally long validation prompts, without duplicating or padding
    # task text. The threshold is recorded as a support limitation, not a
    # hidden quality transformation.
    for index in range(0, len(ds), stride):
        row = ds[index]
        text = f"{row['ctx']} Continue the passage:"
        ids = tokenizer.encode(text, add_special_tokens=True)
        if len(ids) < 64:
            continue
        rows.append({"sample_id": f"hellaswag_validation_{int(row['ind']):06d}", "prompt": text, "input_ids": ids[:512], "label": int(row["label"]), "endings": row["endings"], "prompt_token_count": min(len(ids), 512)})
        if len(rows) == count:
            break
    if len(rows) != count:
        raise RuntimeError(f"only found {len(rows)} natural HellaSwag prompts with at least 64 tokens; need {count}")
    return rows


def score_hf(model, sample: dict[str, object], device: torch.device) -> dict[str, object]:
    ids = torch.tensor([sample["input_ids"]], device=device, dtype=torch.long)
    past = None
    output = hf_call(model, ids, past)
    logits = output.logits[:, -1, :]
    past = output.past_key_values
    nll, top, step_logits = [], [], []
    for target in sample["target_ids"]:
        current = logits[0].detach().float().cpu()
        logp = F.log_softmax(current, dim=-1)
        nll.append(float(-logp[int(target)].item()))
        top.append(int(current.argmax().item()))
        step_logits.append(current)
        output = hf_call(model, torch.tensor([[int(target)]], device=device), past)
        logits, past = output.logits[:, -1, :], output.past_key_values
    del output, past, logits, ids
    return {"nll": nll, "top1": top, "step_logits": step_logits}


def score_kivi(model, sample: dict[str, object], device: torch.device) -> dict[str, object]:
    return score_hf(model, sample, device)


def score_swift(model, sample: dict[str, object], device: torch.device) -> dict[str, object]:
    seq = [int(x) for x in sample["input_ids"]]
    output = model.forward([seq], [0], [], return_logits=True)
    logits = output[0] if isinstance(output, (list, tuple)) else output
    nll, top, step_logits = [], [], []
    for offset, target in enumerate(sample["target_ids"], start=1):
        current = logits[0].detach().float().cpu()
        logp = F.log_softmax(current, dim=-1)
        nll.append(float(-logp[int(target)].item()))
        top.append(int(current.argmax().item()))
        step_logits.append(current)
        output = model.forward([[int(target)]], [0], [len(seq) + offset], return_logits=True)
        logits = output[0] if isinstance(output, (list, tuple)) else output
    model.free_seqs_resources([0])
    sync(device)
    return {"nll": nll, "top1": top, "step_logits": step_logits}


def _generation_row(generated, step_logits, eos_ids, max_steps):
    nonfinite = any(not bool(torch.isfinite(x).all()) for x in step_logits)
    terminated_at = next((i + 1 for i, token in enumerate(generated) if token in eos_ids), None)
    effective = generated[:terminated_at] if terminated_at is not None else generated
    pathological = False
    if len(effective) >= 8 and any(len(set(effective[i - 8:i])) == 1 for i in range(8, len(effective) + 1)):
        pathological = True
    if len(effective) >= 12:
        for i in range(12, len(effective) + 1):
            if effective[i - 12:i] == effective[i - 6:i] * 2:
                pathological = True
                break
    return {"tokens": effective, "step_logits": step_logits[:len(effective)], "terminated_by_eos": terminated_at is not None, "termination_step": terminated_at, "termination_reason": "eos" if terminated_at is not None else "max_tokens", "nonfinite_logits": nonfinite, "repetition_pathology": pathological, "safety_failure": nonfinite, "termination_failure": False, "requested_max_tokens": max_steps}


def generate_hf(model, ids: list[int], steps: int, device: torch.device, eos_ids: set[int]) -> dict[str, object]:
    output = hf_call(model, torch.tensor([ids], device=device, dtype=torch.long))
    past = output.past_key_values
    logits = output.logits[:, -1, :]
    generated, step_logits = [], []
    for _ in range(steps):
        token = int(logits.argmax(-1)[0].item())
        generated.append(token)
        step_logits.append(logits[0].detach().float().cpu())
        if token in eos_ids:
            break
        output = hf_call(model, torch.tensor([[token]], device=device), past)
        past = output.past_key_values
        logits = output.logits[:, -1, :]
    del output, past, logits
    return _generation_row(generated, step_logits, eos_ids, steps)


def generate_swift(model, ids: list[int], steps: int, device: torch.device, eos_ids: set[int]) -> dict[str, object]:
    output = model.forward([ids], [0], [], return_logits=True)
    logits = output[0] if isinstance(output, (list, tuple)) else output
    generated, step_logits = [], []
    for step in range(steps):
        token = int(logits.argmax(-1)[0].item())
        generated.append(token)
        step_logits.append(logits[0].detach().float().cpu())
        if token in eos_ids:
            break
        output = model.forward([[token]], [0], [len(ids) + step + 1], return_logits=True)
        logits = output[0] if isinstance(output, (list, tuple)) else output
    model.free_seqs_resources([0])
    sync(device)
    return _generation_row(generated, step_logits, eos_ids, steps)


def divergence(candidate_logits: torch.Tensor, reference_logits: torch.Tensor) -> dict[str, float]:
    candidate = candidate_logits.float()
    reference = reference_logits.float()
    candidate_logp = F.log_softmax(candidate, dim=-1)
    reference_logp = F.log_softmax(reference, dim=-1)
    reference_prob = reference_logp.exp()
    return {
        "kl_reference_to_candidate": float((reference_prob * (reference_logp - candidate_logp)).sum().item()),
        "logit_mse": float(F.mse_loss(candidate, reference).item()),
    }


def release(model, device):
    del model
    cleanup()
    sync(device)


@torch.inference_mode()
def run(args: argparse.Namespace) -> dict[str, object]:
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(args.budget_fraction, device)
    path = str(Path(args.model_path or MODEL_PATHS[args.model_family]).resolve())

    # Tokenizer and held-out sampling are method-independent and are frozen
    # before loading the candidate model.
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    long_samples = heldout_inputs(tokenizer, args.context_tokens, args.samples)
    free_samples = hellaswag_inputs(tokenizer, args.samples)

    if args.method == "swiftllm_fp16":
        reference, _, ref_config = load_swift(path, args.context_tokens, args.generation_tokens, 1, device, args.budget_fraction)
        reference_score_fn = score_swift
        reference_gen_fn = generate_swift
        reference_kind = "same_native_swiftllm_fp16_control"
    else:
        reference, _, ref_config = load_hf("hf_fp16", path, device, 16, 32, 32)
        reference_score_fn = score_hf
        reference_gen_fn = generate_hf
        reference_kind = "corrected_transformers_fp16_control"
    reference_long = []
    reference_free = []
    for sample in long_samples:
        reference_long.append(reference_score_fn(reference, sample, device))
    eos_value = tokenizer.eos_token_id
    eos_ids = {int(x) for x in eos_value} if isinstance(eos_value, (list, tuple, set)) else {int(eos_value)}
    for sample in free_samples:
        reference_free.append(reference_gen_fn(reference, sample["input_ids"], args.generation_tokens, device, eos_ids))
    release(reference, device)
    reference = None

    if args.method == "swiftllm_fp16":
        candidate, candidate_tokenizer, method_config = load_swift(path, args.context_tokens, args.generation_tokens, 1, device, args.budget_fraction)
        score_fn = score_swift
        gen_fn = generate_swift
    else:
        bits = int(args.bits)
        candidate, candidate_tokenizer, method_config = load_hf(args.method, path, device, bits, 32, 32)
        score_fn = score_kivi
        gen_fn = generate_hf

    long_rows = []
    for sample, reference_row in zip(long_samples, reference_long):
        candidate_row = score_fn(candidate, sample, device)
        deltas = [a - b for a, b in zip(candidate_row["nll"], reference_row["nll"])]
        paired = [divergence(c, r) for c, r in zip(candidate_row["step_logits"], reference_row["step_logits"])]
        long_rows.append({"sample_id": sample["sample_id"], "window_start": sample["window_start"], "query": sample["query"], "nll_delta": sum(deltas) / len(deltas), "reference_nll": sum(reference_row["nll"]) / len(reference_row["nll"]), "candidate_nll": sum(candidate_row["nll"]) / len(candidate_row["nll"]), "top1_match": sum(a == b for a, b in zip(candidate_row["top1"], reference_row["top1"])) / len(deltas), "kl_mean": sum(x["kl_reference_to_candidate"] for x in paired) / len(paired), "logit_mse_mean": sum(x["logit_mse"] for x in paired) / len(paired), "per_token_divergence": [{"nll_delta": delta, "top1_match": candidate_top == reference_top, **stats} for delta, candidate_top, reference_top, stats in zip(deltas, candidate_row["top1"], reference_row["top1"], paired)], "target_tokens": len(deltas)})

    free_rows = []
    for sample, reference_row in zip(free_samples, reference_free):
        candidate_row = gen_fn(candidate, sample["input_ids"], args.generation_tokens, device, eos_ids)
        reference_tokens = reference_row["tokens"]
        candidate_tokens = candidate_row["tokens"]
        paired = [divergence(c, r) for c, r in zip(candidate_row["step_logits"], reference_row["step_logits"])]
        free_rows.append({"sample_id": sample["sample_id"], "label": sample["label"], "candidate_tokens": candidate_tokens, "reference_tokens": reference_tokens, "prefix_token_agreement": sum(a == b for a, b in zip(candidate_tokens, reference_tokens)) / max(1, len(reference_tokens)), "per_step_divergence": [{"top1_match": candidate_row["step_logits"][i].argmax().item() == reference_row["step_logits"][i].argmax().item(), **stats} for i, stats in enumerate(paired)], "candidate_termination": {k: v for k, v in candidate_row.items() if k not in {"tokens", "step_logits"}}, "reference_termination": {k: v for k, v in reference_row.items() if k not in {"tokens", "step_logits"}}, "candidate_text": tokenizer.decode(candidate_tokens, skip_special_tokens=True), "reference_text": tokenizer.decode(reference_tokens, skip_special_tokens=True), "generated_tokens": len(candidate_tokens)})

    release(candidate, device)
    long_deltas = [row["nll_delta"] for row in long_rows]
    return {"schema": "kv-capacity-quality-v2", "status": "completed", "model_family": args.model_family, "model_path": path, "model_files": model_files(path), "method": args.method, "method_config": method_config, "reference": {"kind": reference_kind, "config": ref_config}, "provenance": {"git_head": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(), "source_files": list(SOURCE_FILES), "source_sha256": quality_source_sha256(), "kivi_commit": subprocess.check_output(["git", "-C", str(ROOT / "vendor/public/KIVI"), "rev-parse", "HEAD"], text=True).strip(), "swiftllm_upstream_commit": (ROOT / "vendor/swiftLLM/UPSTREAM_COMMIT").read_text().strip()}, "sampling": {"long_context_dataset": "wikitext-2-raw-v1/validation", "long_context_count": len(long_rows), "long_context_window_tokens": args.context_tokens, "continuation_tokens": 32, "long_context_query": {"kind": "identified_continuation_boundary", "recorded_per_sample": True, "planted": False}, "free_running_dataset": "Rowan/hellaswag/validation", "free_running_prompt_min_tokens": 64, "free_running_count": len(free_rows), "generation_tokens": args.generation_tokens, "eos_token_ids": sorted(eos_ids), "heldout_loaded_before_candidate": True, "artificial_prompt_repetition": False, "eos_handled": True}, "long_context_information_use": {"per_sample": long_rows, "nll_delta_bootstrap95": bootstrap(long_deltas, args.seed + 1), "top1_match_mean": sum(row["top1_match"] for row in long_rows) / len(long_rows), "kl_mean": sum(row["kl_mean"] for row in long_rows) / len(long_rows), "logit_mse_mean": sum(row["logit_mse_mean"] for row in long_rows) / len(long_rows)}, "free_running_generation": {"per_sample": free_rows, "prefix_token_agreement_mean": sum(row["prefix_token_agreement"] for row in free_rows) / len(free_rows), "per_step_divergence_recorded": True, "termination_records": True, "pathology_records": True, "sample_count": len(free_rows)}, "quality_timing_separate": True, "scope": "held-out task probes; no capacity or latency value is taken from this run", "environment": environment(device, args.budget_fraction), "invocation": {"argv": sys.argv, "cwd": os.getcwd()}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--method", choices=("swiftllm_fp16", "kivi2", "kivi4"), required=True)
    parser.add_argument("--model-family", choices=tuple(MODEL_PATHS), required=True)
    parser.add_argument("--model-path")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--context-tokens", type=int, default=2048)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--generation-tokens", type=int, default=64)
    parser.add_argument("--bits", type=int, choices=(2, 4), default=2)
    parser.add_argument("--budget-fraction", type=float, default=.90)
    parser.add_argument("--seed", type=int, default=20261021)
    args = parser.parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    try:
        result = run(args)
    except Exception as exc:
        result = {"schema": "kv-capacity-quality-v2", "status": "blocked", "method": args.method, "model_family": args.model_family, "error": {"type": type(exc).__name__, "message": str(exc)}}
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": args.output, "status": result.get("status", "completed"), "method": args.method, "model_family": args.model_family}))


if __name__ == "__main__":
    main()
