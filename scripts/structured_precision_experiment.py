#!/usr/bin/env python3
"""Model-level structured weight precision experiment.

This is an offline numerical experiment, not a serving implementation.  It
uses the same symmetric per-input-group fake-quantization proxy as the
SwiftLLM research fork, but evaluates all attention projections (Q/K/V/O) and
one FFN block unit (gate/up/down together).  A unit is one projection or one
layer's FFN block.  Single-unit calibration measurements feed deterministic
budgeted policy heuristics; every selected policy is then run on held-out
language-model text and a held-out multiple-choice task set.

Important scope boundaries:
* weights are quantize/dequantized FP16 tensors, never packed low-bit kernels;
* KV-cache quantization is not used;
* task accuracy is teacher-forced multiple-choice scoring, not free generation;
* all storage comparisons use explicit representation-aware bit counts.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
SWIFT_ROOT = ROOT / "vendor" / "swiftLLM"
if str(SWIFT_ROOT) not in sys.path:
    sys.path.insert(0, str(SWIFT_ROOT))
try:
    from swiftllm.precision import quantize_dequantize  # noqa: E402
except ModuleNotFoundError as error:
    if error.name != "swiftllm_c":
        raise
    spec = importlib.util.spec_from_file_location("research_precision", SWIFT_ROOT / "swiftllm" / "precision.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the shared precision helper") from error
    precision_module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = precision_module
    spec.loader.exec_module(precision_module)
    quantize_dequantize = precision_module.quantize_dequantize

BITS = (4, 8, 16)
UNIT_NAMES = ("q", "k", "v", "o", "ffn")
GROUP_SIZE = 128
SCALE_BITS = 16


@dataclass(frozen=True)
class MatrixShape:
    """Shape and representation cost metadata for one weight matrix."""

    name: str
    out_features: int
    in_features: int

    @property
    def numel(self) -> int:
        return self.out_features * self.in_features

    @property
    def padded_in_features(self) -> int:
        return ((self.in_features + GROUP_SIZE - 1) // GROUP_SIZE) * GROUP_SIZE

    @property
    def quantized_numel(self) -> int:
        return self.out_features * self.padded_in_features

    @property
    def scale_count(self) -> int:
        return self.out_features * ((self.in_features + GROUP_SIZE - 1) // GROUP_SIZE)


@dataclass(frozen=True)
class Unit:
    layer: int
    name: str
    matrices: tuple[MatrixShape, ...]

    @property
    def key(self) -> str:
        return f"layer_{self.layer:03d}/{self.name}"

    @property
    def numel(self) -> int:
        return sum(matrix.numel for matrix in self.matrices)

    def storage(self, bits: int) -> dict[str, int | float | bool]:
        """Return exact modeled storage bits for this unit.

        FP16 is represented directly and therefore pays no quantization scale
        or zero-point overhead.  Quantized matrices use padded groups of
        ``GROUP_SIZE`` input channels, one FP16 symmetric scale per group, and
        no zero point.  There is no unmodeled metadata in this representation;
        the result is explicitly marked as a model of the proxy format.
        """
        if bits == 16:
            weight_bits = 16 * self.numel
            padding_bits = 0
            scale_bits = 0
            zero_point_bits = 0
        elif bits in (4, 8):
            quantized_numel = sum(matrix.quantized_numel for matrix in self.matrices)
            weight_bits = bits * quantized_numel
            padding_bits = bits * sum(matrix.quantized_numel - matrix.numel for matrix in self.matrices)
            scale_bits = SCALE_BITS * sum(matrix.scale_count for matrix in self.matrices)
            zero_point_bits = 0
        else:
            raise ValueError(f"unsupported bits: {bits}")
        total = weight_bits + scale_bits + zero_point_bits
        return {
            "bits": bits,
            "weight_payload_bits": weight_bits,
            "padding_bits": padding_bits,
            "scale_bits": scale_bits,
            "zero_point_bits": zero_point_bits,
            "metadata_bits": 0,
            "total_bits": total,
            "total_bytes": total / 8,
            "includes_padding": bits != 16,
            "includes_scale": bits != 16,
            "includes_zero_point": False,
        }


@dataclass
class Profile:
    name: str
    kind: str
    bits: dict[str, int]
    target_bits: int | None = None
    optimizer: str | None = None
    fixed_fp16_bits: int = 0

    def storage(self, units: list[Unit]) -> dict[str, int | float | bool]:
        totals: dict[str, int] = {
            "weight_payload_bits": 0,
            "padding_bits": 0,
            "scale_bits": 0,
            "zero_point_bits": 0,
            "metadata_bits": 0,
            "total_bits": 0,
        }
        for unit in units:
            cost = unit.storage(self.bits[unit.key])
            for key in totals:
                totals[key] += int(cost[key])
        totals["weight_payload_bits"] += self.fixed_fp16_bits
        totals["total_bits"] += self.fixed_fp16_bits
        totals["fixed_fp16_bits"] = self.fixed_fp16_bits
        totals["total_bytes"] = totals["total_bits"] / 8
        totals["profile_unit_count"] = len(units)
        totals["representation"] = "FP16 direct or symmetric groupwise fake-quantized payload; no packed metadata"
        return totals

    def as_record(self, units: list[Unit]) -> dict[str, Any]:
        storage = self.storage(units)
        return {
            "name": self.name,
            "kind": self.kind,
            "optimizer": self.optimizer,
            "target_bits": self.target_bits,
            "unit_bits": self.bits,
            "storage": storage,
        }


class ProfileState:
    """Apply profiles while retaining original weights on CPU.

    Keeping the original copy on CPU avoids an extra full model-sized GPU copy,
    which is important for the 8B confirmation model.  Only units whose bit
    assignment changes are transferred and quantized.
    """

    def __init__(self, model: torch.nn.Module, units: list[Unit], device: torch.device):
        self.model = model
        self.units = units
        self.device = device
        self.original: dict[str, list[torch.Tensor]] = {}
        self.parameters: dict[str, list[torch.nn.Parameter]] = {}
        for unit in units:
            params: list[torch.nn.Parameter] = []
            original: list[torch.Tensor] = []
            layer = model.model.layers[unit.layer]
            for matrix in unit.matrices:
                parameter = parameter_for_matrix(layer, matrix.name)
                params.append(parameter)
                original.append(parameter.detach().cpu().clone())
            self.parameters[unit.key] = params
            self.original[unit.key] = original
        self.active = {unit.key: 16 for unit in units}

    @torch.inference_mode()
    def set_profile(self, profile: Profile | None = None) -> None:
        target = {unit.key: 16 for unit in self.units} if profile is None else profile.bits
        for unit in self.units:
            bits = int(target[unit.key])
            if bits == self.active[unit.key]:
                continue
            for parameter, source in zip(self.parameters[unit.key], self.original[unit.key]):
                if bits == 16:
                    parameter.copy_(source.to(device=self.device, dtype=parameter.dtype))
                else:
                    source_device = source.to(device=self.device, dtype=parameter.dtype)
                    parameter.copy_(quantize_dequantize(source_device, bits, group_size=GROUP_SIZE))
            self.active[unit.key] = bits

    @torch.inference_mode()
    def restore(self) -> None:
        self.set_profile(None)


def parameter_for_matrix(layer: torch.nn.Module, name: str) -> torch.nn.Parameter:
    if name in UNIT_NAMES[:4]:
        return getattr(layer.self_attn, f"{name}_proj").weight
    if name in {"ffn_gate", "ffn_up", "ffn_down"}:
        if name == "ffn_gate":
            return layer.mlp.gate_proj.weight
        if name == "ffn_up":
            return layer.mlp.up_proj.weight
        return layer.mlp.down_proj.weight
    raise KeyError(name)


def fixed_fp16_storage_bits(model: torch.nn.Module, units: list[Unit]) -> int:
    """Count non-Q/K/V/O/FFN parameters kept directly in FP16.

    Embeddings, LM head (when untied), norms, and any other parameters are
    fixed across policy comparisons but are included in the reported model
    weight ledger instead of silently omitted from the budget.
    """
    unit_parameter_ids = set()
    for unit in units:
        layer = model.model.layers[unit.layer]
        unit_parameter_ids.update(id(parameter_for_matrix(layer, matrix.name)) for matrix in unit.matrices)
    fixed_numel = sum(parameter.numel() for parameter in model.parameters() if id(parameter) not in unit_parameter_ids)
    return int(fixed_numel * 16)


def build_units(model: torch.nn.Module) -> list[Unit]:
    units: list[Unit] = []
    for layer_id, layer in enumerate(model.model.layers):
        matrices: dict[str, tuple[str, ...]] = {
            "q": ("q",),
            "k": ("k",),
            "v": ("v",),
            "o": ("o",),
            "ffn": ("ffn_gate", "ffn_up", "ffn_down"),
        }
        for unit_name, matrix_names in matrices.items():
            shapes = []
            for matrix_name in matrix_names:
                parameter = parameter_for_matrix(layer, matrix_name)
                shapes.append(MatrixShape(matrix_name, parameter.shape[0], parameter.shape[1]))
            units.append(Unit(layer_id, unit_name, tuple(shapes)))
    return units


def profile_from_unit_bits(name: str, kind: str, units: list[Unit], values: dict[str, int], **kwargs: Any) -> Profile:
    return Profile(name=name, kind=kind, bits={unit.key: int(values[unit.key]) for unit in units}, **kwargs)


def uniform_profile(units: list[Unit], bits: int, name: str | None = None, fixed_fp16_bits: int = 0) -> Profile:
    return profile_from_unit_bits(name or f"uniform_w{bits}", "uniform", units, {unit.key: bits for unit in units}, target_bits=bits, fixed_fp16_bits=fixed_fp16_bits)


def manual_storage_cases() -> dict[str, Any]:
    """Small exact cases used by tests and the artifact verifier."""
    divisible = Unit(0, "q", (MatrixShape("q", 2, 128),))
    nondivisible = Unit(0, "q", (MatrixShape("q", 2, 129),))
    return {
        "fp16_no_scale": divisible.storage(16),
        "w4_divisible": divisible.storage(4),
        "w4_nondivisible": nondivisible.storage(4),
        "checks": {
            "fp16_is_weight_only": divisible.storage(16)["total_bits"] == 2 * 128 * 16,
            "w4_scale_is_one_per_row_group": divisible.storage(4)["scale_bits"] == 2 * 16,
            "padding_is_counted": nondivisible.storage(4)["padding_bits"] == (2 * 256 - 2 * 129) * 4,
            "fp16_has_no_scale": divisible.storage(16)["scale_bits"] == 0,
        },
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision() -> str | None:
    marker = SWIFT_ROOT / "UPSTREAM_COMMIT"
    if marker.exists():
        return marker.read_text().strip()
    try:
        return subprocess.check_output(["git", "-C", str(SWIFT_ROOT), "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def load_text_sequences(tokenizer: Any, split: str, count: int, seq_len: int, cache_dir: str | None) -> list[dict[str, Any]]:
    dataset = load_dataset(
        "wikitext",
        "wikitext-2-raw-v1",
        split=split,
        cache_dir=cache_dir,
        download_mode="reuse_dataset_if_exists",
    )
    text = "\n".join(row["text"] for row in dataset if row["text"].strip())
    ids = tokenizer(text, add_special_tokens=False, return_attention_mask=False)["input_ids"]
    needed = count * seq_len
    if len(ids) < needed:
        raise RuntimeError(f"Wikitext {split} has only {len(ids)} tokens; need {needed}")
    records = []
    for index in range(count):
        start = index * seq_len
        records.append({"sample_id": f"wikitext_{split}_{index:04d}", "input_ids": ids[start : start + seq_len]})
    return records


def load_task_examples(tokenizer: Any, count: int, seed: int, cache_dir: str | None) -> list[dict[str, Any]]:
    dataset = load_dataset(
        "Rowan/hellaswag",
        split="validation",
        cache_dir=cache_dir,
        download_mode="reuse_dataset_if_exists",
    )
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    records = []
    for index in indices[:count]:
        row = dataset[index]
        prompt = f"{row['activity_label']}: {row['ctx_a']} {row['ctx_b'].capitalize()}"
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        choices = [tokenizer(" " + ending, add_special_tokens=False)["input_ids"] for ending in row["endings"]]
        records.append({
            "sample_id": f"hellaswag_{index:05d}",
            "prompt": prompt,
            "prompt_ids": prompt_ids,
            "choice_ids": choices,
            "label": int(row["label"]),
        })
    return records


def batch_records(records: list[dict[str, Any]], batch_size: int) -> Iterable[list[dict[str, Any]]]:
    for start in range(0, len(records), batch_size):
        yield records[start : start + batch_size]


def lm_batch_metrics(reference_logits: torch.Tensor, candidate_logits: torch.Tensor, input_ids: torch.Tensor) -> list[dict[str, float | int | str]]:
    labels = input_ids[:, 1:]
    ref = reference_logits[:, :-1].float()
    cand = candidate_logits[:, :-1].float()
    ref_logp = F.log_softmax(ref, dim=-1)
    cand_logp = F.log_softmax(cand, dim=-1)
    token_count = labels.shape[1]
    records = []
    for row in range(input_ids.shape[0]):
        row_ref = ref[row]
        row_cand = cand[row]
        row_ref_logp = ref_logp[row]
        row_cand_logp = cand_logp[row]
        row_labels = labels[row]
        ref_nll = -row_ref_logp.gather(-1, row_labels[:, None]).mean().item()
        cand_nll = -row_cand_logp.gather(-1, row_labels[:, None]).mean().item()
        delta = row_cand - row_ref
        kl = (row_ref_logp.exp() * (row_ref_logp - row_cand_logp)).sum(dim=-1).mean().item()
        records.append({
            "nll": cand_nll,
            "reference_nll": ref_nll,
            "nll_delta": cand_nll - ref_nll,
            "perplexity": math.exp(min(cand_nll, 30.0)),
            "reference_perplexity": math.exp(min(ref_nll, 30.0)),
            "logit_mse": delta.square().mean().item(),
            "logit_rmse": delta.square().mean().sqrt().item(),
            "kl_reference_to_candidate": kl,
            "top1_match": (row_ref.argmax(dim=-1) == row_cand.argmax(dim=-1)).float().mean().item(),
            "token_count": token_count,
        })
    return records


def summarize_records(records: list[dict[str, Any]]) -> dict[str, float]:
    if not records:
        return {}
    keys = [key for key, value in records[0].items() if isinstance(value, (float, int)) and key not in {"token_count"}]
    summary = {key: float(np.mean([float(record[key]) for record in records])) for key in keys}
    if "nll" in summary:
        summary["perplexity_from_mean_nll"] = math.exp(min(summary["nll"], 30.0))
    if "reference_nll" in summary:
        summary["reference_perplexity_from_mean_nll"] = math.exp(min(summary["reference_nll"], 30.0))
    return summary


def evaluate_lm(
    model: torch.nn.Module,
    state: ProfileState,
    profile: Profile,
    records: list[dict[str, Any]],
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    """Evaluate a profile against a fresh FP16 reference for paired metrics."""
    state.set_profile(None)
    result_records: list[dict[str, Any]] = []
    for batch in batch_records(records, batch_size):
        input_ids = torch.tensor([record["input_ids"] for record in batch], device=device, dtype=torch.long)
        with torch.inference_mode():
            reference_logits = model(input_ids=input_ids, use_cache=False, return_dict=True).logits
        state.set_profile(profile)
        with torch.inference_mode():
            candidate_logits = model(input_ids=input_ids, use_cache=False, return_dict=True).logits
        result_records.extend(
            {"sample_id": record["sample_id"], **metrics}
            for record, metrics in zip(batch, lm_batch_metrics(reference_logits, candidate_logits, input_ids))
        )
        del input_ids, reference_logits, candidate_logits
        state.set_profile(None)
    state.set_profile(profile)
    return {"summary": summarize_records(result_records), "per_sample": result_records}


def evaluate_task(model: torch.nn.Module, state: ProfileState, profile: Profile, records: list[dict[str, Any]], device: torch.device, batch_size: int) -> dict[str, Any]:
    state.set_profile(profile)
    per_sample = []
    for batch in batch_records(records, batch_size):
        flattened: list[tuple[dict[str, Any], int, list[int]]] = []
        for record in batch:
            for choice_index, choice_ids in enumerate(record["choice_ids"]):
                ids = record["prompt_ids"] + choice_ids
                flattened.append((record, choice_index, ids))
        max_len = max(len(item[2]) for item in flattened)
        input_ids = torch.zeros((len(flattened), max_len), device=device, dtype=torch.long)
        attention = torch.zeros_like(input_ids)
        for row, (_, _, ids) in enumerate(flattened):
            input_ids[row, : len(ids)] = torch.tensor(ids, device=device)
            attention[row, : len(ids)] = 1
        with torch.inference_mode():
            logits = model(input_ids=input_ids, attention_mask=attention, use_cache=False, return_dict=True).logits.float()
        scores: dict[str, list[float]] = {}
        for row, (record, choice_index, ids) in enumerate(flattened):
            prompt_len = len(record["prompt_ids"])
            choice_len = len(record["choice_ids"][choice_index])
            positions = torch.arange(prompt_len - 1, prompt_len + choice_len - 1, device=device)
            token_ids = input_ids[row, prompt_len : prompt_len + choice_len]
            score = F.log_softmax(logits[row, positions], dim=-1).gather(-1, token_ids[:, None]).sum().item()
            scores.setdefault(record["sample_id"], []).append(score)
        for record in batch:
            choice_scores = scores[record["sample_id"]]
            predicted = int(np.argmax(choice_scores))
            per_sample.append({
                "sample_id": record["sample_id"],
                "label": record["label"],
                "predicted": predicted,
                "choice_scores": choice_scores,
                "correct": int(predicted == record["label"]),
            })
        del input_ids, attention, logits
    state.set_profile(None)
    return {
        "summary": {"accuracy": float(np.mean([row["correct"] for row in per_sample])) if per_sample else float("nan"), "count": len(per_sample)},
        "per_sample": per_sample,
    }


def bootstrap_mean(values: list[float], seed: int, iterations: int = 1000) -> dict[str, float | int]:
    if not values:
        return {"mean": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan"), "count": 0, "bootstrap_iterations": iterations}
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(array), size=(iterations, len(array)))
    means = array[samples].mean(axis=1)
    return {
        "mean": float(array.mean()),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
        "count": int(len(array)),
        "bootstrap_iterations": iterations,
    }


def bootstrap_difference(candidate: list[float], reference: list[float], seed: int, iterations: int = 1000) -> dict[str, float | int]:
    if len(candidate) != len(reference) or not candidate:
        return {"mean": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan"), "count": 0, "bootstrap_iterations": iterations}
    delta = np.asarray(candidate, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    return bootstrap_mean(delta.tolist(), seed, iterations)


def unit_risk(single: dict[str, Any], unit: Unit, bits: int) -> float:
    record = single[unit.key][str(bits)]
    # Negative calibration deltas are not treated as free quality gains when
    # allocating bits; they are sampling noise or an interaction artifact.
    return max(0.0, float(record["summary"].get("nll_delta", 0.0)))


def predicted_risk(units: list[Unit], single: dict[str, Any], bits: dict[str, int]) -> float:
    return float(sum(unit_risk(single, unit, bits[unit.key],) for unit in units))


def select_nearest(candidates: list[Profile], units: list[Unit], target_storage: int, single: dict[str, Any]) -> Profile:
    feasible = [candidate for candidate in candidates if int(candidate.storage(units)["total_bits"]) <= target_storage]
    if not feasible:
        feasible = candidates
    return min(
        feasible,
        key=lambda candidate: (
            abs(int(candidate.storage(units)["total_bits"]) - target_storage),
            predicted_risk(units, single, candidate.bits),
            candidate.name,
        ),
    )


def projection_policy(units: list[Unit], target_storage: int, single: dict[str, Any], fixed_fp16_bits: int = 0) -> Profile:
    candidates = []
    for values in itertools.product(BITS, repeat=len(UNIT_NAMES)):
        by_name = dict(zip(UNIT_NAMES, values))
        bits = {unit.key: by_name[unit.name] for unit in units}
        candidates.append(Profile("projection_only_candidate", "projection_only", bits, target_bits=8, optimizer="exhaustive 5^3 enumeration", fixed_fp16_bits=fixed_fp16_bits))
    selected = select_nearest(candidates, units, target_storage, single)
    selected.name = "projection_only_budgeted"
    return selected


def greedy_policy(units: list[Unit], target_storage: int, single: dict[str, Any], grouping: str, priority: bool = False, fixed_fp16_bits: int = 0) -> Profile:
    bits = {unit.key: 4 for unit in units}
    groups: list[list[Unit]]
    if grouping == "layer":
        groups = [[unit for unit in units if unit.layer == layer] for layer in sorted({unit.layer for unit in units})]
    elif grouping == "layer_by_projection":
        groups = [[unit] for unit in units]
    else:
        raise ValueError(grouping)
    current = Profile("working", grouping, bits, target_bits=8, fixed_fp16_bits=fixed_fp16_bits)
    while True:
        current_cost = int(current.storage(units)["total_bits"])
        options: list[tuple[float, float, str, list[Unit], int]] = []
        for group in groups:
            old_bits = bits[group[0].key]
            if any(bits[unit.key] != old_bits for unit in group):
                continue
            for new_bits in BITS:
                if new_bits <= old_bits:
                    continue
                candidate_bits = dict(bits)
                for unit in group:
                    candidate_bits[unit.key] = new_bits
                candidate = Profile("working", grouping, candidate_bits, target_bits=8, fixed_fp16_bits=fixed_fp16_bits)
                cost = int(candidate.storage(units)["total_bits"])
                if cost > target_storage:
                    continue
                reduction = predicted_risk(units, single, bits) - predicted_risk(units, single, candidate_bits)
                extra = cost - current_cost
                if extra <= 0:
                    continue
                ratio = reduction / extra
                name = ",".join(unit.key for unit in group)
                priority_value = 0.0
                if priority:
                    # V-first is deliberately interpretable rather than
                    # sensitivity-optimized: it prefers V upgrades, then Q/K,
                    # then O/FFN, with layer order as the final tie-break.
                    priority_value = sum(1.0 if unit.name == "v" else 0.0 for unit in group)
                options.append((ratio, priority_value, name, group, new_bits))
        if not options:
            break
        if priority:
            options.sort(key=lambda item: (-item[1], -item[0], item[2]))
        else:
            options.sort(key=lambda item: (-item[0], item[2], item[4]))
        _, _, _, chosen_group, chosen_bits = options[0]
        for unit in chosen_group:
            bits[unit.key] = chosen_bits
    name = "v_priority_budgeted" if priority else f"{grouping}_budgeted"
    kind = "heuristic" if priority else grouping
    return Profile(name, kind, bits, target_bits=8, optimizer="deterministic greedy upgrade under true storage budget", fixed_fp16_bits=fixed_fp16_bits)


def category_exact_options(category_units: list[Unit], single: dict[str, Any]) -> list[tuple[int, float, dict[str, int]]]:
    """Best assignment for each (number of W4, number of FP16) count."""
    # Dynamic programming keeps the lowest calibration risk for every count
    # pair, while retaining the actual layer assignment for reproducibility.
    states: dict[tuple[int, int], tuple[float, dict[str, int]]] = {(0, 0): (0.0, {})}
    for unit in category_units:
        next_states: dict[tuple[int, int], tuple[float, dict[str, int]]] = {}
        for (n4, n16), (risk, assignment) in states.items():
            for bits in BITS:
                new_counts = (n4 + int(bits == 4), n16 + int(bits == 16))
                new_assignment = dict(assignment)
                new_assignment[unit.key] = bits
                new_value = (risk + unit_risk(single, unit, bits), new_assignment)
                old = next_states.get(new_counts)
                if old is None or (new_value[0], sorted(new_assignment.items())) < (old[0], sorted(old[1].items())):
                    next_states[new_counts] = new_value
        states = next_states
    options = []
    for (n4, n16), (risk, assignment) in states.items():
        cost = sum(unit.storage(assignment[unit.key])["total_bits"] for unit in category_units)
        baseline = sum(unit.storage(8)["total_bits"] for unit in category_units)
        options.append((int(cost - baseline), risk, assignment))
    return options


def exact_layer_by_projection_policy(units: list[Unit], single: dict[str, Any], fixed_fp16_bits: int = 0) -> Profile:
    """Meet-in-the-middle DP for an exactly uniform-W8 storage total.

    The search is exact for the measured additive calibration objective and
    the integer representation ledger. It is not an assertion that the
    additive objective predicts the combined model (the interaction check is
    run afterward).
    """
    by_name = [[unit for unit in units if unit.name == name] for name in UNIT_NAMES]
    options = [category_exact_options(category, single) for category in by_name]

    def combine(groups: list[list[tuple[int, float, dict[str, int]]]]) -> dict[int, tuple[float, dict[str, int]]]:
        states: dict[int, tuple[float, dict[str, int]]] = {0: (0.0, {})}
        for group in groups:
            next_states: dict[int, tuple[float, dict[str, int]]] = {}
            for delta, (risk, assignment) in states.items():
                for option_delta, option_risk, option_assignment in group:
                    new_delta = delta + option_delta
                    new_risk = risk + option_risk
                    new_assignment = assignment | option_assignment
                    old = next_states.get(new_delta)
                    if old is None or (new_risk, sorted(new_assignment.items())) < (old[0], sorted(old[1].items())):
                        next_states[new_delta] = (new_risk, new_assignment)
            states = next_states
        return states

    left = combine(options[:2])
    right = combine(options[2:])
    best: tuple[float, dict[str, int]] | None = None
    for delta, (risk, assignment) in left.items():
        other = right.get(-delta)
        if other is None:
            continue
        candidate = (risk + other[0], assignment | other[1])
        if best is None or (candidate[0], sorted(candidate[1].items())) < (best[0], sorted(best[1].items())):
            best = candidate
    if best is None:
        raise RuntimeError("no exact layer-by-projection assignment matches the uniform W8 ledger")
    profile = Profile(
        "layer_by_projection_exact",
        "layer_by_projection",
        best[1],
        target_bits=8,
        optimizer="exact integer-budget meet-in-the-middle DP over unit-type count states",
        fixed_fp16_bits=fixed_fp16_bits,
    )
    target = int(uniform_profile(units, 8, fixed_fp16_bits=fixed_fp16_bits).storage(units)["total_bits"])
    if int(profile.storage(units)["total_bits"]) != target:
        raise AssertionError("exact structured profile does not match uniform W8 total")
    return profile


def policy_set(units: list[Unit], single: dict[str, Any], fixed_fp16_bits: int = 0) -> list[Profile]:
    target = int(uniform_profile(units, 8, fixed_fp16_bits=fixed_fp16_bits).storage(units)["total_bits"])
    policies = [uniform_profile(units, 4, fixed_fp16_bits=fixed_fp16_bits), uniform_profile(units, 8, fixed_fp16_bits=fixed_fp16_bits), uniform_profile(units, 16, fixed_fp16_bits=fixed_fp16_bits)]
    policies.append(projection_policy(units, target, single, fixed_fp16_bits))
    policies.append(greedy_policy(units, target, single, "layer", fixed_fp16_bits=fixed_fp16_bits))
    policies.append(greedy_policy(units, target, single, "layer_by_projection", fixed_fp16_bits=fixed_fp16_bits))
    policies.append(greedy_policy(units, target, single, "layer_by_projection", priority=True, fixed_fp16_bits=fixed_fp16_bits))
    policies.append(exact_layer_by_projection_policy(units, single, fixed_fp16_bits))
    return policies


def single_unit_measurements(model: torch.nn.Module, state: ProfileState, units: list[Unit], calibration: list[dict[str, Any]], device: torch.device, batch_size: int) -> dict[str, Any]:
    records: dict[str, Any] = {}
    for index, unit in enumerate(units, start=1):
        print(f"single-unit [{index}/{len(units)}] {unit.key}", flush=True)
        records[unit.key] = {}
        for bits in BITS:
            profile = Profile(f"single_{unit.key}_w{bits}", "single_unit", {item.key: (bits if item.key == unit.key else 16) for item in units})
            if bits == 16:
                metric_result = {"summary": {"nll_delta": 0.0, "logit_mse": 0.0, "kl_reference_to_candidate": 0.0}, "per_sample": []}
            else:
                metric_result = evaluate_lm(model, state, profile, calibration, device, batch_size)
            records[unit.key][str(bits)] = metric_result
    state.restore()
    return records


def additive_prediction(single: dict[str, Any], units: list[Unit], profile: Profile, sample_index: int) -> dict[str, float]:
    prediction = {"nll_delta": 0.0, "logit_mse": 0.0, "kl_reference_to_candidate": 0.0}
    for unit in units:
        bits = profile.bits[unit.key]
        if bits == 16:
            continue
        per_sample = single[unit.key][str(bits)].get("per_sample", [])
        if not per_sample or sample_index >= len(per_sample):
            continue
        for key in prediction:
            prediction[key] += max(0.0, float(per_sample[sample_index].get(key, 0.0)))
    return prediction


def interaction_summary(single: dict[str, Any], units: list[Unit], profile: Profile, actual: dict[str, Any]) -> dict[str, Any]:
    actual_rows = actual["per_sample"]
    predictions = [additive_prediction(single, units, profile, index) for index in range(len(actual_rows))]
    result = {}
    for key in ("nll_delta", "logit_mse", "kl_reference_to_candidate"):
        predicted = [row[key] for row in predictions]
        measured = [float(row[key]) for row in actual_rows]
        result[key] = {
            "predicted_mean": float(np.mean(predicted)) if predicted else float("nan"),
            "measured_mean": float(np.mean(measured)) if measured else float("nan"),
            "measured_minus_predicted": float(np.mean(measured) - np.mean(predicted)) if measured else float("nan"),
            "ratio_measured_to_predicted": float(np.mean(measured) / np.mean(predicted)) if predicted and np.mean(predicted) > 0 else None,
            "paired_difference_bootstrap": bootstrap_difference(measured, predicted, 1731),
        }
    return result


def attach_quality_intervals(result: dict[str, Any], seed: int, iterations: int = 1000) -> dict[str, Any]:
    rows = result["per_sample"]
    metrics = {}
    for key in ("nll", "nll_delta", "logit_mse", "kl_reference_to_candidate", "top1_match"):
        values = [float(row[key]) for row in rows]
        metric_offsets = {"nll": 11, "nll_delta": 23, "logit_mse": 37, "kl_reference_to_candidate": 41, "top1_match": 53}
        metrics[key] = bootstrap_mean(values, seed + metric_offsets[key], iterations)
    if result.get("task"):
        values = [float(row["correct"]) for row in result["task"]["per_sample"]]
        metrics["task_accuracy"] = bootstrap_mean(values, seed + 901, iterations)
    result["bootstrap_95"] = metrics
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path")
    parser.add_argument("--output")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--calibration-samples", type=int, default=16)
    parser.add_argument("--heldout-samples", type=int, default=32)
    parser.add_argument("--task-samples", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--cache-dir", default=os.environ.get("HF_DATASETS_CACHE"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--storage-self-test", action="store_true")
    args = parser.parse_args()
    if args.storage_self_test:
        print(json.dumps(manual_storage_cases(), indent=2))
        return
    if not args.model_path or not args.output:
        parser.error("--model-path and --output are required unless --storage-self-test is used")
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA is required for the full experiment")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    model_path = Path(args.model_path).expanduser().resolve()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map={"": device},
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    units = build_units(model)
    fixed_bits = fixed_fp16_storage_bits(model, units)
    state = ProfileState(model, units, device)
    calibration = load_text_sequences(tokenizer, "train", args.calibration_samples, args.seq_len, args.cache_dir)
    heldout = load_text_sequences(tokenizer, "validation", args.heldout_samples, args.seq_len, args.cache_dir)
    task = load_task_examples(tokenizer, args.task_samples, args.seed + 7, args.cache_dir)
    started = time.time()
    single = single_unit_measurements(model, state, units, calibration, device, args.batch_size)
    policies = policy_set(units, single, fixed_bits)
    # Keep policy evaluation separate from policy selection: assignments use
    # calibration single-unit results only, while headline quality is held out.
    policy_results: list[dict[str, Any]] = []
    for index, profile in enumerate(policies, start=1):
        print(f"policy [{index}/{len(policies)}] {profile.name} storage={profile.storage(units)['total_bits']} bits", flush=True)
        calibration_result = evaluate_lm(model, state, profile, calibration, device, args.batch_size)
        heldout_result = evaluate_lm(model, state, profile, heldout, device, args.batch_size)
        task_result = evaluate_task(model, state, profile, task, device, args.batch_size)
        row = {
            "profile": profile.as_record(units),
            "calibration": calibration_result,
            "heldout": heldout_result,
            "task": task_result,
            "interaction_calibration": interaction_summary(single, units, profile, calibration_result),
        }
        row["heldout"] = attach_quality_intervals(row["heldout"], args.seed + index * 100, args.bootstrap_iterations)
        row["task"]["bootstrap_95"] = bootstrap_mean(
            [float(item["correct"]) for item in row["task"]["per_sample"]],
            args.seed + index * 100 + 901,
            args.bootstrap_iterations,
        )
        policy_results.append(row)
    state.restore()

    # The oracle is deliberately computed only over profiles actually executed
    # at the same modeled storage cost (or the closest reported cost). It is an
    # upper bound on query routing, not a deployable policy.
    target = int(uniform_profile(units, 8, fixed_fp16_bits=fixed_bits).storage(units)["total_bits"])
    min_gap = min(abs(int(row["profile"]["storage"]["total_bits"]) - target) for row in policy_results)
    oracle_candidates = [row for row in policy_results if abs(int(row["profile"]["storage"]["total_bits"]) - target) == min_gap]
    global_best = min(oracle_candidates, key=lambda row: row["heldout"]["summary"].get("nll", float("inf")))
    by_sample = []
    for sample_index, sample in enumerate(heldout):
        choices = []
        for row in oracle_candidates:
            candidate_row = next(item for item in row["heldout"]["per_sample"] if item["sample_id"] == sample["sample_id"])
            choices.append((float(candidate_row["nll"]), row["profile"]["name"], candidate_row))
        best = min(choices, key=lambda item: item[0])
        by_sample.append({"sample_id": sample["sample_id"], "chosen_profile": best[1], "nll": best[0]})
    oracle_nll = [row["nll"] for row in by_sample]
    global_nll = [next(item["nll"] for item in global_best["heldout"]["per_sample"] if item["sample_id"] == row["sample_id"]) for row in by_sample]
    query_oracle = {
        "budget_target_bits": target,
        "candidate_storage_gap_bits": min_gap,
        "candidate_profiles": [row["profile"] for row in oracle_candidates],
        "global_profile": global_best["profile"],
        "oracle_is_upper_bound": True,
        "global_nll": bootstrap_mean(global_nll, args.seed + 5001, args.bootstrap_iterations),
        "oracle_nll": bootstrap_mean(oracle_nll, args.seed + 5002, args.bootstrap_iterations),
        "oracle_minus_global_nll": bootstrap_difference(oracle_nll, global_nll, args.seed + 5003, args.bootstrap_iterations),
        "oracle_mse_reduction_fraction": None,
        "choices": by_sample,
    }
    global_mse = [next(item["logit_mse"] for item in global_best["heldout"]["per_sample"] if item["sample_id"] == row["sample_id"]) for row in by_sample]
    # MSE oracle uses the same per-sample candidate choice selected by NLL only
    # and is reported as descriptive, not as a second optimization target.
    oracle_mse = []
    for row in by_sample:
        selected = next(item for item in oracle_candidates if item["profile"]["name"] == row["chosen_profile"])
        oracle_mse.append(next(item["logit_mse"] for item in selected["heldout"]["per_sample"] if item["sample_id"] == row["sample_id"]))
    query_oracle["oracle_mse_reduction_fraction"] = 1.0 - float(np.mean(oracle_mse)) / float(np.mean(global_mse)) if np.mean(global_mse) > 0 else 0.0
    query_oracle["global_logit_mse"] = bootstrap_mean(global_mse, args.seed + 5004, args.bootstrap_iterations)
    query_oracle["oracle_logit_mse"] = bootstrap_mean(oracle_mse, args.seed + 5005, args.bootstrap_iterations)
    query_oracle["oracle_to_global_logit_mse_ratio"] = float(np.mean(oracle_mse) / np.mean(global_mse)) if np.mean(global_mse) > 0 else None

    fp16_row = next(row for row in policy_results if row["profile"]["name"] == "uniform_w16")
    fp16_rows = fp16_row["heldout"]["per_sample"]
    fp16_noop = {
        "max_abs_nll_delta": max((abs(float(row["nll_delta"])) for row in fp16_rows), default=0.0),
        "max_logit_mse": max((float(row["logit_mse"]) for row in fp16_rows), default=0.0),
        "max_kl_reference_to_candidate": max((float(row["kl_reference_to_candidate"]) for row in fp16_rows), default=0.0),
        "min_top1_match": min((float(row["top1_match"]) for row in fp16_rows), default=1.0),
        "defined_tolerance": {"nll_delta": 0.0, "logit_mse": 0.0, "kl": 0.0, "top1_mismatch": 0},
    }

    calibration_best = min(policy_results, key=lambda row: row["calibration"]["summary"].get("nll", float("inf")))
    calibration_structured = min(
        [row for row in policy_results if row["profile"]["kind"] != "uniform"],
        key=lambda row: row["calibration"]["summary"].get("nll", float("inf")),
    )

    result = {
        "schema_version": 2,
        "experiment": "structured_layer_by_projection_weight_precision",
        "created_unix": time.time(),
        "elapsed_seconds": time.time() - started,
        "scope": {
            "weight_projection_quantization": True,
            "kv_cache_quantization": False,
            "native_low_bit_kernel": False,
            "scheduler_or_router": False,
            "ffn_unit": "gate_proj + up_proj + down_proj as one layer-level unit",
        },
        "proxy": {
            "kind": "symmetric_weight_only_fake_quantization",
            "bits": list(BITS),
            "group_size": GROUP_SIZE,
            "scale_bits": SCALE_BITS,
            "zero_point": False,
            "fp16_scale_overhead": False,
            "representation_note": "Quantized payloads count input-channel padding and one FP16 scale per group; FP16 counts direct weights only; packed headers are not modeled.",
            "native_speedup_claim": False,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
            "swiftllm_upstream_commit": git_revision(),
            "swiftllm_research_diff_sha256": sha256(ROOT / "references/swiftllm-research.diff"),
            "requirements_lock_sha256": sha256(ROOT / "requirements-lock.txt"),
            "datasets_version": __import__("datasets").__version__,
        },
        "model": {
            "path": str(model_path),
            "config": json.loads((model_path / "config.json").read_text()),
            "files": [{"name": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)} for path in sorted(model_path.iterdir()) if path.is_file() and path.suffix in {".json", ".safetensors"}],
        },
        "data": {
            "calibration": {"dataset": "wikitext-2-raw-v1/train", "count": len(calibration), "seq_len": args.seq_len, "sample_ids": [row["sample_id"] for row in calibration]},
            "heldout": {"dataset": "wikitext-2-raw-v1/validation", "count": len(heldout), "seq_len": args.seq_len, "sample_ids": [row["sample_id"] for row in heldout]},
            "task": {"dataset": "Rowan/hellaswag/validation", "count": len(task), "seed": args.seed + 7, "metric": "teacher-forced multiple-choice accuracy"},
            "limitation": "Counts are explicit CLI-selected samples; they are not a claim of benchmark-scale uncertainty. Increase counts for publication-level estimates.",
        },
        "storage_accounting": {
            "manual_cases": manual_storage_cases(),
            "fixed_fp16_bits": fixed_bits,
            "uniform_profiles": {str(bits): uniform_profile(units, bits, fixed_fp16_bits=fixed_bits).storage(units) for bits in BITS},
            "unit_shapes": [{"key": unit.key, "name": unit.name, "layer": unit.layer, "matrices": [matrix.__dict__ | {"numel": matrix.numel, "padded_in_features": matrix.padded_in_features, "scale_count": matrix.scale_count} for matrix in unit.matrices]} for unit in units],
        },
        "single_unit_sensitivity": single,
        "policy_search": {
            "candidate_generation": "single-unit calibration NLL deltas with exhaustive projection enumeration, greedy budgeted upgrades, V-first heuristic, and exact integer-budget DP",
            "combined_calibration_best_profile": calibration_best["profile"],
            "combined_calibration_best_structured_profile": calibration_structured["profile"],
            "combined_calibration_metrics_are_used_for_selection": True,
            "heldout_metrics_are_not_used_for_selection": True,
            "additive_prediction_is_not_final_selection_evidence": True,
        },
        "policies": policy_results,
        "query_oracle": query_oracle,
        "verification": {
            "all_fp16_noop": fp16_noop,
            "selection_uses_calibration_only": True,
            "headline_uses_heldout": True,
            "additive_model_is_candidate_generator_only": True,
            "interaction_adaptation": "All candidate policies are re-executed as combined profiles on calibration and held-out data; additive predictions are not treated as quality evidence, and no native/router phase is opened after interaction failure.",
            "bootstrap_iterations": args.bootstrap_iterations,
            "budgets_exact_only_when_total_bits_equal": True,
        },
    }
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
