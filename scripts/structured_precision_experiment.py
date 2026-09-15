#!/usr/bin/env python3
"""Interaction-aware structured weight-precision evidence experiment.

This is an offline numerical experiment for the *weight* proxy only.  It
starts at a uniform W8 model, measures every Q/K/V/O/FFN unit relative to that
same W8 model, and searches by executing combined profiles on calibration
shards.  The single-unit measurements are move proposals only: profile
selection uses measured combined-model calibration results and a shard
stability rule.

The proxy is symmetric groupwise quantize/dequantize back to FP16.  It is not a
packed W4/W8 kernel and says nothing about serving speed, KV-cache precision,
or scheduler behavior.
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
STORAGE_FIELDS = (
    "weight_payload_bits",
    "padding_bits",
    "scale_bits",
    "zero_point_bits",
    "metadata_bits",
    "total_bits",
)


@dataclass(frozen=True)
class MatrixShape:
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

    def storage(self, bits: int) -> dict[str, int | bool]:
        if bits == 16:
            weight_payload_bits = 16 * self.numel
            padding_bits = scale_bits = zero_point_bits = 0
        elif bits in (4, 8):
            quantized_numel = sum(matrix.quantized_numel for matrix in self.matrices)
            weight_payload_bits = bits * quantized_numel
            padding_bits = bits * sum(matrix.quantized_numel - matrix.numel for matrix in self.matrices)
            scale_bits = SCALE_BITS * sum(matrix.scale_count for matrix in self.matrices)
            zero_point_bits = 0
        else:
            raise ValueError(f"unsupported bits: {bits}")
        total_bits = weight_payload_bits + scale_bits + zero_point_bits
        return {
            "bits": bits,
            "weight_payload_bits": weight_payload_bits,
            "padding_bits": padding_bits,
            "scale_bits": scale_bits,
            "zero_point_bits": zero_point_bits,
            "metadata_bits": 0,
            "total_bits": total_bits,
            "total_bytes": total_bits // 8,
            "includes_padding": bits != 16,
            "includes_scale": bits != 16,
            "includes_zero_point": False,
        }


@dataclass(frozen=True)
class Profile:
    name: str
    kind: str
    bits: dict[str, int]
    target_bits: int | None = None
    optimizer: str | None = None
    fixed_fp16_bits: int = 0

    def signature(self, units: list[Unit]) -> str:
        return ",".join(str(self.bits[unit.key]) for unit in units)

    def storage(self, units: list[Unit]) -> dict[str, int | str]:
        totals: dict[str, int] = {field: 0 for field in STORAGE_FIELDS}
        for unit in units:
            cost = unit.storage(int(self.bits[unit.key]))
            for field in STORAGE_FIELDS:
                totals[field] += int(cost[field])
        totals["weight_payload_bits"] += self.fixed_fp16_bits
        totals["total_bits"] += self.fixed_fp16_bits
        totals["fixed_fp16_bits"] = self.fixed_fp16_bits
        totals["total_bytes"] = totals["total_bits"] // 8
        totals["profile_unit_count"] = len(units)
        totals["representation"] = "FP16 direct or symmetric groupwise fake-quantized payload; no packed metadata"
        return totals

    def as_record(self, units: list[Unit], baseline: Profile | None = None) -> dict[str, Any]:
        storage = self.storage(units)
        record: dict[str, Any] = {
            "name": self.name,
            "kind": self.kind,
            "optimizer": self.optimizer,
            "target_bits": self.target_bits,
            "signature": self.signature(units),
            "unit_bits": self.bits,
            "storage": storage,
        }
        if baseline is not None:
            record["storage_delta_vs_uniform_w8"] = storage_delta(storage, baseline.storage(units))
        return record


class ProfileState:
    """Apply profiles while retaining one original CPU copy per unit."""

    def __init__(self, model: torch.nn.Module, units: list[Unit], device: torch.device):
        self.model = model
        self.units = units
        self.device = device
        self.original: dict[str, list[torch.Tensor]] = {}
        self.parameters: dict[str, list[torch.nn.Parameter]] = {}
        for unit in units:
            layer = model.model.layers[unit.layer]
            params: list[torch.nn.Parameter] = []
            originals: list[torch.Tensor] = []
            for matrix in unit.matrices:
                parameter = parameter_for_matrix(layer, matrix.name)
                params.append(parameter)
                originals.append(parameter.detach().cpu().clone())
            self.parameters[unit.key] = params
            self.original[unit.key] = originals
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
    if name == "ffn_gate":
        return layer.mlp.gate_proj.weight
    if name == "ffn_up":
        return layer.mlp.up_proj.weight
    if name == "ffn_down":
        return layer.mlp.down_proj.weight
    raise KeyError(name)


def build_units(model: torch.nn.Module) -> list[Unit]:
    units: list[Unit] = []
    unit_matrices = {
        "q": ("q",),
        "k": ("k",),
        "v": ("v",),
        "o": ("o",),
        "ffn": ("ffn_gate", "ffn_up", "ffn_down"),
    }
    for layer_id, layer in enumerate(model.model.layers):
        for unit_name, matrix_names in unit_matrices.items():
            shapes = []
            for matrix_name in matrix_names:
                parameter = parameter_for_matrix(layer, matrix_name)
                shapes.append(MatrixShape(matrix_name, parameter.shape[0], parameter.shape[1]))
            units.append(Unit(layer_id, unit_name, tuple(shapes)))
    return units


def fixed_fp16_storage_bits(model: torch.nn.Module, units: list[Unit]) -> int:
    unit_parameter_ids = set()
    for unit in units:
        layer = model.model.layers[unit.layer]
        unit_parameter_ids.update(id(parameter_for_matrix(layer, matrix.name)) for matrix in unit.matrices)
    fixed_numel = sum(parameter.numel() for parameter in model.parameters() if id(parameter) not in unit_parameter_ids)
    return int(fixed_numel * 16)


def profile_from_unit_bits(name: str, kind: str, units: list[Unit], values: dict[str, int], **kwargs: Any) -> Profile:
    return Profile(name, kind, {unit.key: int(values[unit.key]) for unit in units}, **kwargs)


def uniform_profile(units: list[Unit], bits: int, name: str | None = None, fixed_fp16_bits: int = 0) -> Profile:
    return profile_from_unit_bits(
        name or f"uniform_w{bits}",
        "uniform",
        units,
        {unit.key: bits for unit in units},
        target_bits=bits,
        fixed_fp16_bits=fixed_fp16_bits,
    )


def storage_delta(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, int]:
    result = {field: int(candidate[field]) - int(baseline[field]) for field in STORAGE_FIELDS}
    result["fixed_fp16_bits"] = int(candidate.get("fixed_fp16_bits", 0)) - int(baseline.get("fixed_fp16_bits", 0))
    result["total_bytes"] = result["total_bits"] // 8
    return result


def manual_storage_cases() -> dict[str, Any]:
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
            "bytes_are_exact_integers": all(isinstance(divisible.storage(bits)["total_bytes"], int) for bits in BITS),
        },
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision(path: Path) -> str | None:
    marker = path / "UPSTREAM_COMMIT"
    if marker.exists():
        return marker.read_text().strip()
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def git_source_provenance() -> dict[str, str | None]:
    try:
        branch = subprocess.check_output(["git", "-C", str(ROOT), "branch", "--show-current"], text=True).strip()
        commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        branch = commit = None
    return {"branch": branch, "commit_at_run": commit, "base_commit": commit}


def tokenize_split(tokenizer: Any, split: str, cache_dir: str | None) -> list[int]:
    dataset = load_dataset(
        "wikitext",
        "wikitext-2-raw-v1",
        split=split,
        cache_dir=cache_dir,
        download_mode="reuse_dataset_if_exists",
    )
    text = "\n".join(row["text"] for row in dataset if row["text"].strip())
    return tokenizer(text, add_special_tokens=False, return_attention_mask=False)["input_ids"]


def dispersed_windows(
    tokenizer: Any,
    split: str,
    shards: int,
    samples_per_shard: int,
    seq_len: int,
    seed: int,
    cache_dir: str | None,
) -> tuple[list[list[dict[str, Any]]], dict[str, Any]]:
    """Choose deterministic random, dispersed, non-overlapping token windows.

    Starts are sampled independently inside evenly spaced corpus buckets.  The
    bucket construction is only a sampling mechanism; all quality decisions
    use the actual combined-model execution on each shard.
    """
    if shards < 1 or samples_per_shard < 1:
        raise ValueError("shards and samples_per_shard must be positive")
    ids = tokenize_split(tokenizer, split, cache_dir)
    max_start = len(ids) - seq_len
    if max_start < 0:
        raise RuntimeError(f"Wikitext {split} has only {len(ids)} tokens; need {seq_len}")
    total = shards * samples_per_shard
    if max_start + 1 < total:
        raise RuntimeError(f"not enough windows in Wikitext {split} for {total} samples")
    rng = random.Random(seed)
    starts: list[int] = []
    bucket_width = max(1, (max_start + 1) // total)
    for bucket in range(total):
        lo = min(max_start, bucket * bucket_width)
        hi = min(max_start, (bucket + 1) * bucket_width - 1)
        choices = list(range(lo, hi + 1))
        rng.shuffle(choices)
        selected = next((candidate for candidate in choices if all(abs(candidate - old) >= seq_len for old in starts)), None)
        if selected is None:
            # This fallback is deterministic and is only relevant for tiny
            # synthetic corpora where a bucket is narrower than a window.
            candidates = list(range(max_start + 1))
            rng.shuffle(candidates)
            selected = next((candidate for candidate in candidates if all(abs(candidate - old) >= seq_len for old in starts)), None)
        if selected is None:
            raise RuntimeError(f"could not select non-overlapping windows for Wikitext {split}")
        starts.append(selected)
    starts.sort()
    intervals = [[start, start + seq_len] for start in starts]
    if any(right > left for (_, right), (left, _) in zip(intervals, intervals[1:])):
        raise AssertionError("dispersed window selector produced overlap")
    result = [[] for _ in range(shards)]
    for index, start in enumerate(starts):
        shard_id = index % shards
        result[shard_id].append({
            "sample_id": f"wikitext_{split}_shard{shard_id:02d}_window{index:04d}_start{start:08d}",
            "input_ids": ids[start : start + seq_len],
            "window_start": start,
            "window_end": start + seq_len,
            "shard": shard_id,
            "split": split,
        })
    metadata = {
        "dataset": f"wikitext-2-raw-v1/{split}",
        "count": total,
        "shard_count": shards,
        "samples_per_shard": samples_per_shard,
        "seq_len": seq_len,
        "selection_seed": seed,
        "selection_method": "randomized evenly spaced token buckets with non-overlap rejection",
        "sample_ids": [record["sample_id"] for shard in result for record in shard],
        "shards": [
            {
                "shard_id": shard_id,
                "sample_ids": [record["sample_id"] for record in shard_records],
                "window_starts": [record["window_start"] for record in shard_records],
                "window_intervals": [[record["window_start"], record["window_end"]] for record in shard_records],
            }
            for shard_id, shard_records in enumerate(result)
        ],
        "all_windows_non_overlapping": True,
        "dispersed_start_min": min(starts),
        "dispersed_start_max": max(starts),
    }
    return result, metadata


def batch_records(records: list[dict[str, Any]], batch_size: int) -> Iterable[tuple[int, list[dict[str, Any]]]]:
    for start in range(0, len(records), batch_size):
        yield start, records[start : start + batch_size]


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
    keys = [key for key, value in records[0].items() if isinstance(value, (float, int)) and key != "token_count"]
    summary = {key: float(np.mean([float(record[key]) for record in records])) for key in keys}
    if "nll" in summary:
        summary["perplexity_from_mean_nll"] = math.exp(min(summary["nll"], 30.0))
    if "reference_nll" in summary:
        summary["reference_perplexity_from_mean_nll"] = math.exp(min(summary["reference_nll"], 30.0))
    return summary


def build_reference_cache(
    model: torch.nn.Module,
    state: ProfileState,
    profile: Profile,
    shards: list[list[dict[str, Any]]],
    device: torch.device,
    batch_size: int,
) -> list[list[torch.Tensor]]:
    """Cache reference logits on CPU once; candidates still execute normally."""
    state.set_profile(profile)
    cache: list[list[torch.Tensor]] = []
    with torch.inference_mode():
        for records in shards:
            shard_cache: list[torch.Tensor] = []
            for _, batch in batch_records(records, batch_size):
                input_ids = torch.tensor([record["input_ids"] for record in batch], device=device, dtype=torch.long)
                logits = model(input_ids=input_ids, use_cache=False, return_dict=True).logits.detach().cpu().half()
                shard_cache.extend(row for row in logits)
                del input_ids, logits
            cache.append(shard_cache)
    state.restore()
    return cache


def evaluate_lm_against_reference(
    model: torch.nn.Module,
    state: ProfileState,
    profile: Profile,
    records: list[dict[str, Any]],
    reference_logits: list[torch.Tensor],
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    if len(records) != len(reference_logits):
        raise ValueError("reference cache and records differ")
    state.set_profile(profile)
    result_records: list[dict[str, Any]] = []
    with torch.inference_mode():
        for start, batch in batch_records(records, batch_size):
            input_ids = torch.tensor([record["input_ids"] for record in batch], device=device, dtype=torch.long)
            reference = torch.stack(reference_logits[start : start + len(batch)]).to(device=device)
            candidate = model(input_ids=input_ids, use_cache=False, return_dict=True).logits
            result_records.extend(
                {"sample_id": record["sample_id"], **metrics}
                for record, metrics in zip(batch, lm_batch_metrics(reference, candidate, input_ids))
            )
            del input_ids, reference, candidate
    state.restore()
    return {"summary": summarize_records(result_records), "per_sample": result_records}


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


def calibration_stability(result: dict[str, Any], shard_count: int, tolerance: float = 0.002) -> dict[str, Any]:
    shard_deltas = [float(shard["summary"].get("nll_delta", 0.0)) for shard in result.get("shards", [])]
    mean = float(np.mean(shard_deltas)) if shard_deltas else float("nan")
    worst = float(max(shard_deltas)) if shard_deltas else float("nan")
    std = float(np.std(shard_deltas)) if shard_deltas else float("nan")
    improved = sum(delta < 0.0 for delta in shard_deltas)
    required = max(2, math.ceil(shard_count * 2 / 3))
    return {
        "shard_nll_deltas": shard_deltas,
        "mean_nll_delta": mean,
        "worst_shard_nll_delta": worst,
        "shard_std_nll_delta": std,
        "improved_shard_count": improved,
        "required_improved_shards": required,
        "tolerance_nll": tolerance,
        "stable_improvement": bool(mean < 0.0 and improved >= required and worst <= tolerance),
        "stable_rank_score": float(mean + 0.5 * max(0.0, worst) + 0.25 * std) if shard_deltas else float("inf"),
    }


def evaluate_calibration(
    model: torch.nn.Module,
    state: ProfileState,
    profile: Profile,
    calibration_shards: list[list[dict[str, Any]]],
    reference_cache: list[list[torch.Tensor]],
    device: torch.device,
    batch_size: int,
    stability_tolerance: float,
) -> dict[str, Any]:
    shard_results = []
    all_records = []
    for shard_id, (records, references) in enumerate(zip(calibration_shards, reference_cache)):
        result = evaluate_lm_against_reference(model, state, profile, records, references, device, batch_size)
        shard_results.append({"shard_id": shard_id, "summary": result["summary"], "per_sample": result["per_sample"]})
        all_records.extend(result["per_sample"])
    result = {"summary": summarize_records(all_records), "per_sample": all_records, "shards": shard_results}
    result["stability"] = calibration_stability(result, len(calibration_shards), stability_tolerance)
    return result


def zero_calibration_result(reference: dict[str, Any], shard_count: int) -> dict[str, Any]:
    records = []
    shards = []
    for shard_id in range(shard_count):
        shards.append({"shard_id": shard_id, "summary": {"nll_delta": 0.0, "logit_mse": 0.0, "kl_reference_to_candidate": 0.0}, "per_sample": []})
    return {
        "summary": {"nll_delta": 0.0, "logit_mse": 0.0, "kl_reference_to_candidate": 0.0},
        "per_sample": records,
        "shards": shards,
        "stability": {"reference": reference, "stable_improvement": False},
    }


def profile_rank(record: dict[str, Any]) -> tuple[float, float, float, str]:
    stability = record["calibration"]["stability"]
    return (
        float(stability.get("stable_rank_score", float("inf"))),
        float(stability.get("worst_shard_nll_delta", float("inf"))),
        float(stability.get("shard_std_nll_delta", float("inf"))),
        str(record["profile"]["signature"]),
    )


def single_unit_measurements(
    model: torch.nn.Module,
    state: ProfileState,
    units: list[Unit],
    baseline: Profile,
    calibration_shards: list[list[dict[str, Any]]],
    reference_cache: list[list[torch.Tensor]],
    device: torch.device,
    batch_size: int,
    stability_tolerance: float,
) -> dict[str, Any]:
    records: dict[str, Any] = {}
    for index, unit in enumerate(units, start=1):
        print(f"W8-centered marginal [{index}/{len(units)}] {unit.key}", flush=True)
        records[unit.key] = {}
        for bits in BITS:
            profile = Profile(
                f"w8_marginal_{unit.key}_to_w{bits}",
                "w8_centered_marginal",
                {item.key: (bits if item.key == unit.key else 8) for item in units},
                target_bits=8,
                fixed_fp16_bits=baseline.fixed_fp16_bits,
            )
            if bits == 8:
                calibration = zero_calibration_result("uniform_w8", len(calibration_shards))
                measured = False
            else:
                calibration = evaluate_calibration(
                    model, state, profile, calibration_shards, reference_cache, device, batch_size, stability_tolerance
                )
                measured = True
            records[unit.key][str(bits)] = {
                "from_bits": 8,
                "to_bits": bits,
                "measured_combined_profile": measured,
                "profile": profile.as_record(units, baseline),
                "storage_delta_vs_uniform_w8": storage_delta(profile.storage(units), baseline.storage(units)),
                "calibration": calibration,
            }
    state.restore()
    return records


def projection_profiles(units: list[Unit], fixed_fp16_bits: int) -> list[Profile]:
    profiles = []
    for index, values in enumerate(itertools.product(BITS, repeat=len(UNIT_NAMES))):
        by_name = dict(zip(UNIT_NAMES, values))
        bits = {unit.key: by_name[unit.name] for unit in units}
        profiles.append(Profile(
            f"projection_only_{index:03d}",
            "projection_only",
            bits,
            target_bits=8,
            optimizer="exhaustive 3^5 enumeration (243 assignments)",
            fixed_fp16_bits=fixed_fp16_bits,
        ))
    return profiles


def marginal_proposal_ranks(single: dict[str, Any], units: list[Unit]) -> tuple[list[Unit], list[Unit]]:
    # These ranks only bound deterministic move generation.  They are never
    # used to score or select a final combined profile.
    def score(unit: Unit, bits: int) -> float:
        return float(single[unit.key][str(bits)]["calibration"]["summary"].get("nll_delta", 0.0))
    downgrades = sorted(units, key=lambda unit: (score(unit, 4), unit.key))
    upgrades = sorted(units, key=lambda unit: (score(unit, 16), unit.key))
    return downgrades, upgrades


def neighbor_profiles(
    anchor: Profile,
    units: list[Unit],
    target_storage: int,
    fixed_fp16_bits: int,
    downgrade_rank: list[Unit],
    upgrade_rank: list[Unit],
    proposal_width: int,
) -> list[tuple[Profile, str]]:
    current = anchor.bits
    proposals: dict[str, tuple[Profile, str, tuple[Any, ...]]] = {}
    def add(bits: dict[str, int], move: str, order: tuple[Any, ...]) -> None:
        candidate = Profile("search_candidate", "interaction_aware_search", dict(bits), target_bits=8, optimizer="bounded deterministic beam/coordinate search; actual combined calibration ranking", fixed_fp16_bits=fixed_fp16_bits)
        if int(candidate.storage(units)["total_bits"]) <= target_storage and candidate.signature(units) != anchor.signature(units):
            old = proposals.get(candidate.signature(units))
            value = (candidate, move, order)
            if old is None or order < old[2]:
                proposals[candidate.signature(units)] = value

    # Coordinate moves let accepted profiles be recomputed rather than treating
    # the W8 margins as a complete allocation search.
    for rank, unit in enumerate(downgrade_rank[:proposal_width]):
        if current[unit.key] != 4:
            bits = dict(current)
            bits[unit.key] = 4
            add(bits, f"coordinate:{unit.key}->{4}", (0, rank, unit.key, 4))
    for rank, unit in enumerate(upgrade_rank[:proposal_width]):
        if current[unit.key] != 16:
            bits = dict(current)
            bits[unit.key] = 16
            add(bits, f"coordinate:{unit.key}->{16}", (1, rank, unit.key, 16))
    # Explicit paired 8->4 downgrade plus compensating 8->16 upgrade.
    for down_rank, down in enumerate(downgrade_rank[:proposal_width]):
        if current[down.key] != 8:
            continue
        for up_rank, up in enumerate(upgrade_rank[:proposal_width]):
            if up.key == down.key or current[up.key] != 8:
                continue
            bits = dict(current)
            bits[down.key] = 4
            bits[up.key] = 16
            add(bits, f"paired:{down.key}->4,{up.key}->16", (2, down_rank, up_rank, down.key, up.key))
    return [(item[0], item[1]) for item in sorted(proposals.values(), key=lambda item: item[2])]


def actual_calibration_frontier(records: list[dict[str, Any]], target_storage: int) -> list[dict[str, Any]]:
    feasible = [record for record in records if int(record["profile"]["storage"]["total_bits"]) <= target_storage]
    frontier = []
    for candidate in feasible:
        candidate_cost = int(candidate["profile"]["storage"]["total_bits"])
        candidate_quality = float(candidate["calibration"]["summary"].get("nll_delta", float("inf")))
        dominated = any(
            int(other["profile"]["storage"]["total_bits"]) <= candidate_cost
            and float(other["calibration"]["summary"].get("nll_delta", float("inf"))) <= candidate_quality
            and (
                int(other["profile"]["storage"]["total_bits"]) < candidate_cost
                or float(other["calibration"]["summary"].get("nll_delta", float("inf"))) < candidate_quality
            )
            for other in feasible
        )
        if not dominated:
            frontier.append(candidate)
    return sorted(frontier, key=lambda record: (int(record["profile"]["storage"]["total_bits"]), profile_rank(record)))


def paired_against_w8(result: dict[str, Any], iterations: int, seed: int) -> dict[str, Any]:
    rows = result["per_sample"]
    metrics = {}
    for offset, key in enumerate(("nll_delta", "logit_mse", "kl_reference_to_candidate", "top1_match"), start=1):
        metrics[key] = bootstrap_mean([float(row[key]) for row in rows], seed + offset, iterations)
    return metrics


def attach_heldout_evidence(result: dict[str, Any], iterations: int, seed: int) -> dict[str, Any]:
    result["bootstrap_seed"] = seed
    result["bootstrap_95"] = paired_against_w8(result, iterations, seed)
    result["paired_nll_difference_vs_uniform_w8"] = result["bootstrap_95"]["nll_delta"]
    result["paired_metric_differences_vs_uniform_w8"] = result["bootstrap_95"]
    return result


def model_file_records(model_path: Path) -> list[dict[str, Any]]:
    return [
        {"name": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(model_path.iterdir())
        if path.is_file() and path.suffix in {".json", ".safetensors"}
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path")
    parser.add_argument("--output")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--calibration-shards", type=int, default=3)
    parser.add_argument("--calibration-samples-per-shard", type=int, default=4)
    parser.add_argument("--heldout-shards", type=int, default=4)
    parser.add_argument("--heldout-samples-per-shard", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--cache-dir", default=os.environ.get("HF_DATASETS_CACHE"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--search-evaluation-budget", type=int, default=96)
    parser.add_argument("--search-rounds", type=int, default=3)
    parser.add_argument("--beam-width", type=int, default=2)
    parser.add_argument("--proposal-width", type=int, default=8)
    parser.add_argument("--stability-tolerance-nll", type=float, default=0.002)
    parser.add_argument("--run-mode", choices=("exploration", "confirmation"), default="exploration")
    parser.add_argument("--skip-projection-enumeration", action="store_true", help="confirmation mode: skip the 1B-only 3^5 projection sanity check")
    parser.add_argument("--skip-w8-marginals", action="store_true", help="confirmation mode: use deterministic structural move proposals instead of repeating 1B marginal sweeps")
    parser.add_argument("--gate-artifact", default=str(ROOT / "results/sensitivity/llama32_1b_interaction_aware.json"), help="1B gate artifact required before confirmation mode")
    parser.add_argument("--storage-self-test", action="store_true")
    args = parser.parse_args()
    if args.storage_self_test:
        print(json.dumps(manual_storage_cases(), indent=2))
        return
    if not args.model_path or not args.output:
        parser.error("--model-path and --output are required unless --storage-self-test is used")
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA is required for the full experiment")
    gate_artifact = Path(args.gate_artifact).expanduser().resolve()
    gate_data: dict[str, Any] | None = None
    if args.run_mode == "confirmation":
        if not gate_artifact.exists():
            raise RuntimeError(f"confirmation requires the 1B gate artifact: {gate_artifact}")
        gate_data = json.loads(gate_artifact.read_text())
        if gate_data.get("final_gate", {}).get("decision") != "OPEN_8B_CONFIRMATION":
            raise RuntimeError("confirmation is gated on a prior 1B OPEN_8B_CONFIRMATION decision")
        if not args.skip_projection_enumeration or not args.skip_w8_marginals:
            raise RuntimeError("confirmation mode requires the explicit 1B-only projection/marginal scope exclusions")

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
    calibration_shards, calibration_data = dispersed_windows(
        tokenizer, "train", args.calibration_shards, args.calibration_samples_per_shard, args.seq_len, args.seed + 101, args.cache_dir
    )
    baseline = uniform_profile(units, 8, fixed_fp16_bits=fixed_bits)
    print("building uniform-W8 calibration reference", flush=True)
    calibration_reference = build_reference_cache(model, state, baseline, calibration_shards, device, args.batch_size)
    started = time.time()

    # Every subsequent quality number is paired to this same combined W8
    # reference.  The cache is only logits, not a quality shortcut.
    evaluation_registry: dict[str, dict[str, Any]] = {}
    def register(profile: Profile, calibration: dict[str, Any], origin: str) -> dict[str, Any]:
        signature = profile.signature(units)
        row = evaluation_registry.get(signature)
        if row is None:
            row = {"profile": profile.as_record(units, baseline), "calibration": calibration, "origins": [origin]}
            evaluation_registry[signature] = row
        elif origin not in row["origins"]:
            row["origins"].append(origin)
        return row

    base_calibration = evaluate_calibration(
        model, state, baseline, calibration_shards, calibration_reference, device, args.batch_size, args.stability_tolerance_nll
    )
    register(baseline, base_calibration, "uniform_w8_reference")

    if args.skip_w8_marginals:
        if args.run_mode != "confirmation":
            raise ValueError("W8 marginal sweeps may only be skipped in confirmation mode")
        single = {}
        downgrade_rank = list(units)
        upgrade_rank = list(reversed(units))
        print("skipping repeated W8-centered marginal sweep for the gated confirmation; using deterministic structural move proposals", flush=True)
    else:
        single = single_unit_measurements(
            model, state, units, baseline, calibration_shards, calibration_reference, device, args.batch_size, args.stability_tolerance_nll
        )

    # Do not select one projection assignment using a proxy.  Execute every
    # one of the 3^5 assignments, recording exact storage even when it is far
    # below or above the W8 budget.  The feasible/near-budget counts are
    # reported separately for the requested budgeted sanity check.
    projection_all = projection_profiles(units, fixed_bits)
    projection_records = []
    if args.skip_projection_enumeration:
        if args.run_mode != "confirmation":
            raise ValueError("projection enumeration may only be skipped in confirmation mode")
        print("skipping 1B-only projection enumeration for the gated 8B confirmation", flush=True)
    else:
        for index, profile in enumerate(projection_all, start=1):
            print(f"projection exhaustive [{index}/{len(projection_all)}] {profile.name}", flush=True)
            row = evaluation_registry.get(profile.signature(units))
            if row is None:
                row = register(
                    profile,
                    evaluate_calibration(model, state, profile, calibration_shards, calibration_reference, device, args.batch_size, args.stability_tolerance_nll),
                    "projection_exhaustive_3^5",
                )
            else:
                if "projection_exhaustive_3^5" not in row["origins"]:
                    row["origins"].append("projection_exhaustive_3^5")
            projection_records.append(row)

    target_storage = int(baseline.storage(units)["total_bits"])
    # One percent is a declared reporting band; all 243 are nevertheless
    # executed, so no near-budget assignment is hidden by a prefilter.
    projection_tolerance = max(1, int(target_storage * 0.01))
    projection_costs = [int(row["profile"]["storage"]["total_bits"]) for row in projection_records]
    feasible_projection = [row for row in projection_records if int(row["profile"]["storage"]["total_bits"]) <= target_storage]
    near_projection = [row for row in projection_records if abs(int(row["profile"]["storage"]["total_bits"]) - target_storage) <= projection_tolerance]

    if not args.skip_w8_marginals:
        downgrade_rank, upgrade_rank = marginal_proposal_ranks(single, units)
    search_records: list[dict[str, Any]] = []
    search_history = []
    search_new_evaluations = 0
    beam = [evaluation_registry[baseline.signature(units)]]
    for round_index in range(args.search_rounds):
        anchors = list(beam)
        generated: list[tuple[Profile, str, str]] = []
        for anchor_row in anchors:
            anchor_profile = Profile(
                anchor_row["profile"]["name"],
                anchor_row["profile"]["kind"],
                {key: int(value) for key, value in anchor_row["profile"]["unit_bits"].items()},
                target_bits=8,
                fixed_fp16_bits=fixed_bits,
            )
            for profile, move in neighbor_profiles(
                anchor_profile, units, target_storage, fixed_bits, downgrade_rank, upgrade_rank, args.proposal_width
            ):
                generated.append((profile, move, anchor_row["profile"]["signature"]))
        # Dedup before consuming the explicit search budget.  Proposal ordering
        # is deterministic; final ranking below uses only actual shard runs.
        dedup: dict[str, tuple[Profile, str, str]] = {}
        for item in generated:
            dedup.setdefault(item[0].signature(units), item)
        round_evaluated: list[dict[str, Any]] = []
        for signature, (profile, move, anchor_signature) in sorted(dedup.items()):
            row = evaluation_registry.get(signature)
            if row is None:
                if search_new_evaluations >= args.search_evaluation_budget:
                    break
                print(f"interaction search round={round_index} eval={search_new_evaluations + 1}/{args.search_evaluation_budget} {move}", flush=True)
                row = register(
                    profile,
                    evaluate_calibration(model, state, profile, calibration_shards, calibration_reference, device, args.batch_size, args.stability_tolerance_nll),
                    f"interaction_search_round_{round_index}",
                )
                search_new_evaluations += 1
            elif f"interaction_search_round_{round_index}" not in row["origins"]:
                row["origins"].append(f"interaction_search_round_{round_index}")
            if row not in search_records:
                search_records.append(row)
            round_evaluated.append(row)
        # Keep W8 as an anchor and retain the best actual combined candidates
        # as exploration anchors.  A noisy candidate can therefore generate a
        # next-round neighborhood, but it cannot change final held-out claims.
        alternatives = [row for row in search_records if row["profile"]["signature"] != baseline.signature(units)]
        beam = [evaluation_registry[baseline.signature(units)]] + sorted(alternatives, key=profile_rank)[: args.beam_width]
        search_history.append({
            "round": round_index,
            "anchor_signatures": [row["profile"]["signature"] for row in anchors],
            "generated_unique_count": len(dedup),
            "evaluated_signatures": [row["profile"]["signature"] for row in round_evaluated],
            "accepted_anchor_signatures": [row["profile"]["signature"] for row in beam],
            "ranking": "actual combined calibration stable_rank_score, then worst shard, then shard std, then signature",
        })
        if search_new_evaluations >= args.search_evaluation_budget:
            break

    pool = [row for row in projection_records + search_records if int(row["profile"]["storage"]["total_bits"]) <= target_storage]
    # Registry rows may occur in both lists; profile signatures are the unit of
    # deduplication for the final frontier.
    pool_by_signature = {row["profile"]["signature"]: row for row in pool}
    frontier = actual_calibration_frontier(list(pool_by_signature.values()), target_storage)
    if not any(row["profile"]["signature"] == baseline.signature(units) for row in frontier):
        frontier.insert(0, evaluation_registry[baseline.signature(units)])

    # Held-out windows are not loaded or evaluated until after search and are
    # from the disjoint Wikitext validation split.
    heldout_shards, heldout_data = dispersed_windows(
        tokenizer, "validation", args.heldout_shards, args.heldout_samples_per_shard, args.seq_len, args.seed + 202, args.cache_dir
    )
    heldout_records = [record for shard in heldout_shards for record in shard]
    heldout_reference = build_reference_cache(model, state, baseline, heldout_shards, device, args.batch_size)
    frontier_results = []
    for index, row in enumerate(frontier):
        profile = Profile(
            row["profile"]["name"],
            row["profile"]["kind"],
            {key: int(value) for key, value in row["profile"]["unit_bits"].items()},
            target_bits=8,
            fixed_fp16_bits=fixed_bits,
        )
        print(f"heldout frontier [{index + 1}/{len(frontier)}] {profile.name}", flush=True)
        heldout_result = evaluate_lm_against_reference(
            model, state, profile, heldout_records, [item for shard in heldout_reference for item in shard], device, args.batch_size
        )
        heldout_result = attach_heldout_evidence(heldout_result, args.bootstrap_iterations, args.seed + 7000 + index * 31)
        frontier_results.append({
            "profile": row["profile"],
            "calibration": row["calibration"],
            "origins": row["origins"],
            "heldout_vs_uniform_w8": heldout_result,
        })
    state.restore()

    baseline_signature = baseline.signature(units)
    gate_candidates = []
    for row in frontier_results:
        if row["profile"]["signature"] == baseline_signature:
            continue
        nll = row["heldout_vs_uniform_w8"]["paired_nll_difference_vs_uniform_w8"]
        stable = bool(row["calibration"]["stability"].get("stable_improvement", False))
        positive_signal = float(nll["mean"]) < 0.0 and float(nll["ci95_high"]) < 0.0
        gate_candidates.append({
            "profile": row["profile"],
            "storage_leq_uniform_w8": int(row["profile"]["storage"]["total_bits"]) <= target_storage,
            "heldout_paired_nll": nll,
            "positive_confidence_signal": positive_signal,
            "stable_calibration": stable,
            "eligible": bool(positive_signal and stable),
        })
    eligible = [candidate for candidate in gate_candidates if candidate["eligible"]]
    gate_passed = bool(eligible)
    if args.run_mode == "confirmation":
        gate_model = "Llama 3.1 8B"
        gate_decision = "CONFIRMED_8B_REOPEN_NATIVE_KERNEL_GATE" if gate_passed else "NO_GO_CLOSE_STRUCTURED_WEIGHT_PRECISION"
        confirmation = {
            "opened": True,
            "model": "Llama 3.1 8B",
            "reason": "1B gate passed; this is the gated 8B confirmation run",
        }
        next_direction = "reopen native-kernel feasibility; keep scheduler, KV quantization, and routing work paused" if gate_passed else "move to KV-cache precision/serving behavior; close weight-allocation search"
    else:
        gate_model = "Llama 3.2 1B"
        gate_decision = "OPEN_8B_CONFIRMATION" if gate_passed else "NO_GO_CLOSE_STRUCTURED_WEIGHT_PRECISION"
        confirmation = {
            "opened": gate_passed,
            "model": "Llama 3.1 8B",
            "reason": "opened only after the 1B gate" if gate_passed else "1B did not show a stable positive held-out Pareto signal; no 8B run was opened",
        }
        next_direction = "run the gated Llama 3.1 8B confirmation before reopening native kernels" if gate_passed else "move to KV-cache precision/serving behavior; keep weight-allocation search and native kernels paused"
    gate = {
        "run_mode": args.run_mode,
        "model": gate_model,
        "uniform_w8_storage_bits": target_storage,
        "criterion": "non-uniform profile at equal-or-lower modeled storage, paired held-out NLL CI upper bound < 0, and stable improvement on at least two of three calibration shards with declared tolerance",
        "candidate_checks": gate_candidates,
        "eligible_profiles": [candidate["profile"] for candidate in eligible],
        "gate_passed": gate_passed,
        "one_b_gate_passed": gate_passed if args.run_mode == "exploration" else None,
        "decision": gate_decision,
        "confirmation": confirmation,
        "next_direction": next_direction,
    }

    all_candidate_evaluations = list(evaluation_registry.values())
    result = {
        "schema_version": 3,
        "run_mode": args.run_mode,
        "experiment": "interaction_aware_w8_centered_structured_weight_precision",
        "created_unix": time.time(),
        "elapsed_seconds": time.time() - started,
        "scope": {
            "weight_projection_quantization": True,
            "kv_cache_quantization": False,
            "native_low_bit_kernel": False,
            "scheduler_or_router": False,
            "query_specific_oracle": False,
            "ffn_unit": "gate_proj + up_proj + down_proj as one layer-level unit",
        },
        "source_provenance": {
            "required_branch": "structured-precision-evidence",
            "required_base_commit": "b5fc47744d6c4aa1a46206439cdb6b70c29a88ad",
            **git_source_provenance(),
            "gate_artifact": str(gate_artifact) if args.run_mode == "confirmation" else None,
            "gate_artifact_sha256": sha256(gate_artifact) if args.run_mode == "confirmation" else None,
            "gate_artifact_created_unix": gate_data.get("created_unix") if gate_data is not None else None,
            "gate_artifact_decision": gate_data.get("final_gate", {}).get("decision") if gate_data is not None else None,
            "swiftllm_upstream_commit": git_revision(SWIFT_ROOT),
            "swiftllm_research_diff_sha256": sha256(ROOT / "references/swiftllm-research.diff"),
            "requirements_lock_sha256": sha256(ROOT / "requirements-lock.txt"),
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
            "datasets_version": __import__("datasets").__version__,
        },
        "model": {
            "path": str(model_path),
            "config": json.loads((model_path / "config.json").read_text()),
            "files": model_file_records(model_path),
        },
        "data": {
            "calibration": calibration_data,
            "heldout": heldout_data,
            "search_uses_sample_ids": calibration_data["sample_ids"],
            "validation_loaded_after_search": True,
            "calibration_heldout_ids_disjoint": set(calibration_data["sample_ids"]).isdisjoint(set(heldout_data["sample_ids"])),
            "limitation": "Wikitext is a language-quality proxy; this phase does not use a small HellaSwag subset as a decision criterion.",
        },
        "storage_accounting": {
            "manual_cases": manual_storage_cases(),
            "fixed_fp16_bits": fixed_bits,
            "uniform_profiles": {str(bits): uniform_profile(units, bits, fixed_fp16_bits=fixed_bits).storage(units) for bits in BITS},
            "unit_shapes": [
                {
                    "key": unit.key,
                    "name": unit.name,
                    "layer": unit.layer,
                    "matrices": [matrix.__dict__ | {"numel": matrix.numel, "padded_in_features": matrix.padded_in_features, "scale_count": matrix.scale_count} for matrix in unit.matrices],
                }
                for unit in units
            ],
        },
        "w8_centered_marginals": {
            "skipped_for_confirmation": args.skip_w8_marginals,
            "background_profile": baseline.as_record(units),
            "definition": "one unit changed from W8 while every other Q/K/V/O/FFN unit remains W8; actual combined calibration execution for W4 and FP16",
            "records": single,
            "all_units_measured": len(single) == len(units),
            "measured_targets": [] if args.skip_w8_marginals else [4, 16],
        },
        "projection_enumeration": {
            "description": "exhaustive projection-only enumeration across all 3^5 = 243 assignments in exploration; skipped only for the gated 8B confirmation",
            "optimizer_description": "exhaustive 3^5 enumeration (243 assignments)",
            "total_assignments": len(projection_all),
            "executed_assignments": len(projection_records),
            "skipped_for_confirmation": args.skip_projection_enumeration,
            "unique_signatures": len({row["profile"]["signature"] for row in projection_records}),
            "budget_target_bits": target_storage,
            "budget_tolerance_bits_for_close_bracket": projection_tolerance,
            "feasible_count": len(feasible_projection),
            "closely_bracketed_count": len(near_projection),
            "all_feasible_and_near_budget_executed": not args.skip_projection_enumeration,
            "assignment_cost_min_bits": min(projection_costs) if projection_costs else None,
            "assignment_cost_max_bits": max(projection_costs) if projection_costs else None,
            "evaluations": projection_records,
        },
        "interaction_aware_search": {
            "starting_profile": baseline.as_record(units),
            "candidate_generation": "bounded deterministic beam/coordinate search from W8; includes explicit feasible 8->4 plus compensating 8->16 moves and recomputes moves around accepted profiles",
            "marginals_only_propose_moves": True,
            "final_ranking_uses_actual_combined_calibration": True,
            "ranking": "stable_rank_score from actual shard NLL deltas, then worst shard and shard dispersion; deterministic signature tie-break",
            "evaluation_budget": args.search_evaluation_budget,
            "new_unique_evaluations": search_new_evaluations,
            "rounds": search_history,
            "search_records": search_records,
            "final_calibration_frontier": [row["profile"] for row in frontier],
        },
        "combined_candidate_evaluations": all_candidate_evaluations,
        "heldout_frontier": frontier_results,
        "final_gate": gate,
        "verification": {
            "reference_profile": "uniform_w8",
            "calibration_reference_is_combined_uniform_w8": True,
            "heldout_reference_is_combined_uniform_w8": True,
            "heldout_not_used_for_search": True,
            "all_candidates_have_actual_combined_calibration": True,
            "exact_integer_storage_bits_and_bytes": True,
            "bootstrap_iterations": args.bootstrap_iterations,
            "search_budget_enforced": search_new_evaluations <= args.search_evaluation_budget,
            "no_8b_run_before_gate": True,
            "final_gate_decision_recorded": True,
            "run_mode": args.run_mode,
            "gate_artifact_required_for_confirmation": str(gate_artifact),
        },
    }
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {output_path}")
    print(json.dumps({"decision": gate["decision"], "frontier_profiles": len(frontier), "projection_assignments": len(projection_records)}, indent=2))


if __name__ == "__main__":
    main()
