#!/usr/bin/env python3
"""Aggregate only the frozen, auditable v2 capacity records.

Historical screening/experimental files remain in the tree, but are not
silently mixed with the final primary comparison.  The selected records are
explicitly scoped by directory and the raw per-cell artifacts remain the
source of truth.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/kv-capacity-v2"
DECISION_TIMING_DIRS = {"primary-current", "primary-expandable", "confirmation-current", "confirmation-final", "confirmation-postfix"}
EQUAL_BATCH_DIR = "equal-batch-final"
EQUAL_BATCH_RERUN_DIR = "equal-batch-rerun"
MEMORY_DIRS = {"memory-current", "memory-final", "memory-postfix"}
QUALITY_DIR = "quality-final"


def bootstrap(values: list[float], seed: int, n: int = 4000) -> dict[str, object]:
    if not values:
        return {"count": 0}
    values_np = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = values_np[rng.integers(0, len(values_np), size=(n, len(values_np)))].mean(axis=1)
    return {
        "count": int(len(values_np)),
        "mean": float(values_np.mean()),
        "median": float(np.median(values_np)),
        "p95": float(np.quantile(values_np, 0.95)),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
        "bootstrap_iterations": n,
    }


def is_selected_timing(path: Path, data: dict) -> bool:
    if data.get("schema") != "kv-capacity-cell-v2" or data.get("instrumentation") != "timing":
        return False
    if path.parent.name in {EQUAL_BATCH_DIR, EQUAL_BATCH_RERUN_DIR}:
        return True
    if path.parent.name not in DECISION_TIMING_DIRS:
        return False
    if path.parent.name == "primary-current":
        # Retained as a pre-common-allocator diagnostic; the selected baseline
        # uses the same expandable-segment setting as accepted candidates.
        return False
    if path.parent.name == "confirmation-current":
        # The first *_current attempts were made without the allocator setting
        # used by the accepted KIVI runs, and the Swift files predate the
        # Llama-3.1 loader correction.  Keep them on disk but exclude them.
        if data.get("method") == "swiftllm_fp16":
            return False
        return "expandable" in path.name or "boundary" in path.name
    return True


def load_timing() -> list[tuple[Path, dict, str]]:
    rows = []
    for path in sorted(OUT.rglob("*.json")):
        if path.name == "summary.json":
            continue
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if is_selected_timing(path, data):
            scope = "equal_batch" if path.parent.name in {EQUAL_BATCH_DIR, EQUAL_BATCH_RERUN_DIR} else "capacity"
            rows.append((path, data, scope))
    return rows


def load_memory() -> list[tuple[Path, dict]]:
    rows = []
    for path in sorted(OUT.rglob("*.json")):
        if path.parent.name not in MEMORY_DIRS:
            continue
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if data.get("schema") == "kv-capacity-cell-v2" and data.get("instrumentation") == "memory":
            rows.append((path, data))
    return rows


def cell_record(path: Path, data: dict, scope: str) -> dict:
    row = {
        "artifact": str(path.relative_to(ROOT)),
        "scope": scope,
        "method": data.get("method"),
        "model_family": data.get("model_family"),
        "context_tokens": data.get("context_tokens"),
        "decode_tokens": data.get("decode_tokens"),
        "batch_size": data.get("batch_size"),
        "instrumentation": data.get("instrumentation"),
        "status": data.get("status"),
        "feasible": bool(data.get("feasible", False)),
        "source_sha256": data.get("provenance", {}).get("source_sha256"),
        "visible_devices": data.get("budget", {}).get("visible_devices"),
    }
    if data.get("status") != "completed":
        row["failure"] = data.get("error", {})
        return row
    repetitions = data["run"]["repetitions"]
    decode_per_token = [float(rep["decode_ms"]) / int(data["decode_tokens"]) for rep in repetitions]
    prefill = [float(rep["prefill_ms"]) for rep in repetitions]
    peaks = [max(int(rep["peak"]["peak_allocated_bytes"]), int(rep["peak"]["peak_reserved_bytes"])) for rep in repetitions]
    row.update({
        "repetitions": len(repetitions),
        "decode_per_token_ms": bootstrap(decode_per_token, 1000 + int(data["batch_size"])),
        "prefill_ms": bootstrap(prefill, 2000 + int(data["batch_size"])),
        "peak_budget_bytes_max": max(peaks),
        "declared_budget_bytes": int(data["budget"]["declared_budget_bytes"]),
        "total_device_memory_bytes": int(data["budget"]["total_device_memory_bytes"]),
        "completed_decode_steps": data.get("completed_decode_steps"),
        "timing_boundary": data["run"].get("timing_boundary"),
    })
    row["aggregate_output_tokens_per_second_at_median"] = int(data["batch_size"]) * 1000.0 / row["decode_per_token_ms"]["median"]
    return row


def capacity_rows(records: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = {}
    for row in records:
        if row["scope"] == "capacity" and row["status"] == "completed" and row["feasible"]:
            key = (row["model_family"], row["context_tokens"], row["decode_tokens"], row["method"])
            grouped.setdefault(key, []).append(row)
    result = []
    for key, rows in sorted(grouped.items()):
        # All accepted final rows have the same repeated-run policy.  If a
        # future rerun adds evidence, prefer the largest completed batch, then
        # the row with more repeats.
        best = max(rows, key=lambda x: (int(x["batch_size"]), int(x.get("repetitions", 0))))
        result.append({
            "model_family": key[0], "context_tokens": key[1], "decode_tokens": key[2], "method": key[3],
            "largest_feasible_batch_observed": int(best["batch_size"]), "selected_artifact": best["artifact"],
            "repetitions": best["repetitions"], "peak_budget_bytes_max": best["peak_budget_bytes_max"], "declared_budget_bytes": best["declared_budget_bytes"], "total_device_memory_bytes": best["total_device_memory_bytes"], "peak_budget_fraction": best["peak_budget_bytes_max"] / best["declared_budget_bytes"],
            "decode_per_token_ms": best["decode_per_token_ms"], "prefill_ms": best["prefill_ms"],
            "aggregate_output_tokens_per_second": best["aggregate_output_tokens_per_second_at_median"],
        })
    return result


def compare_capacity(capacities: list[dict]) -> list[dict]:
    grouped: dict[tuple, dict[str, dict]] = {}
    for row in capacities:
        grouped.setdefault((row["model_family"], row["context_tokens"], row["decode_tokens"]), {})[row["method"]] = row
    comparisons = []
    for workload, methods in sorted(grouped.items()):
        baseline = methods.get("hf_fp16")
        if not baseline:
            continue
        for candidate_name in ("kivi4", "kivi2", "swiftllm_fp16"):
            candidate = methods.get(candidate_name)
            if not candidate:
                continue
            med_ratio = candidate["decode_per_token_ms"]["median"] / baseline["decode_per_token_ms"]["median"]
            p95_ratio = candidate["decode_per_token_ms"]["p95"] / baseline["decode_per_token_ms"]["p95"]
            prefill_ratio = candidate["prefill_ms"]["median"] / baseline["prefill_ms"]["median"]
            comparisons.append({
                "workload": {"model_family": workload[0], "context_tokens": workload[1], "decode_tokens": workload[2]},
                "mixed_method": candidate_name, "fp16_method": "hf_fp16",
                "candidate_batch": candidate["largest_feasible_batch_observed"], "fp16_batch": baseline["largest_feasible_batch_observed"],
                "batch_difference": candidate["largest_feasible_batch_observed"] - baseline["largest_feasible_batch_observed"],
                "batch_gain": candidate["largest_feasible_batch_observed"] / baseline["largest_feasible_batch_observed"],
                "decode_latency_ratio_median": med_ratio, "decode_latency_ratio_p95": p95_ratio, "prefill_latency_ratio_median": prefill_ratio,
                "latency_gate_median_pass": med_ratio <= 1.25, "latency_gate_p95_pass": p95_ratio <= 1.25,
                "prefill_gate_pass": prefill_ratio <= 1.50,
                "capacity_gate_pass": candidate["largest_feasible_batch_observed"] > baseline["largest_feasible_batch_observed"],
                "throughput_ratio_at_each_method_capacity": candidate["aggregate_output_tokens_per_second"] / baseline["aggregate_output_tokens_per_second"],
                "throughput_win": candidate["aggregate_output_tokens_per_second"] > baseline["aggregate_output_tokens_per_second"],
                "declared_latency_quality_gates_pass": med_ratio <= 1.25 and p95_ratio <= 1.25 and prefill_ratio <= 1.50,
            })
    return comparisons


def compare_equal_batch(records: list[dict]) -> list[dict]:
    rows = [r for r in records if r["scope"] == "equal_batch" and r["status"] == "completed" and r["feasible"]]
    grouped: dict[tuple, dict[str, dict]] = {}
    for row in rows:
        grouped.setdefault((row["model_family"], row["context_tokens"], row["decode_tokens"], row["batch_size"]), {})[row["method"]] = row
    result = []
    for key, methods in sorted(grouped.items()):
        baseline = methods.get("hf_fp16")
        if not baseline:
            continue
        for candidate_name in ("kivi4", "kivi2", "swiftllm_fp16"):
            candidate = methods.get(candidate_name)
            if not candidate:
                continue
            result.append({
                "workload": {"model_family": key[0], "context_tokens": key[1], "decode_tokens": key[2], "batch_size": key[3]},
                "candidate_method": candidate_name, "baseline_method": "hf_fp16", "candidate_prefill_ms": candidate["prefill_ms"], "baseline_prefill_ms": baseline["prefill_ms"],
                "decode_latency_ratio_median": candidate["decode_per_token_ms"]["median"] / baseline["decode_per_token_ms"]["median"],
                "decode_latency_ratio_p95": candidate["decode_per_token_ms"]["p95"] / baseline["decode_per_token_ms"]["p95"],
                "prefill_latency_ratio_median": candidate["prefill_ms"]["median"] / baseline["prefill_ms"]["median"],
                "candidate_decode_per_token_ms": candidate["decode_per_token_ms"], "baseline_decode_per_token_ms": baseline["decode_per_token_ms"],
                "throughput_ratio": candidate["aggregate_output_tokens_per_second_at_median"] / baseline["aggregate_output_tokens_per_second_at_median"],
                "decode_latency_gate_pass": candidate["decode_per_token_ms"]["median"] <= 1.25 * baseline["decode_per_token_ms"]["median"] and candidate["decode_per_token_ms"]["p95"] <= 1.25 * baseline["decode_per_token_ms"]["p95"],
                "prefill_gate_pass": candidate["prefill_ms"]["median"] <= 1.50 * baseline["prefill_ms"]["median"],
                "same_batch": True,
            })
    return result


def load_additional_requests() -> list[dict]:
    result = []
    for path in sorted((OUT / "additional-request").glob("*.json")):
        data = json.loads(path.read_text())
        result.append({
            "artifact": str(path.relative_to(ROOT)), "status": data.get("status"), "model_family": data.get("model_family"),
            "context_tokens": data.get("context_tokens"), "base_batch": data.get("base_batch"), "exact_pool_additional_status": data.get("exact_pool", {}).get("additional_request", {}).get("status"),
            "one_extra_pool_additional_status": data.get("one_extra_pool", {}).get("additional_request", {}).get("status"),
            "exact_pool_peak": data.get("exact_pool", {}).get("after_base", {}).get("allocator"), "one_extra_pool_peak": data.get("one_extra_pool", {}).get("additional_request", {}).get("allocator"),
        })
    return result


def load_quality() -> list[dict]:
    result = []
    for path in sorted((OUT / QUALITY_DIR).glob("*.json")):
        data = json.loads(path.read_text())
        if data.get("status") != "completed":
            result.append({"artifact": str(path.relative_to(ROOT)), "status": data.get("status"), "error": data.get("error")})
            continue
        long = data["long_context_information_use"]
        free = data["free_running_generation"]
        result.append({
            "artifact": str(path.relative_to(ROOT)), "status": "completed", "model_family": data["model_family"], "method": data["method"], "context_tokens": data["sampling"].get("long_context_window_tokens"),
            "source_sha256": data.get("provenance", {}).get("source_sha256"), "reference": data["reference"],
            "long_context_nll_delta_bootstrap95": long["nll_delta_bootstrap95"], "long_context_top1_match_mean": long["top1_match_mean"],
            "long_context_kl_mean": long.get("kl_mean"), "long_context_logit_mse_mean": long.get("logit_mse_mean"),
            "long_context_samples": len(long["per_sample"]), "free_running_samples": len(free["per_sample"]),
            "free_running_prefix_token_agreement_mean": free["prefix_token_agreement_mean"],
            "free_running_termination_records": free.get("termination_records", False), "free_running_pathology_records": free.get("pathology_records", False),
            "artificial_prompt_repetition": data["sampling"].get("artificial_prompt_repetition"),
        })
    return result


def main() -> None:
    timing = load_timing()
    records = [cell_record(path, data, scope) for path, data, scope in timing]
    memories = load_memory()
    capacities = capacity_rows(records)
    comparisons = compare_capacity(capacities)
    equal_batch = compare_equal_batch(records)
    quality = load_quality()
    additional_requests = load_additional_requests()
    quality_by_workload = {(q.get("model_family"), q.get("method"), q.get("context_tokens")): q for q in quality if q.get("status") == "completed"}
    sensitivity = {"memory_margin_capacity_counts": {}, "latency_capacity_counts": {}, "latency_equal_batch_counts": {}, "nll_quality_counts": {}}
    for margin in (0.05, 0.10, 0.15):
        key = f"{int(margin * 100)}%_margin"
        sensitivity["memory_margin_capacity_counts"][key] = {}
        for family in ("llama31_8b", "llama32_1b"):
            for method in ("hf_fp16", "kivi2", "kivi4", "swiftllm_fp16"):
                selected = [x for x in capacities if x["model_family"] == family and x["method"] == method]
                sensitivity["memory_margin_capacity_counts"][key][f"{family}:{method}"] = {"feasible_workloads": sum(x["peak_budget_bytes_max"] <= x["total_device_memory_bytes"] * (1.0 - margin) for x in selected), "workloads": len(selected)}
    for limit in (1.10, 1.25, 1.50):
        key = f"{limit:.2f}x"
        sensitivity["latency_capacity_counts"][key] = {}
        sensitivity["latency_equal_batch_counts"][key] = {}
        for method in ("kivi2", "kivi4"):
            cap = [x for x in comparisons if x["workload"]["model_family"] == "llama31_8b" and x["mixed_method"] == method]
            eq = [x for x in equal_batch if x["workload"]["model_family"] == "llama31_8b" and x["candidate_method"] == method]
            sensitivity["latency_capacity_counts"][key][method] = sum(x["decode_latency_ratio_median"] <= limit and x["decode_latency_ratio_p95"] <= limit and x["prefill_latency_ratio_median"] <= 1.50 for x in cap)
            sensitivity["latency_equal_batch_counts"][key][method] = sum(x["decode_latency_ratio_median"] <= limit and x["decode_latency_ratio_p95"] <= limit and x["prefill_gate_pass"] for x in eq)
    for limit in (0.01, 0.02, 0.05):
        key = f"+{limit:.2f}_nats"
        sensitivity["nll_quality_counts"][key] = {}
        for method in ("kivi2", "kivi4"):
            rows = [x for x in quality if x.get("model_family") == "llama31_8b" and x.get("method") == method and x.get("status") == "completed"]
            sensitivity["nll_quality_counts"][key][method] = {"pass": sum(x["long_context_nll_delta_bootstrap95"]["ci95_high"] <= limit for x in rows), "tested": len(rows)}
    qualifying_equal_batch = []
    for row in equal_batch:
        workload = row["workload"]
        q = quality_by_workload.get((workload["model_family"], row["candidate_method"], workload["context_tokens"]))
        row["quality_nll_gate_pass"] = bool(q and q["long_context_nll_delta_bootstrap95"]["ci95_high"] <= 0.02)
        row["qualifies_declared_gate"] = bool(row["candidate_method"] in {"kivi2", "kivi4"} and row["quality_nll_gate_pass"] and row["decode_latency_gate_pass"] and row["prefill_gate_pass"] and row["throughput_ratio"] > 1.0)
        if row["qualifies_declared_gate"]:
            qualifying_equal_batch.append(row)
    plan = ROOT / "docs/kv-capacity-plan-v2.md"
    summary = {
        "schema": "kv-capacity-summary-v2",
        "plan_sha256": hashlib.sha256(plan.read_bytes()).hexdigest(),
        "accepted_timing_policy": {"directories": sorted(DECISION_TIMING_DIRS | {EQUAL_BATCH_DIR, EQUAL_BATCH_RERUN_DIR}), "excluded_historical_files_retained": True, "primary_baseline": "corrected Transformers hf_fp16", "secondary_baseline": "post-fix SwiftLLM dense FP16"},
        "attempted_cell_count": len(records), "completed_cell_count": sum(r["status"] == "completed" for r in records),
        "feasible_cell_count": sum(r["status"] == "completed" and r["feasible"] for r in records),
        "failure_cell_count": sum(r["status"] != "completed" for r in records),
        "declared_budget_bytes": sorted({r.get("declared_budget_bytes") for r in records if r.get("declared_budget_bytes")}),
        "records": records, "largest_feasible_capacity": capacities, "same_budget_comparisons": comparisons, "equal_batch_comparisons": equal_batch, "additional_request_allocation": additional_requests,
        "memory_instrumented_artifacts": [str(path.relative_to(ROOT)) for path, _ in memories], "memory_trace_count": len(memories), "quality": quality,
        "quality_policy": {"nll_upper_limit_nats_per_token": 0.02, "long_context_minimum_samples": 8, "free_running_minimum_samples": 8, "decode_latency_multiplier": 1.25, "prefill_latency_multiplier": 1.50}, "sensitivity_results": sensitivity, "qualifying_equal_batch_gains": qualifying_equal_batch, "verdict": "GO" if qualifying_equal_batch else "NO_GO",
        "sensitivity": {"memory_margin_fractions": [0.05, 0.10, 0.15], "latency_multipliers": [1.10, 1.25, 1.50], "nll_limits": [0.01, 0.02, 0.05]},
        "notes": {"latency_for_failures": "not assigned", "capacity_requires_complete_decode_and_budget_check": True, "workspace_attribution": "allocator peak is measured; allocator does not expose per-allocation workspace owner", "additional_request_allocation": "fresh full-model batch cells are measured; no online arrival/admission trace was implemented"},
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"output": str(OUT / "summary.json"), "attempted": len(records), "completed": summary["completed_cell_count"], "feasible": summary["feasible_cell_count"], "failures": summary["failure_cell_count"], "capacities": len(capacities), "equal_batch": len(equal_batch), "quality": len(quality), "additional_requests": len(additional_requests)}))


if __name__ == "__main__":
    main()
