#!/usr/bin/env python3
"""Aggregate v2 cell artifacts without inventing values for failures."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/kv-capacity-v2"


def bootstrap(values: list[float], seed: int, n: int = 4000) -> dict[str, object]:
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {"count": 0}
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(n, len(values)))].mean(axis=1)
    return {"count": int(len(values)), "mean": float(values.mean()), "median": float(np.median(values)), "p95": float(np.quantile(values, .95)), "ci95_low": float(np.quantile(means, .025)), "ci95_high": float(np.quantile(means, .975)), "bootstrap_iterations": n}


def load_cells() -> list[tuple[Path, dict]]:
    rows = []
    for path in sorted(OUT.rglob("*.json")):
        if path.name == "summary.json" or "quality" in path.parts or any(part.startswith("memory") for part in path.parts):
            continue
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if data.get("schema") == "kv-capacity-cell-v2" and data.get("instrumentation") == "timing":
            # Keep development and contaminated-device attempts on disk, but
            # do not let them define the decision summary.
            if path.parent.name == "screen" or (data.get("model_family") == "llama31_8b" and path.parent.name == "final" and "freegpu" not in path.name):
                continue
            rows.append((path, data))
    return rows


def load_memory() -> list[tuple[Path, dict]]:
    rows = []
    for path in sorted(OUT.rglob("*.json")):
        if not any(part.startswith("memory") for part in path.parts):
            continue
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if data.get("schema") == "kv-capacity-cell-v2":
            rows.append((path, data))
    return rows


def main() -> None:
    cells = load_cells()
    records = []
    for path, data in cells:
        row = {"artifact": str(path.relative_to(ROOT)), "method": data.get("method"), "model_family": data.get("model_family"), "context_tokens": data.get("context_tokens"), "decode_tokens": data.get("decode_tokens"), "batch_size": data.get("batch_size"), "instrumentation": data.get("instrumentation"), "status": data.get("status"), "feasible": data.get("feasible", False), "cold_start_recorded": bool(data.get("run", {}).get("cold_start"))}
        if data.get("status") == "completed":
            reps = data["run"]["repetitions"]
            per_token = [float(rep["decode_ms"]) / int(data["decode_tokens"]) for rep in reps]
            prefill = [float(rep["prefill_ms"]) for rep in reps]
            peaks = [max(int(rep["peak"]["peak_allocated_bytes"]), int(rep["peak"]["peak_reserved_bytes"])) for rep in reps]
            row.update({"decode_per_token_ms": bootstrap(per_token, 1000 + int(data["batch_size"])), "prefill_ms": bootstrap(prefill, 2000 + int(data["batch_size"])), "peak_budget_bytes_max": max(peaks), "declared_budget_bytes": int(data["budget"]["declared_budget_bytes"]), "completed_decode_steps": data.get("completed_decode_steps")})
            row["aggregate_output_tokens_per_second_at_median"] = int(data["batch_size"]) * 1000.0 / row["decode_per_token_ms"]["median"]
        else:
            row["failure"] = data.get("error", {})
            if data.get("budget") or data.get("environment"):
                source = data.get("budget", data.get("environment", {}))
                row["declared_budget_bytes"] = source.get("declared_budget_bytes")
        records.append(row)

    grouped: dict[tuple, list[dict]] = {}
    for row in records:
        if row["status"] == "completed" and row["feasible"]:
            grouped.setdefault((row["model_family"], row["context_tokens"], row["decode_tokens"], row["method"]), []).append(row)
    capacities = []
    def evidence_priority(row: dict) -> int:
        artifact = row["artifact"]
        if "confirmation-cold" in artifact:
            return 6
        if "confirmation-extended-confirm" in artifact:
            return 5
        if "confirmation-rerun" in artifact:
            return 4
        if "/confirmation/" in artifact:
            return 3
        if "confirmation-extended" in artifact:
            return 2
        return 1
    for (family, context, decode, method), rows in sorted(grouped.items()):
        best = max(rows, key=lambda x: (int(x["batch_size"]), evidence_priority(x)))
        capacities.append({"model_family": family, "context_tokens": context, "decode_tokens": decode, "method": method, "largest_feasible_batch_observed": best["batch_size"], "selected_artifact": best["artifact"], "peak_budget_fraction": best["peak_budget_bytes_max"] / best["declared_budget_bytes"], "decode_per_token_ms": best["decode_per_token_ms"], "aggregate_output_tokens_per_second": best["aggregate_output_tokens_per_second_at_median"]})

    by_workload = {}
    for row in capacities:
        key = (row["model_family"], row["context_tokens"], row["decode_tokens"])
        by_workload.setdefault(key, {})[row["method"]] = row
    total_device_bytes = BUDGET = 22766439628 / 0.90
    sensitivity = []
    for fraction in (0.85, 0.90, 0.95):
        limit = total_device_bytes * fraction
        grouped_sensitivity: dict[tuple, list[dict]] = {}
        for row in records:
            if row["status"] == "completed" and row["feasible"] and row.get("peak_budget_bytes_max", math.inf) <= limit:
                grouped_sensitivity.setdefault((row["model_family"], row["context_tokens"], row["decode_tokens"], row["method"]), []).append(row)
        for key, rows in sorted(grouped_sensitivity.items()):
            selected = max(rows, key=lambda x: int(x["batch_size"]))
            sensitivity.append({"margin_fraction": 1.0 - fraction, "budget_fraction": fraction, "workload": {"model_family": key[0], "context_tokens": key[1], "decode_tokens": key[2]}, "method": key[3], "largest_feasible_batch_observed": selected["batch_size"], "artifact": selected["artifact"]})

    comparisons = []
    for key, methods in sorted(by_workload.items()):
        fp = methods.get("swiftllm_fp16")
        if not fp:
            continue
        for method in ("kivi4", "kivi2"):
            candidate = methods.get(method)
            if not candidate:
                continue
            fp_ms = fp["decode_per_token_ms"]["median"]
            cand_ms = candidate["decode_per_token_ms"]["median"]
            comparisons.append({"workload": {"model_family": key[0], "context_tokens": key[1], "decode_tokens": key[2]}, "mixed_method": method, "fp16_method": "swiftllm_fp16", "batch_gain": int(candidate["largest_feasible_batch_observed"]) / int(fp["largest_feasible_batch_observed"]), "batch_difference": int(candidate["largest_feasible_batch_observed"]) - int(fp["largest_feasible_batch_observed"]), "decode_latency_ratio": cand_ms / fp_ms, "throughput_ratio_at_each_method_capacity": candidate["aggregate_output_tokens_per_second"] / fp["aggregate_output_tokens_per_second"], "latency_limit_1_25_pass": cand_ms <= 1.25 * fp_ms, "throughput_win": candidate["aggregate_output_tokens_per_second"] > fp["aggregate_output_tokens_per_second"]})

    quality = []
    for path in sorted((OUT / "quality").glob("*.json")):
        data = json.loads(path.read_text())
        if data.get("status") == "blocked":
            quality.append({"artifact": str(path.relative_to(ROOT)), "status": "blocked", "error": data.get("error")})
            continue
        long = data["long_context_information_use"]
        free = data["free_running_generation"]
        quality.append({"artifact": str(path.relative_to(ROOT)), "status": "completed", "model_family": data["model_family"], "method": data["method"], "reference": data["reference"], "long_context_nll_delta_bootstrap95": long["nll_delta_bootstrap95"], "long_context_candidate_nll_mean": float(np.mean([x["candidate_nll"] for x in long["per_sample"]])), "long_context_top1_match_mean": long["top1_match_mean"], "free_running_prefix_token_agreement_mean": free["prefix_token_agreement_mean"], "free_running_samples": free["sample_count"]})

    plan = ROOT / "docs/kv-capacity-plan-v2.md"
    summary = {"schema": "kv-capacity-summary-v2", "plan_sha256": hashlib.sha256(plan.read_bytes()).hexdigest(), "attempted_cell_count": len(records), "completed_cell_count": sum(row["status"] == "completed" for row in records), "feasible_cell_count": sum(row["status"] == "completed" and row["feasible"] for row in records), "failure_cell_count": sum(row["status"] != "completed" for row in records), "cold_start_recorded_count": sum(row["cold_start_recorded"] for row in records), "declared_budget_bytes": sorted({row.get("declared_budget_bytes") for row in records if row.get("declared_budget_bytes")}), "records": records, "largest_feasible_capacity": capacities, "same_budget_comparisons": comparisons, "budget_sensitivity": sensitivity, "memory_instrumented_artifacts": [str(path.relative_to(ROOT)) for path, _ in load_memory()], "quality": quality, "notes": {"screening_failures_are_retained": True, "latency_for_failures": "not assigned", "capacity_requires_complete_decode_and_budget_check": True, "screen_root_development_artifacts_excluded": True}}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"output": str(OUT / 'summary.json'), "attempted": len(records), "completed": summary["completed_cell_count"], "feasible": summary["feasible_cell_count"], "failures": summary["failure_cell_count"]}))


if __name__ == "__main__":
    main()
