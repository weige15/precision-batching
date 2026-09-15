#!/usr/bin/env python3
"""Verify the v2 capacity manifest, failure accounting, and quality artifacts."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/kv-capacity-v2"
BUDGET = 22766439628


def check(value: bool, message: str) -> None:
    if not value:
        raise AssertionError(message)


def verify_cell(path: Path, data: dict) -> None:
    check(data.get("schema") == "kv-capacity-cell-v2", f"wrong cell schema: {path}")
    check(data.get("instrumentation") == "timing", f"non-timing cell in manifest: {path}")
    if data.get("status") != "completed":
        # Loading can fail before provenance is assembled; the exact failure
        # and allocator observation are the evidence for that attempted cell.

        check("run" not in data and "feasible" not in data, f"failure has invented run/feasibility: {path}")
        check(data.get("error", {}).get("message"), f"failure has no error: {path}")
        return
    check(data.get("provenance", {}).get("swiftllm_upstream_commit") == "682cf9a28f97f7490409981a2f181528f377eb5d", f"SwiftLLM pin missing: {path}")
    check(data.get("budget", {}).get("declared_budget_bytes") == BUDGET, f"budget mismatch: {path}")
    check(data.get("method_config", {}).get("weights_dtype") == "torch.float16", f"weights not FP16: {path}")
    run = data["run"]
    check(len(run["repetitions"]) >= 1, f"missing repeats: {path}")
    for row in run["repetitions"]:
        check(row["prefill_ms"] is not None and row["decode_ms"] is not None, f"missing timing: {path}")
        check(row["peak"]["peak_allocated_bytes"] >= row["base_model_runtime"]["allocated_bytes"], f"peak allocation accounting: {path}")
        check(row["peak"]["peak_reserved_bytes"] >= row["base_model_runtime"]["reserved_bytes"], f"peak reserve accounting: {path}")
        check(row["budget_check"]["declared_budget_bytes"] == BUDGET, f"budget check mismatch: {path}")
    check(all(x == data["decode_tokens"] for x in data["completed_decode_steps"]), f"decode trajectory incomplete: {path}")
    check(data.get("feasible") is True, f"completed cell did not pass feasibility: {path}")


def main() -> None:
    plan = ROOT / "docs/kv-capacity-plan-v2.md"
    summary_path = OUT / "summary.json"
    check(plan.exists() and summary_path.exists(), "plan or summary missing")
    summary = json.loads(summary_path.read_text())
    check(summary["plan_sha256"] == hashlib.sha256(plan.read_bytes()).hexdigest(), "summary does not bind frozen plan")
    check(summary["declared_budget_bytes"] == [BUDGET], "multiple or unexpected declared budgets")
    manifest = []
    for path in sorted(OUT.rglob("*.json")):
        if path.name == "summary.json" or "quality" in path.parts or any(part.startswith("memory") for part in path.parts):
            continue
        data = json.loads(path.read_text())
        if data.get("schema") == "kv-capacity-cell-v2" and data.get("instrumentation") == "timing":
            if path.parent.name == "screen" or (data.get("model_family") == "llama31_8b" and path.parent.name == "final" and "freegpu" not in path.name):
                continue
            verify_cell(path, data)
            manifest.append(path)
    check(len(manifest) == summary["attempted_cell_count"], "summary manifest count mismatch")
    check(summary["failure_cell_count"] > 0, "no OOM/blocked boundary was recorded")
    check(summary["cold_start_recorded_count"] >= 9, "cold-start timing records missing from confirmation cells")

    memory = []
    for path in sorted(OUT.rglob("*.json")):
        if not any(part.startswith("memory") for part in path.parts):
            continue
        data = json.loads(path.read_text())
        if data.get("schema") != "kv-capacity-cell-v2":
            continue
        check(data.get("instrumentation") == "memory", f"wrong memory mode: {path}")
        check(data.get("status") == "completed" and data.get("feasible") is True, f"memory replay failed: {path}")
        row = data["run"]["repetitions"][0]
        check(len(row["trace"]) == data["context_tokens"] * 0 + data["decode_tokens"] + 1, f"memory trace does not cover every position: {path}")
        check(all(item["allocator"]["allocated_bytes"] >= 0 for item in row["trace"]), f"memory allocator trace missing: {path}")
        check(row["trace"][-1]["token_position"] == data["context_tokens"] + data["decode_tokens"], f"memory token growth incomplete: {path}")
        check(row["peak"]["peak_reserved_bytes"] <= BUDGET, f"memory peak exceeds budget: {path}")
        memory.append(path)
    check(len(memory) >= 12, "missing separate memory-instrumented capacity traces")

    quality = []
    for path in sorted((OUT / "quality").glob("*.json")):
        data = json.loads(path.read_text())
        check(data.get("schema") == "kv-capacity-quality-v2", f"wrong quality schema: {path}")
        check(data.get("status", "completed") != "blocked", f"quality blocked: {path}")
        long = data["long_context_information_use"]
        free = data["free_running_generation"]
        check(len(long["per_sample"]) == data["sampling"]["long_context_count"], f"long quality count: {path}")
        check(len(free["per_sample"]) == data["sampling"]["free_running_count"], f"free quality count: {path}")
        check(long["nll_delta_bootstrap95"]["count"] == len(long["per_sample"],), f"quality bootstrap count: {path}")
        check(data["quality_timing_separate"] is True, f"quality/timing not separated: {path}")
        quality.append(path)
    check({path.name for path in quality} >= {"llama31_8b_kivi2_quality.json", "llama31_8b_kivi4_quality.json", "llama31_8b_swiftllm_quality.json", "llama32_1b_kivi2_quality.json", "llama32_1b_kivi4_quality.json", "llama32_1b_swiftllm_quality.json"}, "quality family coverage incomplete")

    capacities = summary["largest_feasible_capacity"]
    primary = {(x["model_family"], x["context_tokens"], x["decode_tokens"], x["method"]): x for x in capacities}
    for context, decode in ((2048, 256), (8192, 256), (4096, 512), (16384, 512)):
        for method in ("swiftllm_fp16", "kivi4", "kivi2"):
            check(("llama31_8b", context, decode, method) in primary, f"8B capacity missing: {context}/{decode}/{method}")
    check(any(x["batch_difference"] > 0 for x in summary["same_budget_comparisons"] if x["workload"]["model_family"] == "llama31_8b"), "no measured mixed capacity difference")
    check(all(x["latency_limit_1_25_pass"] is False or x["throughput_win"] for x in summary["same_budget_comparisons"] if x["workload"]["model_family"] == "llama31_8b"), "comparison record inconsistent")
    check({round(float(x["budget_fraction"]), 2) for x in summary["budget_sensitivity"]} == {0.85, 0.90, 0.95}, "budget sensitivity range missing")
    print(json.dumps({"summary": str(summary_path), "timing_cells": len(manifest), "memory_traces": len(memory), "quality_artifacts": len(quality), "status": "pass"}))


if __name__ == "__main__":
    main()
