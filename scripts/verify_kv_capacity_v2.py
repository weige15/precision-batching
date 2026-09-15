#!/usr/bin/env python3
"""Strict verifier for the selected v2 capacity evidence."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/kv-capacity-v2"
BUDGET = 22766439628
WORKLOADS = ((2048, 256), (8192, 256), (4096, 512), (16384, 512))
METHODS = {"hf_fp16", "kivi2", "kivi4", "swiftllm_fp16"}


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def verify_record(row: dict) -> None:
    path = ROOT / row["artifact"]
    check(path.exists(), f"missing raw artifact: {path}")
    data = json.loads(path.read_text())
    check(data.get("schema") == "kv-capacity-cell-v2", f"wrong cell schema: {path}")
    check(data.get("instrumentation") == "timing", f"wrong instrumentation: {path}")
    if data.get("status") != "completed":
        check(data.get("error", {}).get("message"), f"failure lacks exact error: {path}")
        check("run" not in data and "feasible" not in data, f"failure invents a run/feasibility: {path}")
        return
    check(data.get("feasible") is True, f"completed selected cell is not feasible: {path}")
    check(data.get("method") in METHODS, f"unknown method: {path}")
    check(data.get("budget", {}).get("declared_budget_bytes") == BUDGET, f"budget mismatch: {path}")
    check(data.get("method_config", {}).get("weights_dtype") == "torch.float16", f"weights are not FP16: {path}")
    check(data.get("method_config", {}).get("kv_backend"), f"KV backend missing: {path}")
    run = data["run"]
    reps = run.get("repetitions", [])
    check(len(reps) >= 5, f"selected capacity cell lacks five warmed repeats: {path}")
    check(len(data.get("completed_decode_steps", [])) == len(reps), f"decode repeat manifest mismatch: {path}")
    check(all(step == data["decode_tokens"] for step in data["completed_decode_steps"]), f"decode trajectory incomplete: {path}")
    for rep in reps:
        check(rep.get("prefill_ms") is not None and rep.get("decode_ms") is not None, f"missing complete timing: {path}")
        peak = rep.get("peak", {})
        base = rep.get("base_model_runtime", {})
        check(peak.get("peak_allocated_bytes", -1) >= base.get("allocated_bytes", 0), f"allocated peak accounting: {path}")
        check(peak.get("peak_reserved_bytes", -1) >= base.get("reserved_bytes", 0), f"reserved peak accounting: {path}")
        budget = rep.get("budget_check", {})
        check(budget.get("declared_budget_bytes") == BUDGET, f"budget check missing: {path}")
        check(budget.get("within_budget") is True, f"selected repeat exceeds budget: {path}")
    boundary = run.get("timing_boundary", "")
    check("complete" in boundary and "quality" in boundary.lower(), f"timing boundary is not complete/model-only: {path}")


def verify_memory(path: Path) -> None:
    data = json.loads(path.read_text())
    check(data.get("schema") == "kv-capacity-cell-v2", f"wrong memory schema: {path}")
    check(data.get("instrumentation") == "memory" and data.get("status") == "completed", f"memory replay incomplete: {path}")
    check(data.get("feasible") is True, f"memory replay infeasible: {path}")
    row = data["run"]["repetitions"][0]
    trace = row.get("trace", [])
    check(len(trace) == data["decode_tokens"] + 1, f"memory trace misses a position: {path}")
    positions = [item.get("token_position") for item in trace]
    check(positions == list(range(data["context_tokens"], data["context_tokens"] + data["decode_tokens"] + 1)), f"memory positions are not contiguous: {path}")
    check(all(item.get("allocator", {}).get("allocated_bytes", -1) >= 0 for item in trace), f"allocator trace missing: {path}")
    check(row["peak"]["peak_reserved_bytes"] <= BUDGET and row["peak"]["peak_allocated_bytes"] <= BUDGET, f"memory peak exceeds budget: {path}")
    check(data.get("run", {}).get("timing_boundary", "").startswith("not a latency run"), f"memory replay mislabeled as latency: {path}")


def verify_quality(path: Path) -> dict:
    data = json.loads(path.read_text())
    check(data.get("schema") == "kv-capacity-quality-v2" and data.get("status") == "completed", f"quality incomplete: {path}")
    check(data.get("provenance", {}).get("source_sha256"), f"quality provenance missing: {path}")
    check(data.get("model_files"), f"quality model manifest missing: {path}")
    sampling = data["sampling"]
    long = data["long_context_information_use"]
    free = data["free_running_generation"]
    check(sampling.get("long_context_count", 0) >= 8 and len(long["per_sample"]) == sampling["long_context_count"], f"long quality support incomplete: {path}")
    check(sampling.get("free_running_count", 0) >= 8 and len(free["per_sample"]) == sampling["free_running_count"], f"free quality support incomplete: {path}")
    check(sampling.get("artificial_prompt_repetition") is False, f"quality uses artificial prompt repetition: {path}")
    check(sampling.get("eos_handled") is True and free.get("termination_records") is True and free.get("pathology_records") is True, f"free-running safeguards absent: {path}")
    check(free.get("per_step_divergence_recorded") is True, f"free-running divergence absent: {path}")
    check(long.get("kl_mean") is not None and long.get("logit_mse_mean") is not None, f"long-context divergence metrics absent: {path}")
    for sample in long["per_sample"]:
        check(sample.get("query", {}).get("kind") == "identified_continuation_boundary", f"identified query missing: {path}")
        check(len(sample.get("per_token_divergence", [])) == sample["target_tokens"], f"per-token divergence missing: {path}")
    for sample in free["per_sample"]:
        term = sample.get("candidate_termination", {})
        check("termination_reason" in term and "repetition_pathology" in term and "safety_failure" in term, f"termination/pathology record missing: {path}")
        check("per_step_divergence" in sample, f"free per-step divergence missing: {path}")
    interval = long["nll_delta_bootstrap95"]
    return {"path": str(path.relative_to(ROOT)), "method": data["method"], "model_family": data["model_family"], "context_tokens": data["sampling"].get("long_context_window_tokens"), "nll_upper": interval["ci95_high"], "nll_gate_pass": interval["ci95_high"] <= 0.02}


def main() -> None:
    summary_path = OUT / "summary.json"
    plan = ROOT / "docs/kv-capacity-plan-v2.md"
    check(summary_path.exists() and plan.exists(), "summary or frozen plan missing")
    summary = json.loads(summary_path.read_text())
    check(summary.get("plan_sha256") == hashlib.sha256(plan.read_bytes()).hexdigest(), "summary is not bound to frozen plan")
    check(summary.get("declared_budget_bytes") == [BUDGET], "unexpected budget")
    records = summary.get("records", [])
    check(records and summary["attempted_cell_count"] == len(records), "timing manifest count mismatch")
    check(summary["failure_cell_count"] > 0, "no recorded OOM boundary")
    for row in records:
        verify_record(row)
    completed_capacity = [r for r in records if r["scope"] == "capacity" and r["status"] == "completed" and r["feasible"]]
    check(all(r.get("repetitions", 0) >= 5 for r in completed_capacity), "a selected capacity success lacks repeated measurements")
    capacities = summary["largest_feasible_capacity"]
    by_key = {(r["model_family"], r["context_tokens"], r["decode_tokens"], r["method"]): r for r in capacities}
    for context, decode in WORKLOADS:
        for method in METHODS:
            check(("llama31_8b", context, decode, method) in by_key, f"8B capacity missing: {context}/{decode}/{method}")
    equal = summary.get("equal_batch_comparisons", [])
    check(len(equal) >= 9, "equal-batch evidence incomplete")
    check(all(row.get("same_batch") is True for row in equal), "equal-batch manifest is not equal-batch")
    additional = summary.get("additional_request_allocation", [])
    check(additional, "actual additional-request allocation artifact missing")
    completed_additional = []
    for row in additional:
        path = ROOT / row["artifact"]
        data = json.loads(path.read_text())
        if data.get("status") != "completed":
            check(data.get("error", {}).get("message"), f"blocked additional-request probe lacks error: {path}")
            continue
        completed_additional.append(path)
        check(data.get("schema") == "kv-additional-request-v2", f"wrong additional-request schema: {path}")
        check(row.get("exact_pool_additional_status") == "failed", f"exact-full pool unexpectedly admitted request: {path}")
        check(row.get("one_extra_pool_additional_status") == "completed", f"one-extra pool did not admit request: {path}")
        exact = data["exact_pool"]["after_base"]["blocks"]
        extra = data["one_extra_pool"]["additional_request"]["blocks"]
        check(exact["free_blocks"] == 0 and extra["free_blocks"] == 0, f"block accounting does not show real additional allocation: {path}")
        check(extra["allocated_blocks"] > exact["allocated_blocks"], f"additional request did not consume blocks: {path}")
        check(data["one_extra_pool"]["additional_request"]["allocator"]["peak_reserved_bytes"] <= BUDGET, f"additional request peak exceeds budget: {path}")
    check(completed_additional, "no completed additional-request allocation probe")
    memory_paths = []
    for rel in summary.get("memory_instrumented_artifacts", []):
        path = ROOT / rel
        verify_memory(path)
        memory_paths.append(path)
    check(len(memory_paths) >= 10, "insufficient separate memory traces")
    quality_results = []
    for path in sorted((OUT / "quality-final").glob("*.json")):
        quality_results.append(verify_quality(path))
    check({(r["model_family"], r["method"]) for r in quality_results} >= {(family, method) for family in ("llama31_8b", "llama32_1b") for method in ("kivi2", "kivi4", "swiftllm_fp16")}, "quality family coverage incomplete")
    comparisons = [r for r in summary["same_budget_comparisons"] if r["workload"]["model_family"] == "llama31_8b" and r["mixed_method"] in {"kivi2", "kivi4"}]
    check(len(comparisons) == 8, "8B candidate comparison grid incomplete")
    qualifying = summary.get("qualifying_equal_batch_gains", [])
    check(summary.get("verdict") == "GO" and qualifying, "decision does not contain a qualifying gain")
    check(any(x["candidate_method"] == "kivi4" and x["workload"]["context_tokens"] == 16384 and x["throughput_ratio"] > 1.0 and x["quality_nll_gate_pass"] for x in qualifying), "expected replicated 8B K4 long-context throughput gain missing")
    q_by_workload = {(r["model_family"], r["method"], r["context_tokens"]): r for r in quality_results}
    capacity_qualifiers = []
    for row in comparisons:
        q = q_by_workload.get(("llama31_8b", row["mixed_method"], row["workload"]["context_tokens"]))
        if q and q["nll_gate_pass"] and row["capacity_gate_pass"] and row["declared_latency_quality_gates_pass"]:
            capacity_qualifiers.append(row)
    check(any(x["mixed_method"] == "kivi4" and x["workload"]["context_tokens"] == 8192 and x["workload"]["decode_tokens"] == 256 for x in capacity_qualifiers), "expected 8B K4 c8192 capacity gain missing")
    print(json.dumps({"summary": str(summary_path), "timing_cells": len(records), "memory_traces": len(memory_paths), "quality_artifacts": len(quality_results), "qualifying_equal_batch_gains": len(qualifying), "qualifying_capacity_gains": len(capacity_qualifiers), "status": "pass"}))


if __name__ == "__main__":
    main()
