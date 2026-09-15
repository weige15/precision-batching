#!/usr/bin/env python3
"""Check the evidence artifacts produced by the baseline commands.

This is an artifact consistency check, not a substitute for reviewing the
research scope or rerunning experiments.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "682cf9a28f97f7490409981a2f181528f377eb5d"
QAQ_COMMIT = "f8d47e0967c5c5f67f156c1f391a02b5cbd8183f"


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized_log(path: Path) -> list[str]:
    return [line for line in path.read_text().splitlines() if "Model creation time" not in line]


def verify_matrix(path: Path, expected_prompts: int, expected_layers: int | None = None) -> None:
    data = json.loads(path.read_text())
    check(data["environment"]["swiftllm_commit"] == COMMIT, f"wrong SwiftLLM commit in {path}")
    check(data["environment"]["swiftllm_upstream_commit"] == COMMIT, f"missing upstream provenance in {path}")
    check(
        data["environment"]["swiftllm_research_diff_sha256"] == file_sha256(ROOT / "references/swiftllm-research.diff"),
        f"missing research-tree provenance in {path}",
    )
    check(data["correctness"]["fp16_control"]["prefill_max_abs_logit_error"] <= 0.0, f"FP16 prefill control failed: {path}")
    check(data["correctness"]["fp16_control"]["decode_max_abs_logit_error"] <= 0.0, f"FP16 decode control failed: {path}")
    check("weighted_budget_oracle_comparison" in data, f"missing weighted budget oracle in {path}")
    all_layer = [r for r in data["results"] if r["config"]["scope"] == "all_layers"]
    check(all_layer and "weighted_storage_bits" in all_layer[0]["config"], f"missing weighted storage accounting in {path}")
    check("unweighted_fixed_oracle_comparison" in data, f"missing fixed oracle in {path}")
    check("unweighted_sum_oracle_comparison" in data, f"missing unweighted diagnostic in {path}")
    expected = {
        (q, k, v) for q, k, v in itertools.product((4, 8, 16), repeat=3)
    }
    actual = {
        (r["config"]["q_bits"], r["config"]["k_bits"], r["config"]["v_bits"])
        for r in all_layer
    }
    check(actual == expected, f"incomplete Q/K/V matrix in {path}")
    check(len(data["prompts"]) == expected_prompts, f"unexpected prompt count in {path}")
    for result in data["results"]:
        check(len(result["per_query"]) == expected_prompts, f"missing per-query records in {path}")
        for query in result["per_query"]:
            check(query["prefill_positions"], f"missing prefill positions in {path}")
            check(query["decode_positions"], f"missing decode positions in {path}")
    if expected_layers is not None:
        local = [r for r in data["results"] if r["config"]["scope"] == "single_layer"]
        check(len(local) == expected_layers * 6, f"incomplete layer sweep in {path}")


def verify_structured(path: Path, expected_calibration: int, expected_heldout: int, expected_task: int, expected_layers: int) -> None:
    data = json.loads(path.read_text())
    check(data["schema_version"] == 2, f"wrong structured schema in {path}")
    check(data["experiment"] == "structured_layer_by_projection_weight_precision", f"wrong structured experiment in {path}")
    check(data["scope"]["kv_cache_quantization"] is False, f"structured experiment must not quantize KV cache: {path}")
    check(data["scope"]["native_low_bit_kernel"] is False, f"structured experiment is not a native kernel: {path}")
    check(data["proxy"]["fp16_scale_overhead"] is False, f"FP16 scale overhead was charged: {path}")
    check(all(data["storage_accounting"]["manual_cases"]["checks"].values()), f"manual storage checks failed: {path}")
    check(data["environment"]["swiftllm_research_diff_sha256"] == file_sha256(ROOT / "references/swiftllm-research.diff"), f"structured provenance is stale: {path}")
    units = data["storage_accounting"]["unit_shapes"]
    fixed_fp16_bits = int(data["storage_accounting"]["fixed_fp16_bits"])
    config = data["model"]["config"]
    expected_fixed_numel = int(config["vocab_size"]) * int(config["hidden_size"]) * (1 + int(not config.get("tie_word_embeddings", False)))
    expected_fixed_numel += (2 * int(config["num_hidden_layers"]) + 1) * int(config["hidden_size"])
    check(fixed_fp16_bits == expected_fixed_numel * 16, f"fixed FP16 ledger does not match Llama non-unit parameters: {path}")
    check(len(units) == expected_layers * 5, f"expected Q/K/V/O/FFN unit for every layer: {path}")
    unit_by_key = {unit["key"]: unit for unit in units}
    check(len(unit_by_key) == len(units), f"duplicate structured units: {path}")

    def unit_cost(unit: dict, bits: int) -> dict[str, int]:
        weight = padding = scales = 0
        for matrix in unit["matrices"]:
            numel = int(matrix["numel"])
            if bits == 16:
                weight += 16 * numel
            else:
                quantized_numel = int(matrix["out_features"]) * int(matrix["padded_in_features"])
                weight += bits * quantized_numel
                padding += bits * (quantized_numel - numel)
                scales += 16 * int(matrix["scale_count"])
        return {"weight_payload_bits": weight, "padding_bits": padding, "scale_bits": scales, "zero_point_bits": 0, "metadata_bits": 0, "total_bits": weight + scales}

    for bits, expected in data["storage_accounting"]["uniform_profiles"].items():
        total = {key: 0 for key in ("weight_payload_bits", "padding_bits", "scale_bits", "zero_point_bits", "metadata_bits", "total_bits")}
        for unit in units:
            costs = unit_cost(unit, int(bits))
            for key in total:
                total[key] += costs[key]
        total["weight_payload_bits"] += fixed_fp16_bits
        total["total_bits"] += fixed_fp16_bits
        for key in total:
            check(total[key] == expected[key], f"storage formula mismatch for uniform W{bits} {key}: {path}")
        if int(bits) == 16:
            check(total["scale_bits"] == 0, f"FP16 unexpectedly has scales: {path}")

    sensitivity = data["single_unit_sensitivity"]
    check(set(sensitivity) == set(unit_by_key), f"single-unit sensitivity does not cover every unit: {path}")
    for unit_key, measurements in sensitivity.items():
        check(set(measurements) == {"4", "8", "16"}, f"missing 4/8/16 sensitivity for {unit_key}: {path}")
        check("summary" in measurements["4"] and "summary" in measurements["8"], f"missing sensitivity summaries for {unit_key}: {path}")

    check(data["policy_search"]["combined_calibration_metrics_are_used_for_selection"] is True, f"combined calibration selection missing: {path}")
    check(data["policy_search"]["heldout_metrics_are_not_used_for_selection"] is True, f"heldout selection leakage: {path}")
    check(data["policy_search"]["additive_prediction_is_not_final_selection_evidence"] is True, f"additive selection leakage: {path}")
    required_kinds = {"uniform", "projection_only", "layer", "layer_by_projection", "heuristic"}
    policies = data["policies"]
    check(required_kinds <= {row["profile"]["kind"] for row in policies}, f"missing structured policy kind: {path}")
    check({row["profile"]["name"] for row in policies} >= {"uniform_w4", "uniform_w8", "uniform_w16"}, f"missing uniform baselines: {path}")
    for row in policies:
        profile = row["profile"]
        total = {key: 0 for key in ("weight_payload_bits", "padding_bits", "scale_bits", "zero_point_bits", "metadata_bits", "total_bits")}
        check(set(profile["unit_bits"]) == set(unit_by_key), f"profile does not assign every unit {profile['name']}: {path}")
        for key, bits in profile["unit_bits"].items():
            check(key in unit_by_key and int(bits) in (4, 8, 16), f"invalid profile assignment {key}: {path}")
            costs = unit_cost(unit_by_key[key], int(bits))
            for field in total:
                total[field] += costs[field]
        total["weight_payload_bits"] += fixed_fp16_bits
        total["total_bits"] += fixed_fp16_bits
        for field in total:
            check(total[field] == profile["storage"][field], f"profile storage mismatch {profile['name']} {field}: {path}")
        check(profile["storage"]["fixed_fp16_bits"] == fixed_fp16_bits, f"fixed FP16 storage mismatch {profile['name']}: {path}")
        check(len(row["calibration"]["per_sample"]) == expected_calibration, f"calibration count mismatch {profile['name']}: {path}")
        check(len(row["heldout"]["per_sample"]) == expected_heldout, f"heldout count mismatch {profile['name']}: {path}")
        check(len(row["task"]["per_sample"]) == expected_task, f"task count mismatch {profile['name']}: {path}")
        check("bootstrap_95" in row["heldout"], f"missing heldout confidence interval {profile['name']}: {path}")
        check("bootstrap_95" in row["task"], f"missing task confidence interval {profile['name']}: {path}")
    calibration_ids = set(data["data"]["calibration"]["sample_ids"])
    heldout_ids = set(data["data"]["heldout"]["sample_ids"])
    check(calibration_ids.isdisjoint(heldout_ids), f"calibration and heldout samples overlap: {path}")
    fp16_noop = data["verification"]["all_fp16_noop"]
    check(fp16_noop["max_abs_nll_delta"] <= 0.0, f"FP16 NLL no-op failed: {path}")
    check(fp16_noop["max_logit_mse"] <= 0.0, f"FP16 logit no-op failed: {path}")
    check(fp16_noop["max_kl_reference_to_candidate"] <= 0.0, f"FP16 KL no-op failed: {path}")
    check(fp16_noop["min_top1_match"] >= 1.0, f"FP16 top-1 no-op failed: {path}")
    check(data["verification"]["selection_uses_calibration_only"] is True, f"selection leakage flag: {path}")
    check(data["verification"]["headline_uses_heldout"] is True, f"headline heldout flag: {path}")
    check(data["verification"]["additive_model_is_candidate_generator_only"] is True, f"additive leakage flag: {path}")
    check("combined profiles" in data["verification"]["interaction_adaptation"], f"missing interaction adaptation: {path}")
    check("oracle_to_global_logit_mse_ratio" in data["query_oracle"], f"missing oracle logit-MSE comparison: {path}")
    check(data["query_oracle"]["oracle_to_global_logit_mse_ratio"] >= 0, f"invalid oracle logit-MSE ratio: {path}")
    check(data["verification"]["budgets_exact_only_when_total_bits_equal"] is True, f"budget equality flag: {path}")


def main() -> None:
    check((ROOT / "vendor/swiftLLM/UPSTREAM_COMMIT").read_text().strip() == COMMIT, "research marker is not pinned")
    check((ROOT / "vendor/swiftLLM-upstream/UPSTREAM_COMMIT").read_text().strip() == COMMIT, "clean marker is not pinned")
    check((ROOT / "vendor/qaq/UPSTREAM_COMMIT").read_text().strip() == QAQ_COMMIT, "QAQ marker is not pinned")
    provenance = json.loads((ROOT / "results/baseline/research-provenance.json").read_text())
    check(provenance["swiftllm_upstream_commit"] == COMMIT, "research provenance commit is wrong")
    check(provenance["swiftllm_research_diff_sha256"] == file_sha256(ROOT / "references/swiftllm-research.diff"), "research provenance diff is stale")
    check(provenance["requirements_lock_sha256"] == file_sha256(ROOT / "requirements-lock.txt"), "requirements lock provenance is stale")
    check(
        normalized_log(ROOT / "results/baseline/swiftllm_unmodified_1b.log")
        == normalized_log(ROOT / "results/baseline/swiftllm_precision_noop_1b.log"),
        "SwiftLLM baseline/no-op outputs differ",
    )
    verify_matrix(ROOT / "results/sensitivity/llama32_1b_qkv_matrix.json", 8)
    verify_matrix(ROOT / "results/sensitivity/llama31_8b_qkv_matrix.json", 8)
    verify_matrix(ROOT / "results/sensitivity/llama32_1b_qkv_matrix_layer_sweep.json", 8, 16)
    verify_matrix(ROOT / "results/sensitivity/llama31_8b_qkv_matrix_layer_sweep_4prompts.json", 4, 32)
    verify_structured(ROOT / "results/sensitivity/llama32_1b_structured.json", 16, 32, 32, 16)
    verify_structured(ROOT / "results/sensitivity/llama31_8b_structured.json", 8, 16, 16, 32)
    for required in (
        "docs/baseline.md",
        "docs/claim-evidence.md",
        "docs/feasibility-report.md",
        "docs/papers.md",
        "docs/source-map.md",
        "references/qaq-2403.04643.pdf",
        "references/morphserve-2506.02006-v2.pdf",
        "references/swiftllm-research.diff",
        "requirements-lock.txt",
        "results/baseline/research-provenance.json",
    ):
        check((ROOT / required).exists(), f"missing required artifact: {required}")
    print("artifact verification passed; this does not make blocked native-kernel questions complete")


if __name__ == "__main__":
    main()
