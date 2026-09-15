#!/usr/bin/env python3
"""Verify baseline artifacts and the actual interaction-aware experiment.

Declarative flags are not sufficient here.  The structured verifier recomputes
storage, checks that W8-background marginal profiles really ran, verifies
window/shard separation and non-leakage, verifies all combined candidate
executions (including all 243 projection assignments), and checks the recorded
search and 1B gate provenance.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "682cf9a28f97f7490409981a2f181528f377eb5d"
QAQ_COMMIT = "f8d47e0967c5c5f67f156c1f391a02b5cbd8183f"
BASE_EXPERIMENT_COMMIT = "b5fc47744d6c4aa1a46206439cdb6b70c29a88ad"
KV_LEGACY_SOURCE_FILES = (
    "vendor/swiftLLM/swiftllm/worker/kv_cache.py",
    "vendor/swiftLLM/swiftllm/worker/kernels/kvcache_mgmt.py",
    "vendor/swiftLLM/swiftllm/worker/layers/transformer_layer.py",
    "vendor/swiftLLM/swiftllm/worker/layers/post_layer.py",
    "vendor/swiftLLM/swiftllm/worker/model.py",
    "vendor/swiftLLM/swiftllm/engine_config.py",
    "scripts/kv_precision_experiment.py",
)
KV_OPTIMIZED_SOURCE_FILES = (
    "vendor/swiftLLM/swiftllm/worker/kv_cache.py",
    "vendor/swiftLLM/swiftllm/worker/kernels/kvcache_mgmt.py",
    "vendor/swiftLLM/swiftllm/worker/kernels/segmented_paged_attn.py",
    "vendor/swiftLLM/swiftllm/worker/layers/transformer_layer.py",
    "vendor/swiftLLM/swiftllm/worker/layers/post_layer.py",
    "vendor/swiftLLM/swiftllm/worker/model.py",
    "vendor/swiftLLM/swiftllm/engine_config.py",
    "scripts/kv_precision_experiment.py",
)
LEGACY_KV_HEAD = "1351139bfab5fa1a629af5352687aa8aa45e0914"
LEGACY_KV_SOURCE_SHA256 = "5849053a31cd96b7d222af5568d1fc3ab5ca3bdbff957973bfc8421c6fc2b28d"
BREAK_EVEN_SOURCE_FILES = (
    "scripts/kv_break_even_experiment.py",
    "scripts/kv_precision_experiment.py",
    "vendor/swiftLLM/swiftllm/worker/kv_cache.py",
    "vendor/swiftLLM/swiftllm/worker/kernels/segmented_paged_attn.py",
    "vendor/swiftLLM/swiftllm/worker/kernels/paged_attn.py",
    "vendor/swiftLLM/swiftllm/worker/model.py",
    "vendor/swiftLLM/csrc/src/block_swapping.cpp",
)


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_fingerprint(files: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    for relative in files:
        digest.update(relative.encode("utf-8"))
        digest.update((ROOT / relative).read_bytes())
    return digest.hexdigest()


def normalized_log(path: Path) -> list[str]:
    return [line for line in path.read_text().splitlines() if "Model creation time" not in line]


def verify_matrix(path: Path, expected_prompts: int, expected_layers: int | None = None) -> None:
    data = json.loads(path.read_text())
    check(data["environment"]["swiftllm_commit"] == COMMIT, f"wrong SwiftLLM commit in {path}")
    check(data["environment"]["swiftllm_upstream_commit"] == COMMIT, f"missing upstream provenance in {path}")
    check(data["environment"]["swiftllm_research_diff_sha256"] == file_sha256(ROOT / "references/swiftllm-research.diff"), f"stale research provenance in {path}")
    check(data["correctness"]["fp16_control"]["prefill_max_abs_logit_error"] <= 0.0, f"FP16 prefill control failed: {path}")
    check(data["correctness"]["fp16_control"]["decode_max_abs_logit_error"] <= 0.0, f"FP16 decode control failed: {path}")
    expected = set(itertools.product((4, 8, 16), repeat=3))
    all_layer = [row for row in data["results"] if row["config"]["scope"] == "all_layers"]
    actual = {(row["config"]["q_bits"], row["config"]["k_bits"], row["config"]["v_bits"]) for row in all_layer}
    check(actual == expected, f"incomplete Q/K/V matrix in {path}")
    check(len(data["prompts"]) == expected_prompts, f"unexpected prompt count in {path}")
    for row in data["results"]:
        check(len(row["per_query"]) == expected_prompts, f"missing per-query records in {path}")
        for query in row["per_query"]:
            check(query["prefill_positions"] and query["decode_positions"], f"missing positions in {path}")
    if expected_layers is not None:
        local = [row for row in data["results"] if row["config"]["scope"] == "single_layer"]
        check(len(local) == expected_layers * 6, f"incomplete layer sweep in {path}")


def intervals_are_disjoint(metadata: dict) -> bool:
    intervals = []
    for shard in metadata["shards"]:
        for start, end in shard["window_intervals"]:
            intervals.append((int(start), int(end)))
    intervals.sort()
    return all(left[1] <= right[0] for left, right in zip(intervals, intervals[1:]))


def expected_unit_cost(unit: dict, bits: int) -> dict[str, int]:
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
    return {
        "weight_payload_bits": weight,
        "padding_bits": padding,
        "scale_bits": scales,
        "zero_point_bits": 0,
        "metadata_bits": 0,
        "total_bits": weight + scales,
    }


def expected_profile_storage(profile: dict, units: dict[str, dict], fixed_bits: int) -> dict[str, int]:
    total = {field: 0 for field in ("weight_payload_bits", "padding_bits", "scale_bits", "zero_point_bits", "metadata_bits", "total_bits")}
    for key, bits in profile["unit_bits"].items():
        cost = expected_unit_cost(units[key], int(bits))
        for field in total:
            total[field] += cost[field]
    total["weight_payload_bits"] += fixed_bits
    total["total_bits"] += fixed_bits
    return total


def close(actual: float, expected: float, where: str) -> None:
    check(math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-7), f"numeric summary mismatch at {where}: {actual} != {expected}")


def recompute_summary(records: list[dict]) -> dict[str, float]:
    if not records:
        return {}
    keys = [key for key, value in records[0].items() if isinstance(value, (float, int)) and key != "token_count"]
    summary = {key: float(np.mean([float(record[key]) for record in records])) for key in keys}
    if "nll" in summary:
        summary["perplexity_from_mean_nll"] = math.exp(min(summary["nll"], 30.0))
    if "reference_nll" in summary:
        summary["reference_perplexity_from_mean_nll"] = math.exp(min(summary["reference_nll"], 30.0))
    return summary


def recompute_bootstrap(values: list[float], seed: int, iterations: int) -> dict[str, float | int]:
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


def verify_profile_shape(profile: dict, units: dict[str, dict], ordered_keys: list[str], where: str) -> None:
    check(set(profile["unit_bits"]) == set(ordered_keys), f"profile does not assign exactly every unit at {where}")
    check(profile["signature"] == ",".join(str(profile["unit_bits"][key]) for key in ordered_keys), f"profile signature mismatch at {where}")
    check(all(int(bits) in {4, 8, 16} for bits in profile["unit_bits"].values()), f"invalid profile bit assignment at {where}")


def verify_calibration_execution(
    calibration: dict,
    expected_count: int,
    expected_shards: int,
    where: str,
    metadata: dict | None = None,
) -> None:
    check(len(calibration["per_sample"]) == expected_count, f"wrong combined calibration sample count at {where}")
    check(len(calibration["shards"]) == expected_shards, f"missing calibration shard execution at {where}")
    ids = []
    for shard_id, shard in enumerate(calibration["shards"]):
        shard_ids = [row["sample_id"] for row in shard["per_sample"]]
        ids.extend(shard_ids)
        if metadata is not None:
            check(shard_ids == metadata["shards"][shard_id]["sample_ids"], f"calibration IDs do not match declared shard {shard_id} at {where}")
        for key, value in recompute_summary(shard["per_sample"]).items():
            close(float(shard["summary"][key]), value, f"{where}/shard{shard_id}/{key}")
    overall_ids = [row["sample_id"] for row in calibration["per_sample"]]
    check(ids == overall_ids, f"shard records do not match combined calibration records at {where}")
    if metadata is not None:
        check(overall_ids == metadata["sample_ids"], f"calibration records do not match declared sample IDs at {where}")
    check(len(set(ids)) == len(ids), f"duplicate calibration sample record at {where}")
    for key, value in recompute_summary(calibration["per_sample"]).items():
        close(float(calibration["summary"][key]), value, f"{where}/{key}")
    check("stability" in calibration, f"missing shard stability calculation at {where}")
    stability = calibration["stability"]
    shard_deltas = [float(shard["summary"].get("nll_delta", 0.0)) for shard in calibration["shards"]]
    check(len(stability.get("shard_nll_deltas", [])) == expected_shards, f"stability is not shard-aware at {where}")
    for actual, expected in zip(stability["shard_nll_deltas"], shard_deltas):
        close(float(actual), expected, f"{where}/stability/shard_delta")
    mean = float(np.mean(shard_deltas))
    worst = float(max(shard_deltas))
    std = float(np.std(shard_deltas))
    improved = sum(delta < 0.0 for delta in shard_deltas)
    required = max(2, math.ceil(expected_shards * 2 / 3))
    tolerance = float(stability.get("tolerance_nll", 0.002))
    close(float(stability["mean_nll_delta"]), mean, f"{where}/stability/mean")
    close(float(stability["worst_shard_nll_delta"]), worst, f"{where}/stability/worst")
    close(float(stability["shard_std_nll_delta"]), std, f"{where}/stability/std")
    check(int(stability["improved_shard_count"]) == improved, f"stability improvement count mismatch at {where}")
    check(int(stability["required_improved_shards"]) == required, f"stability threshold mismatch at {where}")
    check(stability["stable_improvement"] == bool(mean < 0.0 and improved >= required and worst <= tolerance), f"stability decision mismatch at {where}")
    close(float(stability["stable_rank_score"]), mean + 0.5 * max(0.0, worst) + 0.25 * std, f"{where}/stability/rank")


def verify_heldout_execution(result: dict, expected_ids: set[str], where: str) -> None:
    rows = result["per_sample"]
    check({row["sample_id"] for row in rows} == expected_ids, f"heldout IDs mismatch at {where}")
    check(len(rows) == len(expected_ids), f"heldout count mismatch at {where}")
    for key, value in recompute_summary(rows).items():
        close(float(result["summary"][key]), value, f"{where}/summary/{key}")
    check(isinstance(result.get("bootstrap_seed"), int), f"heldout bootstrap seed missing at {where}")
    iterations = int(result["bootstrap_95"]["nll_delta"]["bootstrap_iterations"])
    for offset, key in enumerate(("nll_delta", "logit_mse", "kl_reference_to_candidate", "top1_match"), start=1):
        expected = recompute_bootstrap([float(row[key]) for row in rows], int(result["bootstrap_seed"]) + offset, iterations)
        actual = result["bootstrap_95"][key]
        for field in expected:
            if field in {"count", "bootstrap_iterations"}:
                check(int(actual[field]) == int(expected[field]), f"bootstrap count mismatch at {where}/{key}")
            else:
                close(float(actual[field]), float(expected[field]), f"{where}/bootstrap/{key}/{field}")
    check(result["paired_nll_difference_vs_uniform_w8"] == result["bootstrap_95"]["nll_delta"], f"paired NLL record is not the stored bootstrap at {where}")
    check(result["paired_metric_differences_vs_uniform_w8"] == result["bootstrap_95"], f"paired metric record is not the stored bootstrap at {where}")


def verify_interaction_aware(path: Path) -> None:
    data = json.loads(path.read_text())
    check(data["schema_version"] == 3, f"wrong interaction-aware schema in {path}")
    check(data["experiment"] == "interaction_aware_w8_centered_structured_weight_precision", f"wrong interaction-aware experiment in {path}")
    provenance = data["source_provenance"]
    check(provenance["required_branch"] == "structured-precision-evidence", f"wrong source branch declaration in {path}")
    check(provenance["required_base_commit"] == BASE_EXPERIMENT_COMMIT, f"wrong source base commit in {path}")
    current_branch = subprocess.check_output(["git", "-C", str(ROOT), "branch", "--show-current"], text=True).strip()
    current_head = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    check(current_branch == provenance["branch"] == "structured-precision-evidence", f"artifact branch does not match current checkout in {path}")
    artifact_commit = provenance["commit_at_run"]
    is_ancestor = subprocess.run(
        ["git", "-C", str(ROOT), "merge-base", "--is-ancestor", artifact_commit, current_head],
        check=False,
    ).returncode == 0
    check(is_ancestor and artifact_commit == BASE_EXPERIMENT_COMMIT, f"artifact commit is not a historical ancestor of the current checkout in {path}")
    check(provenance["commit_at_run"] == BASE_EXPERIMENT_COMMIT, f"artifact was not run from the requested base commit in {path}")
    check(provenance["base_commit"] == BASE_EXPERIMENT_COMMIT, f"artifact base commit provenance is stale in {path}")
    check(provenance["swiftllm_upstream_commit"] == COMMIT, f"wrong SwiftLLM provenance in {path}")
    check(provenance["swiftllm_research_diff_sha256"] == file_sha256(ROOT / "references/swiftllm-research.diff"), f"stale research diff provenance in {path}")
    run_mode = data.get("run_mode", "exploration")
    log_path = path.with_suffix(".log")
    check(log_path.exists(), f"missing experiment execution log for {path}")
    log_text = log_path.read_text()
    check("wrote " in log_text, f"experiment log has no successful write marker for {path}")
    if run_mode == "exploration":
        check("projection exhaustive [243/243]" in log_text, f"1B execution log lacks the final projection run for {path}")
    else:
        check("skipping 1B-only projection enumeration" in log_text, f"8B confirmation log lacks scope marker for {path}")
        gate_path = Path(provenance["gate_artifact"])
        check(gate_path.exists(), f"missing prior 1B gate artifact for {path}")
        check(provenance["gate_artifact_sha256"] == file_sha256(gate_path), f"prior 1B gate hash mismatch for {path}")
        gate_data = json.loads(gate_path.read_text())
        check(gate_data["final_gate"]["decision"] == "OPEN_8B_CONFIRMATION", f"8B confirmation was not gated by a passing 1B artifact for {path}")
        check(provenance["gate_artifact_decision"] == "OPEN_8B_CONFIRMATION", f"8B gate decision provenance mismatch for {path}")
        check(float(provenance["gate_artifact_created_unix"]) < float(data["created_unix"]), f"8B artifact predates its 1B gate artifact for {path}")
    check(data["scope"]["kv_cache_quantization"] is False, f"KV cache was included in {path}")
    check(data["scope"]["native_low_bit_kernel"] is False, f"native kernel was included in {path}")
    check(data["scope"]["scheduler_or_router"] is False, f"scheduler/router work was included in {path}")
    check(data["scope"]["query_specific_oracle"] is False, f"query oracle was included in {path}")
    check(data["proxy"]["fp16_scale_overhead"] is False, f"FP16 scale overhead was charged in {path}")
    check(all(data["storage_accounting"]["manual_cases"]["checks"].values()), f"manual storage cases failed in {path}")
    check(data["verification"]["exact_integer_storage_bits_and_bytes"] is True, f"integer storage evidence missing in {path}")

    units_list = data["storage_accounting"]["unit_shapes"]
    units = {unit["key"]: unit for unit in units_list}
    ordered_unit_keys = [unit["key"] for unit in units_list]
    config = data["model"]["config"]
    expected_layers = int(config["num_hidden_layers"])
    if run_mode == "exploration":
        check(expected_layers == 16 and int(config["hidden_size"]) == 2048, f"exploration artifact is not Llama 3.2 1B in {path}")
    else:
        check(expected_layers == 32 and int(config["hidden_size"]) == 4096, f"confirmation artifact is not Llama 3.1 8B in {path}")
    check(len(units_list) == expected_layers * 5, f"missing Q/K/V/O/FFN units in {path}")
    check(len(units) == len(units_list), f"duplicate unit keys in {path}")
    fixed_bits = int(data["storage_accounting"]["fixed_fp16_bits"])
    expected_fixed_numel = int(config["vocab_size"]) * int(config["hidden_size"]) * (1 + int(not config.get("tie_word_embeddings", False)))
    expected_fixed_numel += (2 * expected_layers + 1) * int(config["hidden_size"])
    check(fixed_bits == expected_fixed_numel * 16, f"fixed parameter ledger mismatch in {path}")

    uniform = data["storage_accounting"]["uniform_profiles"]
    for bits_text, recorded in uniform.items():
        assignment = {key: int(bits_text) for key in units}
        fake_profile = {"unit_bits": assignment}
        expected = expected_profile_storage(fake_profile, units, fixed_bits)
        for field in expected:
            check(int(recorded[field]) == expected[field], f"uniform storage mismatch {bits_text}/{field} in {path}")
        check(int(recorded["total_bytes"]) * 8 == int(recorded["total_bits"]), f"non-integer uniform bytes in {path}")

    calibration = data["data"]["calibration"]
    heldout = data["data"]["heldout"]
    check(int(calibration["shard_count"]) >= 3, f"fewer than three calibration shards in {path}")
    check(calibration["dataset"].endswith("/train"), f"calibration is not Wikitext train in {path}")
    check(heldout["dataset"].endswith("/validation"), f"heldout is not Wikitext validation in {path}")
    check(calibration["all_windows_non_overlapping"] is True and heldout["all_windows_non_overlapping"] is True, f"window overlap flag missing in {path}")
    for window_set in (calibration, heldout):
        check("randomized" in window_set["selection_method"] and "bucket" in window_set["selection_method"], f"window selection is not declared dispersed/randomized in {path}")
        check(int(window_set["dispersed_start_max"]) - int(window_set["dispersed_start_min"]) >= int(window_set["seq_len"]), f"window set is not dispersed in {path}")
    check(intervals_are_disjoint(calibration) and intervals_are_disjoint(heldout), f"dispersed windows overlap in {path}")
    cal_ids = set(calibration["sample_ids"])
    held_ids = set(heldout["sample_ids"])
    check(cal_ids.isdisjoint(held_ids), f"calibration/heldout IDs overlap in {path}")
    check(data["data"]["calibration_heldout_ids_disjoint"] is True, f"non-leakage result is false in {path}")
    check(data["data"]["validation_loaded_after_search"] is True, f"validation loading order was not recorded in {path}")
    check(set(data["data"]["search_uses_sample_ids"]) == cal_ids, f"search uses data outside calibration shards in {path}")
    expected_cal = int(calibration["count"])
    expected_hold = int(heldout["count"])

    marginals = data["w8_centered_marginals"]
    baseline_signature = marginals["background_profile"]["signature"]
    check(baseline_signature == ",".join(["8"] * len(units)), f"marginal background is not uniform W8 in {path}")
    if data.get("run_mode", "exploration") == "exploration":
        check(marginals["skipped_for_confirmation"] is False, f"exploration skipped W8 margins in {path}")
        check(marginals["all_units_measured"] is True, f"W8 marginal coverage flag is false in {path}")
        check(marginals["measured_targets"] == [4, 16], f"W8 marginal targets are incomplete in {path}")
        check(set(marginals["records"]) == set(units), f"W8 margins do not cover every unit in {path}")
        for unit_key, records in marginals["records"].items():
            check(set(records) == {"4", "8", "16"}, f"incomplete W8 marginal bit records for {unit_key} in {path}")
            for bits_text, record in records.items():
                profile = record["profile"]
                differing = [key for key, bits in profile["unit_bits"].items() if int(bits) != 8]
                check(differing == ([] if bits_text == "8" else [unit_key]), f"marginal profile is not one-unit W8-centered at {unit_key}/{bits_text} in {path}")
                verify_profile_shape(profile, units, ordered_unit_keys, f"marginal {unit_key}/{bits_text}")
                expected_marginal = expected_profile_storage(profile, units, fixed_bits)
                for field in expected_marginal:
                    check(int(profile["storage"][field]) == expected_marginal[field], f"marginal storage mismatch at {unit_key}/{bits_text}/{field}")
                check(int(record["storage_delta_vs_uniform_w8"]["total_bits"]) == int(profile["storage_delta_vs_uniform_w8"]["total_bits"]), f"marginal storage delta not recorded at {unit_key}/{bits_text}")
                if bits_text in {"4", "16"}:
                    check(record["measured_combined_profile"] is True, f"unexecuted W8 marginal at {unit_key}/{bits_text} in {path}")
                    verify_calibration_execution(record["calibration"], expected_cal, int(calibration["shard_count"]), f"marginal {unit_key}/{bits_text}", calibration)
    else:
        check(marginals["skipped_for_confirmation"] is True, f"confirmation did not record the marginal-sweep scope exception in {path}")
        check(marginals["all_units_measured"] is False and not marginals["records"], f"confirmation unexpectedly contains a partial marginal sweep in {path}")

    projection = data["projection_enumeration"]
    check(projection["optimizer_description"] == "exhaustive 3^5 enumeration (243 assignments)", f"projection optimizer description typo/staleness in {path}")
    check(int(projection["total_assignments"]) == 243, f"projection enumeration is not 243 in {path}")
    if data.get("run_mode", "exploration") == "exploration":
        check(int(projection["executed_assignments"]) == 243, f"not all projection assignments were executed in {path}")
        check(int(projection["unique_signatures"]) == 243, f"projection assignments were deduplicated incorrectly in {path}")
        projection_patterns = {
            tuple(int(row["profile"]["unit_bits"][f"layer_000/{name}"]) for name in ("q", "k", "v", "o", "ffn"))
            for row in projection["evaluations"]
        }
        check(projection_patterns == set(itertools.product((4, 8, 16), repeat=5)), f"projection enumeration does not cover every 3^5 type assignment in {path}")
        check(projection["all_feasible_and_near_budget_executed"] is True, f"feasible/near-budget projection coverage missing in {path}")
        for index, row in enumerate(projection["evaluations"]):
            # The all-W8 assignment is also the uniform reference row; it is
            # deduplicated by signature but still counts as an executed projection
            # assignment.  Every other enumerated row must retain projection kind.
            is_uniform_reference = row["profile"]["signature"] == baseline_signature
            check(row["profile"]["kind"] == ("uniform" if is_uniform_reference else "projection_only"), f"projection row has wrong kind in {path}")
            check("projection_exhaustive_3^5" in row["origins"], f"projection row was not actually executed in {path}")
            verify_calibration_execution(row["calibration"], expected_cal, int(calibration["shard_count"]), f"projection {index}", calibration)
    else:
        check(data["run_mode"] == "confirmation", f"unknown interaction-aware run mode in {path}")
        check(projection["skipped_for_confirmation"] is True, f"confirmation did not record projection skip in {path}")
        check(int(projection["executed_assignments"]) == 0 and not projection["evaluations"], f"confirmation unexpectedly has partial projection enumeration in {path}")

    search = data["interaction_aware_search"]
    check(search["marginals_only_propose_moves"] is True, f"search incorrectly treats margins as objective in {path}")
    check(search["final_ranking_uses_actual_combined_calibration"] is True, f"search does not rank actual combined runs in {path}")
    check("actual shard" in search["ranking"], f"actual combined-calibration search ranking is not documented in {path}")
    check(int(search["new_unique_evaluations"]) <= int(search["evaluation_budget"]), f"search evaluation budget exceeded in {path}")
    eval_by_signature = {row["profile"]["signature"]: row for row in data["combined_candidate_evaluations"]}
    check(len(eval_by_signature) == len(data["combined_candidate_evaluations"]), f"combined profiles were not deduplicated in {path}")
    for row in data["combined_candidate_evaluations"]:
        verify_profile_shape(row["profile"], units, ordered_unit_keys, f"combined {row['profile']['signature']}")
        verify_calibration_execution(row["calibration"], expected_cal, int(calibration["shard_count"]), f"combined {row['profile']['signature']}", calibration)
        check(int(row["profile"]["storage"]["total_bytes"]) * 8 == int(row["profile"]["storage"]["total_bits"]), f"non-integer candidate bytes in {path}")
        expected = expected_profile_storage(row["profile"], units, fixed_bits)
        for field in expected:
            check(int(row["profile"]["storage"][field]) == expected[field], f"candidate storage mismatch {field} in {path}")
        check(int(row["profile"]["storage_delta_vs_uniform_w8"]["total_bits"]) == int(row["profile"]["storage"]["total_bits"]) - int(uniform["8"]["total_bits"]), f"candidate storage delta mismatch in {path}")
    search_evidence_rows = [row for row in data["combined_candidate_evaluations"] if any(origin.startswith("interaction_search_round_") for origin in row["origins"])]
    check(len(search_evidence_rows) == int(search["new_unique_evaluations"]), f"search budget count does not match actual search-origin profiles in {path}")
    for history in search["rounds"]:
        check(history["ranking"].startswith("actual combined calibration"), f"search round did not use actual ranking in {path}")
        for signature in history["accepted_anchor_signatures"] + history["evaluated_signatures"]:
            check(signature in eval_by_signature, f"search references an unevaluated profile {signature} in {path}")

    frontier = data["heldout_frontier"]
    check(frontier, f"empty heldout frontier in {path}")
    frontier_signatures = set()
    for row in frontier:
        signature = row["profile"]["signature"]
        check(signature not in frontier_signatures, f"duplicate final frontier profile in {path}")
        check(signature in eval_by_signature, f"frontier profile has no combined calibration execution in {path}")
        check(row["calibration"] == eval_by_signature[signature]["calibration"], f"frontier calibration was not copied from the executed profile in {path}")
        verify_profile_shape(row["profile"], units, ordered_unit_keys, f"frontier {signature}")
        frontier_signatures.add(signature)
        expected = expected_profile_storage(row["profile"], units, fixed_bits)
        for field in expected:
            check(int(row["profile"]["storage"][field]) == expected[field], f"frontier storage mismatch in {path}")
        hold = row["heldout_vs_uniform_w8"]
        verify_heldout_execution(hold, held_ids, f"frontier {signature}")
        check(len(hold["per_sample"]) == expected_hold, f"frontier heldout count mismatch in {path}")
        paired = hold["paired_nll_difference_vs_uniform_w8"]
        check(paired["count"] == expected_hold and paired["ci95_low"] <= paired["mean"] <= paired["ci95_high"], f"missing paired heldout NLL bootstrap in {path}")
        check(set(hold["paired_metric_differences_vs_uniform_w8"]) >= {"nll_delta", "logit_mse", "kl_reference_to_candidate"}, f"paired metric evidence incomplete in {path}")

    gate = data["final_gate"]
    frontier_by_signature = {row["profile"]["signature"] for row in frontier}
    check(all(candidate["profile"]["signature"] in frontier_by_signature for candidate in gate["candidate_checks"]), f"gate references a profile outside the evaluated frontier in {path}")
    check(gate["decision"] in {"NO_GO_CLOSE_STRUCTURED_WEIGHT_PRECISION", "OPEN_8B_CONFIRMATION", "CONFIRMED_8B_REOPEN_NATIVE_KERNEL_GATE"}, f"invalid final gate decision in {path}")
    recomputed_eligible = []
    for candidate in gate["candidate_checks"]:
        paired = candidate["heldout_paired_nll"]
        gate_profile = next(row["profile"] for row in frontier if row["profile"]["signature"] == candidate["profile"]["signature"])
        storage_leq = int(gate_profile["storage"]["total_bits"]) <= int(gate["uniform_w8_storage_bits"])
        check(candidate["storage_leq_uniform_w8"] == storage_leq, f"gate storage flag is not computed from exact profile storage at {path}")
        positive = float(paired["mean"]) < 0.0 and float(paired["ci95_high"]) < 0.0
        eligible = bool(candidate["storage_leq_uniform_w8"] and positive and candidate["stable_calibration"])
        check(candidate["positive_confidence_signal"] == positive, f"gate confidence flag is not computed from paired NLL at {path}")
        check(candidate["eligible"] == eligible, f"gate eligibility flag is not computed from evidence at {path}")
        if eligible:
            recomputed_eligible.append(candidate["profile"]["signature"])
    check(bool(gate["gate_passed"]) == bool(recomputed_eligible), f"final gate boolean is inconsistent with candidate evidence in {path}")
    check({profile["signature"] for profile in gate["eligible_profiles"]} == set(recomputed_eligible), f"final gate eligible profiles are stale in {path}")
    if data.get("run_mode", "exploration") == "exploration":
        check(gate["one_b_gate_passed"] == (gate["decision"] == "OPEN_8B_CONFIRMATION"), f"1B gate decision mismatch in {path}")
        if gate["decision"] == "NO_GO_CLOSE_STRUCTURED_WEIGHT_PRECISION":
            check(gate["confirmation"]["opened"] is False, f"8B confirmation opened after a failed 1B gate in {path}")
            check("no 8B run" in gate["confirmation"]["reason"], f"failed 1B gate does not record the stopped 8B decision in {path}")
    else:
        check(gate["one_b_gate_passed"] is None, f"8B confirmation incorrectly reports a new 1B gate in {path}")
        check(gate["confirmation"]["opened"] is True, f"8B confirmation provenance missing in {path}")
        check(gate["decision"] in {"CONFIRMED_8B_REOPEN_NATIVE_KERNEL_GATE", "NO_GO_CLOSE_STRUCTURED_WEIGHT_PRECISION"}, f"invalid 8B confirmation decision in {path}")
    check(data["verification"]["heldout_not_used_for_search"] is True, f"heldout leakage flag in {path}")
    check(data["verification"]["no_8b_run_before_gate"] is True, f"8B ordering provenance missing in {path}")
    check(data["verification"]["final_gate_decision_recorded"] is True, f"final gate provenance missing in {path}")


SHAPES_FOR_KV = {
    "llama32_1b": {"num_kv_heads": 8, "head_dim": 64},
    "llama31_8b": {"num_kv_heads": 8, "head_dim": 128},
}


def expected_kv_page_bytes(shape: dict[str, int], page_format: str, group_size: int = 128) -> int:
    numel = 16 * int(shape["num_kv_heads"]) * int(shape["head_dim"])
    if page_format == "fp16":
        return numel * 4
    groups = math.ceil(numel / group_size)
    payload = numel * 2 if page_format == "int8" else math.ceil(numel / 2) * 2 if page_format == "int4" else None
    check(payload is not None, f"unknown KV page format {page_format}")
    return int(payload) + groups * 4


def verify_kv_provenance(data: dict, path: Path) -> None:
    source_files = tuple(data["provenance"]["source_files"])
    if source_files == KV_LEGACY_SOURCE_FILES:
        # These large legacy grids were executed before the optimized path was
        # added.  Keep them as historical controls, but bind them to their
        # recorded commit rather than silently accepting a stale current hash.
        check(data["provenance"].get("git_head") == LEGACY_KV_HEAD, f"legacy KV provenance commit mismatch in {path}")
        check(data["provenance"]["source_sha256"] == LEGACY_KV_SOURCE_SHA256, f"legacy KV source hash mismatch in {path}")
    else:
        check(source_files == KV_OPTIMIZED_SOURCE_FILES, f"KV source manifest mismatch in {path}")
        check(data["provenance"]["source_sha256"] == source_fingerprint(KV_OPTIMIZED_SOURCE_FILES), f"optimized KV artifact was not produced by the current source manifest in {path}")


def verify_kv_mechanism(path: Path) -> None:
    data = json.loads(path.read_text())
    check(data["schema"] == "live-kv-precision-v1", f"wrong KV mechanism schema in {path}")
    check(data["provenance"]["branch"] == "structured-precision-evidence", f"KV artifact branch mismatch in {path}")
    check(data["provenance"]["swiftllm_upstream_commit"] == COMMIT, f"KV SwiftLLM pin mismatch in {path}")
    verify_kv_provenance(data, path)
    check(data["scope"]["quantizer_is_new_contribution"] is False, f"new quantizer claimed in {path}")
    check(data["scope"]["scheduler_changed"] is False and data["scope"]["cpu_gpu_hierarchy_changed"] is False, f"forbidden serving scope changed in {path}")
    check(data["comparisons"]["queue_or_refuse_capacity"]["reclaimed_bytes"] == 0, f"queue baseline changed in {path}")
    check(data["comparisons"]["morphserve"]["superiority_claim"] is False, f"MorphServe overclaim in {path}")
    check(len(data["conversion"]) == 180, f"KV batch/context conversion grid incomplete in {path}")
    converted_total = 0
    for row in data["conversion"]:
        shape = SHAPES_FOR_KV[row["model_family"]]
        fp16_bytes = expected_kv_page_bytes(shape, "fp16")
        target_bytes = expected_kv_page_bytes(shape, row["target_format"])
        resident = int(row["resident_pages"])
        converted = int(row["converted_pages"])
        check(converted == round(resident * float(row["fraction"])), f"conversion count/fraction mismatch in {path}")
        check(int(row["before_logical_bytes"]) == resident * fp16_bytes, f"FP16 storage mismatch in {path}")
        expected_after = (resident - converted) * fp16_bytes + converted * target_bytes
        check(int(row["after_logical_bytes"]) == expected_after, f"compressed storage mismatch in {path}")
        check(int(row["reclaimed_bytes"]) == int(row["before_logical_bytes"]) - int(row["after_logical_bytes"]), f"reclaim arithmetic mismatch in {path}")
        raw = row["raw_page_conversions"]
        check(len(raw) == converted and len({(r["block_id"], r["layer_id"]) for r in raw}) == converted, f"raw conversion coverage mismatch in {path}")
        check(sum(int(r["reclaimed_bytes"]) for r in raw) == int(row["reclaimed_bytes"]), f"raw conversion total mismatch in {path}")
        for conversion in raw:
            check(int(conversion["before_bytes"]) == fp16_bytes and int(conversion["after_bytes"]) == target_bytes, f"raw page byte mismatch in {path}")
        counts = row["metadata_counts"]
        check(counts["k"].get(row["target_format"], 0) == converted and counts["v"].get(row["target_format"], 0) == converted, f"page metadata mismatch in {path}")
        check(row["has_unreclaimed_shadow"] is False, f"source shadow retained in {path}")
        check("before_torch_memory_allocated" in row and "after_torch_memory_allocated" in row, f"actual allocator measurements missing in {path}")
        if converted:
            check(int(row["after_torch_memory_allocated"]) < int(row["before_torch_memory_allocated"]), f"actual GPU allocation did not decrease in {path}")
            check(int(row["torch_allocated_delta"]) > 0, f"actual GPU reclaim is not positive in {path}")
        converted_total += converted
    check(converted_total > 0, f"no KV page was converted in {path}")

    expected_trial_keys = {f"batch{batch}_context{context}" for batch in (1, 4, 8) for context in (128, 512, 1024)}
    for family_result in data["families"]:
        shape = SHAPES_FOR_KV[family_result["model_family"]]
        check(set(family_result["trials"]) == expected_trial_keys, f"KV batch/context grid incomplete in {path}")
        for trial in family_result["trials"].values():
            records = [trial["fp16"], *trial["static"], *trial["mixed"]]
            for record in records:
                timing = record["attention"]
                check(len(timing["replicate_elapsed_ms"]) == int(timing["repetitions"]) == 3, f"attention repeats missing in {path}")
                check(float(timing["per_decode_ms_median"]) > 0 and int(timing["stored_payload_bytes_processed_per_iteration"]) > 0, f"attention measurement missing in {path}")
                storage = record["storage"]
                check(int(storage["allocated_payload_bytes_including_pending"]) == int(storage["logical_payload_bytes"]), f"attention record has pending source storage in {path}")
            for target in ("int8", "int4"):
                static = next(r for r in trial["static"] if r["target_format"] == target)
                check(static["storage"]["logical_payload_bytes"] == int(static["resident_pages"]) * expected_kv_page_bytes(shape, target), f"static {target} bytes mismatch in {path}")
                mixed = next(r for r in trial["mixed"] if r["target_format"] == target and r["compressed_fraction"] == 0.5)
                pages = int(mixed["resident_pages"])
                compressed = round(pages * 0.5)
                expected = (pages - compressed) * expected_kv_page_bytes(shape, "fp16") + compressed * expected_kv_page_bytes(shape, target)
                check(mixed["storage"]["logical_payload_bytes"] == expected, f"mixed {target} bytes mismatch in {path}")
    for row in data["overlap"]:
        check(row["supported"] is True and float(row["max_error_after_event_ordering"]) <= 1e-5, f"overlap correctness failed in {path}")
        check(row["pending_before_consumer_read"] is True and row["pending_after_consumer_read"] is True, f"consumer did not exercise pending event ordering in {path}")
        check(row["has_unreclaimed_shadow_after_sync"] is False, f"overlap source shadow retained in {path}")
        check(float(row["work_alone_ms"]) > 0 and float(row["overlap_wall_ms"]) > 0, f"overlap timing missing in {path}")
    check({row["target_format"] for row in data["overlap"]} == {"int8", "int4"}, f"overlap formats incomplete in {path}")


def verify_kv_quality(path: Path) -> None:
    data = json.loads(path.read_text())
    check(data["schema"] == "live-kv-precision-v1", f"wrong KV quality schema in {path}")
    check(data["provenance"]["swiftllm_upstream_commit"] == COMMIT, f"quality SwiftLLM pin mismatch in {path}")
    verify_kv_provenance(data, path)
    check(len(data.get("quality", [])) == 1, f"quality artifact should contain one run in {path}")
    run = data["quality"][0]
    check(Path(run["model_path"]).exists(), f"quality checkpoint unavailable in {path}")
    baseline = run["baseline"]
    check(baseline["kind"] == "unchanged_fp16_dense", f"quality baseline is not dense FP16 in {path}")
    generation_tokens = int(baseline["generation_tokens"])
    check(len(baseline["tokens"]) == generation_tokens and len(baseline["latency"]["decode_ms"]) == generation_tokens - 1, f"baseline trace incomplete in {path}")
    names = {variant["name"] for variant in run["variants"]}
    required = {"page_fp16_control", "static_int8", "static_int4", "dynamic_old_int4_50_both", "dynamic_recent_int4_50_both", "dynamic_old_int8_50_early", "dynamic_old_int8_50_late", "dynamic_old_int8_50_k_only", "dynamic_old_int8_50_v_only"}
    required |= {f"dynamic_{recency}_int8_{fraction}_both" for recency in ("old", "recent") for fraction in (25, 50, 75)}
    check(required <= names, f"quality policy grid incomplete in {path}")
    optimized_quality = tuple(data["provenance"]["source_files"]) == KV_OPTIMIZED_SOURCE_FILES
    for variant in run["variants"]:
        quality = variant["quality"]
        check(len(quality["per_step"]) == generation_tokens, f"quality per-step trace incomplete in {path}/{variant['name']}")
        agree = [bool(row["top1_agrees"]) for row in quality["per_step"]]
        close(float(quality["forced_prefix_top1_agreement"]), sum(agree) / len(agree), f"{path}/{variant['name']}/agreement")
        close(float(quality["baseline_token_nll_mean"]), sum(float(row["nll"]) for row in quality["per_step"]) / generation_tokens, f"{path}/{variant['name']}/nll")
        close(float(quality["reference_fp16_token_nll_mean"]), sum(float(row["reference_nll"]) for row in quality["per_step"]) / generation_tokens, f"{path}/{variant['name']}/reference_nll")
        close(float(quality["paired_nll_delta_mean"]), sum(float(row["nll_delta"]) for row in quality["per_step"]) / generation_tokens, f"{path}/{variant['name']}/nll_delta")
        check(len(variant["latency"]["decode_ms"]) == generation_tokens - 1, f"quality latency trace incomplete in {path}/{variant['name']}")
        if variant["name"] != "page_fp16_control":
            check("page_fp16_reference" in quality, f"quality is confounded by missing page FP16 control in {path}/{variant['name']}")
        storage = variant["storage"]
        if variant["name"] == "page_fp16_control":
            check(float(variant["fraction"]) == 0.0 and variant["target_format"] == "fp16", f"page FP16 control metadata is stale in {path}")
            check(storage["metadata_counts"]["k"].get("fp16") == storage["resident_pages"], f"page FP16 K control format mismatch in {path}")
            check(storage["metadata_counts"]["v"].get("fp16") == storage["resident_pages"], f"page FP16 V control format mismatch in {path}")
        if variant["name"].startswith("static_"):
            check(float(variant["fraction"]) == 1.0 and variant["demoted_pages"] == 0, f"static quality metadata is stale in {path}/{variant['name']}")
            check(storage["metadata_counts"]["k"].get(variant["target_format"]) == storage["resident_pages"], f"static K format mismatch in {path}/{variant['name']}")
            check(storage["metadata_counts"]["v"].get(variant["target_format"]) == storage["resident_pages"], f"static V format mismatch in {path}/{variant['name']}")
        check(storage["no_dense_fp16_shadow"] is True, f"quality source shadow retained in {path}/{variant['name']}")
        if variant["demoted_pages"]:
            check(len(variant["demotions"]) == variant["demoted_pages"], f"demotion trace incomplete in {path}/{variant['name']}")
            if optimized_quality and "int8" in variant["name"] and tuple(variant["components"]) == ("k", "v"):
                conversion = variant.get("conversion")
                check(conversion is not None and conversion["batched"] is True, f"optimized INT8 quality conversion was not batched in {path}/{variant['name']}")
                check(int(conversion["pages"]) == int(variant["demoted_pages"]), f"optimized INT8 conversion page count mismatch in {path}/{variant['name']}")
                check(int(conversion["reclaimed_bytes"]) > 0 and float(conversion["elapsed_ms"]) >= 0, f"optimized INT8 conversion metrics missing in {path}/{variant['name']}")
            for demotion in variant["demotions"]:
                check(int(demotion["result"]["before_bytes"]) > int(demotion["result"]["after_bytes"]), f"quality demotion did not reclaim bytes in {path}/{variant['name']}")
        else:
            check(variant["name"] in {"page_fp16_control", "static_int8", "static_int4"}, f"unexpected quality variant has no conversion in {path}/{variant['name']}")
    check(data["comparisons"]["queue_or_refuse_capacity"]["reclaimed_bytes"] == 0, f"quality queue comparison missing in {path}")
    check(data["comparisons"]["morphserve"]["superiority_claim"] is False, f"quality MorphServe comparison overclaims in {path}")


def verify_optimized_smoke(path: Path) -> None:
    data = json.loads(path.read_text())
    check(data["schema"] == "live-kv-optimized-int8-v1", f"wrong optimized smoke schema in {path}")
    provenance = data["provenance"]
    check(provenance["swiftllm_upstream_commit"] == COMMIT, f"optimized smoke SwiftLLM pin mismatch in {path}")
    check(tuple(provenance["source_files"]) == KV_OPTIMIZED_SOURCE_FILES, f"optimized smoke source manifest mismatch in {path}")
    check(provenance["source_sha256"] == source_fingerprint(KV_OPTIMIZED_SOURCE_FILES), f"optimized smoke source hash mismatch in {path}")
    check({int(row["model_shape"]["head_dim"]) for row in data["smoke"]} == {64, 128}, f"optimized smoke model-shape coverage missing in {path}")
    for row in data["smoke"]:
        conversion = row["batch_conversion"]
        check(conversion["target_format"] == "int8" and conversion["has_unreclaimed_shadow"] is False, f"optimized smoke conversion state missing in {path}")
        check(int(conversion["pages"]) == int(row["converted_pages"]) and int(conversion["reclaimed_bytes"]) > 0, f"optimized smoke conversion accounting missing in {path}")
        check(float(row["correctness"]["max_abs_error_vs_reference"]) <= 0.01 and row["correctness"]["optimized_has_nan"] is False, f"optimized smoke oracle check failed in {path}")
        check(float(row["attention"]["per_decode_ms_median"]) > 0, f"optimized smoke timing missing in {path}")


def verify_break_even(path: Path) -> None:
    data = json.loads(path.read_text())
    check(data["schema"] == "kv-memory-pressure-break-even-v1", f"wrong break-even schema in {path}")
    provenance = data["provenance"]
    check(provenance["swiftllm_upstream_commit"] == COMMIT, f"break-even SwiftLLM pin mismatch in {path}")
    check(tuple(BREAK_EVEN_SOURCE_FILES) == tuple(data["provenance"].get("source_files", BREAK_EVEN_SOURCE_FILES)), f"break-even source manifest missing in {path}")
    check(provenance["source_sha256"] == source_fingerprint(BREAK_EVEN_SOURCE_FILES), f"break-even source hash mismatch in {path}")
    scope = data["scope"]
    check(scope["scheduler_implemented"] is False and scope["scheduler_changed"] is False, f"break-even widened into scheduler work in {path}")
    check(scope["compression_format"].startswith("INT8"), f"break-even format is not INT8 in {path}")
    expected_cells = {(family, batch, context, fraction) for family in SHAPES_FOR_KV for batch in (1, 4, 8) for context in (128, 512, 2048) for fraction in (0.25, 0.5, 0.75)}
    actual_cells = {(row["model_family"], int(row["batch"]), int(row["context_tokens"]), float(row["compressed_fraction"])) for row in data["cells"]}
    check(actual_cells == expected_cells, f"break-even dimension grid incomplete in {path}")
    for row in data["cells"]:
        shape = SHAPES_FOR_KV[row["model_family"]]
        fraction = float(row["compressed_fraction"])
        pages = int(row["resident_pages"])
        selected = round(pages * fraction)
        check(int(row["selected_pages"]) == selected, f"selected page count mismatch in {path}")
        conversion = row["conversion"]
        check(conversion["batched_gpu_operation"] is True and int(conversion["pages"]) == selected, f"batched conversion evidence missing in {path}")
        fp16_bytes = expected_kv_page_bytes(shape, "fp16")
        int8_bytes = expected_kv_page_bytes(shape, "int8")
        check(int(conversion["before_bytes"]) == selected * fp16_bytes, f"conversion input bytes mismatch in {path}")
        check(int(conversion["after_bytes"]) == selected * int8_bytes, f"conversion output bytes mismatch in {path}")
        check(int(conversion["reclaimed_bytes"]) == int(conversion["before_bytes"]) - int(conversion["after_bytes"]), f"conversion reclaim arithmetic mismatch in {path}")
        check("gpu_memory_allocated_before_bytes" in conversion and "gpu_memory_allocated_after_bytes" in conversion, f"compression allocator observations missing in {path}")
        if selected:
            check(int(conversion["gpu_memory_allocated_delta_bytes"]) > 0, f"compression did not reclaim measured GPU allocation in {path}")
        memory = row["memory"]
        layers = int(shape["num_layers"]) if "num_layers" in shape else (16 if shape["head_dim"] == 64 else 32)
        check(int(memory["reclaimed_bytes_all_layers"]) == int(conversion["reclaimed_bytes"]) * layers, f"all-layer reclaim scaling mismatch in {path}")
        check(int(memory["capacity_bytes_avoided_or_reclaimed"]) == int(memory["reclaimed_bytes_all_layers"]), f"capacity accounting mismatch in {path}")
        restoration = row["restoration"]
        check(restoration["batched_gpu_operation"] is True and int(restoration["pages"]) == selected, f"restoration metadata malformed in {path}")
        check("gpu_memory_allocated_before_bytes" in restoration and "gpu_memory_allocated_after_bytes" in restoration, f"restoration allocator observations missing in {path}")
        offload = row["cpu_offload"]
        check(offload["mechanism"] == "swiftllm_c.swap_blocks" and offload["allocator_is_preallocated"] is True, f"existing CPU/GPU swap evidence missing in {path}")
        offload_pages = int(row["offload_selected_pages"])
        check(int(offload["selected_pages"]) == offload_pages, f"offload selected-page record mismatch in {path}")
        check(int(offload["bytes_reclaimed_as_reusable_capacity"]) == offload_pages * fp16_bytes * layers, f"offload capacity mismatch in {path}")
        check(int(offload["bytes_reclaimed_as_reusable_capacity"]) >= int(memory["target_deficit_bytes"]), f"offload did not meet matched deficit in {path}")
        check(int(offload["bytes_reclaimed_as_reusable_capacity"]) - int(memory["target_deficit_bytes"]) < fp16_bytes * layers, f"offload page-granularity overshoot is too large in {path}")
        check("physical_hbm_delta_bytes" in offload, f"physical HBM offload observation missing in {path}")
        actions = row["actions"]
        check(set(actions) == {"queue", "cpu_offload", "int8_compression"}, f"action records incomplete in {path}")
        check(float(actions["queue"]["transition_latency_ms"]) == 0.0 and float(actions["queue"]["quality_change"]) == 0.0, f"queue action record malformed in {path}")
        check(int(actions["queue"]["capacity_bytes_avoided"]) == int(memory["capacity_bytes_avoided_or_reclaimed"]), f"queue capacity record mismatch in {path}")
        check(int(actions["cpu_offload"]["capacity_bytes_reclaimed"]) == int(offload["bytes_reclaimed_as_reusable_capacity"]), f"offload action record mismatch in {path}")
        check(float(actions["int8_compression"]["transition_latency_ms"]) == float(conversion["elapsed_ms"]), f"compression transition record mismatch in {path}")
        check(int(actions["int8_compression"]["capacity_bytes_reclaimed"]) == int(memory["reclaimed_bytes_all_layers"]), f"compression capacity record mismatch in {path}")
        check(float(actions["int8_compression"]["persistent_per_token_ms"]) > 0, f"compression persistent timing missing in {path}")
        check(row["quality_delta"]["int8_compression"] is not None and 0.0 <= float(row["quality_delta"]["top1_agreement"]) <= 1.0, f"compression quality probe missing in {path}")
        check(float(row["quality_delta"]["queue"]) == 0.0 and float(row["quality_delta"]["cpu_offload"]) == 0.0, f"lossless action quality records missing in {path}")
        check(float(row["oracle"]["max_abs_error_vs_optimized"]) <= 0.01, f"optimized attention oracle mismatch in {path}")
        baseline = row["baseline_dense_fp16"]
        fast = row["optimized_mixed_int8"]
        check(float(baseline["per_token_wall_ms"]) > 0 and float(fast["per_token_wall_ms"]) > 0, f"attention timing missing in {path}")
        costs = {int(r["horizon_decode_iterations"]): r for r in row["break_even"]}
        check(set(costs) == set((1, 4, 8, 16, 32, 64)), f"horizon grid incomplete in {path}")
        for horizon, costs_row in costs.items():
            expected_queue = horizon * float(baseline["per_token_wall_ms"])
            expected_offload = float(offload["transition_wall_ms"])
            expected_compression = float(conversion["elapsed_ms"]) + float(restoration["elapsed_ms"]) + horizon * (float(fast["per_token_wall_ms"]) - float(baseline["per_token_wall_ms"]))
            close(float(costs_row["queue_cost_ms"]), expected_queue, f"{path}/queue/{horizon}")
            close(float(costs_row["offload_cost_ms"]), expected_offload, f"{path}/offload/{horizon}")
            close(float(costs_row["compression_cost_ms"]), expected_compression, f"{path}/compression/{horizon}")
            check(costs_row["winner"] in {"queue", "cpu_offload", "int8_compression"}, f"invalid break-even winner in {path}")


def main() -> None:
    check((ROOT / "vendor/swiftLLM/UPSTREAM_COMMIT").read_text().strip() == COMMIT, "research marker is not pinned")
    check((ROOT / "vendor/swiftLLM-upstream/UPSTREAM_COMMIT").read_text().strip() == COMMIT, "clean marker is not pinned")
    check((ROOT / "vendor/qaq/UPSTREAM_COMMIT").read_text().strip() == QAQ_COMMIT, "QAQ marker is not pinned")
    provenance = json.loads((ROOT / "results/baseline/research-provenance.json").read_text())
    check(provenance["swiftllm_upstream_commit"] == COMMIT, "research provenance commit is wrong")
    check(provenance["swiftllm_research_diff_sha256"] == file_sha256(ROOT / "references/swiftllm-research.diff"), "research provenance diff is stale")
    check(provenance["requirements_lock_sha256"] == file_sha256(ROOT / "requirements-lock.txt"), "requirements lock provenance is stale")
    check(normalized_log(ROOT / "results/baseline/swiftllm_unmodified_1b.log") == normalized_log(ROOT / "results/baseline/swiftllm_precision_noop_1b.log"), "baseline/no-op outputs differ")
    check(normalized_log(ROOT / "results/baseline/swiftllm_unmodified_1b.log") == normalized_log(ROOT / "results/baseline/swiftllm_precision_noop_kv_phase.log"), "current KV-phase default outputs differ")
    verify_matrix(ROOT / "results/sensitivity/llama32_1b_qkv_matrix.json", 8)
    verify_matrix(ROOT / "results/sensitivity/llama31_8b_qkv_matrix.json", 8)
    verify_matrix(ROOT / "results/sensitivity/llama32_1b_qkv_matrix_layer_sweep.json", 8, 16)
    verify_matrix(ROOT / "results/sensitivity/llama31_8b_qkv_matrix_layer_sweep_4prompts.json", 4, 32)
    verify_interaction_aware(ROOT / "results/sensitivity/llama32_1b_interaction_aware.json")
    verify_interaction_aware(ROOT / "results/sensitivity/llama31_8b_interaction_aware.json")
    verify_kv_mechanism(ROOT / "results/sensitivity/kv_precision_mechanism.json")
    verify_kv_quality(ROOT / "results/sensitivity/kv_precision_quality_1b.json")
    verify_kv_quality(ROOT / "results/sensitivity/kv_precision_quality_8b.json")
    verify_kv_quality(ROOT / "results/sensitivity/kv_precision_quality_batched_1b.json")
    verify_kv_quality(ROOT / "results/sensitivity/kv_precision_quality_batched_8b.json")
    verify_optimized_smoke(ROOT / "results/sensitivity/kv_precision_optimized_smoke_final.json")
    verify_break_even(ROOT / "results/sensitivity/kv_break_even_study.json")
    invocations = json.loads((ROOT / "results/baseline/invocations.json").read_text())["commands"]
    invocation_names = {row["name"] for row in invocations}
    check("interaction_aware_structured_llama32_1b" in invocation_names, "1B interaction-aware invocation is missing")
    check("gated_interaction_aware_confirmation_llama31_8b" in invocation_names, "8B gated confirmation invocation is missing")
    check("artifact_verifier" in invocation_names and "unit_tests" in invocation_names, "current verification invocations are missing")
    for required in (
        "docs/baseline.md",
        "docs/claim-evidence.md",
        "docs/feasibility-report.md",
        "docs/completion-audit.md",
        "docs/papers.md",
        "docs/source-map.md",
        "references/qaq-2403.04643.pdf",
        "references/morphserve-2506.02006-v2.pdf",
        "references/swiftllm-research.diff",
        "references/swiftllm-kv-page.diff",
        "requirements-lock.txt",
        "results/baseline/research-provenance.json",
        "results/baseline/unit-tests-kv-phase.log",
        "results/baseline/artifact-verification-kv-phase.log",
        "results/baseline/noop-output-comparison-kv-phase.txt",
        "results/baseline/swiftllm_precision_noop_kv_phase.log",
        "results/sensitivity/kv_precision_mechanism.json",
        "results/sensitivity/kv_precision_mechanism.log",
        "results/sensitivity/kv_precision_quality_1b.json",
        "results/sensitivity/kv_precision_quality_1b.log",
        "results/sensitivity/kv_precision_quality_8b.json",
        "results/sensitivity/kv_precision_quality_8b.log",
        "results/sensitivity/kv_precision_quality_batched_1b.json",
        "results/sensitivity/kv_precision_quality_batched_8b.json",
        "results/sensitivity/kv_precision_optimized_smoke_final.json",
        "results/sensitivity/kv_break_even_study.json",
    ):
        check((ROOT / required).exists(), f"missing required artifact: {required}")
    print("artifact verification passed: historical structured gate plus live-KV page storage, conversion, mixed attention, overlap, and quality traces verified")


if __name__ == "__main__":
    main()
