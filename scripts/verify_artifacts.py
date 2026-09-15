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
    check(current_head == provenance["commit_at_run"] == BASE_EXPERIMENT_COMMIT, f"artifact commit does not match requested checkout in {path}")
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


def main() -> None:
    check((ROOT / "vendor/swiftLLM/UPSTREAM_COMMIT").read_text().strip() == COMMIT, "research marker is not pinned")
    check((ROOT / "vendor/swiftLLM-upstream/UPSTREAM_COMMIT").read_text().strip() == COMMIT, "clean marker is not pinned")
    check((ROOT / "vendor/qaq/UPSTREAM_COMMIT").read_text().strip() == QAQ_COMMIT, "QAQ marker is not pinned")
    provenance = json.loads((ROOT / "results/baseline/research-provenance.json").read_text())
    check(provenance["swiftllm_upstream_commit"] == COMMIT, "research provenance commit is wrong")
    check(provenance["swiftllm_research_diff_sha256"] == file_sha256(ROOT / "references/swiftllm-research.diff"), "research provenance diff is stale")
    check(provenance["requirements_lock_sha256"] == file_sha256(ROOT / "requirements-lock.txt"), "requirements lock provenance is stale")
    check(normalized_log(ROOT / "results/baseline/swiftllm_unmodified_1b.log") == normalized_log(ROOT / "results/baseline/swiftllm_precision_noop_1b.log"), "baseline/no-op outputs differ")
    verify_matrix(ROOT / "results/sensitivity/llama32_1b_qkv_matrix.json", 8)
    verify_matrix(ROOT / "results/sensitivity/llama31_8b_qkv_matrix.json", 8)
    verify_matrix(ROOT / "results/sensitivity/llama32_1b_qkv_matrix_layer_sweep.json", 8, 16)
    verify_matrix(ROOT / "results/sensitivity/llama31_8b_qkv_matrix_layer_sweep_4prompts.json", 4, 32)
    verify_interaction_aware(ROOT / "results/sensitivity/llama32_1b_interaction_aware.json")
    verify_interaction_aware(ROOT / "results/sensitivity/llama31_8b_interaction_aware.json")
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
        "requirements-lock.txt",
        "results/baseline/research-provenance.json",
    ):
        check((ROOT / required).exists(), f"missing required artifact: {required}")
    print("artifact verification passed: W8-centered/confirmation evidence, combined search, 1B 243 projection runs, heldout separation, and final gates verified")


if __name__ == "__main__":
    main()
