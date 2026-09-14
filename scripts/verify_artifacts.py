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
