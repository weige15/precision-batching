"""Validate the checked-in public low-bit KV reproduction artifacts."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def commit(path: str) -> str:
    return subprocess.check_output(["git", "-C", str(ROOT / path), "rev-parse", "HEAD"], text=True).strip()


def load(path: str):
    return json.loads((ROOT / path).read_text())


def main() -> None:
    manifest = load("results/public/source_manifest.json")
    expected = {entry["name"]: entry for entry in manifest["implementations"] if entry["commit"]}
    for name, entry in expected.items():
        if name == "QAQ":
            actual = (ROOT / "vendor/qaq/UPSTREAM_COMMIT").read_text().strip()
        else:
            actual = commit({"KIVI": "vendor/public/KIVI", "KVQuant": "vendor/public/KVQuant", "Kitty": "vendor/public/Kitty"}[name])
        assert actual == entry["commit"], (name, actual, entry["commit"])

    kivi = [load(path) for path in (
        "results/public/kivi_llama32_1b_2bit.json",
        "results/public/kivi_llama32_1b_4bit.json",
        "results/public/kivi_llama32_1b_2bit_ctx256.json",
        "results/public/kivi_llama32_1b_batch4_ctx256_2bit.json",
        "results/public/kivi_llama31_8b_2bit.json",
    )]
    assert all(x["method"] == "KIVI" for x in kivi)
    assert all(x["model_config"]["gqa_groups"] == 4 for x in kivi)
    assert all(x["summary"]["cache_reduction_ratio"] > 0 for x in kivi)
    assert all(x["summary"]["decode_slowdown"] > 1 for x in kivi)
    assert {x["kivi_config"]["k_bits"] for x in kivi} == {2, 4}

    kitty = [load(path) for path in (
        "results/public/kitty_llama_gqa_kernel.json",
        "results/public/kitty_llama_gqa_kernel_ctx512.json",
    )]
    assert all(x["shape"]["gqa_groups"] == 4 for x in kitty)
    assert all(x["summary"]["finite_output"] for x in kitty)
    assert all(x["storage"]["capacity_reduction_ratio"] > 0 for x in kitty)
    assert all(x["summary"]["mean_abs_error_vs_unquantized"] >= 0 for x in kitty)

    assert "num_key_value_groups == 1" in (ROOT / "vendor/public/KVQuant/deployment/transformers/src/transformers/models/llama/modeling_llama.py").read_text()
    assert "dequantized_cache" in (ROOT / "vendor/qaq/src/quantizer.py").read_text()
    assert not (ROOT / "docs/public-kv-study.md").read_text().count("Minima-KV") < 3
    print("public low-bit KV artifact verification passed")


if __name__ == "__main__":
    main()
