# SwiftLLM FP16 vs Transformers FP16 diagnosis

**Status:** fixed for the tested local Llama 3.2 1B Instruct and Llama 3.1 8B checkpoints; capacity-study follow-up remains required.

## Scope and method

The comparison used the local checkpoints below, the same tokenizer-produced token IDs, FP16 weights, and RTX 3090 CUDA runs:

- Llama 3.2 1B Instruct: snapshot `9213176726f574b556790deb65791e0c5aa438b6`
- Llama 3.1 8B: snapshot `d04e592bb4f6aa9cfee91e2e20afa771667e1d4b`

`Transformers` used the eager Llama implementation as the numerical reference. SwiftLLM used its unchanged FP16 projection path and vLLM FlashAttention prefill path. The diagnostic records embedding, layer input, attention RMSNorm, raw Q/K/V, post-RoPE Q/K, attention context, O projection, FFN norm/activation/output, layer output, final norm, and logits. It also records GQA shapes, positions, causal masking, and all loaded weight comparisons.

Artifacts:

- `results/fp16-diagnosis/pre-fix-1b-baseline.json`
- `results/fp16-diagnosis/pre-fix-1b.json` (post-RoPE intervention)
- `results/fp16-diagnosis/post-fix-1b.json`
- `results/fp16-diagnosis/post-fix-1b-long.json`
- `results/fp16-diagnosis/post-fix-8b-v2.json`
- `results/fp16-diagnosis/decode-1b.json`
- `results/fp16-diagnosis/decode-8b.json`
- `results/fp16-diagnosis/capacity-regression-8b.json`

The reproducible tools are `scripts/swiftllm_fp16_diagnosis.py` and `scripts/swiftllm_fp16_decode_check.py`.

## Confirmed findings

1. **The earliest material 1B divergence was incorrect Llama 3 RoPE construction.**
   Before the fix, embedding was exact, attention RMSNorm and raw Q/K/V differed only by ordinary FP16 kernel rounding, then layer 0 post-RoPE Q/K diverged sharply: Q max absolute error `2.2080`, K `1.2988`, with relative L2 errors `0.0599` and `0.0561`. The old SwiftLLM code split frequencies at one arbitrary dimension boundary and applied two constant position scales. Transformers applies Llama 3 wavelength cutoffs plus smooth interpolation.

2. **The RoPE fix is exact at the cached table level.**
   The corrected implementation in `vendor/swiftLLM/swiftllm/worker/model.py` follows Transformers' Llama 3 inverse-frequency formula, computes in FP32, and casts the table to FP16. In both post-fix model artifacts, `rope.cos_half` and `rope.sin_half` have zero error against Transformers.

3. **The intervention is causal evidence, not just correlation.**
   In the pre-fix 1B run, replacing SwiftLLM's post-RoPE Q/K at every layer with the matched Transformers tensors changed final-logit max error from `0.25` to `0.015625` and relative L2 from `0.01562` to `0.000794`. The intervention is recorded in `pre-fix-1b.json`; the no-intervention control is `pre-fix-1b-baseline.json`.

4. **A separate Llama 3.1 8B loader bug was confirmed.**
   The old auto-detection treated any dictionary-valued `rope_scaling` as “llama3.2” and therefore loaded the tied embedding matrix as `lm_head` even for Llama 3.1 8B, whose config has `tie_word_embeddings: false`. The pre-fix 8B trace had exact hidden-state agreement through the layers but one weight-audit mismatch (`lm_head`), then final logits max error `18.1772` and a wrong top-1 token. `infer_model_version()` now uses `tie_word_embeddings`, and the post-fix audit checks 259/259 tensors exactly.

5. **The fixed FP16 path is numerically aligned within a measured FP16 envelope.**
   The envelope used here is finite tensors, relative L2 below `0.005`, and matching top-1 prediction; final-logit max error is also reported rather than hidden. Post-fix results:

   | checkpoint / probe | final-logit max abs | relative L2 | top-1 |
   |---|---:|---:|---|
   | 1B, 9-token prefill | 0.0195 | 0.00115 | match |
   | 1B, 47-token prefill | 0.0156 | 0.00106 | match |
   | 8B, 9-token prefill | 0.0234 | 0.00204 | match |
   | 1B, decode after 9-token prefill | 0.0195 | 0.00112 | match |
   | 8B, decode after 9-token prefill | 0.0234 | 0.00107 | match |

   Later intermediate max errors are scale-dependent (up to `0.125` for 1B and `0.25` for 8B), while their relative L2 errors stay below `0.005`; this is why the report does not impose a scale-independent absolute bound on every hidden tensor.

6. **Weights, GQA, positions, masking, and KV layout were checked.**

   - All 131 1B and all 259 8B loaded tensors, including concatenated `[up, gate]` FFN storage, matched the safetensor source exactly.
   - The traces report 32 Q heads / 8 KV heads for both checkpoints, with head dimensions 64 / 128 respectively; the post-RoPE Q/K shapes agree.
   - Position IDs are the matched sequence positions `0..N-1`; the decode check uses the next position `N`.
   - Transformers eager attention reports maximum future causal attention weight `0.0`; SwiftLLM invokes FlashAttention with `causal=True`.
   - The decode check reconstructs SwiftLLM's physical `[block, layer, KV-head, offset, head-dim]` cache into logical `[layer, KV-head, token, head-dim]`. Maximum K/V differences are `0.02344/0.00513` for 1B and `0.01563/0.00500` for 8B. Prefill and one cached decode step both match top-1 predictions.

## Fixes made

- `vendor/swiftLLM/swiftllm/worker/model.py`: replace the arbitrary Llama 3 RoPE split with the Transformers wavelength-cutoff and smooth-interpolation formula; use the configured maximum position range.
- `vendor/swiftLLM/swiftllm/worker/weight.py`: select the Llama 3.2 tied-embedding convention from `tie_word_embeddings`, not from the presence of `rope_scaling`.
- `tests/test_precision.py`: add a regression test for Llama 3.1 versus 3.2 loader selection.
- New diagnostic scripts and artifacts listed above.

No quantization, mixed-precision policy, scheduler, or offload behavior was changed.

## Verification and regression evidence

Passed:

```text
29 repository unit tests: OK
py_compile for changed/source diagnostic files: OK
git diff --check: OK
scripts/verify_artifacts.py: passed
scripts/verify_public_kv.py: passed
```

Because `model.py` is part of the KV quality provenance, the four affected quality artifacts were rerun with their recorded invocations (`kv_precision_quality_1b.json`, `kv_precision_quality_8b.json`, and both batched artifacts), and the optimized smoke plus break-even artifacts were regenerated. `verify_artifacts.py` then passed.

A small post-fix 8B RTX 3090 capacity/timing regression (`capacity-regression-8b.json`) completed at context 2048, decode 16, batch 1, dense FP16 SwiftLLM: warm prefill `486--488 ms`, warm decode `344--345 ms`, peak reserved `16.68 GB` versus the declared `22.77 GB` budget, with 129 FP16 K/V pages.

## Supported but uncertain findings

- The two fixes are sufficient for the tested 1B/8B checkpoints, the tested prefill lengths, and one cached decode step. They do not prove every long-context position or every batch/interleaving path.
- The diagnostic uses one request for the intermediate comparison. It checks GQA and a physical KV reconstruction, but it does not exhaustively compare multi-request block-table permutations.
- The measured FP16 envelope is a practical RTX 3090 / PyTorch 2.4 / Triton-vLLM-kernel envelope, not a mathematical guarantee for every GPU, driver, kernel, or prompt.

## Blocked questions

- `scripts/verify_kv_capacity_v2.py` could not be accepted as a green gate: the current working tree contains additional `results/kv-capacity-v2/*-current` timing and memory files whose count is not reflected in the checked-in `summary.json` (`summary attempted_cell_count=214`), so it fails at “summary manifest count mismatch.” This is an artifact-set/manifest blocker, not an FP16 numerical failure.
- A full post-fix rerun of every historical capacity cell was not performed. The capacity regression is evidence that the corrected path runs under budget, not a replacement capacity curve.

## Remaining uncertainty and capacity impact

SwiftLLM is now a valid **FP16 numerical baseline** for the tested local Llama 3.2 1B and Llama 3.1 8B configurations, subject to the scope above. The prior capacity conclusions that directly use SwiftLLM FP16 timings or its 8B FP16 batch boundaries should be treated as provisional until the capacity matrix is rerun against this corrected source. The memory arithmetic itself is unchanged, but timing, output correctness, provenance hashes, and any conclusion comparing a compressed method against SwiftLLM FP16 need refreshed evidence. The KIVI quantization implementation, scheduler, and offload claims were not changed by this patch.

The prior capacity report must therefore not be cited as a fully verified corrected-baseline result until its current-file manifest is reconciled and at least the primary SwiftLLM FP16 boundary cells are rerun.

## Prompt-to-artifact completion checklist

| Requirement | Evidence | State |
|---|---|---|
| Compare matched token IDs on local checkpoints | `token_ids`, model paths, and tokenizer metadata in all diagnosis JSON | confirmed |
| Locate earliest divergence through embedding/layers/QKV/attention/FFN/final norm/logits | ordered `comparison.rows` in `pre-fix-1b-baseline.json` and post-fix artifacts | confirmed |
| Check loading/RoPE/GQA/KV/positions/masking | exact weight audit, RoPE table rows, GQA metadata, decode KV rows, position/masking metadata | confirmed for tested scope |
| Replace suspect intermediate and test later realignment | pre-fix no-op versus post-RoPE intervention artifacts | confirmed |
| Apply only evidence-backed fixes | two source diffs above; no quantization/scheduler/offload edits | confirmed |
| Verify FP16 logits and token predictions | prefill/decode artifacts for 1B and 8B plus long 1B probe | confirmed for tested scope |
| Rerun affected quality tests and verifiers | regenerated quality/smoke/break-even artifacts; `verify_artifacts.py` and public verifier passed | confirmed |
| Run repository tests and static checks | 29 tests, pycompile, diff check | confirmed |
| Small timing/capacity regression | `capacity-regression-8b.json` | confirmed |
| State previous capacity impact and uncertainty | this report and blocked capacity verifier result | confirmed |
