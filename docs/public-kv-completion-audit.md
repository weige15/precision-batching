# Completion audit: public low-bit KV reproduction objective

This audit is against the active public-implementation reproduction objective,
not against the historical SwiftLLM structured-weight objective. It was run in
the isolated `low-bit-kv-reproduction` worktree.

## Concrete deliverables

1. Exact public source/repository/commit provenance for KIVI, KVQuant, QAQ,
   Kitty, and Minima-KV classification.
2. Per-method availability, completeness, RTX 3090/Ampere support,
   Llama-family/GQA support, actual low-bit layout, and optimized decode-path
   determination.
3. Local reproduction evidence for viable methods: resident KV bytes,
   latency/throughput against FP16, quality, and recent-high/old-low behavior.
4. Explicit portability adaptations and exact failure/block reasons.
5. Secondary interpretation against completed SwiftLLM FP16/KV reference data.
6. Integration recommendation naming a method, precision, restrictions, code,
   commit, and minimum adaptation.
7. Final confirmed/approximate/failed/blocked/uncertain categories.

## Prompt-to-artifact checklist

| Requirement | Evidence inspected | Audit result |
|---|---|---|
| Separate branch/worktree and GPU/output isolation | `git worktree list`; branch `low-bit-kv-reproduction`; all new artifacts under `results/public/`; original worktree remains dirty but untouched | Pass |
| Exact upstream repository and commit preserved | `.gitmodules`, `git submodule status`, `results/public/source_manifest.json`; QAQ `vendor/qaq/UPSTREAM_COMMIT` | Pass: KIVI `876b4d2`, KVQuant `57a2383`, Kitty `dfd2c07`, QAQ `f8d47e0`; Minima explicitly unresolved |
| Public/complete-enough status for each named method | `docs/public-kv-study.md` source audit; submodule trees; `results/public/minima_kv_search_summary.json` | Pass with QAQ/Minima blocked classifications |
| Ampere/RTX 3090 support | `results/public/kivi_build.log` includes `-gencode=arch=compute_86,code=sm_86`; `results/public/kvquant_build.log`; Kitty kernel run JSON device/capability | Pass for KIVI/Kitty; KVQuant compile pass but model blocked; QAQ no optimized CUDA path |
| Llama-family shape and GQA support | KIVI JSON model configs (32 Q/8 KV for both checkpoints); Kitty JSON shape (32/8/D64); KVQuant source assertion/log | Pass for KIVI; kernel-only Kitty; KVQuant blocked; QAQ not established |
| Actual precision/layout recorded | source manifest and report; KIVI JSON cache layer shapes; Kitty cache tensor bytes; source paths | Pass |
| Actual optimized decode path distinguished from offline quantization | KIVI source `models/llama_kivi.py`, `quant/new_pack.py`, `quant/matmul.py`, CUDA build/run; Kitty `kitty_attention.py`; KVQuant `quant_cuda`; QAQ `quantizer.py`/`evaluator.py` | Pass |
| Actual GPU KV-memory reduction | KIVI JSON resident cache tensor-byte sums for 1B/8B, K2/K4, batch 1/4; Kitty JSON static tensor capacity sums | Pass, with allocator/workspace caveat explicitly reported |
| Decode latency/throughput vs method FP16 baseline | KIVI matched HF FP16 synchronized wall-clock samples; Kitty FP16 SDPA CUDA-event samples | Pass for viable local paths; no KVQuant E2E due blocker |
| Quality degradation | KIVI paired teacher-forced NLL/top-1 for 1B context128/256 and 8B context128; Kitty fused-output error | Pass as controlled probes; broad task/perplexity remains open |
| Recent-high/old-low behavior | KIVI residual FP16 window plus old packed state; Kitty FP16 sink/local/query buffers; source layouts | Pass structurally; no SwiftLLM live-demotion transition claimed |
| Author kernels/layouts used | KIVI public CUDA/Triton paths directly invoked; Kitty public Triton quant-pack/qk/sv directly invoked; no replacement low-bit kernel | Pass |
| Every compatibility patch documented | `scripts/reproduce_kivi.py` compatibility metadata; report shim and model/submodule/dependency boundaries; KVQuant logs | Pass |
| Matched SwiftLLM reference comparison | report cites prior batch8/context1024 SwiftLLM raw/reference results and bounds the cross-system comparison | Pass as secondary, non-apples-to-apples evidence |
| Determine whether 3.8x/6.8x is reference-path-specific | KIVI optimized E2E ~1.84--2.02x; Kitty kernel 4.35--4.45x; prior SwiftLLM reference values | Supported conclusion: not intrinsic, but not proven entirely artifact |
| No forbidden scheduler/new kernel/policy work | `git diff`; new files are probes/report/verifier; source method kernels are upstream submodules | Pass |
| Integration recommendation | report recommendation: KIVI K2/V2 first, Kitty alternative, KVQuant GQA port blocked, QAQ/Minima excluded | Pass |
| Final evidence categories and uncertainty | report sections `Confirmed`, `Approximate`, `Failed`, `Blocked`, `Remaining uncertainty` | Pass |

## Commands and observed verification

Commands run in the isolated worktree:

```bash
git submodule status
.venv/bin/python scripts/verify_public_kv.py
.venv/bin/python -m py_compile scripts/reproduce_kivi.py scripts/probe_kitty.py scripts/verify_public_kv.py
PYTHONPATH="$PWD/vendor/swiftLLM:$PWD/vendor/swiftLLM/csrc" \
  .venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/verify_artifacts.py
```

Observed:

- `verify_public_kv.py`: passed.
- Python compilation: passed.
- Repository tests: **19 passed** after building the ordinary SwiftLLM C++
  extension required by imports.
- The historical `verify_artifacts.py` intentionally failed its branch
  provenance check because historical structured artifacts say
  `structured-precision-evidence`, while this isolated study is on
  `low-bit-kv-reproduction`. This is not evidence against the public study;
  it is recorded in `results/public/validation_summary.txt` rather than
  bypassed or relabeled.

The public study's own verifier recomputes source commit pins, GQA coverage,
positive resident-cache reduction, optimized-path result presence, Kitty finite
outputs, and the source-level KVQuant/QAQ boundaries. Raw timing, cache shape,
quality, build, and block logs remain alongside the JSON artifacts.

## Missing or deliberately bounded requirements

- No public implementation in this set provides a drop-in SwiftLLM paged
  adapter. KIVI's minimum adapter is described but not implemented, as required
  by the reproduction-only boundary.
- KIVI quality evidence is short, one-prompt, paired forced-prefix evidence;
  no broad task or perplexity claim is made.
- Kitty is kernel-only for local Llama shapes because its pinned model wrapper
  is Qwen3-only.
- KVQuant's MHA-only model seam and old dependency contract prevent a faithful
  local Llama 3.x GQA run; removing that assertion would be a portability port,
  not a faithful reproduction.
- Minima-KV remains unidentified; a canonical author URL/commit is required
  before further work.
- Cross-system batch/context equivalence is limited: KIVI has batch 1 and batch
  4 context-256 probes, while the prior SwiftLLM stress reference is batch 8
  context 1024. Therefore no exact apples-to-apples speed claim is made.

These are explicitly reported uncertainties/blockers, not silently treated as
achieved. The evidence is sufficient for the requested foundation choice and
for rejecting the current transparent SwiftLLM mixed-reference path as a
universal low-bit performance proxy. It is not sufficient to approve a serving
scheduler or production integration.
