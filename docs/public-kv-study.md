# Public low-bit KV-cache reproduction and portability study

## Decision and scope

**Best locally working foundation: KIVI at its upstream 2-bit/4-bit KV path.**
It is the only candidate in this study that ran end to end with both local
Llama 3.2 1B and Llama 3.1 8B GQA checkpoints on an RTX 3090, while retaining
packed low-bit KV state and using an optimized CUDA/Triton decode path. It is
not a drop-in SwiftLLM page kernel: its cache is a contiguous per-layer legacy
layout with a residual FP16 window, K per-channel packing, and V per-token
packing.

This study is on the separate `low-bit-kv-reproduction` worktree/branch. It
does not modify the concurrent memory-reclamation worktree, its uncommitted
files, outputs, or benchmark artifacts. The public source checkouts are pinned
as submodules in `vendor/public/`; exact provenance is in
[`results/public/source_manifest.json`](../results/public/source_manifest.json).

The question is reproduction and portability, not new quantizer or kernel
design. No scheduler, router, memory-pressure controller, CPU swap policy,
MorphServe reproduction, or new low-bit kernel was implemented.

## Source and implementation audit

| Method | Public pinned source | Local model/GQA support | Actual layout | Optimized decode path | Local classification |
|---|---|---|---|---|---|
| **KIVI** | `jy-yuan/KIVI`, `876b4d2d08e3b1d5f70d0969c299d8c7c42ddfb6` | **Confirmed**: Llama 3.2 1B and Llama 3.1 8B, both 32 Q / 8 KV heads | K per-channel over token groups; V per-token over head groups; packed int32 codes, FP16 scale/min metadata, FP16 residual window; 2/4-bit | **Confirmed**: upstream Triton packing plus `quant/kivi_gemv` CUDA GEMV consumed by incremental decode | End-to-end reproduction, with explicit Transformers/flash-attention portability shim |
| **KVQuant** | `SqueezeAILab/KVQuant`, `57a238357f0ffe50084670fcd5781c9848f80ea2`, paper deployment tree | **Blocked for local Llama 3.x GQA**: source asserts `num_key_value_groups == 1` at `deployment/transformers/src/transformers/models/llama/modeling_llama.py:1408`; local config is 32/8 | Packed 2/3/4-bit K/V state with NUQ lookup tables and optional sparse outliers | **Confirmed in source**: `quant_cuda` fused append and K/Q or attention matmul kernels | CUDA build succeeded for `sm_86`; model reproduction blocked by GQA and old custom-Transformers dependency |
| **Kitty** | `Summer-Summer/Kitty`, `dfd2c07b407d6b407179359207c612ab631f3ed1` | Kernel-level GQA portability confirmed with Llama 3.2 dimensions; **end-to-end Llama integration absent** in this commit (current wrapper is Qwen3; required submodules are not initialized) | 128-token pages, uint8 payloads, FP16 metadata, FP16 sink/local buffers; K2/V2 with 25% K channels boosted to K4 | **Confirmed at kernel level**: upstream Triton quantize-pack and fused qk/sv attention | Synthetic GQA kernel reproduction; not an end-to-end Llama claim |
| **QAQ** | `ClubieDong/QAQ-KVCacheQuantization`, `f8d47e0967c5c5f67f156c1f391a02b5cbd8183f` | Offline evaluator source only; local GQA/3090 optimized support not established | `src/quantizer.py` dequantizes and returns FP16 tensors; integer payload is not retained for attention | **Negative**: no CUDA/Triton/C++ decode kernel; `src/evaluator.py` feeds dequantized FP16 `past_key_values` to ordinary Transformers | Not viable for the prioritized low-bit storage + optimized-inference reproduction |
| **Minima-KV** | No canonical author-linked public repository identified | Blocked | Unknown | Blocked | GitHub searches returned unrelated “minimal KVM”/MiniMax repositories; no name-collision repository was substituted. Evidence: `results/public/minima_kv_search_summary.json` |

### Why the implementations are not interchangeable

KIVI and KVQuant consume different quantization layouts and attention contracts.
KIVI's CUDA entry point expects packed cache dimensions and K/V metadata arranged
for its per-channel/per-token scheme; it is not a page-table-aware kernel.
KVQuant's source is explicitly MHA-only at the model seam. Kitty is already
page-oriented and has a GQA-aware fused attention kernel, but its checked-out
model integration is Qwen3-only and its buffers include a fixed FP16 sink,
query buffer, and local value buffer. QAQ is a quality simulator, not a
low-bit serving implementation.

## Local reproductions

Kitty kernel timings use CUDA events after warmup; KIVI end-to-end timings use
synchronized wall-clock intervals after warmup. KIVI compares its own public
model path against the ordinary Transformers FP16 model with the same checkpoint,
prompt token count, teacher-forced continuation, GPU, and model shape. The
reported cache bytes are the sum of the actual resident cache tensors (packed
codes plus metadata and FP16 residuals); allocator readings and peak workspace
are retained separately in each JSON artifact.

### KIVI end-to-end results

| Checkpoint / configuration | FP16 cache | KIVI cache | KV reduction | FP16 decode | KIVI decode | Slowdown | Paired NLL delta | top-1 agreement |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Llama 3.2 1B, 128 prompt, 16 decode, K2/V2 | 2,097,152 B | 1,212,416 B | **42.19%** | 15.89 ms/token | 32.18 ms/token | **2.02x** | +0.0510 | 100% |
| Llama 3.2 1B, 128 prompt, 16 decode, K4/V4 | 2,097,152 B | 1,671,168 B | **20.31%** | 15.66 ms/token | 30.87 ms/token | **1.97x** | +0.00189 | 100% |
| Llama 3.2 1B, 256 prompt, 8 decode, K2/V2 | 4,194,304 B | 1,998,848 B | **52.34%** | 16.61 ms/token | 30.54 ms/token | **1.84x** | +0.1296 | 87.5% |
| Llama 3.2 1B, batch 4, 256 prompt, 8 decode, K2/V2 | 16,777,216 B | 7,995,392 B | **52.34%** | 19.22 ms/token | 42.10 ms/token | **1.94x** | +0.1327 | 87.5% |
| Llama 3.1 8B, 128 prompt, 8 decode, K2/V2 | 8,388,608 B | 4,849,664 B | **42.19%** | 32.07 ms/token | 63.12 ms/token | **1.97x** | +0.00032 | 100% |

The source code and model traces confirm GQA operation: each local checkpoint
has 32 query heads, 8 KV heads, and four query heads per KV head. The 3090
CUDA extension was built with `TORCH_CUDA_ARCH_LIST=8.6`; the full commands and
compatibility shim are in `scripts/reproduce_kivi.py` and
`results/public/kivi_build.log`.

The 1B 2-bit context-256 result shows why the quality conclusion is not a
blanket 2-bit guarantee: it is one repeated prompt and only eight forced
positions, but it loses one top-1 position and has a larger paired NLL delta.
The 8B result is encouraging but also short and single-prompt. These are
reproduction probes, not a task-suite or perplexity certification.

The post-prefill allocator fields should not be confused with cache bytes:
KIVI's transient quantization and model-output workspace produces substantial
peak allocation, especially on the first run. The artifact records
`allocated_delta_bytes` and `peak_delta_bytes` for this reason. The robust
memory result is the direct tensor accounting of the resident packed cache,
not a claim that every allocator peak is reduced.

### Kitty optimized GQA kernel result

Because the pinned Kitty tree has no Llama wrapper, the probe uses the exact
Llama 3.2 1B dimensions (`B=1`, `H_Q=32`, `H_KV=8`, `D=64`) and the public
`KittyCache`, Triton packing, and fused qk/sv kernels. It compares the fused
kernel with PyTorch FP16 SDPA on the same unquantized synthetic K/V values.

| Static capacity | Dense FP16 K/V | Kitty allocated cache tensors | Capacity reduction | Kitty attention | FP16 SDPA | Slowdown | Mean abs error |
|---|---:|---:|---:|---:|---:|---:|---:|
| 512 tokens, 288-token prompt + decode | 1,048,576 B | 632,896 B | **39.64%** | 0.1915 ms | 0.0430 ms | **4.45x** | 0.0406 |
| 2048 tokens, 288-token prompt + decode | 4,194,304 B | 1,155,328 B | **72.45%** | 0.1915 ms | 0.0440 ms | **4.35x** | 0.0406 |

Outputs were finite and GQA head mapping was exercised by the public kernel.
These results are explicitly approximate/portability-modified: they do not
represent Kitty end-to-end Llama throughput or quality. They do show that a
page-oriented fused low-bit attention implementation can run on Ampere and
that the current SwiftLLM mixed-reference slowdown is not the only possible
implementation behavior.

### KVQuant build and block

`vendor/public/KVQuant/deployment/kvquant/setup_cuda.py` compiled its large
`quant_cuda` extension for `sm_86` with the local CUDA 12.4 toolkit and
PyTorch 2.4.0. The local Llama 3.2 config has 32 query heads and 8 KV heads;
the upstream deployment source independently contains an MHA-only assertion
(`num_key_value_groups == 1`) at the model seam. The actual import attempt was
blocked one step earlier by the custom Transformers dependency requiring
`tokenizers<0.19`, while the shared environment has `tokenizers==0.21.4`.
No quality, latency, or memory number is reported for KVQuant because it never
reached a valid local model execution. This is a failed portability
reproduction, not a failure of the published MHA implementation's kernels.

## Relationship to the previous SwiftLLM results

The completed SwiftLLM reference phase measured at batch 8/context 1024:

- 50% mixed INT8: about **3.8x** the FP16 page-reference attention time;
- 50% mixed INT4: about **6.8x**;
- full static INT8: about **6.5x**;
- full static INT4: about **12.5x**.

Those numbers came from the deliberately transparent Python/PyTorch page
attention path, which materializes and dequantizes each page. They are valid
measurements of that path, not of low-bit KV in general.

The KIVI end-to-end path is a directly relevant counterexample: it retains
packed low-bit storage and runs optimized CUDA/Triton decode, yet its matched
batch-1 and batch-4 decode slowdowns are roughly 1.84--2.02x at the tested
contexts. Kitty's
fused kernel is slower than FP16 SDPA in the synthetic GQA probe, but its
4.35--4.45x penalty differs from the 6.8x INT4 mixed-reference result and its
cache/layout/model boundary is different.

Therefore the previous 3.8x/6.8x observation is **not an intrinsic low-bit-KV
law** and is substantially attributable to the correctness-reference path.
It is not safe to say the entire gap is an artifact: KIVI still slows down on
this 3090 workload, Kitty is also slower in its isolated kernel probe, and the
comparisons differ in layout, batch, context, and baseline. A native SwiftLLM
page kernel would need its own end-to-end gate.

## Integration recommendation

1. **Foundation to adapt first: KIVI 2-bit K/V path**, restricted initially to
   old/resident pages and the 1B/8B Llama GQA shapes already reproduced. It has
   the strongest local evidence: complete source, pinned CUDA code, actual
   packed storage, GQA execution, and two model families.
2. **Do not copy KIVI into the scheduler yet.** The minimum integration seam is
   a SwiftLLM page adapter that stores each page in KIVI's packed K/V layout,
   retains its FP16 residual window, exposes page metadata, and invokes the
   existing KIVI CUDA GEMV/decode contract with SwiftLLM's block table and
   GQA head mapping. Appending/evicting across residual boundaries must be
   tested before any reclamation policy is considered.
3. **Use KIVI's 4-bit mode only as a quality/memory comparison**, not as the
   default recommendation: the local 1B probe reclaimed only 20.3% after
   metadata/residual overhead and was still about 1.97x slower.
4. **Keep Kitty as the page-kernel alternative**, not the first integration:
   its page layout is architecturally closer to SwiftLLM, but the pinned tree
   lacks a Llama wrapper and the isolated kernel was 4.35--4.45x slower than
   FP16 SDPA here.
5. **Do not pursue KVQuant without a deliberate GQA port**, and classify QAQ
   and Minima-KV as non-foundations for this objective.

This recommendation is a kernel/layout portability finding only. It does not
authorize scheduler, queue, priority, CPU/GPU swap, or memory-pressure policy
work in the separate goal.

## Reproduction commands

Initialize the pinned public sources and use the existing repository Python
environment (the worktree's `.venv` is a local symlink in this machine):

```bash
git submodule update --init --recursive
CUDA_HOME=/usr/local/cuda-12.4 \
  TORCH_CUDA_ARCH_LIST=8.6 MAX_JOBS=2 \
  .venv/bin/python vendor/public/KIVI/quant/setup.py build_ext --inplace
CUDA_HOME=/usr/local/cuda-12.4 \
  TORCH_CUDA_ARCH_LIST=8.6 MAX_JOBS=2 \
  .venv/bin/python vendor/public/KVQuant/deployment/kvquant/setup_cuda.py build_ext --inplace

CUDA_VISIBLE_DEVICES=1 \
PYTHONPATH="$PWD/vendor/public/KIVI:$PWD/vendor/public/KIVI/quant" \
.venv/bin/python scripts/reproduce_kivi.py \
  --model-path /nfs/home/s314511048/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6 \
  --output results/public/kivi_llama32_1b_2bit.json --device cuda:0 \
  --prompt-tokens 128 --decode-tokens 16 --warmups 1 --repeats 3 \
  --bits 2 --group-size 32 --residual-length 32

CUDA_VISIBLE_DEVICES=3 \
.venv/bin/python scripts/probe_kitty.py \
  --output results/public/kitty_llama_gqa_kernel.json --device cuda:0 \
  --max-length 2048 --prompt-tokens 288 --warmups 5 --repeats 30
```

The other KIVI and Kitty variants, including the batch-4/context-256 run,
use the same scripts and are fully recorded in their JSON `invocation` fields.
The KVQuant GQA failure is preserved in
`results/public/kvquant_gqa_blocked.log`; the Minima-KV search evidence is in
`results/public/minima_kv_search_summary.json`.

## Findings by evidence status

### Confirmed reproductions

- Exact public commits and repository URLs are pinned in the source manifest.
- KIVI's CUDA extension builds for RTX 3090/SM86 and runs packed K2/V2 and
  K4/V4 decode with local Llama 3.2 1B and Llama 3.1 8B GQA checkpoints.
- KIVI produces direct resident-cache reductions of 42.19% at the 128-token
  probe and 52.34% at the 256-token probe for K2/V2; K4/V4 gives 20.31% in
  the 1B 128-token probe.
- KIVI's optimized end-to-end decode latency and paired quality traces are
  preserved, including the 8B GQA run.
- Kitty's public Triton page packing and fused qk/sv attention run on the RTX
  3090 for the local Llama GQA dimensions and show real low-bit storage,
  finite output, and measured latency/error.
- KVQuant's custom CUDA extension compiles for SM86, while its GQA model seam
  is demonstrably blocked.

### Approximate or portability-modified reproductions

- KIVI uses a `vllm_flash_attn` shim for the public source's `flash_attn`
  import and Transformers >=4.51 star-import compatibility. The upstream
  KIVI source files were not modified; this is not a bit-for-bit original
  environment reproduction.
- Kitty is a synthetic GQA kernel probe, not a Llama end-to-end run.
- The local quality probes are paired forced-prefix NLL/top-1 checks, not
  broad downstream tasks or perplexity.

### Failed reproductions

- KVQuant did not execute a local Llama 3.x decode because its deployment
  model asserts MHA-only (`num_key_value_groups == 1`) and its pinned custom
  Transformers dependency conflicts with the shared environment. Its source
  kernels are still a credible optimized MHA implementation.
- Kitty's current end-to-end Llama run could not be performed because the
  checked-out model wrapper is Qwen3-only and the required submodule setup was
  not present; the kernel-only probe did run.

### Blocked methods

- QAQ is blocked from the priority claim by design: it returns dequantized
  FP16 cache tensors and has no optimized low-bit decode path.
- Minima-KV is blocked because no canonical public repository/commit could be
  identified under that exact name; unrelated repositories were intentionally
  excluded.

### Remaining uncertainty

- KIVI's 2-bit quality needs a real multi-prompt/task or perplexity study at
  the serving contexts relevant to SwiftLLM; the short probes are not a policy
  guarantee.
- KIVI's contiguous per-layer layout and residual-window transitions need an
  actual SwiftLLM page-table adapter benchmark before integration is approved.
- The batch-8/context-1024 SwiftLLM comparison still needs a native packed
  mixed-page kernel for an apples-to-apples answer; current cross-system
  evidence only establishes that the reference-path slowdown is not universal.
- KVQuant may be valuable after a proper GQA port, but that port is outside
  this reproduction-only phase.

## Completion audit against the active objective

| Explicit requirement | Evidence checked | Status |
|---|---|---|
| Work independently from memory-reclamation goal | Separate `low-bit-kv-reproduction` worktree/branch; no changes in original dirty worktree | Confirmed |
| Preserve exact upstream repositories/commits | Git submodules plus `results/public/source_manifest.json` | Confirmed |
| Determine public/completeness/Ampere/Llama/GQA/layout/optimized path per viable method | Source audit table; KIVI/KVQuant/Kitty/QAQ source paths; Minima search artifacts | Confirmed, with blocked classifications |
| Prioritize actual low-bit storage plus optimized inference | KIVI and Kitty reproduced; KVQuant source/build verified; QAQ excluded | Confirmed |
| Measure actual memory reduction | Resident tensor-byte accounting in KIVI/Kitty JSON; allocator/peak fields retained | Confirmed with allocator limitation explicitly reported |
| Measure decode latency/throughput vs own FP16 baseline | KIVI end-to-end matched baseline; Kitty kernel vs FP16 SDPA | Confirmed for viable local paths |
| Measure quality degradation | Paired NLL/top-1 in all KIVI JSON; Kitty numerical error | Confirmed as controlled probes, not broad quality |
| Measure mixed recent-high/old-low behavior where public implementation supports it | KIVI residual FP16 window + quantized old state; Kitty sink/local buffers; source layouts | Confirmed structurally; no dynamic SwiftLLM transition run |
| Use author kernels/layouts where possible | Upstream KIVI CUDA/Triton and Kitty Triton paths invoked; no replacement low-bit kernel | Confirmed |
| Document every portability adaptation | KIVI script compatibility metadata and report; Kitty synthetic boundary; KVQuant dependency/GQA blockers | Confirmed |
| Compare matched SwiftLLM FP16 secondarily | Existing SwiftLLM reference numbers and prior raw JSON/logs cited; no claim of apples-to-apples equivalence | Confirmed with boundary |
| Decide whether prior 3.8x/6.8x result is reference-path-specific | KIVI ~2x optimized E2E counterexample, Kitty independent kernel result, prior SwiftLLM results | Supported conclusion, not absolute proof |
| Do not implement scheduler/new quantizer/custom new kernel/policy | Scope statement, only adapters/probes, upstream kernels invoked | Confirmed |
| Concise integration recommendation | Recommendation section naming KIVI 2-bit, restrictions, artifacts, minimum adaptation | Confirmed |
| Final categories and uncertainty | Confirmed / approximate / failed / blocked / remaining uncertainty sections | Confirmed |

The study is complete as a reproduction and portability decision under these
explicit boundaries. It does **not** claim production readiness or authorize
opening the separate memory-reclamation scheduler goal.
