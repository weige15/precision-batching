# Mixed-precision KV-cache measurement audit v2

**Run date:** 2026-09-15  
**Scope:** corrected measurement baseline and claim audit only; no scheduler,
router, quantizer, CPU-swap policy, or weight-precision search was added.

## Decision

The previous `kv-break-even` action ranking is **withdrawn as a capacity or
latency ranking**. Its raw artifacts remain historical evidence, but they mix
one-layer conversion/attention with all-layer swapping, use a defective dense
layout construction, select physical pages globally across requests, omit
packed-arena copies from the memory ledger, and model no request-level
scheduling effects. The old KIVI cache-reduction number is also corrected:
`DynamicCache` accounting counted only V and its snapshot was at prefill while
its shape summary was after decode.

The corrected evidence supports the following narrower disposition:

* **Ready for the next fixed-GPU-memory capacity experiment:** the unchanged
  SwiftLLM dense FP16 allocator/block-table path. It is the only path here with
  an honest native reusable-block contract and no compression-dependent
  capacity interpretation.
* **Measurement seam, not capacity-ready:** the SwiftLLM page INT8 path. It
  now has all-layer conversion, nontrivial mapping, native segmented-attention
  correctness, explicit packed-copy accounting, allocator observations, and
  release observations. Its packed arenas approximately duplicate live page
  payload, so its logical reclaim must not be reported as reusable capacity
  without a storage redesign.
* **Not a SwiftLLM capacity backend:** KIVI. Its corrected native public path
  is useful latency/quality evidence, but its contiguous per-layer layout and
  residual-window state have no SwiftLLM page-table adapter.

No result requires compression to win. No positive full-model or continuous-
batching compression claim is made.

## Objective and boundaries

The concrete deliverable was an evidence-backed corrected baseline that:

1. preserves the reviewed repository, prior artifacts, public source pins, and
   unrelated changes;
2. fixes or exposes K/V accounting, timing, token-position, peak-memory,
   ownership/lifetime, layout/permutation, physical-page selection, and
   one-layer/all-layer defects;
3. measures logical payload, unique live storage, allocator allocated/reserved
   memory, peak memory, and explicitly observed release/reusability separately;
4. separates performance from quality scoring and tests prefill, repeated
   decode append, page boundaries, conversion/restoration, and release;
5. runs the corrected native paths on the available authorized RTX 3090 with
   local Llama checkpoints where their interfaces support it;
6. audits KIVI, KVQuant, Kitty, QAQ, and the unresolved Minima-KV search without
   combining incompatible implementations into a fabricated ranking; and
7. leaves a claim-by-claim disposition with confirmed, uncertain, blocked, and
   remaining-uncertainty sections.

The v2 changes are `scripts/reproduce_kivi.py`,
`scripts/kv_measurement_v2.py`, `scripts/verify_artifacts.py`, and
`tests/test_kv_measurement_v2.py`. All new measurements are under
`results/kv-measurement-v2/`. Existing `results/public/` and
`results/sensitivity/` artifacts were not rewritten.

## Provenance, environment, and isolation

The reviewed remote commit was
`8c0a0bfcc6bf87461c104603718ed2dc507df390`. The starting local checkout was
`e74c42c064fac54f9cda0c1b74680f1fa6d350d0`; it was fast-forwarded with
`git merge --ff-only origin/main`, not reset. The actual local commit used for
v2 artifacts is therefore also
`8c0a0bfcc6bf87461c104603718ed2dc507df390`, on
`structured-precision-evidence`.

At the end of this audit, tracked dirty files are:

* `README.md`
* `scripts/reproduce_kivi.py`
* `scripts/verify_artifacts.py`

New untracked audit files are the v2 scripts, test, report, raw artifact
folder, and the pre-existing unrelated `vendor/qaq/assets/` directory. The
tracked dirty-diff SHA-256 at report creation is
`0614ff2689e3a1fccffa384cbf179e43cd4d74490ed6b9023edc444634703402`.
Each GPU artifact also records the status and tracked-diff hash observed when
it was produced; see its `provenance` object.

Submodule revisions recorded by the v2 runs:

```text
vendor/public/KIVI                                  876b4d2d08e3b1d5f70d0969c299d8c7c42ddfb6
vendor/public/KVQuant                               57a238357f0ffe50084670fcd5781c9848f80ea2
vendor/public/Kitty                                dfd2c07b407d6b407179359207c612ab631f3ed1
Kitty/third_party/lm-evaluation-harness             62617a8cbd7f3beddc2a9c029536e0e387976beb
Kitty/third_party/transformers                      37f8b0b53512e6aae0cfd15746c133c101783178
```

The run used Linux x86-64, Python 3.12.3, PyTorch 2.4.0+cu121, Triton 3.0.0,
Transformers 4.51.3, vllm-flash-attn 2.6.2, CUDA toolkit 12.4.99, and driver
580.159.03. All GPUs are RTX 3090, compute capability 8.6, 24,576 MiB. The
snapshot was recorded in
`results/kv-measurement-v2/environment_snapshot.json`,
`nvidia-smi.csv`, and `nvidia-processes.csv`.

The synthetic SwiftLLM run used physical GPU 0 and the checkpoint smoke used
physical GPU 3 via `CUDA_VISIBLE_DEVICES`; corrected KIVI 1B and 8B runs used
physical GPUs 1 and 2. At the device snapshot GPUs 0--3 had 1 MiB used. Other
users' processes remained on GPUs 4--6 and were not stopped or modified.

## Corrected measurement contract

### Accounting

`reproduce_kivi.py` now recursively inventories every tensor in either a
Transformers `DynamicCache` or KIVI's legacy tuple. It sums K and V logical
bytes, records tensor shape/dtype, and deduplicates underlying
`untyped_storage()` pointers for unique live storage. The v2 SwiftLLM ledger
additionally inventories page K/V payloads, FP16 scales, CPU format metadata,
compact FP16/INT8 packed arenas, slot maps, and the call-local segmented-
attention workspace. The workspace formula covers `segment_valid`, `mid_o`,
`mid_log`, and the four int32 segment descriptors; its logical size is 33,360 B
(1B) or 66,128 B (8B), and its post-return live size is verified as zero. It
reports:

* `logical_payload_bytes`: mathematical live page/cache payload plus scales;
* `unique_live_storage_bytes`: deduplicated backing storage;
* metadata bytes and packed-arena/map bytes separately;
* pending old-page bytes and shadow state;
* `torch.cuda.memory_allocated` and `memory_reserved`;
* reset-window peak allocated/reserved values; and
* allocator state before and after explicit cache release.

Logical bytes are never relabeled as physical HBM free or reusable scheduler
capacity. For dense SwiftLLM, the preallocated cache is reported as capacity
of the native block pool. For the page path, release is directly observed, but
packed copies are charged and no request-level capacity claim is made.

### Timing and quality

KIVI v2 performance performs one prefill model call and a model-only decode
loop with input tensors prepared before timing. It synchronizes once at the
phase boundary, not once per token. Cross entropy, top-1 comparison, and
host scalar extraction are in `run_quality`, outside performance timing. The
memory replay intentionally synchronizes after every token and is labeled
memory evidence, not latency evidence.

SwiftLLM v2 uses CUDA events around attention/conversion and a separate
synchronized checkpoint forward trace. Cold packed-arena construction and
steady-state attention are reported separately. The raw records include
repetition counts and timing units.

## Defect-by-defect audit

| Audit target | What was reproduced or inspected | Corrected disposition |
|---|---|---|
| KIVI `DynamicCache` omitted K | The old `cache_bytes` used `zip(key_cache, value_cache)` but summed only `value`. `test_dynamic_cache_accounting_includes_k_and_v_and_deduplicates_aliases` fails under the old implementation and passes with the recursive inventory. | Fixed in `scripts/reproduce_kivi.py`; old baseline cache numbers are withdrawn. |
| Quality contaminated decode timing | The old KIVI timed loop called cross entropy, `.item()`, and top-1 `.item()` inside the timed loop. | Fixed by `run_performance`/`run_quality` separation; v2 timing boundary is recorded and verified. |
| Different memory positions | Old `cache_bytes` was captured after prefill, while `cache_summary` described `past` after decode. | V2 records prefill, every decode position, decode end, and release. 1B traces have positions 128--144; 8B traces have 128--136. |
| Decode peaks omitted | Old KIVI recorded only the prefill peak, including full prefill logits. | V2 records reset-window prefill/decode peaks and full allocator allocated/reserved fields. Peak includes necessary model output/workspace; it is not claimed to be pure KV. |
| Token-major dense pages reshaped as head-major | The old break-even harness used `.reshape(..., kv_heads, block, dim)` on `[block, kv_heads, dim]` pages. The v2 dense construction uses `permute(0,1,3,2,4).contiguous()`. | Regression test uses non-symmetric token/head values; GPU FP16 dense/page control uses a non-identity table and GQA. |
| Packed copies omitted | `PagedKVCache.packed_attention_storage()` creates copied FP16/INT8 arenas and maps that `storage_summary()` did not count. | V2 includes packed arena/map bytes and a cold peak. For 1B mixed state, logical payload is 2,629,632 B, packed arena/map is 2,630,400 B, and unique live storage is 5,260,224 B. For 8B those values are 10,518,528 B, 10,520,064 B, and 21,038,976 B. |
| Cold arena construction hidden | Old break-even timing warmed `optimized_attention` before timing and then charged only later transitions. | V2 records `cold_allocator_peak` and steady CUDA-event timing; no cold cost is silently treated as steady state. |
| Global page IDs across requests | Old selectors used the first/last global physical IDs. | `select_request_local_blocks` follows each request's block-table row; the regression test uses `[[5,1,9],[7,3,8]]` and expects `[5,7]` for old pages. |
| One-layer conversion/attention vs all-layer swap | Old break-even used a one-layer `PagedKVCache` but all-layer `swap_blocks` and multiplied only bytes. | V2 conversion uses all 16/32 layers. It deliberately does not publish a repaired queue/offload/compression ranking. |
| Request-level scheduling omitted | Neither old nor v2 harness implements arrival traces, queue admission, preemption, or a controller. | All queue/action ranking and throughput implications remain unmeasured. |
| Append/residual/page boundaries | V2 synthetic run appends at positions 31, 32, 33 with a 16-token page; checkpoint smoke uses the same 31-token prompt and positions 32, 33, 34. | Direct traces pass; compressed old page and newly allocated FP16 page behavior are exposed. |
| Lifetime/release | V2 releases previous dynamic pasts between repetitions and explicitly frees page blocks before allocator snapshots. Pending/shadow state is retained until synchronization in the page implementation. | Release observations are present; allocator release is not equated with global HBM capacity. |

The old break-even source fingerprint is retained in its artifact and its
historical verifier check remains unchanged. It is not substituted with v2
numbers. The v2 verifier adds checks; it does not remove or weaken historical
checks.

## Corrected directly observed measurements

### SwiftLLM native page path

Artifact: [`results/kv-measurement-v2/swiftllm_native_v2.json`](../results/kv-measurement-v2/swiftllm_native_v2.json). The synthetic case is batch 2,
31 prefill tokens, three page capacity slots per request, two resident pages
per request, all 16 or 32 layers, GQA 32 Q heads / 8 KV heads, and a reversed
physical block table. The old-page selection is `[2, 5]` in both request rows,
not a global prefix.

| shape | FP16 prefill logical | all-layer INT8 conversion | reclaim | packed/map bytes | unique mixed storage | cold extra allocated | steady attention |
|---|---:|---:|---:|---:|---:|---:|---:|
| Llama 3.2 1B | 2,097,152 B | 32 pages, 55.357 ms | 516,096 B | 2,630,400 B | 5,260,224 B | 2,676,224 B | 0.686 ms / 2 iterations |
| Llama 3.1 8B | 8,388,608 B | 64 pages, 4.213 ms | 2,064,384 B | 10,520,064 B | 21,038,976 B | 10,607,104 B | 0.673 ms / 2 iterations |

Conversion timing is a direct observation of this small batched synthetic
case and is visibly variable; it is not generalized into a throughput law.
The corrected FP16 dense/page control had maximum absolute errors of
`0.00146484375` (1B) and `0.0009765625` (8B). The optimized mixed path had
maximum absolute errors of `0.00048828125` and `0.0009765625`, respectively,
and finite outputs. The 1B/8B page payload releases observed
`3,145,728`/`12,582,912` allocated bytes and `8,388,608`/`31,457,280` reserved
bytes after explicit cache destruction and `empty_cache()`.

### Checkpoint-backed SwiftLLM smoke

The same artifact's `checkpoint_probe` uses the local Llama 3.2 1B Instruct
checkpoint, 31 prefill tokens, and three native decode appends crossing the
page boundary. It runs a dense FP16 model and a fresh FP16 page-store model
with the first old physical page demoted to INT8 across all 16 layers.
Quality is not scored in this smoke; greedy logits only are used to drive
necessary decode inputs, and timing contains no quality scoring.

* Dense trace positions are 31, 32, 33, 34 with logical used-page bytes
  `1,048,576`, `1,048,576`, `1,572,864`, `1,572,864`.
* Dynamic INT8 conversion is 16 pages, `524,288 -> 266,240` B, reclaiming
  `258,048` B; the selected block is recorded in the artifact.
* Dynamic positions are 31, 32, 33, 34 with logical bytes
  `1,048,576`, `790,528`, `1,314,816`, `1,314,816`.
* Both methods released their explicitly allocated native model/cache state;
  exact before/after allocator observations are in the JSON.

The checkpoint smoke is a correctness/lifetime boundary test, not a full-model
capacity result: the page model uses the research one-tensor-per-page store
and the dense model preallocates its native block pool.

### Corrected KIVI public path

Artifacts:

* [`kivi_llama32_1b_v2.json`](../results/kv-measurement-v2/kivi_llama32_1b_v2.json)
* [`kivi_llama31_8b_v2.json`](../results/kv-measurement-v2/kivi_llama31_8b_v2.json)

Both use KIVI commit `876b4d2d...`, the author CUDA GEMV/Triton path, and the
local GQA checkpoints. KIVI source remains unmodified; the compatibility shim
is recorded in each artifact. Repetitions were 3 (1B) and 2 (8B), with one
warmup. Decode timing is model-only and in milliseconds per forced token
(total decode time divided by 16 or 8 tokens).

| checkpoint | corrected FP16 K+V cache at prefill | corrected KIVI cache | logical reduction | decode slowdown | paired NLL delta | top-1 |
|---|---:|---:|---:|---:|---:|---:|
| Llama 3.2 1B | 4,194,304 B | 1,212,416 B | 71.09375% | 2.016x | +0.0510052 | 100% |
| Llama 3.1 8B | 16,777,216 B | 4,849,664 B | 71.09375% | 2.070x | +0.0003198 | 100% |

These cache values are at the same prefill position and include both K and V.
The old 42.1875% reduction was an accounting artifact, not a corrected
capacity result. At decode end, the 1B caches were 4,718,592 B (FP16) and
1,523,712 B (KIVI); the 8B caches were 17,825,792 B and 5,472,256 B. Per-
position traces contain 17 and 9 snapshots, respectively. The KIVI logical
cache is not reusable SwiftLLM block capacity because its packed layout and
residual window differ.

The NLL values are controlled one-prompt forced-prefix measurements. They do
not certify task quality, long-horizon quality, or free-running generation.

## Public implementation audit

`docs/public-kv-study.md`, `docs/public-kv-completion-audit.md`,
`results/public/source_manifest.json`, the pinned submodules, and
`verify_public_kv.py` were inspected and verified. The public verifier passed.
The dispositions are:

* **KIVI:** complete enough for local Llama 3.2 1B and Llama 3.1 8B GQA runs;
  actual packed low-bit storage and optimized decode confirmed. Corrected v2
  runs are above.
* **KVQuant:** CUDA extension build for SM86 passed, but its model seam
  asserts `num_key_value_groups == 1`; local Llama 3.x is 32/8 GQA. No local
  Llama latency, quality, or capacity number is claimed.
* **Kitty:** public Triton page packing and fused kernel run for synthetic
  32/8/D64 GQA dimensions, but the pinned tree has no Llama wrapper. Its
  results remain kernel-only and are not combined with KIVI or SwiftLLM.
* **QAQ:** the pinned evaluator dequantizes to FP16 before ordinary model
  attention and has no optimized low-bit decode path. Its quality simulation
  is not evidence of low-bit serving capacity.
* **Minima-KV:** the exact public identity remains unresolved; the checked
  search artifact intentionally excludes unrelated KVM/MiniMax repositories.

The public artifacts' historical KIVI timings were contaminated by quality
scoring and their baseline cache fields omitted K. Their qualitative
optimized-path evidence remains useful, but exact cache-reduction and
latency claims are superseded by the v2 KIVI artifacts. Existing public and
SwiftLLM raw artifacts were preserved.

## Verification record

### Commands and exit codes

The successful commands and exits are preserved in
`results/kv-measurement-v2/command-status.txt`, `attempt-log.txt`, and the
individual JSON `invocation` objects.

| Command/check | Result |
|---|---|
| `git submodule update --init --recursive` | pass, exit 0; all pinned public submodules initialized |
| KIVI build from the repository root | failed, exit 1: wrong working directory (`csrc/gemv_cuda.cu` not found) |
| KIVI build from `vendor/public/KIVI/quant` with wrong relative interpreter | failed, exit 127; exact failed attempt retained |
| KIVI build from its quant directory with `../../../../.venv/bin/python` | pass, exit 0; SM86 CUDA extension |
| `kv_measurement_v2.py` synthetic/native final run | pass, exit 0; two shapes, 2 repetitions, 2 attention iterations |
| corrected KIVI Llama 3.2 1B run | pass, exit 0; 1 warmup, 3 repetitions, 16 decode tokens |
| corrected KIVI Llama 3.1 8B run | pass, exit 0; 1 warmup, 2 repetitions, 8 decode tokens |
| checkpoint probe first attempt | failed, exit 1: fixture selected dense mode instead of page-store mode |
| checkpoint probe corrected rerun | pass, exit 0; Llama 3.2 1B, 31+3 positions |
| documented unittest discovery | pass, exit 0; **28 tests passed** |
| `py_compile` for v2 scripts/test/verifier | pass, exit 0 |
| `scripts/verify_public_kv.py` | pass, exit 0 |
| `scripts/verify_artifacts.py` | pass, exit 0; historical checks plus v2 checks |
| `git diff --check` | pass, exit 0 |

The 28 unit tests ran on CPU and are not GPU performance evidence. GPU
performance/correctness evidence is only the RTX 3090 artifact set described
above. No historical verifier condition was deleted or relaxed to force a
pass; v2 checks are additive.

### Source fingerprints

* SwiftLLM native v2 source fingerprint:
  `34fb8127ef1f1be24832642c8289fce64da67ef9ef350a8f46484fd15217684f`.
* Corrected KIVI v2 source fingerprint:
  `95c1114b993b76d0323ce040015955af6f3f28dd4a368249b275e432e3121b55`.
* SwiftLLM upstream pin:
  `682cf9a28f97f7490409981a2f181528f377eb5d`.
* Public KIVI pin:
  `876b4d2d08e3b1d5f70d0969c299d8c7c42ddfb6`.

The exact source file lists are in the artifact provenance and are recomputed
by `scripts/verify_artifacts.py`. That verifier also recomputes every recorded
KIVI cache tensor logical/unique total and every native page/metadata/packed
ledger total from the raw tensor inventory; release deltas are checked as
before-minus-after arithmetic.

## Claim-by-claim completion audit

| Requirement / named evidence surface | Artifact or check | Status |
|---|---|---|
| Actual local commit, reviewed remote commit, dirty state | v2 artifact `provenance`; `git status`, diff hash, fast-forward record above | **Confirmed** |
| Preserve unrelated changes and historical evidence | Existing `results/public`, `results/sensitivity`, `vendor/qaq/assets/`; only new v2 files plus two code/verifier edits | **Confirmed** |
| SwiftLLM/KIVI/public pins and submodules | `docs/baseline.md`, `docs/available-assets.md`, `docs/public-kv-study.md`, submodule status, source manifest | **Confirmed** |
| Environment and authorized RTX 3090 | environment snapshot, GPU CSVs, artifact hardware/provenance | **Confirmed** |
| `docs/kv-break-even-report.md` audited | Defect table; v1 ranking withdrawn rather than multiplied or relabeled | **Confirmed, narrowed** |
| `scripts/kv_break_even_experiment.py` inspected | Dense reshape, global selection, one/all-layer mismatch, packed warmup and analytical queue term identified | **Confirmed audit; v1 not reused for ranking** |
| `scripts/reproduce_kivi.py` corrected | K+V recursive accounting, separate performance/quality, per-position/peak/release traces | **Confirmed** |
| `scripts/verify_artifacts.py` extended | Additive v2 checks for ledgers, mappings, peaks, checkpoint smoke, and public v2 | **Confirmed** |
| SwiftLLM cache/attention ownership | v2 page/scale/metadata/packed/map ledger; source inspection of `kv_cache.py` and segmented attention | **Confirmed for measured path; full serving allocator remains open** |
| Dense/page FP16 logical input/output match | non-identity GQA control, max errors 0.001465/0.000977, regression test | **Confirmed within recorded FP16 kernel tolerance** |
| Optimized INT8 correctness and no intentional quantization conflation | segmented path vs page oracle, max errors 0.000488/0.000977, finite output | **Confirmed for measured shapes** |
| KIVI K/V payload/scales/residual accounting | recursive legacy tuple inventory and same-position corrected artifacts | **Confirmed for returned KIVI cache; not a SwiftLLM capacity claim** |
| Unique aliased storage | CPU alias regression plus per-tensor storage IDs | **Confirmed** |
| Allocated/reserved/peak memory | v2 snapshots and reset-window peaks | **Confirmed for isolated processes; allocator fragmentation under serving pressure remains uncertain** |
| Reusable capacity after release | explicit page free and model/cache release observations | **Confirmed as release observations; only dense native block pool is capacity-ready** |
| Prefill, repeated decode append, residual/page boundaries | SwiftLLM synthetic positions 31/32/33 and checkpoint 31/32/33/34; KIVI traces | **Confirmed** |
| Previous repetitions released before next measurement | `run_performance`, `run_memory_trace`, `release_model`, artifact release fields | **Confirmed** |
| Avoidable host sync/quality outside timing | timing boundary fields and source/test checks | **Confirmed** |
| Necessary inference/cache update inside timing | native model/attention forward calls remain inside CUDA-event or synchronized wall boundary | **Confirmed for measured harness** |
| Request-local page selection | reversed block table, expected selection, unit test | **Confirmed** |
| All-layer conversion rather than guessed scaling | 16/32-layer conversion keys and exact byte arithmetic | **Confirmed** |
| No repaired queue/offload/compression ranking fabricated | v2 `verdict` says request-level capacity not run; no action table added | **Confirmed** |
| Existing public implementations inspected | source manifest, public study, KIVI/KVQuant/Kitty/QAQ/Minima dispositions | **Confirmed with blocked methods** |
| Existing tests and raw artifacts retained | old verifier and artifacts pass; v2 raw JSON/logs linked | **Confirmed** |
| Relevant tests, documented discovery, artifact verifier | 28 tests, pycompile, `verify_artifacts.py`, `verify_public_kv.py`, all exit records above | **Confirmed** |
| Small checkpoint-backed rerun on corrected native paths | corrected KIVI 1B/8B actual paths plus SwiftLLM native Llama 3.2 1B smoke | **Confirmed as smoke, not full-model capacity** |
| New artifact verification not weakening historical checks | additive verifier calls; historical checks still pass | **Confirmed** |
| Final report categories and changed conclusions | this report: confirmed, supported but uncertain, blocked, remaining uncertainty | **Confirmed** |

## Confirmed findings

1. The KIVI DynamicCache baseline accounting defect is real and changes the
   reported same-position logical reduction from 42.19% to 71.09% for the
   corrected 1B/8B probes. This is a measurement correction, not evidence that
   71.09% is reusable SwiftLLM capacity.
2. Quality scoring and per-token host synchronization were inside the old KIVI
   decode timer. The v2 timer excludes them and records its boundary.
3. Old and final cache snapshots referred to different token positions. V2
   position traces grow through decode and include decode peaks and release.
4. The old dense-page reshape was a real token/head layout hazard. The v2
   permutation and nontrivial GQA/block-table test expose and correct it.
5. The segmented SwiftLLM path retains copied packed arenas. Once charged, a
   mixed 1B/8B state has roughly 2x page payload unique live storage before
   temporary workspace. Logical compression alone is therefore not capacity.
6. V2 all-layer conversion and checkpoint page-boundary append/release traces
   pass. The optimized path matches its explicit oracle within the recorded
   sub-0.001 absolute-error range for these cases.
7. Corrected KIVI 2-bit packed decode runs on both local Llama GQA checkpoints
   and remains slower than the corrected FP16 control by about 2.02--2.07x in
   this small probe. The result is direct optimized-path evidence, not a
   universal quantization law.
8. All 28 repository tests, Python compilation, public verification, additive
   artifact verification, and diff checks pass. CPU tests remain separate from
   GPU evidence.
9. Historical source pins, model availability, public method boundaries,
   baseline/no-op behavior, and the prior quality-control observations remain
   valid within their documented scopes. The old KIVI memory/timing numbers and
   old SwiftLLM action ranking do not.

## Supported but uncertain findings

* Charging packed copies is likely the dominant capacity problem for the
  current SwiftLLM page implementation, but allocator fragmentation and
  concurrent allocator pressure could change the exact overhead.
* The v2 synthetic conversion times and attention times are useful boundary
  observations, not a full request-level latency model. The checkpoint dynamic
  smoke shows large boundary-sensitive timings in a tiny one-request run; it
  does not establish a stable throughput ranking.
* KIVI's corrected short probe supports a real low-bit optimized implementation
  on Ampere and a roughly 2x slowdown, but its quality evidence remains one
  prompt and forced-prefix. It cannot establish perplexity, task quality,
  long-horizon drift, or free-running output quality.
* FP16 dense/page correctness in the v2 control supports the permutation fix;
  it does not prove every future shape, block-table stride, or multi-request
  CUDA-graph contract.
* The v2 page path is a credible starting measurement seam, but a fixed-GPU
  capacity result would need a packed arena or zero-copy page design with its
  own allocator ledger. The dense native allocator is the defensible next
  capacity backend now.

## Blocked questions

* **Full-model fixed-memory compression capacity:** not run. SwiftLLM v2's
  checkpoint smoke is a small one-request boundary test, and KIVI has no
  SwiftLLM adapter. Proceeding requires an adapter/fixed arena that accounts
  for payloads, scales, metadata, packed copies, residuals, workspace, and
  release at request level.
* **Continuous batching, arrivals, queueing, preemption, and throughput:** not
  run by design. Evidence needed is a trace-driven experiment with matched
  requests and a fixed GPU memory budget, not multiplication of one-layer
  timings.
* **Full-model CPU offload/compression action comparison:** not rerun after the
  v1 audit. The existing C++ swap mechanism remains lossless and historical,
  but no corrected mixed-action ranking is claimed.
* **KVQuant local Llama GQA:** blocked by its MHA-only assertion and dependency
  contract; a deliberate GQA port would be a new portability change.
* **Kitty end-to-end Llama:** blocked by the pinned tree's missing Llama
  wrapper; only the synthetic GQA kernel evidence is valid.
* **QAQ optimized serving:** blocked by its FP16-dequantizing evaluator and no
  fused low-bit decode path.
* **Minima-KV:** blocked until an author-linked canonical repository or commit
  is identified.
* **Broad quality:** no multi-prompt/task/perplexity or free-running quality
  rerun was required for this corrected measurement baseline. The exact
  evidence needed is a held-out multi-prompt suite with paired logits,
  generation traces, and confidence intervals for each supported backend.

## Remaining uncertainty and next evidence

The following earlier claims must be narrowed or withdrawn:

* Withdraw the old KIVI `42.19%` cache reduction as a corrected K/V number; it
  was V-only baseline accounting at a different position.
* Withdraw the exact KIVI old decode timings as uncontaminated performance;
  use the v2 model-only timings instead.
* Withdraw the `kv-break-even-report.md` global winner/action map as a valid
  capacity or latency ranking. Its logical byte and native-kernel correctness
  observations may remain as historical enabling evidence, but its dense
  layout, selection, packed-copy, layer-unit, and missing scheduling defects
  prevent the ranking claim.
* Narrow all SwiftLLM INT8 logical-reclaim language to “logical payload
  reduction” unless packed copies and allocator reuse are included.
* Narrow KIVI quality to two short forced-prefix probes and do not infer task
  quality or throughput.

The next experiment should start with the native SwiftLLM dense FP16 allocator
and its existing block manager, under a fixed GPU budget and a trace that
records prefill/decode/release per request. Compression should be reconsidered
only after a zero-copy or arena-backed adapter can pass the v2 ledger and
oracle checks. The next required evidence is a clean matched request-level
run with per-request page tables, all-layer transition timings, allocator
allocated/reserved/peak values, reusable free-block counts, and a held-out
quality suite. Nothing in this audit supports an inference about continuous
batching, user-visible latency, quality, or throughput beyond the direct
measurements listed above.
