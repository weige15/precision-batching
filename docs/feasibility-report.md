# Feasibility report: query-/Q/K/V-aware precision

## Executive decision

**Conditional go for a narrowly scoped native-kernel feasibility phase; no-go for implementing a full precision-aware continuous-batching scheduler yet.**

The local numerical proxy shows a repeatable and sizable **projection** asymmetry: in both Llama 3.2 1B and Llama 3.1 8B, all-layer 4-bit V-projection perturbation produces substantially more logit error than an otherwise identical Q-only or K-only perturbation. One-projection layer sweeps also show multi-fold layer variation. This is enough evidence to justify measuring a native kernel and a better policy objective.

It is not enough to claim that query-specific scheduling will beat a fixed profile. In the tested matrix, the equal 8-bit profile was safer than heterogeneous profiles at the same coarse average budget, and the offline query oracle reported zero decode-MSE reduction at the tested 4/8/16 budget comparisons. The proxy is also not a low-bit execution kernel. A full runtime scheduler, KV-cache quantization, bit-plane swapping, and production INT4 implementation should remain deferred until native overhead and query-dependent quality gains are established.

## Reproduction surface

The experiment is fully driven by:

```bash
MODEL=/path/to/local/llama-checkpoint
CUDA_VISIBLE_DEVICES=5 .venv/bin/python scripts/qkv_sensitivity.py \
  --model-path "$MODEL" \
  --output results/sensitivity/run.json \
  --max-prompts 8 --decode-tokens 4 --bits 4 8 16 --device cuda:0
```

For the checked-in runs:

- `results/sensitivity/llama32_1b_qkv_matrix.json`: 8 controlled prompts, 4 reference decode tokens, 27 all-layer profiles in `{4,8,16}^3`, 16-layer Llama 3.2 1B.
- `results/sensitivity/llama32_1b_qkv_matrix_layer_sweep.json`: same prompts, plus 96 one-projection/one-layer profiles across 16 layers.
- `results/sensitivity/llama31_8b_qkv_matrix.json`: 8 prompts, 4 decode tokens, 27 profiles, 32-layer Llama 3.1 8B.
- `results/sensitivity/llama31_8b_qkv_matrix_layer_sweep_4prompts.json`: same 8B sample, plus 192 one-projection/one-layer profiles across 32 layers.

Each JSON contains model file hashes, environment metadata, prompt records, per-position prefill/decode metrics, aggregate metrics, per-config elapsed time, and both unweighted diagnostic and weighted-budget oracle comparisons. Weighted storage bits use the actual Q/K/V parameter counts for the model's GQA shape plus one FP16 scale per 128-weight group; packed metadata and runtime overhead are not included. The sensitivity execution uses Hugging Face Llama modules with the shared proxy helper, not SwiftLLM's Triton/PagedAttention kernels; SwiftLLM is the serving baseline and metadata integration target. Decode comparisons feed the same reference continuation token to the candidate model, so they measure paired logit drift rather than confounding precision error with divergent sampled text.

## Confirmed findings

1. The unmodified pinned SwiftLLM baseline runs on the local Llama 3.2 1B checkpoint. On a free RTX 3090, the upstream offline example reported 37,975 GPU KV blocks and a 4.78-GB profiled runtime peak. Output is preserved in `results/baseline/swiftllm_unmodified_1b.log`; earlier runs with a competing GPU process were discarded from this baseline.
2. The research fork's all-FP16/no-op output matches the unmodified output exactly after excluding the model-creation-time line. The sensitivity JSONs' FP16 controls have an explicitly defined exact tolerance (`max_abs_logit_error <= 0.0`, zero top-1 mismatches); unit tests and the metadata smoke test pass.
3. The scheduler remains length/block/batch based. Precision metadata is carried beside the request and is not used by `Scheduler.get_next_batch`.
4. `PrecisionProfile` represents separate Q/K/V/O/FFN bits per request. The default is all FP16. The metadata is passed through `Engine` → `LlamaModel.forward` → `LlamaInferState` → Q/K/V/O/FFN projection call sites.
5. The local available MorphServe checkpoint is AWQ W4 and cannot be loaded by the pinned ordinary SwiftLLM weight loader. No unsupported conversion was silently attempted.

## Supported but sample-limited findings

### Projection asymmetry

Aggregate all-layer 4-bit-only perturbations (other two projections at FP16) are:

| model / phase | Q only MSE | K only MSE | V only MSE | V/Q |
|---|---:|---:|---:|---:|
| Llama 3.2 1B / prefill | 0.01902 | 0.01799 | 0.08432 | 4.43x |
| Llama 3.2 1B / decode | 0.04160 | 0.02545 | 0.13605 | 3.27x |
| Llama 3.1 8B / prefill | 0.04311 | 0.05282 | 0.14014 | 3.25x |
| Llama 3.1 8B / decode | 0.01476 | 0.01134 | 0.07989 | 5.41x |

This is consistent with a useful structured precision hypothesis, but the direction is not identical to QAQ's key/value **cache** result because this experiment perturbs projection weights, not cached vectors. It should not be overinterpreted as a universal K/V ordering.

At 4-bit all-layer perturbation, prefill top-1 match was 93.0% for 1B and 96.2% for 8B; decode top-1 match was 96.9% and 93.8%, respectively, over the recorded position samples. At 8-bit all three projections, decode logit MSE fell to `7.53e-4` (1B) and `4.20e-4` (8B), with 100% decode top-1 match in these samples.

### Layer structure

The one-projection/one-layer 4-bit decode-MSE ranges were:

| model | Q range | K range | V range |
|---|---:|---:|---:|
| Llama 3.2 1B | 0.000985–0.00522 | 0.000937–0.00346 | 0.00445–0.01969 |
| Llama 3.1 8B | 0.000079–0.00118 | 0.000085–0.000645 | 0.00134–0.01102 |

The layer sweep therefore supports nonuniform layer allocation as a worthwhile next question. It does not yet provide confidence intervals or a robust transferable layer order.

### Prefill/decode and overhead

The harness records both phases. On the checked-in samples, decode drift is sometimes larger (1B) and sometimes smaller (8B) than prefill drift, so position phase should remain an explicit policy dimension rather than an assumption. Four decode steps are insufficient to make a long-generation claim.

Per-config elapsed time includes model projection/attention execution but excludes the preceding weight mutation. For the current eight-prompt runs, all-layer `(4,4,4)` versus `(16,16,16)` took 0.919 s versus 0.828 s for the 1B sample and 1.481 s versus 1.433 s for the 8B sample. These are **eager fake-quantization proxy overhead measurements only**, not low-bit speedup measurements; the candidate weights are dequantized FP16 tensors and use ordinary PyTorch kernels.

### Query-specific oracle

The JSON `unweighted_fixed_oracle_comparison` chooses the lowest decode logit MSE independently for each prompt among tested profiles within ±2/3 **unweighted** bit of the fixed all-equal profile. This is a diagnostic, not a memory comparison. At fixed 4, 8, and 16 unweighted-bit comparisons, the recorded oracle reduction is 0.0 for both models because the equal profile is the only strong choice at those coarse budgets.

The `weighted_budget_oracle_comparison` holds weighted Q+K+V storage bits constant and compares the best single profile with a per-prompt oracle. It finds a modest 2.65% reduction for Llama 3.2 1B at 6.79 weighted storage bits (swapping K/V allocation) but only 0.013% for Llama 3.1 8B at the same weighted budget in the full eight-prompt matrix. The separate unweighted-sum diagnostic can show larger differences (1.19% for 1B and 2.73% for 8B), but those are not memory-comparable and are not used for the go/no-go decision. This is a deliberately favorable offline oracle, not a realizable runtime scheduler, and the small sample/metric effect is not yet enough to justify dynamic scheduling.

## Blocked questions

- Native packed INT4/INT8 Q/K/V kernels and their actual memory/latency behavior are not implemented.
- The eager proxy does not model activation quantization, AWQ scales, kernel packing, or hardware-specific accumulation.
- QAQ's cache quantization policy is not reproduced on LLaMA 2 tasks; this work only uses the primary paper/code to define the cache-vs-projection distinction and research hypotheses.
- MorphServe's dynamic layer swapping and KVResizer are not reproduced; the local study is intentionally upstream-compatible and stops before those systems features.
- No online mixed-profile continuous-batching trace has been run.
- Prompt count, task coverage, decode length, and random-seed coverage are too small for a general quality claim.

## Recommended next iteration

1. Implement a single native or library-backed W8 projection microkernel with the same group scale convention and compare quality and wall time against this proxy.
2. Expand the paired evaluation to a calibrated QAQ-style KV-cache experiment and task-level metrics, with enough prompts for confidence intervals.
3. Revisit query-specific allocation using a richer budget grid and a quality metric tied to next-token loss/task accuracy; retain the zero-gain oracle result as a baseline.
4. Only if native overhead is competitive and the larger oracle study demonstrates a material quality/memory advantage, prototype a scheduler policy. Do not add dynamic KV-cache compression, CPU/GPU bit swapping, or production kernels before those gates pass.
