# Structured layer-by-projection precision report

## Decision summary

The current evidence does **not** establish a held-out quality/storage Pareto advantage for structured layer-by-projection weight precision over uniform precision. The calibration-only combined-profile selector picks the greedy layer-by-projection candidate in both models, but that candidate is worse on held-out next-token NLL; the best held-out W8 candidate is uniform W8. The favorable per-query oracle has only small, metric-dependent NLL headroom and worse NLL-selected logit MSE.

This is a model-level result for the checked-in symmetric groupwise fake-quantization proxy. It is not a claim that native W4/W8 kernels are impossible; it is a gate against implementing them for a structured-policy advantage that has not yet been demonstrated.

### Separate go/no-go decisions

1. **Static layer-by-projection mixed precision: NO-GO for implementation as the next production direction.** The selected structured policy has no held-out Pareto advantage over uniform W8 and is worse in both models. A future study may revisit this with a different quantizer or larger benchmark, but the current result does not justify the static policy.
2. **Native W4/W8 kernel implementation: NO-GO under the requested gate.** The proxy is not a kernel, but the objective requires a defensible structured quality/storage Pareto advantage before native-kernel work receives a go. That gate is not met. Uniform-kernel work would be a separate question.
3. **Query-conditioned precision-aware continuous batching: NO-GO.** The favorable offline oracle's NLL headroom is only about 0.16–0.17% and its NLL-selected logit MSE is worse. This is not substantial, metric-consistent headroom. Do not train a router or alter the scheduler.

## Reproduction

The main experiment is `scripts/structured_precision_experiment.py`. It uses local Hugging Face datasets and local model snapshots; it does not contact a serving scheduler.

```bash
# Build the pinned SwiftLLM extension once if the checkout imports need it.
(cd vendor/swiftLLM/csrc && ../../../.venv/bin/python setup.py build_ext --inplace)

MODEL=/nfs/home/s314511048/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6
CUDA_VISIBLE_DEVICES=5 .venv/bin/python scripts/structured_precision_experiment.py \
  --model-path "$MODEL" \
  --output results/sensitivity/llama32_1b_structured.json \
  --device cuda:0 --calibration-samples 16 --heldout-samples 32 \
  --task-samples 32 --seq-len 128 --batch-size 2 --bootstrap-iterations 1000

MODEL=/nfs/home/s314511048/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b
CUDA_VISIBLE_DEVICES=5 .venv/bin/python scripts/structured_precision_experiment.py \
  --model-path "$MODEL" \
  --output results/sensitivity/llama31_8b_structured.json \
  --device cuda:0 --calibration-samples 8 --heldout-samples 16 \
  --task-samples 16 --seq-len 128 --batch-size 1 --bootstrap-iterations 1000
```

The exact local counts are intentional and recorded in each JSON. The 1B run is the broad exploration (16 calibration sequences, 32 held-out sequences, 32 HellaSwag items); the 8B run is a confirmation under the available compute (8, 16, and 16). These are larger and more relevant than the original eight prompts, but still not publication-scale benchmark uncertainty. The smaller 8B sample is an explicit limitation, not silently treated as definitive.

The verification surface is:

```bash
.venv/bin/python scripts/structured_precision_experiment.py --storage-self-test
PYTHONPATH="$PWD/vendor/swiftLLM:$PWD/vendor/swiftLLM/csrc" \
  .venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/verify_artifacts.py
```

## Experimental design

### Units and policies

Every transformer layer has five independently measured units: Q, K, V, O, and an FFN block consisting of gate, up, and down projections quantized together. Each unit is measured at 4, 8, and 16 bits. The 16-bit record is the explicit no-op control. Candidate profiles are selected from calibration single-unit NLL deltas using:

- uniform W4/W8/FP16;
- projection-only exhaustive enumeration over one bit per Q/K/V/O/FFN type;
- layer-only deterministic greedy upgrades, with one bit for all five units in a layer;
- layer-by-projection deterministic greedy upgrades, plus an exact integer-budget meet-in-the-middle DP assignment;
- an interpretable V-first upgrade heuristic.

The selected profiles are executed as combined profiles on both calibration and held-out text. Combined quality is not inferred from the single-unit measurements. The artifact reports predicted-vs-measured additive error and bootstrap intervals for the interaction residual.

### Storage accounting

The compared model weight ledger includes every Q/K/V/O/FFN unit plus all remaining model parameters (embeddings, untied LM head, norms, and other parameters) as fixed FP16 storage. For each quantized matrix it counts:

- bit payload for input-channel-padded groups of 128;
- one FP16 symmetric scale per group;
- zero zero-point bits, because the proxy is symmetric;
- no unmodeled packed-header bits (explicitly recorded as an unknown for a native format).

FP16 matrices count direct 16-bit weights and **do not pay scale overhead**. All profile totals are stored as integer bits and bytes in the raw JSON. Profiles are called exactly equal only when their integer total-bit counts are equal. Greedy mixed profiles can be slightly below W8 because removing FP16 scale overhead makes discrete 4/8/16 assignments difficult to match; the exact integer-budget DP profile is included to remove that ambiguity.

This is a true representation-aware **weight** budget for the modeled format, not total serving memory: KV-cache storage, allocator blocks, kernel packing, and scheduler metadata are outside this experiment and unchanged across profiles.

### Held-out metrics

The headline objective is held-out next-token NLL/perplexity on Wikitext-2 validation, paired against a fresh FP16 reference for logit MSE, KL, and top-1 agreement. A held-out HellaSwag validation subset supplies a teacher-forced multiple-choice accuracy metric. Calibration uses Wikitext-2 train and never selects a profile using held-out values. Sequence/item bootstrap 95% intervals are recorded in each policy row.

The per-query oracle chooses, after seeing held-out NLL, the best among the executed profiles at the closest modeled W8 storage. It is therefore favorable and an upper bound, not a realizable router.

## Results: Llama 3.2 1B exploration

The model has 16 layers and the W8 unit-plus-fixed-weight ledger is 12,110,036,992 bits (1,513,754,624 bytes). Uniform W4 is 8,217,722,880 bits and uniform FP16 is 19,773,030,400 bits.

| profile | true total bits | gap vs W8 | held-out NLL delta vs FP16 | bootstrap 95% interval | HellaSwag accuracy |
|---|---:|---:|---:|---:|---:|
| uniform W4 | 8,217,722,880 | -3,892,314,112 | 0.23993 | [0.20946, 0.27081] | 17/32 |
| uniform W8 | 12,110,036,992 | 0 | 0.00339 | [0.00153, 0.00507] | 18/32 |
| projection-only | 12,110,036,992 | 0 | 0.00339 | [0.00161, 0.00501] | 18/32 |
| layer-only | 12,110,036,992 | 0 | 0.00339 | [0.00174, 0.00509] | 18/32 |
| layer-by-projection (greedy) | 12,108,988,416 | -1,048,576 | 0.01871 | [0.00987, 0.02850] | 18/32 |
| V-first heuristic | 12,108,857,344 | -1,179,648 | 0.02384 | [0.01663, 0.03153] | 18/32 |
| layer-by-projection (exact W8) | 12,110,036,992 | 0 | 0.01642 | [0.00759, 0.02499] | 18/32 |
| uniform FP16 | 19,773,030,400 | +7,662,993,408 | 0 | [0, 0] | 18/32 |

The exact W8-budget oracle candidate set contains uniform W8, projection-only, layer-only, and the exact layer-by-projection assignment. The combined calibration selector chose the greedy layer-by-projection profile, whose held-out NLL delta is 0.01871; the best held-out candidate is uniform W8. The favorable held-out NLL oracle reduces NLL by 0.00527 [0.00225, 0.00874] in absolute terms (about 0.17% of global NLL); it selects mixed profiles on some sequences. However, the NLL-selected oracle has a logit-MSE ratio of **13.99x** versus global W8 (equivalent to a -1,299% reduction), so this is small metric-specific headroom rather than a robust quality-routing opportunity.

The single-unit calibration surface does show nonuniformity: mean W4 NLL deltas by unit type were Q -0.00002, K 0.00091, V 0.00115, O 0.00142, and FFN 0.00814. This is a sensitivity observation, not evidence that a combined allocation is additive or beneficial. The combined layer-by-projection profile's calibration NLL interaction was -0.05937 (measured minus nonnegative additive prediction), with a bootstrap interval [-0.06910, -0.05135]; that large mismatch is direct evidence against using the single-unit sum as a quality predictor without further interaction modeling.

## Results: Llama 3.1 8B confirmation

The model has 32 layers and the W8 unit-plus-fixed-weight ledger is 73,522,020,352 bits (9,190,252,544 bytes). Uniform W4 is 45,604,732,928 bits and uniform FP16 is 128,484,179,968 bits.

| profile | true total bits | gap vs W8 | held-out NLL delta vs FP16 | bootstrap 95% interval | HellaSwag accuracy |
|---|---:|---:|---:|---:|---:|
| uniform W4 | 45,604,732,928 | -27,917,287,424 | 0.13662 | [0.10353, 0.17587] | 10/16 |
| uniform W8 | 73,522,020,352 | 0 | 0.00067 | [-0.00151, 0.00282] | 10/16 |
| projection-only | 73,522,020,352 | 0 | 0.00067 | [-0.00147, 0.00278] | 10/16 |
| layer-only | 73,494,757,376 | -27,262,976 | 0.00806 | [0.00365, 0.01283] | 10/16 |
| layer-by-projection (greedy) | 73,507,340,288 | -14,680,064 | 0.03819 | [0.02492, 0.05441] | 10/16 |
| V-first heuristic | 73,512,583,168 | -9,437,184 | 0.03537 | [0.02208, 0.04886] | 9/16 |
| layer-by-projection (exact W8) | 73,522,020,352 | 0 | 0.02793 | [0.01237, 0.04618] | 10/16 |
| uniform FP16 | 128,484,179,968 | +54,962,159,616 | 0 | [0, 0] | 10/16 |

The combined calibration selector again chose the greedy layer-by-projection profile, whose held-out NLL delta is 0.03819; the best held-out candidate is uniform W8. The exact W8-budget oracle has a favorable held-out NLL reduction of 0.00376 [0.00058, 0.00733] (about 0.16% of global NLL), but its NLL-selected logit-MSE ratio is **8.83x** versus global W8 (equivalent to a -783% reduction). This is not substantial, metric-consistent headroom for a router. The smaller 8B sample makes the estimate a confirmation, not a universal benchmark claim.

The additive model is again invalid as a direct allocation predictor: the layer-by-projection calibration interaction was -0.08156 NLL (measured minus predicted), with bootstrap interval [-0.09901, -0.06650]. The sign and magnitude should not be interpreted as a general beneficial interaction; it means the independent-unit error sum overpredicts this combined profile on this calibration sample.

## Corrected Q/K/V observation

The older checked-in Q/K/V matrix remains useful as a secondary observation. V-only W4 perturbation has larger aggregate logit MSE than either Q-only or K-only in both models and both recorded phases:

| model / phase | Q-only | K-only | V-only |
|---|---:|---:|---:|
| Llama 3.2 1B / prefill | 0.01902 | 0.01799 | 0.08432 |
| Llama 3.2 1B / decode | 0.04160 | 0.02545 | 0.13605 |
| Llama 3.1 8B / prefill | 0.04311 | 0.05282 | 0.14014 |
| Llama 3.1 8B / decode | 0.01476 | 0.01134 | 0.07989 |

The defensible statement is **V is more sensitive than Q and K in this checked proxy/sample**. There is no fixed Q-vs-K ordering: Q is above K in 1B decode and 8B decode, while K is above Q in 8B prefill (and slightly below Q in 1B prefill). The repository no longer treats Q>K or K>Q as a universal rule. These are weight-projection measurements, not QAQ's KV-cache quantization result.

## Confirmed findings

- The pinned SwiftLLM baseline and research no-op remain byte-for-byte equal after excluding the creation-time line; the nine local unit tests and artifact verifier pass.
- Q/K/V/O/FFN units are measured at 4/8/16 bits for every layer in both models. FFN is explicitly gate/up/down as a single block unit.
- Held-out text NLL, paired logits/KL/top-1, task accuracy, and bootstrap intervals are present; headline profile selection is calibration-only.
- FP16 has no scale overhead; quantized costs include group padding and FP16 scales, and fixed non-unit model weights are included in the ledger.
- Combined-profile execution, rather than proxy arithmetic alone, is used for all policy claims. The additive single-unit predictor is demonstrably unreliable in these runs.
- The robust Q/K/V observation is V-vs-Q/K, not a fixed Q-vs-K ordering.

## Supported but uncertain findings

- The model-level mixed profiles are worse than uniform W8 in both checked models, but the proxy is symmetric fake quantization rather than AWQ/GPTQ or a packed native representation.
- HellaSwag accuracy is flat at these small subsets and therefore cannot distinguish the policies. The Wikitext validation samples are useful paired evidence, not a complete task-quality evaluation.
- Layer and unit sensitivity varies, but this variance does not transfer additively to a useful combined policy. The 8B confirmation has only 16 held-out text sequences and 16 task items.
- The comparison excludes native packed-header/alignment costs and KV-cache memory. Those omissions are constant or unknown for the modeled weight-policy comparison, but they prevent an end-to-end serving-memory claim.

## Blocked questions and explicit non-work

- Native packed W4/W8 kernels, AWQ/GPTQ calibration, and realistic kernel timing were not implemented.
- KV-cache quantization, CPU/GPU bit-plane swapping, precision-aware continuous batching, and query routers were not implemented.
- MorphServe and QAQ published task/serving results were not independently reproduced.
- Long free-generation quality and broad benchmark confidence remain open.

The existing per-request metadata path remains an instrumentation seam only. It must not be mistaken for an efficient mixed-profile execution design.
