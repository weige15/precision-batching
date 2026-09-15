# Final interaction-aware structured weight-precision report

## Decision

**B — close structured weight precision for this proxy and move the next
research direction to KV-cache precision/serving behavior.**

The required staged gate was followed:

1. Llama 3.2 1B exploration found several W8-budget structured profiles with
   negative paired held-out NLL differences and stable calibration behavior.
   This opened the gated confirmation.
2. Llama 3.1 8B confirmation found no candidate with a positive held-out
   confidence signal. Its best apparent NLL improvements had bootstrap CIs
   crossing zero. The confirmation therefore failed the repeatability gate.
3. No native-kernel, scheduler, router, KV-cache, swapping, or serving-path
   work is opened by this result.

The result is negative for the tested symmetric groupwise fake-quantization
proxy and search budget. It is not a claim that every quantizer or native
format is impossible.

## Scope correction to the previous audit

The previous structured run detected strong non-additive interaction, but its
projection-only, layer-only, greedy layer-by-projection, and exact-budget DP
profiles were chosen from additive single-unit risks measured with every other
unit at FP16. Re-running those profiles as combined models did **not** make the
search interaction-aware. Those results are preserved in:

- `results/sensitivity/llama32_1b_structured.json`
- `results/sensitivity/llama31_8b_structured.json`

Their negative claim is narrowed to the specific additive-selected candidates
that were tested. They are not evidence that an interaction-aware search was
exhausted. The corrected experiment is recorded separately in:

- `results/sensitivity/llama32_1b_interaction_aware.json`
- `results/sensitivity/llama31_8b_interaction_aware.json`

The optimizer description is explicitly **exhaustive 3^5 enumeration (243
assignments)**, correcting the former reversed-exponent description.

## Experimental procedure

### W8-centered measurements

For every Q, K, V, O, and per-layer FFN unit (gate + up + down together), the
1B run starts from one full uniform-W8 model. It executes:

- one profile changing only that unit W8 → W4;
- one profile changing only that unit W8 → FP16;
- a W8 record as the paired background control.

All other units remain W8. Each measured profile records paired NLL, logit
MSE/RMSE, KL, top-1 agreement, per-sample results, per-shard results, and the
exact integer storage delta versus uniform W8. The single-unit records only
propose search moves; they are not the final search objective.

### Data separation

Calibration uses deterministic randomized, evenly spaced, non-overlapping
Wikitext-2 raw train windows:

| run | calibration | held-out validation | calibration shards |
|---|---:|---:|---:|
| 1B exploration | 12 windows | 64 windows | 3 × 4 |
| 8B confirmation | 6 windows | 32 windows | 3 × 2 |

The validation windows are dispersed and non-overlapping within the validation
split and are loaded only after the search has completed. Search sample IDs are
verified to be calibration IDs and disjoint from validation IDs. HellaSwag is
not used as a decision criterion in this phase.

### Interaction-aware search

Starting from uniform W8, the bounded deterministic beam/coordinate search:

- generates feasible W8 → W4 downgrades and paired compensating W8 → FP16
  upgrades;
- recomputes neighborhoods around accepted profiles for subsequent rounds;
- deduplicates profiles and enforces an explicit evaluation budget;
- ranks candidates using actual combined-model calibration NLL across shards,
  with worst-shard and dispersion tie-breaks;
- uses W8-centered margins only to bound deterministic proposal order.

The 1B search budget was 96 new combined profiles over three rounds with beam
width 2. It also executed all 243 projection-only assignments, recording exact
storage for each; 112 were at or below the W8 budget and 7 were within the
declared 1% close-bracket band. The 8B confirmation intentionally skipped the
1B-only exhaustive projection sanity check and repeated marginal sweep, using
48 actual combined search evaluations instead.

The final calibration frontier is the actual nondominated combined-profile
frontier under the modeled W8 storage budget. Every surviving frontier profile
is then evaluated directly against uniform W8 on held-out validation windows.

### Storage model

For each matrix, quantized storage counts input-channel padding, bit payload,
and one FP16 symmetric scale per group of 128. FP16 counts direct 16-bit
weights and no scale overhead. Fixed non-unit parameters (embeddings, LM head,
and norms) remain FP16 and are included. Totals and deltas are exact integer
bits and integer bytes; native packed headers/alignment are explicitly outside
the proxy.

## Results

### Llama 3.2 1B exploration

- Uniform W8: **12,110,036,992 bits / 1,513,754,624 bytes**.
- Uniform W4: 8,217,722,880 bits / 1,027,215,360 bytes.
- Uniform FP16: 19,773,030,400 bits / 2,471,628,800 bytes.
- 80 W8-centered units measured at W4 and FP16.
- 243/243 projection-only assignments executed as combined profiles.
- 339 unique combined calibration profiles, 96 new search evaluations.
- 23 final held-out frontier profiles, each directly paired to W8 over 64
  validation windows.

Five frontier search profiles passed the 1B provisional gate. A representative
one downgraded `layer_004/v` and `layer_004/ffn` to W4, used 205,520,896 fewer
modeled bits (25,690,112 fewer bytes), and had paired held-out NLL difference
`-0.00421` with 95% bootstrap interval `[-0.00746, -0.00090]`. The calibration
improvement was stable under the declared three-shard rule. This provisional
signal was enough to open the 8B confirmation, but was not treated as final.

### Llama 3.1 8B confirmation

- Uniform W8: **73,522,020,352 bits / 9,190,252,544 bytes**.
- 48 new actual combined search evaluations under the declared budget.
- 11 held-out frontier profiles, each directly paired to W8 over 32 dispersed
  validation windows.
- No 8B candidate passed both the positive-confidence and stable-calibration
  gate.

The best apparent candidates were below W8 storage, and some were stable on
calibration, but their held-out paired NLL confidence intervals crossed zero:

| modeled storage delta vs W8 | paired NLL mean | 95% bootstrap CI |
|---:|---:|---:|
| -50,855,936 bits (-6,356,992 bytes) | -0.00227 | [-0.00558, 0.00105] |
| -17,825,792 bits (-2,228,224 bytes) | -0.00212 | [-0.00555, 0.00130] |
| -16,777,216 bits (-2,097,152 bytes) | -0.00251 | [-0.00594, 0.00078] |

Because the upper confidence bound was not below zero, the 8B result does not
confirm a repeatable Pareto advantage. The final artifact records
`NO_GO_CLOSE_STRUCTURED_WEIGHT_PRECISION` and the next direction as
KV-cache precision/serving behavior.

## Verification and reproduction

The artifact verifier checks actual procedure, not only flags or strings:

- W8-centered marginal coverage and one-unit profile structure;
- exact profile storage and signed deltas in bits/bytes;
- three-shard dispersed-window separation and held-out non-leakage;
- actual combined calibration executions for every candidate;
- all 243 projection assignments in the 1B exploration;
- deterministic search history, deduplication, and budget;
- paired held-out frontier metrics and bootstrap intervals;
- staged 1B/8B gate decisions and confirmation ordering.

Run:

```bash
.venv/bin/python scripts/structured_precision_experiment.py --storage-self-test
PYTHONPATH="$PWD/vendor/swiftLLM:$PWD/vendor/swiftLLM/csrc" \
  .venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/verify_artifacts.py
```

The recorded commands and output logs are the `*_interaction_aware.log` files
next to the two JSON artifacts. The 1B exploration command was:

```bash
CUDA_VISIBLE_DEVICES=3 .venv/bin/python scripts/structured_precision_experiment.py \
  --model-path /nfs/home/s314511048/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6 \
  --output results/sensitivity/llama32_1b_interaction_aware.json --device cuda:0 \
  --calibration-shards 3 --calibration-samples-per-shard 4 \
  --heldout-shards 4 --heldout-samples-per-shard 16 --batch-size 2 \
  --search-evaluation-budget 96 --search-rounds 3 --beam-width 2 \
  --proposal-width 8 --bootstrap-iterations 1000
```

After that artifact's `OPEN_8B_CONFIRMATION` gate, the confirmation used a
bounded actual search; it intentionally skipped the 1B-only exhaustive
projection check and repeated marginal sweep:

```bash
CUDA_VISIBLE_DEVICES=3 .venv/bin/python scripts/structured_precision_experiment.py \
  --run-mode confirmation --skip-projection-enumeration --skip-w8-marginals \
  --model-path /nfs/home/s314511048/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b \
  --output results/sensitivity/llama31_8b_interaction_aware.json --device cuda:0 \
  --calibration-shards 3 --calibration-samples-per-shard 2 \
  --heldout-shards 4 --heldout-samples-per-shard 8 --batch-size 1 \
  --search-evaluation-budget 48 --search-rounds 2 --beam-width 2 \
  --proposal-width 8 --bootstrap-iterations 1000
```

The experiment remains an offline fake-quant proxy: do not infer native speedup
or serving-memory behavior from it.

## Explicitly deferred work

Do not implement native W4/W8 kernels, query routing, precision-aware
scheduling, KV-cache quantization, swapping, or serving-path optimization as a
consequence of this experiment. Query-specific oracle work remains paused
because no multiple globally competitive profiles survived the 8B gate.
