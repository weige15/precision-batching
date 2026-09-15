# Available local assets

Paths are machine-local and are not required to exist in a clean checkout. The experiment JSON records the model file hashes used for every run.

| Asset | Local path | Use / status |
|---|---|---|
| Llama 3.2 1B Instruct | `/nfs/home/s314511048/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6` | Primary SwiftLLM baseline and 8/16-layer sensitivity runs |
| Llama 3.1 8B | `/nfs/home/s314511048/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b` | Secondary 27-profile and 32-layer sensitivity runs |
| MorphServe Llama 3.1 8B AWQ W4 G128 zero-point | `/nfs/home/s314511048/.cache/morphserve/llama31-8b-autoawq-w4-g128-zp` | Available, but blocked for ordinary pinned SwiftLLM loader because it contains AWQ-packed tensors |
| Wikitext-2 raw train/validation Arrow cache | `/nfs/home/s314511048/.cache/huggingface/datasets/Salesforce___wikitext/wikitext-2-raw-v1` | Calibration and held-out next-token evaluation source |
| HellaSwag validation cache | `/nfs/home/s314511048/.cache/huggingface/datasets/Rowan___hellaswag` | Held-out teacher-forced task accuracy source |

The primary model has 16 layers, hidden size 2,048, 32 query heads, 8 KV heads, and a 2,048-MiB `model.safetensors` file. The secondary model has 32 layers, hidden size 4,096, 32 query heads, and 8 KV heads. The AWQ asset is approximately 5.4 GB and its `config.json` declares `quant_method: awq`, `bits: 4`, `group_size: 128`, and `zero_point: true`; its two weight files are approximately 4.0 GB and 1.7 GB.

See:

- `results/baseline/environment_snapshot.json`
- `results/baseline/environment_snapshot_llama31_8b.json`
- `results/baseline/environment_snapshot_morphserve_awq.json`
- the `model.files` sections of each sensitivity JSON
