# Primary-source notes: QAQ and MorphServe

These notes distinguish what the papers actually study from the projection-precision question in this repository. The PDFs used for the notes are checked in under `references/` and have the SHA-256 values below.

## QAQ

- **Paper:** Shichen Dong, Wen Cheng, Jiayu Qin, and Wei Wang, “QAQ: Quality Adaptive Quantization for LLM KV Cache,” arXiv:2403.04643, 2024.
- **Primary source:** <https://arxiv.org/abs/2403.04643>
- **Code:** <https://github.com/ClubieDong/KVCacheQuantization>, fetched at commit `f8d47e0967c5c5f67f156c1f391a02b5cbd8183f` (recorded in `vendor/qaq/UPSTREAM_COMMIT` and `results/baseline/qaq_git_verification.txt`).
- **Local PDF:** `references/qaq-2403.04643.pdf`
- **PDF SHA-256:** `cfc68f45ba813fe7dade1d6ace48f5fa83be13ee172e7c732f7c1aea24822fab`

### Confirmed relevant content

1. QAQ is a **KV-cache** quantization method, not a Q/K/V projection-weight quantization method. Section 3.1 derives different sensitivities for key and value cache and reports an individual key/value experiment on LLaMA 2-7B with 1,000 HellaSwag questions.
2. Section 4.1 makes key-cache precision depend on the squared query norm and value-cache precision depend on attention values. Section 4.2 uses an attention window to handle exceptions to persistence of attention importance and preserves outliers in full precision.
3. The method is applied during autoregressive generation to previously generated K/V cache entries. Its reported headline claim is nearly 10x KV-cache compression with negligible reported accuracy loss on its LLaMA 2 7B/13B downstream-task experiments. Those numerical claims are paper claims, not results reproduced here.
4. The released implementation (`vendor/qaq/src/quantizer.py`) exposes separate key and value quantizers and quantization levels (`token`, `layer`, `head`). It does not expose separate Q/K/V **projection** weight profiles.

### Implication for this project

QAQ is direct motivation for testing structured K/V sensitivity and query dependence, but it does not establish that runtime precision changes for the **Q, K, and V projection GEMMs** are useful. The present harness therefore tests that missing link separately and labels its weight fake-quantization proxy.

## MorphServe

- **Paper:** Zhaoyuan Su, Zeyu Zhang, Tingfeng Lan, Zirui Wang, Haiying Shen, Juncheng Yang, and Yue Cheng, “MorphServe: Efficient and Workload-Aware LLM Serving via Runtime Quantized Layer Swapping and KV Cache Resizing,” arXiv:2506.02006v2, 2026-01-07.
- **Primary source:** <https://arxiv.org/abs/2506.02006>
- **Local PDF:** `references/morphserve-2506.02006-v2.pdf`
- **PDF SHA-256:** `e2c0f12fcbc5188a04078a31a732acaa95e93e9662aff4b766a0b9d6a73fbe30`

### Confirmed relevant content

1. MorphServe is implemented on top of SwiftLLM and proposes two runtime mechanisms: **LayerSwapper** (selective decoder-layer replacement with pre-quantized alternatives) and **KVResizer** (elastic allocation/deallocation of KV-cache blocks under memory pressure).
2. Section 4.2 ranks layers offline using Layer Transformation Sensitivity, Layer Replacement Sensitivity, and Model Degradation Sensitivity, combined as a Layer Importance Score (LIS). This is layer-level, not per-request Q/K/V projection sensitivity.
3. Section 4.3 describes preloaded full/quantized layer variants and asynchronous in-place swapping. Section 4.4 explicitly says KVResizer does not quantize or compress existing KV caches; it reallocates capacity made available by weight quantization.
4. Section 5 reports the paper's serving results on Llama/Vicuna workloads, including claimed SLO/TTFT improvements. The paper reports an implementation addition of approximately 2,200 Python lines and 500 C++/CUDA lines on SwiftLLM. These results are not treated as evidence of a native mixed-QKV kernel here.

### Implication for this project

MorphServe supports the systems hypothesis that offline sensitivity can drive runtime layer-level adaptation, and it supplies a relevant serving architecture. It does not answer whether per-request, per-projection Q/K/V precision has enough query-dependent value to justify finer-grained scheduling. The baseline deliberately stops before implementing MorphServe's full layer swapping or KV resizing.

## Scope boundary

Neither paper is evidence that this repository's eager FP16-after-dequantization proxy provides low-bit speedup. All local low-bit timings are overhead measurements only. Native packed kernels, KV-cache quantization, dynamic CPU/GPU bit-plane movement, and a precision-aware continuous-batching scheduler remain out of scope until the sensitivity evidence warrants them.
