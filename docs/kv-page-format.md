# Live KV-page format and mechanism

## Scope

The SwiftLLM baseline keeps its original dense FP16 KV cache and Triton
PagedAttention path. The runtime CLI exposes only `dense_fp16`; the page
formats remain directly constructible by the offline research harness for
reproducibility, but are not runtime actions after the break-even no-go. There
is no scheduler decision, priority policy, CPU/GPU residual hierarchy,
weight swapping, or query router in this phase.

A logical page is one `(physical_block_id, layer_id)` pair. Its K and V
components have independent format metadata, so K-only and V-only demotions
are representable. The cache stores metadata in a CPU tensor shaped
`[num_blocks, num_layers, 2]`, with separate K/V format codes. The SwiftLLM
block table continues to map a request's sequence page to a physical block.

## Representation

For a page with shape `[block_size, num_kv_heads, head_dim]`:

- FP16 stores K and V directly as FP16 tensors and has no scale tensor.
- INT8 stores one signed byte per value and one FP16 scale per group for K and
  V independently.
- INT4 stores two signed values per byte as offset nibbles and one FP16 scale
  per group for K and V independently.

The documented rule is symmetric per-group max-absolute quantization, with
`group_size=128`:

```text
qmax = 2^(bits - 1) - 1
scale = max(abs(x_group), eps) / qmax
q = round(x / scale), clamped to [-2^(bits-1), qmax]
reconstruction = q * scale
```

INT4's signed range is `[-8, 7]`, encoded as `q + 8` in each nibble. Padding
for a final incomplete group is included in the payload count. Scales are
FP16 to make the representation and byte accounting explicit. This is a
standard conservative baseline, not a new quantizer or a claim of novelty.

The page record owns its payload. Synchronous demotion replaces the FP16
record with the compressed record and returns a conversion record containing
before/after bytes, elapsed time, and temporary source bytes. Asynchronous
demotion runs on a separate CUDA stream, publishes an event dependency, and
retains the source until an explicit `synchronize()` safely reclaims it.
Consumers enqueue a wait on their current stream before decoding the published
payload, so unrelated work can overlap. A pending source is visible as
transient workspace and is not counted as reclaimed capacity.

## Mixed attention

The optional page-store path writes new pages in the configured initial format,
preserves a page's current K/V formats on append, and retains the transparent
`page_attention_for_layer` FP16-dequantization implementation as a correctness
oracle. The follow-up also provides
`worker/kernels/segmented_paged_attn.py`: format-specialized FP16 and INT8
segments read native payloads, dequantize INT8 in registers, and reduce their
online-softmax partials without materializing the mixed cache to FP16. The
transformer uses that path for matching FP16/INT8 pages and falls back to the
oracle for INT4 or unsupported K/V combinations.

`PagedKVCache.demote_pages_batch()` and `promote_pages_batch()` measure the
conversion/reversal data path as batched GPU operations. The matched
queue/offload/compression outcome is in
[`kv-break-even-report.md`](kv-break-even-report.md). Page-store mode still
deliberately rejects existing SwiftLLM CPU/GPU swapping rather than pretending
that compressed pages can be swapped by the FP16 extension; the lossless swap
comparison is measured separately with the unchanged dense mode.
