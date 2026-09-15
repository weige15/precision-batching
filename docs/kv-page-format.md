# Live KV-page format and mechanism

## Scope

The SwiftLLM baseline keeps its original dense FP16 KV cache and Triton
PagedAttention path. The optional research path is enabled with
`--kv-page-format fp16|int8|int4`; the default `dense_fp16` is unchanged.
There is no scheduler decision, priority policy, CPU/GPU residual hierarchy,
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
preserves a page's current K/V formats on append, and reads each page through
a layer-aware FP16 dequantization reference before ordinary PyTorch attention.
A decode batch can therefore contain FP16, INT8, and INT4 pages at once. The
reference is deliberately transparent rather than a production fused kernel;
its timings are an upper-bound/feasibility result for this implementation,
not evidence of low-bit acceleration.

`LlamaModel.demote_kv_pages()` is the only control-plane hook needed to convert
already-live pages. Page-store mode deliberately rejects existing SwiftLLM
CPU/GPU swapping instead of pretending that compressed pages can be swapped by
the FP16 extension.
