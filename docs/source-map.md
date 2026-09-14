# Pinned SwiftLLM execution map

The research fork is `vendor/swiftLLM`. The byte-for-byte source snapshot before research edits is `vendor/swiftLLM-upstream`; both are pinned to upstream commit `682cf9a28f97f7490409981a2f181528f377eb5d`. The tracked edit is also captured in `references/swiftllm-research.diff`.

Line numbers below refer to the research fork at the time this document was written. Use `nl -ba` after edits to refresh them.

## Request to scheduler

1. `vendor/swiftLLM/swiftllm/server/api_server.py:17-39` validates `/generate` JSON into `RawRequest`, including optional `precision_profile` metadata; validation errors are returned as HTTP 422.
2. `vendor/swiftLLM/swiftllm/server/engine.py:65-86` creates a `Request`, queues it, and exposes streaming/non-streaming request completion.
3. `vendor/swiftLLM/swiftllm/server/engine.py:89-112` batches raw prompts, calls Ray `TokenizationEngine.batched_tokenize`, fills `prompt_token_ids`/`prompt_len`, then calls `Scheduler.on_requests_arrival`.
4. `vendor/swiftLLM/swiftllm/server/tokenization_engine.py:6-14` owns the tokenizer actor and returns batched token IDs.
5. `vendor/swiftLLM/swiftllm/server/scheduler.py:33-120` maintains FCFS `waiting_q`, `running_q`, and `swapped_q`; `get_next_batch` admits prefills, decodes, and swaps based only on sequence lengths, block capacity, and batch/token limits. It does not inspect precision metadata.
6. `vendor/swiftLLM/swiftllm/server/engine.py:115-171` asks the scheduler for a batch, performs KV swaps, constructs prefill/decode inputs, and now passes one `PrecisionProfile` per request to `model.forward`. The scheduler decisions and arguments remain unchanged.

## Model and Q/K/V path

1. `vendor/swiftLLM/swiftllm/worker/model.py:253-378` flattens batch inputs, derives prefill/decode positions, allocates PagedAttention blocks, builds `LlamaInferState`, and calls `_forward`.
2. `vendor/swiftLLM/swiftllm/worker/model.py:229-251` calls the embedding layer, each transformer layer, and the final projection.
3. `vendor/swiftLLM/swiftllm/worker/layers/transformer_layer.py:31-174` executes one layer. The projection call sites are:
   - Q: lines 54-60, `self.weight.q_proj`;
   - K: lines 61-67, `self.weight.k_proj`;
   - V: lines 68-74, `self.weight.v_proj`;
   - rotary embedding: lines 76-81;
   - KV-cache store: lines 83-96;
   - prefill FlashAttention: lines 100-112;
   - decode PagedAttention: lines 116-133;
   - O: lines 135-142, `self.weight.o_proj`;
   - FFN up/gate and down: lines 150-168.
4. `vendor/swiftLLM/swiftllm/worker/kernels/linear.py:6-43` keeps the original `torch.nn.functional.linear` operation for missing/all-FP16 metadata. For opted-in low-bit metadata it invokes the explicit eager fake-quantize/dequantize proxy and computes request-row groups separately; it does not pack weights or claim acceleration.
5. `vendor/swiftLLM/swiftllm/worker/kernels/kvcache_mgmt.py:81-121` writes K and V into the FP16 paged cache using the block table and layer ID.
6. `vendor/swiftLLM/swiftllm/worker/kernels/paged_attn.py:152-221` reads the FP16 K/V cache for decode attention through the block table.
7. `vendor/swiftLLM/swiftllm/worker/model.py:138-178` allocates FP16 GPU/CPU K/V cache and initializes the block managers. KV-cache quantization is not implemented.

## Metadata seam

- `vendor/swiftLLM/swiftllm/precision.py:22-105` defines immutable `PrecisionProfile(q_bits, k_bits, v_bits, o_bits, ffn_bits)`, validation, JSON mapping, and the clearly labeled numerical proxy.
- `vendor/swiftLLM/swiftllm/server/structs.py:17-69` attaches a profile to `RawRequest` and copies it to `Request`.
- `vendor/swiftLLM/swiftllm/worker/infer_state.py:28-35` carries the profile list and flattened-token-to-request mapping.
- `vendor/swiftLLM/swiftllm/worker/model.py:262-304` validates one profile per request and constructs the flattened mapping.

## Baseline interpretation

The upstream code's normal path is preserved in `vendor/swiftLLM-upstream`; `results/baseline/swiftllm_unmodified_1b.log` is a run from that code before the research edits. `results/baseline/swiftllm_precision_noop_1b.log` is the corresponding research-fork run. Their generated token lines match exactly; model initialization time is intentionally excluded from that comparison.
