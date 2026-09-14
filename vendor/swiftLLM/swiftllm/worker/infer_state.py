import dataclasses

import torch

from swiftllm.precision import PrecisionProfile

@dataclasses.dataclass
class LlamaInferState:
    batch_size: int
    num_tokens: int

    seq_ids: torch.Tensor   # [batch_size]
    softmax_scale: float    # Equal to 1/sqrt(head_dim)

    num_prefill_seqs: int
    num_prefill_tokens: int
    prefill_seq_start_locs: torch.Tensor # [batch_size]
    prefill_seq_start_locs_with_end: torch.Tensor # [batch_size+1], = prefill_seq_start_locs + [num_prefill_tokens]
    prefill_seq_lens: torch.Tensor # [batch_size]
    max_prefill_len: int

    num_decoding_seqs: int
    decoding_seq_lens: torch.Tensor # [batch_size]
    max_decoding_len: int

    seq_block_size: int
    num_seq_blocks: int

    position_cos: torch.Tensor	# [num_tokens, hidden_size]
    position_sin: torch.Tensor	# [num_tokens, hidden_size]

    ignore_kvcache: bool    # Skip storing the key/value cache, useful when profiling the number of kv blocks

    # Request-level precision metadata.  ``token_request_ids`` maps each row
    # in the flattened input to an entry in ``precision_profiles``.  Empty
    # metadata means the legacy all-FP16 path.
    precision_profiles: list[PrecisionProfile] = dataclasses.field(default_factory=list)
    token_request_ids: list[int] = dataclasses.field(default_factory=list)
