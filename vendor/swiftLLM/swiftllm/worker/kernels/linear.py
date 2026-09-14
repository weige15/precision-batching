import torch

from swiftllm.precision import PrecisionProfile, quantize_dequantize


def linear(
    a: torch.Tensor,	# [a, b]
    w: torch.Tensor,		# [c, b]
    *,
    projection: str | None = None,
    precision_profiles: list[PrecisionProfile] | None = None,
    token_request_ids: list[int] | None = None,
) -> torch.Tensor:		# [a, c]
    """Run a projection, optionally using the explicit eager proxy.

    The no-profile/all-FP16 branch is intentionally the original one-line
    operation. Low-bit profiles are only a research proxy: weights are
    quantized and immediately dequantized before ``F.linear``. No packed
    representation or acceleration is claimed.
    """
    # pylint: disable=not-callable
    if projection is None or not precision_profiles:
        return torch.nn.functional.linear(a, w)

    bits = [profile.bits_for(projection) for profile in precision_profiles]
    if all(bit == 16 for bit in bits):
        return torch.nn.functional.linear(a, w)
    if token_request_ids is None or len(token_request_ids) != a.shape[0]:
        raise ValueError("token_request_ids are required for non-FP16 request profiles")

    # Different requests may select different precision for the same
    # projection. Compute each group separately; this is deliberately simple
    # and is not intended to be a serving implementation.
    output = torch.empty((a.shape[0], w.shape[0]), device=a.device, dtype=a.dtype)
    for request_id, bit in enumerate(bits):
        rows = [index for index, owner in enumerate(token_request_ids) if owner == request_id]
        if not rows:
            continue
        row_index = torch.tensor(rows, device=a.device, dtype=torch.long)
        quantized_weight = quantize_dequantize(w, bit)
        output.index_copy_(0, row_index, torch.nn.functional.linear(a.index_select(0, row_index), quantized_weight))
    return output
