"""Precision metadata and explicit numerical proxy helpers.

The serving path only needs a small, immutable request-level description of
precision.  Native low-bit kernels are intentionally *not* part of this
module.  ``quantize_dequantize`` is an offline/eager proxy for experiments;
it materializes a dequantized FP tensor and must not be interpreted as a
low-bit speedup.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any, Literal


Projection = Literal["q", "k", "v", "o", "ffn"]
SUPPORTED_BITS = (4, 8, 16)


@dataclasses.dataclass(frozen=True)
class PrecisionProfile:
    """Per-request weight precision metadata for the model projections.

    A profile is deliberately independent of scheduling.  The default profile
    is all-FP16 and is the only profile used unless a caller opts in.
    """

    q_bits: int = 16
    k_bits: int = 16
    v_bits: int = 16
    o_bits: int = 16
    ffn_bits: int = 16

    def __post_init__(self) -> None:
        for name in ("q_bits", "k_bits", "v_bits", "o_bits", "ffn_bits"):
            bits = getattr(self, name)
            if bits not in SUPPORTED_BITS:
                raise ValueError(f"{name} must be one of {SUPPORTED_BITS}, got {bits!r}")

    @classmethod
    def default(cls) -> "PrecisionProfile":
        return cls()

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "PrecisionProfile":
        """Build a profile from API/config JSON, rejecting unknown fields."""
        if value is None:
            return cls.default()
        allowed = {"q_bits", "k_bits", "v_bits", "o_bits", "ffn_bits"}
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"unknown precision profile fields: {sorted(unknown)}")
        return cls(**{key: value[key] for key in allowed if key in value})

    def bits_for(self, projection: Projection) -> int:
        return getattr(self, f"{projection}_bits")

    def as_dict(self) -> dict[str, int]:
        return dataclasses.asdict(self)


# Kept as a function so the offline harness and the optional eager model proxy
# share exactly one clearly labeled quantization rule.
def quantize_dequantize(weight, bits: int, group_size: int = 128):
    """Symmetric per-input-group weight-only fake quantization.

    The return value is still an FP tensor.  This is a numerical proxy for
    sensitivity experiments, not a packed low-bit representation or a native
    execution kernel.
    """
    import torch

    if bits == 16:
        return weight
    if bits not in (4, 8):
        raise ValueError(f"proxy supports 4, 8, or 16 bits, got {bits}")
    original_shape = weight.shape
    input_size = original_shape[-1]
    padded_size = ((input_size + group_size - 1) // group_size) * group_size
    flat_weight = weight.float().reshape(-1, input_size)
    if padded_size != input_size:
        flat_weight = torch.nn.functional.pad(flat_weight, (0, padded_size - input_size))
    flat = flat_weight.reshape(-1, group_size)
    qmax = (1 << (bits - 1)) - 1
    scale = flat.abs().amax(dim=-1, keepdim=True).clamp_min(torch.finfo(torch.float32).eps) / qmax
    quantized = torch.round(flat / scale).clamp(-qmax - 1, qmax)
    dequantized = (quantized * scale).reshape(-1, padded_size)[:, :input_size]
    return dequantized.reshape(original_shape).to(dtype=weight.dtype)


def all_fp16(profiles: list[PrecisionProfile] | None) -> bool:
    """Whether a profile list is empty/default-only and can use the old path."""
    return not profiles or all(profile == PrecisionProfile.default() for profile in profiles)
