"""Explicit mixed-format paged KV-cache storage.

This module is a small research mechanism, not a production cache allocator.
A logical page is one ``(block_id, layer_id)`` pair and contains independent K
and V payloads.  Existing FP16 pages and symmetric INT8/INT4 pages can coexist.
Compressed payloads own their storage; after a synchronous conversion the old
FP16 tensor is no longer retained.
"""

from __future__ import annotations

import dataclasses
import math
import time
from collections import Counter
from collections.abc import Iterable
from typing import Literal

import torch
import torch.nn.functional as F


PageFormat = Literal["fp16", "int8", "int4"]
SUPPORTED_PAGE_FORMATS: tuple[PageFormat, ...] = ("fp16", "int8", "int4")
_FORMAT_BITS = {"fp16": 16, "int8": 8, "int4": 4}
_FORMAT_CODES = {"fp16": 0, "int8": 1, "int4": 2}


def _check_format(value: str) -> PageFormat:
    if value not in SUPPORTED_PAGE_FORMATS:
        raise ValueError(f"page format must be one of {SUPPORTED_PAGE_FORMATS}, got {value!r}")
    return value  # type: ignore[return-value]


def _payload_bytes(payload: torch.Tensor | None) -> int:
    return 0 if payload is None else payload.numel() * payload.element_size()


def _encode_tensor(
    value: torch.Tensor,
    page_format: PageFormat,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor | None, int]:
    """Encode one page tensor with symmetric per-group max-abs quantization."""
    page_format = _check_format(page_format)
    value = value.contiguous()
    if page_format == "fp16":
        return value.to(dtype=torch.float16), None, value.numel()

    flat = value.float().reshape(-1)
    padded_numel = math.ceil(flat.numel() / group_size) * group_size
    if padded_numel != flat.numel():
        flat = F.pad(flat, (0, padded_numel - flat.numel()))
    groups = flat.reshape(-1, group_size)
    qmax = (1 << (_FORMAT_BITS[page_format] - 1)) - 1
    scales = groups.abs().amax(dim=1).clamp_min(torch.finfo(torch.float32).eps) / qmax
    quantized = torch.round(groups / scales[:, None]).clamp(-qmax - 1, qmax).to(torch.int8)
    quantized = quantized.reshape(-1)
    if page_format == "int8":
        payload = quantized
    else:
        # Symmetric INT4 uses signed values [-8, 7], stored as two offset
        # nibbles per byte.  group_size is even by contract below.
        unsigned = (quantized.to(torch.int16) + 8).to(torch.uint8)
        payload = unsigned[0::2] | (unsigned[1::2] << 4)
    return payload, scales.to(torch.float16), padded_numel


def _decode_tensor(
    payload: torch.Tensor,
    scales: torch.Tensor | None,
    page_format: PageFormat,
    shape: tuple[int, ...],
    padded_numel: int,
    group_size: int,
) -> torch.Tensor:
    page_format = _check_format(page_format)
    if page_format == "fp16":
        return payload.reshape(shape)
    if scales is None:
        raise ValueError("compressed page is missing scales")
    if page_format == "int8":
        quantized = payload.to(torch.float32).reshape(-1)
    else:
        low = payload & 0x0F
        high = (payload >> 4) & 0x0F
        quantized = torch.stack((low, high), dim=1).reshape(-1)[:padded_numel].to(torch.float32) - 8
    scale_values = scales.to(torch.float32).repeat_interleave(group_size)[:padded_numel]
    return (quantized * scale_values)[: math.prod(shape)].reshape(shape).to(torch.float16)


@dataclasses.dataclass
class KVPage:
    """One logical page with independently formatted K and V components."""

    block_id: int
    layer_id: int
    k_format: PageFormat
    v_format: PageFormat
    k_payload: torch.Tensor
    v_payload: torch.Tensor
    k_scales: torch.Tensor | None
    v_scales: torch.Tensor | None
    shape: tuple[int, ...]
    k_padded_numel: int
    v_padded_numel: int
    group_size: int

    @property
    def allocated_bytes(self) -> int:
        return sum((
            _payload_bytes(self.k_payload),
            _payload_bytes(self.v_payload),
            _payload_bytes(self.k_scales),
            _payload_bytes(self.v_scales),
        ))

    @property
    def has_fp16_payload(self) -> bool:
        return self.k_format == "fp16" or self.v_format == "fp16"

    def decode(self, component: str) -> torch.Tensor:
        if component == "k":
            return _decode_tensor(
                self.k_payload, self.k_scales, self.k_format,
                self.shape, self.k_padded_numel, self.group_size,
            )
        if component == "v":
            return _decode_tensor(
                self.v_payload, self.v_scales, self.v_format,
                self.shape, self.v_padded_numel, self.group_size,
            )
        raise ValueError(f"component must be 'k' or 'v', got {component!r}")


@dataclasses.dataclass(frozen=True)
class ConversionResult:
    block_id: int
    layer_id: int
    target_format: str
    components: tuple[str, ...]
    before_bytes: int
    after_bytes: int
    elapsed_ms: float
    temporary_bytes: int


class PagedKVCache:
    """A page-granular KV cache suitable for mechanism experiments.

    The store intentionally uses one tensor record per logical page.  This is
    slower and more allocator-heavy than a production arena, but makes storage
    ownership and conversion accounting observable.  ``metadata_codes`` is
    CPU-resident explicit metadata indexed by ``[block_id, layer_id]``; K and V
    have separate code planes so K-only/V-only policies are representable.
    """

    def __init__(
        self,
        num_blocks: int,
        num_layers: int,
        num_kv_heads: int,
        block_size: int,
        head_dim: int,
        device: torch.device | str,
        default_format: PageFormat = "fp16",
        group_size: int = 128,
    ) -> None:
        if group_size <= 0 or group_size % 2:
            raise ValueError("group_size must be a positive even integer")
        if num_blocks <= 0 or num_layers <= 0 or block_size <= 0 or head_dim <= 0:
            raise ValueError("cache dimensions must be positive")
        self.num_blocks = num_blocks
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.block_size = block_size
        self.head_dim = head_dim
        self.device = torch.device(device)
        self.default_format = _check_format(default_format)
        self.group_size = group_size
        self.page_shape = (block_size, num_kv_heads, head_dim)
        self.metadata_codes = torch.full(
            (num_blocks, num_layers, 2),
            _FORMAT_CODES["fp16"],
            dtype=torch.uint8,
            device="cpu",
        )
        self._pages: dict[tuple[int, int], KVPage] = {}
        self._pending: dict[tuple[int, int], tuple[KVPage, torch.cuda.Event]] = {}

    def _key(self, block_id: int, layer_id: int) -> tuple[int, int]:
        if not (0 <= block_id < self.num_blocks and 0 <= layer_id < self.num_layers):
            raise IndexError(f"page ({block_id}, {layer_id}) outside cache")
        return block_id, layer_id

    def _set_metadata(self, page: KVPage) -> None:
        self.metadata_codes[page.block_id, page.layer_id, 0] = _FORMAT_CODES[page.k_format]
        self.metadata_codes[page.block_id, page.layer_id, 1] = _FORMAT_CODES[page.v_format]

    def _make_page(
        self,
        block_id: int,
        layer_id: int,
        k: torch.Tensor,
        v: torch.Tensor,
        k_format: PageFormat,
        v_format: PageFormat,
    ) -> KVPage:
        if tuple(k.shape) != self.page_shape or tuple(v.shape) != self.page_shape:
            raise ValueError(f"page tensors must have shape {self.page_shape}")
        k_payload, k_scales, k_padded = _encode_tensor(k, k_format, self.group_size)
        v_payload, v_scales, v_padded = _encode_tensor(v, v_format, self.group_size)
        return KVPage(
            block_id, layer_id, k_format, v_format,
            k_payload, v_payload, k_scales, v_scales,
            self.page_shape, k_padded, v_padded, self.group_size,
        )

    def _wait_page(self, key: tuple[int, int]) -> None:
        pending = self._pending.get(key)
        if pending is not None:
            # Keep the source alive until an explicit device synchronization;
            # merely enqueueing a wait is not enough to make host destruction
            # safe.  The consumer stream can still overlap conversion.
            torch.cuda.current_stream(self.device).wait_event(pending[1])

    def write(
        self,
        block_id: int,
        layer_id: int,
        k: torch.Tensor,
        v: torch.Tensor,
        token_offset: int = 0,
    ) -> None:
        """Write a token span, preserving the page's current formats."""
        key = self._key(block_id, layer_id)
        self._wait_page(key)
        if k.ndim != 3 or v.ndim != 3 or tuple(k.shape[1:]) != self.page_shape[1:] or tuple(v.shape[1:]) != self.page_shape[1:]:
            raise ValueError("k and v must be [tokens, num_kv_heads, head_dim]")
        if k.shape[0] != v.shape[0] or token_offset < 0 or token_offset + k.shape[0] > self.block_size:
            raise ValueError("K/V token span does not fit in page")
        old = self._pages.get(key)
        if old is None:
            full_k = torch.zeros(self.page_shape, dtype=torch.float16, device=self.device)
            full_v = torch.zeros_like(full_k)
            k_format = v_format = self.default_format
        else:
            full_k = old.decode("k")
            full_v = old.decode("v")
            k_format, v_format = old.k_format, old.v_format
        full_k[token_offset:token_offset + k.shape[0]].copy_(k)
        full_v[token_offset:token_offset + v.shape[0]].copy_(v)
        page = self._make_page(block_id, layer_id, full_k, full_v, k_format, v_format)
        self._pages[key] = page
        self._set_metadata(page)

    def _convert_page_sync(
        self,
        page: KVPage,
        target_format: PageFormat,
        components: tuple[str, ...],
    ) -> KVPage:
        k_format = target_format if "k" in components else page.k_format
        v_format = target_format if "v" in components else page.v_format
        k = page.decode("k") if "k" in components else page.k_payload
        v = page.decode("v") if "v" in components else page.v_payload
        if "k" not in components:
            # _make_page needs a decoded tensor only for changed components;
            # keep the old payload below by rebuilding explicitly.
            k_payload, k_scales, k_padded = page.k_payload, page.k_scales, page.k_padded_numel
        else:
            k_payload, k_scales, k_padded = _encode_tensor(k, k_format, self.group_size)
        if "v" not in components:
            v_payload, v_scales, v_padded = page.v_payload, page.v_scales, page.v_padded_numel
        else:
            v_payload, v_scales, v_padded = _encode_tensor(v, v_format, self.group_size)
        return KVPage(
            page.block_id, page.layer_id, k_format, v_format,
            k_payload, v_payload, k_scales, v_scales,
            page.shape, k_padded, v_padded, page.group_size,
        )

    def demote_page(
        self,
        block_id: int,
        layer_id: int,
        target_format: PageFormat,
        components: Iterable[str] = ("k", "v"),
        synchronize: bool = True,
    ) -> ConversionResult:
        """Convert live page components to a lower precision synchronously."""
        target_format = _check_format(target_format)
        components = tuple(components)
        if not components or any(component not in ("k", "v") for component in components):
            raise ValueError("components must contain k and/or v")
        key = self._key(block_id, layer_id)
        self._wait_page(key)
        page = self._pages[key]
        selected_formats = [page.k_format if component == "k" else page.v_format for component in components]
        if any(_FORMAT_BITS[target_format] >= _FORMAT_BITS[fmt] for fmt in selected_formats):
            raise ValueError("demotion must strictly reduce every selected component's precision")
        before = page.allocated_bytes
        temporary = before
        if self.device.type == "cuda" and synchronize:
            torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        new_page = self._convert_page_sync(page, target_format, components)
        if self.device.type == "cuda" and synchronize:
            torch.cuda.synchronize(self.device)
        elapsed_ms = (time.perf_counter() - start) * 1000
        self._pages[key] = new_page
        self._set_metadata(new_page)
        return ConversionResult(
            block_id, layer_id, target_format, components,
            before, new_page.allocated_bytes, elapsed_ms, temporary,
        )

    def demote_page_async(
        self,
        block_id: int,
        layer_id: int,
        target_format: PageFormat,
        components: Iterable[str] = ("k", "v"),
        stream: torch.cuda.Stream | None = None,
    ) -> None:
        """Schedule conversion and publish it with an event dependency.

        ``read`` inserts a stream wait before using the published payload.  The
        old page is retained only in ``_pending`` until ``synchronize`` is
        called, making transient overlap memory explicit rather than silently
        corrupting a request.
        """
        target_format = _check_format(target_format)
        components = tuple(components)
        if not components or any(component not in ("k", "v") for component in components):
            raise ValueError("components must contain k and/or v")
        if self.device.type != "cuda":
            raise RuntimeError("asynchronous page conversion requires CUDA")
        key = self._key(block_id, layer_id)
        self._wait_page(key)
        old = self._pages[key]
        selected_formats = [old.k_format if component == "k" else old.v_format for component in components]
        if any(_FORMAT_BITS[target_format] >= _FORMAT_BITS[fmt] for fmt in selected_formats):
            raise ValueError("demotion must strictly reduce every selected component's precision")
        stream = stream or torch.cuda.current_stream(self.device)
        source_ready = torch.cuda.Event()
        source_ready.record(torch.cuda.current_stream(self.device))
        stream.wait_event(source_ready)
        with torch.cuda.stream(stream):
            new_page = self._convert_page_sync(old, target_format, components)
            event = torch.cuda.Event()
            event.record(stream)
        self._pages[key] = new_page
        self._pending[key] = (old, event)
        self._set_metadata(new_page)

    def read(self, block_id: int, layer_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        key = self._key(block_id, layer_id)
        self._wait_page(key)
        page = self._pages[key]
        return page.decode("k"), page.decode("v")

    def page(self, block_id: int, layer_id: int) -> KVPage:
        key = self._key(block_id, layer_id)
        self._wait_page(key)
        return self._pages[key]

    def allocated_bytes(self, include_pending: bool = True) -> int:
        current = sum(page.allocated_bytes for page in self._pages.values())
        if include_pending:
            current += sum(old.allocated_bytes for old, _ in self._pending.values())
        return current

    def logical_bytes(self) -> int:
        return sum(page.allocated_bytes for page in self._pages.values())

    def metadata_counts(self) -> dict[str, dict[str, int]]:
        result = {component: Counter() for component in ("k", "v")}
        for page in self._pages.values():
            result["k"][page.k_format] += 1
            result["v"][page.v_format] += 1
        return {component: dict(counts) for component, counts in result.items()}

    def resident_page_keys(self) -> list[tuple[int, int]]:
        return sorted(self._pages)

    def free_blocks(self, block_ids: Iterable[int]) -> None:
        block_ids = set(block_ids)
        for key in list(self._pages):
            if key[0] in block_ids:
                self._wait_page(key)
                del self._pages[key]
                self.metadata_codes[key[0], key[1], :] = _FORMAT_CODES["fp16"]

    def synchronize(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self._pending.clear()

    def has_unreclaimed_shadow(self) -> bool:
        return bool(self._pending)

    def storage_summary(self) -> dict[str, object]:
        return {
            "resident_pages": len(self._pages),
            "metadata_counts": self.metadata_counts(),
            "logical_payload_bytes": self.logical_bytes(),
            "allocated_payload_bytes_including_pending": self.allocated_bytes(True),
            "pending_old_page_bytes": self.allocated_bytes(True) - self.logical_bytes(),
            "has_unreclaimed_shadow": self.has_unreclaimed_shadow(),
        }


def page_attention_for_layer(
    q: torch.Tensor,
    page_cache: PagedKVCache,
    block_table: torch.Tensor,
    seq_ids: torch.Tensor,
    decoding_seq_lens: torch.Tensor,
    model_config,
    engine_config,
    layer_id: int,
    out: torch.Tensor,
) -> None:
    """Layer-aware form used by SwiftLLM's transformer path."""
    for batch_id in range(q.shape[0]):
        seq_id = int(seq_ids[batch_id].item())
        seq_len = int(decoding_seq_lens[batch_id].item())
        num_blocks = math.ceil(seq_len / engine_config.block_size)
        k_pages, v_pages = [], []
        for block_index in range(num_blocks):
            physical_id = int(block_table[seq_id, block_index].item())
            k_page, v_page = page_cache.read(physical_id, layer_id)
            k_pages.append(k_page)
            v_pages.append(v_page)
        k = torch.cat(k_pages, dim=0)[:seq_len]
        v = torch.cat(v_pages, dim=0)[:seq_len]
        k = k.repeat_interleave(model_config.num_q_heads // model_config.num_kv_heads, dim=1)
        v = v.repeat_interleave(model_config.num_q_heads // model_config.num_kv_heads, dim=1)
        scores = torch.einsum("hd,thd->ht", q[batch_id].float(), k.float())
        weights = torch.softmax(scores * (model_config.head_dim ** -0.5), dim=-1)
        result = torch.einsum("ht,thd->hd", weights, v.float())
        out[batch_id].copy_(result.to(dtype=out.dtype).reshape(-1))
