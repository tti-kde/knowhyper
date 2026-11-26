"""Structural feature-channel batching for knowledge contexts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

import torch

from ieforge.knowledge.core.base import IEKnowledgeFeatureChannel
from ieforge.registry import register

if TYPE_CHECKING:
    from ieforge.knowledge.core.schema import KnowledgeContext


class FeatureChannel(Protocol):
    """Converts knowledge contexts into batch-level feature tensors."""

    def collate(self, contexts: Sequence[KnowledgeContext]) -> Mapping[str, Any]:
        """Return batch keys derived from per-example knowledge contexts."""
        ...


@dataclass(frozen=True)
class TensorFeatureSpec:
    """Configuration for batching one tensor from ``graph_features``."""

    source_key: str
    batch_key: str
    pad_dim: int | None = 0
    pad_value: float | int = 0
    required: bool = True

    def __post_init__(self) -> None:
        """Validate pad configuration."""
        if self.pad_dim is not None and self.pad_dim < 0:
            msg = f"pad_dim must be non-negative or None, got {self.pad_dim}"
            raise ValueError(msg)


@register("knowledge_channel", name="tensor_feature")
class TensorFeatureChannel(IEKnowledgeFeatureChannel):
    """Batch configured tensor values from ``KnowledgeContext.graph_features``."""

    def __init__(self, specs: Sequence[TensorFeatureSpec | Mapping[str, Any]]) -> None:
        """Store tensor feature specs."""
        self.specs = tuple(_coerce_spec(spec) for spec in specs)
        if not self.specs:
            msg = "TensorFeatureChannel requires at least one TensorFeatureSpec"
            raise ValueError(msg)

    def collate(self, contexts: Sequence[KnowledgeContext]) -> Mapping[str, Any]:
        """Batch configured feature tensors from contexts."""
        return {
            spec.batch_key: batched
            for spec in self.specs
            if (batched := _batch_feature(contexts, spec)) is not None
        }


def _coerce_spec(spec: TensorFeatureSpec | Mapping[str, Any]) -> TensorFeatureSpec:
    if isinstance(spec, TensorFeatureSpec):
        return spec
    if isinstance(spec, Mapping):
        return TensorFeatureSpec(**dict(spec))
    msg = (
        "TensorFeatureChannel specs must be TensorFeatureSpec or mapping values, "
        f"got {type(spec).__name__}"
    )
    raise TypeError(msg)


def _batch_feature(
    contexts: Sequence[KnowledgeContext],
    spec: TensorFeatureSpec,
) -> torch.Tensor | None:
    values = [
        _feature_value(context, spec, index) for index, context in enumerate(contexts)
    ]
    present = [value for value in values if value is not None]
    if not present:
        if spec.required and contexts:
            msg = f"Required graph feature {spec.source_key!r} missing from example 0."
            raise ValueError(msg)
        return None

    reference = present[0]
    tensors = [
        _missing_tensor(reference, spec) if value is None else value for value in values
    ]
    if spec.pad_dim is None:
        _validate_stack_shapes(tensors, spec)
        return torch.stack(tensors)

    _validate_pad_dim(reference, spec)
    _validate_padded_shapes(tensors, spec)
    max_size = max(tensor.shape[spec.pad_dim] for tensor in tensors)
    return torch.stack([_pad_tensor(tensor, spec, max_size) for tensor in tensors])


def _feature_value(
    context: KnowledgeContext,
    spec: TensorFeatureSpec,
    example_index: int,
) -> torch.Tensor | None:
    if spec.source_key not in context.graph_features:
        if spec.required:
            msg = (
                f"Required graph feature {spec.source_key!r} missing from "
                f"example {example_index}."
            )
            raise ValueError(msg)
        return None

    value = context.graph_features[spec.source_key]
    if not isinstance(value, torch.Tensor):
        msg = (
            f"Graph feature {spec.source_key!r} must be a torch.Tensor for "
            f"example {example_index}, got {type(value).__name__}."
        )
        raise TypeError(msg)
    return value


def _missing_tensor(reference: torch.Tensor, spec: TensorFeatureSpec) -> torch.Tensor:
    shape = list(reference.shape)
    if spec.pad_dim is not None:
        shape[spec.pad_dim] = 0
    return reference.new_full(tuple(shape), spec.pad_value)


def _validate_stack_shapes(
    tensors: Sequence[torch.Tensor],
    spec: TensorFeatureSpec,
) -> None:
    expected = tuple(tensors[0].shape)
    for index, tensor in enumerate(tensors):
        if tuple(tensor.shape) != expected:
            msg = (
                f"Graph feature {spec.source_key!r} shape mismatch for example "
                f"{index}: got {tuple(tensor.shape)}, expected {expected}."
            )
            raise ValueError(msg)


def _validate_pad_dim(reference: torch.Tensor, spec: TensorFeatureSpec) -> None:
    assert spec.pad_dim is not None
    if spec.pad_dim >= reference.dim():
        msg = (
            f"Graph feature {spec.source_key!r} pad_dim {spec.pad_dim} is out "
            f"of range for shape {tuple(reference.shape)}."
        )
        raise ValueError(msg)


def _validate_padded_shapes(
    tensors: Sequence[torch.Tensor],
    spec: TensorFeatureSpec,
) -> None:
    assert spec.pad_dim is not None
    expected_rank = tensors[0].dim()
    expected_non_padded = _non_padded_shape(tensors[0], spec.pad_dim)
    for index, tensor in enumerate(tensors):
        if tensor.dim() != expected_rank:
            msg = (
                f"Graph feature {spec.source_key!r} rank mismatch for example "
                f"{index}: got {tensor.dim()}, expected {expected_rank}."
            )
            raise ValueError(msg)
        if _non_padded_shape(tensor, spec.pad_dim) != expected_non_padded:
            msg = (
                f"Graph feature {spec.source_key!r} non-padded dimensions mismatch "
                f"for example {index}: got {tuple(tensor.shape)}, expected "
                f"dimensions {expected_non_padded} outside pad_dim {spec.pad_dim}."
            )
            raise ValueError(msg)


def _non_padded_shape(tensor: torch.Tensor, pad_dim: int) -> tuple[int, ...]:
    return tuple(size for dim, size in enumerate(tensor.shape) if dim != pad_dim)


def _pad_tensor(
    tensor: torch.Tensor,
    spec: TensorFeatureSpec,
    max_size: int,
) -> torch.Tensor:
    assert spec.pad_dim is not None
    pad_size = max_size - tensor.shape[spec.pad_dim]
    if pad_size == 0:
        return tensor
    shape = list(tensor.shape)
    shape[spec.pad_dim] = pad_size
    padding = tensor.new_full(tuple(shape), spec.pad_value)
    return torch.cat([tensor, padding], dim=spec.pad_dim)


__all__ = ["FeatureChannel", "TensorFeatureChannel", "TensorFeatureSpec"]
