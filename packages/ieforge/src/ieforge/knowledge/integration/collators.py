"""Collator adapters for knowledge-conditioned batches."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, MutableMapping
from dataclasses import dataclass
from typing import Any

import torch

from ieforge.knowledge.core.base import IEKnowledgeCollator
from ieforge.registry import register


@dataclass(frozen=True)
class _PromptPayload:
    embeds: torch.Tensor
    attention_mask: torch.Tensor


@dataclass(frozen=True)
class _InstanceMetadataView:
    """Wraps a retrieval example, overlaying constant instance metadata.

    Reads of stamped keys (e.g. ``dataset``, ``split``) resolve from
    ``_metadata``; all other attribute reads delegate to the wrapped example.
    Lets a query builder read both per-pair fields (``id``, ``text``,
    ``head_mention``) and collator-constant instance metadata uniformly via
    ``getattr``.
    """

    _example: Any
    _metadata: Mapping[str, Any]

    def __getattr__(self, name: str) -> Any:
        meta = object.__getattribute__(self, "_metadata")
        if name in meta:
            return meta[name]
        return getattr(object.__getattribute__(self, "_example"), name)


@register("knowledge_collator", name="knowledge")
class KnowledgeCollator(IEKnowledgeCollator):
    """Attach padded soft-prompt tensors produced by a knowledge pipeline."""

    def __init__(
        self,
        base_collator: Any,
        pipeline: Any,
        prompt_key: str = "knowledge_prompt_embeds",
        prompt_attention_mask_key: str = "knowledge_prompt_attention_mask",
        require_prompts: bool = True,
        instance_metadata: Mapping[str, Any] | None = None,
        feature_channels: Iterable[Any] | None = None,
    ) -> None:
        """Store the wrapped collator and knowledge batch adapters."""
        self.base_collator = base_collator
        self.pipeline = pipeline
        self.prompt_key = prompt_key
        self.prompt_attention_mask_key = prompt_attention_mask_key
        self.require_prompts = require_prompts
        self.instance_metadata = instance_metadata
        self.feature_channels = tuple(feature_channels or ())

    def __call__(self, examples: Iterable[Any]) -> MutableMapping[str, Any]:
        """Collate examples, then append padded knowledge prompt tensors."""
        example_list = list(examples)
        batch = self.base_collator(example_list)
        if not isinstance(batch, MutableMapping):
            msg = (
                "KnowledgeCollator expected base_collator to return a mutable "
                f"mapping, got {type(batch).__name__}."
            )
            raise TypeError(msg)

        expanded = batch.pop("_knowledge_examples", None)
        retrieval_examples = expanded if expanded is not None else example_list
        if self.instance_metadata is not None:
            retrieval_examples = [
                _InstanceMetadataView(example, self.instance_metadata)
                for example in retrieval_examples
            ]
        contexts = [self.pipeline.run(example) for example in retrieval_examples]
        payloads = [
            self._payload_from_context(context, index)
            for index, context in enumerate(contexts)
        ]
        present_payloads = [payload for payload in payloads if payload is not None]
        if present_payloads:
            prompt_embeds, prompt_attention_mask = _pad_prompt_payloads(payloads)
            batch[self.prompt_key] = prompt_embeds
            batch[self.prompt_attention_mask_key] = prompt_attention_mask

        for channel in self.feature_channels:
            feature_batch = channel.collate(contexts)
            if not isinstance(feature_batch, Mapping):
                msg = (
                    "KnowledgeCollator expected feature channel to return a "
                    f"mapping, got {type(feature_batch).__name__}."
                )
                raise TypeError(msg)
            _merge_feature_batch(
                batch,
                feature_batch,
                reserved_keys={self.prompt_key, self.prompt_attention_mask_key},
            )
        return batch

    def _payload_from_context(
        self, context: Any, example_index: int
    ) -> _PromptPayload | None:
        prompt = getattr(context, "prompt_embeds", None)
        if prompt is None:
            if self.require_prompts:
                msg = (
                    "KnowledgeCollator requires prompt_embeds for example "
                    f"{example_index}, but the knowledge pipeline returned none."
                )
                raise ValueError(msg)
            return None

        prompt = _normalize_prompt_embeds(prompt, example_index)
        mask = getattr(context, "prompt_attention_mask", None)
        mask = _normalize_prompt_attention_mask(mask, prompt.size(0), example_index)
        return _PromptPayload(embeds=prompt, attention_mask=mask)


def _merge_feature_batch(
    batch: MutableMapping[str, Any],
    feature_batch: Mapping[str, Any],
    *,
    reserved_keys: set[str],
) -> None:
    for key in feature_batch:
        if key in reserved_keys:
            msg = (
                "KnowledgeCollator feature channel key collision: "
                f"{key!r} is reserved for prompt tensors."
            )
            raise ValueError(msg)
        if key in batch:
            msg = (
                "KnowledgeCollator feature channel key collision: "
                f"{key!r} already exists in the base batch."
            )
            raise ValueError(msg)
    batch.update(feature_batch)


def _normalize_prompt_embeds(value: Any, example_index: int) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        msg = (
            "KnowledgeCollator expected prompt_embeds for example "
            f"{example_index} to be a torch.Tensor, got {type(value).__name__}."
        )
        raise ValueError(msg)  # noqa: TRY004 - Task 3 requires ValueError here.

    if value.dim() == 3:
        if value.size(0) != 1:
            msg = (
                "KnowledgeCollator expected prompt_embeds for example "
                f"{example_index} to have shape [prompt_len, hidden] or "
                f"[1, prompt_len, hidden], got {tuple(value.shape)}."
            )
            raise ValueError(msg)
        value = value.squeeze(0)
    elif value.dim() != 2:
        msg = (
            "KnowledgeCollator expected prompt_embeds for example "
            f"{example_index} to have shape [prompt_len, hidden] or "
            f"[1, prompt_len, hidden], got {tuple(value.shape)}."
        )
        raise ValueError(msg)

    return value


def _normalize_prompt_attention_mask(
    value: Any, prompt_len: int, example_index: int
) -> torch.Tensor:
    if value is None:
        return torch.ones(prompt_len, dtype=torch.long)

    if not isinstance(value, torch.Tensor):
        msg = (
            "KnowledgeCollator expected prompt_attention_mask for example "
            f"{example_index} to be a torch.Tensor, got {type(value).__name__}."
        )
        raise ValueError(msg)  # noqa: TRY004 - Task 3 requires ValueError here.

    if value.dim() == 2:
        if value.size(0) != 1:
            msg = (
                "KnowledgeCollator expected prompt_attention_mask for example "
                f"{example_index} to have shape [prompt_len] or [1, prompt_len], "
                f"got {tuple(value.shape)}."
            )
            raise ValueError(msg)
        value = value.squeeze(0)
    elif value.dim() != 1:
        msg = (
            "KnowledgeCollator expected prompt_attention_mask for example "
            f"{example_index} to have shape [prompt_len] or [1, prompt_len], "
            f"got {tuple(value.shape)}."
        )
        raise ValueError(msg)

    if value.size(0) != prompt_len:
        msg = (
            "KnowledgeCollator prompt_attention_mask length mismatch for example "
            f"{example_index}: got {value.size(0)}, expected {prompt_len}."
        )
        raise ValueError(msg)

    return value


def _pad_prompt_payloads(
    payloads: list[_PromptPayload | None],
) -> tuple[torch.Tensor, torch.Tensor]:
    present_payloads = [payload for payload in payloads if payload is not None]
    if not present_payloads:
        msg = (
            "KnowledgeCollator cannot pad prompts when no prompt payloads are present."
        )
        raise ValueError(msg)

    first = present_payloads[0]
    hidden_size = first.embeds.size(1)
    prompt_device = first.embeds.device
    prompt_dtype = first.embeds.dtype
    mask_device = first.attention_mask.device
    mask_dtype = first.attention_mask.dtype

    for index, payload in enumerate(payloads):
        if payload is None:
            continue
        if payload.embeds.size(1) != hidden_size:
            msg = (
                "KnowledgeCollator prompt hidden-size mismatch: example "
                f"{index} has hidden size {payload.embeds.size(1)}, "
                f"expected {hidden_size}."
            )
            raise ValueError(msg)

    max_prompt_len = max(payload.embeds.size(0) for payload in present_payloads)
    padded_prompts = []
    padded_masks = []
    for payload in payloads:
        if payload is None:
            prompt = torch.empty(
                (0, hidden_size), dtype=prompt_dtype, device=prompt_device
            )
            mask = torch.empty((0,), dtype=mask_dtype, device=mask_device)
        else:
            prompt = payload.embeds.to(device=prompt_device)
            mask = payload.attention_mask.to(device=mask_device)

        padded_prompts.append(_pad_prompt(prompt, max_prompt_len))
        padded_masks.append(_pad_mask(mask, max_prompt_len))

    return torch.stack(padded_prompts), torch.stack(padded_masks)


def _pad_prompt(prompt: torch.Tensor, max_prompt_len: int) -> torch.Tensor:
    pad_len = max_prompt_len - prompt.size(0)
    if pad_len == 0:
        return prompt
    padding = prompt.new_zeros((pad_len, prompt.size(1)))
    return torch.cat([prompt, padding], dim=0)


def _pad_mask(mask: torch.Tensor, max_prompt_len: int) -> torch.Tensor:
    pad_len = max_prompt_len - mask.size(0)
    if pad_len == 0:
        return mask
    padding = mask.new_zeros((pad_len,))
    return torch.cat([mask, padding], dim=0)


__all__ = ["KnowledgeCollator"]
