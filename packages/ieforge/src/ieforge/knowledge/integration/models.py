from __future__ import annotations

from typing import Any

import torch
from hydra.errors import InstantiationException
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from ieforge.models.base import IEModel
from ieforge.registry import register


@register("model", name="knowledge_aware")
class KnowledgeAwareModel(IEModel):
    """IE model wrapper that prepends collated knowledge soft prompts."""

    def __init__(
        self,
        inner: IEModel | dict[str, Any] | DictConfig,
        label_list: list[str],
        extra_token_count: int = 0,
        name: str = "knowledge_aware",
        prompt_key: str = "knowledge_prompt_embeds",
        prompt_attention_mask_key: str = "knowledge_prompt_attention_mask",
        ignore_index: int = -100,
    ) -> None:
        """Wrap an IEModel or Hydra model config with soft-prompt conditioning."""
        super().__init__()
        if extra_token_count < 0:
            msg = f"extra_token_count must be non-negative, got {extra_token_count}"
            raise ValueError(msg)
        self.inner = self._build_inner(inner, label_list, extra_token_count)
        self.name = name
        self.label_list = label_list
        self.extra_token_count = extra_token_count
        self.prompt_key = prompt_key
        self.prompt_attention_mask_key = prompt_attention_mask_key
        self.ignore_index = ignore_index

    def forward(self, batch: dict[str, Any]) -> Any:
        """Run the wrapped model, prepending prompt embeddings when present."""
        prompt_embeds = batch.get(self.prompt_key)
        if prompt_embeds is None:
            return self.inner.forward(batch)
        if not isinstance(prompt_embeds, torch.Tensor):
            msg = f"{self.prompt_key!r} must be a torch.Tensor"
            raise TypeError(msg)
        if prompt_embeds.ndim != 3:
            msg = (
                f"{self.prompt_key!r} must have shape "
                "[batch, prompt_length, hidden_size]"
            )
            raise ValueError(msg)
        if prompt_embeds.shape[1] == 0:
            return self.inner.forward(batch)

        hf_model = self._hf_model()
        embeddings = hf_model.get_input_embeddings()
        input_ids = self._input_ids(batch)
        token_embeds = embeddings(input_ids)
        if token_embeds.ndim != 3:
            msg = "inner model input embeddings must return a 3-D tensor"
            raise ValueError(msg)
        if prompt_embeds.shape[0] != token_embeds.shape[0]:
            msg = (
                f"{self.prompt_key!r} batch size {prompt_embeds.shape[0]} does not "
                f"match input_ids batch size {token_embeds.shape[0]}"
            )
            raise ValueError(msg)
        if prompt_embeds.shape[2] != token_embeds.shape[2]:
            msg = (
                f"{self.prompt_key!r} hidden size {prompt_embeds.shape[2]} does not "
                f"match inner model hidden size {token_embeds.shape[2]}"
            )
            raise ValueError(msg)

        prompt_embeds = prompt_embeds.to(
            device=token_embeds.device,
            dtype=token_embeds.dtype,
        )
        inputs_embeds = torch.cat([prompt_embeds, token_embeds], dim=1)

        attention_mask = self._attention_mask(batch, token_embeds, prompt_embeds)
        kwargs: dict[str, Any] = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
        }
        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is not None:
            kwargs["token_type_ids"] = self._prepend_token_type_ids(
                token_type_ids,
                prompt_embeds.shape[1],
            )
        if "labels" in batch:
            kwargs["labels"] = self._labels(
                batch["labels"],
                input_ids.shape,
                prompt_embeds.shape[1],
            )
        return hf_model(**kwargs)

    @staticmethod
    def _build_inner(
        inner: IEModel | dict[str, Any] | DictConfig,
        label_list: list[str],
        extra_token_count: int,
    ) -> IEModel:
        if isinstance(inner, IEModel):
            return inner
        if not isinstance(inner, (dict, DictConfig)):
            msg = (
                "inner must be an IEModel instance or a Hydra config/dict with "
                "a _target_"
            )
            raise TypeError(msg)

        inner_cfg = OmegaConf.create(inner)
        if OmegaConf.select(inner_cfg, "_target_", default=None) is None:
            msg = "inner config must include a _target_"
            raise ValueError(msg)

        try:
            built = instantiate(
                inner_cfg,
                label_list=label_list,
                extra_token_count=extra_token_count,
            )
        except (TypeError, InstantiationException) as exc:
            if extra_token_count != 0 or "extra_token_count" not in str(exc):
                raise
            built = instantiate(inner_cfg, label_list=label_list)

        if not isinstance(built, IEModel):
            msg = f"inner config instantiated {type(built).__name__}, expected IEModel"
            raise TypeError(msg)
        return built

    def _hf_model(self) -> Any:
        hf_model = getattr(self.inner, "model", None)
        if hf_model is None:
            msg = (
                "KnowledgeAwareModel requires inner.model to reference a "
                "Hugging Face-style module"
            )
            raise TypeError(msg)
        if not hasattr(hf_model, "get_input_embeddings"):
            msg = "inner.model must provide get_input_embeddings()"
            raise TypeError(msg)
        return hf_model

    @staticmethod
    def _input_ids(batch: dict[str, Any]) -> torch.Tensor:
        input_ids = batch.get("input_ids")
        if not isinstance(input_ids, torch.Tensor):
            msg = "batch['input_ids'] must be a torch.Tensor when prompts are present"
            raise TypeError(msg)
        if input_ids.ndim != 2:
            msg = "batch['input_ids'] must have shape [batch, sequence_length]"
            raise ValueError(msg)
        return input_ids

    def _attention_mask(
        self,
        batch: dict[str, Any],
        token_embeds: torch.Tensor,
        prompt_embeds: torch.Tensor,
    ) -> torch.Tensor:
        attention_mask = batch.get("attention_mask")
        if not isinstance(attention_mask, torch.Tensor):
            msg = (
                "batch['attention_mask'] must be a torch.Tensor when prompts "
                "are present"
            )
            raise TypeError(msg)
        if attention_mask.ndim != 2:
            msg = "batch['attention_mask'] must have shape [batch, sequence_length]"
            raise ValueError(msg)
        if attention_mask.shape != token_embeds.shape[:2]:
            msg = (
                "batch['attention_mask'] shape must match input_ids shape before "
                "prompt prepending"
            )
            raise ValueError(msg)

        prompt_mask = batch.get(self.prompt_attention_mask_key)
        if prompt_mask is None:
            prompt_mask = torch.ones(
                prompt_embeds.shape[:2],
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
        else:
            if not isinstance(prompt_mask, torch.Tensor):
                msg = f"{self.prompt_attention_mask_key!r} must be a torch.Tensor"
                raise TypeError(msg)
            if prompt_mask.shape != prompt_embeds.shape[:2]:
                msg = (
                    f"{self.prompt_attention_mask_key!r} shape must match "
                    f"{self.prompt_key!r} first two dimensions"
                )
                raise ValueError(msg)
            prompt_mask = prompt_mask.to(
                device=attention_mask.device,
                dtype=attention_mask.dtype,
            )
        return torch.cat([prompt_mask, attention_mask], dim=1)

    @staticmethod
    def _prepend_token_type_ids(
        token_type_ids: Any,
        prompt_length: int,
    ) -> torch.Tensor:
        if not isinstance(token_type_ids, torch.Tensor):
            msg = "batch['token_type_ids'] must be a torch.Tensor when present"
            raise TypeError(msg)
        if token_type_ids.ndim != 2:
            msg = "batch['token_type_ids'] must have shape [batch, sequence_length]"
            raise ValueError(msg)
        prompt_token_type_ids = torch.zeros(
            (token_type_ids.shape[0], prompt_length),
            dtype=token_type_ids.dtype,
            device=token_type_ids.device,
        )
        return torch.cat([prompt_token_type_ids, token_type_ids], dim=1)

    def _labels(
        self,
        labels: Any,
        input_shape: torch.Size,
        prompt_length: int,
    ) -> Any:
        if not isinstance(labels, torch.Tensor):
            return labels
        if labels.ndim != 2 or tuple(labels.shape) != tuple(input_shape):
            return labels
        prompt_labels = torch.full(
            (labels.shape[0], prompt_length),
            self.ignore_index,
            dtype=labels.dtype,
            device=labels.device,
        )
        return torch.cat([prompt_labels, labels], dim=1)
