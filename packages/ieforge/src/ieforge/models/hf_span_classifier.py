from dataclasses import dataclass
from typing import Any, cast

import torch
from torch import nn
from transformers import AutoModel

from ieforge.models.base import IEModel
from ieforge.registry import register


@dataclass
class SpanClassifierOutput:
    """Model forward output: span-type logits, no loss (loss is computed elsewhere).

    The span feature fields are populated only when the caller asks for them via
    `return_span_features=True`, because `span_representations` is large --
    `[B, W, K, D]`, tens of MB for a normal batch -- and standard training never
    needs it. See `HFSpanClassifier.forward` for the layout contract.
    """

    logits: torch.Tensor
    loss: None = None
    span_representations: torch.Tensor | None = None
    span_indices: torch.Tensor | None = None
    span_mask: torch.Tensor | None = None


SPAN_REP_MODES = ("start_end", "start_end_width", "start_end_pool_width")


class SpanRepLayer(nn.Module):
    """Build a span vector from its boundary words, optionally with width and content.

    `mode` selects how much of the span is represented:

    * ``start_end`` -- boundary words only (the original behaviour).
    * ``start_end_width`` -- adds a learned embedding of the span's word count,
      so the scorer can condition on span length directly.
    * ``start_end_pool_width`` -- also adds a mean pool over the words inside
      the span, giving the scorer lexical evidence from the span interior
      rather than only its edges.

    Boundary-only representations cannot distinguish two spans that share a
    start and end word, and carry no signal about what lies between them, which
    matters most on datasets where entities are usually multi-word.
    """

    def __init__(
        self,
        hidden_size: int,
        span_hidden_dim: int,
        dropout: float = 0.1,
        mode: str = "start_end",
        max_span_width: int = 12,
        width_embedding_dim: int = 64,
    ) -> None:
        """Build the boundary projections and any extra feature branches for `mode`."""
        super().__init__()
        if mode not in SPAN_REP_MODES:
            msg = f"Unknown span_rep_mode {mode!r}; expected one of {SPAN_REP_MODES}"
            raise ValueError(msg)
        self.mode = mode
        self.use_width = mode in ("start_end_width", "start_end_pool_width")
        self.use_pool = mode == "start_end_pool_width"

        self.project_start = nn.Sequential(
            nn.Linear(hidden_size, span_hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(span_hidden_dim * 4, span_hidden_dim),
        )
        self.project_end = nn.Sequential(
            nn.Linear(hidden_size, span_hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(span_hidden_dim * 4, span_hidden_dim),
        )

        joint_dim = span_hidden_dim * 2
        if self.use_pool:
            self.project_pool = nn.Sequential(
                nn.Linear(hidden_size, span_hidden_dim * 4),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(span_hidden_dim * 4, span_hidden_dim),
            )
            joint_dim += span_hidden_dim
        if self.use_width:
            self.width_embeddings = nn.Embedding(max_span_width, width_embedding_dim)
            joint_dim += width_embedding_dim

        self.out_project = nn.Sequential(
            nn.Linear(joint_dim, span_hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(span_hidden_dim * 4, span_hidden_dim),
        )

    def _mean_pool(
        self, word_embeddings: torch.Tensor, span_idx: torch.Tensor
    ) -> torch.Tensor:
        """Average the word embeddings inside each span, inclusive of both ends."""
        num_words = word_embeddings.size(1)
        positions = torch.arange(num_words, device=word_embeddings.device)
        starts = span_idx[:, :, 0].unsqueeze(-1)
        ends = span_idx[:, :, 1].unsqueeze(-1)
        inside = (positions >= starts) & (positions <= ends)
        weights = inside.to(word_embeddings.dtype)
        totals = torch.bmm(weights, word_embeddings)
        return totals / weights.sum(-1, keepdim=True).clamp(min=1.0)

    def forward(
        self,
        word_embeddings: torch.Tensor,
        span_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Gather the configured span features and project them to a joint vector."""
        start_rep = self.project_start(word_embeddings)
        end_rep = self.project_end(word_embeddings)

        start_span = torch.gather(
            start_rep,
            1,
            span_idx[:, :, 0].unsqueeze(-1).expand(-1, -1, start_rep.size(-1)),
        )
        end_span = torch.gather(
            end_rep, 1, span_idx[:, :, 1].unsqueeze(-1).expand(-1, -1, end_rep.size(-1))
        )

        parts = [start_span, end_span]
        if self.use_pool:
            parts.append(self.project_pool(self._mean_pool(word_embeddings, span_idx)))

        cat = torch.cat(parts, dim=-1).relu()
        if self.use_width:
            widths = (span_idx[:, :, 1] - span_idx[:, :, 0]).clamp(
                min=0, max=self.width_embeddings.num_embeddings - 1
            )
            cat = torch.cat([cat, self.width_embeddings(widths)], dim=-1)
        return self.out_project(cat)


class EntityTypeProjection(nn.Module):
    """Project entity type embeddings to the span representation space."""

    def __init__(
        self, hidden_size: int, span_hidden_dim: int, dropout: float = 0.1
    ) -> None:
        """Build the entity-type embedding projection MLP."""
        super().__init__()
        self.project = nn.Sequential(
            nn.Linear(hidden_size, span_hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(span_hidden_dim * 4, span_hidden_dim),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Project raw entity-type embeddings into the span representation space."""
        return self.project(embeddings)


@register("model", name="hf_span_classifier")
class HFSpanClassifier(IEModel):
    """Span classifier backed by a Hugging Face pretrained encoder."""

    def __init__(
        self,
        model_name: str,
        label_list: list[str],
        max_span_width: int = 12,
        span_hidden_dim: int | None = None,
        dropout: float = 0.1,
        span_rep_mode: str = "start_end",
        width_embedding_dim: int = 64,
        name: str = "hf_span_classifier",
        extra_token_count: int = 0,
    ) -> None:
        """Load a pretrained encoder and add a span classification head.

        `label_list` is the bare entity type names (e.g. ["Chemical", "Disease"]),
        not BIO labels -- callers (e.g. SpanNERTask) strip B-/I- prefixes first.
        """
        super().__init__()
        self.name = name
        self.label_list = label_list
        self.max_span_width = max_span_width
        self.num_entity_types = len(label_list)

        try:
            self.encoder = AutoModel.from_pretrained(model_name)
        except ValueError:
            # Fallback for checkpoints without model_type (e.g. prajjwal1/bert-tiny)
            from transformers import BertConfig, BertModel

            config = BertConfig.from_pretrained(model_name)
            self.encoder = BertModel.from_pretrained(model_name, config=config)

        hidden_size: int = self.encoder.config.hidden_size
        self.span_hidden_dim = (
            span_hidden_dim if span_hidden_dim is not None else hidden_size
        )

        if extra_token_count < 0:
            msg = f"extra_token_count must be non-negative, got {extra_token_count}"
            raise ValueError(msg)
        if extra_token_count > 0:
            embeddings = self.encoder.get_input_embeddings()
            current_size = cast("int", embeddings.num_embeddings)
            self.encoder.resize_token_embeddings(current_size + extra_token_count)

        self.span_rep_mode = span_rep_mode
        self.span_rep_layer = SpanRepLayer(
            hidden_size,
            self.span_hidden_dim,
            dropout,
            mode=span_rep_mode,
            max_span_width=max_span_width,
            width_embedding_dim=width_embedding_dim,
        )

        self.entity_type_embeddings = nn.Embedding(self.num_entity_types, hidden_size)
        self.entity_type_projection = EntityTypeProjection(
            hidden_size, self.span_hidden_dim, dropout
        )

    def score_span_representations(
        self, span_representations: torch.Tensor
    ) -> torch.Tensor:
        """Score span vectors against the entity-type embeddings.

        This is the same scorer `forward` uses, exposed so a downstream
        component can re-score span representations it has modified -- for
        example after fusing per-candidate evidence -- without depending on the
        entity-type embedding and projection submodules.

        Args:
            span_representations: ``[B, W, K, D]`` where ``D`` is
                ``span_hidden_dim``. Passing the untouched
                ``SpanClassifierOutput.span_representations`` reproduces
                ``SpanClassifierOutput.logits`` exactly.

        Returns:
            ``[B, W, K, C]`` logits, with the type axis in `label_list` order,
            matching `forward`.

        Gradients flow to both the input and the type scorer, so a fusion layer
        built on top is trainable. The encoder is not involved.
        """
        if span_representations.dim() != 4:
            msg = (
                "span_representations must be [B, W, K, D], got rank "
                f"{span_representations.dim()}"
            )
            raise ValueError(msg)
        if span_representations.size(-1) != self.span_hidden_dim:
            msg = (
                f"span_representations last dim must be span_hidden_dim="
                f"{self.span_hidden_dim}, got {span_representations.size(-1)}"
            )
            raise ValueError(msg)

        type_indices = torch.arange(
            self.num_entity_types, device=span_representations.device
        )
        type_emb = self.entity_type_projection(
            self.entity_type_embeddings(type_indices)
        )
        type_emb = type_emb.unsqueeze(0).expand(span_representations.size(0), -1, -1)
        return torch.einsum("bwkd,bcd->bwkc", span_representations, type_emb)

    def forward(
        self, batch: dict[str, Any], *, return_span_features: bool = False
    ) -> SpanClassifierOutput:
        """Encode tokens, pool to word representations, and classify each candidate span.

        With `return_span_features=True` the output also carries the intermediate
        span tensors, so a downstream component can fuse per-candidate evidence
        without reimplementing the encoder or reaching into submodules. This is
        a supported contract:

        * ``span_representations`` -- ``[B, W, K, D]``, the vectors the type
          scorer consumes, where ``D`` is ``span_hidden_dim``.
        * ``span_indices`` -- ``[B, W, K, 2]`` word-index pairs ``(start, end)``,
          both inclusive. Invalid entries are ``(0, 0)``; use ``span_mask``
          rather than testing for that sentinel.
        * ``span_mask`` -- ``[B, W, K]`` bool, true where the span fits inside
          the example's word count.
        * ``logits`` -- ``[B, W, K, C]``, always returned.

        The first three share axes with ``logits``, so ``logits[b, w, k]`` scores
        the span described by ``span_indices[b, w, k]``. The flag changes only
        what is returned: the encoder runs exactly once either way, the same
        dropout draw is used, and the tensors are the ones the forward pass
        actually used, so gradients flow back through them. Values are identical
        to a call with the flag off.

        Ordering follows the collator's flattening, ``flat = w * max_span_width
        + k``, matching ``span_labels`` and ``span_mask`` in the batch.
        """
        encoder_output = self.encoder(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        token_embeddings = encoder_output.last_hidden_state

        word_starts_list: list[list[int]] = batch["word_starts"]
        num_words_list: list[int] = batch["num_words"]
        batch_size = token_embeddings.shape[0]
        hidden_size = token_embeddings.shape[2]
        max_words = max(num_words_list)
        device = token_embeddings.device

        # Gather each word's first-subword embedding in one batched op. Padded
        # word slots point at token 0 and are zeroed afterwards, matching the
        # zeros-initialised tensor this replaces.
        num_words = torch.as_tensor(num_words_list, device=device)
        word_start_idx = torch.zeros(
            batch_size, max_words, device=device, dtype=torch.long
        )
        for b, starts in enumerate(word_starts_list):
            if starts:
                word_start_idx[b, : len(starts)] = torch.as_tensor(
                    starts, device=device, dtype=torch.long
                )
        word_valid = torch.arange(max_words, device=device) < num_words.unsqueeze(1)
        word_embeddings = token_embeddings.gather(
            1, word_start_idx.unsqueeze(-1).expand(-1, -1, hidden_size)
        )
        word_embeddings = word_embeddings * word_valid.unsqueeze(-1)

        # span_idx depends only on the geometry (max_words, width, num_words),
        # so it broadcasts instead of looping. Out-of-range spans collapse to
        # (0, 0); the loss masks them via span_mask.
        starts_grid = torch.arange(max_words, device=device).unsqueeze(1)
        widths = torch.arange(self.max_span_width, device=device).unsqueeze(0)
        ends_grid = starts_grid + widths
        span_valid = (starts_grid < num_words.view(-1, 1, 1)) & (
            ends_grid < num_words.view(-1, 1, 1)
        )
        span_idx = torch.stack(
            (
                starts_grid.expand(max_words, self.max_span_width) * span_valid.long(),
                ends_grid.expand(max_words, self.max_span_width) * span_valid.long(),
            ),
            dim=-1,
        ).reshape(batch_size, max_words * self.max_span_width, 2)

        span_rep = self.span_rep_layer(word_embeddings, span_idx)
        span_rep = span_rep.view(
            batch_size, max_words, self.max_span_width, self.span_hidden_dim
        )

        logits = self.score_span_representations(span_rep)

        if not return_span_features:
            return SpanClassifierOutput(logits=logits)

        # Reshaped views of the tensors this pass already used, so the caller
        # sees exactly what scored the logits and gradients stay connected.
        return SpanClassifierOutput(
            logits=logits,
            span_representations=span_rep,
            span_indices=span_idx.view(batch_size, max_words, self.max_span_width, 2),
            span_mask=span_valid.view(batch_size, max_words, self.max_span_width),
        )
