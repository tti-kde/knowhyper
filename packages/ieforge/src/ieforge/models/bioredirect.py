"""BioREDirect multi-task model: soft prompts, multi-chunk, token-index pooling."""

from typing import Any

import torch
from torch import nn
from transformers import AutoModel, BertConfig, BertModel

from ieforge.data.schema import BioREDirectOutput
from ieforge.models.base import IEModel
from ieforge.registry import register


class _HiddenTokenPooler(nn.Module):
    """Dense + tanh pooler applied at selected token positions."""

    def __init__(self, hidden_size: int, source_dense: nn.Linear | None = None) -> None:
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        if (
            source_dense is not None
            and source_dense.weight.shape == self.dense.weight.shape
            and source_dense.bias is not None
            and source_dense.bias.shape == self.dense.bias.shape
        ):
            # Mirror official HiddenTokenPooler: init from pretrained BERT pooler.
            self.dense.weight.data.copy_(source_dense.weight.data)
            self.dense.bias.data.copy_(source_dense.bias.data)
        self.activation = nn.Tanh()

    def forward(
        self, hidden_states: torch.Tensor, token_index: torch.Tensor
    ) -> torch.Tensor:
        # hidden_states: (N, T, H); token_index: (N,) -> gather one token per row
        idx = token_index.view(-1, 1, 1).expand(-1, 1, hidden_states.size(-1))
        token_hidden = torch.gather(hidden_states, 1, idx).squeeze(1)  # (N, H)
        return self.activation(self.dense(token_hidden))


@register("model", name="bioredirect")
class BioREDirectModel(IEModel):
    """Faithful port of the official BioREDirect architecture."""

    def __init__(
        self,
        model_name: str,
        novelty_label_count: int,
        direction_label_count: int,
        relation_label_count: int | None = None,
        label_list: list[str] | None = None,
        soft_prompt_len: int = 0,
        num_soft_prompt: int = 1,
        name: str = "bioredirect",
        extra_token_count: int = 0,
    ) -> None:
        """Build the encoder, soft prompts, three poolers, heads, and thresholds.

        The relation head size is taken from ``label_list`` when the runtime
        injects it (``label_list=task.label_list``), falling back to the explicit
        ``relation_label_count``. ``label_list`` must therefore agree with the
        task's relation vocabulary; this keeps the head sized to the data.
        """
        super().__init__()
        self.name = name
        if label_list is not None:
            relation_label_count = len(label_list)
        if relation_label_count is None:
            msg = "BioREDirectModel requires relation_label_count or label_list"
            raise ValueError(msg)
        try:
            self.bert = AutoModel.from_pretrained(model_name)
        except ValueError:
            # Fallback for checkpoints without model_type (e.g. prajjwal1/bert-tiny)
            config = BertConfig.from_pretrained(model_name)
            self.bert = BertModel.from_pretrained(model_name, config=config)
        hidden = self.bert.config.hidden_size
        self.hidden_size = hidden
        self.soft_prompt_len = soft_prompt_len
        self.soft_prompt = nn.Embedding(
            num_soft_prompt, max(soft_prompt_len, 1) * hidden
        )
        self.soft_prompt.weight.data.normal_(
            mean=0.0, std=self.bert.config.initializer_range
        )
        pooler = getattr(self.bert, "pooler", None)
        source = (
            pooler.dense if pooler is not None and hasattr(pooler, "dense") else None
        )
        self.rel_pooler = _HiddenTokenPooler(hidden, source_dense=source)
        self.nov_pooler = _HiddenTokenPooler(hidden, source_dense=source)
        self.dir_pooler = _HiddenTokenPooler(hidden, source_dense=source)
        self.rel_classifier = nn.Linear(hidden, relation_label_count)
        self.nov_classifier = nn.Linear(hidden, novelty_label_count)
        self.dir_classifier = nn.Linear(hidden, direction_label_count)
        self.rel_thresholds = nn.Parameter(torch.full((relation_label_count,), 0.5))
        self.nov_thresholds = nn.Parameter(torch.full((novelty_label_count,), 0.5))
        self.dir_thresholds = nn.Parameter(torch.full((direction_label_count,), 0.5))
        if extra_token_count > 0:
            self.bert.resize_token_embeddings(
                self.bert.get_input_embeddings().num_embeddings + extra_token_count
            )

    def resize_token_embeddings(self, new_num_tokens: int) -> None:
        """Resize the underlying encoder's input embedding table."""
        self.bert.resize_token_embeddings(new_num_tokens)

    def _encode_chunks(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pair_prompt_ids: torch.Tensor,
    ) -> torch.Tensor:
        _b, _num_chunks, t = input_ids.size()
        flat_ids = input_ids.view(-1, t)
        word_embed = self.bert.get_input_embeddings()(flat_ids)  # (B*C, T, H)
        if self.soft_prompt_len > 0:
            prompt = self.soft_prompt(pair_prompt_ids.view(-1))
            prompt = prompt.view(-1, self.soft_prompt_len, self.hidden_size)
            embeds = torch.cat([word_embed, prompt], dim=1)
            extra = torch.ones(
                embeds.size(0),
                self.soft_prompt_len,
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            mask = torch.cat([attention_mask.view(-1, t), extra], dim=1)
        else:
            embeds = word_embed
            mask = attention_mask.view(-1, t)
        return self.bert(inputs_embeds=embeds, attention_mask=mask)[0]  # (B*C, T', H)

    @staticmethod
    def _combine_chunks(pooled: torch.Tensor, b: int, num_chunks: int) -> torch.Tensor:
        pooled = pooled.view(b, num_chunks, -1)  # (B, C, H)
        attn = torch.softmax(torch.matmul(pooled, pooled.transpose(-1, -2)), dim=-1)
        return torch.matmul(attn, pooled)[:, 0, :]  # (B, H)

    def forward(self, batch: dict[str, Any]) -> BioREDirectOutput:
        """Encode chunks, pool the three task tokens, and return head logits."""
        input_ids = batch["input_ids"]
        b, num_chunks, _ = input_ids.size()
        seq = self._encode_chunks(
            input_ids, batch["attention_mask"], batch["pair_prompt_ids"]
        )
        rel_idx = batch["relation_token_index"].view(-1)
        nov_idx = batch["novelty_token_index"].view(-1)
        dir_idx = batch["direction_token_index"].view(-1)
        rel = self._combine_chunks(self.rel_pooler(seq, rel_idx), b, num_chunks)
        nov = self._combine_chunks(self.nov_pooler(seq, nov_idx), b, num_chunks)
        direction = self._combine_chunks(self.dir_pooler(seq, dir_idx), b, num_chunks)
        return BioREDirectOutput(
            relation_logits=self.rel_classifier(rel),
            novelty_logits=self.nov_classifier(nov),
            direction_logits=self.dir_classifier(direction),
        )
