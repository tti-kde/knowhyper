"""Backbone provenance for span-NER proposal export (#218 §0.1).

Candidate JSONL is backbone-specific: the proposal distribution is a property of
the first-pass encoder that produced it. Two failure modes are silent at runtime
and produce a complete, plausible artifact:

1. Exporting a checkpoint trained on backbone A through a model config naming
   backbone B. ``load_state_dict`` only rejects this when a shape differs, and the
   span head shapes match across BERT-base backbones.
2. Pairing one backbone's tokenizer with another's encoder inside one cell. The
   collator tokenizes with one vocabulary while the encoder embeds with another.

The word-embedding row count is the cheapest reliable discriminator: PubMedBERT is
uncased with 30,522 rows, BioLinkBERT has 28,895. Both gates raise rather than warn.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

_WORD_EMBEDDING_SUFFIX = "embeddings.word_embeddings.weight"


def backbone_from_state_dict(state_dict: dict[str, Any]) -> int:
    """Return the encoder vocabulary size recorded in a checkpoint.

    Raises if no word-embedding tensor is present: a renamed key must fail the
    provenance gate loudly rather than silently disable it.
    """
    for key, value in state_dict.items():
        if key.endswith(_WORD_EMBEDDING_SUFFIX):
            shape = getattr(value, "shape", None)
            if shape is None or len(shape) != 2:
                msg = (
                    f"word embedding tensor {key!r} must be 2-D to identify the "
                    f"backbone, got shape {shape!r}"
                )
                raise ValueError(msg)
            return int(shape[0])
    msg = (
        "checkpoint has no word embedding tensor "
        f"(*.{_WORD_EMBEDDING_SUFFIX}); cannot verify the backbone"
    )
    raise ValueError(msg)


@dataclass(frozen=True)
class ProposalProvenance:
    """The backbone identity a proposal export is committed to."""

    model_config: str
    tokenizer_name: str
    encoder_name: str
    vocab_size: int

    def __post_init__(self) -> None:
        """Reject a tokenizer/encoder mismatch at construction time."""
        self.validate()

    def validate(self) -> None:
        if self.tokenizer_name != self.encoder_name:
            msg = (
                f"tokenizer {self.tokenizer_name!r} does not match encoder "
                f"{self.encoder_name!r}; one cell must use a single backbone"
            )
            raise ValueError(msg)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_config": self.model_config,
            "tokenizer_name": self.tokenizer_name,
            "encoder_name": self.encoder_name,
            "vocab_size": self.vocab_size,
        }


def assert_checkpoint_matches_backbone(
    state_dict: dict[str, Any], *, provenance: ProposalProvenance
) -> None:
    """Fail closed when a checkpoint was trained on a different backbone."""
    checkpoint_vocab = backbone_from_state_dict(state_dict)
    if checkpoint_vocab != provenance.vocab_size:
        msg = (
            "checkpoint backbone does not match the requested model: checkpoint "
            f"vocabulary has {checkpoint_vocab} rows but model "
            f"{provenance.model_config!r} ({provenance.encoder_name}) expects "
            f"{provenance.vocab_size}. Candidate JSONL is backbone-specific and "
            "must not be shared across backbones."
        )
        raise ValueError(msg)
