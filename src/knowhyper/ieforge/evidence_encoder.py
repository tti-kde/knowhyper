"""A second, independent, full text encoder -- shared by #189 and #233 Phase A0.

#189's `ParallelTextGatedKnowledgeModel` already solved this problem: load a head-free
encoder from the SAME checkpoint as the inner model so both start identically initialised,
then train it separately. #233 needs exactly that for candidate-owned text, so the builder
is extracted here rather than re-invented -- a new "small encoder" would be a second,
unaudited initialisation and pooling decision.

Two properties this module exists to guarantee.

**The checkpoint follows the anchor.** `checkpoint_of` reads `_name_or_path` off the LOADED
inner config rather than a config key, so a PubMedBERT anchor gets a PubMedBERT evidence
encoder and a BioLinkBERT anchor gets a BioLinkBERT one, automatically. Pinning one arm to
PubMedBERT would make the BioLinkBERT arm a cross-backbone hybrid and the final per-backbone
comparison uninterpretable -- and it would train fine and report a plausible score.

**Pooling is fixed here, before any GPU run.** Attention-masked mean over content tokens,
excluding padding and special tokens. Including padding would make a candidate's embedding a
function of the batch's longest text -- the same candidate encoding differently in different
batches, which is non-reproducible by construction and not something a fingerprint could
describe. Including `[CLS]`/`[SEP]` dilutes short strings far more than long ones, and T0
measured that this corpus is dominated by short synonym strings (median 13 tokens).

NOTE ON THE LR GROUP. This encoder is NOT under `model.inner.`, so `BACKBONE_PREFIX` does
not match it and `difflr_*` would put it in the KNOWLEDGE group at 10x LR. Arms using it
must run `trainer=nolog`. See the `second-bert-and-difflr-are-incompatible` finding.
"""

from __future__ import annotations

from typing import Any

import torch  # noqa: TC002 - used at runtime by masked_mean_pool, not only in hints

#: The one pooling strategy, named so it can be recorded in a manifest and compared across
#: runs. Changing it means a new name, not a silent edit.
POOLING = "masked_mean_no_special"


class EvidenceEncoderError(ValueError):
    """The evidence encoder cannot be built or used consistently."""


def build_evidence_encoder(
    checkpoint: str,
    *,
    extra_token_count: int = 0,
    expected_hidden_size: int | None = None,
) -> Any:
    """Load an independent, head-free encoder from `checkpoint`.

    Lifted from `ParallelTextGatedKnowledgeModel._build_evidence_encoder`, including its
    `BertConfig` fallback: some checkpoints ship no `model_type` in `config.json`
    (`prajjwal1/bert-tiny`, which the unit tests use), and `AutoModel` cannot dispatch
    without it. Without the fallback this code would be untestable offline while working on
    PubMedBERT -- i.e. the tests would skip rather than the model be verified.
    """
    from transformers import AutoModel, BertConfig, BertModel

    if extra_token_count < 0:
        msg = f"extra_token_count must be non-negative, got {extra_token_count}"
        raise EvidenceEncoderError(msg)

    try:
        encoder = AutoModel.from_pretrained(checkpoint)
    except ValueError:
        config = BertConfig.from_pretrained(checkpoint)
        encoder = BertModel.from_pretrained(checkpoint, config=config)

    if extra_token_count > 0:
        # The evidence sequence carries the same added marker tokens as the source, so the
        # embedding table must be resized identically or those ids index out of range.
        embeddings = encoder.get_input_embeddings()
        encoder.resize_token_embeddings(
            int(embeddings.num_embeddings) + extra_token_count
        )

    hidden = int(encoder.config.hidden_size)
    if expected_hidden_size is not None and hidden != int(expected_hidden_size):
        msg = (
            f"evidence encoder hidden size {hidden} != anchor hidden size "
            f"{expected_hidden_size}. The pooled evidence is scattered into the anchor's "
            "span representation, so one shared width is required."
        )
        raise EvidenceEncoderError(msg)
    return encoder


def checkpoint_of(inner: Any) -> str:
    """Return the inner model's own checkpoint id, so both encoders match.

    Read from the LOADED config rather than a config key: a backbone swap that left this
    pointing at the old checkpoint would pair, say, a BioLinkBERT anchor with a PubMedBERT
    evidence encoder. That trains cleanly and reports a plausible score, and it would make
    the per-backbone comparison this campaign reports uninterpretable.
    """
    config = getattr(getattr(inner, "config", None), "_name_or_path", "")
    if not config:
        for attribute in ("model", "bert", "encoder"):
            nested = getattr(inner, attribute, None)
            config = getattr(getattr(nested, "config", None), "_name_or_path", "")
            if config:
                break
    if not config:
        msg = (
            "cannot infer the evidence encoder checkpoint from the anchor model; the "
            "loaded config carries no _name_or_path. Pass the checkpoint explicitly "
            "rather than defaulting, so the two encoders cannot silently diverge."
        )
        raise EvidenceEncoderError(msg)
    return str(config)


def masked_mean_pool(
    states: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    special_tokens_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean-pool `[N, L, H]` over content tokens only, returning `[N, H]`.

    An all-masked row pools to zeros rather than NaN. NaN would propagate through the shared
    head into every candidate's logit, so one candidate with empty text would corrupt the
    whole batch instead of simply carrying no evidence.
    """
    if states.dim() != 3:
        msg = f"states must be [N, L, H], got {tuple(states.shape)}"
        raise EvidenceEncoderError(msg)
    if attention_mask.shape != states.shape[:2]:
        msg = (
            f"attention_mask shape {tuple(attention_mask.shape)} does not match states "
            f"{tuple(states.shape[:2])}"
        )
        raise EvidenceEncoderError(msg)

    keep = attention_mask.bool()
    if special_tokens_mask is not None:
        if special_tokens_mask.shape != states.shape[:2]:
            msg = (
                f"special_tokens_mask shape {tuple(special_tokens_mask.shape)} does not "
                f"match states {tuple(states.shape[:2])}"
            )
            raise EvidenceEncoderError(msg)
        keep = keep & ~special_tokens_mask.bool()

    weights = keep.unsqueeze(-1).to(states.dtype)
    total = (states * weights).sum(dim=1)
    count = weights.sum(dim=1)
    # clamp_min rather than a branch: keeps the graph shape constant, and the numerator is
    # already exactly zero for an all-masked row, so the result is zero rather than NaN.
    return total / count.clamp_min(1.0)


def encoder_manifest(
    encoder: Any, *, checkpoint: str, trainable: bool, tokenizer: str | None = None
) -> dict[str, Any]:
    """Record what actually ran, for the run's manifest.

    #233 requires model revision, tokenizer revision, pooling and trainability to be
    recorded: every one of them changes what the arm measures while leaving the score
    plausible, so none can be reconstructed after the fact from the number alone.
    """
    config = getattr(encoder, "config", None)
    return {
        "checkpoint": str(checkpoint),
        "tokenizer": str(tokenizer) if tokenizer else str(checkpoint),
        "model_type": str(getattr(config, "model_type", "") or ""),
        "hidden_size": int(getattr(config, "hidden_size", 0) or 0),
        "num_hidden_layers": int(getattr(config, "num_hidden_layers", 0) or 0),
        "vocab_size": int(getattr(config, "vocab_size", 0) or 0),
        "pooling": POOLING,
        "trainable": bool(trainable),
        "parameter_count": sum(p.numel() for p in encoder.parameters()),
    }
