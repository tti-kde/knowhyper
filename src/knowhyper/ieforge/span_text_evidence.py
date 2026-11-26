"""#233 Phase A0: candidate-owned TEXT as the evidence channel, plus its lexical control.

Two pieces, kept in one module because they must share the availability signal and the surface
normaliser -- if they drifted apart, `text_real - lexical` would measure the difference in
construction rather than the difference in information.

**`encode_candidate_text`** produces `(kv, kmask, ownership)` in exactly the shapes
`_attend_candidate_evidence` already consumes, so text plugs into the existing residual path
without touching the frozen anchor. That is the whole design constraint: the anchor's logits are
the baseline being preserved, and a text arm that perturbed them would break `zero`'s bitwise
equality with the anchor.

**Not a second segment.** `PairAwareTextKnowledgeCollator` builds one combined `input_ids` +
`token_type_ids`, appending text as sequence B to the *same* encoder. That is what reintroduces
#220's sham/truncation effects, where `text_sham - base` was significantly NEGATIVE in 4/6
cells -- a content-free segment is actively harmful there. So text is encoded separately and
reaches only the residual head.

**Absence is a mask, never a zero.** A zero vector is a legitimate embedding; only the mask can
say "this candidate owns no text". This is the same distinction #171 forced on the link
features, where a zero-filled score was indistinguishable from "linked nothing".

`lexical_features` is the demanding control: it receives the same text-availability signal as
`text_real` but no content, so a tie means the gain was availability plus string matching. It
uses `normalise_surface` -- the SAME normaliser as the T0 coverage figure -- so the control and
the measurement cannot disagree about what counts as a match.
"""

from __future__ import annotations

import torch

from knowhyper.analysis.candidate_text_audit import normalise_surface


class LexicalFeatureError(ValueError):
    """The lexical control's inputs do not line up."""


#: Feature order for `lexical_features`. Recorded so a reader can map a weight back to a
#: feature, and so a reordering shows up as a test failure rather than a silent re-labelling.
LEXICAL_FEATURE_NAMES: tuple[str, ...] = (
    "exact_match",
    "surface_in_evidence",
    "evidence_in_surface",
    "token_overlap",
    "has_text",
)


def encode_candidate_text(
    *,
    texts: list[list[str]],
    mention_count: int,
    hidden_size: int,
    device: torch.device,
    encoder: torch.nn.Module | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode per-mention text into `(kv, kmask, ownership)`.

    One knowledge token per mention, so ownership is diagonal: mention *i* owns token *i* and
    nothing else. That is what "candidate-owned" has to mean -- sharing a token across mentions
    would let one candidate's evidence reach another and make `text_swapped` meaningless.

    `encoder` is injected rather than constructed here so the caller controls its lifetime and,
    critically, its parameter group: a text encoder swept into the differential-LR callback runs
    at 10x because `BACKBONE_PREFIX = "model.inner."` does not match it. Phase A0 therefore uses
    `trainer=nolog`. With `encoder=None` the embedding is a deterministic hash -- enough to test
    the plumbing without a second BERT.
    """
    batch = len(texts)
    width = max(mention_count, 0)
    kv = torch.zeros((batch, width, hidden_size), dtype=torch.float32, device=device)
    kmask = torch.zeros((batch, width), dtype=torch.bool, device=device)
    ownership = torch.zeros((batch, width, width), dtype=torch.bool, device=device)

    for row, row_texts in enumerate(texts):
        for index in range(min(len(row_texts), width)):
            text = str(row_texts[index] or "").strip()
            if not text:
                # Absent: leave the mask False. Do NOT write a zero vector and call it evidence.
                continue
            if encoder is None:
                # Deterministic stand-in so plumbing is testable without a second encoder.
                seed = abs(hash(normalise_surface(text))) % (2**31)
                generator = torch.Generator(device="cpu").manual_seed(seed)
                kv[row, index] = torch.randn(
                    hidden_size, generator=generator, dtype=torch.float32
                ).to(device)
            else:
                kv[row, index] = encoder(text).to(device)
            kmask[row, index] = True
            ownership[row, index, index] = True
    return kv, kmask, ownership


def lexical_features(
    *,
    surfaces: list[str],
    evidence_texts: list[list[str]],
    has_text: list[bool],
) -> torch.Tensor:
    """Surface-match features for the `lexical` control. Carries NO text content.

    Returns `[n, len(LEXICAL_FEATURE_NAMES)]`. `has_text` is included deliberately so the
    control sees the same availability signal as `text_real`; without it, `text_real - lexical`
    would be partly an availability contrast, which is `text_sham`'s job -- and T0 warned the
    observed coverage gap may be linkability rather than text.
    """
    if not (len(surfaces) == len(evidence_texts) == len(has_text)):
        msg = (
            "lexical inputs must align: "
            f"{len(surfaces)} surfaces, {len(evidence_texts)} evidence lists, "
            f"{len(has_text)} availability flags"
        )
        raise LexicalFeatureError(msg)

    rows: list[list[float]] = []
    for surface, evidence, available in zip(
        surfaces, evidence_texts, has_text, strict=True
    ):
        needle = normalise_surface(surface)
        haystacks = [normalise_surface(t) for t in evidence if t]
        exact = float(any(needle and needle == h for h in haystacks))
        inside = float(any(needle and needle in h for h in haystacks))
        contains = float(any(h and h in needle for h in haystacks))
        needle_tokens = set(needle.split())
        overlap = 0.0
        if needle_tokens:
            for haystack in haystacks:
                shared = needle_tokens & set(haystack.split())
                overlap = max(overlap, len(shared) / len(needle_tokens))
        rows.append([exact, inside, contains, overlap, float(available)])
    return torch.tensor(rows, dtype=torch.float32)
