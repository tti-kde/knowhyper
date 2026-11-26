"""Pre-registered contrast families for the #220 span text campaign.

Issue #220 fixes one PRIMARY family and two SECONDARY families, each Holm-corrected
WITHIN itself:

1. ``text_real - text_sham`` -- 6 contrasts (2 backbones x 3 datasets). Does the
   retrieved text's CONTENT matter? This is the claim Table 6 makes.
2. ``text_real - base`` -- 6 contrasts. Total effect of the text route.
3. ``text_sham - base`` -- 6 contrasts. Effect of the added segment/frame alone.

Pooling all 18 into one family would change which contrasts survive, so the split
is part of the pre-registered protocol rather than a caller's choice. Reporting
only family 2 and calling it a knowledge-content effect is an explicit non-goal of
the issue: without the sham, `text_real - base` cannot separate content from the
mere presence of a second segment.

The arithmetic (`paired_delta`, `holm`) is shared with #218 rather than
reimplemented -- the families differ, the statistics do not.
"""

from __future__ import annotations

from dataclasses import dataclass

from knowhyper.span_campaign import BACKBONES, DATASETS
from knowhyper.span_stats import PairedResult, holm, paired_delta

_BACKBONE_NAMES = tuple(BACKBONES)

__all__ = [
    "PairedResult",
    "TextContrast",
    "holm",
    "paired_delta",
    "text_contrast_families",
]


@dataclass(frozen=True)
class TextContrast:
    """One pre-registered comparison between two arms of the same cell."""

    kind: str
    backbone: str
    dataset: str
    treated: str
    control: str

    @property
    def label(self) -> str:
        return f"{self.kind}/{self.backbone}/{self.dataset}"


# (family name, treated arm, control arm). The primary family is first.
_FAMILIES: tuple[tuple[str, str, str], ...] = (
    ("text_real_minus_text_sham", "text_real", "text_sham"),
    ("text_real_minus_base", "text_real", "base"),
    ("text_sham_minus_base", "text_sham", "base"),
)

PRIMARY_FAMILY = _FAMILIES[0][0]


def text_contrast_families(
    *,
    backbones: tuple[str, ...] = _BACKBONE_NAMES,
    datasets: tuple[str, ...] = DATASETS,
) -> dict[str, list[TextContrast]]:
    """Build the one primary and two secondary families for an explicit scope."""
    return {
        name: [
            TextContrast(
                kind=name,
                backbone=backbone,
                dataset=dataset,
                treated=treated,
                control=control,
            )
            for backbone in backbones
            for dataset in datasets
        ]
        for name, treated, control in _FAMILIES
    }
