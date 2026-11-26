"""Small string→index vocabularies for typed-hyperedge encoding.

primehkg validates the FORMAT of `semantics`/`roles` but does not export an enum
of their VALUES; the concrete sets live implicitly in primehkg's builders. These
vocabularies are therefore KnowHyper-curated. Index 0 is always ``<unk>`` so any
out-of-vocab string maps there instead of crashing or silently colliding.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

UNK = "<unk>"


class Vocab:
    """Map a string (or None) to a stable integer index; unknowns → 0 (<unk>)."""

    def __init__(self, values: Sequence[str]) -> None:
        """Build a vocab with ``<unk>`` at index 0 followed by ``values``.

        Idempotent about a leading ``<unk>``: passing a list that already starts
        with ``<unk>`` (e.g. the curated ``ROLE_VOCAB``) does not double it or drop
        a real value, so callers can pass the full curated tuple directly.
        """
        rest = [v for v in values if v != UNK]
        self._tokens: list[str] = [UNK, *rest]
        self._index: dict[str, int] = {t: i for i, t in enumerate(self._tokens)}

    def index(self, s: str | None) -> int:
        """Return the index of ``s``; unknown or None → 0 (``<unk>``)."""
        if s is None:
            return 0
        return self._index.get(s, 0)

    def __len__(self) -> int:
        """Return the vocabulary size (including ``<unk>``)."""
        return len(self._tokens)


# Curated from primehkg/builders/* (moa_triad, contraindication, phenotype,
# drug_disease_polarity, ...) as of 2026-06-16. <unk> covers anything missing.
SEMANTICS_VOCAB: tuple[str, ...] = (
    UNK,
    "moa_triad",
    "contraindication",
    "indication",
    "exposure_disease",
    "drug_disease_phenotype",
    "drug_drug",
    "protein_protein",
    "pathway_membership",
    "entity_attribute",
    "regulation",
)

POLARITY_VOCAB: tuple[str, ...] = (
    UNK,  # "unknown" maps here
    "positive",
    "negative",
    "undirected",
)

ROLE_VOCAB: tuple[str, ...] = (
    UNK,
    "drug",
    "protein",
    "disease",
    "phenotype",
    "exposure",
    "pathway",
    "gene",
    "anchor",
    "attribute:name",
    "attribute:text",
    "attribute:sequence",
    "attribute:smiles",
)

# Relation-signal vocab for ReDaF (relation-gated fusion). The "relation" proxy in
# RE is the dominant sub-HG hyperedge semantics. Fixed, frequency-ordered list built
# from the v0.12.0 ChemProt cache (e95924d8). `<unk>` (id 0) absorbs anything absent.
REDAF_SEMANTICS_VOCAB: tuple[str, ...] = (
    "<unk>",
    "drug_protein",
    "moa_triad",
    "bioprocess_protein",
    "protein_protein",
    "drug_drug",
    "molfunc_protein",
    "cellcomp_protein",
    "regulation",
    "disease_protein",
    "anatomy_protein_present",
    "pathway_protein",
    "molfunc_molfunc",
    "ae_drug_phenotype",
    "anatomy_protein_absent",
    "reaction_event",
    "protein_protein_bundle",
    "contraindication",
    "exposure_exposure",
    "protein_complex",
    "bioprocess_bioprocess",
    "disease_phenotype_context",
    "exposure_bioprocess",
    "pathway_pathway",
    "cellcomp_cellcomp",
    "drug_combination",
    "exposure_disease",
    "exposure_molfunc",
    "indication",
    "exposure_protein",
    "phenotype_protein",
    "disease_protein_bundle",
    "protein_disease_bundle",
    "disease_disease",
    "anatomy_anatomy",
    "phenotype_phenotype",
    "off_label_use",
    "disease_phenotype_negative",
    "drug_disease_shared_target_bundle",
    "exposure_cellcomp",
)

_REDAF_SEMANTICS_INDEX: dict[str, int] = {
    s: i for i, s in enumerate(REDAF_SEMANTICS_VOCAB)
}

REDAF_VOCAB_SIZE = len(REDAF_SEMANTICS_VOCAB)
"""Size of the ReDaF semantics vocab; must equal model.fusion.num_relations."""


def redaf_semantics_id(semantics: str) -> int:
    """Map a hyperedge semantics string to its ReDaF vocab id (0=<unk> if absent)."""
    return _REDAF_SEMANTICS_INDEX.get(semantics, 0)
