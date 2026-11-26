from __future__ import annotations

from typing import Any, Literal

_DrugDiseasePolarity = Literal["positive", "negative", "unknown"]
_PolarityResolution = Literal[
    "mixed_drug_disease_evidence",
    "no_drug_disease_clinical_edge",
]


def canonical_drug_disease_pair(source: str, target: str) -> tuple[str, str] | None:
    """Return ``(drug, disease)`` when *source*/*target* are a drug-disease edge."""
    if source.startswith("primekg:drug:") and target.startswith("primekg:disease:"):
        return (source, target)
    if source.startswith("primekg:disease:") and target.startswith("primekg:drug:"):
        return (target, source)
    return None


def ingest_drug_disease_clinical_edge(
    *,
    source: str,
    target: str,
    triplet: tuple[str, str, str],
    indications: dict[tuple[str, str], set[tuple[str, str, str]]],
    contraindications: dict[tuple[str, str], set[tuple[str, str, str]]],
) -> bool:
    """Store an indication/contraindication triplet under canonical ``(drug, disease)``."""
    pair = canonical_drug_disease_pair(source, target)
    if pair is None:
        return False
    drug, disease = pair
    relation = triplet[1]
    if relation == "indication":
        indications[(drug, disease)].add(triplet)
        return True
    if relation in ("contraindication", "contraindicated"):
        contraindications[(drug, disease)].add(triplet)
        return True
    return False


def resolve_drug_disease_polarity(
    *,
    drug: str,
    disease: str,
    indications: dict[tuple[str, str], set[tuple[str, str, str]]],
    contraindications: dict[tuple[str, str], set[tuple[str, str, str]]],
) -> tuple[_DrugDiseasePolarity, _PolarityResolution | None]:
    """Align composed-family polarity with shared-target bundle rules."""
    has_ind = bool(indications.get((drug, disease)))
    has_contra = bool(contraindications.get((drug, disease)))
    if has_ind and has_contra:
        return "unknown", "mixed_drug_disease_evidence"
    if has_ind:
        return "positive", None
    if has_contra:
        return "negative", None
    return "unknown", "no_drug_disease_clinical_edge"


def moa_triad_rule_trace(
    *, polarity: _DrugDiseasePolarity, resolution: _PolarityResolution | None
) -> dict[str, Any]:
    trace: dict[str, Any] = {
        "builder": "moa_triad",
        "version": 2,
        "kind": "composed",
        "logic": "drug_protein_and_disease_protein_join_on_protein",
        "polarity_rule": "indication_positive_else_contraindication_negative_else_unknown_mixed_to_unknown",
        "polarity": polarity,
    }
    if resolution is not None:
        trace["polarity_resolution"] = resolution
    return trace
