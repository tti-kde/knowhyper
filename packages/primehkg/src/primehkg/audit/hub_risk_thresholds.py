"""Per-semantics top-1 hub_risk thresholds for ``build_report.json``."""

from __future__ import annotations

from typing import Final

# Top-1 node hyperedge share above this value flags ``hub_risk.flagged[semantics]``.
DEFAULT_HUB_RISK_THRESHOLD: Final[float] = 0.2

# Semantics with known hub concentration below the default cutoff on full PrimeKG.
HUB_RISK_THRESHOLDS_BY_SEMANTICS: Final[dict[str, float]] = {
    # ~10.6% top phenotype on shark coverage (see hub-risk thresholds doc).
    "drug_disease_phenotype": 0.10,
    # Small exposure atomics families (shark coverage 2026-05-20).
    "exposure_bioprocess": 0.15,
    "exposure_cellcomp": 0.15,
    "exposure_molfunc": 0.15,
    "exposure_protein": 0.15,
    # Curated / bundle families (higher bar — fewer HE, intentional hubs).
    "drug_disease_shared_target_bundle": 0.25,
    "protein_complex": 0.25,
    "reaction_event": 0.25,
}


def default_hub_risk_threshold() -> float:
    return DEFAULT_HUB_RISK_THRESHOLD


def hub_risk_thresholds_by_semantics() -> dict[str, float]:
    """Return a sorted copy of explicit per-semantics overrides."""
    return {
        k: float(HUB_RISK_THRESHOLDS_BY_SEMANTICS[k])
        for k in sorted(HUB_RISK_THRESHOLDS_BY_SEMANTICS)
    }


def hub_risk_threshold_for_semantics(semantics: str) -> float:
    """Resolve the hub-risk cutoff for a hyperedge semantics slug."""
    sem = semantics.strip() if semantics else "unknown"
    return float(HUB_RISK_THRESHOLDS_BY_SEMANTICS.get(sem, DEFAULT_HUB_RISK_THRESHOLD))


def hub_risk_flagged_semantics(
    top1_ratio_by_semantics: dict[str, float],
) -> dict[str, float]:
    """Return semantics whose ``top1_ratio`` exceeds their configured threshold."""
    out: dict[str, float] = {}
    for sem in sorted(top1_ratio_by_semantics):
        ratio = float(top1_ratio_by_semantics[sem])
        if ratio > hub_risk_threshold_for_semantics(sem):
            out[sem] = ratio
    return out
