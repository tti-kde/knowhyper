from __future__ import annotations

import warnings
from enum import StrEnum
from typing import assert_never


class BuildProfile(StrEnum):
    """Which PrimeKG-derived builder segments ``primehkg-build`` runs by default."""

    PRIMEHKG = "primehkg"  # default: variant-FREE biomedical core
    PRIMEHKG_CLINICAL = "primehkg_clinical"  # opt-in preset: primehkg + ClinVar variant
    SMOKE = "smoke"


_EXPLAINABLE_SLUGS = frozenset(
    {
        "moa_triad",
        "phenotype",
        "disease_protein_bundle",
        "protein_disease_bundle",
        "protein_protein_bundle",
    }
)

_PRIMEHKG_SLUGS = _EXPLAINABLE_SLUGS | frozenset(
    {
        "registry_atomic",
        "drug_disease_shared_target_bundle",
        "corum",
        "reactome",
        "signor",
        "complexportal",
        "drugcomb",
        "stitch",
        "ctd",
    }
)  # NOTE: clinvar removed — default is variant-free
_PRIMEHKG_CLINICAL_SLUGS = _PRIMEHKG_SLUGS | frozenset({"clinvar"})
_SMOKE_SLUGS = _EXPLAINABLE_SLUGS

# single source of truth for the segment vocabulary (include/exclude validation).
# phenotype_composed is a recognized opt-in slug (default OFF in every profile): it drives
# the composed drug_disease_phenotype family, distinct from the atomic `phenotype` builder.
# It needs no extra source (PrimeKG kg.csv, same as `phenotype`), so it is added to the
# vocabulary only — not to any profile base set or _SEGMENT_REQUIRED_SOURCES.
# stitch is now in the default primehkg profile (via _PRIMEHKG_SLUGS); it drives the
# STITCH chemical-protein drug_protein builder and requires the three stitch TSV inputs
# listed in _SEGMENT_REQUIRED_SOURCES (stitch_tsv / unichem_cid_map_tsv / string_ensp_geneid_tsv).
# ctd is now in the default primehkg profile (via _PRIMEHKG_SLUGS); it drives the
# CTD chemical→protein regulation builder and requires --ctd-tsv / --ctd-mesh-cid-tsv
# (plus the shared --unichem-cid-map-tsv) listed in _SEGMENT_REQUIRED_SOURCES.
ALL_SEGMENT_SLUGS = _PRIMEHKG_CLINICAL_SLUGS | frozenset({"phenotype_composed"})

_LEGACY_ALIASES: dict[str, str] = {
    "coverage": BuildProfile.PRIMEHKG.value,
    "coverage_shared": BuildProfile.PRIMEHKG.value,
    "coverage_curated": BuildProfile.PRIMEHKG.value,
    "coverage_shared_curated": BuildProfile.PRIMEHKG.value,
    "full": BuildProfile.PRIMEHKG.value,
}

_REMOVED_WITHOUT_ALIAS = frozenset({"explainable"})


def parse_build_profile(value: str) -> BuildProfile:
    normalized = value.strip().lower()
    if normalized in _REMOVED_WITHOUT_ALIAS:
        msg = (
            f"build profile {value!r} was removed; use {BuildProfile.PRIMEHKG.value!r} "
            f"(production) or {BuildProfile.SMOKE.value!r} (CI/fixtures)"
        )
        raise ValueError(msg)
    if normalized in _LEGACY_ALIASES:
        warnings.warn(
            f"build profile {value!r} is deprecated; use {_LEGACY_ALIASES[normalized]!r}",
            DeprecationWarning,
            stacklevel=2,
        )
        normalized = _LEGACY_ALIASES[normalized]
    try:
        return BuildProfile(normalized)
    except ValueError as exc:
        allowed = ", ".join(p.value for p in BuildProfile)
        msg = f"unknown build profile {value!r}; choose one of: {allowed}"
        raise ValueError(msg) from exc


_SEGMENT_SLUG_ALIASES: dict[str, str] = {
    "registry_atomic_wave": "registry_atomic",
}


def _profile_base_slugs(profile: BuildProfile) -> frozenset[str]:
    """Return the canonical slug set for *profile* with no include/exclude adjustments."""
    if profile is BuildProfile.PRIMEHKG:
        return _PRIMEHKG_SLUGS
    if profile is BuildProfile.PRIMEHKG_CLINICAL:
        return _PRIMEHKG_CLINICAL_SLUGS
    if profile is BuildProfile.SMOKE:
        return _SMOKE_SLUGS
    assert_never(profile)


def segment_enabled(profile: BuildProfile, slug: str) -> bool:
    """Return whether a merge segment slug is in this profile's base preset."""
    slug = _SEGMENT_SLUG_ALIASES.get(slug, slug)
    return slug in _profile_base_slugs(profile)


def effective_segments(
    profile: BuildProfile,
    *,
    include: frozenset[str] = frozenset(),
    exclude: frozenset[str] = frozenset(),
) -> frozenset[str]:
    """Resolve the build-time segment set: (profile base | include) - exclude.

    Unknown slugs (after alias normalization) or slugs in both include and exclude raise
    ValueError. The base comes from the profile preset; include/exclude adjust it per build.
    """
    include = frozenset(_SEGMENT_SLUG_ALIASES.get(s, s) for s in include)
    exclude = frozenset(_SEGMENT_SLUG_ALIASES.get(s, s) for s in exclude)
    unknown = (include | exclude) - ALL_SEGMENT_SLUGS
    if unknown:
        msg = f"unknown segment(s): {sorted(unknown)}; choose from {sorted(ALL_SEGMENT_SLUGS)}"
        raise ValueError(msg)
    overlap = include & exclude
    if overlap:
        msg = (
            "segment(s) in both --include-segments and --exclude-segments: "
            f"{sorted(overlap)}"
        )
        raise ValueError(msg)
    return (_profile_base_slugs(profile) | include) - exclude


def profile_requires_curated(profile: BuildProfile) -> bool:
    """Whether this profile's reference recipe requires curated CORUM/Reactome TSVs."""
    return profile in (BuildProfile.PRIMEHKG, BuildProfile.PRIMEHKG_CLINICAL)
