"""Build link resolver chains from presets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from primehkg.linking.chemical_xref_resolver import ChemicalExternalIdResolver
from primehkg.linking.resolvers import (
    ExactLabelResolver,
    ExactNodeIdResolver,
    ExactXrefResolver,
    FuzzyLabelResolver,
)

if TYPE_CHECKING:
    from primehkg.core.dataset import CoreDataset
    from primehkg.linking.presets import LinkingPreset
    from primehkg.linking.protocols import LinkResolver

# ExactNodeId + ExactXref + ChemicalExternalId + ExactLabel
EXACT_RESOLVER_COUNT = 4


def _lexical_resolver_types() -> tuple[type[object], ...]:
    from primehkg.linking.sapbert_resolver import SapBERTLinkResolver

    return (FuzzyLabelResolver, SapBERTLinkResolver)


@dataclass(frozen=True)
class LinkResolverBuildInfo:
    """Metadata from ``build_link_resolvers`` (sidecar presence, lexical mode)."""

    sapbert_sidecar_missing: bool
    lexical_mode: Literal["fuzzy", "sapbert", "none"]


def build_exact_resolvers(dataset: CoreDataset) -> list[LinkResolver]:
    idx = dataset.xref_index
    return [
        ExactNodeIdResolver(idx),
        ExactXrefResolver(idx),
        ChemicalExternalIdResolver(dataset.chemical_xref_index),
        ExactLabelResolver(idx),
    ]


def build_link_resolvers(
    dataset: CoreDataset,
    preset: LinkingPreset,
) -> tuple[list[LinkResolver], LinkResolverBuildInfo]:
    from primehkg.linking.sapbert_resolver import SapBERTLinkResolver

    built = build_exact_resolvers(dataset)
    enable_fuzzy = bool(preset.get("enable_fuzzy", True))
    if enable_fuzzy:
        wj, wl = preset.get("fuzzy_weights") or (0.3, 0.7)
        built.append(
            FuzzyLabelResolver(dataset.xref_index, w_jaccard=wj, w_levenshtein=wl)
        )
        return built, LinkResolverBuildInfo(
            sapbert_sidecar_missing=False,
            lexical_mode="fuzzy",
        )
    label_index = dataset.label_embedding_index
    if label_index is not None:
        built.append(SapBERTLinkResolver(label_index, preset=preset))
        return built, LinkResolverBuildInfo(
            sapbert_sidecar_missing=False,
            lexical_mode="sapbert",
        )
    return built, LinkResolverBuildInfo(
        sapbert_sidecar_missing=True,
        lexical_mode="none",
    )


def insert_extra_resolvers(
    resolvers: list[LinkResolver],
    extra: list[LinkResolver],
) -> list[LinkResolver]:
    """Insert ``extra`` after the exact resolvers, before lexical."""
    if not extra:
        return list(resolvers)
    if len(resolvers) < EXACT_RESOLVER_COUNT:
        msg = (
            f"expected at least {EXACT_RESOLVER_COUNT} resolvers, got {len(resolvers)}"
        )
        raise ValueError(msg)
    return [
        *resolvers[:EXACT_RESOLVER_COUNT],
        *extra,
        *resolvers[EXACT_RESOLVER_COUNT:],
    ]


def is_lexical_resolver(resolver: LinkResolver) -> bool:
    return type(resolver) in _lexical_resolver_types()
