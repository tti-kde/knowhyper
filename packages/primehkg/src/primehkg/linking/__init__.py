"""Biomedical entity linking for PrimeHKG consumers."""

from primehkg.linking.build import (
    LinkResolverBuildInfo,
    build_link_resolvers,
    insert_extra_resolvers,
)
from primehkg.linking.pipeline import LinkPipeline
from primehkg.linking.presets import DEFAULT_LINKING, SAPBERT_LINKING
from primehkg.linking.xref_index import XrefIndex, normalize_label

__all__ = [
    "DEFAULT_LINKING",
    "SAPBERT_LINKING",
    "LinkPipeline",
    "LinkResolverBuildInfo",
    "XrefIndex",
    "build_link_resolvers",
    "insert_extra_resolvers",
    "normalize_label",
]


def __getattr__(name: str) -> type[LinkPipeline]:
    if name == "DefaultLinkPipeline":
        from primehkg.linking.pipeline import DefaultLinkPipeline

        return DefaultLinkPipeline
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
