from __future__ import annotations

from pathlib import Path

# Dataset *product* names (repo id suffix / documentation label).
PRIMEHKG_CORE_PRODUCT = "primehkg-core"
PRIMEHKG_FEATURES_PRODUCT = "primehkg-features"

# Hugging Face Datasets config names (one schema per config).
PRIMEHKG_CORE_HF_CONFIGS: tuple[str, ...] = ("nodes", "hyperedges", "support")
PRIMEHKG_FEATURES_HF_CONFIGS: tuple[str, ...] = (
    "node_features_text",
    "node_features_sequence",
    "node_features_structure",
    "node_features_property",
    "node_features_embedding",
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DATASET_CARDS = _REPO_ROOT / "docs" / "huggingface" / "dataset-cards"


def hub_config_names_for_product(product: str) -> tuple[str, ...]:
    """Return HF config names for a known product label."""
    if product == PRIMEHKG_CORE_PRODUCT:
        return PRIMEHKG_CORE_HF_CONFIGS
    if product == PRIMEHKG_FEATURES_PRODUCT:
        return PRIMEHKG_FEATURES_HF_CONFIGS
    msg = f"unknown PrimeHKG Hub product: {product!r}"
    raise ValueError(msg)


def default_dataset_card_path(product: str) -> Path:
    """In-repo dataset card template for ``primehkg-hf-push --dataset-readme``."""
    if product == PRIMEHKG_CORE_PRODUCT:
        return _DATASET_CARDS / "primehkg-core.md"
    if product == PRIMEHKG_FEATURES_PRODUCT:
        return _DATASET_CARDS / "primehkg-features.md"
    msg = f"unknown PrimeHKG Hub product: {product!r}"
    raise ValueError(msg)
