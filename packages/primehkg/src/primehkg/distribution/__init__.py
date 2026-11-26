"""Canonical Hugging Face distribution names (Approach A).

Product names (`primehkg-core`, `primehkg-features`) name the **dataset family** and
recommended Hub repo id suffix. Hugging Face ``config_name`` values are the **table**
names (heterogeneous schemas), not the product label.
"""

from primehkg.distribution.hub_layout import (
    PRIMEHKG_CORE_HF_CONFIGS,
    PRIMEHKG_CORE_PRODUCT,
    PRIMEHKG_FEATURES_HF_CONFIGS,
    PRIMEHKG_FEATURES_PRODUCT,
    default_dataset_card_path,
    hub_config_names_for_product,
)

__all__ = [
    "PRIMEHKG_CORE_HF_CONFIGS",
    "PRIMEHKG_CORE_PRODUCT",
    "PRIMEHKG_FEATURES_HF_CONFIGS",
    "PRIMEHKG_FEATURES_PRODUCT",
    "default_dataset_card_path",
    "hub_config_names_for_product",
]
