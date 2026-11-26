"""Knowledge injection channels."""

from ieforge.knowledge.channels.features import (
    FeatureChannel,
    TensorFeatureChannel,
    TensorFeatureSpec,
)
from ieforge.knowledge.channels.text import TextChannel

__all__ = [
    "FeatureChannel",
    "TensorFeatureChannel",
    "TensorFeatureSpec",
    "TextChannel",
]
