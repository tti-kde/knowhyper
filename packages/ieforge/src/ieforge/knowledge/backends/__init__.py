"""Dependency-light knowledge backends."""

from ieforge.knowledge.backends.features import ConstantGraphFeatureInjector
from ieforge.knowledge.backends.prompts import ConstantPromptInjector

__all__ = ["ConstantGraphFeatureInjector", "ConstantPromptInjector"]
