"""KnowHyper's ieforge integration layer.

Importing this package registers KnowHyper's ieforge components (the
cross-attention model and the knowledge injector) via their ``@register``
decorators, and re-exports ``CONFIGS_DIR`` for Hydra search-path wiring.
"""

from knowhyper.ieforge import channels, injector, models, noop, span_knowledge_model
from knowhyper.ieforge._configs import CONFIGS_DIR

__all__ = [
    "CONFIGS_DIR",
    "channels",
    "injector",
    "models",
    "noop",
    "span_knowledge_model",
]
