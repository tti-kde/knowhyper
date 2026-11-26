"""Knowledge augmentors for IEForge datasets."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ieforge.knowledge.core.protocols import KnowledgeChannel, KnowledgeRunner


class KnowledgeAugmentor:
    """Applies a knowledge pipeline through one or more channels."""

    def __init__(
        self,
        pipeline: KnowledgeRunner,
        channels: list[KnowledgeChannel],
    ) -> None:
        """Create an augmentor from a pipeline and channel list."""
        self._pipeline = pipeline
        self._channels = list(channels)

    def augment(self, example: Any) -> Any:
        """Return an augmented copy of an example."""
        context = self._pipeline.run(example)
        augmented = example
        for channel in self._channels:
            augmented = channel.apply(augmented, context)
        return augmented
