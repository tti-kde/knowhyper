from abc import ABC, abstractmethod
from typing import Any


class IETask(ABC):
    """Abstract base class for information extraction tasks."""

    @abstractmethod
    def encode(self, example: Any, tokenizer: Any) -> Any:
        """Convert an IE example into model-consumable features."""

    @abstractmethod
    def decode(self, outputs: Any, batch: dict[str, Any]) -> list:
        """Convert model outputs into task-level predictions."""
