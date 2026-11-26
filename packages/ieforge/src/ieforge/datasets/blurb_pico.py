import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ieforge.data.schema import PICOExample
from ieforge.datasets.base import IEDataset
from ieforge.registry import register

_HF_PICO_REPO = "tti-kde/blurb-pico"


@dataclass(frozen=True)
class _PICOTaskSpec:
    hf_config: str
    label_list: tuple[str, ...]


_PICO_TASK_REGISTRY: dict[str, _PICOTaskSpec] = {
    "ebm_pico": _PICOTaskSpec(
        hf_config="ebm_pico",
        label_list=("O", "I-PAR", "I-INT", "I-OUT"),
    )
}


def _load_local_pico_examples(tsv_path: Path) -> list[PICOExample]:
    examples: list[PICOExample] = []
    current_id: str | None = None
    current_tokens: list[str] = []
    current_labels: list[str] = []

    def flush() -> None:
        nonlocal current_id, current_tokens, current_labels
        if current_id is None:
            return
        examples.append(
            PICOExample(
                id=current_id,
                tokens=tuple(current_tokens),
                labels=tuple(current_labels),
            )
        )
        current_id = None
        current_tokens = []
        current_labels = []

    with tsv_path.open(encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            if not line:
                continue
            if line.startswith("-DOCSTART- (") and line.endswith(")"):
                flush()
                current_id = line[len("-DOCSTART- (") : -1]
                continue
            token, label = line.split("\t")
            current_tokens.append(token)
            current_labels.append(label)
    flush()
    return examples


def load_dataset(repo: str, config: str, split: str, token: str | None) -> Any:
    """Load from the HF Hub, or from ``IEFORGE_DATA_DIR`` when set (mockable in tests)."""
    from ieforge.datasets.local import load_hf_dataset

    return load_hf_dataset(repo, config, split=split, token=token)


@register("dataset", name="blurb_pico")
class BLURBPICODataset(IEDataset):
    """Canonical BLURB EBM PICO dataset wrapper."""

    def __init__(
        self,
        task: str,
        split: str = "train",
        name: str = "blurb_pico",
        data_root: str | Path | None = None,
    ) -> None:
        """Load the requested BLURB PICO task split."""
        if task not in _PICO_TASK_REGISTRY:
            msg = f"Unknown BLURB PICO task {task!r}. Choose from: {list(_PICO_TASK_REGISTRY)}"
            raise ValueError(msg)
        self._task = task
        self.name = name
        spec = _PICO_TASK_REGISTRY[task]
        if data_root is not None:
            tsv_path = Path(data_root) / task / f"{split}.tsv"
            self._examples = _load_local_pico_examples(tsv_path)
            return

        token = os.environ.get("HF_TOKEN")
        hf = load_dataset(_HF_PICO_REPO, spec.hf_config, split=split, token=token)
        self._examples = [
            PICOExample(
                id=str(record["id"]),
                tokens=tuple(record["tokens"]),
                labels=tuple(record["labels"]),
            )
            for record in hf
        ]

    def __len__(self) -> int:
        """Return the number of PICO examples."""
        return len(self._examples)

    def __getitem__(self, index: int) -> PICOExample:
        """Return the PICO example at the given index."""
        return self._examples[index]

    @property
    def label_list(self) -> tuple[str, ...]:
        """Return the immutable label vocabulary for the selected task."""
        return _PICO_TASK_REGISTRY[self._task].label_list
