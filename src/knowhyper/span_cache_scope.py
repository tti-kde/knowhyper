"""Prediction-scoped SubHG cache resolution for #218 §0.3.

The plumbing already exists: `NERPredictionQueryBuilder` puts the JSONL's sha256
into `query.metadata["prediction_fingerprint"]`, and `injector.py` scopes the cache
instance key with `prediction_scoped_instance_key`, giving on-disk entries under
`<subhg-root>/<retrieval-fingerprint>/<dataset>@pred-<sha>/<split>/<id>.json`.

What was missing is the operator-facing half: which scope does a given
backbone x dataset resolve to, is that cache actually complete, and -- the property
the campaign rests on -- are the two backbones' scopes genuinely disjoint. A shared
scope would hand one backbone the other's retrieved graph with no error at all,
because both lattices are valid inputs.

Note the retrieval fingerprint (profile + linker + pins) and the prediction scope are
INDEPENDENT: the former is the parent directory, the latter the `<dataset>@pred-<sha>`
leaf. Two backbones share the retrieval fingerprint -- they must, since retrieval
semantics are a dataset property -- and differ only in the prediction leaf.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path  # noqa: TC003

SPLITS: tuple[str, ...] = ("train", "validation", "test")

# The cache leaf uses the DATASET CONFIG NAME (`cfg.dataset.name`), not the task.
# `InstanceKey.from_metadata` reads the `dataset` metadata field, which the datamodule
# fills from `dataset.name` -- so entries land under `blurb_ner_bc2gm@pred-<sha>`, not
# `bc2gm@pred-<sha>`. Using the task here makes the completeness check inspect a
# directory nothing writes, so every scope reads INCOMPLETE while the cache is built.
_DATASET_CONFIG_NAME = "blurb_ner_{task}".format


@dataclass(frozen=True)
class PredictionScope:
    """One backbone x dataset x split prediction identity."""

    backbone: str
    dataset: str
    split: str
    prediction_sha: str
    rows: int
    path: Path

    @property
    def dataset_config_name(self) -> str:
        """`cfg.dataset.name` for this task, which is what the cache key carries."""
        return _DATASET_CONFIG_NAME(task=self.dataset)

    @property
    def scoped_dataset(self) -> str:
        """The `<dataset.name>@pred-<sha>` leaf the injector writes under."""
        return f"{self.dataset_config_name}@pred-{self.prediction_sha}"


@dataclass(frozen=True)
class ScopeReport:
    """How much of one scope's cache is built."""

    scope: PredictionScope
    cached: int

    @property
    def complete(self) -> bool:
        return self.cached >= self.scope.rows

    @property
    def missing(self) -> int:
        return max(0, self.scope.rows - self.cached)


def _fingerprint_and_rows(path: Path) -> tuple[str, int]:
    """Return the store's 16-hex prediction fingerprint and the row count.

    Must match `NERPredictionStore._load`: sha256 of the RAW BYTES, truncated to 16.
    Recomputing it differently here would report a scope that no cache uses.
    """
    raw = path.read_bytes()
    rows = sum(1 for line in raw.decode("utf-8").splitlines() if line.strip())
    return hashlib.sha256(raw).hexdigest()[:16], rows


def resolve_prediction_scopes(
    predictions_root: Path,
    *,
    backbones: list[str],
    datasets: list[str],
    splits: tuple[str, ...] = SPLITS,
) -> list[PredictionScope]:
    """Resolve every backbone x dataset x split scope under a predictions root.

    Fails closed on a missing or empty split: the knowledge arm would retrieve
    nothing for it and still report a score.
    """
    scopes: list[PredictionScope] = []
    for backbone in backbones:
        for dataset in datasets:
            base = predictions_root / backbone / dataset
            for split in splits:
                path = base / f"{split}.jsonl"
                if not path.is_file():
                    msg = (
                        f"prediction file missing for {backbone}/{dataset}: {path} "
                        f"(split {split!r}); every knowledge cell needs all of {splits}"
                    )
                    raise FileNotFoundError(msg)
                sha, rows = _fingerprint_and_rows(path)
                if rows == 0:
                    msg = f"prediction file has no rows: {path}"
                    raise ValueError(msg)
                scopes.append(
                    PredictionScope(
                        backbone=backbone,
                        dataset=dataset,
                        split=split,
                        prediction_sha=sha,
                        rows=rows,
                        path=path,
                    )
                )
    return scopes


def verify_scope_completeness(
    scope: PredictionScope, retrieval_root: Path
) -> ScopeReport:
    """Count cached instances for a scope under one retrieval fingerprint.

    `retrieval_root` is `<subhg-root>/<retrieval-fingerprint>`; the scope leaf is
    appended. A partial cache is reported as incomplete rather than passing: an arm
    training against a half-built cache silently sees a subset of the graph.
    """
    entries = retrieval_root / scope.scoped_dataset / scope.split
    if not entries.is_dir():
        return ScopeReport(scope=scope, cached=0)
    cached = sum(1 for path in entries.iterdir() if path.suffix == ".json")
    return ScopeReport(scope=scope, cached=cached)


def assert_backbone_scopes_disjoint(scopes: list[PredictionScope]) -> None:
    """Fail when two backbones share a prediction sha for the same dataset/split.

    That would mean byte-identical candidate lattices from two different first-pass
    encoders -- possible only if one backbone's artifact was copied to the other,
    which #218 forbids because it compares architectures against one proposal
    distribution while claiming two.
    """
    seen: dict[tuple[str, str, str], str] = {}
    for scope in scopes:
        key = (scope.dataset, scope.split, scope.prediction_sha)
        if key in seen and seen[key] != scope.backbone:
            msg = (
                f"backbones {seen[key]!r} and {scope.backbone!r} share prediction sha "
                f"{scope.prediction_sha} for {scope.dataset}/{scope.split}; candidate "
                "JSONL must never be shared across backbones"
            )
            raise ValueError(msg)
        seen[key] = scope.backbone
