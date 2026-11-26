"""Prediction-backed knowledge queries for deployable two-pass NER.

Gold ``NERExample.entities`` remain supervision for the base NER collator.  This
module reads first-pass predictions from JSONL and exposes only those predicted
mentions to the knowledge pipeline, preventing an oracle gold-mention leak.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from ieforge.data.schema import Entity
from ieforge.knowledge.core.base import IEKnowledgeQueryBuilder
from ieforge.knowledge.core.schema import Query
from ieforge.registry import register

from knowhyper.cache.keys import InstanceKey
from knowhyper.ieforge.ner_candidates import NERSpanCandidate


@dataclass(frozen=True)
class NERPredictionRecord:
    """One first-pass NER prediction record."""

    text: str
    entities: tuple[Entity, ...]
    candidates: tuple[NERSpanCandidate, ...]


class NERPredictionStore:
    """Load and validate ``{split}.jsonl`` first-pass NER predictions lazily."""

    def __init__(self, root: str | Path | None) -> None:
        if root is None:
            msg = (
                "NER knowledge integration requires first-pass predictions; set "
                "knowledge.pipeline.query_builder.predictions_root"
            )
            raise ValueError(msg)
        self.root = Path(root)
        self._records: dict[str, dict[str, NERPredictionRecord]] = {}
        self._fingerprints: dict[str, str] = {}

    def _load(self, split: str) -> None:
        if split in self._records:
            return
        path = self.root / f"{split}.jsonl"
        if not path.is_file():
            msg = f"NER prediction file does not exist: {path}"
            raise FileNotFoundError(msg)

        raw = path.read_bytes()
        fingerprint = hashlib.sha256(raw).hexdigest()[:16]
        records: dict[str, NERPredictionRecord] = {}
        for line_no, line in enumerate(raw.decode("utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                msg = f"Invalid JSON in {path}:{line_no}"
                raise ValueError(msg) from exc
            prediction_id = str(payload.get("id", ""))
            if not prediction_id:
                msg = f"Missing prediction id in {path}:{line_no}"
                raise ValueError(msg)
            if prediction_id in records:
                msg = f"Duplicate prediction id {prediction_id!r} in {path}"
                raise ValueError(msg)

            text = payload.get("text")
            if not isinstance(text, str):
                msg = f"Missing prediction text for id {prediction_id!r} in {path}"
                raise TypeError(msg)
            raw_entities = payload.get("entities")
            if not isinstance(raw_entities, list):
                msg = f"Invalid entities for id {prediction_id!r} in {path}"
                raise TypeError(msg)
            entities: list[Entity] = []
            for entity_no, raw_entity in enumerate(raw_entities):
                if not isinstance(raw_entity, dict):
                    msg = (
                        f"Invalid entity {entity_no} for id {prediction_id!r} in {path}"
                    )
                    raise TypeError(msg)
                entity_data = cast("dict[str, Any]", raw_entity)
                try:
                    entities.append(
                        Entity(
                            start=int(entity_data["start"]),
                            end=int(entity_data["end"]),
                            text=str(entity_data["text"]),
                            label=str(entity_data["label"]),
                        )
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    msg = (
                        f"Invalid entity {entity_no} for id {prediction_id!r} in {path}"
                    )
                    raise ValueError(msg) from exc
            raw_candidates = payload.get("candidates")
            candidates: list[NERSpanCandidate] = []
            if raw_candidates is None:
                candidates = [
                    NERSpanCandidate(
                        start=entity.start,
                        end=entity.end,
                        text=entity.text,
                        label=entity.label,
                        confidence=1.0,
                        entropy=0.0,
                        source="decoded",
                    )
                    for entity in entities
                ]
            elif not isinstance(raw_candidates, list):
                msg = f"Invalid candidates for id {prediction_id!r} in {path}"
                raise TypeError(msg)
            else:
                for candidate_no, raw_candidate in enumerate(raw_candidates):
                    if not isinstance(raw_candidate, dict):
                        msg = (
                            f"Invalid candidate {candidate_no} for id "
                            f"{prediction_id!r} in {path}"
                        )
                        raise TypeError(msg)
                    candidate_data = cast("dict[str, Any]", raw_candidate)
                    try:
                        candidate = NERSpanCandidate(
                            start=int(candidate_data["start"]),
                            end=int(candidate_data["end"]),
                            text=str(candidate_data["text"]),
                            label=str(candidate_data["label"]),
                            confidence=float(candidate_data["confidence"]),
                            entropy=float(candidate_data["entropy"]),
                            source=str(candidate_data.get("source", "lattice")),
                        )
                    except (KeyError, TypeError, ValueError) as exc:
                        msg = (
                            f"Invalid candidate {candidate_no} for id "
                            f"{prediction_id!r} in {path}"
                        )
                        raise ValueError(msg) from exc
                    if (
                        candidate.start < 0
                        or candidate.end <= candidate.start
                        or candidate.end > len(text)
                        or text[candidate.start : candidate.end] != candidate.text
                    ):
                        msg = (
                            f"Candidate span/text mismatch for id {prediction_id!r} "
                            f"in {path}: {candidate!r}"
                        )
                        raise ValueError(msg)
                    if not (
                        0.0 <= candidate.confidence <= 1.0
                        and 0.0 <= candidate.entropy <= 1.0
                    ):
                        msg = (
                            f"Candidate uncertainty outside [0, 1] for id "
                            f"{prediction_id!r} in {path}"
                        )
                        raise ValueError(msg)
                    candidates.append(candidate)
            records[prediction_id] = NERPredictionRecord(
                text=text,
                entities=tuple(entities),
                candidates=tuple(candidates),
            )

        self._records[split] = records
        self._fingerprints[split] = fingerprint

    def get(self, split: str, prediction_id: str, *, text: str) -> NERPredictionRecord:
        """Return one prediction and fail closed on missing/misaligned input."""
        self._load(split)
        try:
            record = self._records[split][prediction_id]
        except KeyError as exc:
            msg = f"No first-pass NER prediction for split={split!r}, id={prediction_id!r}"
            raise KeyError(msg) from exc
        if record.text != text:
            msg = f"Prediction text mismatch for split={split!r}, id={prediction_id!r}"
            raise ValueError(msg)
        return record

    def fingerprint(self, split: str) -> str:
        """Return the content hash for one split's prediction file."""
        self._load(split)
        return self._fingerprints[split]


def prediction_scoped_instance_key(
    instance: InstanceKey, prediction_fingerprint: str | None
) -> InstanceKey:
    """Scope a cache instance to the exact first-pass prediction file."""
    if not prediction_fingerprint:
        return instance
    return InstanceKey(
        dataset=f"{instance.dataset}@pred-{prediction_fingerprint}",
        split=instance.split,
        instance_id=instance.instance_id,
    )


def _get(obj: Any, name: str) -> Any:
    if isinstance(obj, dict):
        return obj[name]
    return getattr(obj, name)


@register("knowledge_query_builder", name="ner_prediction")
class NERPredictionQueryBuilder(IEKnowledgeQueryBuilder):
    """Build NER knowledge queries exclusively from first-pass predictions."""

    def __init__(
        self,
        *,
        predictions_root: str | Path | None,
        field: str = "text",
        task: str | None = None,
        metadata_fields: list[str] | None = None,
        mention_source: str = "entities",
    ) -> None:
        if mention_source not in ("entities", "candidates"):
            msg = (
                "mention_source must be 'entities' or 'candidates', got "
                f"{mention_source!r}"
            )
            raise ValueError(msg)
        self._field = field
        self._task = task
        self._metadata_fields = list(metadata_fields or [])
        self._mention_source = mention_source
        self._predictions = NERPredictionStore(predictions_root)

    def build(self, example: Any) -> Query:
        """Look up the matching predicted mentions without reading gold entities."""
        text = str(_get(example, self._field))
        split = str(_get(example, "split"))
        prediction_id = str(_get(example, "id"))
        record = self._predictions.get(split, prediction_id, text=text)

        metadata = {name: _get(example, name) for name in self._metadata_fields}
        mentions = (
            record.entities if self._mention_source == "entities" else record.candidates
        )
        metadata["entity_types"] = [mention.label for mention in mentions]
        # Preserve first-pass mention provenance for NER-specific fusion. Query.entities
        # remains the surface-only compatibility path used by the linker, while these
        # offsets let the model restrict knowledge residuals to the tokens that produced
        # the retrieval seed. They are predictions, never gold annotations.
        metadata["entity_spans"] = [
            (mention.start, mention.end) for mention in mentions
        ]
        metadata["entity_confidences"] = [
            float(getattr(mention, "confidence", 1.0)) for mention in mentions
        ]
        metadata["entity_entropies"] = [
            float(getattr(mention, "entropy", 0.0)) for mention in mentions
        ]
        metadata["entity_sources"] = [
            str(getattr(mention, "source", "decoded")) for mention in mentions
        ]
        metadata["prediction_fingerprint"] = self._predictions.fingerprint(split)
        return Query(
            text=text,
            entities=[mention.text for mention in mentions],
            task=self._task,
            metadata=metadata,
        )
