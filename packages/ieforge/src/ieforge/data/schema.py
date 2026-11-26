from dataclasses import dataclass, field
from typing import Any


@dataclass
class Entity:
    """A single named entity span in character offsets."""

    start: int
    end: int
    text: str
    label: str
    mention: list[str] | None = None


@dataclass
class NERExample:
    """A text with its associated named entity annotations."""

    id: str
    text: str
    entities: list[Entity] = field(default_factory=list)


@dataclass
class Relation:
    """A directed relation between two entities, referenced by index."""

    head: int
    tail: int
    label: str


@dataclass
class REExample:
    """A text with entity spans and relation annotations."""

    id: str
    text: str
    entities: list[Entity] = field(default_factory=list)
    relations: list[Relation] = field(default_factory=list)


@dataclass(frozen=True)
class REPairExample:
    """A single relation candidate paired with its source document context."""

    id: str
    doc_id: str
    text: str
    entities: list[Entity] = field(default_factory=list)
    relations: list[Relation] = field(default_factory=list)
    head_idx: int = 0
    tail_idx: int = 0
    label: str = ""


@dataclass(frozen=True)
class BioREDRelation:
    """A BioRED relation with optional novelty annotation."""

    head: int
    tail: int
    label: str
    novelty: str | None = None


@dataclass(frozen=True)
class BioREDExample:
    """A BioRED example with entity and relation annotations."""

    id: str
    text: str
    entities: list[Entity] = field(default_factory=list)
    relations: list[BioREDRelation] = field(default_factory=list)


@dataclass(frozen=True)
class BioREDPairExample:
    """A single BioRED PubTator ID-pair candidate."""

    id: str
    doc_id: str
    text: str
    entities: list[Entity] = field(default_factory=list)
    head_idx: int = 0
    tail_idx: int = 1
    label: str = ""
    novelty: str | None = None
    id1: str = ""
    id2: str = ""
    id1_type: str = ""
    id2_type: str = ""
    is_in_same_sent: bool = False
    min_sents_window: int = 0


@dataclass(frozen=True)
class BioREDPairPrediction:
    """A BioRED pair prediction with PubTator IDs preserved."""

    pmid: str
    id1: str
    id2: str
    id1_type: str
    id2_type: str
    gold_label: str
    pred_label: str
    novelty: str | None = None


@dataclass
class QAExample:
    """A question, reference text, and classification label."""

    id: str
    question: str
    context: str
    label: str


@dataclass(frozen=True)
class DocClassificationExample:
    """A document text paired with one or more classification labels."""

    id: str
    text: str
    labels: tuple[str, ...]


@dataclass(frozen=True)
class SentenceSimilarityExample:
    """A sentence pair with a scalar similarity score."""

    id: str
    sentence1: str
    sentence2: str
    score: float


@dataclass(frozen=True)
class PICOExample:
    """A token-labeled EBM PICO sequence."""

    id: str
    tokens: tuple[str, ...]
    labels: tuple[str, ...]


@dataclass
class BioREDirectOutput:
    """Multi-head logits for BioREDirect (relation / novelty / direction)."""

    relation_logits: Any
    novelty_logits: Any
    direction_logits: Any
    loss: Any | None = None

    @property
    def logits(self) -> Any:
        """Alias the relation head so single-head code paths keep working."""
        return self.relation_logits


@dataclass(frozen=True)
class BioREDirectPairExample:
    """A BioRED pair candidate with directionality annotation."""

    id: str
    doc_id: str
    text: str
    label: str = ""
    novelty: str | None = None
    direction: str = "None"
    subject_id: str = "None"
    id1: str = ""
    id2: str = ""
    id1_type: str = ""
    id2_type: str = ""
    is_in_same_sent: bool = False
    min_sents_window: int = 0


@dataclass(frozen=True)
class BioREDirectPairPrediction:
    """A BioREDirect pair prediction including subject/object direction."""

    pmid: str
    id1: str
    id2: str
    id1_type: str
    id2_type: str
    gold_label: str
    pred_label: str
    novelty: str | None = None
    direction: str = "None"
    gold_direction: str = "None"
