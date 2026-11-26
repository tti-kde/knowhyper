import os
from dataclasses import dataclass
from typing import Any

import nltk

from ieforge.data.schema import Entity, NERExample
from ieforge.datasets.base import IEDataset
from ieforge.registry import register


def _get_sentence_tokenizer() -> nltk.tokenize.PunktSentenceTokenizer:
    """Load the English Punkt sentence tokenizer, downloading it if needed."""
    try:
        return nltk.data.load("tokenizers/punkt_tab/english.pickle")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)
        return nltk.data.load("tokenizers/punkt_tab/english.pickle")


def _load_bigbio_kb(
    record: dict,
    sent_tokenizer: nltk.tokenize.PunktSentenceTokenizer,
    entity_types: frozenset[str],
) -> list[NERExample]:
    """Convert a BigBio KB record to sentence-level NERExamples.

    Only entities whose type is in entity_types are kept.
    Title is treated as a single sentence; abstract is Punkt-split.
    Entity offsets are re-based to sentence start.
    """
    doc_id: str = record["document_id"]

    doc_entities: list[Entity] = [
        Entity(start=start, end=end, text=span_text, label=ent["type"])
        for ent in record["entities"]
        if ent["type"] in entity_types
        for (start, end), span_text in zip(ent["offsets"], ent["text"])
    ]

    examples: list[NERExample] = []
    for passage in record["passages"]:
        passage_text: str = passage["text"][0]
        passage_start: int = passage["offsets"][0][0]

        if passage["type"] == "title":
            sent_spans = [(0, len(passage_text))]
        else:
            sent_spans = list(sent_tokenizer.span_tokenize(passage_text)) or [
                (0, len(passage_text))
            ]

        for sent_start, sent_end in sent_spans:
            sent_text = passage_text[sent_start:sent_end]
            abs_start = passage_start + sent_start
            abs_end = passage_start + sent_end

            sent_entities: list[Entity] = sorted(
                [
                    Entity(
                        start=ent.start - abs_start,
                        end=ent.end - abs_start,
                        text=ent.text,
                        label=ent.label,
                    )
                    for ent in doc_entities
                    if ent.start >= abs_start and ent.end <= abs_end
                ],
                key=lambda e: e.start,
            )
            examples.append(
                NERExample(
                    id=f"{doc_id}_{abs_start}",
                    text=sent_text,
                    entities=sent_entities,
                )
            )

    return examples


def _load_blurb_parquet(record: dict, entity_type: str) -> NERExample:
    """Convert a pre-tokenized BLURB parquet record to a NERExample.

    Reconstructs character-level text and entity offsets from token/tag lists.
    Tokens are joined with single spaces, matching BLURB TSV conventions.
    Tags are single-character BIO ('B'/'I'/'O') as produced by the official
    BLURB preprocessor (label[0] only). entity_type is used as the label for
    all detected spans.
    """
    tokens: list[str] = record["tokens"]
    ner_tags: list[str] = record["ner_tags"]

    char_offset = 0
    token_spans: list[tuple[int, int]] = []
    for tok in tokens:
        token_spans.append((char_offset, char_offset + len(tok)))
        char_offset += len(tok) + 1

    text = " ".join(tokens)

    entities: list[Entity] = []
    i = 0
    while i < len(ner_tags):
        tag = ner_tags[i]
        if tag == "B":
            start = token_spans[i][0]
            end = token_spans[i][1]
            j = i + 1
            while j < len(ner_tags) and ner_tags[j] == "I":
                end = token_spans[j][1]
                j += 1
            entities.append(
                Entity(start=start, end=end, text=text[start:end], label=entity_type)
            )
            i = j
        else:
            i += 1

    return NERExample(id="", text=text, entities=entities)


@dataclass(frozen=True)
class _TaskSpec:
    """Static configuration for a single BLURB NER task."""

    hf_dataset: str
    hf_config: str
    label_list: tuple[str, ...]
    entity_type: str
    hf_split_map: dict[str, str]
    loader: Any


_HF_NER_REPO = "tti-kde/blurb-ner"

_TASK_REGISTRY: dict[str, _TaskSpec] = {
    "bc5cdr_chem": _TaskSpec(
        hf_dataset=_HF_NER_REPO,
        hf_config="bc5cdr-chem",
        label_list=("O", "B-Chemical", "I-Chemical"),
        entity_type="Chemical",
        hf_split_map={"train": "train", "validation": "validation", "test": "test"},
        loader=None,
    ),
    "bc5cdr_disease": _TaskSpec(
        hf_dataset=_HF_NER_REPO,
        hf_config="bc5cdr-disease",
        label_list=("O", "B-Disease", "I-Disease"),
        entity_type="Disease",
        hf_split_map={"train": "train", "validation": "validation", "test": "test"},
        loader=None,
    ),
    "ncbi_disease": _TaskSpec(
        hf_dataset=_HF_NER_REPO,
        hf_config="ncbi-disease",
        label_list=("O", "B-Disease", "I-Disease"),
        entity_type="Disease",
        hf_split_map={"train": "train", "validation": "validation", "test": "test"},
        loader=None,
    ),
    "bc2gm": _TaskSpec(
        hf_dataset=_HF_NER_REPO,
        hf_config="bc2gm",
        label_list=("O", "B-Gene", "I-Gene"),
        entity_type="Gene",
        hf_split_map={"train": "train", "validation": "validation", "test": "test"},
        loader=None,
    ),
    "jnlpba": _TaskSpec(
        hf_dataset=_HF_NER_REPO,
        hf_config="jnlpba",
        label_list=("O", "B-entity", "I-entity"),
        entity_type="entity",
        hf_split_map={"train": "train", "validation": "validation", "test": "test"},
        loader=None,
    ),
}


@register("dataset", name="blurb_ner")
class BLURBNERDataset(IEDataset):
    """Unified dataset for BLURB NER benchmark tasks.

    Supports bc5cdr_chem, bc5cdr_disease, ncbi_disease, bc2gm, and jnlpba.
    Each sentence is a NERExample with character-level entity offsets.
    """

    def __init__(
        self,
        task: str,
        split: str = "train",
        name: str = "blurb_ner",
    ) -> None:
        """Load and preprocess the requested BLURB task split."""
        if task not in _TASK_REGISTRY:
            msg = f"Unknown BLURB task {task!r}. Choose from: {list(_TASK_REGISTRY)}"
            raise ValueError(msg)
        self._task = task
        self.name = name
        spec = _TASK_REGISTRY[task]
        hf_split = spec.hf_split_map.get(split, split)

        from ieforge.datasets.local import load_hf_dataset

        token = os.environ.get("HF_TOKEN")
        hf = load_hf_dataset(
            spec.hf_dataset, spec.hf_config, split=hf_split, token=token
        )
        self._examples: list[NERExample] = []

        if spec.loader is None:
            for i, rec in enumerate(hf):
                ex = _load_blurb_parquet(rec, spec.entity_type)
                self._examples.append(
                    NERExample(id=str(i), text=ex.text, entities=ex.entities)
                )
        else:
            sent_tokenizer = _get_sentence_tokenizer()
            for rec in hf:
                self._examples.extend(
                    spec.loader(rec, sent_tokenizer, frozenset({spec.entity_type}))
                )

    def __len__(self) -> int:
        """Return the number of sentence-level examples."""
        return len(self._examples)

    def __getitem__(self, index: int) -> NERExample:
        """Return the NERExample at the given index."""
        return self._examples[index]

    @property
    def label_list(self) -> list[str]:
        """Return the BIO label vocabulary for this task."""
        return list(_TASK_REGISTRY[self._task].label_list)
