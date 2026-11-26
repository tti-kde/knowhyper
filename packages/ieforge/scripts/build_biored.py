# ruff: noqa: D101,D102,TRY003,EM101,EM102,PERF401

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict

_ENTITY_TYPE_ALIASES = {"SequenceVariant": "GeneOrGeneProduct"}
_SOURCE_TARGET_TYPE_PAIRS = (
    ("ChemicalEntity", "ChemicalEntity"),
    ("ChemicalEntity", "DiseaseOrPhenotypicFeature"),
    ("ChemicalEntity", "GeneOrGeneProduct"),
    ("DiseaseOrPhenotypicFeature", "GeneOrGeneProduct"),
    ("GeneOrGeneProduct", "GeneOrGeneProduct"),
)
_ALLOWED_TYPE_PAIRS = frozenset(_SOURCE_TARGET_TYPE_PAIRS)


@dataclass(frozen=True)
class PubTatorAnnotation:
    pmid: str
    start: int
    end: int
    text: str
    entity_type: str
    identifier: str


@dataclass(frozen=True)
class PubTatorDocument:
    pmid: str
    title: str
    abstract: str
    annotations: tuple[PubTatorAnnotation, ...]
    relations: dict[tuple[str, str], tuple[str, str | None, str]]

    @property
    def text(self) -> str:
        return f"{self.title} {self.abstract}".strip()


def _normalize_entity_type(entity_type: str) -> str:
    base_type = re.sub(r"\s*\(.*?\)\s*$", "", entity_type).strip()
    return _ENTITY_TYPE_ALIASES.get(base_type, base_type)


def _split_ids(identifier: str) -> list[str]:
    return [part.strip() for part in re.split(r"[,;]", identifier) if part.strip()]


def _read_pubtator_documents(path: Path) -> list[PubTatorDocument]:
    documents: list[PubTatorDocument] = []
    pmid = ""
    title = ""
    abstract = ""
    annotations: list[PubTatorAnnotation] = []
    relations: dict[tuple[str, str], tuple[str, str | None, str]] = {}

    def flush() -> None:
        nonlocal pmid, title, abstract, annotations, relations
        if not pmid:
            return
        documents.append(
            PubTatorDocument(
                pmid=pmid,
                title=title,
                abstract=abstract,
                annotations=tuple(annotations),
                relations=dict(relations),
            )
        )
        pmid = ""
        title = ""
        abstract = ""
        annotations = []
        relations = {}

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip("\n")
        if not line:
            flush()
            continue

        pipe_parts = line.split("|", 2)
        if len(pipe_parts) == 3 and pipe_parts[1] in {"t", "a"}:
            line_pmid, section, text = pipe_parts
            if pmid and line_pmid != pmid:
                raise ValueError(
                    f"text PMID {line_pmid!r} does not match current document {pmid!r}"
                )
            pmid = line_pmid
            if section == "t":
                title = text
            else:
                abstract = text
            continue

        fields = line.split("	")
        if len(fields) == 6:
            ann_pmid, start, end, text, entity_type, identifier = fields
            if pmid and ann_pmid != pmid:
                raise ValueError(
                    f"annotation PMID {ann_pmid!r} does not match current document {pmid!r}"
                )
            if not pmid:
                pmid = ann_pmid
            for split_id in _split_ids(identifier):
                annotations.append(
                    PubTatorAnnotation(
                        pmid=ann_pmid,
                        start=int(start),
                        end=int(end),
                        text=text,
                        entity_type=_normalize_entity_type(entity_type),
                        identifier=split_id,
                    )
                )
            continue

        if len(fields) == 4 and fields[3].startswith("Subject:"):
            subj_pmid, raw_id1, raw_id2, subject_field = fields
            if pmid and subj_pmid != pmid:
                raise ValueError(
                    f"relation PMID {subj_pmid!r} does not match current document {pmid!r}"
                )
            if not pmid:
                pmid = subj_pmid
            subject_id = subject_field.split(":", 1)[1].strip() or "None"
            for id1 in _split_ids(raw_id1):
                for id2 in _split_ids(raw_id2):
                    key = (id1, id2) if (id1, id2) in relations else (id2, id1)
                    if key in relations:
                        relation_type, novelty, _ = relations[key]
                        relations[key] = (relation_type, novelty, subject_id)
            continue

        if len(fields) in {4, 5}:
            rel_pmid, relation_type, raw_id1, raw_id2 = fields[:4]
            if pmid and rel_pmid != pmid:
                raise ValueError(
                    f"relation PMID {rel_pmid!r} does not match current document {pmid!r}"
                )
            if not pmid:
                pmid = rel_pmid
            novelty = fields[4] if len(fields) == 5 else None
            for id1 in _split_ids(raw_id1):
                for id2 in _split_ids(raw_id2):
                    relations[(id1, id2)] = (relation_type, novelty, "None")
            continue

        raise ValueError(f"unrecognized PubTator line: {line!r}")

    flush()
    return documents


def _entity_ids_by_first_mention(
    document: PubTatorDocument,
) -> list[tuple[str, str]]:
    seen: set[tuple[str, str]] = set()
    entities: list[tuple[str, str]] = []
    for annotation in sorted(
        document.annotations, key=lambda item: (item.start, item.end)
    ):
        entity = (annotation.identifier, annotation.entity_type)
        if entity in seen:
            continue
        seen.add(entity)
        entities.append(entity)
    return entities


def _allowed_pair(type1: str, type2: str) -> bool:
    return (type1, type2) in _ALLOWED_TYPE_PAIRS or (
        type2,
        type1,
    ) in _ALLOWED_TYPE_PAIRS


def _pair_role_types(
    id1: str,
    type1: str,
    id2: str,
    type2: str,
) -> tuple[str, str, str, str]:
    for source_type, target_type in _SOURCE_TARGET_TYPE_PAIRS:
        if type1 == source_type and type2 == target_type:
            return id1, type1, id2, type2
        if type2 == source_type and type1 == target_type:
            return id2, type2, id1, type1
    raise ValueError(f"disallowed BioRED type pair: {(type1, type2)!r}")


def _marker_for(entity_type: str, role: str, text: str) -> str:
    suffix = "Src" if role == "src" else "Tgt"
    return f"@{entity_type}{suffix}$ {text} @/{entity_type}{suffix}$"


def _marker_text(
    document: PubTatorDocument,
    id1: str,
    type1: str,
    id2: str,
    type2: str,
) -> str:
    replacements_by_span: dict[tuple[int, int], str] = {}
    for annotation in document.annotations:
        span = (annotation.start, annotation.end)
        if annotation.identifier == id1:
            replacements_by_span[span] = _marker_for(type1, "src", annotation.text)
        elif annotation.identifier == id2 and span not in replacements_by_span:
            replacements_by_span[span] = _marker_for(type2, "tgt", annotation.text)

    if not replacements_by_span:
        raise ValueError(
            f"no mentions found for pair {(id1, id2)!r} in {document.pmid}"
        )

    text = document.text
    for (start, end), replacement in sorted(
        replacements_by_span.items(), key=lambda item: item[0], reverse=True
    ):
        text = text[:start] + replacement + text[end:]
    return re.sub(r"\s+", " ", text).strip()


def _sentence_spans(text: str) -> list[tuple[int, int]]:
    spans = [match.span() for match in re.finditer(r"[^.!?]+[.!?]?", text)]
    return [span for span in spans if text[span[0] : span[1]].strip()]


def _same_sentence_metadata(
    document: PubTatorDocument,
    id1: str,
    id2: str,
) -> tuple[bool, int]:
    id1_sentences: set[int] = set()
    id2_sentences: set[int] = set()
    for index, (sentence_start, sentence_end) in enumerate(
        _sentence_spans(document.text)
    ):
        for annotation in document.annotations:
            if sentence_start <= annotation.start and annotation.end <= sentence_end:
                if annotation.identifier == id1:
                    id1_sentences.add(index)
                if annotation.identifier == id2:
                    id2_sentences.add(index)

    if not id1_sentences or not id2_sentences:
        return False, 100
    min_window = min(
        abs(left - right) for left in id1_sentences for right in id2_sentences
    )
    return min_window == 0, min_window


def _relation_for(
    document: PubTatorDocument,
    id1: str,
    id2: str,
) -> tuple[tuple[str, str] | None, str, str | None, str]:
    if (id1, id2) in document.relations:
        label, novelty, subject_id = document.relations[(id1, id2)]
        return (id1, id2), label, novelty, subject_id
    if (id2, id1) in document.relations:
        label, novelty, subject_id = document.relations[(id2, id1)]
        return (id2, id1), label, novelty, subject_id
    return None, "None", None, "None"


def _mentions_by_identifier(
    document: PubTatorDocument,
) -> dict[str, list[str]]:
    mentions: dict[str, list[str]] = {}
    for annotation in document.annotations:
        texts = mentions.setdefault(annotation.identifier, [])
        if annotation.text not in texts:
            texts.append(annotation.text)
    return mentions


def _build_pubtator_records(documents: list[PubTatorDocument]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for document in documents:
        id_mentions = _mentions_by_identifier(document)
        entities = _entity_ids_by_first_mention(document)
        for index, (left_id, left_type) in enumerate(entities[:-1]):
            for right_id, right_type in entities[index + 1 :]:
                if not _allowed_pair(left_type, right_type):
                    continue
                _, relation_label, novelty_label, subject_id = _relation_for(
                    document, left_id, right_id
                )
                id1, id1_type, id2, id2_type = _pair_role_types(
                    left_id,
                    left_type,
                    right_id,
                    right_type,
                )
                if id1 == "-" or id2 == "-":
                    continue
                is_in_same_sent, min_sents_window = _same_sentence_metadata(
                    document, id1, id2
                )
                records.append(
                    {
                        "id": f"{document.pmid}#{id1}-{id2}",
                        "pmid": document.pmid,
                        "id1": id1,
                        "id2": id2,
                        "id1_type": id1_type,
                        "id2_type": id2_type,
                        "id1_mentions": id_mentions.get(id1, []),
                        "id2_mentions": id_mentions.get(id2, []),
                        "is_in_same_sent": is_in_same_sent,
                        "min_sents_window": min_sents_window,
                        "text": _marker_text(document, id1, id1_type, id2, id2_type),
                        "relation_label": relation_label,
                        "novelty_label": novelty_label,
                        "subject_id": subject_id,
                    }
                )
    return records


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build BioRED pair-level records")
    parser.add_argument("--biored-zip", type=Path, required=True)
    parser.add_argument("--hf-repo", default="tti-kde/biored")
    parser.add_argument("--config-name", default="biored")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Write <out-dir>/<config>/<split>-00000-of-00001.parquet locally instead of pushing to the Hub (use $IEFORGE_DATA_DIR/<repo-name>)",
    )
    # Zip member paths for each split. Defaults match the standard BioRED zip
    # layout; override to read the BioREDirect directionality-annotated
    # PubTator files (which carry the ``Subject:`` lines), e.g.
    # ``--train-member bioredirect/bioredirect_train.pubtator``.
    parser.add_argument("--train-member", default="BioRED/Train.PubTator")
    parser.add_argument("--dev-member", default="BioRED/Dev.PubTator")
    parser.add_argument("--test-member", default="BioRED/Test.PubTator")
    return parser.parse_args(argv)


def _read_split_from_zip(archive_path: Path, member: str) -> list[PubTatorDocument]:
    with zipfile.ZipFile(archive_path) as archive:
        payload = archive.read(member)
    with tempfile.NamedTemporaryFile(suffix=".PubTator", delete=False) as tmp_file:
        tmp_path = Path(tmp_file.name)
        tmp_file.write(payload)
    try:
        return _read_pubtator_documents(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)


def _build_records_from_args(
    args: argparse.Namespace,
) -> dict[str, list[dict[str, Any]]]:
    return {
        "train": _build_pubtator_records(
            _read_split_from_zip(args.biored_zip, args.train_member)
        ),
        "validation": _build_pubtator_records(
            _read_split_from_zip(args.biored_zip, args.dev_member)
        ),
        "test": _build_pubtator_records(
            _read_split_from_zip(args.biored_zip, args.test_member)
        ),
    }


def _count_split(records: list[dict[str, Any]]) -> dict[str, int]:
    positive_pairs = sum(1 for record in records if record["relation_label"] != "None")
    return {"pairs": len(records), "positive_pairs": positive_pairs}


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    split_records = _build_records_from_args(args)
    counts = {split: _count_split(records) for split, records in split_records.items()}
    print(json.dumps(counts, indent=2, sort_keys=True))
    if args.dry_run:
        return

    dataset = DatasetDict(
        {split: Dataset.from_list(records) for split, records in split_records.items()}
    )
    if args.out_dir is not None:
        from ieforge.datasets.local import save_parquet_layout

        save_parquet_layout(dataset, args.out_dir, args.config_name)
        return

    token = os.environ.get("HF_TOKEN")
    if token is None:
        raise SystemExit("HF_TOKEN is required for publication")
    dataset.push_to_hub(args.hf_repo, config_name=args.config_name, token=token)


if __name__ == "__main__":
    main()
