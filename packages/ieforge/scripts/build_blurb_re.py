#!/usr/bin/env python
r"""Build and push tti-kde/blurb-re to HuggingFace.

Preprocessing follows the official BLURB data_generation scripts (Gu et al. 2021):
https://microsoft.github.io/BLURB/sample_code/data_generation.tar.gz

ChemProt (bigbio/chemprot chemprot_bigbio_kb):
  - Sentence-split with customized_sent_tokenize: NLTK sent_tokenize then split on
    \t or \n (bigbio joins title+abstract with \n; official raw TSV uses \t)
  - Extract all CHEMICAL x GENE-{N,Y} pairs per sentence
  - Map CPR groups: Upregulator→CPR:3, Downregulator→CPR:4, Agonist→CPR:5,
    Antagonist→CPR:6, Substrate→CPR:9; all others (incl. reverse direction) → "false"
  - Label expansion (ALL splits): unlabeled CHEMICAL-GENE pairs → "false"
  - Dummify (mask method): @{e1_type}$ @{e2_type}$ (e.g. @CHEMICAL$@GENE-N$);
    overlapping entities → @CHEM-GENE$; matches official split_sent/reformat_data

DDI (raw XML from DDICorpus-2013):
  - Parse XML files listed in raw_data/data_generation/indexing/DDI/{split}_files.tsv
    (official BLURB split: train=623, dev=89, test=190 files)
  - Extract all annotated pairs from <pair> elements (ddi="true/false" + type)
  - ddi="false" → "DDI-false"; ddi="true" type=X → "DDI-X" (lowercase, DDI- prefix)
  - Multi-span charOffset (e.g. "107-107;124-144"): use first and last digit positions
  - Dummify (mask method): sort entities by position, replace both with @DRUG$
  - Test: 190 files from test_dataset/ (DDICorpus Test for DDI Extraction task)

GAD:
  - BLURB-compatible split counts: train=4261, validation=535, test=534
  - Preprocess from raw_data/GAD/1/{train.tsv,test.tsv}
  - That raw TSV layout can be supplied by the full official BLURB download workflow
    (`download_BLURB_data.sh` fetching `REdata.zip`); the sample-code archive alone is
    not sufficient
  - Train split is sliced into train/validation (first 4261 rows for train, rest for validation)
  - Test split validates the official header row, then uses the remaining rows as test instances

Usage:
    HF_TOKEN=<token> uv run python scripts/build_blurb_re.py
    # or build locally without pushing (reader layout for IEFORGE_DATA_DIR):
    uv run python scripts/build_blurb_re.py --dataset chemprot \
        --out-dir "$IEFORGE_DATA_DIR/blurb-re"

    DDI_CORPUS_DIR defaults to raw_data/ddi/DDICorpus (relative to repo root).
    Download: https://github.com/isegura/DDICorpus/raw/master/DDICorpus-2013.zip
    Also requires raw_data/data_generation/ from data_generation.tar.gz.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from pathlib import Path

import nltk

# Ensure sentence tokenizer is available
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)


@dataclass
class _RERecord:
    id: str
    text: str
    entities: list[dict] = field(default_factory=list)
    relations: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# ChemProt
# ---------------------------------------------------------------------------

_CHEMPROT_CPR_MAP = {
    "Upregulator": "CPR:3",
    "Downregulator": "CPR:4",
    "Agonist": "CPR:5",
    "Antagonist": "CPR:6",
    "Substrate": "CPR:9",
}


def _chemprot_sent_tokenize(text: str) -> list[str]:
    r"""Mirror official customized_sent_tokenize: NLTK then split on \t/\n.

    bigbio joins title+abstract with \n; official raw TSV uses \t. Both are
    treated as sentence boundaries, matching preprocessor.py exactly.
    """
    sents_raw = nltk.sent_tokenize(text)
    output_sents: list[str] = []
    for sent in sents_raw:
        parts = sent.replace("\n", "\t").split("\t")
        output_sents.extend(p for p in parts if p)
    return output_sents


def _chemprot_split_sent(
    e1_s: int, e1_e: int, e2_s: int, e2_e: int, sent: str
) -> list[str]:
    """Mirror official split_sent: 5 chunks (non-overlapping) or 3 (overlapping)."""
    if e1_e > e2_s:
        entity_s = min(e1_s, e2_s)
        entity_e = max(e1_e, e2_e)
        pos_list = [entity_s, entity_e]
    else:
        pos_list = [e1_s, e1_e, e2_s, e2_e]
    spans = zip([0, *pos_list], [*pos_list, len(sent)])
    return [sent[s:e] for s, e in spans]


def _chemprot_entities(
    text: str,
    *,
    e1_type: str,
    e2_type: str,
    e1_mention: str,
    e2_mention: str,
    record_id: str,
) -> list[dict]:
    """Build ChemProt entity dicts with marker spans into the dummified text.

    Overlapping entities collapse to one ``@CHEM-GENE$`` marker; in that case
    both entities share the single marker span (mentions stay distinct).

    Invariant note: for the overlapping case each entity's ``text`` is its own
    type marker (e.g. ``@CHEMICAL$``) while its ``start``/``end`` span the fused
    ``@CHEM-GENE$`` token, so ``text[start:end] != entity["text"]`` there. This is
    inert for the shipped BLURB RE config (``passthrough`` representation +
    ``full_text`` strategy never slice by span), but a span-slicing
    ``entity_representation`` would mis-align on these records.
    """
    overlapping = "@CHEM-GENE$" in text
    if overlapping:
        span = _marker_span(text, "@CHEM-GENE$", 0)
        _assert_marker_span(text, *span, "@CHEM-GENE$", record_id=record_id)
        e1_span = e2_span = span
    else:
        m1, m2 = f"@{e1_type}$", f"@{e2_type}$"
        e1_span = _marker_span(text, m1, 0)
        _assert_marker_span(text, *e1_span, m1, record_id=record_id)
        e2_span = _marker_span(text, m2, e1_span[1])
        _assert_marker_span(text, *e2_span, m2, record_id=record_id)
    return [
        {
            "text": f"@{e1_type}$",
            "label": e1_type,
            "role": "head",
            "mention": e1_mention,
            "start": e1_span[0],
            "end": e1_span[1],
        },
        {
            "text": f"@{e2_type}$",
            "label": e2_type,
            "role": "tail",
            "mention": e2_mention,
            "start": e2_span[0],
            "end": e2_span[1],
        },
    ]


def _convert_chemprot(split: str) -> list[_RERecord]:
    r"""Convert bigbio/chemprot (KB format) to sentence-level RE records.

    Exactly mirrors official BLURB preprocessor.py reformat_data():
    - customized_sent_tokenize (\t/\n split after NLTK)
    - all CHEMICAL x GENE-{N,Y} pairs, label from both-direction rel lookup
    - dummify with @{type}$ tokens; overlapping entities -> @CHEM-GENE$
    """
    from datasets import load_dataset

    token = os.environ.get("HF_TOKEN")
    ds = load_dataset("bigbio/chemprot", "chemprot_bigbio_kb", split=split, token=token)
    records: list[_RERecord] = []

    for doc_idx, doc in enumerate(ds):
        passage = doc["passages"][0]
        text: str = passage["text"][0]
        offset_base: int = passage["offsets"][0][0]

        # Sentence splitting — mirrors customized_sent_tokenize + char-offset tracking
        sents = _chemprot_sent_tokenize(text)
        pos = 0
        sent_spans: list[tuple[int, int]] = []
        for sent in sents:
            idx = text.find(sent, pos)
            sent_spans.append((idx + offset_base, idx + offset_base + len(sent)))
            pos = idx + len(sent)

        # Relation map: check both directions (official checks (e1,e2) and (e2,e1))
        rel_map: dict[tuple[str, str], str] = {}
        for rel in doc["relations"]:
            cpr = _CHEMPROT_CPR_MAP.get(rel["type"])
            if cpr:
                rel_map[(rel["arg1_id"], rel["arg2_id"])] = cpr

        for sent_idx, (sent_start, sent_end) in enumerate(sent_spans):
            sent_ents = [
                e
                for e in doc["entities"]
                if e["offsets"][0][0] >= sent_start and e["offsets"][0][1] <= sent_end
            ]
            chemicals = [e for e in sent_ents if e["type"] == "CHEMICAL"]
            genes = [e for e in sent_ents if e["type"] in ("GENE-N", "GENE-Y")]
            if not chemicals or not genes:
                continue

            sent_text = text[sent_start - offset_base : sent_end - offset_base]

            for chem in chemicals:
                for gene in genes:
                    # Mirror: first entity = lower offset (e1), second = higher (e2)
                    if chem["offsets"][0][0] < gene["offsets"][0][0]:
                        e1, e2 = chem, gene
                    else:
                        e1, e2 = gene, chem

                    # Label: check both (e1,e2) and (e2,e1) directions
                    label = rel_map.get(
                        (e1["id"], e2["id"]),
                        rel_map.get((e2["id"], e1["id"]), "false"),
                    )

                    # Sentence-relative offsets
                    off = sent_start
                    e1s = e1["offsets"][0][0] - off
                    e1e = e1["offsets"][0][1] - off
                    e2s = e2["offsets"][0][0] - off
                    e2e = e2["offsets"][0][1] - off

                    chunks = _chemprot_split_sent(e1s, e1e, e2s, e2e, sent_text)
                    if len(chunks) == 5:
                        c1, _, c3, _, c5 = chunks
                        dummified = f"{c1}@{e1['type']}${c3}@{e2['type']}${c5}"
                    else:
                        c1, _, c3 = chunks
                        dummified = f"{c1}@CHEM-GENE${c3}"

                    # clean_sent: remove \n \t \r
                    for ch in ("\n", "\t", "\r"):
                        dummified = dummified.replace(ch, " ")

                    records.append(
                        _RERecord(
                            id=f"chemprot_{split}_{doc_idx}_{sent_idx}_{e1['id']}_{e2['id']}",
                            text=dummified,
                            entities=_chemprot_entities(
                                dummified,
                                e1_type=e1["type"],
                                e2_type=e2["type"],
                                e1_mention=e1["text"],
                                e2_mention=e2["text"],
                                record_id=(
                                    f"chemprot_{split}_{doc_idx}_{sent_idx}"
                                    f"_{e1['id']}_{e2['id']}"
                                ),
                            ),
                            relations=[{"head": 0, "tail": 1, "label": label}],
                        )
                    )

    return records


# ---------------------------------------------------------------------------
# DDI (from raw XML, following official BLURB data_generation scripts)
# ---------------------------------------------------------------------------


_EXPECTED_SPLIT_COUNTS = {
    "chemprot": {"train": 18035, "validation": 11268, "test": 15745},
    "ddi": {"train": 25296, "validation": 2496, "test": 5716},
    "gad": {"train": 4261, "validation": 535, "test": 534},
}


def _validate_expected_counts(
    dataset_name: str, records_by_split: dict[str, list[_RERecord]]
) -> None:
    expected = _EXPECTED_SPLIT_COUNTS[dataset_name]
    actual = {split: len(records) for split, records in records_by_split.items()}
    if actual != expected:
        msg = f"{dataset_name} split counts mismatch: expected {expected}, got {actual}"
        raise ValueError(msg)


def _ddi_extract_span(span_str: str) -> tuple[int, int]:
    """Extract (start, end) from DDI charOffset string.

    Follows official BLURB logic: extract all digit sequences, use first as
    start and last as end (inclusive), then add 1 to end.
    Works for both "107-144" and multi-span "107-107;124-144".
    """
    import re as _re

    digits = [int(d) for d in _re.findall(r"\d+", span_str)]
    return digits[0], digits[-1] + 1


def _marker_span(text: str, marker: str, search_from: int) -> tuple[int, int]:
    """Return (start, end) of `marker` in `text`, searching from `search_from`.

    Used to locate a dummified entity token (e.g. ``@CHEMICAL$``) inside the
    masked sentence so the stored span satisfies ``text[start:end] == marker``.
    Raises ``ValueError`` if `marker` is empty or not found at or after
    `search_from`.
    """
    if not marker:
        msg = "marker must be a non-empty string"
        raise ValueError(msg)
    start = text.find(marker, search_from)
    if start < 0:
        msg = f"marker {marker!r} not found in text from offset {search_from}: {text!r}"
        raise ValueError(msg)
    return start, start + len(marker)


def _assert_marker_span(
    text: str, start: int, end: int, marker: str, *, record_id: str
) -> None:
    """Fail loudly if a computed span does not slice out its marker token."""
    if text[start:end] != marker:
        msg = (
            f"marker span invariant violated for {record_id}: "
            f"text[{start}:{end}]={text[start:end]!r} != marker {marker!r}"
        )
        raise ValueError(msg)


def _ddi_dummify(
    text: str, e1_start: int, e1_end: int, e2_start: int, e2_end: int
) -> str:
    """Replace both entity spans with @DRUG$ (mask method, position-sorted).

    Matches official BLURB preprocessor mask method: both entities get the
    same @DRUG$ token. Spans are replaced in reverse position order.
    """
    spans = sorted(
        [(e1_start, e1_end), (e2_start, e2_end)],
        key=lambda x: x[0],
        reverse=True,
    )
    result = text
    for start, end in spans:
        result = result[:start] + "@DRUG$" + result[end:]
    return result


def _ddi_entities(
    text: str,
    *,
    e1_mention: str,
    e2_mention: str,
    record_id: str,
) -> list[dict]:
    """Build DDI entity dicts with @DRUG$ marker spans into the dummified text.

    Two distinct markers get distinct spans (the second search starts after the
    first). When overlapping/multi-span source entities collapse to a single
    ``@DRUG$`` token, both entities share that one span (mentions stay distinct),
    mirroring the ChemProt ``@CHEM-GENE$`` rule.
    """
    span1 = _marker_span(text, "@DRUG$", 0)
    _assert_marker_span(text, *span1, "@DRUG$", record_id=record_id)
    if text.count("@DRUG$") == 1:
        span2 = span1
    else:
        span2 = _marker_span(text, "@DRUG$", span1[1])
        _assert_marker_span(text, *span2, "@DRUG$", record_id=record_id)
    return [
        {
            "text": "@DRUG$",
            "label": "Drug",
            "role": "head",
            "mention": e1_mention,
            "start": span1[0],
            "end": span1[1],
        },
        {
            "text": "@DRUG$",
            "label": "Drug",
            "role": "tail",
            "mention": e2_mention,
            "start": span2[0],
            "end": span2[1],
        },
    ]


def _gad_entities(text: str, *, record_id: str) -> list[dict]:
    """Build GAD entity dicts with marker spans; mention is null (pre-blinded).

    The source sentence already contains ``@GENE$``/``@DISEASE$`` markers, but no
    surface mention is recoverable, so ``mention`` is ``None``.
    """

    def locate(marker: str) -> tuple[int, int]:
        try:
            span = _marker_span(text, marker, 0)
        except ValueError as exc:
            msg = f"{record_id}: {exc}"
            raise ValueError(msg) from exc
        _assert_marker_span(text, *span, marker, record_id=record_id)
        return span

    gene_span = locate("@GENE$")
    disease_span = locate("@DISEASE$")
    return [
        {
            "text": "@GENE$",
            "label": "Gene",
            "role": "head",
            "mention": None,
            "start": gene_span[0],
            "end": gene_span[1],
        },
        {
            "text": "@DISEASE$",
            "label": "Disease",
            "role": "tail",
            "mention": None,
            "start": disease_span[0],
            "end": disease_span[1],
        },
    ]


def _build_gad(
    gad_root: Path,
) -> tuple[list[_RERecord], list[_RERecord], list[_RERecord]]:
    r"""Build BLURB-compatible GAD train/validation/test splits.

    The documented GAD raw format is TSV with a header row and three columns:
    `index\tsentence\tlabel`. Train data is sliced from the first 4261 data
    rows, validation gets the remainder, and test keeps the header out of the
    returned instances.
    """
    input_dir = gad_root / "1"
    train_path = input_dir / "train.tsv"
    test_path = input_dir / "test.tsv"
    if not train_path.exists() or not test_path.exists():
        msg = (
            f"GAD raw files not found under {input_dir}. "
            "Expected train.tsv and test.tsv from the BLURB-compatible layout."
        )
        raise FileNotFoundError(msg)

    def parse_tsv(path: Path, *, has_header: bool) -> list[tuple[str, str, str]]:
        rows: list[tuple[str, str, str]] = []
        for line_no, raw_line in enumerate(path.read_text().splitlines(), start=1):
            if has_header and line_no == 1:
                columns = raw_line.split("\t")
                if len(columns) != 3 or columns != ["index", "sentence", "label"]:
                    msg = (
                        f"Unexpected GAD test.tsv header in {path}: "
                        f"expected 'index\tsentence\tlabel', got {raw_line!r}"
                    )
                    raise ValueError(msg)
                continue

            columns = raw_line.split("\t")
            if len(columns) != 3:
                msg = (
                    f"Unexpected GAD row format in {path} line {line_no}: "
                    "expected three tab-separated columns"
                )
                raise ValueError(msg)
            rows.append((columns[0].strip(), columns[1], columns[2].strip()))
        return rows

    def to_records(rows: list[tuple[str, str, str]], split: str) -> list[_RERecord]:
        records: list[_RERecord] = []
        for index, (row_id, sentence, label) in enumerate(rows):
            record_id = row_id or str(index)
            records.append(
                _RERecord(
                    id=f"gad_{split}_{record_id}",
                    text=sentence,
                    entities=_gad_entities(
                        sentence, record_id=f"gad_{split}_{record_id}"
                    ),
                    relations=[{"head": 0, "tail": 1, "label": label}],
                )
            )
        return records

    train_rows = parse_tsv(train_path, has_header=True)
    test_rows = parse_tsv(test_path, has_header=True)

    return (
        to_records(train_rows[:4261], "train"),
        to_records(train_rows[4261:], "validation"),
        to_records(test_rows, "test"),
    )


def _parse_ddi_xml(path: Path) -> list[_RERecord]:
    """Parse a single DDI XML file and return one record per annotated pair."""
    import xml.etree.ElementTree as ET

    records: list[_RERecord] = []
    try:
        tree = ET.parse(path)  # noqa: S314
    except ET.ParseError:
        return records

    root = tree.getroot()
    for sent in root.iter("sentence"):
        sent_id = sent.get("id", "")
        text = sent.get("text", "")

        if sent.find("pair") is None:
            continue

        # Build entity map: id → (start, end, mention)
        entities: dict[str, tuple[int, int, str]] = {}
        for ent in sent.findall("entity"):
            eid = ent.get("id", "")
            offset_str = ent.get("charOffset", "")
            if not offset_str:
                continue
            try:
                span = _ddi_extract_span(offset_str)
            except (ValueError, IndexError):
                continue
            entities[eid] = (span[0], span[1], ent.get("text", ""))

        for pair in sent.findall("pair"):
            e1_id = pair.get("e1", "")
            e2_id = pair.get("e2", "")
            ddi_flag = pair.get("ddi", "false")
            ddi_type = pair.get("type", "")
            pair_id = pair.get("id", "")

            if e1_id not in entities or e2_id not in entities:
                continue

            e1_start, e1_end, e1_mention = entities[e1_id]
            e2_start, e2_end, e2_mention = entities[e2_id]

            # Ensure e1 is the first entity (position order), matching official script
            if e1_start > e2_start:
                e1_start, e1_end, e1_mention, e2_start, e2_end, e2_mention = (
                    e2_start,
                    e2_end,
                    e2_mention,
                    e1_start,
                    e1_end,
                    e1_mention,
                )

            # Label: "DDI-false" or "DDI-<type>" (lowercase, DDI- prefix for all)
            if ddi_flag == "true" and ddi_type:
                label = f"DDI-{ddi_type.lower()}"
            else:
                label = "DDI-false"

            dummified = _ddi_dummify(text, e1_start, e1_end, e2_start, e2_end)

            records.append(
                _RERecord(
                    id=f"{sent_id}_{pair_id}",
                    text=dummified,
                    entities=_ddi_entities(
                        dummified,
                        e1_mention=e1_mention,
                        e2_mention=e2_mention,
                        record_id=f"{sent_id}_{pair_id}",
                    ),
                    relations=[{"head": 0, "tail": 1, "label": label}],
                )
            )
    return records


def _load_files_from_index(
    index_file: Path,
    corpus_dir: Path,
    train_dir_name: str = "Train",
    test_dir_name: str = "Test/Test for DDI Extraction task",
) -> list[_RERecord]:
    """Load DDI records for files listed in an official BLURB index file."""
    records: list[_RERecord] = []
    for raw_line in index_file.read_text().splitlines():
        entry = raw_line.strip()
        if not entry:
            continue
        # e.g. "train_dataset/DrugBank/Foo.xml" or "test_dataset/DrugBank/Bar.xml"
        parts = entry.split("/", 1)
        dataset_type = parts[0]  # "train_dataset" or "test_dataset"
        rel_path = parts[1]  # "DrugBank/Foo.xml"

        if dataset_type == "train_dataset":
            full_path = corpus_dir / train_dir_name / rel_path
        else:
            full_path = corpus_dir / test_dir_name / rel_path

        records.extend(_parse_ddi_xml(full_path))
    return records


def _build_ddi(
    corpus_dir: Path | None = None,
    index_dir: Path | None = None,
) -> tuple[list[_RERecord], list[_RERecord], list[_RERecord]]:
    """Build DDI train/validation/test splits using official BLURB index files.

    Uses raw_data/data_generation/indexing/DDI/{split}_files.tsv to reproduce
    the exact same file split as the official BLURB data_generation scripts.
    """
    repo_root = Path(__file__).parent.parent
    if corpus_dir is None:
        corpus_dir = repo_root / "raw_data" / "ddi" / "DDICorpus"
    if index_dir is None:
        index_dir = repo_root / "raw_data" / "data_generation" / "indexing" / "DDI"

    if not corpus_dir.exists():
        msg = (
            f"DDI corpus not found at {corpus_dir}. "
            "Download DDICorpus-2013.zip from https://github.com/isegura/DDICorpus "
            "and extract to raw_data/ddi/"
        )
        raise FileNotFoundError(msg)
    if not index_dir.exists():
        msg = (
            f"BLURB index not found at {index_dir}. "
            "Download data_generation.tar.gz from "
            "https://microsoft.github.io/BLURB/sample_code/data_generation.tar.gz "
            "and extract to raw_data/"
        )
        raise FileNotFoundError(msg)

    train_records = _load_files_from_index(index_dir / "train_files.tsv", corpus_dir)
    dev_records = _load_files_from_index(index_dir / "dev_files.tsv", corpus_dir)
    test_records = _load_files_from_index(index_dir / "test_files.tsv", corpus_dir)

    return train_records, dev_records, test_records


# ---------------------------------------------------------------------------
# Push to HuggingFace
# ---------------------------------------------------------------------------


def _push_config(
    records_by_split: dict[str, list[_RERecord]],
    config_name: str,
    repo_id: str,
    token: str | None,
    out_dir: Path | None = None,
) -> None:
    from datasets import Dataset, DatasetDict

    def to_hf_dict(records: list[_RERecord]) -> dict[str, list]:
        return {
            "id": [r.id for r in records],
            "text": [r.text for r in records],
            "entities": [r.entities for r in records],
            "relations": [r.relations for r in records],
        }

    dd = DatasetDict(
        {
            split: Dataset.from_dict(to_hf_dict(recs))
            for split, recs in records_by_split.items()
        }
    )
    for split, recs in records_by_split.items():
        print(f"  {split}: {len(recs)} records")
    if out_dir is not None:
        from ieforge.datasets.local import save_parquet_layout

        save_parquet_layout(dd, out_dir, config_name)
        print(f"Wrote {config_name} to {out_dir}")
        return
    dd.push_to_hub(repo_id, config_name=config_name, token=token)
    print(f"Pushed {config_name} to {repo_id}")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=("chemprot", "ddi", "gad", "all"),
        default="all",
    )
    parser.add_argument("--hf-repo", default="tti-kde/blurb-re")
    parser.add_argument("--gad-root", type=Path, default=Path("raw_data") / "GAD")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Write <out-dir>/<config>/<split>-00000-of-00001.parquet locally instead of pushing to the Hub (use $IEFORGE_DATA_DIR/<repo-name>)",
    )
    return parser.parse_args(argv)


def _build_selected_datasets(
    args: argparse.Namespace,
) -> dict[str, dict[str, list[_RERecord]]]:
    selected: dict[str, dict[str, list[_RERecord]]] = {}
    if args.dataset in {"chemprot", "all"}:
        selected["chemprot"] = {
            "train": _convert_chemprot("train"),
            "validation": _convert_chemprot("validation"),
            "test": _convert_chemprot("test"),
        }
    if args.dataset in {"ddi", "all"}:
        ddi_train, ddi_val, ddi_test = _build_ddi()
        selected["ddi"] = {
            "train": ddi_train,
            "validation": ddi_val,
            "test": ddi_test,
        }
    if args.dataset in {"gad", "all"}:
        gad_train, gad_val, gad_test = _build_gad(args.gad_root)
        selected["gad"] = {
            "train": gad_train,
            "validation": gad_val,
            "test": gad_test,
        }
    return selected


def main(argv: list[str] | None = None) -> None:
    """Build and push BLURB RE datasets to HuggingFace."""
    args = _parse_args(argv)
    token = os.environ.get("HF_TOKEN")
    if not token and not args.dry_run and args.out_dir is None:
        msg = "HF_TOKEN not set"
        raise RuntimeError(msg)

    datasets = _build_selected_datasets(args)
    for name, records_by_split in datasets.items():
        if records_by_split and any(records_by_split.values()):
            _validate_expected_counts(name, records_by_split)
        if args.dry_run:
            counts = {
                split: len(records) for split, records in records_by_split.items()
            }
            print(f"Verified {name}: {counts}")
            continue
        if args.out_dir is not None:
            _push_config(records_by_split, name, args.hf_repo, None, args.out_dir)
            continue
        assert token is not None
        _push_config(records_by_split, name, args.hf_repo, token)

    if args.dry_run:
        print("Done.")


if __name__ == "__main__":
    main()
