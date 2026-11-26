from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Iterator

from primehkg.ids import normalize_node_id


@dataclass(frozen=True, slots=True)
class PrimeKGLoadError(ValueError):
    """Raised when a PrimeKG CSV row cannot be parsed/validated in strict mode."""

    path: str
    row_number: int
    message: str

    def __str__(self) -> str:  # pragma: no cover (tiny)
        """Format the error with path and row context."""
        return f"PrimeKG load error at {self.path}:{self.row_number}: {self.message}"


_ColSpecName = Literal["primekg_kg", "srto", "spo"]

_MISSING_HEADER_MSG = "PrimeKG kg.csv is missing a header row."
_INVALID_HEADER_MSG = (
    "PrimeKG kg.csv header must contain either "
    "{x_id, relation, y_id} or {s, r, t} or {subject, predicate, object}."
)
_EMPTY_SUBJECT_MSG = "empty subject after normalization"
_EMPTY_RELATION_MSG = "empty relation"
_EMPTY_OBJECT_MSG = "empty object after normalization"


def _detect_columns(fieldnames: list[str] | None) -> tuple[_ColSpecName, str, str, str]:
    if not fieldnames:
        msg = _MISSING_HEADER_MSG
        raise ValueError(msg)

    stripped: list[str] = []
    for c in fieldnames:
        if c is None:
            continue
        stripped.append(c.strip())

    if len(set(stripped)) != len(stripped):
        counts = Counter(stripped)
        dupes = sorted([k for k, c in counts.items() if c > 1])
        msg = (
            "PrimeKG kg.csv header contains duplicate column names after stripping whitespace: "
            + ", ".join(dupes)
        )
        raise ValueError(msg)

    cols = {c.strip(): c for c in fieldnames if c is not None}

    # PrimeKG kg.csv commonly uses x_id / relation / y_id.
    if {"x_id", "relation", "y_id"} <= set(cols):
        return ("primekg_kg", cols["x_id"], cols["relation"], cols["y_id"])

    # Simple triplet CSV.
    if {"s", "r", "t"} <= set(cols):
        return ("srto", cols["s"], cols["r"], cols["t"])

    # Subject-predicate-object.
    if {"subject", "predicate", "object"} <= set(cols):
        return ("spo", cols["subject"], cols["predicate"], cols["object"])

    msg = _INVALID_HEADER_MSG
    raise ValueError(msg)


def _primekg_typed_node_id(raw_id: str, raw_type: str) -> str:
    """Build a canonical node_id from PrimeKG's (id, type) columns.

    PrimeKG `kg.csv` from Dataverse commonly uses numeric ids plus `x_type`/`y_type`
    values like `gene/protein`, `drug`, `disease`, `phenotype`.
    """
    rid = (raw_id or "").strip()
    rty = (raw_type or "").strip().lower()
    if not rid:
        return ""
    if not rty:
        return rid
    if rty in {"gene/protein", "protein", "gene"}:
        return f"primekg:protein:{rid}"
    if rty == "drug":
        return f"primekg:drug:{rid}"
    if rty == "disease":
        return f"primekg:disease:{rid}"
    if rty == "phenotype":
        return f"primekg:phenotype:{rid}"
    return f"primekg:{rty}:{rid}"


def _strip_header_map(fieldnames: list[str] | None) -> dict[str, str]:
    if not fieldnames:
        return {}
    return {c.strip(): c for c in fieldnames if c is not None}


def _endpoint_name_fields(
    spec: _ColSpecName,
    cols: dict[str, str],
    row: dict[str, str | None],
) -> dict[str, str]:
    """Collect optional endpoint display strings from wide PrimeKG-style rows."""
    extra: dict[str, str] = {}

    def _add_side(
        *,
        name_key: str,
        src_key: str | None,
        side: Literal["s", "t"],
        evidence: str,
    ) -> None:
        if name_key not in cols:
            return
        name = (row.get(cols[name_key]) or "").strip()
        if not name:
            return
        if side == "s":
            extra["name_s"] = name
            extra["evidence_s"] = evidence
            out_src = "src_s"
        else:
            extra["name_t"] = name
            extra["evidence_t"] = evidence
            out_src = "src_t"
        if src_key and src_key in cols:
            src = (row.get(cols[src_key]) or "").strip()
            if src:
                extra[out_src] = src

    if spec == "primekg_kg":
        _add_side(
            name_key="x_name",
            src_key="x_source",
            side="s",
            evidence="kg_csv_x_name",
        )
        _add_side(
            name_key="y_name",
            src_key="y_source",
            side="t",
            evidence="kg_csv_y_name",
        )
    elif spec == "srto":
        if "s_name" in cols:
            name = (row.get(cols["s_name"]) or "").strip()
            if name:
                extra["name_s"] = name
                extra["evidence_s"] = "kg_csv_s_name"
        if "t_name" in cols:
            name = (row.get(cols["t_name"]) or "").strip()
            if name:
                extra["name_t"] = name
                extra["evidence_t"] = "kg_csv_t_name"
    elif spec == "spo":
        if "subject_name" in cols:
            name = (row.get(cols["subject_name"]) or "").strip()
            if name:
                extra["name_s"] = name
                extra["evidence_s"] = "kg_csv_subject_name"
        if "object_name" in cols:
            name = (row.get(cols["object_name"]) or "").strip()
            if name:
                extra["name_t"] = name
                extra["evidence_t"] = "kg_csv_object_name"

    return extra


def iter_primekg_rows(
    path: str | Path,
    *,
    strict: bool = True,
    encoding: str = "utf-8",
) -> Iterator[dict[str, str]]:
    """Yield one dict per valid CSV row: ``s``, ``r``, ``t`` plus optional endpoint labels.

    Optional keys (when present in the header and non-empty in the row):

    - ``name_s`` / ``name_t``: endpoint display strings
    - ``src_s`` / ``src_t``: row-level source attribution (e.g. ``x_source`` / ``y_source``)
    - ``evidence_s`` / ``evidence_t``: provenance tokens for how the name was derived

    See ``collect_primekg_endpoint_name_updates`` for how this is used in v1 ingestion.
    """
    p = Path(path)

    def _fail(msg: str) -> None:
        raise ValueError(msg)

    with p.open("r", encoding=encoding, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames) if reader.fieldnames else None
        spec, s_col, r_col, t_col = _detect_columns(fieldnames)
        cols = _strip_header_map(fieldnames)

        for row_number, row in enumerate(reader, start=2):
            try:
                raw_s = (row.get(s_col) or "").strip()
                raw_r = (row.get(r_col) or "").strip()
                raw_t = (row.get(t_col) or "").strip()

                if spec == "primekg_kg":
                    x_type_col = cols.get("x_type")
                    y_type_col = cols.get("y_type")
                    if x_type_col:
                        s = _primekg_typed_node_id(raw_s, row.get(x_type_col) or "")
                    else:
                        s = normalize_node_id(raw_s)
                    if y_type_col:
                        t = _primekg_typed_node_id(raw_t, row.get(y_type_col) or "")
                    else:
                        t = normalize_node_id(raw_t)
                else:
                    s = normalize_node_id(raw_s)
                    t = normalize_node_id(raw_t)
                r = raw_r

                if not s:
                    _fail(_EMPTY_SUBJECT_MSG)
                if not r:
                    _fail(_EMPTY_RELATION_MSG)
                if not t:
                    _fail(_EMPTY_OBJECT_MSG)

                rec: dict[str, str] = {"s": s, "r": r, "t": t}
                rec.update(_endpoint_name_fields(spec, cols, row))
                yield rec
            except (ValueError, TypeError, KeyError, csv.Error) as e:
                if strict:
                    raise PrimeKGLoadError(
                        path=str(p),
                        row_number=row_number,
                        message=str(e) or e.__class__.__name__,
                    ) from e
                continue


def load_primekg_triplets(
    path: str | Path,
    *,
    strict: bool = True,
    encoding: str = "utf-8",
) -> list[dict[str, str]]:
    """Load PrimeKG `kg.csv` as a list of (s, r, t) dict triplets.

    - Deterministic ordering: preserves CSV read order.
    - Normalization: `s` and `t` are normalized via `primehkg.ids.normalize_node_id`.
    - Robustness: by default (`strict=True`), raises `PrimeKGLoadError` on bad rows.
    """
    return [
        {"s": rec["s"], "r": rec["r"], "t": rec["t"]}
        for rec in iter_primekg_rows(path, strict=strict, encoding=encoding)
    ]
