from __future__ import annotations

import csv
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from primehkg.io.primekg import _primekg_typed_node_id
from primehkg.nodes.feature_refs import text_pointer

if TYPE_CHECKING:
    from collections.abc import Sequence


# High-value drug text modalities that get a feature_refs text pointer.
# Names MUST match the features-side modality names in
# primehkg.features.extract_drug_tab.MODALITY_COLUMNS so the entity-view join
# (entity_node_id, source_modality) resolves into node_features_embedding.
DRUG_TEXT_MODALITIES: tuple[str, ...] = (
    "description",
    "indication",
    "mechanism_of_action",
    "pharmacodynamics",
)


def build_node_index_to_node_id(
    path: str | Path, *, strict: bool = True
) -> dict[int, str]:
    """Build a mapping from PrimeKG `node_index` (x_index/y_index) to canonical `node_id`.

    This supports the official Dataverse `kg.csv` format (numeric ids + x_type/y_type).
    """
    p = Path(path)

    def _fail(msg: str) -> None:
        raise ValueError(msg)

    with p.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            msg = "PrimeKG kg.csv is missing a header row."
            _fail(msg)

        fieldnames_obj = reader.fieldnames
        if fieldnames_obj is None:
            msg = "PrimeKG kg.csv is missing a header row."
            _fail(msg)
        fieldnames_seq = cast("Sequence[str]", fieldnames_obj)
        fieldnames = list(fieldnames_seq)
        cols = {c.strip(): c for c in fieldnames if c is not None}
        required = {
            "x_index",
            "y_index",
            "x_id",
            "y_id",
            "x_type",
            "y_type",
        }
        missing = required - set(cols)
        if missing:
            msg = f"PrimeKG kg.csv missing required columns for feature join: {sorted(missing)}"
            _fail(msg)

        out: dict[int, str] = {}
        collisions: set[int] = set()
        for row_number, row in enumerate(reader, start=2):
            try:
                for side in ("x", "y"):
                    idx_raw = (row.get(cols[f"{side}_index"]) or "").strip()
                    if not idx_raw:
                        continue
                    idx = int(idx_raw)
                    if idx in collisions:
                        continue
                    nid = _primekg_typed_node_id(
                        (row.get(cols[f"{side}_id"]) or "").strip(),
                        (row.get(cols[f"{side}_type"]) or "").strip(),
                    )
                    if not nid:
                        continue
                    prev = out.get(idx)
                    if prev is None:
                        out[idx] = nid
                    elif prev != nid:
                        msg = (
                            f"node_index collision at row {row_number}: {idx} maps to "
                            f"{prev!r} and {nid!r}"
                        )
                        if strict:
                            _fail(msg)
                        # In non-strict mode, drop this join key entirely to avoid incorrect joins.
                        out.pop(idx, None)
                        collisions.add(idx)
                        continue
            except (ValueError, TypeError) as exc:
                if strict:
                    msg = (
                        f"failed to read node_index mapping at row {row_number}: {exc}"
                    )
                    raise ValueError(msg) from exc
                continue

        return out


def read_primekg_disease_features_tab(
    path: str | Path, *, index_to_node_id: dict[int, str], strict: bool = True
) -> dict[str, dict[str, Any]]:
    """Read `disease_features.tab` and convert to `node_updates` keyed by node_id."""
    p = Path(path)
    with p.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if not reader.fieldnames:
            msg = "PrimeKG disease_features.tab is missing a header row."
            raise ValueError(msg)
        cols = {c.strip(): c for c in reader.fieldnames if c is not None}
        required = {"node_index", "mondo_id", "mondo_name"}
        missing = required - set(cols)
        if missing:
            msg = f"PrimeKG disease_features.tab missing required columns: {sorted(missing)}"
            raise ValueError(msg)

        out: dict[str, dict[str, Any]] = {}
        for row_number, row in enumerate(reader, start=2):
            idx_raw = (row.get(cols["node_index"]) or "").strip()
            if not idx_raw:
                continue
            try:
                idx = int(idx_raw)
            except ValueError:
                if strict:
                    msg = f"invalid node_index at row {row_number}: {idx_raw!r}"
                    raise ValueError(msg) from None
                continue
            node_id = index_to_node_id.get(idx)
            if not node_id:
                if strict:
                    msg = f"node_index not found in kg.csv join map at row {row_number}: {idx}"
                    raise ValueError(msg)
                continue

            mondo_id = (row.get(cols["mondo_id"]) or "").strip()
            mondo_name = (row.get(cols["mondo_name"]) or "").strip().strip('"')
            if not mondo_id and not mondo_name:
                continue

            upd = out.setdefault(
                node_id,
                {
                    "source": "PrimeKG",
                    "source_version": p.name,
                    "evidence": "primekg_disease_features",
                    "source_meta_patch": {
                        "primekg_disease_features": {
                            "node_index": idx,
                            "row_count": 0,
                            "mondo_ids": [],
                            "mondo_names": [],
                        }
                    },
                },
            )
            upd_any = upd
            smp = cast("dict[str, Any]", upd_any.setdefault("source_meta_patch", {}))
            feat = cast(
                "dict[str, Any]",
                smp.setdefault(
                    "primekg_disease_features",
                    {
                        "node_index": idx,
                        "row_count": 0,
                        "mondo_ids": [],
                        "mondo_names": [],
                    },
                ),
            )
            feat["row_count"] = int(feat.get("row_count") or 0) + 1

            if mondo_id:
                ids = cast("list[str]", feat.setdefault("mondo_ids", []))
                ids.append(mondo_id)
                feat["mondo_ids"] = sorted(set(ids))

            if mondo_name:
                names_ = cast("list[str]", feat.setdefault("mondo_names", []))
                names_.append(mondo_name)
                feat["mondo_names"] = sorted(set(names_))
            if mondo_name:
                names = cast("list[dict[str, Any]]", upd_any.setdefault("names", []))
                names.append(
                    {
                        "name": mondo_name,
                        "source": "PrimeKG",
                        "source_version": p.name,
                        "evidence": "primekg_disease_features",
                    }
                )
            out[node_id] = upd

        return out


def read_primekg_drug_features_tab(
    path: str | Path, *, index_to_node_id: dict[int, str], strict: bool = True
) -> dict[str, dict[str, Any]]:
    """Read `drug_features.tab` and convert to `node_updates` keyed by node_id.

    Stores the raw drug text fields in ``source_meta_patch.primekg_drug_features``
    and emits ``feature_refs.text`` indirection pointers (into ``primehkg-features``)
    for each present modality in ``DRUG_TEXT_MODALITIES``
    (description / indication / mechanism_of_action / pharmacodynamics).
    No xref/name inference.
    """
    p = Path(path)
    with p.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if not reader.fieldnames:
            msg = "PrimeKG drug_features.tab is missing a header row."
            raise ValueError(msg)
        cols = {c.strip(): c for c in reader.fieldnames if c is not None}
        if "node_index" not in cols:
            msg = "PrimeKG drug_features.tab missing required columns: ['node_index']"
            raise ValueError(msg)

        out: dict[str, dict[str, Any]] = {}
        for row_number, row in enumerate(reader, start=2):
            idx_raw = (row.get(cols["node_index"]) or "").strip()
            if not idx_raw:
                continue
            try:
                idx = int(idx_raw)
            except ValueError:
                if strict:
                    msg = f"invalid node_index at row {row_number}: {idx_raw!r}"
                    raise ValueError(msg) from None
                continue
            node_id = index_to_node_id.get(idx)
            if not node_id:
                if strict:
                    msg = f"node_index not found in kg.csv join map at row {row_number}: {idx}"
                    raise ValueError(msg)
                continue

            # In-scope drug text modalities (strip surrounding quotes); only these
            # four get a feature_refs text pointer (see DRUG_TEXT_MODALITIES).
            values = {
                m: (row.get(cols.get(m, "")) or "").strip().strip('"')
                for m in DRUG_TEXT_MODALITIES
            }
            present = {m: v for m, v in values.items() if v}

            if not present:
                continue

            upd = out.setdefault(
                node_id,
                {
                    "source": "PrimeKG",
                    "source_version": p.name,
                    "evidence": "primekg_drug_features",
                    "source_meta_patch": {
                        "primekg_drug_features": {
                            "node_index": idx,
                            "row_count": 0,
                            "description": None,
                            "indication": None,
                            "mechanism_of_action": None,
                            "pharmacodynamics": None,
                        }
                    },
                },
            )
            upd_any = upd
            smp = cast("dict[str, Any]", upd_any.setdefault("source_meta_patch", {}))
            feat = cast(
                "dict[str, Any]",
                smp.setdefault(
                    "primekg_drug_features",
                    {
                        "node_index": idx,
                        "row_count": 0,
                        "description": None,
                        "indication": None,
                        "mechanism_of_action": None,
                        "pharmacodynamics": None,
                    },
                ),
            )
            feat["row_count"] = int(feat.get("row_count") or 0) + 1
            for m, v in present.items():
                if feat.get(m) is None:
                    feat[m] = v
            # Keep the last parsed index for visibility/debug (should be stable).
            feat["node_index"] = idx

            # Indirection pointers into primehkg-features for present modalities.
            text_refs = cast("dict[str, Any]", upd_any.setdefault("feature_refs", {}))
            existing = cast("list[dict[str, str]]", text_refs.setdefault("text", []))
            have = {r.get("modality") for r in existing}
            for m in DRUG_TEXT_MODALITIES:
                if m in present and m not in have:
                    existing.append(
                        text_pointer(
                            modality=m, source="PrimeKG", source_version=p.name
                        )
                    )
            out[node_id] = upd
        return out


def read_primekg_nodes_tab(
    path: str | Path,
    *,
    strict: bool = True,
) -> dict[str, dict[str, Any]]:
    """Read PrimeKG `nodes.tab` and convert to `node_updates` keyed by canonical node_id."""
    p = Path(path)
    with p.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if not reader.fieldnames:
            msg = "PrimeKG nodes.tab is missing a header row."
            raise ValueError(msg)
        cols = {c.strip(): c for c in reader.fieldnames if c is not None}
        required = {"node_index", "node_id", "node_type", "node_name", "node_source"}
        missing = required - set(cols)
        if missing:
            msg = f"PrimeKG nodes.tab missing required columns: {sorted(missing)}"
            raise ValueError(msg)

        out: dict[str, dict[str, Any]] = {}
        for row_number, row in enumerate(reader, start=2):
            try:
                idx_raw = (row.get(cols["node_index"]) or "").strip().strip('"')
                raw_id = (row.get(cols["node_id"]) or "").strip().strip('"')
                raw_type = (row.get(cols["node_type"]) or "").strip().strip('"')
                raw_name = (row.get(cols["node_name"]) or "").strip().strip('"')
                raw_source = (row.get(cols["node_source"]) or "").strip().strip('"')

                if not idx_raw or not raw_id or not raw_type:
                    continue
                idx = int(idx_raw)
                node_id = _primekg_typed_node_id(raw_id, raw_type)
                if not node_id:
                    continue

                upd = out.setdefault(
                    node_id,
                    {
                        "source": "PrimeKG",
                        "source_version": p.name,
                        "evidence": "primekg_nodes_tab",
                        "source_meta_patch": {
                            "primekg_nodes_tab": {
                                "node_index": idx,
                                "node_type": raw_type or None,
                                "node_source": raw_source or None,
                            }
                        },
                    },
                )
                upd_any = upd
                if raw_name:
                    names = cast(
                        "list[dict[str, Any]]", upd_any.setdefault("names", [])
                    )
                    names.append(
                        {
                            "name": raw_name,
                            "source": "PrimeKG",
                            "source_version": p.name,
                            "evidence": "primekg_nodes_tab",
                        }
                    )
            except (ValueError, TypeError) as exc:
                if strict:
                    msg = f"failed to read nodes.tab at row {row_number}: {exc}"
                    raise ValueError(msg) from exc
                continue

        return out
