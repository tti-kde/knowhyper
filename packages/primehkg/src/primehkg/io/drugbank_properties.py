"""Parse DrugBank full-database XML for physicochemical property exports."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from primehkg.io.drugbank_structures import _local_tag

# DrugBank calculated/experimental property `kind` → registry `property_id`.
DRUGBANK_KIND_TO_PROPERTY_ID: dict[str, str] = {
    "Molecular Weight": "molecular_weight",
    "logP": "alogp",
    "ALOGP": "alogp",
    "Polar Surface Area (PSA)": "psa",
    "PSA": "psa",
    "Hydrogen Bond Acceptor Count": "hba",
    "Hydrogen Bond Donor Count": "hbd",
}


def _parse_numeric_value(raw: str) -> float | None:
    text = raw.strip().strip('"')
    if not text:
        return None
    try:
        return float(text.replace(",", ""))
    except ValueError:
        return None


def _properties_from_elem(properties_elem: ET.Element | None) -> dict[str, float]:
    if properties_elem is None:
        return {}
    out: dict[str, float] = {}
    for prop in properties_elem:
        if _local_tag(prop.tag) != "property":
            continue
        kind: str | None = None
        value: str | None = None
        for child in prop:
            child_tag = _local_tag(child.tag)
            if child_tag == "kind":
                kind = (child.text or "").strip()
            elif child_tag == "value":
                value = (child.text or "").strip()
        if not kind or not value or kind == "SMILES":
            continue
        property_id = DRUGBANK_KIND_TO_PROPERTY_ID.get(kind)
        if property_id is None:
            continue
        numeric = _parse_numeric_value(value)
        if numeric is None:
            continue
        out[property_id] = numeric
    return out


def load_drugbank_properties_index(
    xml_path: str | Path,
) -> dict[str, dict[str, float]]:
    """Return DrugBank id → {property_id: value} from XML calculated/experimental props."""
    path = Path(xml_path)
    if not path.is_file():
        msg = f"DrugBank XML not found: {path}"
        raise FileNotFoundError(msg)

    index: dict[str, dict[str, float]] = {}
    for _event, elem in ET.iterparse(path, events=("end",)):  # noqa: S314
        if _local_tag(elem.tag) != "drug":
            continue
        drugbank_id: str | None = None
        props: dict[str, float] = {}
        for child in elem:
            child_tag = _local_tag(child.tag)
            if child_tag == "drugbank-id" and child.get("primary") == "true":
                text = (child.text or "").strip()
                if text:
                    drugbank_id = text
            elif child_tag in {"calculated-properties", "experimental-properties"}:
                for pid, val in _properties_from_elem(child).items():
                    props.setdefault(pid, val)
        if drugbank_id and props:
            existing = index.setdefault(drugbank_id, {})
            existing.update(props)
        elem.clear()
    return index
