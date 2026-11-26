"""Parse DrugBank full-database XML for maintainer-owned SMILES exports."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path


def _local_tag(tag: str) -> str:
    if "}" in tag:
        return tag.rsplit("}", 1)[-1]
    return tag


def _property_smiles(properties_elem: ET.Element | None) -> str | None:
    if properties_elem is None:
        return None
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
        if kind == "SMILES" and value:
            return value
    return None


def load_drugbank_smiles_index(xml_path: str | Path) -> dict[str, str]:
    """Return DrugBank id (e.g. ``DB00001``) → canonical SMILES from XML."""
    path = Path(xml_path)
    if not path.is_file():
        msg = f"DrugBank XML not found: {path}"
        raise FileNotFoundError(msg)

    index: dict[str, str] = {}
    for _event, elem in ET.iterparse(path, events=("end",)):  # noqa: S314
        if _local_tag(elem.tag) != "drug":
            continue
        drugbank_id: str | None = None
        smiles: str | None = None
        for child in elem:
            child_tag = _local_tag(child.tag)
            if child_tag == "drugbank-id" and child.get("primary") == "true":
                text = (child.text or "").strip()
                if text:
                    drugbank_id = text
            elif child_tag in {"calculated-properties", "experimental-properties"}:
                candidate = _property_smiles(child)
                if candidate and smiles is None:
                    smiles = candidate
        if drugbank_id and smiles:
            index.setdefault(drugbank_id, smiles)
        elem.clear()
    return index
