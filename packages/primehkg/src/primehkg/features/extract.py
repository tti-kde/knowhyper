from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Any


def _strip_header_map(fieldnames: list[str] | None) -> dict[str, str]:
    if not fieldnames:
        return {}
    return {h.strip(): h for h in fieldnames if h is not None}


_UNIPROT_FUNCTION_COL_ALIASES = ("Function [CC]", "Function", "cc_function")


def _uniprot_function_col(cols: dict[str, str]) -> str | None:
    for alias in _UNIPROT_FUNCTION_COL_ALIASES:
        if alias in cols:
            return cols[alias]
    return None


def _uniprot_row_node_id(
    row: dict[str, str | None], cols: dict[str, str]
) -> str | None:
    if "node_id" in cols:
        node_id = (row.get(cols["node_id"]) or "").strip()
        if node_id:
            return node_id
    for gene_col in ("Gene ID", "gene_id", "entrez_id", "GeneID"):
        if gene_col in cols:
            gene_id = (row.get(cols[gene_col]) or "").strip()
            if gene_id:
                return f"primekg:protein:{gene_id}"
    if "Entry" in cols:
        entry = (row.get(cols["Entry"]) or "").strip()
        if entry:
            return f"primekg:protein:{entry}"
    return None


def read_uniprot_sequence_feature_rows(
    path: str | Path,
    *,
    allowed_node_ids: frozenset[str] | None = None,
    delimiter: str = "\t",
) -> list[dict[str, Any]]:
    """Extract protein sequence payloads for ``node_features_sequence``."""
    p = Path(path)
    rows_out: list[dict[str, Any]] = []
    with p.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        if reader.fieldnames is None:
            return rows_out
        cols = _strip_header_map(list(reader.fieldnames))
        if "Sequence" not in cols:
            return rows_out
        if (
            "Entry" not in cols
            and "node_id" not in cols
            and not any(
                c in cols for c in ("Gene ID", "gene_id", "entrez_id", "GeneID")
            )
        ):
            return rows_out
        source_version = p.name
        for row in reader:
            sequence = (row.get(cols["Sequence"]) or "").strip()
            if not sequence:
                continue
            node_id = _uniprot_row_node_id(row, cols)
            if not node_id:
                continue
            if allowed_node_ids is not None and node_id not in allowed_node_ids:
                continue
            rows_out.append(
                {
                    "node_id": node_id,
                    "sequence_type": "protein",
                    "sequence": sequence,
                    "source": "UniProt",
                    "source_version": source_version,
                    "license_tag": None,
                }
            )
    rows_out.sort(
        key=lambda r: (
            str(r["node_id"]),
            str(r["sequence_type"]),
            str(r["source"]),
            str(r["source_version"]),
        )
    )
    return rows_out


def read_nucleotide_sequence_feature_rows(
    path: str | Path,
    *,
    allowed_node_ids: frozenset[str] | None = None,
    delimiter: str = "\t",
) -> list[dict[str, Any]]:
    """Extract nucleotide sequence payloads for ``node_features_sequence``."""
    p = Path(path)
    rows_out: list[dict[str, Any]] = []
    with p.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        if reader.fieldnames is None:
            return rows_out
        cols = _strip_header_map(list(reader.fieldnames))
        if "Sequence" not in cols:
            return rows_out
        source_version = p.name
        for row in reader:
            sequence = (row.get(cols["Sequence"]) or "").strip()
            if not sequence:
                continue
            node_id: str | None = None
            if "node_id" in cols:
                node_id = (row.get(cols["node_id"]) or "").strip() or None
            if not node_id and "entrez_id" in cols:
                entrez = (row.get(cols["entrez_id"]) or "").strip()
                if entrez.isdigit():
                    node_id = f"primekg:protein:{entrez}"
            if not node_id:
                continue
            if allowed_node_ids is not None and node_id not in allowed_node_ids:
                continue
            accession = (row.get(cols.get("refseq_rna_accession", "")) or "").strip()
            rows_out.append(
                {
                    "node_id": node_id,
                    "sequence_type": "nucleotide",
                    "sequence": sequence,
                    "source": "NCBI RefSeq",
                    "source_version": accession or source_version,
                    "license_tag": None,
                }
            )
    rows_out.sort(
        key=lambda r: (
            str(r["node_id"]),
            str(r["sequence_type"]),
            str(r["source"]),
            str(r["source_version"]),
        )
    )
    return rows_out


def read_uniprot_text_feature_rows(
    path: str | Path,
    *,
    allowed_node_ids: frozenset[str] | None = None,
    delimiter: str = "\t",
) -> list[dict[str, Any]]:
    """Extract protein text payloads from UniProt export TSV (``node_features_text``)."""
    p = Path(path)
    rows_out: list[dict[str, Any]] = []
    with p.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        if reader.fieldnames is None:
            return rows_out
        cols = _strip_header_map(list(reader.fieldnames))
        if not any(k in cols for k in ("Protein names", "Gene Names", "Entry")):
            return rows_out
        source_version = p.name
        for row in reader:
            node_id = _uniprot_row_node_id(row, cols)
            if not node_id:
                continue
            if allowed_node_ids is not None and node_id not in allowed_node_ids:
                continue
            protein_name = (row.get(cols.get("Protein names", "")) or "").strip()
            gene_name = (
                row.get(cols.get("Gene Names", ""))
                or row.get(cols.get("Gene Names (primary)", ""))
                or ""
            ).strip()
            entry = (row.get(cols.get("Entry", "")) or "").strip()
            if protein_name:
                text, source_field = protein_name, "Protein names"
            elif gene_name:
                text, source_field = gene_name, "Gene Names"
            elif entry:
                text, source_field = entry, "Entry"
            else:
                text, source_field = "", ""
            function_col = _uniprot_function_col(cols)
            function_text = (
                (row.get(function_col) or "").strip() if function_col else ""
            )
            if not text and not function_text:
                continue
            if text:
                rows_out.append(
                    _text_row(
                        node_id=node_id,
                        modality="description",
                        text=text,
                        source="UniProt",
                        source_version=source_version,
                        source_field=source_field,
                        payload_class="short_label",
                    )
                )
            if function_text:
                rows_out.append(
                    _text_row(
                        node_id=node_id,
                        modality="definition",
                        text=function_text,
                        source="UniProt",
                        source_version=source_version,
                        source_field="Function",
                        payload_class="semantic_definition",
                    )
                )
    rows_out.sort(
        key=lambda r: (
            str(r["node_id"]),
            str(r["modality"]),
            str(r["source"]),
            str(r["source_version"]),
        )
    )
    return rows_out


def read_chembl_smiles_feature_rows(
    path: str | Path,
    *,
    allowed_node_ids: frozenset[str] | None = None,
    delimiter: str = "\t",
) -> list[dict[str, Any]]:
    """Extract SMILES payloads for ``node_features_sequence`` (``sequence_type=smiles``)."""
    p = Path(path)
    rows_out: list[dict[str, Any]] = []
    with p.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        if reader.fieldnames is None:
            return rows_out
        cols = _strip_header_map(list(reader.fieldnames))
        if "node_id" not in cols or "canonical_smiles" not in cols:
            return rows_out
        source_version = p.name
        for row in reader:
            node_id = (row.get(cols["node_id"]) or "").strip()
            smiles = (row.get(cols["canonical_smiles"]) or "").strip()
            if not node_id or not smiles:
                continue
            if allowed_node_ids is not None and node_id not in allowed_node_ids:
                continue
            rows_out.append(
                {
                    "node_id": node_id,
                    "sequence_type": "smiles",
                    "sequence": smiles,
                    "source": "ChEMBL",
                    "source_version": source_version,
                    "license_tag": None,
                }
            )
    rows_out.sort(
        key=lambda r: (
            str(r["node_id"]),
            str(r["sequence_type"]),
            str(r["source"]),
            str(r["source_version"]),
        )
    )
    return rows_out


def read_chembl_property_feature_rows(
    path: str | Path,
    *,
    allowed_node_ids: frozenset[str] | None = None,
    delimiter: str = "\t",
    registry: Any | None = None,
) -> list[dict[str, Any]]:
    """Extract ChEMBL numeric columns as ``node_features_property`` rows."""
    from primehkg.features.property_rows import property_row
    from primehkg.policy.node_property_registry import (
        NodePropertyRegistry,
        chembl_column_to_property_id,
        load_node_property_registry,
    )

    reg: NodePropertyRegistry = (
        registry
        if isinstance(registry, NodePropertyRegistry)
        else load_node_property_registry()
    )
    col_to_pid = chembl_column_to_property_id(reg)
    if not col_to_pid:
        return []

    p = Path(path)
    rows_out: list[dict[str, Any]] = []
    with p.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        if reader.fieldnames is None:
            return rows_out
        cols = _strip_header_map(list(reader.fieldnames))
        if "node_id" not in cols:
            return rows_out
        source_version = p.name
        chembl_id_col = cols.get("chembl_id")
        for row in reader:
            node_id = (row.get(cols["node_id"]) or "").strip()
            if not node_id:
                continue
            if allowed_node_ids is not None and node_id not in allowed_node_ids:
                continue
            chembl_id_val = ""
            if chembl_id_col is not None:
                chembl_id_val = (row.get(chembl_id_col) or "").strip()
            row_source = "ChEMBL" if chembl_id_val else "PubChem"
            for col_name, property_id in col_to_pid.items():
                col = cols.get(col_name)
                if col is None:
                    continue
                raw = (row.get(col) or "").strip()
                if not raw:
                    continue
                prop_def = reg.by_id[property_id]
                try:
                    if prop_def.dtype == "int":
                        value: float | int = int(float(raw.replace(",", "")))
                    else:
                        value = float(raw.replace(",", ""))
                except ValueError:
                    continue
                rows_out.append(
                    property_row(
                        node_id=node_id,
                        property_id=property_id,
                        value_numeric=value,
                        dtype=prop_def.dtype,
                        source=row_source,
                        source_version=source_version,
                        unit=prop_def.unit,
                    )
                )
    rows_out.sort(
        key=lambda r: (
            str(r["node_id"]),
            str(r["property_id"]),
            str(r["source"]),
            str(r["source_version"]),
        )
    )
    return rows_out


def read_drugbank_property_feature_rows(
    path: str | Path,
    *,
    allowed_node_ids: frozenset[str] | None = None,
    delimiter: str = "\t",
    registry: Any | None = None,
) -> list[dict[str, Any]]:
    """Extract DrugBank property TSV columns as ``node_features_property`` rows."""
    from primehkg.features.property_rows import property_row
    from primehkg.policy.node_property_registry import (
        NodePropertyRegistry,
        drugbank_column_to_property_id,
        load_node_property_registry,
    )

    reg: NodePropertyRegistry = (
        registry
        if isinstance(registry, NodePropertyRegistry)
        else load_node_property_registry()
    )
    col_to_pid = drugbank_column_to_property_id(reg)
    if not col_to_pid:
        return []

    p = Path(path)
    rows_out: list[dict[str, Any]] = []
    with p.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        if reader.fieldnames is None:
            return rows_out
        cols = _strip_header_map(list(reader.fieldnames))
        if "node_id" not in cols:
            return rows_out
        source_version = p.name
        for row in reader:
            node_id = (row.get(cols["node_id"]) or "").strip()
            if not node_id:
                continue
            if allowed_node_ids is not None and node_id not in allowed_node_ids:
                continue
            for col_name, property_id in col_to_pid.items():
                col = cols.get(col_name)
                if col is None:
                    continue
                raw = (row.get(col) or "").strip()
                if not raw:
                    continue
                prop_def = reg.by_id[property_id]
                try:
                    if prop_def.dtype == "int":
                        value: float | int = int(float(raw.replace(",", "")))
                    else:
                        value = float(raw.replace(",", ""))
                except ValueError:
                    continue
                rows_out.append(
                    property_row(
                        node_id=node_id,
                        property_id=property_id,
                        value_numeric=value,
                        dtype=prop_def.dtype,
                        source="DrugBank",
                        source_version=source_version,
                        unit=prop_def.unit,
                    )
                )
    rows_out.sort(
        key=lambda r: (
            str(r["node_id"]),
            str(r["property_id"]),
            str(r["source"]),
            str(r["source_version"]),
        )
    )
    return rows_out


def read_drugbank_smiles_feature_rows(
    path: str | Path,
    *,
    allowed_node_ids: frozenset[str] | None = None,
    delimiter: str = "\t",
) -> list[dict[str, Any]]:
    """Extract DrugBank SMILES fallback rows for ``node_features_sequence``."""
    p = Path(path)
    rows_out: list[dict[str, Any]] = []
    with p.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        if reader.fieldnames is None:
            return rows_out
        cols = _strip_header_map(list(reader.fieldnames))
        if "node_id" not in cols:
            return rows_out
        smiles_col = cols.get("canonical_smiles") or cols.get("smiles")
        if smiles_col is None:
            return rows_out
        source_version = p.name
        for row in reader:
            node_id = (row.get(cols["node_id"]) or "").strip()
            smiles = (row.get(smiles_col) or "").strip()
            if not node_id or not smiles:
                continue
            if allowed_node_ids is not None and node_id not in allowed_node_ids:
                continue
            rows_out.append(
                {
                    "node_id": node_id,
                    "sequence_type": "smiles",
                    "sequence": smiles,
                    "source": "DrugBank",
                    "source_version": source_version,
                    "license_tag": None,
                }
            )
    rows_out.sort(
        key=lambda r: (
            str(r["node_id"]),
            str(r["sequence_type"]),
            str(r["source"]),
            str(r["source_version"]),
        )
    )
    return rows_out


def merge_smiles_gap_fill(
    primary_rows: list[dict[str, Any]],
    fallback_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Append fallback SMILES rows only for nodes missing ``sequence_type=smiles``."""
    covered = {
        row["node_id"]
        for row in primary_rows
        if row.get("sequence_type") == "smiles" and row.get("node_id")
    }
    merged = list(primary_rows)
    for row in fallback_rows:
        node_id = row.get("node_id")
        if not isinstance(node_id, str) or not node_id or node_id in covered:
            continue
        merged.append(row)
        covered.add(node_id)
    return merged


def _text_row(
    *,
    node_id: str,
    modality: str,
    text: str,
    source: str,
    source_version: str,
    source_field: str,
    payload_class: str,
    upstream_source: str | None = None,
    upstream_release: str | None = None,
    upstream_record_id: str | None = None,
    provenance_chain: list[Any] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "node_id": node_id,
        "modality": modality,
        "text": text,
        "language": "en",
        "source": source,
        "source_version": source_version,
        "source_field": source_field,
        "payload_class": payload_class,
        "license_tag": None,
        "content_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }
    if upstream_source is not None:
        row["upstream_source"] = upstream_source
    if upstream_release is not None:
        row["upstream_release"] = upstream_release
    if upstream_record_id is not None:
        row["upstream_record_id"] = upstream_record_id
    if provenance_chain is not None:
        row["provenance_chain"] = provenance_chain
    return row


def read_mondo_text_feature_rows(
    path: str | Path,
    *,
    mondo_obo: Path | None = None,
    allowed_node_ids: frozenset[str] | None = None,
    delimiter: str = "\t",
) -> list[dict[str, Any]]:
    """Extract MONDO text features.

    With ``mondo_obo``, emit OBO-derived ``definition`` rows (``semantic_definition``)
    plus optional ``mondo_name`` short labels. Without ``mondo_obo``, emit only
    ``mondo_name`` short labels — map ``definition`` is never emitted (fail-closed
    so packaged ``allow`` pins cannot stamp PrimeKG map prose as OBO-complete).
    """
    from primehkg.features.mondo_obo import (
        normalize_mondo_id,
        parse_mondo_obo_definitions,
    )

    p = Path(path)
    obo_definitions: dict[str, str] | None = None
    if mondo_obo is not None:
        obo_definitions = parse_mondo_obo_definitions(Path(mondo_obo))

    rows_out: list[dict[str, Any]] = []
    with p.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        if reader.fieldnames is None:
            return rows_out
        cols = _strip_header_map(list(reader.fieldnames))
        if "node_id" not in cols:
            return rows_out
        source_version = p.name
        for row in reader:
            node_id = (row.get(cols["node_id"]) or "").strip()
            if not node_id:
                continue
            if allowed_node_ids is not None and node_id not in allowed_node_ids:
                continue
            mondo_name = (row.get(cols.get("mondo_name", "")) or "").strip()
            mondo_id = (row.get(cols.get("mondo_id", "")) or "").strip()

            # Fail-closed: map ``definition`` is never emitted. Publishable
            # semantic definitions require ``mondo_obo`` (OBO-derived text).
            # Without OBO, only ``mondo_name`` short_label rows are allowed.
            if obo_definitions is not None:
                if mondo_id:
                    obo_def = obo_definitions.get(normalize_mondo_id(mondo_id))
                    if obo_def:
                        rows_out.append(
                            _text_row(
                                node_id=node_id,
                                modality="definition",
                                text=obo_def,
                                source="MONDO",
                                source_version=source_version,
                                source_field="definition",
                                payload_class="semantic_definition",
                            )
                        )
                if mondo_name:
                    rows_out.append(
                        _text_row(
                            node_id=node_id,
                            modality="description",
                            text=mondo_name,
                            source="MONDO",
                            source_version=source_version,
                            source_field="mondo_name",
                            payload_class="short_label",
                        )
                    )
                continue

            if mondo_name:
                rows_out.append(
                    _text_row(
                        node_id=node_id,
                        modality="description",
                        text=mondo_name,
                        source="MONDO",
                        source_version=source_version,
                        source_field="mondo_name",
                        payload_class="short_label",
                    )
                )
            # intentionally ignore map ``definition`` when mondo_obo is unset
    rows_out.sort(
        key=lambda r: (
            str(r["node_id"]),
            str(r["modality"]),
            str(r["source"]),
            str(r["source_version"]),
        )
    )
    return rows_out


def read_hgnc_text_feature_rows(
    path: str | Path,
    *,
    allowed_node_ids: frozenset[str] | None = None,
    delimiter: str = "\t",
) -> list[dict[str, Any]]:
    """Map HGNC export → node_features_text (modality=description).

    Emits rows for ``primekg:protein:{entrez_id}`` only. ``primekg:gene:HGNC:*`` is not
    used for features on reference builds (PrimeKG ``gene/protein`` → protein nodes).
    """
    p = Path(path)
    rows_out: list[dict[str, Any]] = []
    with p.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        if reader.fieldnames is None:
            return rows_out
        cols = _strip_header_map(list(reader.fieldnames))
        if "hgnc_id" not in cols or "symbol" not in cols:
            return rows_out
        source_version = p.name
        for row in reader:
            hgnc_id = (row.get(cols["hgnc_id"]) or "").strip()
            symbol = (row.get(cols["symbol"]) or "").strip()
            if not hgnc_id or not symbol:
                continue
            name = (row.get(cols.get("name", "")) or "").strip()
            if name:
                text, source_field = name, "name"
            else:
                text, source_field = symbol, "symbol"
            entrez_id = (row.get(cols.get("entrez_id", "")) or "").strip()
            if entrez_id.isdigit():
                protein_node_id = f"primekg:protein:{entrez_id}"
                if allowed_node_ids is None or protein_node_id in allowed_node_ids:
                    rows_out.append(
                        _text_row(
                            node_id=protein_node_id,
                            modality="description",
                            text=text,
                            source="HGNC",
                            source_version=source_version,
                            source_field=source_field,
                            payload_class="short_label",
                        )
                    )
    rows_out.sort(
        key=lambda r: (
            str(r["node_id"]),
            str(r["modality"]),
            str(r["source"]),
            str(r["source_version"]),
        )
    )
    return rows_out


def read_hpo_text_feature_rows(
    path: str | Path,
    *,
    allowed_node_ids: frozenset[str] | None = None,
    delimiter: str = "\t",
) -> list[dict[str, Any]]:
    """Extract text from HPO mapping TSV (``hp_label`` or optional ``definition``)."""
    p = Path(path)
    rows_out: list[dict[str, Any]] = []
    with p.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        if reader.fieldnames is None:
            return rows_out
        cols = _strip_header_map(list(reader.fieldnames))
        if "node_id" not in cols:
            return rows_out
        source_version = p.name
        for row in reader:
            node_id = (row.get(cols["node_id"]) or "").strip()
            if not node_id:
                continue
            if allowed_node_ids is not None and node_id not in allowed_node_ids:
                continue
            definition = (row.get(cols.get("definition", "")) or "").strip()
            hp_label = (row.get(cols.get("hp_label", "")) or "").strip()
            if definition:
                text, modality, source_field = definition, "definition", "definition"
                payload_class = "semantic_definition"
            elif hp_label:
                text, modality, source_field = hp_label, "description", "hp_label"
                payload_class = "short_label"
            else:
                continue
            rows_out.append(
                _text_row(
                    node_id=node_id,
                    modality=modality,
                    text=text,
                    source="HPO",
                    source_version=source_version,
                    source_field=source_field,
                    payload_class=payload_class,
                )
            )
    rows_out.sort(
        key=lambda r: (
            str(r["node_id"]),
            str(r["modality"]),
            str(r["source"]),
            str(r["source_version"]),
        )
    )
    return rows_out


def read_go_text_feature_rows(
    path: str | Path,
    *,
    allowed_node_ids: frozenset[str] | None = None,
    delimiter: str = "\t",
) -> list[dict[str, Any]]:
    """Extract GO definition text from ``go_map.tsv``."""
    p = Path(path)
    rows_out: list[dict[str, Any]] = []
    with p.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        if reader.fieldnames is None:
            return rows_out
        cols = _strip_header_map(list(reader.fieldnames))
        if "node_id" not in cols:
            return rows_out
        source_version = p.name
        for row in reader:
            node_id = (row.get(cols["node_id"]) or "").strip()
            if not node_id:
                continue
            if allowed_node_ids is not None and node_id not in allowed_node_ids:
                continue
            definition = (row.get(cols.get("definition", "")) or "").strip()
            if not definition:
                continue
            rows_out.append(
                _text_row(
                    node_id=node_id,
                    modality="definition",
                    text=definition,
                    source="GO",
                    source_version=source_version,
                    source_field="definition",
                    payload_class="semantic_definition",
                )
            )
    rows_out.sort(
        key=lambda r: (
            str(r["node_id"]),
            str(r["modality"]),
            str(r["source"]),
            str(r["source_version"]),
        )
    )
    return rows_out


def read_pathway_text_feature_rows(
    path: str | Path,
    *,
    allowed_node_ids: frozenset[str] | None = None,
    delimiter: str = "\t",
) -> list[dict[str, Any]]:
    """Extract pathway description text from ``pathway_map.tsv``."""
    p = Path(path)
    rows_out: list[dict[str, Any]] = []
    with p.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        if reader.fieldnames is None:
            return rows_out
        cols = _strip_header_map(list(reader.fieldnames))
        if "node_id" not in cols:
            return rows_out
        source_version = p.name
        for row in reader:
            node_id = (row.get(cols["node_id"]) or "").strip()
            if not node_id:
                continue
            if allowed_node_ids is not None and node_id not in allowed_node_ids:
                continue
            description = (row.get(cols.get("description", "")) or "").strip()
            if not description:
                continue
            rows_out.append(
                _text_row(
                    node_id=node_id,
                    modality="description",
                    text=description,
                    source="pathway",
                    source_version=source_version,
                    source_field="description",
                    payload_class="short_label",
                )
            )
    rows_out.sort(
        key=lambda r: (
            str(r["node_id"]),
            str(r["modality"]),
            str(r["source"]),
            str(r["source_version"]),
        )
    )
    return rows_out


def read_anatomy_text_feature_rows(
    path: str | Path,
    *,
    allowed_node_ids: frozenset[str] | None = None,
    delimiter: str = "\t",
) -> list[dict[str, Any]]:
    """Extract anatomy description text from ``anatomy_map.tsv``."""
    p = Path(path)
    rows_out: list[dict[str, Any]] = []
    with p.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        if reader.fieldnames is None:
            return rows_out
        cols = _strip_header_map(list(reader.fieldnames))
        if "node_id" not in cols:
            return rows_out
        source_version = p.name
        for row in reader:
            node_id = (row.get(cols["node_id"]) or "").strip()
            if not node_id:
                continue
            if allowed_node_ids is not None and node_id not in allowed_node_ids:
                continue
            description = (row.get(cols.get("description", "")) or "").strip()
            if not description:
                continue
            rows_out.append(
                _text_row(
                    node_id=node_id,
                    modality="description",
                    text=description,
                    source="anatomy",
                    source_version=source_version,
                    source_field="description",
                    payload_class="short_label",
                )
            )
    rows_out.sort(
        key=lambda r: (
            str(r["node_id"]),
            str(r["modality"]),
            str(r["source"]),
            str(r["source_version"]),
        )
    )
    return rows_out


def read_exposure_text_feature_rows(
    path: str | Path,
    *,
    allowed_node_ids: frozenset[str] | None = None,
    delimiter: str = "\t",
) -> list[dict[str, Any]]:
    """Extract exposure description text from ``exposure_map.tsv``."""
    p = Path(path)
    rows_out: list[dict[str, Any]] = []
    with p.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        if reader.fieldnames is None:
            return rows_out
        cols = _strip_header_map(list(reader.fieldnames))
        if "node_id" not in cols:
            return rows_out
        source_version = p.name
        for row in reader:
            node_id = (row.get(cols["node_id"]) or "").strip()
            if not node_id:
                continue
            if allowed_node_ids is not None and node_id not in allowed_node_ids:
                continue
            description = (row.get(cols.get("description", "")) or "").strip()
            if not description:
                continue
            rows_out.append(
                _text_row(
                    node_id=node_id,
                    modality="description",
                    text=description,
                    source="exposure",
                    source_version=source_version,
                    source_field="description",
                    payload_class="short_label",
                )
            )
    rows_out.sort(
        key=lambda r: (
            str(r["node_id"]),
            str(r["modality"]),
            str(r["source"]),
            str(r["source_version"]),
        )
    )
    return rows_out


def read_chebi_text_feature_rows(
    path: str | Path,
    *,
    allowed_node_ids: frozenset[str] | None = None,
    delimiter: str = "\t",
) -> list[dict[str, Any]]:
    """Extract ChEBI definition text from a chebi_definitions.tsv."""
    p = Path(path)
    rows_out: list[dict[str, Any]] = []
    with p.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        if reader.fieldnames is None:
            return rows_out
        cols = _strip_header_map(list(reader.fieldnames))
        if "chebi_node_id" not in cols or "definition" not in cols:
            return rows_out
        source_version = p.name
        for row in reader:
            node_id = (row.get(cols["chebi_node_id"]) or "").strip()
            definition = (row.get(cols["definition"]) or "").strip()
            if not node_id or not definition:
                continue
            if allowed_node_ids is not None and node_id not in allowed_node_ids:
                continue
            rows_out.append(
                _text_row(
                    node_id=node_id,
                    modality="definition",
                    text=definition,
                    source="chebi",
                    source_version=source_version,
                    source_field="definition",
                    payload_class="semantic_definition",
                )
            )
    rows_out.sort(
        key=lambda r: (
            str(r["node_id"]),
            str(r["modality"]),
            str(r["source"]),
            str(r["source_version"]),
        )
    )
    return rows_out


def structure_skeleton_feature_rows(
    skeleton: dict[str, Any],
    *,
    source_version: str = "structure_skeleton_v1",
) -> list[dict[str, Any]]:
    """Map ``build_structure_feature_skeleton`` output to ``node_features_structure`` rows."""
    node_ids = skeleton.get("node_ids")
    if not isinstance(node_ids, list):
        msg = "skeleton missing node_ids list"
        raise TypeError(msg)
    kind = str(skeleton.get("kind") or "structure")
    rows_out: list[dict[str, Any]] = []
    for node_id in node_ids:
        if not isinstance(node_id, str) or not node_id:
            continue
        rows_out.append(
            {
                "node_id": node_id,
                "structure_ref": f"primehkg:derived:{kind}",
                "format": "derived_scalar",
                "source": "PrimeHKG",
                "source_version": source_version,
            }
        )
    rows_out.sort(key=lambda r: str(r["node_id"]))
    return rows_out
