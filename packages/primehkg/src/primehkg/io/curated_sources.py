from __future__ import annotations

import csv
import gzip
import re
from pathlib import Path
from statistics import median
from typing import TYPE_CHECKING, Any, TypedDict

if TYPE_CHECKING:
    from collections.abc import Sequence


class CorumComplex(TypedDict):
    """A single CORUM complex's participant list and optional display name."""

    uniprot_ids: list[str]
    complex_name: str | None


class ComplexPortalComplex(TypedDict):
    """A single Complex Portal complex's UniProt participants and display name."""

    uniprot_ids: list[str]
    name: str


class ReactomeReactionRow(TypedDict):
    """A single Reactome reaction participant row."""

    entity_id: str
    role: str


def _raise_on_duplicate_stripped_headers(
    fieldnames: Sequence[str | None], *, source: str
) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()

    for raw in fieldnames:
        if raw is None:
            continue
        stripped = raw.strip()
        if stripped in seen:
            duplicates.add(stripped)
            continue
        seen.add(stripped)

    if duplicates:
        msg = (
            f"{source} has duplicate header name(s) after stripping whitespace: "
            f"{sorted(duplicates)}"
        )
        raise ValueError(msg)


def load_corum_participants(path: str | Path) -> dict[str, CorumComplex]:
    """Load CORUM complex participants from a TSV.

    Required headers:
    - complex_id
    - uniprot_id

    Optional headers:
    - complex_name
    """
    p = Path(path)
    with p.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if not reader.fieldnames:
            msg = "CORUM TSV is missing a header row."
            raise ValueError(msg)

        _raise_on_duplicate_stripped_headers(reader.fieldnames, source="CORUM TSV")
        cols = {c.strip(): c for c in reader.fieldnames if c is not None}
        required = {"complex_id", "uniprot_id"}
        missing = required - set(cols)
        if missing:
            msg = f"CORUM TSV missing required columns: {sorted(missing)}"
            raise ValueError(msg)

        has_name = "complex_name" in cols

        participants: dict[str, set[str]] = {}
        complex_names: dict[str, set[str]] = {}

        for row in reader:
            complex_id = (row.get(cols["complex_id"]) or "").strip()
            uniprot_id = (row.get(cols["uniprot_id"]) or "").strip()
            if not complex_id or not uniprot_id:
                continue

            participants.setdefault(complex_id, set()).add(uniprot_id)
            if has_name:
                name_raw = (row.get(cols["complex_name"]) or "").strip()
                if name_raw:
                    complex_names.setdefault(complex_id, set()).add(name_raw)

        out: dict[str, CorumComplex] = {}
        for complex_id in sorted(participants):
            if complex_id in complex_names and len(complex_names[complex_id]) > 1:
                msg = (
                    "CORUM TSV has conflicting complex_name values for complex_id "
                    f"{complex_id!r}: {sorted(complex_names[complex_id])}"
                )
                raise ValueError(msg)

            out[complex_id] = {
                "uniprot_ids": sorted(participants[complex_id]),
                "complex_name": (
                    min(complex_names[complex_id])
                    if complex_id in complex_names
                    else None
                ),
            }
        return out


_UNIPROT_AC_RE = re.compile(
    r"^(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})$"
)


def _normalize_complexportal_token(token: str) -> str | None:
    """Return a bare UniProt AC for a participant token, or ``None`` to skip it.

    Strips the trailing ``(stoichiometry)``, drops chain/isoform suffixes
    (``-PRO_*`` / ``-<digits>``), and keeps only UniProt-AC-shaped tokens.
    """
    core = token.split("(", 1)[0].strip().upper()
    if not core:
        return None
    base = core.split("-", 1)[0]
    if _UNIPROT_AC_RE.match(base):
        return base
    return None


def load_complexportal_participants(
    path: str | Path,
) -> dict[str, ComplexPortalComplex]:
    """Load Complex Portal human complexes from the ``complextab/9606.tsv`` TSV.

    Uses the ``Expanded participant list`` column (nested complexes already flattened to base
    molecules). Keeps UniProt protein participants only; CHEBI / RNA / nested / other tokens are
    skipped. Stoichiometry and chain/isoform suffixes are stripped.
    """
    p = Path(path)
    with p.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if not reader.fieldnames:
            msg = "Complex Portal TSV is missing a header row."
            raise ValueError(msg)

        _raise_on_duplicate_stripped_headers(
            reader.fieldnames, source="Complex Portal TSV"
        )
        cols = {c.strip(): c for c in reader.fieldnames if c is not None}
        required = {"#Complex ac", "Expanded participant list"}
        missing = required - set(cols)
        if missing:
            msg = f"Complex Portal TSV missing required columns: {sorted(missing)}"
            raise ValueError(msg)
        has_name = "Recommended name" in cols

        out: dict[str, ComplexPortalComplex] = {}
        for row in reader:
            complex_id = (row.get(cols["#Complex ac"]) or "").strip()
            if not complex_id:
                continue
            expanded = (row.get(cols["Expanded participant list"]) or "").strip()
            acs: set[str] = set()
            for token in expanded.split("|"):
                ac = _normalize_complexportal_token(token)
                if ac:
                    acs.add(ac)
            name = ""
            if has_name:
                name = (row.get(cols["Recommended name"]) or "").strip()
            out[complex_id] = {"uniprot_ids": sorted(acs), "name": name}
        return out


def load_signor_rows(path: str | Path) -> list[dict[str, str]]:
    """Read a SIGNOR human TSV (no header) into positional dicts.

    Columns (0-indexed): [1] type_a [2] id_a [3] db_a [5] type_b [6] id_b [7] db_b
    [8] effect [9] mechanism [27] score. Rows with < 10 columns, or empty
    id_a / id_b, are skipped. A PMID (first all-digit field at index >= 19, if
    present) is captured as ``pmid``. The confidence ``score`` (raw string) is
    empty when the row is too short.
    """
    p = Path(path)
    raw = p.read_text(encoding="utf-8", errors="replace").splitlines()
    out: list[dict[str, str]] = []
    reader = csv.reader(raw, delimiter="\t")
    for row in reader:
        if len(row) < 10:
            continue
        id_a = row[2].strip()
        id_b = row[6].strip()
        if not id_a or not id_b:
            continue
        pmid = ""
        for cell in row[19:]:
            c = cell.strip()
            if c.isdigit():
                pmid = c
                break
        out.append(
            {
                "type_a": row[1].strip(),
                "id_a": id_a,
                "db_a": row[3].strip(),
                "type_b": row[5].strip(),
                "id_b": id_b,
                "db_b": row[7].strip(),
                "effect": row[8].strip(),
                "mechanism": row[9].strip(),
                "pmid": pmid,
                "score": row[27].strip() if len(row) > 27 else "",
            }
        )
    return out


def load_reactome_reaction_rows(
    path: str | Path,
) -> dict[str, list[ReactomeReactionRow]]:
    """Load Reactome reaction rows from a TSV.

    Required headers:
    - reaction_id
    - entity_id
    - role
    """
    p = Path(path)
    with p.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if not reader.fieldnames:
            msg = "Reactome TSV is missing a header row."
            raise ValueError(msg)

        _raise_on_duplicate_stripped_headers(reader.fieldnames, source="Reactome TSV")
        cols = {c.strip(): c for c in reader.fieldnames if c is not None}
        required = {"reaction_id", "entity_id", "role"}
        missing = required - set(cols)
        if missing:
            msg = f"Reactome TSV missing required columns: {sorted(missing)}"
            raise ValueError(msg)

        grouped: dict[str, list[ReactomeReactionRow]] = {}
        for row in reader:
            reaction_id = (row.get(cols["reaction_id"]) or "").strip()
            entity_id = (row.get(cols["entity_id"]) or "").strip()
            role = (row.get(cols["role"]) or "").strip()
            if not reaction_id or not entity_id or not role:
                continue
            grouped.setdefault(reaction_id, []).append(
                {"entity_id": entity_id, "role": role}
            )

        out: dict[str, list[ReactomeReactionRow]] = {}
        for reaction_id in sorted(grouped):
            out[reaction_id] = sorted(
                grouped[reaction_id], key=lambda r: (r["role"], r["entity_id"])
            )
        return out


class DrugCombPair(TypedDict):
    """An aggregated DrugComb drug pair (one per unordered resolved node pair)."""

    node_ids: list[str]
    median_zip: float | None
    n_measurements: int
    n_cell_lines: int
    drug_names: list[str]


def load_drugcomb_drug_identifiers(path: str | Path) -> dict[str, str]:
    """Load a DrugComb name -> drugbank_id map (TSV with columns ``name``, ``drugbank_id``).

    The fetch step expands the DrugComb identifiers xlsx into one TSV row per ``dname`` and per
    ``;``-split synonym, each paired with that drug's ``drugbank_id``. Keys are lowercased; rows
    with an empty ``drugbank_id`` are dropped. On a key collision the first non-empty mapping wins.
    """
    p = Path(path)
    out: dict[str, str] = {}
    with p.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if not reader.fieldnames:
            msg = "DrugComb identifiers TSV is missing a header row."
            raise ValueError(msg)
        _raise_on_duplicate_stripped_headers(
            reader.fieldnames, source="DrugComb identifiers TSV"
        )
        cols = {c.strip(): c for c in reader.fieldnames if c is not None}
        required = {"name", "drugbank_id"}
        missing = required - set(cols)
        if missing:
            msg = (
                f"DrugComb identifiers TSV missing required columns: {sorted(missing)}"
            )
            raise ValueError(msg)
        for row in reader:
            name = (row.get(cols["name"]) or "").strip().lower()
            db = (row.get(cols["drugbank_id"]) or "").strip()
            if not name or not db:
                continue
            out.setdefault(name, db)
    return out


def load_drugcomb_pairs(
    summary_csv: str | Path,
    *,
    name_to_drugbank: dict[str, str],
    known_node_ids: set[str] | None = None,
) -> list[DrugCombPair]:
    """Group DrugComb summary rows into one aggregated pair per unordered resolved node pair.

    Resolves ``drug_row`` / ``drug_col`` (lowercased) via ``name_to_drugbank`` to
    ``primekg:drug:<DB#####>``. Rows are skipped when either drug is unresolved, (with
    ``known_node_ids`` given) a node id is absent, or the two resolve to the same node (self-loop).
    Per surviving pair: ``median_zip`` = median of parseable ``synergy_zip`` values (None if none),
    ``n_measurements`` = surviving row count, ``n_cell_lines`` = distinct ``cell_line_name``.
    """
    p = Path(summary_csv)
    groups: dict[tuple[str, str], dict[str, Any]] = {}
    with p.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            msg = "DrugComb summary CSV is missing a header row."
            raise ValueError(msg)
        _raise_on_duplicate_stripped_headers(
            reader.fieldnames, source="DrugComb summary CSV"
        )
        cols = {c.strip(): c for c in reader.fieldnames if c is not None}
        required = {"drug_row", "drug_col", "cell_line_name", "synergy_zip"}
        missing = required - set(cols)
        if missing:
            msg = f"DrugComb summary CSV missing required columns: {sorted(missing)}"
            raise ValueError(msg)

        def resolve(name: str) -> str | None:
            db = name_to_drugbank.get(name.strip().lower())
            if not db:
                return None
            nid = f"primekg:drug:{db}"
            if known_node_ids is not None and nid not in known_node_ids:
                return None
            return nid

        for row in reader:
            a = resolve(row.get(cols["drug_row"]) or "")
            b = resolve(row.get(cols["drug_col"]) or "")
            if a is None or b is None or a == b:
                continue
            key = (a, b) if a < b else (b, a)
            g = groups.setdefault(
                key, {"zips": [], "n": 0, "cells": set(), "names": set()}
            )
            g["n"] += 1
            cell = (row.get(cols["cell_line_name"]) or "").strip()
            if cell:
                g["cells"].add(cell)
            for nm in (row.get(cols["drug_row"]), row.get(cols["drug_col"])):
                if nm and nm.strip():
                    g["names"].add(nm.strip())
            raw_zip = (row.get(cols["synergy_zip"]) or "").strip()
            try:
                g["zips"].append(float(raw_zip))
            except ValueError:
                pass

    out: list[DrugCombPair] = []
    for (a, b), g in groups.items():
        zips = g["zips"]
        out.append(
            {
                "node_ids": sorted([a, b]),
                "median_zip": float(median(zips)) if zips else None,
                "n_measurements": int(g["n"]),
                "n_cell_lines": len(g["cells"]),
                "drug_names": sorted(g["names"]),
            }
        )
    return out


class ClinVarVariant(TypedDict):
    """A clinically-significant ClinVar variant (GRCh38 row)."""

    variation_id: str
    gene_id: str | None
    mondo_numbers: list[str]
    clinical_significance: str
    review_status: str
    # modality (0.8.1) — empty string when the source column/cell is absent
    hgvs_name: str
    variant_type: str
    chromosome: str
    start: str
    stop: str
    ref_allele: str
    alt_allele: str
    rs_id: str


_CLINVAR_SIG_IN_SCOPE = {
    "pathogenic",
    "likely pathogenic",
    "benign",
    "likely benign",
}


def _normalize_mondo_number(token: str) -> str | None:
    """Strip MONDO prefix(es) and return the canonical int-string (leading zeros removed).

    ``"MONDO:MONDO:0013342"`` -> ``"13342"``; ``"8019"`` -> ``"8019"``. Returns None if no digits.
    """
    t = token.strip()
    while t.upper().startswith("MONDO:"):
        t = t[len("MONDO:") :]
    digits = t.strip()
    if not digits.isdigit():
        return None
    return str(int(digits))


def read_mondo_to_primekg_disease(path: str | Path) -> dict[str, str]:
    """Map each MONDO number in `nodes.tab` disease rows -> its `primekg:disease:<node_id>`.

    Disease `node_id` is the MONDO number (single, e.g. ``8019``) or a ``_``-joined grouped string
    (e.g. ``8099_12497_12498``). Each ``_``-split member -> ``primekg:disease:<full node_id>`` so a
    grouped member resolves to its group's node. First occurrence wins.
    """
    p = Path(path)
    out: dict[str, str] = {}
    with p.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if not reader.fieldnames:
            msg = "nodes.tab is missing a header row."
            raise ValueError(msg)
        _raise_on_duplicate_stripped_headers(reader.fieldnames, source="nodes.tab")
        cols = {c.strip(): c for c in reader.fieldnames if c is not None}
        required = {"node_id", "node_type"}
        missing = required - set(cols)
        if missing:
            msg = f"nodes.tab missing required columns: {sorted(missing)}"
            raise ValueError(msg)
        for row in reader:
            ntype = (row.get(cols["node_type"]) or "").strip().strip('"')
            if ntype != "disease":
                continue
            node_id = (row.get(cols["node_id"]) or "").strip().strip('"')
            if not node_id:
                continue
            disease_nid = f"primekg:disease:{node_id}"
            for member in node_id.split("_"):
                norm = _normalize_mondo_number(member)
                if norm and norm not in out:
                    out[norm] = disease_nid
    return out


def load_clinvar_variants(path: str | Path) -> list[ClinVarVariant]:
    """Load clinically-significant GRCh38 ClinVar variants from variant_summary.txt.gz.

    ReviewStatus is optional; when absent, review_status is empty.
    """
    p = Path(path)
    out: list[ClinVarVariant] = []
    with gzip.open(p, "rt", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        try:
            header = next(reader)
        except StopIteration as exc:
            msg = "ClinVar variant_summary is empty."
            raise ValueError(msg) from exc
        header = [h.lstrip("#").strip() for h in header]
        _raise_on_duplicate_stripped_headers(header, source="ClinVar variant_summary")
        idx = {name: i for i, name in enumerate(header)}
        required = {
            "VariationID",
            "GeneID",
            "ClinicalSignificance",
            "PhenotypeIDS",
            "Assembly",
        }
        missing = required - set(idx)
        if missing:
            msg = f"ClinVar variant_summary missing required columns: {sorted(missing)}"
            raise ValueError(msg)
        i_vid, i_gene = idx["VariationID"], idx["GeneID"]
        i_sig, i_pid = idx["ClinicalSignificance"], idx["PhenotypeIDS"]
        i_asm = idx["Assembly"]
        i_rev = idx.get("ReviewStatus")
        i_name = idx.get("Name")
        i_type = idx.get("Type")
        i_chr = idx.get("Chromosome")
        i_start = idx.get("Start")
        i_stop = idx.get("Stop")
        i_ref = idx.get("ReferenceAllele")
        i_alt = idx.get("AlternateAllele")
        i_rs = idx.get("RS# (dbSNP)")

        def _cell(row: list[str], i: int | None) -> str:
            return row[i].strip() if i is not None and len(row) > i else ""

        for row in reader:
            if len(row) <= i_asm or row[i_asm].strip() != "GRCh38":
                continue
            sig = row[i_sig].strip() if len(row) > i_sig else ""
            sig_key = sig.split("/")[0].strip().lower()
            if sig_key not in _CLINVAR_SIG_IN_SCOPE:
                continue
            vid = row[i_vid].strip() if len(row) > i_vid else ""
            if not vid:
                continue
            gene = row[i_gene].strip() if len(row) > i_gene else ""
            gene_id = gene if gene not in ("-1", "") else None
            pid = row[i_pid] if len(row) > i_pid else ""
            mondo: list[str] = []
            for raw_token in pid.replace("|", ",").split(","):
                tok = raw_token.strip()
                if tok.upper().startswith("MONDO:"):
                    norm = _normalize_mondo_number(tok)
                    if norm and norm not in mondo:
                        mondo.append(norm)
            review = _cell(row, i_rev)
            rs_raw = _cell(row, i_rs)
            rs_id = rs_raw if rs_raw not in ("-1", "") else ""
            out.append(
                {
                    "variation_id": vid,
                    "gene_id": gene_id,
                    "mondo_numbers": mondo,
                    "clinical_significance": sig,
                    "review_status": review,
                    "hgvs_name": _cell(row, i_name),
                    "variant_type": _cell(row, i_type),
                    "chromosome": _cell(row, i_chr),
                    "start": _cell(row, i_start),
                    "stop": _cell(row, i_stop),
                    "ref_allele": _cell(row, i_ref),
                    "alt_allele": _cell(row, i_alt),
                    "rs_id": rs_id,
                }
            )
    return out


def load_stitch_rows(path: str | Path) -> list[dict[str, str]]:
    """Read normalized stitch.tsv (header: chemical, protein, combined_score)."""
    p = Path(path)
    with p.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if not reader.fieldnames:
            msg = "STITCH TSV must have a header row"
            raise ValueError(msg)
        _raise_on_duplicate_stripped_headers(reader.fieldnames, source="STITCH TSV")
        cols = {c.strip(): c for c in reader.fieldnames if c is not None}
        required = {"chemical", "protein", "combined_score"}
        missing = required - set(cols)
        if missing:
            msg = f"STITCH TSV missing required columns: {sorted(missing)}"
            raise ValueError(msg)
        out: list[dict[str, str]] = []
        for row in reader:
            chemical = (row.get(cols["chemical"]) or "").strip()
            protein = (row.get(cols["protein"]) or "").strip()
            score = (row.get(cols["combined_score"]) or "").strip()
            if not chemical or not protein:
                continue
            out.append({"chemical": chemical, "protein": protein, "score": score})
        return out


def load_unichem_cid_map(path: str | Path) -> dict[str, dict[str, str | None]]:
    """Read normalized unichem_cid_map.tsv (header: pubchem_cid, drugbank, chebi).

    Returns ``{bare_cid: {"drugbank": <id|None>, "chebi": <id|None>}}``. Empty or
    whitespace-only cells become ``None``.
    """
    p = Path(path)
    out: dict[str, dict[str, str | None]] = {}
    with p.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if not reader.fieldnames:
            msg = "UniChem CID map must have a header row"
            raise ValueError(msg)
        _raise_on_duplicate_stripped_headers(
            reader.fieldnames, source="UniChem CID map"
        )
        cols = {c.strip(): c for c in reader.fieldnames if c is not None}
        required = {"pubchem_cid", "drugbank", "chebi"}
        missing = required - set(cols)
        if missing:
            msg = f"UniChem CID map missing required columns: {sorted(missing)}"
            raise ValueError(msg)
        for row in reader:
            cid = (row.get(cols["pubchem_cid"]) or "").strip()
            if not cid:
                continue
            drugbank = (row.get(cols["drugbank"]) or "").strip() or None
            chebi = (row.get(cols["chebi"]) or "").strip() or None
            out[cid] = {"drugbank": drugbank, "chebi": chebi}
    return out


def load_string_ensp_geneid(path: str | Path) -> dict[str, str]:
    """Read normalized string_ensp_geneid.tsv (header: ensp, geneid)."""
    p = Path(path)
    out: dict[str, str] = {}
    with p.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if not reader.fieldnames:
            msg = "STRING ENSP map must have a header row"
            raise ValueError(msg)
        _raise_on_duplicate_stripped_headers(
            reader.fieldnames, source="STRING ENSP map"
        )
        cols = {c.strip(): c for c in reader.fieldnames if c is not None}
        required = {"ensp", "geneid"}
        missing = required - set(cols)
        if missing:
            msg = f"STRING ENSP map missing required columns: {sorted(missing)}"
            raise ValueError(msg)
        for row in reader:
            ensp = (row.get(cols["ensp"]) or "").strip()
            geneid = (row.get(cols["geneid"]) or "").strip()
            if ensp and geneid:
                out.setdefault(ensp, geneid)
    return out


def load_ctd_rows(path: str | Path) -> list[dict[str, str]]:
    """Read normalized ctd.tsv (header: chemical_mesh, gene_entrez, action, pmid)."""
    p = Path(path)
    with p.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if not reader.fieldnames:
            msg = "CTD TSV must have a header row"
            raise ValueError(msg)
        _raise_on_duplicate_stripped_headers(reader.fieldnames, source="CTD TSV")
        cols = {c.strip(): c for c in reader.fieldnames if c is not None}
        required = {"chemical_mesh", "gene_entrez", "action", "pmid"}
        missing = required - set(cols)
        if missing:
            msg = f"CTD TSV missing required columns: {sorted(missing)}"
            raise ValueError(msg)
        out: list[dict[str, str]] = []
        for row in reader:
            mesh = (row.get(cols["chemical_mesh"]) or "").strip()
            gene = (row.get(cols["gene_entrez"]) or "").strip()
            action = (row.get(cols["action"]) or "").strip()
            pmid = (row.get(cols["pmid"]) or "").strip()
            if not mesh or not gene:
                continue
            out.append(
                {
                    "chemical_mesh": mesh,
                    "gene_entrez": gene,
                    "action": action,
                    "pmid": pmid,
                }
            )
        return out


def load_ctd_mesh_cid(path: str | Path) -> dict[str, str]:
    """Read normalized ctd_mesh_cid.tsv (header: mesh, pubchem_cid)."""
    p = Path(path)
    out: dict[str, str] = {}
    with p.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if not reader.fieldnames:
            msg = "CTD mesh->cid TSV must have a header row"
            raise ValueError(msg)
        _raise_on_duplicate_stripped_headers(
            reader.fieldnames, source="CTD mesh->cid TSV"
        )
        cols = {c.strip(): c for c in reader.fieldnames if c is not None}
        required = {"mesh", "pubchem_cid"}
        missing = required - set(cols)
        if missing:
            msg = f"CTD mesh->cid TSV missing required columns: {sorted(missing)}"
            raise ValueError(msg)
        for row in reader:
            mesh = (row.get(cols["mesh"]) or "").strip()
            cid = (row.get(cols["pubchem_cid"]) or "").strip()
            if mesh and cid:
                out.setdefault(mesh, cid)
    return out


def load_ctd_mesh_pubchem_v2(path: str | Path) -> list[dict[str, str]]:
    """Read lossless ``ctd_mesh_pubchem_v2.tsv`` rows.

    Returns one dict per ``(mesh_id, pubchem_cid)`` with ``source`` /
    ``source_version``. Exact duplicate rows are collapsed; order is
    ``(mesh_id, pubchem_cid)``.
    """
    p = Path(path)
    seen: set[tuple[str, str, str, str]] = set()
    with p.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if not reader.fieldnames:
            msg = "CTD mesh_pubchem_v2 must have a header row"
            raise ValueError(msg)
        _raise_on_duplicate_stripped_headers(
            reader.fieldnames, source="CTD mesh_pubchem_v2"
        )
        cols = {c.strip(): c for c in reader.fieldnames if c is not None}
        required = {"mesh_id", "pubchem_cid", "source", "source_version"}
        missing = required - set(cols)
        if missing:
            msg = f"CTD mesh_pubchem_v2 missing required columns: {sorted(missing)}"
            raise ValueError(msg)
        for row in reader:
            mesh = (row.get(cols["mesh_id"]) or "").strip()
            cid = (row.get(cols["pubchem_cid"]) or "").strip()
            source = (row.get(cols["source"]) or "").strip()
            version = (row.get(cols["source_version"]) or "").strip()
            if mesh and cid:
                seen.add((mesh, cid, source, version))
    return [
        {
            "mesh_id": mesh,
            "pubchem_cid": cid,
            "source": source,
            "source_version": version,
        }
        for mesh, cid, source, version in sorted(seen)
    ]


def index_ctd_mesh_pubchem_v2(
    rows: list[dict[str, str]],
) -> dict[str, list[str]]:
    """Index v2 mesh→CID rows as ``{mesh_id: [cid, ...]}`` (sorted unique CIDs)."""
    out: dict[str, set[str]] = {}
    for row in rows:
        mesh = (row.get("mesh_id") or "").strip()
        cid = (row.get("pubchem_cid") or "").strip()
        if mesh and cid:
            out.setdefault(mesh, set()).add(cid)
    return {mesh: sorted(cids) for mesh, cids in sorted(out.items())}


def load_unichem_pubchem_xref_v2(path: str | Path) -> list[dict[str, str]]:
    """Read lossless long-form ``unichem_pubchem_xref_v2.tsv`` rows."""
    p = Path(path)
    seen: set[tuple[str, str, str, str, str]] = set()
    with p.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if not reader.fieldnames:
            msg = "UniChem pubchem_xref_v2 must have a header row"
            raise ValueError(msg)
        _raise_on_duplicate_stripped_headers(
            reader.fieldnames, source="UniChem pubchem_xref_v2"
        )
        cols = {c.strip(): c for c in reader.fieldnames if c is not None}
        required = {
            "pubchem_cid",
            "namespace",
            "external_id",
            "source",
            "source_version",
        }
        missing = required - set(cols)
        if missing:
            msg = f"UniChem pubchem_xref_v2 missing required columns: {sorted(missing)}"
            raise ValueError(msg)
        for row in reader:
            cid = (row.get(cols["pubchem_cid"]) or "").strip()
            ns = (row.get(cols["namespace"]) or "").strip().lower()
            eid = (row.get(cols["external_id"]) or "").strip()
            source = (row.get(cols["source"]) or "").strip()
            version = (row.get(cols["source_version"]) or "").strip()
            if cid and ns and eid:
                seen.add((cid, ns, eid, source, version))
    return [
        {
            "pubchem_cid": cid,
            "namespace": ns,
            "external_id": eid,
            "source": source,
            "source_version": version,
        }
        for cid, ns, eid, source, version in sorted(seen)
    ]


def index_unichem_pubchem_xref_v2(
    rows: list[dict[str, str]],
) -> dict[str, list[tuple[str, str]]]:
    """Index v2 UniChem rows as ``{cid: [(namespace, external_id), ...]}``."""
    out: dict[str, set[tuple[str, str]]] = {}
    for row in rows:
        cid = (row.get("pubchem_cid") or "").strip()
        ns = (row.get("namespace") or "").strip().lower()
        eid = (row.get("external_id") or "").strip()
        if cid and ns and eid:
            out.setdefault(cid, set()).add((ns, eid))
    return {cid: sorted(pairs) for cid, pairs in sorted(out.items())}
