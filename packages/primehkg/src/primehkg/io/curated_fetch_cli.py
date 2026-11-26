"""CLI: download curated sources and convert to PrimeHKG TSV contracts."""

from __future__ import annotations

import csv
import gzip
import html
import json
import re
import sys
from pathlib import Path
from time import sleep
from typing import TYPE_CHECKING, Any

import typer
from typer.main import get_command

from primehkg.io.build_manifest import sha256_file
from primehkg.io.curated_node_ids import curated_chebi_node_id
from primehkg.io.http_download import download_file, get_json
from primehkg.io.reactome_content_service import resolve_reaction_participants

if TYPE_CHECKING:
    from collections.abc import Iterator

app = typer.Typer(
    rich_markup_mode="rich",
    help="Fetch curated sources (CORUM/Reactome) and write PrimeHKG-ready TSVs.",
)

# These URLs are best-effort defaults; allow overrides on the CLI.
DEFAULT_CORUM_URL = (
    "https://mips.helmholtz-muenchen.de/fastapi-corum/public/file/"
    "download_current_file?file_id=complete&file_format=txt"
)
DEFAULT_REACTOME_REACTION_EXPORTER_URL = (
    "https://reactome.org/download/current/reactome_reaction_exporter.txt"
)
DEFAULT_CHEBI_PE_URL = (
    "https://reactome.org/download/current/ChEBI2Reactome_PE_Reactions.txt"
)
DEFAULT_CHEBI_COMPOUNDS_URL = (
    "https://ftp.ebi.ac.uk/pub/databases/chebi/flat_files/compounds.tsv.gz"
)
DEFAULT_CHEBI_NAMES_URL = (
    "https://ftp.ebi.ac.uk/pub/databases/chebi/flat_files/names.tsv.gz"
)
DEFAULT_REACTOME_CONTENT_SERVICE_BASE = "https://reactome.org/ContentService/data"
DEFAULT_SIGNOR_URL = "https://signor.uniroma2.it/getData.php?organism=9606"
DEFAULT_CLINVAR_URL = (
    "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz"
)
DEFAULT_COMPLEXPORTAL_URL = (
    "https://ftp.ebi.ac.uk/pub/databases/intact/complex/current/complextab/9606.tsv"
)
DEFAULT_DRUGCOMB_SUMMARY_URL = (
    "https://zenodo.org/api/records/11102665/files/summary_table_v1.4.csv/content"
)
DEFAULT_DRUGCOMB_IDENTIFIERS_URL = (
    "https://zenodo.org/api/records/18449193/files/"
    "DrugComb_drug_identifiers.xlsx/content"
)
DEFAULT_STITCH_LINKS_URL = (
    "http://stitch.embl.de/download/protein_chemical.links.v5.0/"
    "9606.protein_chemical.links.v5.0.tsv.gz"
)
DEFAULT_UNICHEM_DRUGBANK_URL = (
    "https://ftp.ebi.ac.uk/pub/databases/chembl/UniChem/data/"
    "wholeSourceMapping/src_id2/src2src22.txt.gz"
)
DEFAULT_UNICHEM_CHEBI_URL = (
    "https://ftp.ebi.ac.uk/pub/databases/chembl/UniChem/data/"
    "wholeSourceMapping/src_id7/src7src22.txt.gz"
)
DEFAULT_STRING_ALIASES_URL = (
    "https://stringdb-downloads.org/download/protein.aliases.v12.0/"
    "9606.protein.aliases.v12.0.txt.gz"
)
DEFAULT_CTD_CHEM_GENE_URL = "https://ctdbase.org/reports/CTD_chem_gene_ixns.tsv.gz"
DEFAULT_CTD_CHEMICALS_URL = "https://ctdbase.org/reports/CTD_chemicals.tsv.gz"

OUT_DIR_OPT = typer.Option(
    ...,
    "--out-dir",
    file_okay=False,
    dir_okay=True,
    help="Output directory for converted TSVs and sources.json.",
)
CORUM_URL_OPT = typer.Option(
    DEFAULT_CORUM_URL,
    "--corum-url",
    help="CORUM download URL (allComplexes.txt or equivalent).",
)
REACTOME_PROTEIN_ROLE_URL_OPT = typer.Option(
    DEFAULT_REACTOME_REACTION_EXPORTER_URL,
    "--reactome-protein-role-url",
    help="Reactome reaction exporter URL (default: reactome_reaction_exporter.txt).",
)
CHEBI_PE_URL_OPT = typer.Option(
    DEFAULT_CHEBI_PE_URL,
    "--chebi-pe-url",
    help="ChEBI2Reactome_PE_Reactions.txt URL (used by --reactome-mode exporter-chebi).",
)
CHEBI_COMPOUNDS_URL_OPT = typer.Option(
    DEFAULT_CHEBI_COMPOUNDS_URL,
    "--chebi-compounds-url",
    help="ChEBI compounds.tsv.gz URL (EBI flat-file; emits chebi_names.tsv).",
)
CHEBI_NAMES_URL_OPT = typer.Option(
    DEFAULT_CHEBI_NAMES_URL,
    "--chebi-names-url",
    help="ChEBI names.tsv.gz URL (EBI flat-file; emits chebi_synonyms.tsv).",
)
CHEBI_NAMES_VERSION_OPT = typer.Option(
    "chebi:unknown",
    "--chebi-names-version",
    help=(
        "ChEBI release pin recorded in sources.json and chebi_definitions_meta.json "
        "(e.g. chebi:235). Pass a concrete release id — not chebi:unknown — before "
        "promoting chebi/definition to allow in text_provenance_registry.yaml."
    ),
)
CORUM_VERSION_OPT = typer.Option(
    "unknown",
    "--corum-version",
    help="Version string recorded in sources.json (and passed to primehkg-build).",
)
REACTOME_VERSION_OPT = typer.Option(
    "unknown",
    "--reactome-version",
    help="Version string recorded in sources.json (and passed to primehkg-build).",
)
TIMEOUT_S_OPT = typer.Option(
    60.0,
    "--timeout-s",
    min=1.0,
    help="Per-download timeout in seconds.",
)
INSECURE_SSL_OPT = typer.Option(
    False,
    "--insecure-ssl",
    help="Disable SSL certificate verification for downloads (use only when necessary).",
)
REACTOME_MODE_OPT = typer.Option(
    "exporter",
    "--reactome-mode",
    help=(
        "Reactome conversion mode: exporter (fast) or content-service (includes ChEBI roles; slow)"
        " or exporter-chebi (bulk protein roles + ChEBI participants, no API)."
    ),
)
REACTOME_MAX_REACTIONS_OPT = typer.Option(
    2000,
    "--reactome-max-reactions",
    min=1,
    help="Safety cap for content-service mode: max reactions to query.",
)
REACTOME_CONTENT_SERVICE_BASE_OPT = typer.Option(
    DEFAULT_REACTOME_CONTENT_SERVICE_BASE,
    "--reactome-content-service-base",
    help="Base URL for Reactome ContentService (advanced).",
)
REACTOME_SLEEP_MS_OPT = typer.Option(
    50,
    "--reactome-sleep-ms",
    min=0,
    help="Sleep between ContentService requests (ms). Default is conservative to avoid rate limits.",
)
ALLOW_HTTP_CONTENT_SERVICE_OPT = typer.Option(
    False,
    "--allow-http-content-service",
    help="Allow http:// for ContentService base URL (discouraged).",
)
SIGNOR_URL_OPT = typer.Option(
    DEFAULT_SIGNOR_URL,
    "--signor-url",
    help="SIGNOR human regulation TSV URL.",
)
SIGNOR_VERSION_OPT = typer.Option(
    "signor:unknown",
    "--signor-version",
    help="Version string recorded for SIGNOR in sources.json.",
)
COMPLEXPORTAL_URL_OPT = typer.Option(
    DEFAULT_COMPLEXPORTAL_URL,
    "--complexportal-url",
    help="Complex Portal human complexes TSV URL.",
)
COMPLEXPORTAL_VERSION_OPT = typer.Option(
    "complexportal:unknown",
    "--complexportal-version",
    help="Version string recorded for Complex Portal in sources.json.",
)
DRUGCOMB_SUMMARY_URL_OPT = typer.Option(
    DEFAULT_DRUGCOMB_SUMMARY_URL,
    "--drugcomb-summary-url",
    help="DrugComb summary measurements CSV URL (Zenodo).",
)
DRUGCOMB_IDENTIFIERS_URL_OPT = typer.Option(
    DEFAULT_DRUGCOMB_IDENTIFIERS_URL,
    "--drugcomb-identifiers-url",
    help="DrugComb drug identifiers xlsx URL (Zenodo).",
)
DRUGCOMB_VERSION_OPT = typer.Option(
    "drugcomb:unknown",
    "--drugcomb-version",
    help="Version string recorded for DrugComb in sources.json.",
)
CLINVAR_URL_OPT = typer.Option(
    DEFAULT_CLINVAR_URL,
    "--clinvar-url",
    help="ClinVar variant_summary.txt.gz URL (NCBI FTP).",
)
CLINVAR_VERSION_OPT = typer.Option(
    "clinvar:unknown",
    "--clinvar-version",
    help="Version string recorded for ClinVar in sources.json.",
)
STITCH_LINKS_URL_OPT = typer.Option(
    DEFAULT_STITCH_LINKS_URL,
    "--stitch-links-url",
    help="STITCH 9606 protein-chemical links.tsv.gz URL (emits stitch.tsv).",
)
UNICHEM_DRUGBANK_URL_OPT = typer.Option(
    DEFAULT_UNICHEM_DRUGBANK_URL,
    "--unichem-drugbank-url",
    help="UniChem DrugBank<->PubChem (src2src22.txt.gz) URL.",
)
UNICHEM_CHEBI_URL_OPT = typer.Option(
    DEFAULT_UNICHEM_CHEBI_URL,
    "--unichem-chebi-url",
    help="UniChem ChEBI<->PubChem (src7src22.txt.gz) URL.",
)
STRING_ALIASES_URL_OPT = typer.Option(
    DEFAULT_STRING_ALIASES_URL,
    "--string-aliases-url",
    help="STRING 9606 protein.aliases.txt.gz URL (emits string_ensp_geneid.tsv).",
)
CTD_CHEM_GENE_URL_OPT = typer.Option(
    DEFAULT_CTD_CHEM_GENE_URL,
    "--ctd-chem-gene-url",
    help="CTD chem-gene interactions URL.",
)
CTD_CHEMICALS_URL_OPT = typer.Option(
    DEFAULT_CTD_CHEMICALS_URL,
    "--ctd-chemicals-url",
    help="CTD chemicals (MeSH->PubChem CID) URL.",
)


def _convert_drugcomb_identifiers_xlsx(xlsx_path: Path, out_tsv: Path) -> None:
    """Expand the DrugComb identifiers xlsx into a (name, drugbank_id) TSV.

    One row per ``dname`` and per ``;``-split ``synonyms`` entry, paired with the row's
    ``drugbank_id``. Rows without a DB##### drugbank_id are skipped. ``openpyxl`` is a fetch-only
    dependency (not required by the builder/runtime).
    """
    import openpyxl

    wb = openpyxl.load_workbook(xlsx_path, read_only=True)
    try:
        ws = wb[wb.sheetnames[0]]
        rows = ws.iter_rows(values_only=True)
        header = [str(c).strip() if c is not None else "" for c in next(rows)]
        idx = {name: i for i, name in enumerate(header)}
        i_dn, i_db, i_syn = (
            idx.get("dname"),
            idx.get("drugbank_id"),
            idx.get("synonyms"),
        )
        if i_dn is None or i_db is None:
            msg = "DrugComb identifiers xlsx missing 'dname' or 'drugbank_id' column"
            raise ValueError(msg)
        seen: set[tuple[str, str]] = set()
        lines = ["name\tdrugbank_id"]
        for row in rows:
            db = row[i_db] if i_db < len(row) else None
            db = str(db).strip() if db is not None else ""
            if not db or not db.startswith("DB"):
                continue
            names: list[str] = []
            dn = row[i_dn] if i_dn < len(row) else None
            if dn:
                names.append(str(dn).strip())
            if i_syn is not None and i_syn < len(row) and row[i_syn]:
                names.extend(s.strip() for s in str(row[i_syn]).split(";"))
            for nm in names:
                if nm and (nm, db) not in seen:
                    seen.add((nm, db))
                    lines.append(f"{nm}\t{db}")
        out_tsv.write_text("\n".join(lines) + "\n", encoding="utf-8")
    finally:
        wb.close()


def _argv_for_cli(argv: list[str] | None) -> list[str]:
    a = list(argv) if argv is not None else list(sys.argv[1:])
    if a and a[0] == "run":
        return a[1:]
    return a


def _write_sources_json(out_dir: Path, sources: list[dict[str, Any]]) -> None:
    (out_dir / "sources.json").write_text(
        json.dumps(sources, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _convert_corum_all_complexes_to_tsv(src_path: Path, out_tsv: Path) -> None:
    """Convert CORUM allComplexes.txt to PrimeHKG `corum.tsv`.

    We only need: complex_id, uniprot_id, complex_name.
    """
    # CORUM formats vary by release; we support a pragmatic subset:
    # - complex id column often named "ComplexID" or "complex_id"
    # - subunit UniProt column often contains semicolon-delimited accessions
    # - complex name column "ComplexName" or similar
    raw = src_path.read_text(encoding="utf-8", errors="replace").splitlines()
    reader = csv.DictReader(raw, delimiter="\t")
    if not reader.fieldnames:
        msg = "CORUM download is missing a header row."
        raise ValueError(msg)

    cols = {c.strip().lower(): c for c in reader.fieldnames if c is not None}
    cid_col = cols.get("complexid") or cols.get("complex_id") or cols.get("complex id")
    name_col = (
        cols.get("complexname") or cols.get("complex_name") or cols.get("complex name")
    )
    sub_col = (
        cols.get("subunits (uniprot ids)")
        or cols.get("subunits(uniprot ids)")
        or cols.get("subunits_uniprot_id")
        or cols.get("subunits")
        or cols.get("uniprot_id")
        or cols.get("uniprot ids")
    )
    if not cid_col or not sub_col:
        msg = "CORUM download missing required columns for conversion."
        raise ValueError(msg)

    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    with out_tsv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["complex_id", "uniprot_id", "complex_name"])
        for row in reader:
            cid = (row.get(cid_col) or "").strip()
            if not cid:
                continue
            name = (row.get(name_col) or "").strip() if name_col else ""
            sub = (row.get(sub_col) or "").strip()
            if not sub:
                continue
            # Common CORUM delimiter for multiple UniProt IDs is ';'
            for uid in [p.strip() for p in sub.replace(",", ";").split(";")]:
                if uid:
                    w.writerow([cid, uid, name])


def _iter_reactome_rows(src_path: Path) -> list[tuple[str, str, str]]:
    """Read a Reactome export and emit (reaction_id, entity_id, role) rows.

    Supports two formats:
    - `reactome_reaction_exporter.txt` (preferred): header includes `reaction_id`, `uniprot_acc`,
      `role_in_reaction` with a list like `[input, output]`.
    - `ProteinRoleReaction.txt` (fallback): `uniprot`, `role`, `reaction_id` per row (no header).
    """
    raw = src_path.read_text(encoding="utf-8", errors="replace").splitlines()
    if not raw:
        return []

    if raw[0].startswith("pathway_id\treaction_id\t"):
        reader = csv.DictReader(raw, delimiter="\t")
        out: list[tuple[str, str, str]] = []
        for row in reader:
            reaction_id = (row.get("reaction_id") or "").strip()
            uniprot = (row.get("uniprot_acc") or "").strip()
            roles = (row.get("role_in_reaction") or "").strip()
            if not reaction_id or not uniprot or not roles:
                continue
            cleaned = roles.strip().strip("[]")
            for role in [p.strip() for p in cleaned.split(",") if p.strip()]:
                role_norm = role.lower().replace(" ", "_")
                if role_norm == "positive":
                    role_norm = "positive_regulator"
                elif role_norm == "negative":
                    role_norm = "negative_regulator"
                out.append((reaction_id, uniprot, role_norm))
        return out

    reader = csv.reader(raw, delimiter="\t")
    out = []
    for row in reader:
        if len(row) < 3:
            continue
        uniprot, role, reaction_id = (row[0].strip(), row[1].strip(), row[2].strip())
        if not uniprot or not role or not reaction_id:
            continue
        out.append((reaction_id, uniprot, role.strip().lower().replace(" ", "_")))
    return out


def _iter_chebi_pe_rows(src_path: Path) -> list[tuple[str, str, str]]:
    """Read ChEBI2Reactome_PE_Reactions.txt -> (reaction_id, "chebi:<id>", "participant").

    Tab-separated, NO header. Columns:
      [0] ChEBI_id  [1] PE_stId  [2] PE_name  [3] reaction_stId  [4] url
      [5] reaction_name  [6] evidence  [7] species
    Keeps only species == "Homo sapiens" and reaction_id "R-HSA-<digits>". The ChEBI
    node_id is produced via ``curated_chebi_node_id``; role is always "participant".
    """
    raw = src_path.read_text(encoding="utf-8", errors="replace").splitlines()
    out: list[tuple[str, str, str]] = []
    reader = csv.reader(raw, delimiter="\t")
    for row in reader:
        if len(row) < 8:
            continue
        chebi_id = row[0].strip()
        reaction_id = row[3].strip()
        species = row[7].strip()
        if not chebi_id or species != "Homo sapiens":
            continue
        local = reaction_id.removeprefix("R-HSA-")
        if local == reaction_id or not local.isdigit():
            continue
        out.append(
            (reaction_id, curated_chebi_node_id(f"CHEBI:{chebi_id}"), "participant")
        )
    return out


_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def _clean_chebi_definition(raw: str) -> str:
    if not raw:
        return ""
    text = html.unescape(raw)
    text = _HTML_TAG_RE.sub("", text)
    text = _WS_RE.sub(" ", text).strip()
    if not text or text.lower() == "null":
        return ""
    return text


def _iter_chebi_compound_rows(src_path: Path) -> list[tuple[str, str, str]]:
    """Read ChEBI compounds.tsv.gz -> (chebi_node_id, name, definition) rows.

    Tab-separated WITH a lowercase header. The real EBI schema columns are
    ``id name status_id source parent_id merge_type chebi_accession definition
    ascii_name stars modified_on release_date``. The ``id`` column is a bare
    integer; the ``CHEBI:<n>``-prefixed accession lives in ``chebi_accession``,
    which we use to build the node id via ``curated_chebi_node_id``. Rows with an
    empty ``chebi_accession``, empty ``name``, or a ``name`` literally equal to
    ``"null"`` are skipped. The ``definition`` column is optional — if absent,
    definitions come out as empty strings.
    """
    with gzip.open(src_path, "rt", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f, delimiter="\t")
        fieldnames = reader.fieldnames or []
        header = {h.strip(): h for h in fieldnames if h is not None}
        name_col = header.get("name")
        accession_col = header.get("chebi_accession")
        definition_col = header.get("definition")
        missing = [
            col
            for col, resolved in (
                ("name", name_col),
                ("chebi_accession", accession_col),
            )
            if resolved is None
        ]
        if missing:
            msg = f"ChEBI compounds.tsv missing required columns: {missing}"
            raise ValueError(msg)
        out: list[tuple[str, str, str]] = []
        for row in reader:
            accession = (row.get(accession_col) or "").strip()
            name = (row.get(name_col) or "").strip()
            if not accession or not name or name.lower() == "null":
                continue
            definition_raw = (row.get(definition_col) or "") if definition_col else ""
            definition = _clean_chebi_definition(definition_raw)
            out.append((curated_chebi_node_id(accession), name, definition))
        return out


def _convert_chebi_compounds_to_tsv(src_path: Path, out_tsv: Path) -> None:
    """Write chebi_names.tsv (header + deduped, node_id-sorted rows).

    When the dictionary has multiple rows per accession, keep the first NAME
    in node_id-sorted, name-sorted order (deterministic).
    """
    by_id: dict[str, str] = {}
    for node_id, name, _definition in sorted(_iter_chebi_compound_rows(src_path)):
        by_id.setdefault(node_id, name)
    if not by_id:
        msg = f"ChEBI compounds file {src_path} yielded 0 (id, name) rows — likely a bad/empty download or schema change"
        raise ValueError(msg)
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    with out_tsv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["chebi_node_id", "name"])
        for node_id in sorted(by_id):
            w.writerow([node_id, by_id[node_id]])


_CHEBI_SYNONYM_TYPES = frozenset({"SYNONYM", "IUPAC NAME"})


def _clean_chebi_surface(name: str, ascii_name: str) -> str:
    raw = (ascii_name or name or "").strip()
    if not raw or raw.lower() == "null":
        return ""
    if "<" in raw:
        raw = _HTML_TAG_RE.sub("", html.unescape(raw))
        raw = _WS_RE.sub(" ", raw).strip()
    return raw


def _convert_chebi_names_gz_to_synonyms_tsv(
    names_gz: Path, compounds_gz: Path, out_tsv: Path
) -> None:
    id_to_node: dict[str, str] = {}
    with gzip.open(compounds_gz, "rt", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            cid = (row.get("id") or "").strip()
            acc = (row.get("chebi_accession") or "").strip()
            if cid and acc:
                id_to_node[cid] = curated_chebi_node_id(acc)
    rows: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    with gzip.open(names_gz, "rt", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if (row.get("type") or "").strip() not in _CHEBI_SYNONYM_TYPES:
                continue
            if (row.get("language_code") or "").strip() != "en":
                continue
            cid = (row.get("compound_id") or "").strip()
            node_id = id_to_node.get(cid)
            if not node_id:
                continue
            surface = _clean_chebi_surface(
                row.get("name") or "", row.get("ascii_name") or ""
            )
            if not surface:
                continue
            name_type = (row.get("type") or "").strip()
            key = (node_id, surface, name_type)
            if key in seen:
                continue
            seen.add(key)
            rows.append(key)
    if not rows:
        msg = (
            f"ChEBI names file {names_gz} yielded 0 synonym rows — "
            "likely a bad/empty download or schema change"
        )
        raise ValueError(msg)
    rows.sort()
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    with out_tsv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["chebi_node_id", "surface", "name_type"])
        for node_id, surface, name_type in rows:
            w.writerow([node_id, surface, name_type])


def _convert_chebi_definitions_to_tsv(src_path: Path, out_tsv: Path) -> None:
    """Write chebi_definitions.tsv (header + deduped, node_id-sorted rows).

    Only rows whose cleaned definition is non-empty are written. When the
    dictionary has multiple rows per accession, keep the first definition in
    node_id-sorted order (deterministic).
    """
    by_id: dict[str, str] = {}
    for node_id, _name, definition in sorted(_iter_chebi_compound_rows(src_path)):
        if definition:
            by_id.setdefault(node_id, definition)
    if not by_id:
        msg = (
            f"ChEBI compounds file {src_path} yielded 0 (id, definition) rows "
            "— likely a bad/empty download or schema change"
        )
        raise ValueError(msg)
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    with out_tsv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["chebi_node_id", "definition"])
        for node_id in sorted(by_id):
            w.writerow([node_id, by_id[node_id]])


def _write_chebi_definitions_meta(
    *,
    out_meta: Path,
    definitions_tsv: Path,
    chebi_names_version: str,
    compounds_url: str,
    downloaded_at: str,
) -> None:
    """Write chebi_definitions_meta.json alongside chebi_definitions.tsv."""
    meta = {
        "upstream_source": "ChEBI",
        "upstream_release": chebi_names_version,
        "retrieved_at": downloaded_at[:10],
        "source_url": compounds_url,
        "sha256": sha256_file(definitions_tsv),
        "downloaded_at": downloaded_at,
    }
    out_meta.parent.mkdir(parents=True, exist_ok=True)
    out_meta.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _ctd_reader(src_path: Path) -> Iterator[dict[str, str]]:
    """Yield CTD data rows as dicts keyed by the '# <columns>' header line.

    CTD report files prefix metadata/comment lines with ``#``; the LAST such line
    before the data is the tab-separated column-name row. Data rows are not
    ``#``-prefixed.
    """
    with gzip.open(src_path, "rt", encoding="utf-8", errors="replace") as f:
        header: list[str] | None = None
        for line in f:
            if line.startswith("#"):
                stripped = line.lstrip("#").strip()
                if "\t" in stripped:
                    header = [c.strip() for c in stripped.split("\t")]
                continue
            if header is None:
                msg = "CTD file: no '# <columns>' header line found before data"
                raise ValueError(msg)
            parts = line.rstrip("\n").split("\t")
            yield dict(zip(header, parts, strict=False))


def _convert_ctd_chem_gene_to_tsv(src_path: Path, out_tsv: Path) -> None:
    """Normalize CTD_chem_gene_ixns.gz -> ctd.tsv (chemical_mesh, gene_entrez, action, pmid), human only."""
    rows: list[tuple[str, str, str, str]] = []
    for row in _ctd_reader(src_path):
        if (
            not {
                "ChemicalID",
                "GeneID",
                "OrganismID",
                "InteractionActions",
                "PubMedIDs",
            }
            <= row.keys()
        ):
            msg = f"CTD chem_gene header missing expected columns; got {sorted(row)}"
            raise ValueError(msg)
        if (row.get("OrganismID") or "").strip() != "9606":
            continue
        mesh = (row.get("ChemicalID") or "").strip().removeprefix("MESH:")
        gene = (row.get("GeneID") or "").strip()
        action = (row.get("InteractionActions") or "").strip()
        pmid = (row.get("PubMedIDs") or "").strip()
        if not mesh or not gene or not action:
            continue
        rows.append((mesh, gene, action, pmid))
    if not rows:
        msg = f"CTD chem_gene file {src_path} yielded 0 rows — likely a bad/empty download or schema change"
        raise ValueError(msg)
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    with out_tsv.open("w", encoding="utf-8", newline="") as g:
        w = csv.writer(g, delimiter="\t")
        w.writerow(["chemical_mesh", "gene_entrez", "action", "pmid"])
        for r in rows:
            w.writerow(list(r))


def _convert_ctd_chemicals_to_mesh_cid_tsv(src_path: Path, out_tsv: Path) -> None:
    """Normalize CTD_chemicals.gz -> mesh,pubchem_cid (MESH:/CID: stripped; CID-less skipped).

    Scalar / first-win contract kept for CTD builders. Prefer
    :func:`_convert_ctd_chemicals_to_mesh_pubchem_v2_tsv` for lossless consumers.
    """
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    by_mesh: dict[str, str] = {}
    for row in _ctd_reader(src_path):
        if not {"ChemicalID", "PubChemCID"} <= row.keys():
            msg = f"CTD chemicals header missing expected columns; got {sorted(row)}"
            raise ValueError(msg)
        mesh = (row.get("ChemicalID") or "").strip().removeprefix("MESH:")
        cid = (row.get("PubChemCID") or "").strip().removeprefix("CID:")
        if mesh and cid:
            by_mesh.setdefault(mesh, cid)
    if not by_mesh:
        msg = f"CTD chemicals file {src_path} yielded 0 mesh->cid rows — likely a bad/empty download or schema change"
        raise ValueError(msg)
    with out_tsv.open("w", encoding="utf-8", newline="") as g:
        w = csv.writer(g, delimiter="\t")
        w.writerow(["mesh", "pubchem_cid"])
        for mesh in sorted(by_mesh):
            w.writerow([mesh, by_mesh[mesh]])


def _convert_ctd_chemicals_to_mesh_pubchem_v2_tsv(
    src_path: Path, out_tsv: Path, *, source_version: str
) -> None:
    """Lossless CTD chemicals → ``ctd_mesh_pubchem_v2.tsv`` (one MeSH x CID per row).

    Keeps every distinct ``(mesh_id, pubchem_cid)`` pair; exact duplicates only are
    collapsed. No first-win / ``setdefault`` selection.
    """
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    pairs: set[tuple[str, str]] = set()
    for row in _ctd_reader(src_path):
        if not {"ChemicalID", "PubChemCID"} <= row.keys():
            msg = f"CTD chemicals header missing expected columns; got {sorted(row)}"
            raise ValueError(msg)
        mesh = (row.get("ChemicalID") or "").strip().removeprefix("MESH:")
        cid = (row.get("PubChemCID") or "").strip().removeprefix("CID:")
        if mesh and cid:
            pairs.add((mesh, cid))
    if not pairs:
        msg = (
            f"CTD chemicals file {src_path} yielded 0 mesh->cid rows — "
            "likely a bad/empty download or schema change"
        )
        raise ValueError(msg)
    version = (source_version or "").strip() or "ctd:unknown"
    with out_tsv.open("w", encoding="utf-8", newline="") as g:
        w = csv.writer(g, delimiter="\t")
        w.writerow(["mesh_id", "pubchem_cid", "source", "source_version"])
        for mesh, cid in sorted(pairs):
            w.writerow([mesh, cid, "CTD", version])


def _convert_stitch_links_to_tsv(src_path: Path, out_tsv: Path) -> None:
    """Normalize STITCH 9606 links.gz -> stitch.tsv (chemical, protein, combined_score)."""
    rows: list[tuple[str, str, str]] = []
    with gzip.open(src_path, "rt", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f, delimiter="\t")
        fns = reader.fieldnames or []
        header = {h.strip(): h for h in fns if h is not None}
        for col in ("chemical", "protein", "combined_score"):
            if col not in header:
                msg = f"STITCH links missing column {col!r}; got {sorted(header)}"
                raise ValueError(msg)
        for row in reader:
            c = (row.get(header["chemical"]) or "").strip()
            p = (row.get(header["protein"]) or "").strip()
            s = (row.get(header["combined_score"]) or "").strip()
            if c and p and s:
                rows.append((c, p, s))
    if not rows:
        msg = f"STITCH links file {src_path} yielded 0 rows — likely a bad/empty download or schema change"
        raise ValueError(msg)
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    with out_tsv.open("w", encoding="utf-8", newline="") as g:
        w = csv.writer(g, delimiter="\t")
        w.writerow(["chemical", "protein", "combined_score"])
        for c, p, s in rows:
            w.writerow([c, p, s])


def _convert_unichem_pair_to_cid_map_tsv(
    src_path: Path, out_tsv: Path, *, kind: str
) -> None:
    """Normalize a UniChem pair file into a partial cid map TSV.

    Reads ``srcN <-> src22`` (PubChem) and writes (pubchem_cid, drugbank, chebi)
    filling only the ``kind`` column. ``kind`` is 'drugbank' or 'chebi'. The
    PubChem column is the header containing '22'.
    """
    if kind not in ("drugbank", "chebi"):
        msg = f"kind must be drugbank|chebi, got {kind!r}"
        raise ValueError(msg)
    with gzip.open(src_path, "rt", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        if header is None or len(header) < 2:
            msg = "UniChem pair file must have a 2-column header"
            raise ValueError(msg)
        pubchem_idx = next((i for i, h in enumerate(header) if "22" in h), None)
        if pubchem_idx is None:
            msg = f"UniChem pair file: no PubChem (src22) column in header {header}"
            raise ValueError(msg)
        other_idx = (
            1 - pubchem_idx if len(header) == 2 else (0 if pubchem_idx == 1 else 1)
        )
        by_cid: dict[str, str] = {}
        for row in reader:
            if len(row) <= max(pubchem_idx, other_idx):
                continue
            cid = row[pubchem_idx].strip()
            other = row[other_idx].strip()
            if cid and other:
                by_cid.setdefault(cid, other)
    if not by_cid:
        msg = f"UniChem pair file {src_path} yielded 0 rows — likely a bad/empty download or schema change"
        raise ValueError(msg)
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    with out_tsv.open("w", encoding="utf-8", newline="") as g:
        w = csv.writer(g, delimiter="\t")
        w.writerow(["pubchem_cid", "drugbank", "chebi"])
        for cid in sorted(by_cid):
            db = by_cid[cid] if kind == "drugbank" else ""
            ch = by_cid[cid] if kind == "chebi" else ""
            w.writerow([cid, db, ch])


def _merge_unichem_cid_maps(drugbank_tsv: Path, chebi_tsv: Path, out_tsv: Path) -> None:
    """Merge the two partial cid maps into one (pubchem_cid, drugbank, chebi).

    Scalar / first-win contract kept for STITCH/CTD builders. Prefer
    :func:`_convert_unichem_pairs_to_pubchem_xref_v2_tsv` for lossless consumers.
    """
    merged: dict[str, dict[str, str]] = {}
    for path, key in ((drugbank_tsv, "drugbank"), (chebi_tsv, "chebi")):
        with path.open(encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                cid = (row.get("pubchem_cid") or "").strip()
                val = (row.get(key) or "").strip()
                if cid and val:
                    merged.setdefault(cid, {"drugbank": "", "chebi": ""})[key] = val
    if not merged:
        msg = (
            f"UniChem cid maps {drugbank_tsv} + {chebi_tsv} merged to 0 rows — "
            "likely a bad/empty download or schema change"
        )
        raise ValueError(msg)
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    with out_tsv.open("w", encoding="utf-8", newline="") as g:
        w = csv.writer(g, delimiter="\t")
        w.writerow(["pubchem_cid", "drugbank", "chebi"])
        for cid in sorted(merged):
            w.writerow([cid, merged[cid]["drugbank"], merged[cid]["chebi"]])


def _iter_unichem_pair_rows(
    src_path: Path, *, namespace: str
) -> Iterator[tuple[str, str, str]]:
    """Yield ``(pubchem_cid, namespace, bare_external_id)`` from a UniChem pair file."""
    if namespace not in ("drugbank", "chebi"):
        msg = f"namespace must be drugbank|chebi, got {namespace!r}"
        raise ValueError(msg)
    with gzip.open(src_path, "rt", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        if header is None or len(header) < 2:
            msg = "UniChem pair file must have a 2-column header"
            raise ValueError(msg)
        pubchem_idx = next((i for i, h in enumerate(header) if "22" in h), None)
        if pubchem_idx is None:
            msg = f"UniChem pair file: no PubChem (src22) column in header {header}"
            raise ValueError(msg)
        other_idx = (
            1 - pubchem_idx if len(header) == 2 else (0 if pubchem_idx == 1 else 1)
        )
        for row in reader:
            if len(row) <= max(pubchem_idx, other_idx):
                continue
            cid = row[pubchem_idx].strip()
            other = row[other_idx].strip()
            if not cid or not other:
                continue
            if namespace == "chebi":
                other = other.removeprefix("CHEBI:").removeprefix("chebi:")
            yield cid, namespace, other


def _convert_unichem_pairs_to_pubchem_xref_v2_tsv(
    drugbank_src: Path,
    chebi_src: Path,
    out_tsv: Path,
    *,
    source_version: str,
) -> None:
    """Lossless UniChem pairs → long-form ``unichem_pubchem_xref_v2.tsv``."""
    triples: set[tuple[str, str, str]] = set()
    for path, ns in ((drugbank_src, "drugbank"), (chebi_src, "chebi")):
        for triple in _iter_unichem_pair_rows(path, namespace=ns):
            triples.add(triple)
    if not triples:
        msg = (
            f"UniChem pair files {drugbank_src} + {chebi_src} yielded 0 rows — "
            "likely a bad/empty download or schema change"
        )
        raise ValueError(msg)
    version = (source_version or "").strip() or "unichem:unknown"
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    with out_tsv.open("w", encoding="utf-8", newline="") as g:
        w = csv.writer(g, delimiter="\t")
        w.writerow(
            ["pubchem_cid", "namespace", "external_id", "source", "source_version"]
        )
        for cid, ns, eid in sorted(triples):
            w.writerow([cid, ns, eid, "UniChem", version])


def _convert_string_aliases_to_ensp_geneid_tsv(src_path: Path, out_tsv: Path) -> None:
    """Normalize STRING 9606 aliases.gz -> ensp,geneid (UniProt_DR_GeneID rows only)."""
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    seen: dict[str, str] = {}
    with gzip.open(src_path, "rt", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)  # header line (#string_protein_id ...)
        for row in reader:
            if len(row) < 3:
                continue
            ensp, alias, source = row[0].strip(), row[1].strip(), row[2].strip()
            if source == "UniProt_DR_GeneID" and ensp and alias:
                seen.setdefault(ensp, alias)
    if not seen:
        msg = f"STRING aliases file {src_path} yielded 0 rows — likely a bad/empty download or schema change"
        raise ValueError(msg)
    with out_tsv.open("w", encoding="utf-8", newline="") as g:
        w = csv.writer(g, delimiter="\t")
        w.writerow(["ensp", "geneid"])
        for ensp in sorted(seen):
            w.writerow([ensp, seen[ensp]])


def _convert_reactome_to_tsv(src_path: Path, out_tsv: Path) -> None:
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    with out_tsv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["reaction_id", "entity_id", "role"])
        for reaction_id, entity_id, role in _iter_reactome_rows(src_path):
            w.writerow([reaction_id, entity_id, role])


def _write_reactome_rows(rows: list[tuple[str, str, str]], out_tsv: Path) -> None:
    """Write (reaction_id, entity_id, role) rows as reactome.tsv (header + sorted-unique rows)."""
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    with out_tsv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["reaction_id", "entity_id", "role"])
        for reaction_id, entity_id, role in sorted(set(rows)):
            w.writerow([reaction_id, entity_id, role])


def _convert_reactome_content_service_to_tsv(
    *,
    exporter_path: Path,
    out_tsv: Path,
    base_url: str,
    timeout_s: float,
    max_reactions: int,
    sleep_ms: int,
    allow_http: bool,
    insecure_ssl: bool,
) -> None:
    reaction_ids: list[str] = sorted(
        {rid for rid, _, _ in _iter_reactome_rows(exporter_path)}
    )
    reaction_ids = reaction_ids[: max(0, int(max_reactions))]

    base = base_url.rstrip("/")

    def _fetch(key: str) -> object:
        return get_json(
            url=f"{base}/query/{key}",
            timeout_s=timeout_s,
            allow_http=allow_http,
            insecure_ssl=bool(insecure_ssl),
        )

    entity_cache: dict[str, list[str]] = {}

    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    with out_tsv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["reaction_id", "entity_id", "role"])
        for rid in reaction_ids:
            try:
                rows = resolve_reaction_participants(
                    rid, fetch_json=_fetch, entity_cache=entity_cache
                )
            except Exception as exc:
                typer.echo(f"[warn] skipping reaction {rid}: {exc}", err=True)
                continue
            for role, node_id in rows:
                w.writerow([rid, node_id, role])
            if sleep_ms:
                sleep(float(sleep_ms) / 1000.0)


@app.command()
def run_command(
    out_dir: Path = OUT_DIR_OPT,
    corum_url: str = CORUM_URL_OPT,
    reactome_protein_role_url: str = REACTOME_PROTEIN_ROLE_URL_OPT,
    chebi_pe_url: str = CHEBI_PE_URL_OPT,
    corum_version: str = CORUM_VERSION_OPT,
    reactome_version: str = REACTOME_VERSION_OPT,
    timeout_s: float = TIMEOUT_S_OPT,
    insecure_ssl: bool = INSECURE_SSL_OPT,
    reactome_mode: str = REACTOME_MODE_OPT,
    reactome_max_reactions: int = REACTOME_MAX_REACTIONS_OPT,
    reactome_content_service_base: str = REACTOME_CONTENT_SERVICE_BASE_OPT,
    reactome_sleep_ms: int = REACTOME_SLEEP_MS_OPT,
    allow_http_content_service: bool = ALLOW_HTTP_CONTENT_SERVICE_OPT,
    signor_url: str = SIGNOR_URL_OPT,
    signor_version: str = SIGNOR_VERSION_OPT,
    complexportal_url: str = COMPLEXPORTAL_URL_OPT,
    complexportal_version: str = COMPLEXPORTAL_VERSION_OPT,
    drugcomb_summary_url: str = DRUGCOMB_SUMMARY_URL_OPT,
    drugcomb_identifiers_url: str = DRUGCOMB_IDENTIFIERS_URL_OPT,
    drugcomb_version: str = DRUGCOMB_VERSION_OPT,
    clinvar_url: str = CLINVAR_URL_OPT,
    clinvar_version: str = CLINVAR_VERSION_OPT,
    chebi_compounds_url: str = CHEBI_COMPOUNDS_URL_OPT,
    chebi_names_version: str = CHEBI_NAMES_VERSION_OPT,
    emit_chebi_names: bool = typer.Option(
        True,
        "--emit-chebi-names/--no-emit-chebi-names",
        help="Fetch ChEBI compounds.tsv.gz and emit chebi_names.tsv.",
    ),
    emit_chebi_definitions: bool = typer.Option(
        True,
        "--emit-chebi-definitions/--no-emit-chebi-definitions",
        help="Fetch ChEBI compounds.tsv.gz and emit chebi_definitions.tsv.",
    ),
    chebi_names_url: str = CHEBI_NAMES_URL_OPT,
    emit_chebi_synonyms: bool = typer.Option(
        False,
        "--emit-chebi-synonyms/--no-emit-chebi-synonyms",
        help=(
            "Fetch ChEBI names.tsv.gz (and compounds.tsv.gz if needed) and emit "
            "chebi_synonyms.tsv. Default OFF (names.tsv.gz is large)."
        ),
    ),
    stitch_links_url: str = STITCH_LINKS_URL_OPT,
    unichem_drugbank_url: str = UNICHEM_DRUGBANK_URL_OPT,
    unichem_chebi_url: str = UNICHEM_CHEBI_URL_OPT,
    string_aliases_url: str = STRING_ALIASES_URL_OPT,
    emit_stitch: bool = typer.Option(
        False,
        "--emit-stitch/--no-emit-stitch",
        help=(
            "Fetch STITCH links + UniChem CID maps + STRING aliases and emit "
            "stitch.tsv / unichem_cid_map.tsv / string_ensp_geneid.tsv. "
            "Default OFF (STITCH is large and CC-BY-gated)."
        ),
    ),
    ctd_chem_gene_url: str = CTD_CHEM_GENE_URL_OPT,
    ctd_chemicals_url: str = CTD_CHEMICALS_URL_OPT,
    emit_ctd: bool = typer.Option(
        False,
        "--emit-ctd/--no-emit-ctd",
        help=(
            "Fetch CTD chem_gene + chemicals and emit ctd.tsv / ctd_mesh_cid.tsv. "
            "Default OFF (CTD terms: research use, cite and notify CTD)."
        ),
    ),
) -> None:
    # Ensure `Path` is used at runtime (ruff TC003).
    out_dir = Path(str(out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    corum_dl = download_file(
        url=corum_url,
        out_path=out_dir / "corum_download.txt",
        timeout_s=timeout_s,
        insecure_ssl=bool(insecure_ssl),
    )
    reactome_dl = download_file(
        url=reactome_protein_role_url,
        out_path=out_dir / "reactome_protein_role_reaction.txt",
        timeout_s=timeout_s,
        insecure_ssl=bool(insecure_ssl),
    )

    _convert_corum_all_complexes_to_tsv(corum_dl.path, out_dir / "corum.tsv")
    chebi_pe_dl = None
    if reactome_mode == "content-service":
        _convert_reactome_content_service_to_tsv(
            exporter_path=reactome_dl.path,
            out_tsv=out_dir / "reactome.tsv",
            base_url=str(reactome_content_service_base),
            timeout_s=float(timeout_s),
            max_reactions=int(reactome_max_reactions),
            sleep_ms=int(reactome_sleep_ms),
            allow_http=bool(allow_http_content_service),
            insecure_ssl=bool(insecure_ssl),
        )
    elif reactome_mode == "exporter-chebi":
        chebi_pe_dl = download_file(
            url=chebi_pe_url,
            out_path=out_dir / "ChEBI2Reactome_PE_Reactions.txt",
            timeout_s=timeout_s,
            insecure_ssl=bool(insecure_ssl),
        )
        merged_rows = _iter_reactome_rows(reactome_dl.path) + _iter_chebi_pe_rows(
            chebi_pe_dl.path
        )
        _write_reactome_rows(merged_rows, out_dir / "reactome.tsv")
    else:
        _convert_reactome_to_tsv(reactome_dl.path, out_dir / "reactome.tsv")

    signor_dl = download_file(
        url=signor_url,
        out_path=out_dir / "signor.tsv",
        timeout_s=timeout_s,
        insecure_ssl=bool(insecure_ssl),
    )
    complexportal_dl = download_file(
        url=complexportal_url,
        out_path=out_dir / "complexportal.tsv",
        timeout_s=timeout_s,
        insecure_ssl=bool(insecure_ssl),
    )
    drugcomb_summary_dl = download_file(
        url=drugcomb_summary_url,
        out_path=out_dir / "drugcomb_summary.csv",
        timeout_s=timeout_s,
        insecure_ssl=bool(insecure_ssl),
    )
    drugcomb_identifiers_dl = download_file(
        url=drugcomb_identifiers_url,
        out_path=out_dir / "drugcomb_drug_identifiers.xlsx",
        timeout_s=timeout_s,
        insecure_ssl=bool(insecure_ssl),
    )
    _convert_drugcomb_identifiers_xlsx(
        out_dir / "drugcomb_drug_identifiers.xlsx",
        out_dir / "drugcomb_drug_identifiers.tsv",
    )
    clinvar_dl = download_file(
        url=clinvar_url,
        out_path=out_dir / "clinvar_variant_summary.txt.gz",
        timeout_s=timeout_s,
        insecure_ssl=bool(insecure_ssl),
    )
    chebi_names_dl = None
    if emit_chebi_names:
        chebi_names_dl = download_file(
            url=chebi_compounds_url,
            out_path=out_dir / "chebi_compounds.tsv.gz",
            timeout_s=timeout_s,
            insecure_ssl=bool(insecure_ssl),
        )
        _convert_chebi_compounds_to_tsv(
            chebi_names_dl.path, out_dir / "chebi_names.tsv"
        )

    chebi_definitions_dl = None
    if emit_chebi_definitions:
        definitions_tsv = out_dir / "chebi_definitions.tsv"
        if chebi_names_dl is not None:
            _convert_chebi_definitions_to_tsv(chebi_names_dl.path, definitions_tsv)
            chebi_definitions_dl = chebi_names_dl
        else:
            chebi_definitions_dl = download_file(
                url=chebi_compounds_url,
                out_path=out_dir / "chebi_compounds.tsv.gz",
                timeout_s=timeout_s,
                insecure_ssl=bool(insecure_ssl),
            )
            _convert_chebi_definitions_to_tsv(
                chebi_definitions_dl.path, definitions_tsv
            )
        _write_chebi_definitions_meta(
            out_meta=out_dir / "chebi_definitions_meta.json",
            definitions_tsv=definitions_tsv,
            chebi_names_version=chebi_names_version,
            compounds_url=chebi_compounds_url,
            downloaded_at=chebi_definitions_dl.downloaded_at,
        )

    chebi_synonyms_dl = None
    if emit_chebi_synonyms:
        chebi_synonyms_dl = download_file(
            url=chebi_names_url,
            out_path=out_dir / "chebi_names.tsv.gz",
            timeout_s=timeout_s,
            insecure_ssl=bool(insecure_ssl),
        )
        if chebi_names_dl is not None:
            compounds_gz = chebi_names_dl.path
        elif chebi_definitions_dl is not None:
            compounds_gz = chebi_definitions_dl.path
        else:
            chebi_compounds_for_synonyms = download_file(
                url=chebi_compounds_url,
                out_path=out_dir / "chebi_compounds.tsv.gz",
                timeout_s=timeout_s,
                insecure_ssl=bool(insecure_ssl),
            )
            compounds_gz = chebi_compounds_for_synonyms.path
        _convert_chebi_names_gz_to_synonyms_tsv(
            chebi_synonyms_dl.path, compounds_gz, out_dir / "chebi_synonyms.tsv"
        )

    stitch_dl = None
    unichem_db_dl = None
    unichem_chebi_dl = None
    string_aliases_dl = None
    ctd_chem_gene_dl = None
    ctd_chemicals_dl = None
    if emit_stitch:
        stitch_dl = download_file(
            url=stitch_links_url,
            out_path=out_dir / "stitch_links.tsv.gz",
            timeout_s=timeout_s,
            insecure_ssl=bool(insecure_ssl),
        )
        _convert_stitch_links_to_tsv(stitch_dl.path, out_dir / "stitch.tsv")
        unichem_db_dl = download_file(
            url=unichem_drugbank_url,
            out_path=out_dir / "unichem_src2src22.txt.gz",
            timeout_s=timeout_s,
            insecure_ssl=bool(insecure_ssl),
        )
        unichem_chebi_dl = download_file(
            url=unichem_chebi_url,
            out_path=out_dir / "unichem_src7src22.txt.gz",
            timeout_s=timeout_s,
            insecure_ssl=bool(insecure_ssl),
        )
        _convert_unichem_pair_to_cid_map_tsv(
            unichem_db_dl.path, out_dir / "_unichem_db.tsv", kind="drugbank"
        )
        _convert_unichem_pair_to_cid_map_tsv(
            unichem_chebi_dl.path, out_dir / "_unichem_chebi.tsv", kind="chebi"
        )
        _merge_unichem_cid_maps(
            out_dir / "_unichem_db.tsv",
            out_dir / "_unichem_chebi.tsv",
            out_dir / "unichem_cid_map.tsv",
        )
        unichem_stamp = (
            (unichem_db_dl.downloaded_at or "")[:10]
            or (unichem_chebi_dl.downloaded_at or "")[:10]
            or "unknown"
        )
        _convert_unichem_pairs_to_pubchem_xref_v2_tsv(
            unichem_db_dl.path,
            unichem_chebi_dl.path,
            out_dir / "unichem_pubchem_xref_v2.tsv",
            source_version=f"unichem:{unichem_stamp}",
        )
        string_aliases_dl = download_file(
            url=string_aliases_url,
            out_path=out_dir / "string_aliases.txt.gz",
            timeout_s=timeout_s,
            insecure_ssl=bool(insecure_ssl),
        )
        _convert_string_aliases_to_ensp_geneid_tsv(
            string_aliases_dl.path, out_dir / "string_ensp_geneid.tsv"
        )

    if emit_ctd:
        ctd_chem_gene_dl = download_file(
            url=ctd_chem_gene_url,
            out_path=out_dir / "ctd_chem_gene_ixns.tsv.gz",
            timeout_s=timeout_s,
            insecure_ssl=bool(insecure_ssl),
        )
        _convert_ctd_chem_gene_to_tsv(ctd_chem_gene_dl.path, out_dir / "ctd.tsv")
        ctd_chemicals_dl = download_file(
            url=ctd_chemicals_url,
            out_path=out_dir / "ctd_chemicals.tsv.gz",
            timeout_s=timeout_s,
            insecure_ssl=bool(insecure_ssl),
        )
        _convert_ctd_chemicals_to_mesh_cid_tsv(
            ctd_chemicals_dl.path, out_dir / "ctd_mesh_cid.tsv"
        )
        ctd_stamp = (ctd_chemicals_dl.downloaded_at or "")[:10] or "unknown"
        _convert_ctd_chemicals_to_mesh_pubchem_v2_tsv(
            ctd_chemicals_dl.path,
            out_dir / "ctd_mesh_pubchem_v2.tsv",
            source_version=f"ctd:{ctd_stamp}",
        )

    sources = [
        {
            "name": "CORUM",
            "version": corum_version,
            "url": corum_dl.url,
            "sha256": corum_dl.sha256,
            "downloaded_at": corum_dl.downloaded_at,
        },
        {
            "name": "Reactome",
            "version": reactome_version,
            "url": reactome_dl.url,
            "sha256": reactome_dl.sha256,
            "downloaded_at": reactome_dl.downloaded_at,
        },
    ]
    if chebi_pe_dl is not None:
        sources.append(
            {
                "name": "Reactome_ChEBI_PE",
                "version": reactome_version,
                "url": chebi_pe_dl.url,
                "sha256": chebi_pe_dl.sha256,
                "downloaded_at": chebi_pe_dl.downloaded_at,
            }
        )
    sources.append(
        {
            "name": "SIGNOR",
            "version": signor_version,
            "url": signor_dl.url,
            "sha256": signor_dl.sha256,
            "downloaded_at": signor_dl.downloaded_at,
        }
    )
    sources.append(
        {
            "name": "Complex Portal",
            "version": complexportal_version,
            "url": complexportal_dl.url,
            "sha256": complexportal_dl.sha256,
            "downloaded_at": complexportal_dl.downloaded_at,
        }
    )
    sources.append(
        {
            "name": "DrugComb",
            "version": drugcomb_version,
            "url": drugcomb_summary_dl.url,
            "sha256": drugcomb_summary_dl.sha256,
            "downloaded_at": drugcomb_summary_dl.downloaded_at,
        }
    )
    sources.append(
        {
            "name": "DrugComb_identifiers",
            "version": drugcomb_version,
            "url": drugcomb_identifiers_dl.url,
            "sha256": drugcomb_identifiers_dl.sha256,
            "downloaded_at": drugcomb_identifiers_dl.downloaded_at,
        }
    )
    sources.append(
        {
            "name": "ClinVar",
            "version": clinvar_version,
            "url": clinvar_dl.url,
            "sha256": clinvar_dl.sha256,
            "downloaded_at": clinvar_dl.downloaded_at,
        }
    )
    if chebi_names_dl is not None:
        sources.append(
            {
                "name": "chebi_names",
                "version": chebi_names_version,
                "url": chebi_names_dl.url,
                "sha256": chebi_names_dl.sha256,
                "downloaded_at": chebi_names_dl.downloaded_at,
            }
        )
    if chebi_definitions_dl is not None:
        sources.append(
            {
                "name": "chebi_definitions",
                "version": chebi_names_version,
                "url": chebi_definitions_dl.url,
                "sha256": chebi_definitions_dl.sha256,
                "downloaded_at": chebi_definitions_dl.downloaded_at,
            }
        )
    if chebi_synonyms_dl is not None:
        sources.append(
            {
                "name": "chebi_synonyms",
                "version": chebi_names_version,
                "url": chebi_synonyms_dl.url,
                "sha256": chebi_synonyms_dl.sha256,
                "downloaded_at": chebi_synonyms_dl.downloaded_at,
            }
        )
    if stitch_dl is not None:
        sources.append(
            {
                "name": "STITCH",
                "version": "stitch:v5.0",
                "url": stitch_dl.url,
                "sha256": stitch_dl.sha256,
                "downloaded_at": stitch_dl.downloaded_at,
            }
        )
    if unichem_db_dl is not None:
        sources.append(
            {
                "name": "UniChem_drugbank",
                "version": "unichem:unknown",
                "url": unichem_db_dl.url,
                "sha256": unichem_db_dl.sha256,
                "downloaded_at": unichem_db_dl.downloaded_at,
            }
        )
    if unichem_chebi_dl is not None:
        sources.append(
            {
                "name": "UniChem_chebi",
                "version": "unichem:unknown",
                "url": unichem_chebi_dl.url,
                "sha256": unichem_chebi_dl.sha256,
                "downloaded_at": unichem_chebi_dl.downloaded_at,
            }
        )
    if string_aliases_dl is not None:
        sources.append(
            {
                "name": "STRING_aliases",
                "version": "string:v12.0",
                "url": string_aliases_dl.url,
                "sha256": string_aliases_dl.sha256,
                "downloaded_at": string_aliases_dl.downloaded_at,
            }
        )
    if ctd_chem_gene_dl is not None:
        sources.append(
            {
                "name": "CTD_chem_gene_ixns",
                "version": "ctd:latest",
                "url": ctd_chem_gene_dl.url,
                "sha256": ctd_chem_gene_dl.sha256,
                "downloaded_at": ctd_chem_gene_dl.downloaded_at,
            }
        )
    if ctd_chemicals_dl is not None:
        sources.append(
            {
                "name": "CTD_chemicals",
                "version": "ctd:latest",
                "url": ctd_chemicals_dl.url,
                "sha256": ctd_chemicals_dl.sha256,
                "downloaded_at": ctd_chemicals_dl.downloaded_at,
            }
        )
    _write_sources_json(out_dir, sources)


def main(argv: list[str] | None = None) -> int:
    cmd = get_command(app)
    args_list = _argv_for_cli(argv)
    try:
        rc = cmd.main(
            args=args_list,
            prog_name="primehkg-curated-fetch",
            standalone_mode=False,
        )
    except SystemExit as e:
        if isinstance(e.code, int):
            return e.code
        return 1
    return int(rc) if rc is not None else 0


def entrypoint() -> None:
    raise SystemExit(main())
