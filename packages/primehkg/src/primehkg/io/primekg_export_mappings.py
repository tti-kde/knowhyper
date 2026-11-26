"""Build maintainer-owned extended export TSVs from PrimeKG + public APIs."""

from __future__ import annotations

import csv
import json
import re
import time
import urllib.error
from collections import Counter, deque
from dataclasses import dataclass, field
from http.client import RemoteDisconnected
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlencode

from primehkg.io.chembl_molecule import (
    fetch_chembl_molecule_properties,
    fetch_chembl_xref_for_drugbank_id,
)
from primehkg.io.drugbank_properties import load_drugbank_properties_index
from primehkg.io.drugbank_structures import load_drugbank_smiles_index
from primehkg.io.http_download import download_file, get_json
from primehkg.io.primekg import _primekg_typed_node_id
from primehkg.io.primekg_feature_tabs import build_node_index_to_node_id
from primehkg.io.pubchem_properties import fetch_pubchem_properties_for_drugbank_id
from primehkg.nodes.normalize import normalize_xref_id

DEFAULT_HGNC_JSON_URL = (
    "https://storage.googleapis.com/public-download-files/hgnc/json/json/"
    "hgnc_complete_set.json"
)
DEFAULT_HPO_OBO_URL = "https://github.com/obophenotype/human-phenotype-ontology/releases/latest/download/hp.obo"
DEFAULT_PHENOTYPE_HPOA_URL = (
    "https://github.com/obophenotype/human-phenotype-ontology/"
    "releases/latest/download/phenotype.hpoa"
)
DEFAULT_MONDO_OBO_URL = (
    "https://github.com/monarch-initiative/mondo/releases/latest/download/mondo.obo"
)
_COHORT_FRACTION_RE = re.compile(r"^(\d+)/(\d+)$")
_HPO_FREQUENCY_HP_RE = re.compile(r"^HP:004028[0-5]$")
HPO_FREQUENCY_ORDINAL: dict[str, int] = {
    "HP:0040285": 0,
    "HP:0040284": 1,
    "HP:0040283": 2,
    "HP:0040282": 3,
    "HP:0040281": 4,
    "HP:0040280": 5,
}
DEFAULT_GO_BASIC_OBO_URL = "http://purl.obolibrary.org/obo/go/go-basic.obo"
GO_NODE_TYPES = frozenset(
    {"biological_process", "molecular_function", "cellular_component"}
)
UNIPROT_HUMAN_STREAM = (
    "https://rest.uniprot.org/uniprotkb/stream?format=tsv&fields=accession,"
    "length,mass,protein_name,gene_primary,cc_function,sequence,xref_hgnc,xref_geneid"
    "&query=organism_id:9606"
)
DEFAULT_GENE2REFSEQ_URL = "https://ftp.ncbi.nlm.nih.gov/gene/DATA/gene2refseq.gz"
EFETCH_PROTEIN_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
EFETCH_NUCLEOTIDE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
USER_AGENT = "primehkg-export-mappings/0.1"
NUCLEOTIDE_EXPORT_SCOPE_OR_FAIL_RNA = "or_fail_rna"
NUCLEOTIDE_EXPORT_SCOPE_ALL_MISSING_PROTEIN = "all_missing_protein"
NUCLEOTIDE_TSV_FIELDNAMES = [
    "node_id",
    "entrez_id",
    "refseq_rna_accession",
    "Sequence",
    "Gene symbol",
]
UNIPROT_TSV_FIELDNAMES = [
    "node_id",
    "Entry",
    "Length",
    "Mass",
    "Protein names",
    "Gene Names",
    "Function",
    "Cross-reference (HGNC)",
    "Sequence",
    "Gene ID",
]


@dataclass(frozen=True)
class ExportCounts:
    """Row counts written for each extended export artifact."""

    mondo_map: int = 0
    hpo_map: int = 0
    go_map: int = 0
    pathway_map: int = 0
    anatomy_map: int = 0
    exposure_map: int = 0
    uniprot: int = 0
    nucleotide_map: int = 0
    chembl_drugs: int = 0
    drugbank_smiles: int = 0
    drugbank_properties: int = 0
    hgnc: int = 0


CHEMBL_DRUG_PROPERTY_COLUMNS: tuple[str, ...] = (
    "molecular_weight",
    "alogp",
    "psa",
    "hba",
    "hbd",
)

DRUGBANK_PROPERTIES_FIELDNAMES: tuple[str, ...] = (
    "node_id",
    "drugbank_id",
    *CHEMBL_DRUG_PROPERTY_COLUMNS,
)


def _strip_quotes(value: str) -> str:
    return (value or "").strip().strip('"')


def _tsv_data_row_count(path: Path) -> int:
    if not path.is_file() or path.stat().st_size == 0:
        return 0
    with path.open("r", encoding="utf-8", newline="") as f:
        row_count = sum(1 for _ in f)
    return max(0, row_count - 1)


def _existing_export_count(path: Path, *, min_rows: int, force: bool) -> int | None:
    if force:
        return None
    count = _tsv_data_row_count(path)
    if count >= min_rows:
        return count
    return None


def write_mondo_map_tsv(
    *,
    kg_csv: Path,
    disease_features_tab: Path,
    out_path: Path,
) -> int:
    """Write ``mondo_map.tsv`` (node_id, mondo_id, mondo_name, definition)."""
    index_to_node_id = build_node_index_to_node_id(kg_csv, strict=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    with (
        disease_features_tab.open("r", encoding="utf-8", newline="") as src,
        out_path.open("w", encoding="utf-8", newline="") as dst,
    ):
        reader = csv.DictReader(src, delimiter="\t")
        if not reader.fieldnames:
            msg = "disease_features.tab is missing a header row"
            raise ValueError(msg)
        cols = {c.strip(): c for c in reader.fieldnames if c is not None}
        required = {"node_index", "mondo_id", "mondo_name"}
        missing = required - set(cols)
        if missing:
            msg = f"disease_features.tab missing columns: {sorted(missing)}"
            raise ValueError(msg)
        writer = csv.DictWriter(
            dst,
            fieldnames=["node_id", "mondo_id", "mondo_name", "definition"],
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in reader:
            idx_raw = (row.get(cols["node_index"]) or "").strip()
            if not idx_raw:
                continue
            try:
                idx = int(idx_raw)
            except ValueError:
                continue
            node_id = index_to_node_id.get(idx)
            if not node_id:
                continue
            mondo_id_raw = (row.get(cols["mondo_id"]) or "").strip()
            mondo_name = _strip_quotes(row.get(cols["mondo_name"]) or "")
            if not mondo_id_raw and not mondo_name:
                continue
            mondo_id = normalize_xref_id("mondo", mondo_id_raw) if mondo_id_raw else ""
            definition = ""
            if "mondo_definition" in cols:
                definition = _strip_quotes(row.get(cols["mondo_definition"]) or "")
            writer.writerow(
                {
                    "node_id": node_id,
                    "mondo_id": mondo_id,
                    "mondo_name": mondo_name,
                    "definition": definition,
                }
            )
            rows_written += 1
    return rows_written


def _iter_nodes_tab_rows(nodes_tab: Path) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    with nodes_tab.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if not reader.fieldnames:
            msg = "nodes.tab is missing a header row"
            raise ValueError(msg)
        cols = {c.strip(): c for c in reader.fieldnames if c is not None}
        required = {"node_id", "node_type", "node_name"}
        missing = required - set(cols)
        if missing:
            msg = f"nodes.tab missing columns: {sorted(missing)}"
            raise ValueError(msg)
        for row in reader:
            raw_id = _strip_quotes(row.get(cols["node_id"]) or "")
            raw_type = _strip_quotes(row.get(cols["node_type"]) or "")
            raw_name = _strip_quotes(row.get(cols["node_name"]) or "")
            if not raw_id or not raw_type:
                continue
            node_id = _primekg_typed_node_id(raw_id, raw_type)
            if not node_id:
                continue
            rows.append((node_id, raw_type, raw_name))
    return rows


@dataclass(frozen=True, slots=True)
class HpoTermStats:
    """Per-HPO-term counts from ``hp.obo`` ``[Term]`` blocks."""

    synonym_count: int
    parent_count: int
    xref_count: int
    is_obsolete: int  # 0 or 1


@dataclass(frozen=True, slots=True)
class HpoGraphMetrics:
    """Per-HPO-term ontology graph metrics from ``hp.obo`` ``is_a`` edges."""

    ontology_depth: int
    descendant_count: int


@dataclass(frozen=True, slots=True)
class HpoaTermAggregates:
    """Per-HPO-term aggregates from ``phenotype.hpoa``."""

    annotation_count: int
    disease_count: int
    frequency_row_count: int
    cohort_fraction_mean: float | None
    negative_annotation_count: int = 0
    frequency_hp_term_count: int = 0
    frequency_ordinal_mode: int | None = None
    primekg_disease_count: int = 0


@dataclass
class _HpoaAccumulator:
    annotation_count: int = 0
    diseases: set[str] = field(default_factory=set)
    frequency_row_count: int = 0
    cohort_fractions: list[float] = field(default_factory=list)
    negative_annotation_count: int = 0
    frequency_ordinals: Counter[int] = field(default_factory=Counter)
    primekg_diseases: set[str] = field(default_factory=set)


def _parse_cohort_fraction(raw: str) -> float | None:
    m = _COHORT_FRACTION_RE.match((raw or "").strip())
    if not m:
        return None
    n, d = int(m.group(1)), int(m.group(2))
    if d <= 0:
        return None
    return n / d


def _frequency_ordinal_mode(counter: Counter[int]) -> int | None:
    if not counter:
        return None
    return max(counter.items(), key=lambda kv: (kv[1], kv[0]))[0]


def _normalize_hpoa_database_id(raw: str) -> str:
    s = (raw or "").strip()
    if not s:
        return ""
    upper = s.upper()
    if upper.startswith("ORPHANET:"):
        local = s.split(":", 1)[1].strip()
        return f"ORPHA:{local}" if local else ""
    if upper.startswith(("OMIM:", "ORPHA:", "MONDO:", "OMIMPS:")):
        return upper.replace("OMIMPS:", "OMIM:")
    return s


def _hpo_frequency_ordinal(frequency: str) -> int | None:
    try:
        hp_id = normalize_xref_id("hpo", frequency)
    except ValueError:
        return None
    if not _HPO_FREQUENCY_HP_RE.match(hp_id):
        return None
    return HPO_FREQUENCY_ORDINAL.get(hp_id)


def parse_mondo_disease_xrefs(path: Path) -> dict[str, str]:
    """Map ``OMIM:`` / ``ORPHA:`` / ``MONDO:`` database ids to ``MONDO:`` CURIEs."""
    out: dict[str, str] = {}
    current_mondo: str | None = None
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped == "[Term]":
                current_mondo = None
                continue
            if stripped.startswith("[") and stripped != "[Term]":
                current_mondo = None
                continue
            if stripped.startswith("id: MONDO:"):
                current_mondo = normalize_xref_id(
                    "mondo", stripped.removeprefix("id: ").strip()
                )
                out[current_mondo] = current_mondo
                continue
            if current_mondo is None or not stripped.startswith("xref:"):
                continue
            rest = stripped.removeprefix("xref:").strip()
            xref_raw = rest.split("{", 1)[0].strip()
            xref_id = _normalize_hpoa_database_id(xref_raw)
            if xref_id:
                out[xref_id] = current_mondo
    return out


def _primekg_disease_bridge_from_mondo_map(
    mondo_map_path: Path,
) -> tuple[frozenset[str], dict[str, str]]:
    node_ids: set[str] = set()
    mondo_to_node: dict[str, str] = {}
    with mondo_map_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if reader.fieldnames is None:
            return frozenset(), {}
        for row in reader:
            node_id = (row.get("node_id") or "").strip()
            mondo_id_raw = (row.get("mondo_id") or "").strip()
            if not node_id or not mondo_id_raw:
                continue
            try:
                mondo_id = normalize_xref_id("mondo", mondo_id_raw)
            except ValueError:
                continue
            node_ids.add(node_id)
            mondo_to_node[mondo_id] = node_id
    return frozenset(node_ids), mondo_to_node


def parse_phenotype_hpoa_aggregates(
    path: Path,
    *,
    aspects: frozenset[str] = frozenset({"P"}),
    database_id_to_mondo: dict[str, str] | None = None,
    primekg_disease_node_ids: frozenset[str] | None = None,
    mondo_id_to_node_id: dict[str, str] | None = None,
) -> dict[str, HpoaTermAggregates]:
    """Aggregate phenotype.hpoa rows per normalized ``HP:`` id."""
    accum: dict[str, _HpoaAccumulator] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        lines = [line for line in f if not line.startswith("#")]
    if not lines:
        return {}
    reader = csv.DictReader(lines, delimiter="\t")
    if reader.fieldnames is None:
        return {}
    for row in reader:
        hp_raw = (row.get("hpo_id") or "").strip()
        if not hp_raw:
            continue
        hp_id = normalize_xref_id("hpo", hp_raw)
        if not hp_id:
            continue
        acc = accum.setdefault(hp_id, _HpoaAccumulator())
        qualifier = (row.get("qualifier") or "").strip().upper()
        if qualifier == "NOT":
            acc.negative_annotation_count += 1
            continue
        aspect = (row.get("aspect") or "").strip()
        if aspect not in aspects:
            continue
        acc.annotation_count += 1
        database_id = (row.get("database_id") or "").strip()
        if database_id:
            acc.diseases.add(database_id)
            if (
                database_id_to_mondo is not None
                and primekg_disease_node_ids is not None
                and mondo_id_to_node_id is not None
            ):
                db_key = _normalize_hpoa_database_id(database_id)
                mondo_id = database_id_to_mondo.get(db_key)
                if mondo_id:
                    node_id = mondo_id_to_node_id.get(mondo_id)
                    if node_id and node_id in primekg_disease_node_ids:
                        acc.primekg_diseases.add(database_id)
        frequency = (row.get("frequency") or "").strip()
        if frequency:
            acc.frequency_row_count += 1
            ordinal = _hpo_frequency_ordinal(frequency)
            if ordinal is not None:
                acc.frequency_ordinals[ordinal] += 1
            frac = _parse_cohort_fraction(frequency)
            if frac is not None:
                acc.cohort_fractions.append(frac)
    out: dict[str, HpoaTermAggregates] = {}
    for hp_id, acc in accum.items():
        mean: float | None = None
        if acc.cohort_fractions:
            mean = sum(acc.cohort_fractions) / len(acc.cohort_fractions)
        freq_hp_count = sum(acc.frequency_ordinals.values())
        out[hp_id] = HpoaTermAggregates(
            annotation_count=acc.annotation_count,
            disease_count=len(acc.diseases),
            frequency_row_count=acc.frequency_row_count,
            cohort_fraction_mean=mean,
            negative_annotation_count=acc.negative_annotation_count,
            frequency_hp_term_count=freq_hp_count,
            frequency_ordinal_mode=_frequency_ordinal_mode(acc.frequency_ordinals),
            primekg_disease_count=len(acc.primekg_diseases),
        )
    return out


def download_mondo_obo(
    *,
    out_path: Path,
    url: str = DEFAULT_MONDO_OBO_URL,
    timeout_s: float = 120.0,
    insecure_ssl: bool = False,
) -> Path:
    download_file(
        url=url,
        out_path=out_path,
        timeout_s=timeout_s,
        user_agent=USER_AGENT,
        insecure_ssl=insecure_ssl,
    )
    return out_path


def download_phenotype_hpoa(
    *,
    out_path: Path,
    url: str = DEFAULT_PHENOTYPE_HPOA_URL,
    timeout_s: float = 120.0,
    insecure_ssl: bool = False,
) -> Path:
    download_file(
        url=url,
        out_path=out_path,
        timeout_s=timeout_s,
        user_agent=USER_AGENT,
        insecure_ssl=insecure_ssl,
    )
    return out_path


def parse_hpo_obo_term_stats(path: Path) -> dict[str, HpoTermStats]:
    """Parse ``[Term]`` blocks into per-id counts (synonym, is_a, xref, obsolete)."""
    stats: dict[str, HpoTermStats] = {}
    current_id: str | None = None
    syn = parent = xref = 0
    obsolete = 0

    def _flush() -> None:
        nonlocal current_id, syn, parent, xref, obsolete
        if current_id and current_id.startswith("HP:"):
            stats[current_id] = HpoTermStats(
                synonym_count=syn,
                parent_count=parent,
                xref_count=xref,
                is_obsolete=obsolete,
            )
        current_id = None
        syn = parent = xref = obsolete = 0

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped == "[Term]":
                _flush()
                continue
            if stripped.startswith("[") and stripped != "[Term]":
                _flush()
                continue
            if stripped == "":
                continue
            if stripped.startswith("id: HP:"):
                current_id = stripped.removeprefix("id: ").strip()
                continue
            if current_id is None:
                continue
            if stripped.startswith("synonym:"):
                syn += 1
            elif stripped.startswith("is_a:"):
                parent += 1
            elif stripped.startswith("xref:"):
                xref += 1
            elif stripped == "is_obsolete: true":
                obsolete = 1
    _flush()
    return stats


def parse_hpo_obo_is_a_parents(path: Path) -> dict[str, tuple[str, ...]]:
    """Parse ``[Term]`` blocks into per-id direct ``is_a`` parent HP CURIEs."""
    parents: dict[str, tuple[str, ...]] = {}
    current_id: str | None = None
    parent_list: list[str] = []

    def _flush() -> None:
        nonlocal current_id, parent_list
        if current_id and current_id.startswith("HP:"):
            parents[current_id] = tuple(parent_list)
        current_id = None
        parent_list = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped == "[Term]":
                _flush()
                continue
            if stripped.startswith("[") and stripped != "[Term]":
                _flush()
                continue
            if stripped == "":
                continue
            if stripped.startswith("id: HP:"):
                current_id = stripped.removeprefix("id: ").strip()
                continue
            if current_id is None:
                continue
            if stripped.startswith("is_a:"):
                rest = stripped.removeprefix("is_a:").strip()
                parent_raw = rest.split("!", 1)[0].strip()
                if parent_raw.startswith("HP:"):
                    parent_list.append(parent_raw)
    _flush()
    return parents


def compute_hpo_graph_metrics(
    parents: dict[str, tuple[str, ...]],
) -> dict[str, HpoGraphMetrics]:
    """Shortest ``is_a`` depth from roots and distinct transitive descendant counts."""
    children: dict[str, list[str]] = {}
    nodes: set[str] = set(parents)
    for child, ps in parents.items():
        for parent in ps:
            nodes.add(parent)
            children.setdefault(parent, []).append(child)

    roots = [n for n in nodes if not parents.get(n, ())]
    depth: dict[str, int] = dict.fromkeys(roots, 0)
    queue: deque[str] = deque(roots)
    while queue:
        parent = queue.popleft()
        parent_depth = depth[parent]
        for child in children.get(parent, []):
            candidate = parent_depth + 1
            if child not in depth or candidate < depth[child]:
                depth[child] = candidate
                queue.append(child)

    for node in nodes:
        depth.setdefault(node, 0)

    memo: dict[str, frozenset[str]] = {}
    visiting: set[str] = set()

    def _descendants(term: str) -> frozenset[str]:
        if term in memo:
            return memo[term]
        if term in visiting:
            return frozenset()
        visiting.add(term)
        out: set[str] = set()
        for child in children.get(term, []):
            out.add(child)
            out |= _descendants(child)
        visiting.discard(term)
        frozen = frozenset(out)
        memo[term] = frozen
        return frozen

    return {
        node: HpoGraphMetrics(
            ontology_depth=depth[node],
            descendant_count=len(_descendants(node)),
        )
        for node in nodes
    }


def parse_hpo_obo_definitions(path: Path) -> dict[str, str]:
    """Parse ``[Term]`` blocks from an HPO OBO file into ``HP:…`` → definition."""
    definitions: dict[str, str] = {}
    current_id: str | None = None
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped == "[Term]":
                current_id = None
                continue
            if stripped == "" or stripped.startswith("["):
                continue
            if stripped.startswith("id: HP:"):
                current_id = stripped.removeprefix("id: ").strip()
                continue
            if (
                current_id
                and stripped.startswith("def: ")
                and current_id not in definitions
            ):
                match = re.match(r'^def: "(.*)" \[\]', stripped)
                if match:
                    text = match.group(1).replace('\\"', '"').replace("\\n", "\n")
                    definitions[current_id] = text
    return definitions


def download_hpo_obo(
    *,
    out_path: Path,
    url: str = DEFAULT_HPO_OBO_URL,
    timeout_s: float = 120.0,
    insecure_ssl: bool = False,
) -> Path:
    download_file(
        url=url,
        out_path=out_path,
        timeout_s=timeout_s,
        user_agent=USER_AGENT,
        insecure_ssl=insecure_ssl,
    )
    return out_path


def normalize_go_id(raw: str) -> str:
    return normalize_xref_id("go", raw)


def parse_go_obo_definitions_from_text(text: str) -> dict[str, str]:
    """Parse ``[Term]`` blocks from GO OBO text into ``GO:…`` → definition."""
    definitions: dict[str, str] = {}
    current_id: str | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == "[Term]":
            current_id = None
            continue
        if stripped == "" or stripped.startswith("["):
            continue
        if stripped.startswith("id: GO:"):
            current_id = stripped.removeprefix("id: ").strip()
            continue
        if (
            current_id
            and stripped.startswith("def: ")
            and current_id not in definitions
        ):
            match = re.match(r'^def: "(.*)" \[', stripped)
            if match:
                text_val = match.group(1).replace('\\"', '"').replace("\\n", "\n")
                definitions[current_id] = text_val
    return definitions


def parse_go_obo_definitions(path: Path) -> dict[str, str]:
    return parse_go_obo_definitions_from_text(path.read_text(encoding="utf-8"))


def download_go_obo(
    *,
    out_path: Path,
    url: str = DEFAULT_GO_BASIC_OBO_URL,
    timeout_s: float = 120.0,
    insecure_ssl: bool = False,
) -> Path:
    download_file(
        url=url,
        out_path=out_path,
        timeout_s=timeout_s,
        user_agent=USER_AGENT,
        insecure_ssl=insecure_ssl,
    )
    return out_path


def write_go_map_tsv(
    *,
    nodes_tab: Path,
    go_obo: Path | None,
    out_path: Path,
) -> int:
    definitions = (
        parse_go_obo_definitions(go_obo)
        if go_obo is not None and go_obo.is_file()
        else {}
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    with out_path.open("w", encoding="utf-8", newline="") as dst:
        writer = csv.DictWriter(
            dst,
            fieldnames=["node_id", "go_id", "go_name", "definition"],
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        for node_id, raw_type, raw_name in _iter_nodes_tab_rows(nodes_tab):
            if raw_type not in GO_NODE_TYPES:
                continue
            local = node_id.split(":")[-1]
            go_id = normalize_go_id(local)
            writer.writerow(
                {
                    "node_id": node_id,
                    "go_id": go_id,
                    "go_name": raw_name,
                    "definition": definitions.get(go_id, ""),
                }
            )
            rows_written += 1
    return rows_written


def write_pathway_map_tsv(*, nodes_tab: Path, out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    with out_path.open("w", encoding="utf-8", newline="") as dst:
        writer = csv.DictWriter(
            dst,
            fieldnames=["node_id", "reactome_id", "pathway_name", "description"],
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        for node_id, raw_type, raw_name in _iter_nodes_tab_rows(nodes_tab):
            if raw_type != "pathway":
                continue
            if not raw_name:
                continue
            reactome_id = node_id.split(":")[-1]
            writer.writerow(
                {
                    "node_id": node_id,
                    "reactome_id": reactome_id,
                    "pathway_name": raw_name,
                    "description": raw_name,
                }
            )
            rows_written += 1
    return rows_written


def write_anatomy_map_tsv(*, nodes_tab: Path, out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    with out_path.open("w", encoding="utf-8", newline="") as dst:
        writer = csv.DictWriter(
            dst,
            fieldnames=["node_id", "anatomy_id", "anatomy_name", "description"],
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        for node_id, raw_type, raw_name in _iter_nodes_tab_rows(nodes_tab):
            if raw_type != "anatomy":
                continue
            if not raw_name:
                continue
            anatomy_id = node_id.split(":")[-1]
            writer.writerow(
                {
                    "node_id": node_id,
                    "anatomy_id": anatomy_id,
                    "anatomy_name": raw_name,
                    "description": raw_name,
                }
            )
            rows_written += 1
    return rows_written


def write_exposure_map_tsv(*, nodes_tab: Path, out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    with out_path.open("w", encoding="utf-8", newline="") as dst:
        writer = csv.DictWriter(
            dst,
            fieldnames=["node_id", "exposure_id", "exposure_name", "description"],
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        for node_id, raw_type, raw_name in _iter_nodes_tab_rows(nodes_tab):
            if raw_type != "exposure":
                continue
            if not raw_name:
                continue
            exposure_id = node_id.split(":")[-1]
            writer.writerow(
                {
                    "node_id": node_id,
                    "exposure_id": exposure_id,
                    "exposure_name": raw_name,
                    "description": raw_name,
                }
            )
            rows_written += 1
    return rows_written


def write_hpo_map_tsv(
    *,
    nodes_tab: Path,
    out_path: Path,
    hpo_definitions: dict[str, str] | None = None,
    hpo_term_stats: dict[str, HpoTermStats] | None = None,
    hpoa_aggregates: dict[str, HpoaTermAggregates] | None = None,
    hpo_graph_metrics: dict[str, HpoGraphMetrics] | None = None,
) -> int:
    """Write ``hpo_map.tsv`` for ``effect/phenotype`` (and ``phenotype``) nodes."""
    phenotype_types = {"effect/phenotype", "phenotype"}
    base_fieldnames = ["node_id", "hp_id", "hp_label", "definition"]
    extended = hpo_term_stats is not None or hpoa_aggregates is not None
    if extended:
        fieldnames = [
            *base_fieldnames,
            "synonym_count",
            "parent_count",
            "xref_count",
            "is_obsolete",
            "hpoa_annotation_count",
            "hpoa_disease_count",
            "hpoa_frequency_row_count",
            "hpoa_cohort_fraction_mean",
            "hpoa_negative_annotation_count",
            "hpoa_frequency_hp_term_count",
            "hpoa_frequency_ordinal_mode",
            "hpoa_primekg_disease_count",
        ]
        if hpo_graph_metrics is not None:
            fieldnames = [*fieldnames, "ontology_depth", "descendant_count"]
    else:
        fieldnames = base_fieldnames
    default_stats = HpoTermStats(0, 0, 0, 0)
    default_hpoa = HpoaTermAggregates(0, 0, 0, None)
    default_graph = HpoGraphMetrics(0, 0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    with out_path.open("w", encoding="utf-8", newline="") as dst:
        writer = csv.DictWriter(
            dst,
            fieldnames=fieldnames,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        for node_id, raw_type, raw_name in _iter_nodes_tab_rows(nodes_tab):
            if raw_type not in phenotype_types:
                continue
            local_id = node_id.split(":")[-1]
            hp_id = normalize_xref_id("hpo", local_id)
            definition = (hpo_definitions or {}).get(hp_id, "")
            row: dict[str, str | int] = {
                "node_id": node_id,
                "hp_id": hp_id,
                "hp_label": raw_name,
                "definition": definition,
            }
            if extended:
                stats = (hpo_term_stats or {}).get(hp_id, default_stats)
                row["synonym_count"] = stats.synonym_count
                row["parent_count"] = stats.parent_count
                row["xref_count"] = stats.xref_count
                row["is_obsolete"] = stats.is_obsolete
                hpoa = (hpoa_aggregates or {}).get(hp_id, default_hpoa)
                row["hpoa_annotation_count"] = hpoa.annotation_count
                row["hpoa_disease_count"] = hpoa.disease_count
                row["hpoa_frequency_row_count"] = hpoa.frequency_row_count
                mean = hpoa.cohort_fraction_mean
                row["hpoa_cohort_fraction_mean"] = "" if mean is None else str(mean)
                row["hpoa_negative_annotation_count"] = hpoa.negative_annotation_count
                row["hpoa_frequency_hp_term_count"] = hpoa.frequency_hp_term_count
                mode = hpoa.frequency_ordinal_mode
                row["hpoa_frequency_ordinal_mode"] = "" if mode is None else str(mode)
                row["hpoa_primekg_disease_count"] = hpoa.primekg_disease_count
                if hpo_graph_metrics is not None:
                    graph = hpo_graph_metrics.get(hp_id, default_graph)
                    row["ontology_depth"] = graph.ontology_depth
                    row["descendant_count"] = graph.descendant_count
            writer.writerow(row)
            rows_written += 1
    return rows_written


def _parse_uniprot_gene_ids(raw: str) -> list[str]:
    values: list[str] = []
    for chunk in (raw or "").replace("|", ";").split(";"):
        item = chunk.strip()
        if item.isdigit():
            values.append(item)
    return values


def _download_uniprot_human_tsv(
    *,
    cache_path: Path,
    timeout_s: float,
    insecure_ssl: bool,
) -> Path:
    download_file(
        url=UNIPROT_HUMAN_STREAM,
        out_path=cache_path,
        timeout_s=timeout_s,
        user_agent=USER_AGENT,
        insecure_ssl=insecure_ssl,
    )
    return cache_path


def _refseq_accession_rank(*, accession: str, status: str) -> tuple[int, int, str]:
    if accession.startswith("NP_"):
        prefix_rank = 0
    elif accession.startswith("XP_"):
        prefix_rank = 1
    elif accession.startswith("WP_"):
        prefix_rank = 2
    else:
        prefix_rank = 3
    status_rank = 0 if status.upper() == "REVIEWED" else 1
    return (prefix_rank, status_rank, accession)


def parse_human_refseq_by_gene(
    gene2refseq_path: Path,
    *,
    gene_ids: set[str],
) -> dict[str, str]:
    """Map NCBI Gene IDs to preferred RefSeq protein accessions (human only)."""
    import gzip

    best: dict[str, tuple[tuple[int, int, str], str]] = {}

    def _consume_lines(lines: Any) -> None:
        for line in lines:
            if not line or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 6:
                continue
            tax_id = parts[0]
            gene_id = parts[1]
            status = parts[2]
            protein = parts[5].strip()
            if tax_id != "9606" or gene_id not in gene_ids:
                continue
            if not protein or protein == "-":
                continue
            rank = _refseq_accession_rank(accession=protein, status=status)
            prev = best.get(gene_id)
            if prev is None or rank < prev[0]:
                best[gene_id] = (rank, protein)

    if gene2refseq_path.suffix == ".gz":
        with gzip.open(gene2refseq_path, "rt", encoding="utf-8") as src:
            _consume_lines(src)
    else:
        with gene2refseq_path.open("r", encoding="utf-8") as src:
            _consume_lines(src)
    return {gene_id: accession for gene_id, (_rank, accession) in best.items()}


def _refseq_rna_accession_rank(*, accession: str, status: str) -> tuple[int, int, str]:
    if accession.startswith("NM_"):
        prefix_rank = 0
    elif accession.startswith("NR_"):
        prefix_rank = 1
    elif accession.startswith(("XM_", "XR_")):
        prefix_rank = 2
    else:
        prefix_rank = 3
    status_rank = 0 if status.upper() == "REVIEWED" else 1
    return (prefix_rank, status_rank, accession)


def parse_human_refseq_rna_by_gene(
    gene2refseq_path: Path,
    *,
    gene_ids: set[str],
) -> dict[str, str]:
    """Map NCBI Gene IDs to preferred RefSeq RNA nucleotide accessions (human only)."""
    import gzip

    best: dict[str, tuple[tuple[int, int, str], str]] = {}

    def _consume_lines(lines: Any) -> None:
        for line in lines:
            if not line or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            tax_id = parts[0]
            gene_id = parts[1]
            status = parts[2]
            rna = parts[3].strip()
            if tax_id != "9606" or gene_id not in gene_ids:
                continue
            if not rna or rna == "-":
                continue
            rank = _refseq_rna_accession_rank(accession=rna, status=status)
            prev = best.get(gene_id)
            if prev is None or rank < prev[0]:
                best[gene_id] = (rank, rna)

    if gene2refseq_path.suffix == ".gz":
        with gzip.open(gene2refseq_path, "rt", encoding="utf-8") as src:
            _consume_lines(src)
    else:
        with gene2refseq_path.open("r", encoding="utf-8") as src:
            _consume_lines(src)
    return {gene_id: accession for gene_id, (_rank, accession) in best.items()}


def _entrez_ids_with_protein_sequence(uniprot_tsv: Path) -> set[str]:
    with_sequence: set[str] = set()
    with uniprot_tsv.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            node_id = (row.get("node_id") or "").strip()
            if not node_id.startswith("primekg:protein:"):
                continue
            entrez = node_id.split(":", 2)[2]
            if (row.get("Sequence") or "").strip():
                with_sequence.add(entrez)
    return with_sequence


def _entrez_ids_with_description_proxy(
    uniprot_tsv: Path,
    hgnc_tsv: Path | None,
) -> set[str]:
    with_text: set[str] = set()
    with uniprot_tsv.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            node_id = (row.get("node_id") or "").strip()
            if not node_id.startswith("primekg:protein:"):
                continue
            entrez = node_id.split(":", 2)[2]
            for col in ("Protein names", "Gene Names"):
                if (row.get(col) or "").strip():
                    with_text.add(entrez)
                    break
    if hgnc_tsv is not None and hgnc_tsv.is_file():
        with hgnc_tsv.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            fieldnames = reader.fieldnames or []
            cols = {h.strip(): h for h in fieldnames if h}
            entrez_col = cols.get("entrez_id")
            symbol_col = cols.get("symbol")
            if entrez_col and symbol_col:
                for row in reader:
                    entrez = (row.get(entrez_col) or "").strip()
                    symbol = (row.get(symbol_col) or "").strip()
                    if entrez.isdigit() and symbol:
                        with_text.add(entrez)
    return with_text


def resolve_nucleotide_export_gene_ids(
    *,
    target_genes: set[str],
    uniprot_tsv: Path,
    hgnc_tsv: Path | None,
    scope: str = NUCLEOTIDE_EXPORT_SCOPE_OR_FAIL_RNA,
) -> set[str]:
    """Select Entrez gene ids for ``write_nucleotide_map_tsv``."""
    with_protein = _entrez_ids_with_protein_sequence(uniprot_tsv)
    if scope == NUCLEOTIDE_EXPORT_SCOPE_ALL_MISSING_PROTEIN:
        return target_genes - with_protein
    if scope != NUCLEOTIDE_EXPORT_SCOPE_OR_FAIL_RNA:
        msg = (
            f"unknown nucleotide export scope {scope!r}; expected "
            f"{NUCLEOTIDE_EXPORT_SCOPE_OR_FAIL_RNA!r} or "
            f"{NUCLEOTIDE_EXPORT_SCOPE_ALL_MISSING_PROTEIN!r}"
        )
        raise ValueError(msg)
    with_text = _entrez_ids_with_description_proxy(uniprot_tsv, hgnc_tsv)
    return target_genes - with_protein - with_text


def fetch_refseq_nucleotide_sequences(
    accessions: list[str],
    *,
    timeout_s: float = 120.0,
    insecure_ssl: bool = False,
    batch_size: int = 100,
    sleep_s: float = 0.34,
) -> dict[str, str]:
    """Fetch RefSeq nucleotide sequences from NCBI efetch (accession → sequence)."""
    from primehkg.io.http_download import fetch_text

    unique = list(dict.fromkeys(accessions))
    sequences: dict[str, str] = {}
    for start in range(0, len(unique), batch_size):
        batch = unique[start : start + batch_size]
        query = urlencode(
            {
                "db": "nucleotide",
                "id": ",".join(batch),
                "rettype": "fasta",
                "retmode": "text",
            }
        )
        text = fetch_text(
            url=f"{EFETCH_NUCLEOTIDE_URL}?{query}",
            timeout_s=timeout_s,
            user_agent=USER_AGENT,
            insecure_ssl=insecure_ssl,
        )
        parsed = parse_fasta_sequences(text)
        for accession in batch:
            base = accession.split(".")[0]
            sequence = parsed.get(base)
            if sequence:
                sequences[accession] = sequence
        if sleep_s > 0 and start + batch_size < len(unique):
            time.sleep(sleep_s)
    return sequences


def write_nucleotide_map_tsv(
    *,
    nodes_tab: Path,
    uniprot_tsv: Path,
    out_path: Path,
    hgnc_tsv: Path | None = None,
    gene2refseq_path: Path | None = None,
    scope: str = NUCLEOTIDE_EXPORT_SCOPE_OR_FAIL_RNA,
    timeout_s: float = 120.0,
    insecure_ssl: bool = False,
) -> int:
    """Export RefSeq RNA sequences for non-coding / OR-fail protein nodes."""
    gene_ids: list[str] = []
    node_id_by_gene: dict[str, str] = {}
    gene_name_by_gene: dict[str, str] = {}
    for node_id, raw_type, raw_name in _iter_nodes_tab_rows(nodes_tab):
        if raw_type != "gene/protein":
            continue
        local_id = node_id.split(":")[-1]
        if not local_id.isdigit():
            continue
        gene_ids.append(local_id)
        node_id_by_gene[local_id] = node_id
        gene_name_by_gene[local_id] = raw_name
    target_genes = set(dict.fromkeys(gene_ids))
    export_genes = resolve_nucleotide_export_gene_ids(
        target_genes=target_genes,
        uniprot_tsv=uniprot_tsv,
        hgnc_tsv=hgnc_tsv,
        scope=scope,
    )
    if not export_genes:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8", newline="") as dst:
            writer = csv.DictWriter(
                dst,
                fieldnames=NUCLEOTIDE_TSV_FIELDNAMES,
                delimiter="\t",
                lineterminator="\n",
            )
            writer.writeheader()
        return 0

    refseq_cache = _resolve_gene2refseq_path(
        out_path=out_path,
        gene2refseq_path=gene2refseq_path,
    )
    if refseq_cache is None:
        msg = "gene2refseq.gz is required for nucleotide export"
        raise FileNotFoundError(msg)

    gene_to_rna = parse_human_refseq_rna_by_gene(
        refseq_cache,
        gene_ids=export_genes,
    )
    if not gene_to_rna:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8", newline="") as dst:
            writer = csv.DictWriter(
                dst,
                fieldnames=NUCLEOTIDE_TSV_FIELDNAMES,
                delimiter="\t",
                lineterminator="\n",
            )
            writer.writeheader()
        return 0

    accession_to_genes: dict[str, list[str]] = {}
    for gene_id, accession in gene_to_rna.items():
        accession_to_genes.setdefault(accession, []).append(gene_id)
    sequences = fetch_refseq_nucleotide_sequences(
        list(accession_to_genes),
        timeout_s=timeout_s,
        insecure_ssl=insecure_ssl,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    with out_path.open("w", encoding="utf-8", newline="") as dst:
        writer = csv.DictWriter(
            dst,
            fieldnames=NUCLEOTIDE_TSV_FIELDNAMES,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        for accession, sequence in sequences.items():
            for gene_id in accession_to_genes.get(accession, []):
                writer.writerow(
                    {
                        "node_id": node_id_by_gene[gene_id],
                        "entrez_id": gene_id,
                        "refseq_rna_accession": accession,
                        "Sequence": sequence,
                        "Gene symbol": gene_name_by_gene.get(gene_id, ""),
                    }
                )
                rows_written += 1
    return rows_written


def parse_fasta_sequences(text: str) -> dict[str, str]:
    """Parse FASTA text into accession (no version) → amino-acid sequence."""
    records: dict[str, str] = {}
    current_key: str | None = None
    chunks: list[str] = []
    for line in text.splitlines():
        if line.startswith(">"):
            if current_key is not None and chunks:
                records[current_key] = "".join(chunks)
            header = line[1:].split()[0]
            accession = header.split("|")[-1] if "|" in header else header
            current_key = accession.split(".")[0]
            chunks = []
            continue
        if current_key is not None and line.strip():
            chunks.append(line.strip())
    if current_key is not None and chunks:
        records[current_key] = "".join(chunks)
    return records


def fetch_refseq_protein_sequences(
    accessions: list[str],
    *,
    timeout_s: float = 120.0,
    insecure_ssl: bool = False,
    batch_size: int = 100,
    sleep_s: float = 0.34,
) -> dict[str, str]:
    """Fetch RefSeq protein sequences from NCBI efetch (accession → sequence)."""
    from primehkg.io.http_download import fetch_text

    unique = list(dict.fromkeys(accessions))
    sequences: dict[str, str] = {}
    for start in range(0, len(unique), batch_size):
        batch = unique[start : start + batch_size]
        query = urlencode(
            {
                "db": "protein",
                "id": ",".join(batch),
                "rettype": "fasta",
                "retmode": "text",
            }
        )
        text = fetch_text(
            url=f"{EFETCH_PROTEIN_URL}?{query}",
            timeout_s=timeout_s,
            user_agent=USER_AGENT,
            insecure_ssl=insecure_ssl,
        )
        parsed = parse_fasta_sequences(text)
        for accession in batch:
            base = accession.split(".")[0]
            sequence = parsed.get(base)
            if sequence:
                sequences[accession] = sequence
        if sleep_s > 0 and start + batch_size < len(unique):
            time.sleep(sleep_s)
    return sequences


def _uniprot_length_mass_from_stream_row(
    row: dict[str, str | None], cols: dict[str, str]
) -> tuple[str, str]:
    length = (row.get(cols.get("Length", "")) or "").strip()
    mass = (row.get(cols.get("Mass", "")) or "").strip()
    return length, mass


def _uniprot_function_from_stream_row(
    row: dict[str, str | None], cols: dict[str, str]
) -> str:
    for key in ("Function [CC]", "Function", "cc_function"):
        if key in cols:
            return (row.get(cols[key]) or "").strip()
    return ""


def _uniprot_row_from_stream(
    *,
    row: dict[str, str | None],
    cols: dict[str, str],
    gene_id: str,
    node_id_by_gene: dict[str, str],
) -> dict[str, str]:
    length, mass = _uniprot_length_mass_from_stream_row(row, cols)
    return {
        "node_id": node_id_by_gene[gene_id],
        "Entry": (row.get(cols.get("Entry", "")) or "").strip(),
        "Length": length,
        "Mass": mass,
        "Protein names": (row.get(cols.get("Protein names", "")) or "").strip(),
        "Gene Names": (row.get(cols.get("Gene Names (primary)", "")) or "").strip(),
        "Function": _uniprot_function_from_stream_row(row, cols),
        "Cross-reference (HGNC)": (row.get(cols.get("HGNC", "")) or "").strip(),
        "Sequence": (row.get(cols.get("Sequence", "")) or "").strip(),
        "Gene ID": gene_id,
    }


def _write_uniprot_rows_from_stream(
    *,
    cache_path: Path,
    out_path: Path,
    target_genes: set[str],
    node_id_by_gene: dict[str, str],
) -> tuple[int, set[str], dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    rows_written = 0
    seen_genes: set[str] = set()
    by_symbol: dict[str, dict[str, str]] = {}
    by_accession: dict[str, dict[str, str]] = {}
    with (
        cache_path.open("r", encoding="utf-8", newline="") as src,
        out_path.open("w", encoding="utf-8", newline="") as dst,
    ):
        reader = csv.DictReader(src, delimiter="\t")
        if reader.fieldnames is None:
            msg = "UniProt stream TSV is missing a header row"
            raise ValueError(msg)
        cols = {h.strip(): h for h in reader.fieldnames if h is not None}
        writer = csv.DictWriter(
            dst,
            fieldnames=UNIPROT_TSV_FIELDNAMES,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in reader:
            entry = (row.get(cols.get("Entry", "")) or "").strip()
            sequence = (row.get(cols.get("Sequence", "")) or "").strip()
            if not entry or not sequence:
                continue
            gene_id_raw = (row.get(cols.get("GeneID", "")) or "").strip()
            matched_genes = [
                gene_id
                for gene_id in _parse_uniprot_gene_ids(gene_id_raw)
                if gene_id in target_genes and gene_id not in seen_genes
            ]
            symbol = (
                (row.get(cols.get("Gene Names (primary)", "")) or "").strip().upper()
            )
            length, mass = _uniprot_length_mass_from_stream_row(row, cols)
            template = {
                "Entry": entry,
                "Length": length,
                "Mass": mass,
                "Protein names": (row.get(cols.get("Protein names", "")) or "").strip(),
                "Gene Names": (
                    row.get(cols.get("Gene Names (primary)", "")) or ""
                ).strip(),
                "Function": _uniprot_function_from_stream_row(row, cols),
                "Cross-reference (HGNC)": (row.get(cols.get("HGNC", "")) or "").strip(),
                "Sequence": sequence,
            }
            if symbol and symbol not in by_symbol:
                by_symbol[symbol] = dict(template)
            accession_key = entry.split(".")[0]
            if accession_key not in by_accession:
                by_accession[accession_key] = dict(template)
            for matched_gene in matched_genes:
                seen_genes.add(matched_gene)
                writer.writerow(
                    _uniprot_row_from_stream(
                        row=row,
                        cols=cols,
                        gene_id=matched_gene,
                        node_id_by_gene=node_id_by_gene,
                    )
                )
                rows_written += 1
    return rows_written, seen_genes, by_symbol, by_accession


def _append_uniprot_template_rows(
    *,
    out_path: Path,
    missing_genes: set[str],
    node_id_by_gene: dict[str, str],
    templates_by_gene: dict[str, dict[str, str]],
) -> tuple[int, set[str]]:
    if not templates_by_gene:
        return 0, set()
    rows_written = 0
    added_genes: set[str] = set()
    with out_path.open("a", encoding="utf-8", newline="") as dst:
        writer = csv.DictWriter(
            dst,
            fieldnames=UNIPROT_TSV_FIELDNAMES,
            delimiter="\t",
            lineterminator="\n",
        )
        for gene_id in sorted(templates_by_gene):
            if gene_id not in missing_genes:
                continue
            template = templates_by_gene[gene_id]
            writer.writerow(
                {
                    "node_id": node_id_by_gene[gene_id],
                    "Gene ID": gene_id,
                    **template,
                }
            )
            rows_written += 1
            added_genes.add(gene_id)
    return rows_written, added_genes


def _append_symbol_uniprot_fallback_rows(
    *,
    out_path: Path,
    missing_genes: set[str],
    node_id_by_gene: dict[str, str],
    gene_name_by_gene: dict[str, str],
    by_symbol: dict[str, dict[str, str]],
) -> tuple[int, set[str]]:
    templates: dict[str, dict[str, str]] = {}
    for gene_id in missing_genes:
        symbol = gene_name_by_gene.get(gene_id, "").strip().upper()
        template = by_symbol.get(symbol)
        if template is not None:
            templates[gene_id] = template
    return _append_uniprot_template_rows(
        out_path=out_path,
        missing_genes=missing_genes,
        node_id_by_gene=node_id_by_gene,
        templates_by_gene=templates,
    )


def _append_hgnc_uniprot_fallback_rows(
    *,
    out_path: Path,
    missing_genes: set[str],
    node_id_by_gene: dict[str, str],
    hgnc_path: Path,
    by_accession: dict[str, dict[str, str]],
) -> tuple[int, set[str]]:
    if not hgnc_path.is_file() or not missing_genes:
        return 0, set()
    templates: dict[str, dict[str, str]] = {}
    with hgnc_path.open("r", encoding="utf-8", newline="") as src:
        reader = csv.DictReader(src, delimiter="\t")
        if not reader.fieldnames:
            return 0, set()
        for row in reader:
            gene_id = (row.get("entrez_id") or "").strip()
            if not gene_id or gene_id not in missing_genes or gene_id in templates:
                continue
            uniprot_ids = (row.get("uniprot_ids") or "").strip()
            if not uniprot_ids:
                continue
            for raw_accession in uniprot_ids.split("|"):
                accession = raw_accession.strip().split(".")[0]
                if not accession:
                    continue
                template = by_accession.get(accession)
                if template is not None:
                    templates[gene_id] = template
                    break
    return _append_uniprot_template_rows(
        out_path=out_path,
        missing_genes=missing_genes,
        node_id_by_gene=node_id_by_gene,
        templates_by_gene=templates,
    )


def _resolve_gene2refseq_path(
    *,
    out_path: Path,
    gene2refseq_path: Path | None,
) -> Path | None:
    if gene2refseq_path is not None and gene2refseq_path.is_file():
        return gene2refseq_path
    for candidate in (out_path.with_name("gene2refseq.gz"), Path("gene2refseq.gz")):
        if candidate.is_file():
            return candidate
    return None


def _append_refseq_fallback_rows(
    *,
    out_path: Path,
    missing_genes: set[str],
    node_id_by_gene: dict[str, str],
    gene2refseq_path: Path,
    timeout_s: float,
    insecure_ssl: bool,
) -> int:
    if not missing_genes:
        return 0
    gene_to_accession = parse_human_refseq_by_gene(
        gene2refseq_path,
        gene_ids=missing_genes,
    )
    if not gene_to_accession:
        return 0
    accession_to_genes: dict[str, list[str]] = {}
    for gene_id, accession in gene_to_accession.items():
        accession_to_genes.setdefault(accession, []).append(gene_id)
    sequences = fetch_refseq_protein_sequences(
        list(accession_to_genes),
        timeout_s=timeout_s,
        insecure_ssl=insecure_ssl,
    )
    rows_written = 0
    with out_path.open("a", encoding="utf-8", newline="") as dst:
        writer = csv.DictWriter(
            dst,
            fieldnames=UNIPROT_TSV_FIELDNAMES,
            delimiter="\t",
            lineterminator="\n",
        )
        for accession, sequence in sequences.items():
            for gene_id in accession_to_genes.get(accession, []):
                writer.writerow(
                    {
                        "node_id": node_id_by_gene[gene_id],
                        "Entry": accession,
                        "Length": "",
                        "Mass": "",
                        "Protein names": "",
                        "Gene Names": "",
                        "Function": "",
                        "Cross-reference (HGNC)": "",
                        "Sequence": sequence,
                        "Gene ID": gene_id,
                    }
                )
                rows_written += 1
    return rows_written


def write_uniprot_tsv(
    *,
    nodes_tab: Path,
    out_path: Path,
    timeout_s: float = 120.0,
    insecure_ssl: bool = False,
    refseq_fallback: bool = True,
    gene2refseq_path: Path | None = None,
    hgnc_path: Path | None = None,
) -> int:
    """Map PrimeKG NCBI gene ids → UniProt (+ symbol/HGNC/RefSeq fallbacks)."""
    gene_ids: list[str] = []
    node_id_by_gene: dict[str, str] = {}
    gene_name_by_gene: dict[str, str] = {}
    for node_id, raw_type, raw_name in _iter_nodes_tab_rows(nodes_tab):
        if raw_type != "gene/protein":
            continue
        local_id = node_id.split(":")[-1]
        if not local_id.isdigit():
            continue
        gene_ids.append(local_id)
        node_id_by_gene[local_id] = node_id
        gene_name_by_gene[local_id] = raw_name
    target_genes = set(dict.fromkeys(gene_ids))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path = out_path.with_name("uniprot_human.tsv")
    if not cache_path.is_file():
        _download_uniprot_human_tsv(
            cache_path=cache_path,
            timeout_s=timeout_s,
            insecure_ssl=insecure_ssl,
        )
    rows_written, seen_genes, by_symbol, by_accession = _write_uniprot_rows_from_stream(
        cache_path=cache_path,
        out_path=out_path,
        target_genes=target_genes,
        node_id_by_gene=node_id_by_gene,
    )

    missing_genes = target_genes - seen_genes
    sym_rows, sym_added = _append_symbol_uniprot_fallback_rows(
        out_path=out_path,
        missing_genes=missing_genes,
        node_id_by_gene=node_id_by_gene,
        gene_name_by_gene=gene_name_by_gene,
        by_symbol=by_symbol,
    )
    rows_written += sym_rows
    seen_genes |= sym_added
    missing_genes = target_genes - seen_genes

    resolved_hgnc = (
        hgnc_path if hgnc_path is not None else out_path.with_name("hgnc.tsv")
    )
    hgnc_rows, hgnc_added = _append_hgnc_uniprot_fallback_rows(
        out_path=out_path,
        missing_genes=missing_genes,
        node_id_by_gene=node_id_by_gene,
        hgnc_path=resolved_hgnc,
        by_accession=by_accession,
    )
    rows_written += hgnc_rows
    seen_genes |= hgnc_added
    missing_genes = target_genes - seen_genes

    if refseq_fallback and missing_genes:
        refseq_cache = _resolve_gene2refseq_path(
            out_path=out_path,
            gene2refseq_path=gene2refseq_path,
        )
        if refseq_cache is None:
            refseq_cache = out_path.with_name("gene2refseq.gz")
            download_file(
                url=DEFAULT_GENE2REFSEQ_URL,
                out_path=refseq_cache,
                timeout_s=max(timeout_s, 600.0),
                user_agent=USER_AGENT,
                insecure_ssl=insecure_ssl,
            )
        rows_written += _append_refseq_fallback_rows(
            out_path=out_path,
            missing_genes=missing_genes,
            node_id_by_gene=node_id_by_gene,
            gene2refseq_path=refseq_cache,
            timeout_s=timeout_s,
            insecure_ssl=insecure_ssl,
        )
    return rows_written


def _pubchem_drugbank_lookup(
    drugbank_id: str,
    *,
    timeout_s: float,
    insecure_ssl: bool,
) -> tuple[str, str, str] | None:
    """Resolve DrugBank id → (chembl_id, smiles, name) via PubChem RegistryID xrefs."""
    url = (
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/xref/RegistryID/"
        f"{quote(drugbank_id)}/property/SMILES,InChIKey/JSON"
    )
    payload: dict[str, object] | None = None
    for attempt in range(3):
        try:
            payload = get_json(
                url=url,
                timeout_s=timeout_s,
                user_agent=USER_AGENT,
                insecure_ssl=insecure_ssl,
            )
            break
        except urllib.error.HTTPError:
            return None
        except (urllib.error.URLError, TimeoutError, OSError, RemoteDisconnected):
            if attempt >= 2:
                return None
            time.sleep(1.0 * (attempt + 1))
    if payload is None:
        return None
    properties = (payload.get("PropertyTable") or {}).get("Properties") or []
    if not properties:
        return None
    props = properties[0]
    smiles = str(props.get("SMILES") or props.get("IsomericSMILES") or "").strip()
    if not smiles:
        return None
    return "", smiles, ""


def _load_smiles_by_node_from_export(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    out: dict[str, str] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None or "node_id" not in reader.fieldnames:
            return {}
        smiles_key = (
            "canonical_smiles"
            if "canonical_smiles" in reader.fieldnames
            else "smiles"
            if "smiles" in reader.fieldnames
            else None
        )
        if smiles_key is None:
            return {}
        for row in reader:
            node_id = (row.get("node_id") or "").strip()
            smiles = (row.get(smiles_key) or "").strip()
            if node_id and smiles:
                out[node_id] = smiles
    return out


def write_drugbank_smiles_tsv(
    *,
    nodes_tab: Path,
    chembl_path: Path,
    drugbank_xml: Path,
    out_path: Path,
) -> int:
    """Write ``drugbank_smiles.tsv`` for drugs missing SMILES in ``chembl_drugs.tsv``.

    SMILES are resolved from a maintainer-owned DrugBank full-database XML export.
    """
    chembl_smiles = _load_smiles_by_node_from_export(chembl_path)
    drugbank_index = load_drugbank_smiles_index(drugbank_xml)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    with out_path.open("w", encoding="utf-8", newline="") as dst:
        writer = csv.DictWriter(
            dst,
            fieldnames=["node_id", "drugbank_id", "canonical_smiles"],
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        for node_id, _raw_type, _ in _iter_nodes_tab_rows(nodes_tab):
            if _raw_type != "drug":
                continue
            if node_id in chembl_smiles:
                continue
            drugbank_id = node_id.split(":")[-1]
            smiles = drugbank_index.get(drugbank_id)
            if not smiles:
                continue
            writer.writerow(
                {
                    "node_id": node_id,
                    "drugbank_id": drugbank_id,
                    "canonical_smiles": smiles,
                }
            )
            rows_written += 1
    return rows_written


def write_drugbank_properties_tsv(
    *,
    nodes_tab: Path,
    drugbank_xml: Path,
    out_path: Path,
) -> int:
    """Write ``drugbank_properties.tsv`` from DrugBank calculated/experimental properties."""
    prop_index = load_drugbank_properties_index(drugbank_xml)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    with out_path.open("w", encoding="utf-8", newline="") as dst:
        writer = csv.DictWriter(
            dst,
            fieldnames=list(DRUGBANK_PROPERTIES_FIELDNAMES),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        for node_id, raw_type, _ in _iter_nodes_tab_rows(nodes_tab):
            if raw_type != "drug":
                continue
            drugbank_id = node_id.split(":")[-1]
            props = prop_index.get(drugbank_id)
            if not props:
                continue
            row: dict[str, str] = {
                "node_id": node_id,
                "drugbank_id": drugbank_id,
            }
            has_numeric = False
            for col in CHEMBL_DRUG_PROPERTY_COLUMNS:
                val = props.get(col)
                if val is not None:
                    row[col] = str(val)
                    has_numeric = True
                else:
                    row[col] = ""
            if not has_numeric:
                continue
            writer.writerow(row)
            rows_written += 1
    return rows_written


def _resolve_drug_property_columns(
    *,
    chembl_id: str,
    drugbank_id: str,
    timeout_s: float,
    insecure_ssl: bool,
) -> dict[str, float]:
    """ChEMBL molecule properties when ``chembl_id`` set; PubChem gap-fill per column."""
    out: dict[str, float] = {}
    cid = chembl_id.strip()
    if cid:
        out.update(
            fetch_chembl_molecule_properties(
                cid, timeout_s=timeout_s, insecure_ssl=insecure_ssl
            )
        )
    pubchem = fetch_pubchem_properties_for_drugbank_id(
        drugbank_id, timeout_s=timeout_s, insecure_ssl=insecure_ssl
    )
    for col in CHEMBL_DRUG_PROPERTY_COLUMNS:
        if col not in out and col in pubchem:
            out[col] = pubchem[col]
    return out


def write_chembl_drugs_tsv(
    *,
    nodes_tab: Path,
    out_path: Path,
    timeout_s: float = 60.0,
    insecure_ssl: bool = False,
    sleep_s: float = 0.15,
    max_drugs: int | None = None,
) -> int:
    """Write ``chembl_drugs.tsv`` with SMILES and optional physicochemical columns.

    SMILES are resolved from PubChem RegistryID xrefs (DrugBank ids). ``chembl_id`` is
    left empty when only PubChem structure resolution is available. Numeric columns use
    ChEMBL when ``chembl_id`` is set, with PubChem gap-fill per column.
    """
    drug_rows: list[tuple[str, str]] = []
    for node_id, raw_type, _ in _iter_nodes_tab_rows(nodes_tab):
        if raw_type != "drug":
            continue
        drugbank_id = node_id.split(":")[-1]
        if not drugbank_id:
            continue
        drug_rows.append((node_id, drugbank_id))
    if max_drugs is not None:
        drug_rows = drug_rows[: max(0, max_drugs)]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    done_node_ids: set[str] = set()
    append_existing = out_path.is_file() and out_path.stat().st_size > 0
    if append_existing:
        with out_path.open("r", encoding="utf-8", newline="") as existing:
            reader = csv.DictReader(existing, delimiter="\t")
            if reader.fieldnames and "node_id" in reader.fieldnames:
                for row in reader:
                    node_id = (row.get("node_id") or "").strip()
                    if node_id:
                        done_node_ids.add(node_id)

    rows_written = len(done_node_ids)
    with out_path.open(
        "a" if append_existing else "w", encoding="utf-8", newline=""
    ) as dst:
        chembl_fieldnames = [
            "node_id",
            "chembl_id",
            "canonical_smiles",
            "chembl_name",
            *CHEMBL_DRUG_PROPERTY_COLUMNS,
        ]
        writer = csv.DictWriter(
            dst,
            fieldnames=chembl_fieldnames,
            delimiter="\t",
            lineterminator="\n",
        )
        if not append_existing:
            writer.writeheader()
            dst.flush()
        for node_id, drugbank_id in drug_rows:
            if node_id in done_node_ids:
                continue
            result = _pubchem_drugbank_lookup(
                drugbank_id,
                timeout_s=timeout_s,
                insecure_ssl=insecure_ssl,
            )
            if result is None:
                if sleep_s > 0:
                    time.sleep(sleep_s)
                continue
            chembl_id, smiles, name = result
            if not chembl_id.strip():
                chembl_id, chembl_name = fetch_chembl_xref_for_drugbank_id(
                    drugbank_id,
                    timeout_s=timeout_s,
                    insecure_ssl=insecure_ssl,
                )
                if chembl_name and not name:
                    name = chembl_name
            props = _resolve_drug_property_columns(
                chembl_id=chembl_id,
                drugbank_id=drugbank_id,
                timeout_s=timeout_s,
                insecure_ssl=insecure_ssl,
            )
            row_out: dict[str, str] = {
                "node_id": node_id,
                "chembl_id": chembl_id,
                "canonical_smiles": smiles,
                "chembl_name": name,
            }
            for col in CHEMBL_DRUG_PROPERTY_COLUMNS:
                val = props.get(col)
                row_out[col] = str(val) if val is not None else ""
            writer.writerow(row_out)
            dst.flush()
            rows_written += 1
            done_node_ids.add(node_id)
            if sleep_s > 0:
                time.sleep(sleep_s)
    return rows_written


def write_hgnc_tsv(
    *,
    out_path: Path,
    url: str = DEFAULT_HGNC_JSON_URL,
    timeout_s: float = 120.0,
    insecure_ssl: bool = False,
) -> int:
    """Download HGNC complete set JSON and write ``hgnc.tsv``."""
    cache_json = out_path.with_suffix(".json")
    download_file(
        url=url,
        out_path=cache_json,
        timeout_s=timeout_s,
        user_agent=USER_AGENT,
        insecure_ssl=insecure_ssl,
    )
    payload = json.loads(cache_json.read_text(encoding="utf-8"))
    docs = payload.get("response", {}).get("docs") or []
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    with out_path.open("w", encoding="utf-8", newline="") as dst:
        writer = csv.DictWriter(
            dst,
            fieldnames=[
                "hgnc_id",
                "symbol",
                "name",
                "uniprot_ids",
                "ensembl_gene_id",
                "entrez_id",
            ],
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        for doc in docs:
            hgnc_id = str(doc.get("hgnc_id") or "").strip()
            symbol = str(doc.get("symbol") or "").strip()
            if not hgnc_id or not symbol:
                continue
            uniprot_ids = doc.get("uniprot_ids") or []
            if isinstance(uniprot_ids, list):
                uniprot_value = "|".join(str(v) for v in uniprot_ids if v)
            else:
                uniprot_value = str(uniprot_ids)
            writer.writerow(
                {
                    "hgnc_id": hgnc_id.removeprefix("HGNC:"),
                    "symbol": symbol,
                    "name": str(doc.get("name") or "").strip(),
                    "uniprot_ids": uniprot_value,
                    "ensembl_gene_id": str(doc.get("ensembl_gene_id") or "").strip(),
                    "entrez_id": str(doc.get("entrez_id") or "").strip(),
                }
            )
            rows_written += 1
    return rows_written


def export_extended_mappings(
    *,
    primekg_dir: Path,
    out_dir: Path,
    skip_network: bool = False,
    include_hpo_definitions: bool = True,
    timeout_s: float = 120.0,
    insecure_ssl: bool = False,
    max_drugs: int | None = None,
    force: bool = False,
    drugbank_xml: Path | None = None,
) -> ExportCounts:
    """Generate all extended export TSVs under ``out_dir``."""
    kg_csv = primekg_dir / "kg.csv"
    nodes_tab = primekg_dir / "nodes.tab"
    disease_features_tab = primekg_dir / "disease_features.tab"
    for path in (kg_csv, nodes_tab, disease_features_tab):
        if not path.is_file():
            msg = f"PrimeKG input missing: {path}"
            raise FileNotFoundError(msg)

    out_dir.mkdir(parents=True, exist_ok=True)
    mondo_path = out_dir / "mondo_map.tsv"
    mondo_count = _existing_export_count(mondo_path, min_rows=40_000, force=force)
    if mondo_count is None:
        mondo_count = write_mondo_map_tsv(
            kg_csv=kg_csv,
            disease_features_tab=disease_features_tab,
            out_path=mondo_path,
        )

    hpo_path = out_dir / "hpo_map.tsv"
    hpo_count = _existing_export_count(hpo_path, min_rows=15_000, force=force)
    hpo_definitions: dict[str, str] | None = None
    hpo_term_stats: dict[str, HpoTermStats] | None = None
    hpoa_aggregates: dict[str, HpoaTermAggregates] | None = None
    hpo_graph_metrics: dict[str, HpoGraphMetrics] | None = None
    if hpo_count is None:
        if include_hpo_definitions and not skip_network:
            obo_path = out_dir / "hpo.obo"
            if force or not obo_path.is_file():
                download_hpo_obo(
                    out_path=obo_path,
                    timeout_s=timeout_s,
                    insecure_ssl=insecure_ssl,
                )
            hpo_definitions = parse_hpo_obo_definitions(obo_path)
            hpo_term_stats = parse_hpo_obo_term_stats(obo_path)
            parents = parse_hpo_obo_is_a_parents(obo_path)
            hpo_graph_metrics = compute_hpo_graph_metrics(parents)
            hpoa_path = out_dir / "phenotype.hpoa"
            if force or not hpoa_path.is_file():
                download_phenotype_hpoa(
                    out_path=hpoa_path,
                    timeout_s=timeout_s,
                    insecure_ssl=insecure_ssl,
                )
            mondo_obo_path = out_dir / "mondo.obo"
            if force or not mondo_obo_path.is_file():
                download_mondo_obo(
                    out_path=mondo_obo_path,
                    timeout_s=timeout_s,
                    insecure_ssl=insecure_ssl,
                )
            database_id_to_mondo = parse_mondo_disease_xrefs(mondo_obo_path)
            primekg_disease_node_ids, mondo_id_to_node_id = (
                _primekg_disease_bridge_from_mondo_map(mondo_path)
            )
            hpoa_aggregates = parse_phenotype_hpoa_aggregates(
                hpoa_path,
                database_id_to_mondo=database_id_to_mondo,
                primekg_disease_node_ids=primekg_disease_node_ids,
                mondo_id_to_node_id=mondo_id_to_node_id,
            )
        hpo_count = write_hpo_map_tsv(
            nodes_tab=nodes_tab,
            out_path=hpo_path,
            hpo_definitions=hpo_definitions,
            hpo_term_stats=hpo_term_stats,
            hpoa_aggregates=hpoa_aggregates,
            hpo_graph_metrics=hpo_graph_metrics,
        )

    pathway_path = out_dir / "pathway_map.tsv"
    pathway_count = _existing_export_count(pathway_path, min_rows=2_000, force=force)
    if pathway_count is None:
        pathway_count = write_pathway_map_tsv(
            nodes_tab=nodes_tab,
            out_path=pathway_path,
        )

    anatomy_path = out_dir / "anatomy_map.tsv"
    anatomy_count = _existing_export_count(anatomy_path, min_rows=10_000, force=force)
    if anatomy_count is None:
        anatomy_count = write_anatomy_map_tsv(
            nodes_tab=nodes_tab,
            out_path=anatomy_path,
        )

    exposure_path = out_dir / "exposure_map.tsv"
    exposure_count = _existing_export_count(exposure_path, min_rows=700, force=force)
    if exposure_count is None:
        exposure_count = write_exposure_map_tsv(
            nodes_tab=nodes_tab,
            out_path=exposure_path,
        )

    go_path = out_dir / "go_map.tsv"
    go_count = _existing_export_count(go_path, min_rows=40_000, force=force)
    if go_count is None:
        go_obo_path: Path | None = None
        if not skip_network:
            go_obo_path = out_dir / "go-basic.obo"
            if force or not go_obo_path.is_file():
                download_go_obo(
                    out_path=go_obo_path,
                    timeout_s=timeout_s,
                    insecure_ssl=insecure_ssl,
                )
        go_count = write_go_map_tsv(
            nodes_tab=nodes_tab,
            go_obo=go_obo_path,
            out_path=go_path,
        )

    uniprot_count = 0
    nucleotide_count = 0
    chembl_count = 0
    drugbank_smiles_count = 0
    drugbank_properties_count = 0
    hgnc_count = 0
    if not skip_network:
        hgnc_path = out_dir / "hgnc.tsv"
        hgnc_count = _existing_export_count(hgnc_path, min_rows=40_000, force=force)
        if hgnc_count is None:
            hgnc_count = write_hgnc_tsv(
                out_path=hgnc_path,
                timeout_s=timeout_s,
                insecure_ssl=insecure_ssl,
            )

        uniprot_path = out_dir / "uniprot.tsv"
        uniprot_count = _existing_export_count(
            uniprot_path, min_rows=26_000, force=force
        )
        if uniprot_count is None:
            uniprot_count = write_uniprot_tsv(
                nodes_tab=nodes_tab,
                out_path=uniprot_path,
                timeout_s=timeout_s,
                insecure_ssl=insecure_ssl,
                hgnc_path=hgnc_path,
            )
        nucleotide_path = out_dir / "nucleotide_map.tsv"
        nucleotide_count = _existing_export_count(
            nucleotide_path, min_rows=1, force=force
        )
        if nucleotide_count is None and uniprot_path.is_file():
            nucleotide_count = write_nucleotide_map_tsv(
                nodes_tab=nodes_tab,
                uniprot_tsv=uniprot_path,
                out_path=nucleotide_path,
                hgnc_tsv=hgnc_path,
                timeout_s=timeout_s,
                insecure_ssl=insecure_ssl,
            )
        elif nucleotide_count is None:
            nucleotide_count = 0
        chembl_path = out_dir / "chembl_drugs.tsv"
        chembl_count = _existing_export_count(chembl_path, min_rows=7_000, force=force)
        if chembl_count is None:
            chembl_count = write_chembl_drugs_tsv(
                nodes_tab=nodes_tab,
                out_path=chembl_path,
                timeout_s=timeout_s,
                insecure_ssl=insecure_ssl,
                max_drugs=max_drugs,
                sleep_s=0.05,
            )
        if drugbank_xml is not None:
            drugbank_smiles_path = out_dir / "drugbank_smiles.tsv"
            drugbank_smiles_count = _existing_export_count(
                drugbank_smiles_path,
                min_rows=1,
                force=force,
            )
            if drugbank_smiles_count is None:
                drugbank_smiles_count = write_drugbank_smiles_tsv(
                    nodes_tab=nodes_tab,
                    chembl_path=chembl_path,
                    drugbank_xml=drugbank_xml,
                    out_path=drugbank_smiles_path,
                )
            drugbank_properties_path = out_dir / "drugbank_properties.tsv"
            drugbank_properties_count = _existing_export_count(
                drugbank_properties_path,
                min_rows=1,
                force=force,
            )
            if drugbank_properties_count is None:
                drugbank_properties_count = write_drugbank_properties_tsv(
                    nodes_tab=nodes_tab,
                    drugbank_xml=drugbank_xml,
                    out_path=drugbank_properties_path,
                )

    manifest = {
        "primekg_dir": str(primekg_dir.resolve()),
        "out_dir": str(out_dir.resolve()),
        "skip_network": skip_network,
        "counts": {
            "mondo_map": mondo_count,
            "hpo_map": hpo_count,
            "go_map": go_count,
            "pathway_map": pathway_count,
            "anatomy_map": anatomy_count,
            "exposure_map": exposure_count,
            "uniprot": uniprot_count,
            "nucleotide_map": nucleotide_count,
            "chembl_drugs": chembl_count,
            "drugbank_smiles": drugbank_smiles_count,
            "drugbank_properties": drugbank_properties_count,
            "hgnc": hgnc_count,
        },
    }
    (out_dir / "export_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return ExportCounts(
        mondo_map=mondo_count,
        hpo_map=hpo_count,
        go_map=go_count,
        pathway_map=pathway_count,
        anatomy_map=anatomy_count,
        exposure_map=exposure_count,
        uniprot=uniprot_count,
        nucleotide_map=nucleotide_count,
        chembl_drugs=chembl_count,
        drugbank_smiles=drugbank_smiles_count,
        drugbank_properties=drugbank_properties_count,
        hgnc=hgnc_count,
    )
