from __future__ import annotations

_EMPTY_NAMESPACE = "empty namespace"
_EMPTY_ID = "empty id"
_INVALID_HGNC_ID = "invalid hgnc id (expected digits)"
_INVALID_HPO_ID = "invalid hpo id (expected digits or HP:digits)"
_INVALID_GO_ID = "invalid go id (expected digits or GO:digits)"
_INVALID_MONDO_ID = "invalid mondo id (expected digits or MONDO:digits)"


def normalize_namespace(ns: str) -> str:
    out = (ns or "").strip().lower()
    if not out:
        raise ValueError(_EMPTY_NAMESPACE)
    return out


def normalize_xref_id(namespace: str, raw_id: str) -> str:
    ns = normalize_namespace(namespace)
    rid = (raw_id or "").strip()
    if not rid:
        raise ValueError(_EMPTY_ID)

    if ns == "umls":
        return rid.upper()

    if ns == "mondo":
        r = rid.upper()
        local = r.removeprefix("MONDO:")
        if not local.isdigit():
            raise ValueError(_INVALID_MONDO_ID)
        return f"MONDO:{local.zfill(7)}"

    if ns == "mesh":
        # Disease / chemical MeSH: canonical CURIE MESH:D… / MESH:C…
        r = rid.strip().upper()
        local = r.removeprefix("MESH:")
        if not local:
            raise ValueError(_EMPTY_ID)
        return f"MESH:{local}"

    if ns == "hpo":
        # Human Phenotype Ontology: canonical CURIE HP:0000001 (7-digit local id).
        r = rid.strip().upper()
        num = r[3:].strip() if r.startswith("HP:") else r
        if not num.isdigit():
            raise ValueError(_INVALID_HPO_ID)
        return f"HP:{num.zfill(7)}"

    if ns == "drugbank":
        # DrugBank IDs are typically like DB00945.
        r = rid.strip()
        if ":" in r:
            # Accept CURIE-like inputs (e.g. drugbank:DB00945) and keep only the id payload.
            r = r.split(":", 1)[1]
        return r.upper()

    if ns == "hgnc":
        r = rid.strip().upper()
        while ":" in r:
            head, tail = r.split(":", 1)
            if head != "HGNC":
                break
            r = tail
        # HGNC IDs are numeric; avoid minting HGNC:<symbol>.
        if not r.isdigit():
            raise ValueError(_INVALID_HGNC_ID)
        return f"HGNC:{r}"

    if ns == "go":
        r = rid.strip().upper()
        num = r[3:].strip() if r.startswith("GO:") else r
        if not num.isdigit():
            raise ValueError(_INVALID_GO_ID)
        return f"GO:{num.zfill(7)}"

    if ns == "uniprot":
        # UniProt accessions are case-insensitive; normalize to uppercase.
        r = rid.strip()
        if ":" in r:
            head, tail = r.split(":", 1)
            if head.strip().lower() in {"uniprot", "uniprotkb"}:
                r = tail
        return r.upper()

    if ns == "reactome":
        r = rid.strip()
        if r.lower().startswith("r-hsa-"):
            return f"R-HSA-{r[6:]}"
        return r

    return rid


def infer_node_type_and_name(node_id: str) -> tuple[str, str]:
    """Best-effort ``node_type`` / display ``name`` for v0 minimal rows.

    Recognizes ``primekg:<category>:<local>`` (and longer tails) and sets
    ``node_type=<category>``, ``name=<last segment>``. ``chebi:<local>`` curated
    small-molecule ids are typed ``metabolite``. Anything else uses
    ``node_type="unknown"`` and ``name=node_id``.
    """
    parts = node_id.split(":")
    if len(parts) >= 2 and parts[0] == "chebi":
        name = parts[-1].strip() or node_id
        return "metabolite", name
    if len(parts) >= 2 and parts[0] == "clinvar":
        name = parts[-1].strip() or node_id
        return "variant", name
    if len(parts) >= 3 and parts[0] == "primekg":
        node_type = parts[1].strip() or "unknown"
        name = parts[-1].strip() or node_id
        return node_type, name
    if len(parts) >= 2:
        node_type = parts[0].strip() or "unknown"
        name = parts[-1].strip() or node_id
        return node_type, name
    return "unknown", node_id
