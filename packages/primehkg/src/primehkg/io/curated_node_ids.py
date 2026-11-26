"""Stable ``node_id`` strings for curated CORUM / Reactome builders.

PrimeKG-derived endpoints in this repo use typed ids such as ``primekg:protein:<local>``.
UniProt accessions from curated DB rows are mapped to that form so hyperedges can join
to enriched protein rows (e.g. ``read_uniprot_updates``) and typical PrimeKG-style tests.

Small-molecule ChEBI identifiers remain in a separate ``chebi:`` namespace until an
explicit ChEBI → ``primekg:drug:*`` join table is supplied (see docs).
"""

from __future__ import annotations


def normalize_uniprot_accession(raw: str) -> str:
    """Normalize a UniProt accession column value to a bare AC string (uppercase)."""
    upper = raw.strip().upper()
    for prefix in ("UNIPROTKB:", "UNIPROT:", "UNIPROTID:", "SWISSPROT:"):
        if upper.startswith(prefix):
            return upper.removeprefix(prefix)
    return upper


def primekg_protein_node_id_from_uniprot(
    raw: str, *, uniprot_to_geneid: dict[str, str] | None = None
) -> str:
    """Return ``primekg:protein:<GeneID>`` when the accession resolves via ``uniprot_to_geneid``.

    Falls back to ``primekg:protein:<AC>`` when the map is absent or the accession is not
    found (default None keeps the AC-keyed behavior).
    """
    ac = normalize_uniprot_accession(raw)
    if uniprot_to_geneid:
        gid = uniprot_to_geneid.get(ac)
        if gid:
            return f"primekg:protein:{gid}"
    return f"primekg:protein:{ac}"


def remap_protein_updates_to_geneid(
    node_updates: dict[str, dict],
    uniprot_to_geneid: dict[str, str] | None,
) -> dict[str, dict]:
    """Rekey ``primekg:protein:<AC>`` enrichment updates to ``primekg:protein:<GeneID>``.

    For each update keyed ``primekg:protein:<AC>`` whose AC resolves via
    ``uniprot_to_geneid``, move it to ``primekg:protein:<GeneID>`` so it lands on the same
    unified node the curated builders emit. Non-protein keys and unresolved ACs pass through
    unchanged. On collision (two ACs -> one GeneID) the later entry wins (deterministic by
    input dict order). Empty/None map -> identity.
    """
    if not uniprot_to_geneid:
        return node_updates
    prefix = "primekg:protein:"
    out: dict[str, dict] = {}
    for nid, upd in node_updates.items():
        if isinstance(nid, str) and nid.startswith(prefix):
            ac = nid[len(prefix) :]
            gid = uniprot_to_geneid.get(ac)
            if gid:
                out[f"{prefix}{gid}"] = upd
                continue
        out[nid] = upd
    return out


def curated_chebi_node_id(raw: str) -> str:
    """Normalize a ChEBI reference to ``chebi:<digits/letters>`` (lowercase chebi: prefix)."""
    s = raw.strip()
    upper = s.upper()
    if upper.startswith("CHEBI:"):
        tail = upper.removeprefix("CHEBI:")
        return f"chebi:{tail}"
    return f"chebi:{upper}"


def stitch_chemical_to_node_id(
    stitch_id: str, cid_map: dict[str, dict[str, str | None]]
) -> str | None:
    """Resolve a STITCH chemical id to an existing node id via ``cid_map``.

    STITCH chemical ids are ``CIDm``/``CIDs`` + zero-padded PubChem CID; ``cid_map``
    maps a bare CID -> {"drugbank","chebi"}. Prefers DrugBank
    (``primekg:drug:<DBid>``); falls back to ChEBI (``chebi:<local>`` via
    ``curated_chebi_node_id``). Returns ``None`` when the id has no ``CIDm``/``CIDs``
    prefix or the CID is absent / resolves to neither.
    """
    s = stitch_id.strip()
    if not s.startswith(("CIDm", "CIDs")):
        return None
    bare = s[4:].lstrip("0") or "0"
    entry = cid_map.get(bare)
    if not entry:
        return None
    drugbank = entry.get("drugbank")
    if drugbank:
        return f"primekg:drug:{drugbank}"
    chebi = entry.get("chebi")
    if chebi:
        return curated_chebi_node_id(chebi)
    return None


def stitch_protein_to_node_id(ensp: str, ensp_to_geneid: dict[str, str]) -> str | None:
    """Resolve a STITCH protein id to a node id via ``ensp_to_geneid``.

    Maps a ``9606.ENSP…`` id to ``primekg:protein:<GeneID>``; ``None`` when
    unresolved.
    """
    gid = ensp_to_geneid.get(ensp.strip())
    if not gid:
        return None
    return f"primekg:protein:{gid}"


def ctd_chemical_to_node_id(
    mesh: str,
    mesh_to_cid: dict[str, str],
    cid_map: dict[str, dict[str, str | None]],
) -> str | None:
    """Resolve a CTD chemical MeSH id to an existing node id.

    .. deprecated::
        Scalar / first-win bridge kept for CTD builders. Prefer
        :func:`ctd_chemical_to_node_candidates` for lossless consumer paths.

    MeSH -> PubChem CID (``mesh_to_cid``) -> DrugBank/ChEBI (``cid_map``). Prefers
    DrugBank (``primekg:drug:<DBid>``), falls back to ChEBI (``chebi:<local>`` via
    ``curated_chebi_node_id``). Returns ``None`` if unresolved at any hop.
    """
    if not mesh:
        return None
    cid = mesh_to_cid.get(mesh.strip())
    if not cid:
        return None
    entry = cid_map.get(cid)
    if not entry:
        return None
    drugbank = entry.get("drugbank")
    if drugbank:
        return f"primekg:drug:{drugbank}"
    chebi = entry.get("chebi")
    if chebi:
        return curated_chebi_node_id(chebi)
    return None


def ctd_chemical_to_node_candidates(
    mesh: str,
    mesh_to_cids: dict[str, list[str]],
    cid_xrefs: dict[str, list[tuple[str, str]]],
) -> list[str]:
    """Return all DrugBank/ChEBI node_id candidates for a MeSH chemical.

    Expands every MeSH→CID and every CID→(namespace, external_id) without
    first-win collapse. Does **not** filter by live core membership — callers
    (or ``chemical_xref_v1`` composition) apply that separately.

    Order is deterministic: CIDs sorted, then ``(namespace, external_id)``,
    DrugBank terminals before ChEBI within the same CID group only insofar as
    ``namespace`` sorts (``chebi`` < ``drugbank``); final list is unique
    ``node_id`` values in first-seen order after that expansion.
    """
    if not mesh:
        return []
    cids = mesh_to_cids.get(mesh.strip()) or []
    out: list[str] = []
    seen: set[str] = set()
    for cid in cids:
        for namespace, external_id in cid_xrefs.get(cid) or []:
            if namespace == "drugbank":
                nid = f"primekg:drug:{external_id}"
            elif namespace == "chebi":
                nid = curated_chebi_node_id(external_id)
            else:
                continue
            if nid not in seen:
                seen.add(nid)
                out.append(nid)
    return out
