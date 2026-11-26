from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypedDict

if TYPE_CHECKING:
    from pathlib import Path

from primehkg.ids import he_id
from primehkg.io.curated_sources import ClinVarVariant, load_clinvar_variants


class ClinVarBuilderResult(TypedDict):
    """Return type for :func:`build_clinvar_variant_hyperedges`."""

    hyperedges: list[dict[str, Any]]
    support: list[dict[str, Any]]
    node_updates: dict[str, dict[str, Any]]


_PATHOGENIC = {"pathogenic", "likely pathogenic"}
_BENIGN = {"benign", "likely benign"}


def _disease_polarity(clinical_significance: str) -> str:
    sig = clinical_significance.split("/", maxsplit=1)[0].strip().lower()
    if sig in _PATHOGENIC:
        return "positive"
    if sig in _BENIGN:
        return "negative"
    return "unknown"


def _variant_node_update(v: ClinVarVariant) -> dict[str, Any]:
    """Build a node-enrichment update for a ClinVar variant node.

    name <- HGVS (or VariationID fallback); xrefs <- dbSNP; type/coords <- source_meta provenance.
    Uses only keys consumed by ``enrich_nodes_v1`` (names / xrefs / source_meta_patch).
    """
    display = v.get("hgvs_name") or v["variation_id"]
    upd: dict[str, Any] = {
        "source": "ClinVar",
        "evidence": "clinvar",
        "names": [{"name": display, "source": "ClinVar", "evidence": "clinvar"}],
    }
    rs = v.get("rs_id")
    if rs:
        upd["xrefs"] = {"dbsnp": [f"rs{rs}"]}
    cv: dict[str, str] = {"assembly": "GRCh38"}  # loader pre-filters to GRCh38 rows
    # ClinVarVariant field name -> output schema key (most are identity; variant_type -> type)
    for src_key, dst_key in (
        ("variant_type", "type"),
        ("chromosome", "chromosome"),
        ("start", "start"),
        ("stop", "stop"),
        ("ref_allele", "ref_allele"),
        ("alt_allele", "alt_allele"),
    ):
        val = v.get(src_key)
        if val:
            cv[dst_key] = val
    upd["source_meta_patch"] = {"clinvar_variant": cv}
    return upd


def build_clinvar_variant_hyperedges(
    variant_summary: str | Path,
    *,
    clinvar_version: str,
    mondo_to_disease: dict[str, str],
    known_node_ids: set[str] | None = None,
) -> ClinVarBuilderResult:
    """``variant_gene`` + ``variant_disease`` hyperedges from clinically-significant ClinVar variants.

    A variant is node-ified (``clinvar:<VariationID>``) only if it yields >=1 resolvable edge:
    a gene edge (NCBI GeneID -> ``primekg:protein:<GeneID>``, polarity unknown) and/or disease
    edges (MONDO -> ``primekg:disease:<node_id>`` via ``mondo_to_disease``, polarity from clinical
    significance). The gene/disease endpoint must be in ``known_node_ids`` when given (the
    ``clinvar:<id>`` variant node is novel and always admitted when it has ≥1 resolvable edge).
    """
    variants = load_clinvar_variants(variant_summary)

    hyperedges: list[dict[str, Any]] = []
    support: list[dict[str, Any]] = []
    seen: set[str] = set()
    node_updates: dict[str, dict[str, Any]] = {}

    def _emit(
        semantics: str,
        node_ids: list[str],
        roles: dict[str, str],
        polarity: str,
        variation_id: str,
        sig: str,
        review: str,
    ) -> None:
        rule_trace = {
            "builder": "clinvar",
            "version": 1,
            "kind": "curated_db",
            "clinical_significance": sig,
            "review_status": review,
        }
        hid = he_id(
            semantics=semantics, node_ids=node_ids, roles=roles, rule_trace=rule_trace
        )
        if hid in seen:
            return
        seen.add(hid)
        support_id = f"sup:{hid}"
        support.append(
            {
                "support_id": support_id,
                "sources": [{"name": "ClinVar", "version": clinvar_version}],
                "support_triplets": [
                    {
                        "source": "ClinVar",
                        "variation_id": variation_id,
                        "clinical_significance": sig,
                    }
                ],
                "rule_trace": rule_trace,
                "confidence": None,
            }
        )
        hyperedges.append(
            {
                "he_id": hid,
                "semantics": semantics,
                "evidence_grade": "curated_db",
                "polarity": polarity,
                "node_ids": node_ids,
                "roles": roles,
                "support_id": support_id,
            }
        )

    for v in variants:
        variant_node = f"clinvar:{v['variation_id']}"
        sig = v["clinical_significance"]
        review = v["review_status"]
        emitted_any = False

        if v["gene_id"]:
            gene_node = f"primekg:protein:{v['gene_id']}"
            if known_node_ids is None or gene_node in known_node_ids:
                emitted_any = True
                node_ids = sorted([variant_node, gene_node])
                roles = {variant_node: "variant", gene_node: "gene"}
                _emit(
                    "variant_gene",
                    node_ids,
                    roles,
                    "unknown",
                    v["variation_id"],
                    sig,
                    review,
                )

        disease_polarity = _disease_polarity(sig)
        emitted_diseases: set[str] = set()
        for mondo in v["mondo_numbers"]:
            disease_node = mondo_to_disease.get(mondo)
            if disease_node is None or disease_node in emitted_diseases:
                continue
            if known_node_ids is not None and disease_node not in known_node_ids:
                continue
            emitted_any = True
            emitted_diseases.add(disease_node)
            node_ids = sorted([variant_node, disease_node])
            roles = {variant_node: "variant", disease_node: "disease"}
            _emit(
                "variant_disease",
                node_ids,
                roles,
                disease_polarity,
                v["variation_id"],
                sig,
                review,
            )

        if emitted_any and variant_node not in node_updates:
            node_updates[variant_node] = _variant_node_update(v)

    return {"hyperedges": hyperedges, "support": support, "node_updates": node_updates}
