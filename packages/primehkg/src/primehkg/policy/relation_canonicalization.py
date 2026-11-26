from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

OrientationPolicy = Literal[
    "directed_role",
    "symmetric_sorted",
    "phenotype_owned",
    "unsupported",
]


@dataclass(frozen=True, slots=True)
class CanonicalRelationFact:
    """Canonical endpoint orientation for one normalized PrimeKG relation row."""

    relation: str
    source: str
    target: str
    key: tuple[str, str, str]
    orientation_policy: OrientationPolicy
    symmetric: bool


# PrimeKG node-type prefix per relation (see primekg.io.primekg._primekg_typed_node_id).
ONTOLOGY_PROTEIN_TERM_PREFIX: dict[str, str] = {
    "bioprocess_protein": "biological_process",
    "cellcomp_protein": "cellular_component",
    "molfunc_protein": "molecular_function",
}

ONTOLOGY_TERM_RELATION_PREFIX: dict[str, str] = {
    "bioprocess_bioprocess": "biological_process",
    "cellcomp_cellcomp": "cellular_component",
    "molfunc_molfunc": "molecular_function",
}

ANATOMY_PROTEIN_RELATIONS = frozenset(
    {"anatomy_protein_present", "anatomy_protein_absent"}
)

# exposure -> target entity (PrimeKG node-type prefix).
EXPOSURE_CROSS_TYPE_TARGET: dict[str, str] = {
    "exposure_protein": "protein",
    "exposure_disease": "disease",
    "exposure_bioprocess": "biological_process",
    "exposure_cellcomp": "cellular_component",
    "exposure_molfunc": "molecular_function",
}

PHENOTYPE_PROTEIN_RELATIONS = frozenset({"phenotype_protein"})

DRUG_DISEASE_CLINICAL_RELATIONS = frozenset({"off-label use"})

SYMMETRIC_RELATIONS = frozenset(
    {
        "protein_protein",
        "disease_disease",
        "drug_drug",
        "pathway_pathway",
        "anatomy_anatomy",
        "exposure_exposure",
        "phenotype_phenotype",
        "bioprocess_bioprocess",
        "cellcomp_cellcomp",
        "molfunc_molfunc",
    }
)

PHENOTYPE_OWNED_RELATIONS = frozenset({"drug_effect", "disease_phenotype_positive"})

_DIRECTED_ROLE_RELATIONS = frozenset(
    {
        "pathway_protein",
        "disease_phenotype_negative",
        "disease_protein",
        "drug_protein",
        "indication",
        "contraindication",
        *ONTOLOGY_PROTEIN_TERM_PREFIX,
        *ANATOMY_PROTEIN_RELATIONS,
        *EXPOSURE_CROSS_TYPE_TARGET,
        *PHENOTYPE_PROTEIN_RELATIONS,
        *DRUG_DISEASE_CLINICAL_RELATIONS,
    }
)


def is_phenotype_node(node_id: str) -> bool:
    """PrimeKG HPO nodes use x_type ``effect/phenotype``."""
    return node_id.startswith(("primekg:phenotype:", "primekg:effect/phenotype:"))


def canonicalize_relation_fact(
    relation: str,
    source: str,
    target: str,
) -> CanonicalRelationFact | None:
    rr = relation.strip()
    s = source.strip()
    t = target.strip()
    if not (rr and s and t):
        return None

    if rr in PHENOTYPE_OWNED_RELATIONS:
        return _canonicalize_phenotype_owned(rr, s, t)

    canonical = _canonicalize_triplet_endpoints(rr, s, t)
    if canonical is None:
        return None
    src, tgt = canonical
    if not _triplet_matches_relation_policy(rr, src, tgt):
        return None

    symmetric = rr in SYMMETRIC_RELATIONS
    if symmetric:
        left, right = sorted((src, tgt))
        key = (rr, left, right)
        orientation_policy: OrientationPolicy = "symmetric_sorted"
    else:
        key = (rr, src, tgt)
        orientation_policy = (
            "directed_role" if rr in _DIRECTED_ROLE_RELATIONS else "unsupported"
        )

    return CanonicalRelationFact(
        relation=rr,
        source=src,
        target=tgt,
        key=key,
        orientation_policy=orientation_policy,
        symmetric=symmetric,
    )


def _canonicalize_phenotype_owned(
    relation: str,
    s: str,
    t: str,
) -> CanonicalRelationFact | None:
    if relation == "drug_effect":
        owner_tag = "primekg:drug:"
    elif relation == "disease_phenotype_positive":
        owner_tag = "primekg:disease:"
    else:
        return None

    if s.startswith(owner_tag) and is_phenotype_node(t):
        src, tgt = s, t
    elif t.startswith(owner_tag) and is_phenotype_node(s):
        src, tgt = t, s
    else:
        return None

    return CanonicalRelationFact(
        relation=relation,
        source=src,
        target=tgt,
        key=(relation, src, tgt),
        orientation_policy="phenotype_owned",
        symmetric=False,
    )


def _triplet_matches_relation_policy(relation: str, s: str, t: str) -> bool:
    """Relation-specific endpoint guards aligned with composed bundle builders."""
    if relation == "protein_protein":
        return (
            s.startswith("primekg:protein:")
            and t.startswith("primekg:protein:")
            and s != t
        )
    if relation in {"disease_phenotype_negative", "disease_phenotype_positive"}:
        return s.startswith("primekg:disease:") and is_phenotype_node(t)
    if relation == "drug_drug":
        return (
            s.startswith("primekg:drug:") and t.startswith("primekg:drug:") and s != t
        )
    if relation == "disease_disease":
        return (
            s.startswith("primekg:disease:")
            and t.startswith("primekg:disease:")
            and s != t
        )
    if relation == "pathway_pathway":
        return (
            s.startswith("primekg:pathway:")
            and t.startswith("primekg:pathway:")
            and s != t
        )
    if relation == "pathway_protein":
        return s.startswith("primekg:protein:") and t.startswith("primekg:pathway:")
    ont_prefix = ONTOLOGY_PROTEIN_TERM_PREFIX.get(relation)
    if ont_prefix is not None:
        return s.startswith("primekg:protein:") and t.startswith(
            f"primekg:{ont_prefix}:"
        )
    ont_prefix = ONTOLOGY_TERM_RELATION_PREFIX.get(relation)
    if ont_prefix is not None:
        tag = f"primekg:{ont_prefix}:"
        return s.startswith(tag) and t.startswith(tag) and s != t
    if relation in ANATOMY_PROTEIN_RELATIONS:
        return s.startswith("primekg:protein:") and t.startswith("primekg:anatomy:")
    if relation == "anatomy_anatomy":
        return (
            s.startswith("primekg:anatomy:")
            and t.startswith("primekg:anatomy:")
            and s != t
        )
    target = EXPOSURE_CROSS_TYPE_TARGET.get(relation)
    if target is not None:
        return s.startswith("primekg:exposure:") and t.startswith(f"primekg:{target}:")
    if relation == "exposure_exposure":
        return (
            s.startswith("primekg:exposure:")
            and t.startswith("primekg:exposure:")
            and s != t
        )
    if relation in PHENOTYPE_PROTEIN_RELATIONS:
        return s.startswith("primekg:protein:") and is_phenotype_node(t)
    if relation == "phenotype_phenotype":
        return is_phenotype_node(s) and is_phenotype_node(t) and s != t
    if relation in DRUG_DISEASE_CLINICAL_RELATIONS:
        return s.startswith("primekg:drug:") and t.startswith("primekg:disease:")
    return True


def _canonicalize_triplet_endpoints(
    relation: str, s: str, t: str
) -> tuple[str, str] | None:
    """Return canonical (s, t) for relation-specific orientation, or None to skip."""
    if relation == "pathway_protein":
        if s.startswith("primekg:pathway:") and t.startswith("primekg:protein:"):
            return t, s
        if s.startswith("primekg:protein:") and t.startswith("primekg:pathway:"):
            return s, t
        return None
    ont_prefix = ONTOLOGY_PROTEIN_TERM_PREFIX.get(relation)
    if ont_prefix is not None:
        tag = f"primekg:{ont_prefix}:"
        if s.startswith("primekg:protein:") and t.startswith(tag):
            return s, t
        if s.startswith(tag) and t.startswith("primekg:protein:"):
            return t, s
        return None
    if relation in ONTOLOGY_TERM_RELATION_PREFIX:
        if s.startswith("primekg:") and t.startswith("primekg:"):
            a, b = sorted([s, t])
            return a, b
        return None
    if relation in ANATOMY_PROTEIN_RELATIONS:
        if s.startswith("primekg:anatomy:") and t.startswith("primekg:protein:"):
            return t, s
        if s.startswith("primekg:protein:") and t.startswith("primekg:anatomy:"):
            return s, t
        return None
    if relation == "anatomy_anatomy":
        if s.startswith("primekg:anatomy:") and t.startswith("primekg:anatomy:"):
            a, b = sorted([s, t])
            return a, b
        return None
    target = EXPOSURE_CROSS_TYPE_TARGET.get(relation)
    if target is not None:
        exp_tag = "primekg:exposure:"
        tgt_tag = f"primekg:{target}:"
        if s.startswith(exp_tag) and t.startswith(tgt_tag):
            return s, t
        if s.startswith(tgt_tag) and t.startswith(exp_tag):
            return t, s
        return None
    if relation == "exposure_exposure":
        if s.startswith("primekg:exposure:") and t.startswith("primekg:exposure:"):
            a, b = sorted([s, t])
            return a, b
        return None
    if relation in PHENOTYPE_PROTEIN_RELATIONS:
        if is_phenotype_node(s) and t.startswith("primekg:protein:"):
            return t, s
        if s.startswith("primekg:protein:") and is_phenotype_node(t):
            return s, t
        return None
    if relation == "phenotype_phenotype":
        if is_phenotype_node(s) and is_phenotype_node(t):
            a, b = sorted([s, t])
            return a, b
        return None
    if relation in DRUG_DISEASE_CLINICAL_RELATIONS:
        if s.startswith("primekg:drug:") and t.startswith("primekg:disease:"):
            return s, t
        if s.startswith("primekg:disease:") and t.startswith("primekg:drug:"):
            return t, s
        return None
    if relation == "disease_protein":
        if s.startswith("primekg:disease:") and t.startswith("primekg:protein:"):
            return s, t
        if s.startswith("primekg:protein:") and t.startswith("primekg:disease:"):
            return t, s
        return None
    if relation == "drug_protein":
        if s.startswith("primekg:drug:") and t.startswith("primekg:protein:"):
            return s, t
        if s.startswith("primekg:protein:") and t.startswith("primekg:drug:"):
            return t, s
        return None
    if relation in {"indication", "contraindication"}:
        if s.startswith("primekg:drug:") and t.startswith("primekg:disease:"):
            return s, t
        if s.startswith("primekg:disease:") and t.startswith("primekg:drug:"):
            return t, s
        return None
    if relation in SYMMETRIC_RELATIONS:
        if s != t:
            a, b = sorted([s, t])
            return a, b
        return None
    return s, t
