from __future__ import annotations

from pathlib import Path
from typing import Any

from primehkg.io.build_manifest import sha256_file, stable_json_dumps
from primehkg.policy.atomic_admissions import (
    COVERAGE_REGISTRY_ADMISSIONS,
    active_relations_for_admission,
)
from primehkg.policy.registry import PHENOTYPE_BUILDER_RELATIONS, load_relation_registry
from primehkg.policy.relation_canonicalization import SYMMETRIC_RELATIONS

RELATION_METADATA_SCHEMA_VERSION = "relation_metadata_v1"
VIEWS_SCHEMA_VERSION = "views_v1"

ATOMIC_SEMANTICS_BY_RELATION: dict[str, str] = {
    "drug_effect": "ae_drug_phenotype",
    "disease_phenotype_positive": "disease_phenotype_context",
    "off-label use": "off_label_use",
}

_COMPOSED_SEMANTICS: tuple[str, ...] = (
    "drug_disease_phenotype",
    "moa_triad",
    "drug_disease_shared_target_bundle",
    "protein_protein_bundle",
    "protein_disease_bundle",
    "disease_protein_bundle",
)

_COMPOSED_MECHANISTIC_SEMANTICS: tuple[str, ...] = ("moa_triad",)
_COMPOSED_BUNDLE_SEMANTICS: tuple[str, ...] = (
    "drug_disease_shared_target_bundle",
    "protein_protein_bundle",
    "protein_disease_bundle",
    "disease_protein_bundle",
)
_COMPOSED_COOCCURRENCE_SEMANTICS: tuple[str, ...] = ("drug_disease_phenotype",)

_CURATED_NARY_SEMANTICS: tuple[str, ...] = (
    "protein_complex",
    "reaction_event",
)

_PHENOTYPE_ATOMIC_SEMANTICS: tuple[str, ...] = (
    "ae_drug_phenotype",
    "disease_phenotype_context",
)

_COMPOSED_BUILDER_RELATIONS: dict[str, tuple[str, ...]] = {
    "drug_protein": ("moa_triad", "drug_disease_shared_target_bundle"),
    "disease_protein": (
        "moa_triad",
        "drug_disease_shared_target_bundle",
        "disease_protein_bundle",
        "protein_disease_bundle",
    ),
    "indication": ("moa_triad", "drug_disease_shared_target_bundle"),
    "contraindication": ("moa_triad", "drug_disease_shared_target_bundle"),
    "protein_protein": ("protein_protein_bundle",),
    "drug_effect": ("phenotype",),
    "disease_phenotype_positive": ("phenotype",),
}

_CANONICAL_ROLES_BY_RELATION: dict[str, dict[str, str]] = {
    "anatomy_anatomy": {"source": "anatomy", "target": "anatomy"},
    "anatomy_protein_absent": {"source": "protein", "target": "anatomy"},
    "anatomy_protein_present": {"source": "protein", "target": "anatomy"},
    "bioprocess_bioprocess": {
        "source": "biological_process",
        "target": "biological_process",
    },
    "bioprocess_protein": {"source": "protein", "target": "biological_process"},
    "cellcomp_cellcomp": {
        "source": "cellular_component",
        "target": "cellular_component",
    },
    "cellcomp_protein": {"source": "protein", "target": "cellular_component"},
    "contraindication": {"source": "drug", "target": "disease"},
    "disease_disease": {"source": "disease", "target": "disease"},
    "disease_phenotype_negative": {"source": "disease", "target": "phenotype"},
    "disease_phenotype_positive": {"source": "disease", "target": "phenotype"},
    "disease_protein": {"source": "disease", "target": "protein"},
    "drug_drug": {"source": "drug", "target": "drug"},
    "drug_effect": {"source": "drug", "target": "phenotype"},
    "drug_protein": {"source": "drug", "target": "protein"},
    "exposure_bioprocess": {"source": "exposure", "target": "biological_process"},
    "exposure_cellcomp": {"source": "exposure", "target": "cellular_component"},
    "exposure_disease": {"source": "exposure", "target": "disease"},
    "exposure_exposure": {"source": "exposure", "target": "exposure"},
    "exposure_molfunc": {"source": "exposure", "target": "molecular_function"},
    "exposure_protein": {"source": "exposure", "target": "protein"},
    "indication": {"source": "drug", "target": "disease"},
    "molfunc_molfunc": {
        "source": "molecular_function",
        "target": "molecular_function",
    },
    "molfunc_protein": {"source": "protein", "target": "molecular_function"},
    "off-label use": {"source": "drug", "target": "disease"},
    "pathway_pathway": {"source": "pathway", "target": "pathway"},
    "pathway_protein": {"source": "protein", "target": "pathway"},
    "phenotype_phenotype": {"source": "phenotype", "target": "phenotype"},
    "phenotype_protein": {"source": "protein", "target": "phenotype"},
    "protein_protein": {"source": "protein", "target": "protein"},
}

_ALLOWED_VIEW_FILTER_FIELDS = frozenset({"evidence_grade", "semantics"})
_ALLOWED_VIEW_FILTER_OPS = frozenset({"in", "not_in", "eq"})


def _active_registry_atomic_relations() -> frozenset[str]:
    registry = load_relation_registry()
    out: set[str] = set()
    for admission in COVERAGE_REGISTRY_ADMISSIONS:
        out.update(active_relations_for_admission(admission, registry))
    return frozenset(out)


def _registry_atomic_semantics() -> tuple[str, ...]:
    return tuple(
        sorted(
            ATOMIC_SEMANTICS_BY_RELATION.get(relation, relation)
            for relation in _active_registry_atomic_relations()
        )
    )


def _owner_for_relation(relation: str) -> str:
    if relation in PHENOTYPE_BUILDER_RELATIONS:
        return "phenotype"
    if relation in _active_registry_atomic_relations():
        return "registry_atomic"
    if relation in _COMPOSED_BUILDER_RELATIONS:
        return "composed_support_only"
    return "ignored_with_reason"


def _canonical_roles_for_relation(relation: str) -> dict[str, str]:
    roles = _CANONICAL_ROLES_BY_RELATION.get(relation)
    if roles is not None:
        return dict(roles)
    return {"source": "source", "target": "target"}


def _consumed_by_builders(relation: str) -> list[str]:
    return sorted(_COMPOSED_BUILDER_RELATIONS.get(relation, ()))


def _evidence_interpretation(owner: str) -> str:
    if owner == "registry_atomic":
        return "emitted as kg_atomic by registry_atomic"
    if owner == "phenotype":
        return "emitted as kg_atomic by phenotype builder"
    if owner == "composed_support_only":
        return "consumed as support by composed builders"
    return "not emitted by current builders"


def _consumer_notes(owner: str) -> str:
    if owner == "phenotype":
        return (
            "Phenotype-owned relation: source reverse rows are preserved in "
            "support_triplets, not emitted as duplicate hyperedges."
        )
    if owner == "registry_atomic":
        return "Registry-owned relation: emitted as a canonical kg_atomic hyperedge."
    if owner == "composed_support_only":
        return (
            "Composed-support relation: consumed by composed builders, not emitted "
            "as a standalone atomic hyperedge."
        )
    return "Active relation is not emitted by current builders."


def _semantics_metadata() -> dict[str, dict[str, object]]:
    out: dict[str, dict[str, object]] = {}
    for semantics in _registry_atomic_semantics():
        out[semantics] = {
            "category": "registry_atomic",
            "evidence_grade": "kg_atomic",
            "default_views": _default_view_membership("kg_atomic"),
        }
    for semantics in _PHENOTYPE_ATOMIC_SEMANTICS:
        out[semantics] = {
            "category": "phenotype_atomic",
            "evidence_grade": "kg_atomic",
            "default_views": _default_view_membership("kg_atomic"),
        }
    for semantics in _COMPOSED_SEMANTICS:
        composed_views = list(_default_view_membership("kg_composed"))
        if semantics in _COMPOSED_MECHANISTIC_SEMANTICS:
            composed_views.append("composed_mechanistic_only")
        elif semantics in _COMPOSED_BUNDLE_SEMANTICS:
            composed_views.append("composed_bundle_only")
        elif semantics in _COMPOSED_COOCCURRENCE_SEMANTICS:
            composed_views.append("composed_cooccurrence_only")
        out[semantics] = {
            "category": "composed",
            "evidence_grade": "kg_composed",
            "default_views": composed_views,
        }
    for semantics in _CURATED_NARY_SEMANTICS:
        out[semantics] = {
            "category": "curated_nary",
            "evidence_grade": "curated_db",
            "default_views": _default_view_membership("curated_db"),
        }
    return dict(sorted(out.items()))


def _default_view_membership(evidence_grade: str) -> list[str]:
    if evidence_grade == "kg_atomic":
        return ["full", "atomic_only", "search_default", "learning_default"]
    if evidence_grade == "kg_composed":
        return ["full", "composed_only", "search_default", "learning_default"]
    if evidence_grade == "curated_db":
        return ["full", "search_default", "curated_nary_only"]
    return ["full"]


def build_relation_metadata_sidecar() -> dict[str, Any]:
    registry = load_relation_registry()
    relations: dict[str, Any] = {}
    for relation, entry in sorted(registry.items()):
        if entry.status != "active":
            continue
        owner = _owner_for_relation(relation)
        symmetric = relation in SYMMETRIC_RELATIONS
        atomic_semantics = ATOMIC_SEMANTICS_BY_RELATION.get(relation, relation)
        relations[relation] = {
            "owner": owner,
            "status": entry.status,
            "notes": entry.notes,
            "symmetric": symmetric,
            "directionality": "symmetric" if symmetric else "directed",
            "canonical_roles": _canonical_roles_for_relation(relation),
            "consumed_by_builders": _consumed_by_builders(relation),
            "atomic_semantics": atomic_semantics,
            "evidence_interpretation": _evidence_interpretation(owner),
            "consumer_notes": _consumer_notes(owner),
        }
    return {
        "schema_version": RELATION_METADATA_SCHEMA_VERSION,
        "relations": relations,
        "semantics": _semantics_metadata(),
    }


def build_views_sidecar() -> dict[str, Any]:
    return {
        "schema_version": VIEWS_SCHEMA_VERSION,
        "views": {
            "atomic_only": {
                "description": "Registry and phenotype-owned atomic hyperedges.",
                "filters": [
                    {"field": "evidence_grade", "op": "eq", "value": "kg_atomic"}
                ],
                "scoring_hints": {"prefer_atomic_evidence": True},
            },
            "composed_only": {
                "description": "Composed hyperedges inferred from KG support.",
                "filters": [
                    {"field": "evidence_grade", "op": "eq", "value": "kg_composed"}
                ],
                "scoring_hints": {"prefer_explanatory_bundles": True},
            },
            "composed_mechanistic_only": {
                "description": (
                    "Mechanistic composed hyperedges (drug-disease via shared target)."
                ),
                "filters": [
                    {
                        "field": "semantics",
                        "op": "in",
                        "value": list(_COMPOSED_MECHANISTIC_SEMANTICS),
                    }
                ],
                "scoring_hints": {"prefer_mechanistic_evidence": True},
            },
            "composed_bundle_only": {
                "description": (
                    "Capped aggregation bundles (shared-target / protein / disease)."
                ),
                "filters": [
                    {
                        "field": "semantics",
                        "op": "in",
                        "value": list(_COMPOSED_BUNDLE_SEMANTICS),
                    }
                ],
                "scoring_hints": {"prefer_explanatory_bundles": True},
            },
            "composed_cooccurrence_only": {
                "description": (
                    "Uncapped symptom co-occurrence "
                    "(opt-in phenotype_composed segment)."
                ),
                "filters": [
                    {
                        "field": "semantics",
                        "op": "in",
                        "value": list(_COMPOSED_COOCCURRENCE_SEMANTICS),
                    }
                ],
                "scoring_hints": {"prefer_explanatory_bundles": True},
            },
            "curated_nary_only": {
                "description": "Curated database n-ary hyperedges.",
                "filters": [
                    {
                        "field": "semantics",
                        "op": "in",
                        "value": list(_CURATED_NARY_SEMANTICS),
                    }
                ],
                "scoring_hints": {"prefer_curated_sources": True},
            },
            "full": {
                "description": "All hyperedges emitted by the build.",
                "filters": [],
                "scoring_hints": {},
            },
            "learning_default": {
                "description": "Default view for modeling over KG-derived evidence.",
                "filters": [
                    {
                        "field": "evidence_grade",
                        "op": "in",
                        "value": ["kg_atomic", "kg_composed"],
                    }
                ],
                "scoring_hints": {"prefer_dense_kg_evidence": True},
            },
            "search_default": {
                "description": "Default view for consumer search and retrieval.",
                "filters": [
                    {
                        "field": "evidence_grade",
                        "op": "in",
                        "value": ["kg_atomic", "kg_composed", "curated_db"],
                    }
                ],
                "scoring_hints": {"prefer_explainable_evidence": True},
            },
        },
    }


def validate_views_sidecar(views: dict[str, Any]) -> None:
    if views.get("schema_version") != VIEWS_SCHEMA_VERSION:
        msg = "unsupported views sidecar schema_version"
        raise ValueError(msg)
    view_map = views.get("views")
    if not isinstance(view_map, dict):
        msg = "views sidecar must contain a views mapping"
        raise TypeError(msg)

    for view_name, view in view_map.items():
        if not isinstance(view, dict):
            msg = f"view {view_name!r} must be a mapping"
            raise TypeError(msg)
        filters = view.get("filters", [])
        if not isinstance(filters, list):
            msg = f"view {view_name!r} filters must be a list"
            raise TypeError(msg)
        for filter_spec in filters:
            if not isinstance(filter_spec, dict):
                msg = f"view {view_name!r} filter must be a mapping"
                raise TypeError(msg)
            field = filter_spec.get("field")
            if not isinstance(field, str):
                msg = f"view {view_name!r} filter field must be a string"
                raise TypeError(msg)
            if field not in _ALLOWED_VIEW_FILTER_FIELDS and not (
                field.startswith("metadata.") and len(field) > len("metadata.")
            ):
                msg = f"unsupported view filter field: {field}"
                raise ValueError(msg)
            op = filter_spec.get("op")
            if op not in _ALLOWED_VIEW_FILTER_OPS:
                msg = f"unsupported view filter op: {op}"
                raise ValueError(msg)


def write_consumer_sidecars(out_dir: str | Path) -> dict[str, Any]:
    root = Path(out_dir)
    sidecars = root / "sidecars"
    sidecars.mkdir(parents=True, exist_ok=True)

    relation_metadata = build_relation_metadata_sidecar()
    views = build_views_sidecar()
    validate_views_sidecar(views)

    relation_metadata_path = sidecars / "relation_metadata.json"
    views_path = sidecars / "views.json"
    relation_metadata_path.write_text(
        stable_json_dumps(relation_metadata) + "\n", encoding="utf-8"
    )
    views_path.write_text(stable_json_dumps(views) + "\n", encoding="utf-8")

    return {
        RELATION_METADATA_SCHEMA_VERSION: {
            "schema_version": RELATION_METADATA_SCHEMA_VERSION,
            "path": "sidecars/relation_metadata.json",
            "sha256": sha256_file(relation_metadata_path),
        },
        VIEWS_SCHEMA_VERSION: {
            "schema_version": VIEWS_SCHEMA_VERSION,
            "path": "sidecars/views.json",
            "sha256": sha256_file(views_path),
        },
    }
