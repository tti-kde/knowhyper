from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from primehkg.policy.registry import RelationEntry

ADMISSION_RELATIONS_DRUG_TARGET = frozenset({"disease_protein", "drug_protein"})


@dataclass(frozen=True, slots=True)
class AtomicAdmission:
    """Registry atomic admission: relations emitted as one HE per triplet."""

    admission_id: str
    relations: frozenset[str]
    max_out_edges_per_anchor: int | None = None

    @property
    def wave_id(self) -> str:
        """Deprecated alias for :attr:`admission_id` (0.2.0)."""
        return self.admission_id


ADMISSION_DRUG_TARGET = AtomicAdmission(
    admission_id="pilot_drug_target_joins",
    relations=ADMISSION_RELATIONS_DRUG_TARGET,
)

ADMISSION_RELATIONS_CLINICAL_CONTEXT = frozenset({"indication", "contraindication"})

ADMISSION_CLINICAL_CONTEXT = AtomicAdmission(
    admission_id="pilot_clinical_context",
    relations=ADMISSION_RELATIONS_CLINICAL_CONTEXT,
)

ADMISSION_RELATIONS_PROTEIN_PROTEIN = frozenset({"protein_protein"})

ADMISSION_PROTEIN_PROTEIN = AtomicAdmission(
    admission_id="pilot_protein_protein",
    relations=ADMISSION_RELATIONS_PROTEIN_PROTEIN,
)

ADMISSION_RELATIONS_PHENOTYPE_NEGATIVE = frozenset({"disease_phenotype_negative"})

ADMISSION_PHENOTYPE_NEGATIVE = AtomicAdmission(
    admission_id="pilot_phenotype_negative",
    relations=ADMISSION_RELATIONS_PHENOTYPE_NEGATIVE,
)

ADMISSION_RELATIONS_DRUG_DRUG = frozenset({"drug_drug"})

ADMISSION_DRUG_DRUG = AtomicAdmission(
    admission_id="pilot_drug_drug_capped",
    relations=ADMISSION_RELATIONS_DRUG_DRUG,
)

ADMISSION_RELATIONS_DISEASE_DISEASE = frozenset({"disease_disease"})

ADMISSION_DISEASE_DISEASE = AtomicAdmission(
    admission_id="pilot_disease_disease_capped",
    relations=ADMISSION_RELATIONS_DISEASE_DISEASE,
)

ADMISSION_RELATIONS_PATHWAY_PRIMEKG = frozenset({"pathway_pathway", "pathway_protein"})

ADMISSION_PATHWAY_PRIMEKG = AtomicAdmission(
    admission_id="pilot_pathway_primekg_staging",
    relations=ADMISSION_RELATIONS_PATHWAY_PRIMEKG,
)

ADMISSION_RELATIONS_ONTOLOGY_GO = frozenset(
    {
        "bioprocess_protein",
        "cellcomp_protein",
        "molfunc_protein",
        "bioprocess_bioprocess",
        "cellcomp_cellcomp",
        "molfunc_molfunc",
    }
)

ADMISSION_ONTOLOGY_GO = AtomicAdmission(
    admission_id="pilot_ontology_go_staging",
    relations=ADMISSION_RELATIONS_ONTOLOGY_GO,
)

ADMISSION_RELATIONS_ANATOMY = frozenset(
    {
        "anatomy_protein_present",
        "anatomy_protein_absent",
        "anatomy_anatomy",
    }
)

ADMISSION_ANATOMY = AtomicAdmission(
    admission_id="pilot_anatomy_staging",
    relations=ADMISSION_RELATIONS_ANATOMY,
)

ADMISSION_RELATIONS_EXPOSURE = frozenset(
    {
        "exposure_bioprocess",
        "exposure_cellcomp",
        "exposure_disease",
        "exposure_exposure",
        "exposure_molfunc",
        "exposure_protein",
    }
)

ADMISSION_EXPOSURE = AtomicAdmission(
    admission_id="pilot_exposure_staging",
    relations=ADMISSION_RELATIONS_EXPOSURE,
)

ADMISSION_RELATIONS_PHENOTYPE_ONTOLOGY = frozenset(
    {"phenotype_phenotype", "phenotype_protein"}
)

ADMISSION_PHENOTYPE_ONTOLOGY = AtomicAdmission(
    admission_id="pilot_phenotype_ontology_staging",
    relations=ADMISSION_RELATIONS_PHENOTYPE_ONTOLOGY,
)

ADMISSION_RELATIONS_OFF_LABEL = frozenset({"off-label use"})

ADMISSION_OFF_LABEL = AtomicAdmission(
    admission_id="pilot_off_label_clinical_staging",
    relations=ADMISSION_RELATIONS_OFF_LABEL,
)

COVERAGE_REGISTRY_ADMISSIONS: tuple[AtomicAdmission, ...] = (
    ADMISSION_DRUG_TARGET,
    ADMISSION_CLINICAL_CONTEXT,
    ADMISSION_PROTEIN_PROTEIN,
    ADMISSION_PHENOTYPE_NEGATIVE,
    ADMISSION_DRUG_DRUG,
    ADMISSION_DISEASE_DISEASE,
    ADMISSION_PATHWAY_PRIMEKG,
    ADMISSION_ONTOLOGY_GO,
    ADMISSION_ANATOMY,
    ADMISSION_EXPOSURE,
    ADMISSION_PHENOTYPE_ONTOLOGY,
    ADMISSION_OFF_LABEL,
)


def active_relations_for_admission(
    admission: AtomicAdmission, registry: dict[str, RelationEntry]
) -> frozenset[str]:
    """Return admission relations that are ``active`` in the packaged registry."""
    out: set[str] = set()
    for rel in admission.relations:
        ent = registry.get(rel)
        if ent is not None and ent.status == "active":
            out.add(rel)
    return frozenset(out)
