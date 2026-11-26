"""Deprecated: use :mod:`primehkg.policy.atomic_admissions`."""

from __future__ import annotations

import warnings

from primehkg.policy.atomic_admissions import (
    ADMISSION_ANATOMY as PILOT_WAVE_9,
)
from primehkg.policy.atomic_admissions import (
    ADMISSION_CLINICAL_CONTEXT as PILOT_WAVE_2,
)
from primehkg.policy.atomic_admissions import (
    ADMISSION_DISEASE_DISEASE as PILOT_WAVE_6,
)
from primehkg.policy.atomic_admissions import (
    ADMISSION_DRUG_DRUG as PILOT_WAVE_5,
)
from primehkg.policy.atomic_admissions import (
    ADMISSION_DRUG_TARGET as PILOT_WAVE_1,
)
from primehkg.policy.atomic_admissions import (
    ADMISSION_EXPOSURE as PILOT_WAVE_10,
)
from primehkg.policy.atomic_admissions import (
    ADMISSION_OFF_LABEL as PILOT_WAVE_12,
)
from primehkg.policy.atomic_admissions import (
    ADMISSION_ONTOLOGY_GO as PILOT_WAVE_8,
)
from primehkg.policy.atomic_admissions import (
    ADMISSION_PATHWAY_PRIMEKG as PILOT_WAVE_7,
)
from primehkg.policy.atomic_admissions import (
    ADMISSION_PHENOTYPE_NEGATIVE as PILOT_WAVE_4,
)
from primehkg.policy.atomic_admissions import (
    ADMISSION_PHENOTYPE_ONTOLOGY as PILOT_WAVE_11,
)
from primehkg.policy.atomic_admissions import (
    ADMISSION_PROTEIN_PROTEIN as PILOT_WAVE_3,
)
from primehkg.policy.atomic_admissions import (
    ADMISSION_RELATIONS_ANATOMY as PILOT_WAVE_ANATOMY,
)
from primehkg.policy.atomic_admissions import (
    ADMISSION_RELATIONS_CLINICAL_CONTEXT as PILOT_WAVE_CLINICAL_CONTEXT,
)
from primehkg.policy.atomic_admissions import (
    ADMISSION_RELATIONS_DISEASE_DISEASE as PILOT_WAVE_DISEASE_DISEASE,
)
from primehkg.policy.atomic_admissions import (
    ADMISSION_RELATIONS_DRUG_DRUG as PILOT_WAVE_DRUG_DRUG,
)
from primehkg.policy.atomic_admissions import (
    ADMISSION_RELATIONS_DRUG_TARGET as PILOT_WAVE_DRUG_TARGET_JOINS,
)
from primehkg.policy.atomic_admissions import (
    ADMISSION_RELATIONS_EXPOSURE as PILOT_WAVE_EXPOSURE,
)
from primehkg.policy.atomic_admissions import (
    ADMISSION_RELATIONS_OFF_LABEL as PILOT_WAVE_OFF_LABEL,
)
from primehkg.policy.atomic_admissions import (
    ADMISSION_RELATIONS_ONTOLOGY_GO as PILOT_WAVE_ONTOLOGY_GO,
)
from primehkg.policy.atomic_admissions import (
    ADMISSION_RELATIONS_PATHWAY_PRIMEKG as PILOT_WAVE_PATHWAY_PRIMEKG,
)
from primehkg.policy.atomic_admissions import (
    ADMISSION_RELATIONS_PHENOTYPE_NEGATIVE as PILOT_WAVE_PHENOTYPE_NEGATIVE,
)
from primehkg.policy.atomic_admissions import (
    ADMISSION_RELATIONS_PHENOTYPE_ONTOLOGY as PILOT_WAVE_PHENOTYPE_ONTOLOGY,
)
from primehkg.policy.atomic_admissions import (
    ADMISSION_RELATIONS_PROTEIN_PROTEIN as PILOT_WAVE_PROTEIN_PROTEIN,
)
from primehkg.policy.atomic_admissions import (
    COVERAGE_REGISTRY_ADMISSIONS as COVERAGE_REGISTRY_WAVES,
)
from primehkg.policy.atomic_admissions import (
    AtomicAdmission as AtomicWave,
)
from primehkg.policy.atomic_admissions import (
    active_relations_for_admission as active_relations_for_wave,
)

warnings.warn(
    "primehkg.policy.atomic_waves is deprecated; import primehkg.policy.atomic_admissions",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "COVERAGE_REGISTRY_WAVES",
    "PILOT_WAVE_1",
    "PILOT_WAVE_2",
    "PILOT_WAVE_3",
    "PILOT_WAVE_4",
    "PILOT_WAVE_5",
    "PILOT_WAVE_6",
    "PILOT_WAVE_7",
    "PILOT_WAVE_8",
    "PILOT_WAVE_9",
    "PILOT_WAVE_10",
    "PILOT_WAVE_11",
    "PILOT_WAVE_12",
    "PILOT_WAVE_ANATOMY",
    "PILOT_WAVE_CLINICAL_CONTEXT",
    "PILOT_WAVE_DISEASE_DISEASE",
    "PILOT_WAVE_DRUG_DRUG",
    "PILOT_WAVE_DRUG_TARGET_JOINS",
    "PILOT_WAVE_EXPOSURE",
    "PILOT_WAVE_OFF_LABEL",
    "PILOT_WAVE_ONTOLOGY_GO",
    "PILOT_WAVE_PATHWAY_PRIMEKG",
    "PILOT_WAVE_PHENOTYPE_NEGATIVE",
    "PILOT_WAVE_PHENOTYPE_ONTOLOGY",
    "PILOT_WAVE_PROTEIN_PROTEIN",
    "AtomicWave",
    "active_relations_for_wave",
]
