"""Hyperedge retrieval utilities (v0 + v2 batch hooks)."""

from __future__ import annotations

from primehkg.determinism import RETRIEVAL_DETERMINISM_VERSION
from primehkg.retrieval.batch import (
    DEFAULT_KHOP_BUDGETS,
    BatchQueryResult,
    CandidateMode,
    QuerySpec,
    batch_determinism_version,
    retrieve_hyperedges_batch,
)
from primehkg.retrieval.ewo import ewo_score
from primehkg.retrieval.graph_csr.build import build_graph_csr_from_hyperedges
from primehkg.retrieval.graph_csr.io import read_meta_json, write_graph_csr_sidecar
from primehkg.retrieval.graph_csr.verify import require_matching_fingerprint
from primehkg.retrieval.he_signature_index import (
    HyperedgeSignature,
    SignatureFilter,
    build_he_id_to_signature,
    build_signature_index,
    filter_he_ids_by_signature,
    hyperedge_signature,
    measure_candidate_reduction,
)
from primehkg.retrieval.index import build_inverted_index
from primehkg.retrieval.index_io import (
    load_inverted_index_json,
    load_inverted_index_json_validated,
    save_inverted_index_json,
)
from primehkg.retrieval.khop import (
    BudgetConfig,
    Subhypergraph,
    khop_candidates_k2,
    khop_determinism_version,
    materialize_subhypergraph,
)
from primehkg.retrieval.presets import (
    DEFAULT_RETRIEVAL,
    PHENOTYPE_HUB_BATCH_DEFAULTS,
    PHENOTYPE_HUB_RETRIEVAL,
    make_query_spec,
)
from primehkg.retrieval.retrieve import retrieve_hyperedges
from primehkg.retrieval.truncate import HashBasedTruncator

__all__ = [
    "DEFAULT_KHOP_BUDGETS",
    "DEFAULT_RETRIEVAL",
    "PHENOTYPE_HUB_BATCH_DEFAULTS",
    "PHENOTYPE_HUB_RETRIEVAL",
    "RETRIEVAL_DETERMINISM_VERSION",
    "BatchQueryResult",
    "BudgetConfig",
    "CandidateMode",
    "HashBasedTruncator",
    "HyperedgeSignature",
    "QuerySpec",
    "SignatureFilter",
    "Subhypergraph",
    "batch_determinism_version",
    "build_graph_csr_from_hyperedges",
    "build_he_id_to_signature",
    "build_inverted_index",
    "build_signature_index",
    "ewo_score",
    "filter_he_ids_by_signature",
    "hyperedge_signature",
    "khop_candidates_k2",
    "khop_determinism_version",
    "load_inverted_index_json",
    "load_inverted_index_json_validated",
    "make_query_spec",
    "materialize_subhypergraph",
    "measure_candidate_reduction",
    "read_meta_json",
    "require_matching_fingerprint",
    "retrieve_hyperedges",
    "retrieve_hyperedges_batch",
    "save_inverted_index_json",
    "write_graph_csr_sidecar",
]
