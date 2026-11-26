"""Determinism/version bump policy (single source of truth).

These constants are intended to be:
- included in reproducibility artifacts (e.g. build manifest fingerprints)
- surfaced by retrieval APIs/CLIs when running in strict reproducibility mode

Any behavior change that affects stable outputs MUST bump the corresponding version here:
- role vocabulary mapping
- identifier normalization rules
- filtering rules
"""

from __future__ import annotations

# Global policy version for reproducibility artifacts (manifest fingerprint inputs).
DETERMINISM_POLICY_VERSION = "2026-05-26-v1"

# Retrieval batch API version surfaced in strict mode outputs.
RETRIEVAL_DETERMINISM_VERSION = "2026-04-24-v1"

# Curated conversion behavior (primehkg-curated-fetch) that affects reactome/corum TSV contracts.
CURATED_FETCH_DETERMINISM_VERSION = "2026-04-24-v1"
