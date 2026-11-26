from __future__ import annotations

import hashlib
import unicodedata


def normalize_text_for_hash(text: str) -> str:
    """Normalize text before hashing (NFKC + strip)."""
    return unicodedata.normalize("NFKC", text).strip()


def source_text_hash(text: str) -> str:
    """Return sha256 hex digest of normalized text."""
    normalized = normalize_text_for_hash(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
