"""Lightweight fuzzy string matching for label linking."""

from __future__ import annotations


def token_jaccard(a: str, b: str) -> float:
    ta = {t for t in a.split() if t}
    tb = {t for t in b.split() if t}
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return inter / union if union else 0.0


def normalized_levenshtein(a: str, b: str) -> float:
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    la, lb = len(a), len(b)
    if la < lb:
        a, b = b, a
        la, lb = lb, la
    prev = list(range(lb + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            ins = cur[j - 1] + 1
            delete = prev[j] + 1
            sub = prev[j - 1] + (0 if ca == cb else 1)
            cur.append(min(ins, delete, sub))
        prev = cur
    dist = prev[lb]
    return 1.0 - (dist / max(la, lb))


def combined_fuzzy_score(
    query_norm: str,
    label_norm: str,
    *,
    w_jaccard: float = 0.3,
    w_levenshtein: float = 0.7,
) -> tuple[float, dict[str, float]]:
    j = token_jaccard(query_norm, label_norm)
    lev = normalized_levenshtein(query_norm, label_norm)
    score = w_jaccard * j + w_levenshtein * lev
    return score, {"jaccard": j, "levenshtein": lev, "combined": score}
