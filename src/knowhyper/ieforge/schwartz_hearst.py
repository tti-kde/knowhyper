"""In-document abbreviation extraction (Schwartz & Hearst, PSB 2003).

Finds ``long form (SHORT)`` and ``SHORT (long form)`` parenthetical definitions in
a single document and returns a ``{short_form: long_form}`` map. Validation uses
the character-subsequence rule: every alphanumeric char of the short form must
appear in the long form in order (case-insensitive), the match must begin at a
word start, and the long form is bounded to at most ``len(SF) * 2 + 5`` words to
the left of the parenthesis. Pure, deterministic, no external resources.
"""

from __future__ import annotations

import re

_PAREN = re.compile(r"\(([^()]+)\)")
_WORD = re.compile(r"\w[\w-]*")


def _is_valid_short_form(sf: str) -> bool:
    # 2-10 chars, contains a letter, mostly alphanumeric (rules out phrases/clauses).
    if not (2 <= len(sf) <= 10):
        return False
    if not any(c.isalpha() for c in sf):
        return False
    return sum(c.isalnum() for c in sf) >= max(1, len(sf) - 2)


def _subsequence_match(short: str, long_words: list[str]) -> str | None:
    """Match short's letters as an in-order subsequence of the trailing long form.

    Matched right-to-left, starting at a word boundary; returns the long form on
    success, else ``None``.
    """
    s = [c.lower() for c in short if c.isalnum()]
    if not s:
        return None
    si = len(s) - 1
    start_word = len(long_words)
    for wi in range(len(long_words) - 1, -1, -1):
        word = long_words[wi]
        for c in reversed(word.lower()):
            if si < 0:
                break
            if c.isalnum() and c == s[si]:
                si -= 1
        if si < 0 and word.lower().lstrip()[:1] == s[0]:
            start_word = wi
            break
    if si >= 0:
        return None
    return " ".join(long_words[start_word:])


def extract_abbreviations(text: str) -> dict[str, str]:
    """Extract ``{short_form: long_form}`` definitions from one document.

    Both ``long (SF)`` and ``SF (long)`` orderings are handled; the first valid
    definition for a given short form wins.
    """
    out: dict[str, str] = {}
    for m in _PAREN.finditer(text):
        inner = m.group(1).strip()
        before = text[: m.start()]
        before_words = _WORD.findall(before)
        # Case A: long form before, short form in parens.
        if _is_valid_short_form(inner) and before_words:
            n = min(len(before_words), len(inner.replace(" ", "")) * 2 + 5)
            lf = _subsequence_match(inner, before_words[-n:])
            if lf and inner not in out:
                out[inner] = lf
                continue
        # Case B: short form before (last word), long form in parens.
        inner_words = _WORD.findall(inner)
        if before_words:
            sf = before_words[-1]
            if _is_valid_short_form(sf) and inner_words:
                lf = _subsequence_match(sf, inner_words)
                if lf and sf not in out:
                    out[sf] = lf
    return out
