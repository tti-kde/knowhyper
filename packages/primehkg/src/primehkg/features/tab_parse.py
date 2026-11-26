from __future__ import annotations

import re

_TAB_NUMERIC_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def parse_tab_numeric_cell(raw: str) -> float | None:
    """Parse a PrimeKG feature tab cell as float (plain number or prose)."""
    text = raw.strip().strip('"')
    if not text:
        return None
    try:
        return float(text.replace(",", ""))
    except ValueError:
        pass
    match = _TAB_NUMERIC_RE.search(text)
    if match is None:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None
