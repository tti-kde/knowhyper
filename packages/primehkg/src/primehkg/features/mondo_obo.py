from __future__ import annotations

import re
from pathlib import Path


def normalize_mondo_id(raw: str) -> str:
    """Normalize MONDO CURIEs to seven-digit zero-padded local ids."""
    token = raw.strip()
    local = token
    if local.upper().startswith("MONDO:"):
        local = local.split(":", 1)[1]
    if local.isdigit():
        return f"MONDO:{int(local):07d}"
    return token


def parse_mondo_obo_data_version_from_text(text: str) -> str | None:
    """Return ``data-version`` from OBO header lines before the first ``[Term]``."""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == "[Term]":
            break
        if stripped.startswith("data-version:"):
            return stripped.removeprefix("data-version:").strip()
    return None


def parse_mondo_obo_data_version(path: str | Path) -> str | None:
    return parse_mondo_obo_data_version_from_text(
        Path(path).read_text(encoding="utf-8")
    )


def parse_mondo_obo_definitions_from_text(text: str) -> dict[str, str]:
    """Parse ``[Term]`` blocks from MONDO OBO text into normalized ``MONDO:…`` → definition."""
    definitions: dict[str, str] = {}
    current_id: str | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == "[Term]":
            current_id = None
            continue
        if stripped == "" or stripped.startswith("["):
            continue
        if stripped.startswith("id: MONDO:"):
            current_id = normalize_mondo_id(stripped.removeprefix("id: ").strip())
            continue
        if (
            current_id
            and stripped.startswith("def: ")
            and current_id not in definitions
        ):
            match = re.match(r'^def: "(.*)" \[', stripped)
            if match:
                text_val = match.group(1).replace('\\"', '"').replace("\\n", "\n")
                definitions[current_id] = text_val
    return definitions


def parse_mondo_obo_definitions(path: str | Path) -> dict[str, str]:
    return parse_mondo_obo_definitions_from_text(Path(path).read_text(encoding="utf-8"))
