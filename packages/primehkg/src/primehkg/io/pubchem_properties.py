"""PubChem PUG physicochemical properties for DrugBank RegistryID xrefs."""

from __future__ import annotations

import time
import urllib.error
from http.client import RemoteDisconnected
from typing import Any
from urllib.parse import quote

from primehkg.io.http_download import get_json

_USER_AGENT = "primehkg-export-mappings/0.1"

# PubChem PUG property name → export column / registry column.
PUBCHEM_PROPERTY_TO_COLUMN: dict[str, str] = {
    "MolecularWeight": "molecular_weight",
    "XLogP": "alogp",
    "TPSA": "psa",
    "HBondDonorCount": "hbd",
    "HBondAcceptorCount": "hba",
}

_PUG_PROPERTIES = ",".join(PUBCHEM_PROPERTY_TO_COLUMN)


def fetch_pubchem_properties_for_drugbank_id(
    drugbank_id: str,
    *,
    timeout_s: float = 60.0,
    insecure_ssl: bool = False,
) -> dict[str, float]:
    """Resolve DrugBank id → physicochemical properties via PubChem."""
    dbid = drugbank_id.strip()
    if not dbid:
        return {}
    url = (
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/xref/RegistryID/"
        f"{quote(dbid)}/property/{_PUG_PROPERTIES}/JSON"
    )
    payload: dict[str, Any] | None = None
    for attempt in range(3):
        try:
            payload = get_json(
                url=url,
                timeout_s=timeout_s,
                user_agent=_USER_AGENT,
                insecure_ssl=insecure_ssl,
            )
            break
        except urllib.error.HTTPError:
            return {}
        except (urllib.error.URLError, TimeoutError, OSError, RemoteDisconnected):
            if attempt >= 2:
                return {}
            time.sleep(1.0 * (attempt + 1))
    if payload is None:
        return {}
    properties = (payload.get("PropertyTable") or {}).get("Properties") or []
    if not properties:
        return {}
    props = properties[0]
    out: dict[str, float] = {}
    for pubchem_key, column in PUBCHEM_PROPERTY_TO_COLUMN.items():
        raw = props.get(pubchem_key)
        if raw is None or raw == "":
            continue
        try:
            out[column] = float(raw)
        except (TypeError, ValueError):
            continue
    return out
