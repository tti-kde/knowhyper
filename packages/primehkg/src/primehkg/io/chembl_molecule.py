"""ChEMBL REST molecule physicochemical properties."""

from __future__ import annotations

import time
import urllib.error
from http.client import RemoteDisconnected
from typing import Any
from urllib.parse import quote, urlencode

from primehkg.io.http_download import get_json

_USER_AGENT = "primehkg-export-mappings/0.1"

# ChEMBL molecule_properties field → export column.
CHEMBL_FIELD_TO_COLUMN: dict[str, str] = {
    "full_mwt": "molecular_weight",
    "alogp": "alogp",
    "psa": "psa",
    "hba": "hba",
    "hbd": "hbd",
}


def fetch_chembl_molecule_properties(
    chembl_id: str,
    *,
    timeout_s: float = 60.0,
    insecure_ssl: bool = False,
) -> dict[str, float]:
    """Fetch standard drug properties from ChEMBL molecule endpoint."""
    cid = chembl_id.strip()
    if not cid:
        return {}
    url = f"https://www.ebi.ac.uk/chembl/api/data/molecule/{quote(cid)}.json"
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
    if not isinstance(payload, dict):
        return {}
    mol_props = payload.get("molecule_properties")
    if not isinstance(mol_props, list) or not mol_props:
        return {}
    first = mol_props[0]
    if not isinstance(first, dict):
        return {}
    out: dict[str, float] = {}
    for chembl_field, column in CHEMBL_FIELD_TO_COLUMN.items():
        raw = first.get(chembl_field)
        if raw is None or raw == "":
            continue
        try:
            out[column] = float(raw)
        except (TypeError, ValueError):
            continue
    return out


def fetch_chembl_xref_for_drugbank_id(
    drugbank_id: str,
    *,
    timeout_s: float = 60.0,
    insecure_ssl: bool = False,
) -> tuple[str, str]:
    """Resolve DrugBank id → (molecule_chembl_id, pref_name) via ChEMBL xref search."""
    dbid = drugbank_id.strip()
    if not dbid:
        return "", ""
    query = urlencode(
        {
            "molecule_xrefs__xref_id": dbid,
            "molecule_xrefs__xref_src": "DrugBank",
            "limit": "1",
        }
    )
    url = f"https://www.ebi.ac.uk/chembl/api/data/molecule.json?{query}"
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
            return "", ""
        except (urllib.error.URLError, TimeoutError, OSError, RemoteDisconnected):
            if attempt >= 2:
                return "", ""
            time.sleep(1.0 * (attempt + 1))
    if not isinstance(payload, dict):
        return "", ""
    molecules = payload.get("molecules")
    if not isinstance(molecules, list) or not molecules:
        return "", ""
    first = molecules[0]
    if not isinstance(first, dict):
        return "", ""
    chembl_id = str(first.get("molecule_chembl_id") or "").strip()
    pref_name = str(first.get("pref_name") or "").strip()
    return chembl_id, pref_name
