from __future__ import annotations

import hashlib
import ssl
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DownloadResult:
    """One downloaded file with content hash and timestamp."""

    url: str
    path: Path
    sha256: str
    downloaded_at: str


def download_file(
    *,
    url: str,
    out_path: str | Path,
    timeout_s: float = 60.0,
    user_agent: str = "primehkg-curated-fetch/0.x",
    insecure_ssl: bool = False,
) -> DownloadResult:
    scheme = urllib.parse.urlparse(url).scheme.lower()
    if scheme not in {"https", "http"}:
        msg = f"Unsupported URL scheme for download: {scheme!r}"
        raise ValueError(msg)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    req = urllib.request.Request(url, headers={"User-Agent": user_agent})  # noqa: S310
    ctx = None
    if insecure_ssl:
        # Intentionally opt-out of certificate validation for environments that cannot install
        # required CA roots (e.g. certain academic CA chains). Use only when necessary.
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
    with urllib.request.urlopen(req, timeout=timeout_s, context=ctx) as resp:  # noqa: S310
        data = resp.read()

    out.write_bytes(data)
    sha = hashlib.sha256(data).hexdigest()
    ts = datetime.now(UTC).isoformat()
    return DownloadResult(url=url, path=out, sha256=sha, downloaded_at=ts)


def post_form(
    *,
    url: str,
    form: dict[str, str],
    timeout_s: float = 60.0,
    user_agent: str = "primehkg-curated-fetch/0.x",
    insecure_ssl: bool = False,
) -> bytes:
    """POST ``application/x-www-form-urlencoded`` data and return the response body."""
    scheme = urllib.parse.urlparse(url).scheme.lower()
    if scheme not in {"https", "http"}:
        msg = f"Unsupported URL scheme for POST: {scheme!r}"
        raise ValueError(msg)

    body = urllib.parse.urlencode(form).encode("utf-8")
    req = urllib.request.Request(  # noqa: S310
        url,
        data=body,
        method="POST",
        headers={
            "User-Agent": user_agent,
            "Content-Type": "application/x-www-form-urlencoded",
        },
    )
    ctx = None
    if insecure_ssl:
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
    with urllib.request.urlopen(req, timeout=timeout_s, context=ctx) as resp:  # noqa: S310
        return resp.read()


def fetch_text(
    *,
    url: str,
    timeout_s: float = 60.0,
    user_agent: str = "primehkg-curated-fetch/0.x",
    insecure_ssl: bool = False,
) -> str:
    scheme = urllib.parse.urlparse(url).scheme.lower()
    if scheme not in {"https", "http"}:
        msg = f"Unsupported URL scheme for fetch: {scheme!r}"
        raise ValueError(msg)

    req = urllib.request.Request(url, headers={"User-Agent": user_agent})  # noqa: S310
    ctx = None
    if insecure_ssl:
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
    with urllib.request.urlopen(req, timeout=timeout_s, context=ctx) as resp:  # noqa: S310
        return resp.read().decode("utf-8")


def get_json(
    *,
    url: str,
    timeout_s: float = 60.0,
    user_agent: str = "primehkg-curated-fetch/0.x",
    allow_http: bool = False,
    insecure_ssl: bool = False,
) -> Any:
    scheme = urllib.parse.urlparse(url).scheme.lower()
    if scheme not in {"https", "http"}:
        msg = f"Unsupported URL scheme for download: {scheme!r}"
        raise ValueError(msg)
    if scheme == "http" and not allow_http:
        msg = "Refusing insecure http URL for JSON fetch. Use https or set allow_http=True."
        raise ValueError(msg)

    req = urllib.request.Request(url, headers={"User-Agent": user_agent})  # noqa: S310
    ctx = None
    if insecure_ssl:
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
    with urllib.request.urlopen(req, timeout=timeout_s, context=ctx) as resp:  # noqa: S310
        data = resp.read()
    # Avoid heavy dependency; JSON parsing is done at call sites.
    import json

    return json.loads(data)
