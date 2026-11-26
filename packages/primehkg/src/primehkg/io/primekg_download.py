from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
from contextlib import closing
from pathlib import Path

_DEFAULT_USER_AGENT = "primehkg/0.1 (local-first research)"

# PrimeKG Dataverse (doi:10.7910/DVN/IXA7BM) commonly used file ids.
# These ids are referenced by community notebooks and are stable in practice, but Dataverse may update them.
_PRIMEKG_DATAFILES: dict[str, int] = {
    "kg.csv": 6180620,
    "disease_features.tab": 6180618,
    "drug_features.tab": 6180619,
    "nodes.tab": 6180617,
    "edges.csv": 6180616,
    "README.txt": 6191270,
}


def _download(url: str, out_path: Path, *, user_agent: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not url.startswith(("https://", "http://")):
        msg = "download url must be http(s)"
        raise ValueError(msg)
    req = urllib.request.Request(url, headers={"User-Agent": user_agent})  # noqa: S310
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    try:
        # Dataverse often responds with a 303 redirect to an object store. Some Python
        # versions can keep the connection open without EOF; prefer a Content-Length
        # driven copy loop when available.
        with closing(urllib.request.urlopen(req, timeout=300)) as r:  # noqa: S310
            content_length = r.headers.get("Content-Length")
            expected = (
                int(content_length)
                if content_length and content_length.isdigit()
                else None
            )

            written = 0
            with tmp.open("wb") as f:
                while True:
                    chunk = r.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
                    written += len(chunk)
                    if expected is not None and written >= expected:
                        break
        tmp.replace(out_path)
    finally:
        if tmp.exists() and not out_path.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Download PrimeKG files from Harvard Dataverse."
    )
    p.add_argument(
        "--out-dir",
        required=True,
        help="Output directory where downloaded files are written.",
    )
    p.add_argument(
        "--files",
        default="kg.csv,disease_features.tab,drug_features.tab",
        help=(
            "Comma-separated list of files to download. Known: "
            + ", ".join(sorted(_PRIMEKG_DATAFILES))
        ),
    )
    p.add_argument(
        "--user-agent",
        default=_DEFAULT_USER_AGENT,
        help="User-Agent header to send (Dataverse may reject default clients).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out_dir = Path(args.out_dir)
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        msg = f"error: failed to create --out-dir: {exc}"
        raise SystemExit(msg) from exc

    requested = [x.strip() for x in str(args.files).split(",") if x.strip()]
    if not requested:
        msg = "error: --files must not be empty"
        raise SystemExit(msg)

    unknown = sorted({x for x in requested if x not in _PRIMEKG_DATAFILES})
    if unknown:
        raise SystemExit(
            "error: unknown PrimeKG files: "
            + ", ".join(unknown)
            + " (known: "
            + ", ".join(sorted(_PRIMEKG_DATAFILES))
            + ")"
        )

    for name in requested:
        file_id = _PRIMEKG_DATAFILES[name]
        url = f"https://dataverse.harvard.edu/api/access/datafile/{file_id}"
        dest = out_dir / name
        try:
            _download(url, dest, user_agent=str(args.user_agent))
        except (OSError, ValueError, urllib.error.URLError) as exc:
            msg = f"error: failed to download {name}: {exc}"
            raise SystemExit(msg) from exc
        sys.stdout.write(f"downloaded {name} -> {dest}\n")

    return 0


def entrypoint() -> None:
    raise SystemExit(main())


if __name__ == "__main__":  # pragma: no cover
    entrypoint()
