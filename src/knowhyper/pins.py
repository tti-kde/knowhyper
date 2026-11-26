"""The single source of truth for which PrimeHKG build everything reads.

Import this instead of writing a revision or fingerprint literal. Every value is
parsed from `configs/knowledge/knowhyper.yaml`, so a pin bump reaches Hydra configs
and plain scripts at the same time.

WHY THIS EXISTS. Over 2026-07-27/28 a PrimeHKG bump produced four separate
silent-wrong-data incidents, each costing a re-run, and every one had the same shape:
a literal `"0.12.3"` somewhere that did not move when the central pin did.

  1. The four `*_ev` experiment configs overrode `dataset_revision` to "0.12.3", so
     the upgrade never reached the EV arms. Reported as measured on the new pin; it
     was the old data.
  2. `composite.yaml` passed `${knowhyper.dataset_revision}` (CORE) to
     `resolve_features_dir` (FEATURES). Identical strings while both repos sat at
     0.12.3; the moment they diverged, every EV cell died with
     LocalEntryNotFoundError.
  3. `run_pretrain.py --revision/--fingerprint` defaulted to 0.12.3, so an encoder
     built on the NEW sub-HG cache read node features from the OLD core. Mixed
     conditions, result retracted.
  4. Three tests froze the pin values and so locked in (1) rather than catching it.

The pattern is that a literal is *correct until it isn't*, and nothing fails loudly
when it goes stale -- the run completes and reports a number. Two defences here:

* `CORE_REVISION` / `FEATURES_REVISION` are distinct names. Passing the wrong one is
  now a visible mistake at the call site rather than an invisible coincidence.
* `assert_distinct_repos()` refuses to let a check pass *because* the two revisions
  happen to be equal, which is how (2) hid for a week.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

_CONFIG = Path(__file__).resolve().parent / "configs" / "knowledge" / "knowhyper.yaml"


_KEYS = (
    "primehkg_rev",
    "dataset_revision",
    "dataset_fingerprint",
    "features_revision",
    "features_fingerprint",
    "text_features_revision",
    "text_features_fingerprint",
)


@lru_cache(maxsize=1)
def _pins() -> dict[str, str]:
    """Read the pins out of the `knowhyper:` block.

    Parsed with a real YAML loader, scoped to the `knowhyper:` key, rather than
    regex-matching the file. A regex here has two silent-wrong-value modes that
    matter for a module whose entire job is to never return a wrong pin: it matches
    the first occurrence at ANY indent under ANY parent (so an unrelated block that
    happens to contain `dataset_revision:` wins if it appears earlier), and it does
    not handle single-quoted scalars (`'0.12.13'` came back with the quotes attached,
    which then fails Hub lookup). The file is 130-plus lines and growing, so both get
    more likely, not less.
    """
    import yaml

    doc = yaml.safe_load(_CONFIG.read_text()) or {}
    block = doc.get("knowhyper") if isinstance(doc, dict) else None
    if block is None or not isinstance(block, dict):
        # A config problem, not a caller type error: the file exists but does not
        # have the shape every pin is read out of.
        msg = f"{_CONFIG} has no top-level `knowhyper:` mapping"
        raise RuntimeError(msg)

    from omegaconf import OmegaConf

    out: dict[str, str] = {}
    for key in _KEYS:
        if key not in block:
            msg = f"{_CONFIG} `knowhyper:` block has no {key}"
            raise RuntimeError(msg)
        # Resolve `${oc.env:...}` the same way Hydra does, so the PRIMEHKG_*
        # environment overrides (README, "Using your own PrimeHKG build") reach
        # plain scripts exactly as they reach composed configs.
        out[key] = str(
            OmegaConf.to_container(OmegaConf.create({"v": block[key]}), resolve=True)[
                "v"
            ]
        )
    return out


#: v-prefixed package/cache pin, e.g. "v0.12.14". Feeds cache fingerprints.
PACKAGE_REV = _pins()["primehkg_rev"]

#: primehkg-CORE data tag (no "v"), e.g. "0.12.13". Nodes, hyperedges, sidecars.
CORE_REVISION = _pins()["dataset_revision"]
CORE_FINGERPRINT = _pins()["dataset_fingerprint"]

#: primehkg-FEATURES tag for EV attribute embeddings. NOT the same as CORE.
FEATURES_REVISION = _pins()["features_revision"]
FEATURES_FINGERPRINT = _pins()["features_fingerprint"]

#: primehkg-FEATURES tag for raw provenance-bearing text (R2a). Pinned separately
#: on purpose: refreshing text must not invalidate structure/EV SubHG caches.
TEXT_FEATURES_REVISION = _pins()["text_features_revision"]
TEXT_FEATURES_FINGERPRINT = _pins()["text_features_fingerprint"]

#: Hub repo id OR local directory of a PrimeHKG build (hub layout). Override with
#: PRIMEHKG_CORE / PRIMEHKG_FEATURES; the defaults are the builds the paper used,
#: which are not redistributed (see README).
CORE_REPO = os.environ.get("PRIMEHKG_CORE", "nishide-dev/primehkg-core")
FEATURES_REPO = os.environ.get("PRIMEHKG_FEATURES", "nishide-dev/primehkg-features")


def assert_distinct_repos() -> None:
    """Fail if CORE and FEATURES revisions are equal.

    Not a style check. `composite.yaml` handed the core revision to a features
    resolver and nothing noticed for as long as both repos read "0.12.3" -- the
    coincidence *was* the bug's camouflage. A test that only passes while the two
    agree provides no coverage of the thing it claims to test, so callers that rely
    on the distinction should assert it up front.
    """
    if CORE_REVISION == FEATURES_REVISION:
        msg = (
            f"core and features revisions are both {CORE_REVISION!r}. A wrong-repo "
            "revision cannot be detected in this state -- pin them apart, or skip "
            "any check that depends on telling them apart."
        )
        raise AssertionError(msg)
