#!/usr/bin/env python3
"""Which SubHG cache does an experiment resolve to, and is it already built?

Answers the question that cost several wrong turns on 2026-07-26: two arms that
should share retrieval had drifted apart, and the only way to tell was to read
manifests by hand and guess which directory belonged to which config.

    uv run python scripts/resolve_cache_fingerprint.py khp_biored_xattn biored_retrieved_definitions
    uv run python scripts/resolve_cache_fingerprint.py --all-main-table

A fingerprint tells you the settings differ. It does NOT tell you which is
right -- so when two arms of one dataset disagree, look at the profile diff this
prints, not at the hashes.

The profile must be assembled exactly as PrimeHKGRetriever.__init__ does
(primehkg_backend.py:291) or the hash silently differs from the real one. In
particular `vector_candidate_sidecars` is a SORTED TUPLE of 3-tuples, not the
config's list -- getting that wrong produces a plausible-looking hash that
matches nothing on disk. `--self-check` verifies the recipe against caches whose
fingerprints are known, and runs by default.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from ieforge import CONFIGS_DIR as IEFORGE_CONFIGS_DIR
from omegaconf import OmegaConf

from knowhyper.cache.keys import method_fingerprint
from knowhyper.ieforge._configs import CONFIGS_DIR

# Mirrors PrimeHKGRetriever.__init__'s profile dict, in its order.
_PROFILE_KEYS = (
    "candidate_mode",
    "top_k",
    "hub_penalty",
    "hub_degree_flag_threshold",
    "semantics_weight",
    "evidence_weight",
    "max_hyperedge_size",
    "khop_budget",
)

# (experiment, expected fingerprint) pairs used to validate the recipe itself.
#
# These are PER DATA PIN, because `dataset_fingerprint` is an input to the hash: a
# PrimeHKG upgrade legitimately changes every value here. Keyed by the core
# revision so an upgrade does not silently disable the self-check -- an unkeyed
# table would either fail forever (and get ignored) or be "fixed" by pasting in
# whatever the code currently produces, which validates nothing.
#
# To add a pin: run this script, confirm the caches it names actually exist on
# disk with the right split counts, then record the values here.
_SELF_CHECK_BY_CORE_REV = {
    "0.12.3": (
        ("biored_retrieved_definitions", "970fbf152d120495"),
        ("re_retrieved_definitions", "51fc536bb676a488"),
        ("ddi_retrieved_definitions", "962dbb5c2ecdbc97"),
    ),
    # The NER chain moved to 0.12.13, which left the self-check dormant: with no
    # entry for the current pin `_self_check()` announces "UNVALIDATED" and returns
    # True, so it validated nothing at the revision actually in use. That matters
    # more now that `run_ner_strict_precompute_node.sh` calls `resolve()` to decide
    # whether a precompute wrote anything -- the guard would have been routed through
    # an unvalidated recipe.
    #
    # Both caches below were built on ABCI on 2026-07-28 and confirmed with all three
    # splits at their exact instance counts (4,560 / 4,581 / 4,797), per the runs'
    # own summary.json (`status: completed`, 559,382 and 564,909 JSON files). They
    # also reproduce the documented sharing: khp_ner_ev resolves to the same hash as
    # khp_ner, and khp_ner_ev_strict to the same as khp_ner_strict, which is what
    # makes one precompute serve two arms.
    #
    # NOTE these live under the ABCI KNOWHYPER_HOME, so a local run of this script
    # still reports them NOT BUILT; the self-check compares the FINGERPRINT, not
    # on-disk presence, so it is meaningful either way.
    "0.12.13": (
        ("khp_ner_strict", "71b330506dd2ee18"),
        ("khp_ner", "849c05bd0d2c712a"),
    ),
}

MAIN_TABLE = (
    "khp_re_xattn",
    "khp_re_xattn_ev",
    "khp_ddi_xattn",
    "khp_ddi_xattn_ev",
    "khp_biored_xattn",
    "khp_biored_xattn_ev",
    "re_retrieved_definitions",
    "ddi_retrieved_definitions",
    "biored_retrieved_definitions",
)


def _cache_root() -> Path:
    return (
        Path(os.environ.get("KNOWHYPER_HOME", str(Path.home() / ".knowhyper")))
        / "cache/subhg/v2"
    )


def resolve(experiment: str) -> tuple[str, dict[str, Any], str]:
    """Return (fingerprint, profile, dataset) for an experiment config."""
    with initialize_config_dir(version_base=None, config_dir=str(IEFORGE_CONFIGS_DIR)):
        cfg = compose(
            config_name="train",
            overrides=[
                f"experiment={experiment}",
                f"hydra.searchpath=[file://{CONFIGS_DIR}]",
            ],
        )
    retriever = cfg.knowledge.pipeline.injector.retriever
    resolved = OmegaConf.to_container(retriever, resolve=True)
    inner = OmegaConf.to_container(retriever.inner, resolve=True)

    profile: dict[str, Any] = {k: inner.get(k) for k in _PROFILE_KEYS}
    profile["vector_candidate_sidecars"] = tuple(
        sorted(
            (str(d["sidecar_dir"]), str(d["seed_prefix"]), int(d["top_k_nodes"]))
            for d in (inner.get("vector_candidate_sidecars") or [])
        )
    )
    profile["vector_boost"] = float(inner.get("vector_boost", 1.0))
    if inner.get("provenance_schema") is not None:
        profile["provenance_schema"] = inner["provenance_schema"]
    profile["linker_id"] = resolved["linker_id"]
    for key in ("provenance_schema", "preparation_schema"):
        if resolved.get(key) is not None:
            profile[key] = resolved[key]

    fingerprint = method_fingerprint(
        profile,
        primehkg_rev=cfg.knowhyper.primehkg_rev,
        dataset_fingerprint=cfg.knowhyper.dataset_fingerprint,
        cache_schema_version=cfg.knowhyper.subhg_cache_schema_version,
        retrieval_semantics_version=cfg.knowhyper.retrieval_semantics_version,
    )
    return fingerprint, profile, _dataset_name(cfg)


def _dataset_name(cfg: Any) -> str:
    """Directory name the cache uses under <fingerprint>/.

    `knowledge.dataset` is the core-loader block, not a name, so read the
    dataset group's own identity instead and fall back to whatever single
    directory the built cache contains.
    """
    for path in ("dataset.dataset_name", "dataset.name", "task.dataset"):
        node = cfg
        for part in path.split("."):
            node = node.get(part) if hasattr(node, "get") else None
            if node is None:
                break
        if isinstance(node, str) and node:
            return node.split("/")[-1]
    return ""


def _splits(fingerprint: str, dataset: str) -> str:
    root = _cache_root() / fingerprint
    if not root.is_dir():
        return "NOT BUILT"
    d = root / dataset if dataset else None
    if d is None or not d.is_dir():
        # fall back to the sole dataset directory under this fingerprint
        subdirs = [p for p in root.iterdir() if p.is_dir()]
        if len(subdirs) != 1:
            return "NOT BUILT"
        d = subdirs[0]
    parts = []
    for split in ("train", "validation", "test"):
        n = len(list((d / split).glob("*.json"))) if (d / split).is_dir() else 0
        parts.append(f"{split}={n}" if n else f"{split}=-")
    return "  ".join(parts)


def _core_revision() -> str:
    """The core data revision the config currently pins."""
    import re

    cfg = (
        Path(__file__).resolve().parents[1]
        / "src/knowhyper/configs/knowledge/knowhyper.yaml"
    ).read_text()
    m = re.search(r'^\s*dataset_revision:\s*"?([^"\s#]+)"?', cfg, re.MULTILINE)
    return m.group(1) if m else ""


def _self_check() -> bool:
    rev = _core_revision()
    pairs = _SELF_CHECK_BY_CORE_REV.get(rev)
    if pairs is None:
        # Say so loudly rather than pass by default. Every fingerprint below is
        # unvalidated until someone confirms the caches exist for this pin.
        print(
            f"  self-check: NO known fingerprints for core revision {rev!r} -- "
            "the recipe is UNVALIDATED here."
        )
        print(
            "  Confirm the caches this run names actually exist with the right "
            "split counts, then add them to _SELF_CHECK_BY_CORE_REV."
        )
        return True
    ok = True
    print(f"  (self-check pinned to core revision {rev})")
    for experiment, expected in pairs:
        try:
            got, _, _ = resolve(experiment)
        except Exception as exc:
            print(f"  self-check {experiment}: FAILED to resolve ({exc})")
            ok = False
            continue
        mark = "ok" if got == expected else "MISMATCH"
        if got != expected:
            ok = False
        print(f"  self-check {experiment:30} {got}  expect {expected}  {mark}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("experiments", nargs="*", help="experiment config names")
    ap.add_argument("--all-main-table", action="store_true")
    ap.add_argument("--profile", action="store_true", help="print the full profile")
    ap.add_argument("--skip-self-check", action="store_true")
    args = ap.parse_args()

    names = list(args.experiments)
    if args.all_main_table:
        names = list(MAIN_TABLE)
    if not names:
        ap.error("give experiment names or --all-main-table")

    if not args.skip_self_check:
        print("Validating the fingerprint recipe against known caches:")
        if not _self_check():
            print(
                "\nRecipe does NOT reproduce known fingerprints -- the profile "
                "construction has drifted from PrimeHKGRetriever.__init__. Fix "
                "that before trusting anything below.",
                file=sys.stderr,
            )
            return 1
        print()

    by_fp: dict[str, list[str]] = {}
    for name in names:
        try:
            fingerprint, profile, dataset = resolve(name)
        except Exception as exc:
            print(f"{name:32} -- no knowledge pipeline ({type(exc).__name__})")
            continue
        by_fp.setdefault(fingerprint, []).append(name)
        print(f"{name:32} {fingerprint}  {_splits(fingerprint, dataset)}")
        if args.profile:
            print(json.dumps(profile, indent=2, sort_keys=True, default=str))

    shared = {k: v for k, v in by_fp.items() if len(v) > 1}
    if shared:
        print("\nSharing a cache (same retrieval):")
        for fingerprint, group in shared.items():
            print(f"  {fingerprint}: {', '.join(group)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
