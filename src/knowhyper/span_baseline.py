"""Resolve a frozen #220 span checkpoint for the late-logit verifier, fail-closed.

Variant 1 loads a matched #220 span baseline into `inner` and freezes it. Loading the
wrong checkpoint trains a different experiment than the one being reported, and the shapes
are compatible enough that `load_state_dict` would succeed -- so nothing downstream
notices. Every field the record carries is therefore checked, and any mismatch raises.

Three traps this closes, all found in the Phase 0 audit
(`docs/reports/data/ner_span_late_logit_2026_08_02/phase0_provenance.md`):

1. The checkpoints are NOT under `runs/ner/`. They live in `runs/ner_span_text_phase3`.
   `runs/ner/span019` -- the #202 matched baseline the issue also references -- holds ZERO
   checkpoints (cleaned), so it cannot serve as the frozen base.

2. Seeds are split across trees: `ner_span_text_phase2` holds 0-2 and
   `ner_span_text_phase3` holds 3-9. They are NOT one protocol: `_phase3` additionally
   passes `trainer.callbacks.1.patience=100`, disabling early stopping. Both compose
   `bs=16` / `epochs=10` / `benchmark_nolog`, and their checkpoints are byte-identical in
   size, so batch size and epochs cannot distinguish them -- the patience override is the
   only signal in the record. #231 therefore runs on `_phase3` ONLY and screens on seeds
   3-5, so one protocol covers both the screen and the n=10 confirmation.

3. `ner_span_text` and `ner_span_text_bs16_confirm` also hold a seed-0 base arm for some
   combos. A resolver globbing `runs/ner_span_text*` would find 2-3 candidates for one
   cell and pick silently, so the tree is named explicitly and never globbed.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path  # noqa: TC003 - runtime use in signatures below
from typing import Any

# The ONE tree #231's graph-only campaign reads. Named, never globbed -- see trap 3 above.
BASELINE_TREE = "ner_span_text_phase3"

# #220 re-ran the text arms for seeds 0--2 after fixing their trainer, then completed
# seeds 3--9 in Phase 3. Both trees use benchmark_nolog, bs=16, epochs=10 and
# patience=100, so together they are one protocol. Name each tree explicitly: globbing
# ``ner_span_text*`` also finds the invalid Phase 2 text runs and older probes.
TEXT_REAL_EARLY_TREE = "ner_span_text_phase2b"
TEXT_REAL_LATE_TREE = "ner_span_text_phase3"

# JNLPBA and NCBI-Disease are run after the completed #220 campaign.  Both base and
# text-real arms, all seeds 0--9, share one freshly fixed protocol tree; keeping it
# explicit avoids globbing into any of the invalid/legacy `ner_span_text*` roots.
EXTENSION_TREE = "ner_span_text_extension"
EXTENSION_DATASETS = frozenset({"jnlpba", "ncbi_disease"})

# Decision (a) on #231: screen on seeds that exist under the SAME protocol as the n=10
# confirmation. Seeds 0-2 exist only in `_phase2` (no `patience=100`), so screening there
# and confirming on 0-9 would mix protocols inside one frozen-baseline set.
SCREEN_SEEDS = (3, 4, 5)

# The frozen baseline must have run under exactly this protocol.
_REQUIRED_BATCH_SIZE = 16
_REQUIRED_PATIENCE_OVERRIDE = "trainer.callbacks.1.patience=100"

_CKPT = re.compile(r"epoch_(\d+)-f1_([0-9.]+)\.ckpt")
_BACKBONE_OF_MODEL = {
    "pubmedbert_span_classifier": "pubmedbert",
    "biolinkbert_span_classifier": "biolinkbert",
}


class BaselineMismatchError(RuntimeError):
    """The recorded baseline does not match the cell that wants to freeze it."""


def _expected_experiment(dataset: str, arm: str) -> str:
    """Return the #220 experiment name for this dataset and frozen arm.

    BC5CDR-Chem is the UNSUFFIXED default (`span_ner_base`); the other two carry
    `_<dataset>`. Verified against the recorded overrides, not assumed -- the first version
    of this function required a suffix for all three and correctly rejected all six chem
    cells. Same convention as #218's `khp_span_ner_edge` family, where chem is likewise the
    bare name; a resolver that accepts either form for either dataset would silently pair
    chem cells with a bc2gm baseline.
    """
    stems = {
        "base": "span_ner_base",
        "text_real": "span_ner_retrieved_text",
    }
    try:
        stem = stems[arm]
    except KeyError as error:
        msg = f"unsupported frozen span arm {arm!r}; expected one of {sorted(stems)}"
        raise BaselineMismatchError(msg) from error
    return stem if dataset == "bc5cdr_chem" else f"{stem}_{dataset}"


def _tree_for(*, arm: str, seed: int, dataset: str) -> str:
    if dataset in EXTENSION_DATASETS:
        _expected_experiment(dataset, arm)
        return EXTENSION_TREE
    if arm == "base":
        return BASELINE_TREE
    if arm == "text_real":
        return TEXT_REAL_EARLY_TREE if seed <= 2 else TEXT_REAL_LATE_TREE
    # Keep arm validation in one place and fail before consulting the filesystem.
    _expected_experiment("bc5cdr_chem", arm)
    raise AssertionError


@dataclass(frozen=True)
class ResolvedBaseline:
    """One matched, verified span baseline checkpoint."""

    checkpoint: Path
    arm: str
    source_tree: str
    backbone: str
    dataset: str
    seed: int
    best_epoch: int
    best_val_f1: float
    digest: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "checkpoint": str(self.checkpoint),
            "arm": self.arm,
            "source_tree": self.source_tree,
            "backbone": self.backbone,
            "dataset": self.dataset,
            "seed": self.seed,
            "best_epoch": self.best_epoch,
            "best_val_f1": self.best_val_f1,
            "digest": self.digest,
        }


def _read_overrides(path: Path) -> list[str]:
    if not path.is_file():
        msg = f"baseline overrides missing, cannot verify the match: {path}"
        raise BaselineMismatchError(msg)
    return [
        line.strip().removeprefix("- ").strip().strip("'")
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def _digest(path: Path) -> str:
    """sha256 of the checkpoint, so a cell records WHICH weights it froze.

    Read in chunks: these files are ~500 MB.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_baseline_checkpoint(
    *,
    runs_root: Path,
    backbone: str,
    dataset: str,
    seed: int,
    arm: str = "base",
    compute_digest: bool = True,
) -> ResolvedBaseline:
    """Return the one matched baseline for this cell, or raise.

    `compute_digest=False` skips the sha256 read; use it only where the digest is not being
    recorded, never to weaken the identity check that follows.
    """
    source_tree = _tree_for(arm=arm, seed=seed, dataset=dataset)
    run_dir = runs_root / source_tree / backbone / dataset / arm / f"seed_{seed}"
    checkpoints = sorted((run_dir / "checkpoints").glob("*.ckpt"))
    if not checkpoints:
        msg = (
            f"no baseline checkpoint for {backbone}/{dataset}/seed_{seed} under "
            f"{source_tree}/{arm}: looked in {run_dir / 'checkpoints'}"
        )
        raise BaselineMismatchError(msg)
    if len(checkpoints) > 1:
        names = ", ".join(p.name for p in checkpoints)
        msg = (
            f"{len(checkpoints)} checkpoints in {run_dir}; `save_top_k=1` means one, so "
            f"the run was re-entered or two protocols were mixed: {names}"
        )
        raise BaselineMismatchError(msg)
    checkpoint = checkpoints[0]

    overrides = _read_overrides(run_dir / ".hydra" / "overrides.yaml")
    recorded = dict(
        entry.split("=", 1) for entry in overrides if "=" in entry and " " not in entry
    )

    model = recorded.get("model", "")
    recorded_backbone = _BACKBONE_OF_MODEL.get(model)
    if recorded_backbone != backbone:
        msg = (
            f"baseline backbone mismatch: cell wants {backbone!r} but the run recorded "
            f"model={model!r} ({recorded_backbone!r}). Loading these weights would "
            "succeed and report a different experiment."
        )
        raise BaselineMismatchError(msg)

    experiment = recorded.get("experiment", "")
    expected_experiment = _expected_experiment(dataset, arm)
    if experiment != expected_experiment:
        msg = (
            f"baseline dataset mismatch: cell wants {dataset!r} (experiment "
            f"{expected_experiment!r}) but the run recorded "
            f"experiment={experiment!r}"
        )
        raise BaselineMismatchError(msg)

    if recorded.get("seed") != str(seed):
        msg = (
            f"baseline seed mismatch: directory says seed_{seed} but the run recorded "
            f"seed={recorded.get('seed')!r}"
        )
        raise BaselineMismatchError(msg)

    if _REQUIRED_PATIENCE_OVERRIDE not in overrides:
        msg = (
            f"baseline protocol mismatch: {_REQUIRED_PATIENCE_OVERRIDE!r} absent, so this "
            f"is a `ner_span_text_phase2`-style run with early stopping ACTIVE. Both "
            "protocols compose bs=16/epochs=10, so this override is the only signal; "
            "mixing them inside one frozen-baseline set is what decision (a) avoids."
        )
        raise BaselineMismatchError(msg)

    config = run_dir / ".hydra" / "config.yaml"
    if not config.is_file():
        msg = f"baseline config missing, cannot verify batch_size: {config}"
        raise BaselineMismatchError(msg)
    match = re.search(r"^\s*batch_size:\s*(\d+)\s*$", config.read_text(), re.MULTILINE)
    if match is None or int(match.group(1)) != _REQUIRED_BATCH_SIZE:
        found = match.group(1) if match else "absent"
        msg = (
            f"baseline batch_size mismatch: want {_REQUIRED_BATCH_SIZE}, found {found}. "
            "bs=64 is the RE default that already depressed every span arm in #218."
        )
        raise BaselineMismatchError(msg)

    name = _CKPT.search(checkpoint.name)
    if name is None:
        msg = (
            f"baseline checkpoint name does not carry epoch/f1: {checkpoint.name}. That "
            "filename is the only record of best epoch and best val f1."
        )
        raise BaselineMismatchError(msg)

    return ResolvedBaseline(
        checkpoint=checkpoint,
        arm=arm,
        source_tree=source_tree,
        backbone=backbone,
        dataset=dataset,
        seed=seed,
        best_epoch=int(name.group(1)),
        best_val_f1=float(name.group(2)),
        digest=_digest(checkpoint) if compute_digest else "",
    )
