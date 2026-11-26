"""Read the #231 zero arm from the frozen baseline record instead of training it.

Decision (B) on #231, taken after the Phase 1 smoke measured the collision:

* `knowledge_control="zero"` returns the frozen base logits BEFORE the late head runs, so
  with a frozen inner `logits.requires_grad` is `False` and a Lightning training step dies
  with `element 0 of tensors does not require grad and does not have a grad_fn`. Both zero
  cells failed exactly this way.
* `SpanCrossAttentionKnowledgeModel` never hit it because its inner is trainable -- its zero
  arm still backpropagates into the text encoder.

With a frozen inner the zero arm is not merely *similar* to the frozen baseline, it IS the
frozen baseline: identical weights, identical forward, no trainable parameter in the path.
Its val and test are therefore already recorded in `ner_span_text_phase3`, and re-deriving
them by training would spend GPU hours to reproduce a number that exists.

This keeps `real - zero` a literal column (the issue names that contrast), costs no compute,
and leaves the bitwise exact-zero guarantee resting on an early return rather than on a
multiply-by-zero that could break silently.

**The failure mode this module must prevent is a copied number being read as a measured
one.** Every record carries `source="frozen_baseline_record"` and `trained=False`, and the
aggregate is expected to print them.

Empirical support that the identification is real, **with each number's own source stated**:

* From the Phase 1 smoke (`phase1_smoke.md`): `disease/ev_shuffled` finished at test f1
  `0.854386568069458`, byte-identical to its frozen baseline -- on shuffled evidence the head
  learned to do nothing at all -- while `disease/ev_real` moved by +0.00036.
* From the Phase 3 confirmatory record (`phase3_confirm/cells.jsonl`), NOT the smoke:
  `chem` seed 3 moved by +0.00053 (0.9269563555717468 - 0.9264295101165771).

An earlier version of this docstring attributed the chem figure to the smoke as well. It
cannot be: Phase 1 ran `chem/ev_real` and the failed `chem/ev_zero` only, with no
`chem/ev_shuffled` cell to difference against. The design decision is unaffected -- the
identity was later confirmed bit-identical on all 60 EV-cold cells, checked per seed -- but a
citation that names the wrong phase is how a number later gets defended as evidence it never
was.
"""

from __future__ import annotations

import re
from pathlib import Path  # noqa: TC003 - runtime use in signatures below
from typing import Any

from knowhyper.span_baseline import resolve_baseline_checkpoint

# The keys ieforge logs. `test/f1` exactly -- a loose `[a-z_]*f1` also matches
# `test/macro_f1` and, being last-wins, would silently swap the recorded number.
_TEST_F1 = re.compile(r"\btest/f1\b[^0-9]*([01]\.\d+)")
_TEST_PRECISION = re.compile(r"\btest/precision\b[^0-9]*([01]\.\d+)")
_TEST_RECALL = re.compile(r"\btest/recall\b[^0-9]*([01]\.\d+)")


class ZeroArmError(RuntimeError):
    """The frozen baseline record cannot supply this zero arm."""


def _last(pattern: re.Pattern[str], text: str) -> float | None:
    found = pattern.findall(text)
    return float(found[-1]) if found else None


def read_zero_arm(
    *,
    runs_root: Path,
    backbone: str,
    dataset: str,
    seed: int,
    compute_digest: bool = False,
) -> dict[str, Any]:
    """Return the zero arm's recorded val/test, or raise.

    Delegates identity to `resolve_baseline_checkpoint`, so the backbone / dataset / seed /
    `bs=16` / `patience=100` gates all still apply. Without that, the zero column could come
    from `_phase2` while the real column comes from `_phase3` -- two protocols inside one
    contrast, which is the failure #231 decision (a) exists to avoid.
    """
    resolved = resolve_baseline_checkpoint(
        runs_root=runs_root,
        backbone=backbone,
        dataset=dataset,
        seed=seed,
        compute_digest=compute_digest,
    )
    log = resolved.checkpoint.parent.parent / "driver.log"
    if not log.is_file():
        msg = f"frozen baseline has no driver.log, cannot read its test score: {log}"
        raise ZeroArmError(msg)
    text = log.read_text(errors="replace")
    test = _last(_TEST_F1, text)
    if test is None:
        msg = (
            f"frozen baseline has no test/f1 in {log}; the zero arm cannot be reported "
            "and must not be invented"
        )
        raise ZeroArmError(msg)
    return {
        "backbone": backbone,
        "dataset": dataset,
        "seed": seed,
        "val": resolved.best_val_f1,
        "best_epoch": resolved.best_epoch,
        "test": test,
        "precision": _last(_TEST_PRECISION, text),
        "recall": _last(_TEST_RECALL, text),
        "checkpoint": str(resolved.checkpoint),
        "digest": resolved.digest,
        # Load-bearing provenance: this row was COPIED, not measured in this campaign.
        "source": "frozen_baseline_record",
        "trained": False,
    }
