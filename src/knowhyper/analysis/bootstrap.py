"""Paired test-instance bootstrap and seed-level statistics for RE contrasts.

Two sources of uncertainty are reported side by side and never merged:

* **seed level** -- the paired t statistics over training seeds that every
  recorded aggregate already uses (training randomness);
* **instance level** -- a paired bootstrap over test instances with the ten
  trained models held fixed (test-sample uncertainty of the seed-averaged
  delta).

Both arms of a contrast are resampled with the *same* instance draw, and the
metric is the exact-label micro F1 with the negative class excluded
(:func:`knowhyper.analysis.re_prediction_export.compute_micro_f1`), recomputed
inside every resample from per-instance TP/FP/FN indicators.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import stats as sp_stats

from knowhyper.analysis.re_prediction_export import (
    PredictionRecord,
    read_predictions,
    sha256_file,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

DEFAULT_RESAMPLES = 10_000
DEFAULT_RNG_SEED = 20260917
#: bf16-mixed inference is not bit-reproducible across processes: the F1
#: recomputed from an in-job prediction export drifts from the trained model's
#: own test pass by <= 0.0023 per seed (both signs, mean shift <= 0.0004) over
#: 90 recorded RE cells. Larger drift means the export rebuilt a different model
#: and must not pass.
#:
#: **This is the single definition of the rule** (#314 review F6).
#: ``scripts/run_matrix_cell.py`` re-exports it as ``EXPORT_PARITY_TOL`` and
#: ``finalize_export_parity.py`` imports that, so the write-time warning and the
#: pairing gate cannot drift apart.
DEFAULT_PARITY_TOL = 2.5e-3
#: Below this many training seeds no seed-level test exists at all -- distinct
#: from "the test ran but t is undefined" (a constant delta at n >= 2), which is
#: a real multi-seed result and must keep its DISAGREE handling.
_MIN_SEEDS_FOR_A_TEST = 2
_CHUNK = 1_000


# --- per-instance outcomes -------------------------------------------------


def align_instance_ids(
    groups: Sequence[Sequence[PredictionRecord]],
) -> tuple[str, ...]:
    """Return the sorted instance ids shared by every group, or raise.

    Every group must carry each id once, all groups must carry the same set,
    *and* must agree on each instance's gold label. Pairing two arms whose test
    populations differ is how a linking difference becomes a "result"; pairing
    two arms that disagree on a gold label -- a label-set or label-order change
    between campaigns -- is how a relabelling becomes one, and the delta would
    be silently wrong rather than loudly missing.
    """
    if not groups:
        msg = "at least one prediction group is required"
        raise ValueError(msg)
    reference: dict[str, str] | None = None
    for index, records in enumerate(groups):
        golds = {record.instance_id: record.gold_label for record in records}
        if len(golds) != len(records):
            msg = f"group {index} has duplicate instance ids"
            raise ValueError(msg)
        if reference is None:
            reference = golds
            continue
        if golds.keys() != reference.keys():
            only_ref = len(reference.keys() - golds.keys())
            only_here = len(golds.keys() - reference.keys())
            msg = (
                f"instance ids differ between groups (group {index}: "
                f"{only_ref} missing, {only_here} extra)"
            )
            raise ValueError(msg)
        clashes = sorted(k for k, v in golds.items() if reference[k] != v)
        if clashes:
            first = clashes[0]
            msg = (
                f"gold labels differ between groups (group {index}: "
                f"{len(clashes)} instances, e.g. {first}: "
                f"{reference[first]!r} vs {golds[first]!r})"
            )
            raise ValueError(msg)
    return tuple(sorted(reference or ()))


def instance_outcomes(
    records: Sequence[PredictionRecord],
    instance_ids: Sequence[str],
    *,
    negative_label: str,
) -> np.ndarray:
    """Per-instance ``[tp, fp, fn]`` indicators in ``instance_ids`` order.

    Mirrors ``compute_micro_f1`` exactly: a positive prediction with the wrong
    positive label counts as one FP *and* one FN.
    """
    position = {instance_id: k for k, instance_id in enumerate(instance_ids)}
    out = np.zeros((len(instance_ids), 3), dtype=np.int8)
    for record in records:
        row = position[record.instance_id]
        gold_positive = record.gold_label != negative_label
        pred_positive = record.predicted_label != negative_label
        exact = record.predicted_label == record.gold_label
        if pred_positive and gold_positive and exact:
            out[row, 0] = 1
        elif pred_positive:
            out[row, 1] = 1
        if gold_positive and not exact:
            out[row, 2] = 1
    return out


def f1_from_counts(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray) -> np.ndarray:
    """Vectorised micro F1; ``0.0`` wherever precision + recall is undefined."""
    tp = np.asarray(tp, dtype=np.float64)
    fp = np.asarray(fp, dtype=np.float64)
    fn = np.asarray(fn, dtype=np.float64)
    denominator = 2.0 * tp + fp + fn
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where(denominator > 0, 2.0 * tp / denominator, 0.0)
    return np.asarray(f1, dtype=np.float64)


# --- seed level --------------------------------------------------------------


def seed_level_stats(f1_a: np.ndarray, f1_b: np.ndarray) -> dict[str, Any]:
    """Paired t statistics over seeds for ``f1_a - f1_b``.

    The interval is the t-interval ``mean ± t_{.975, n-1} · sd / sqrt(n)``; the
    p-value is two-sided.  Identical arms give ``p = 1`` rather than NaN.

    Whatever is *not estimable* is ``None``, never a degenerate number: at
    ``n == 1`` there is no seed-level uncertainty to report at all, and a
    constant non-zero delta has an undefined t rather than an infinite one.
    Returning ``0.000`` and a zero-width interval there would read in the record
    as a very strong result obtained from a single run.
    """
    a = np.asarray(f1_a, dtype=np.float64)
    b = np.asarray(f1_b, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1:
        msg = f"per-seed arrays must be 1-d and paired, got {a.shape} vs {b.shape}"
        raise ValueError(msg)
    n = int(a.size)
    delta = a - b
    mean = float(delta.mean())
    sd = float(delta.std(ddof=1)) if n > 1 else None
    t_stat: float | None
    p_value: float | None
    half: float | None
    if n > 1 and sd is not None and sd > 0.0:
        t_stat = mean / (sd / math.sqrt(n))
        p_value = float(2.0 * sp_stats.t.sf(abs(t_stat), df=n - 1))
        half = float(sp_stats.t.ppf(0.975, df=n - 1)) * sd / math.sqrt(n)
    elif n > 1 and mean == 0.0:
        # identical arms: the paired test is defined and answers "no difference"
        t_stat, p_value, half = 0.0, 1.0, 0.0
    else:
        t_stat = p_value = half = None
    return {
        "n_seeds": n,
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "sd_a": float(a.std(ddof=1)) if n > 1 else None,
        "sd_b": float(b.std(ddof=1)) if n > 1 else None,
        "mean_delta": mean,
        "sd_delta": sd,
        "t": t_stat,
        "df": n - 1,
        "p": p_value,
        "ci_low": None if half is None else mean - half,
        "ci_high": None if half is None else mean + half,
        "wins": int((delta > 0).sum()),
        "losses": int((delta < 0).sum()),
        "ties": int((delta == 0).sum()),
        "per_seed_delta": [float(d) for d in delta],
        "estimable": p_value is not None,
    }


def unavailable_seed_stats(missing: Sequence[str]) -> dict[str, Any]:
    """Return the seed-level block for a contrast with incomplete recorded values.

    Every statistic is ``None`` and the arm/seed pairs whose recorded test F1
    could not be found are named, so a gap in the provenance reads as a gap and
    never as a silent fallback to the recomputed numbers.
    """
    return {
        "n_seeds": 0,
        "mean_a": None,
        "mean_b": None,
        "sd_a": None,
        "sd_b": None,
        "mean_delta": None,
        "sd_delta": None,
        "t": None,
        "df": None,
        "p": None,
        "ci_low": None,
        "ci_high": None,
        "wins": None,
        "losses": None,
        "ties": None,
        "per_seed_delta": None,
        "estimable": False,
        "missing_recorded": list(missing),
    }


def holm_adjust(p_values: Sequence[float | None]) -> list[float | None]:
    """Holm-Bonferroni adjusted p-values, in the input order.

    ``None`` entries are not part of the family: an unreportable seed-level p
    (see :func:`seed_level_stats`) cannot be adjusted, and counting it would
    inflate the family size for the contrasts that *are* reportable. The family
    size is therefore the number of non-``None`` entries.
    """
    indexed = [(p, k) for k, p in enumerate(p_values) if p is not None]
    out: list[float | None] = list(p_values)
    k = len(indexed)
    running = 0.0
    for rank, (p, position) in enumerate(sorted(indexed)):
        running = max(running, min(1.0, float(p) * (k - rank)))
        out[position] = running
    return out


# --- instance level ----------------------------------------------------------


def _seed_mean_f1(counts: np.ndarray, outcomes: np.ndarray) -> np.ndarray:
    """Mean over seeds of the resampled F1, for one arm.

    ``counts`` is ``(R, n)`` resample multiplicities, ``outcomes`` is
    ``(S, n, 3)``.  Multiplicities are small integers, so the float32 matmul is
    exact.
    """
    f1_sum = np.zeros(counts.shape[0], dtype=np.float64)
    for seed_outcomes in outcomes:
        totals = counts @ seed_outcomes.astype(np.float32)  # (R, 3)
        f1_sum += f1_from_counts(totals[:, 0], totals[:, 1], totals[:, 2])
    return f1_sum / outcomes.shape[0]


def paired_instance_bootstrap(
    outcomes_a: np.ndarray,
    outcomes_b: np.ndarray,
    *,
    n_resamples: int = DEFAULT_RESAMPLES,
    rng_seed: int = DEFAULT_RNG_SEED,
) -> dict[str, Any]:
    """Percentile 95% interval for the seed-averaged delta over test instances.

    Each resample draws ``n`` instances with replacement *once* and applies the
    same draw to every seed of both arms, so the interval is on the paired
    difference.  Deterministic given ``rng_seed``.
    """
    a = np.asarray(outcomes_a)
    b = np.asarray(outcomes_b)
    if a.ndim != 3 or b.ndim != 3 or a.shape[1:] != b.shape[1:]:
        msg = f"outcome stack shape mismatch: need (S, n, 3) with equal n, got {a.shape}/{b.shape}"
        raise ValueError(msg)
    if n_resamples <= 0:
        msg = "n_resamples must be positive"
        raise ValueError(msg)
    n = a.shape[1]
    observed_a = f1_from_counts(*a.sum(axis=1).T)
    observed_b = f1_from_counts(*b.sum(axis=1).T)
    observed_delta = float(observed_a.mean() - observed_b.mean())

    rng = np.random.default_rng(rng_seed)
    probabilities = np.full(n, 1.0 / n)
    deltas = np.empty(n_resamples, dtype=np.float64)
    f1_a = np.empty(n_resamples, dtype=np.float64)
    f1_b = np.empty(n_resamples, dtype=np.float64)
    for start in range(0, n_resamples, _CHUNK):
        stop = min(start + _CHUNK, n_resamples)
        counts = rng.multinomial(n, probabilities, size=stop - start).astype(np.float32)
        f1_a[start:stop] = _seed_mean_f1(counts, a)
        f1_b[start:stop] = _seed_mean_f1(counts, b)
        deltas[start:stop] = f1_a[start:stop] - f1_b[start:stop]

    low, high = np.percentile(deltas, [2.5, 97.5])
    a_low, a_high = np.percentile(f1_a, [2.5, 97.5])
    b_low, b_high = np.percentile(f1_b, [2.5, 97.5])
    return {
        "n_instances": int(n),
        "n_seeds_a": int(a.shape[0]),
        "n_seeds_b": int(b.shape[0]),
        "n_resamples": int(n_resamples),
        "rng_seed": int(rng_seed),
        "observed_f1_a": float(observed_a.mean()),
        "observed_f1_b": float(observed_b.mean()),
        "observed_delta": observed_delta,
        "ci_low": float(low),
        "ci_high": float(high),
        "ci_a": [float(a_low), float(a_high)],
        "ci_b": [float(b_low), float(b_high)],
        "bootstrap_mean_delta": float(deltas.mean()),
        "bootstrap_sd_delta": float(deltas.std(ddof=1)),
        "fraction_delta_le_zero": float((deltas <= 0).mean()),
    }


# --- loading recorded arms ---------------------------------------------------


@dataclass(frozen=True)
class ArmPredictions:
    """Per-seed test predictions of one recorded arm."""

    dataset: str
    condition: str
    root: Path
    seeds: tuple[int, ...]
    records_by_seed: Mapping[int, Sequence[PredictionRecord]]
    recorded_test_f1: Mapping[int, float | None]
    sha256_by_seed: Mapping[int, str]
    prediction_paths: Mapping[int, Path]


def load_arm(
    root: Path,
    dataset: str,
    condition: str,
    *,
    seeds: Sequence[int],
) -> ArmPredictions:
    """Read ``<root>/<dataset>/<condition>/seed_<s>/export/predictions.jsonl``.

    ``cell.json`` beside it supplies the test F1 the campaign recorded, used
    for the parity check; it may be absent (recorded as ``None``).
    """
    records: dict[int, Sequence[PredictionRecord]] = {}
    recorded: dict[int, float | None] = {}
    hashes: dict[int, str] = {}
    paths: dict[int, Path] = {}
    for seed in seeds:
        cell = Path(root) / dataset / condition / f"seed_{seed}"
        prediction_path = cell / "export" / "predictions.jsonl"
        if not prediction_path.is_file():
            msg = f"missing predictions for {dataset}/{condition}/seed_{seed}: {prediction_path}"
            raise FileNotFoundError(msg)
        records[seed] = read_predictions(prediction_path)
        hashes[seed] = sha256_file(prediction_path)
        paths[seed] = prediction_path
        cell_json = cell / "cell.json"
        value: float | None = None
        if cell_json.is_file():
            raw = json.loads(cell_json.read_text()).get("test_f1")
            value = float(raw) if raw is not None else None
        recorded[seed] = value
    return ArmPredictions(
        dataset=dataset,
        condition=condition,
        root=Path(root),
        seeds=tuple(seeds),
        records_by_seed=records,
        recorded_test_f1=recorded,
        sha256_by_seed=hashes,
        prediction_paths=paths,
    )


# --- one contrast --------------------------------------------------------------


@dataclass(frozen=True)
class ContrastSpec:
    """Which two recorded arms to pair, and where their predictions live."""

    name: str
    dataset: str
    negative_label: str
    arm_a: tuple[str, Path]
    arm_b: tuple[str, Path]
    seeds: tuple[int, ...]
    note: str = ""
    recorded_a: Mapping[int, float] | None = None
    recorded_b: Mapping[int, float] | None = None


@dataclass
class ContrastResult:
    """Everything a table row or a JSON record needs for one contrast."""

    contrast: str
    dataset: str
    negative_label: str
    condition_a: str
    condition_b: str
    root_a: str
    root_b: str
    seeds: tuple[int, ...]
    n_instances: int
    n_seeds: int
    seed: dict[str, Any]
    bootstrap: dict[str, Any]
    per_seed: dict[int, dict[str, Any]]
    parity_ok: bool
    parity_tol: float
    #: ``recorded`` -- the confirmatory block is the recorded per-seed test F1
    #: (what the paper prints); ``unavailable`` -- at least one seed has no
    #: recorded value, so no confirmatory seed statistic exists.
    seed_source: str = "recorded"
    #: Seed statistics recomputed from the re-exported predictions. Secondary,
    #: for transparency: bf16-mixed export drift of a few 1e-4 per seed can flip
    #: a near-tie seed and with it the win count and the t statistic.
    seed_recomputed: dict[str, Any] = field(default_factory=dict)
    #: ``checked`` -- every arm x seed was compared against a recorded value;
    #: ``vacuous`` -- at least one had no recorded value to compare against;
    #: ``unchecked`` -- ``strict_parity`` was off, so a break did not stop it.
    parity_mode: str = "checked"
    #: Number of arm x seed pairs actually compared.
    parity_checked: int = 0
    note: str = ""
    instance_ids_sha256: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def as_row(self) -> dict[str, Any]:
        """Return the flat row the rebuttal table is built from.

        Seed-level fields come from :attr:`seed` -- the recorded statistic --
        and are **blank**, never zero, wherever it is not estimable.

        A block with **no seed-level test at all** (fewer than two seeds) blanks
        the DELTA and the win count too, not just the dispersion. At n_seeds=1
        the difference of two single runs is a number, but it is not a seed-level
        statistic, and a populated `seed_mean_delta` beside
        `seed_source=recorded` read as one (#314 review F2). The instance-level
        columns still carry the observed delta, which is where that quantity
        legitimately lives.

        This keys on `seed_source`, i.e. on the seed COUNT -- the same condition
        the renderer uses. Keying it on `estimable` instead made the TSV blank a
        delta that `aggregate.txt` still printed, so the machine-readable and
        human-readable records disagreed about whether the quantity existed.
        """
        blank = lambda v: "" if v is None else v  # noqa: E731
        has_seed_test = self.seed_source != "not_estimable"
        wins, n = self.seed["wins"], self.seed["n_seeds"]
        return {
            "contrast": self.contrast,
            "dataset": self.dataset,
            "arm_a": self.condition_a,
            "arm_b": self.condition_b,
            "n_instances": self.n_instances,
            "n_seeds": self.n_seeds,
            "observed_f1_a": self.bootstrap["observed_f1_a"],
            "observed_f1_b": self.bootstrap["observed_f1_b"],
            "observed_delta": self.bootstrap["observed_delta"],
            "bootstrap_ci_low": self.bootstrap["ci_low"],
            "bootstrap_ci_high": self.bootstrap["ci_high"],
            "seed_source": self.seed_source,
            "seed_mean_delta": (
                blank(self.seed["mean_delta"]) if has_seed_test else ""
            ),
            "seed_sd": blank(self.seed["sd_delta"]),
            "seed_t": blank(self.seed["t"]),
            "seed_p": blank(self.seed["p"]),
            "seed_ci_low": blank(self.seed["ci_low"]),
            "seed_ci_high": blank(self.seed["ci_high"]),
            "seed_wins": "" if (wins is None or not has_seed_test) else f"{wins}/{n}",
            "seed_recomputed_mean_delta": blank(self.seed_recomputed.get("mean_delta")),
            "seed_recomputed_p": blank(self.seed_recomputed.get("p")),
            "parity_ok": self.parity_ok,
            "parity_mode": self.parity_mode,
            "parity_checked": self.parity_checked,
        }

    def to_json(self) -> str:
        """Serialise the full record, including raw per-seed TP/FP/FN.

        Strict JSON: ``seed_level_stats`` never emits ``inf``/``nan``, so no
        ``Infinity`` token can reach the file and a strict parser can read it.
        """
        payload = asdict(self)
        payload["seeds"] = list(self.seeds)
        payload["per_seed"] = {str(k): v for k, v in self.per_seed.items()}
        return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"

    @classmethod
    def from_json(cls, text: str) -> ContrastResult:
        """Rebuild a record written by :meth:`to_json`.

        The per-contrast JSON is the only durable copy of a contrast that is not
        being recomputed, so a selective re-run re-renders the rest of the record
        from these files instead of dropping them. Round-tripping must be exact:
        ``seeds`` is restored as a tuple and ``per_seed`` is re-keyed by int,
        because both are what the renderers index with -- a string key would make
        a restored row render empty rather than fail.
        """
        payload = json.loads(text)
        unknown = set(payload) - {f.name for f in fields(cls)}
        if unknown:
            msg = f"unknown ContrastResult fields: {sorted(unknown)}"
            raise ValueError(msg)
        payload["seeds"] = tuple(int(s) for s in payload["seeds"])
        payload["per_seed"] = {int(k): v for k, v in payload["per_seed"].items()}
        return cls(**payload)


def _stack(
    arm: ArmPredictions, instance_ids: Sequence[str], negative_label: str
) -> np.ndarray:
    return np.stack(
        [
            instance_outcomes(
                arm.records_by_seed[seed], instance_ids, negative_label=negative_label
            )
            for seed in arm.seeds
        ]
    )


def _recorded(
    override: Mapping[int, float] | None, arm: ArmPredictions, seed: int
) -> float | None:
    if override is not None:
        if seed not in override:
            msg = f"recorded value for seed {seed} missing from spec ({arm.condition})"
            raise ValueError(msg)
        return float(override[seed])
    return arm.recorded_test_f1[seed]


def run_contrast(
    spec: ContrastSpec,
    *,
    n_resamples: int = DEFAULT_RESAMPLES,
    rng_seed: int = DEFAULT_RNG_SEED,
    strict_parity: bool = True,
    parity_tol: float = DEFAULT_PARITY_TOL,
) -> ContrastResult:
    """Load both arms, check parity, and compute seed- and instance-level stats.

    Parity means: identical instance ids and gold labels across every seed of
    both arms, and the F1 recomputed from each prediction file equal (within
    ``parity_tol``) to the test F1 the campaign recorded -- ``spec.recorded_*``
    (copied from the published aggregate) when given, else ``cell.json``.  With
    ``strict_parity`` a break raises; otherwise it is recorded in ``parity_ok``
    and ``parity_mode`` so the row can be shown but not cited.

    **The confirmatory seed-level statistic is computed from the RECORDED
    per-seed test F1** -- the values the paper prints -- not from the F1
    recomputed here. The recomputed value drifts from the trained model's own
    test pass by up to ``parity_tol`` (bf16-mixed inference is not bit-
    reproducible across processes), which is enough to flip a near-tie seed:
    on ``ddi_headline_evwarm_minus_base`` the recorded values give t = 6.09 and
    10/10 wins while the recomputed ones give t = 5.72 and 9/10. The recomputed
    statistics are kept as ``seed_recomputed`` for transparency, and the
    recomputed outcomes -- not the recorded scalars -- are what the parity check
    and the instance bootstrap need, since only they have per-instance detail.
    """
    arm_a = load_arm(spec.arm_a[1], spec.dataset, spec.arm_a[0], seeds=spec.seeds)
    arm_b = load_arm(spec.arm_b[1], spec.dataset, spec.arm_b[0], seeds=spec.seeds)
    groups = [arm_a.records_by_seed[s] for s in spec.seeds] + [
        arm_b.records_by_seed[s] for s in spec.seeds
    ]
    instance_ids = align_instance_ids(groups)
    stack_a = _stack(arm_a, instance_ids, spec.negative_label)
    stack_b = _stack(arm_b, instance_ids, spec.negative_label)
    f1_a = f1_from_counts(*stack_a.sum(axis=1).T)
    f1_b = f1_from_counts(*stack_b.sum(axis=1).T)

    per_seed: dict[int, dict[str, Any]] = {}
    parity_ok = True
    parity_checked = 0
    breaks: list[str] = []
    missing_recorded: list[str] = []
    recorded_a: list[float] = []
    recorded_b: list[float] = []
    for k, seed in enumerate(spec.seeds):
        tp_a, fp_a, fn_a = (int(v) for v in stack_a[k].sum(axis=0))
        tp_b, fp_b, fn_b = (int(v) for v in stack_b[k].sum(axis=0))
        rec_a = _recorded(spec.recorded_a, arm_a, seed)
        rec_b = _recorded(spec.recorded_b, arm_b, seed)
        for label, got, recorded, sink in (
            ("a", f1_a[k], rec_a, recorded_a),
            ("b", f1_b[k], rec_b, recorded_b),
        ):
            if recorded is None:
                missing_recorded.append(f"{label}/seed_{seed}")
                continue
            sink.append(recorded)
            parity_checked += 1
            if abs(float(got) - recorded) > parity_tol:
                parity_ok = False
                breaks.append(
                    f"seed {seed} arm {label}: recomputed {float(got):.6f} "
                    f"vs recorded {recorded:.6f}"
                )
        per_seed[seed] = {
            "f1_a": float(f1_a[k]),
            "f1_b": float(f1_b[k]),
            "recorded_f1_a": rec_a,
            "recorded_f1_b": rec_b,
            "tp_a": tp_a,
            "fp_a": fp_a,
            "fn_a": fn_a,
            "tp_b": tp_b,
            "fp_b": fp_b,
            "fn_b": fn_b,
            "sha256_a": arm_a.sha256_by_seed[seed],
            "sha256_b": arm_b.sha256_by_seed[seed],
            "path_a": str(arm_a.prediction_paths[seed]),
            "path_b": str(arm_b.prediction_paths[seed]),
        }
    if breaks and strict_parity:
        msg = f"parity check failed for {spec.name}: " + "; ".join(breaks)
        raise ValueError(msg)

    seed_recomputed = seed_level_stats(f1_a, f1_b)
    if missing_recorded:
        seed_block = unavailable_seed_stats(missing_recorded)
        seed_source = "unavailable"
    else:
        seed_block = seed_level_stats(np.array(recorded_a), np.array(recorded_b))
        # `not_estimable` is NOT a flavour of `recorded`: with fewer than two
        # training seeds there is no seed-level test to have a result. Labelling
        # such a block `recorded` put a single greedy inference run under
        # "CONFIRMATORY, the paper's statistic" (#314 review F2). Downstream, a
        # blank seed_source is what keeps the TSV's seed fields blank rather than
        # asserting a statistic nobody computed.
        #
        # Keyed on the SEED COUNT, not on `estimable`. `estimable` is false for
        # two different reasons -- n < 2, and n >= 2 with a constant delta, where
        # t is undefined but a real multi-seed result exists. Conflating them
        # labelled a 10-seed contrast in which every seed moved identically (the
        # strongest possible consistent result) "not estimable" and suppressed its
        # DISAGREE line, which is the same over-claiming failure F1 closed,
        # through a rarer door.
        seed_source = (
            "recorded"
            if len(spec.seeds) >= _MIN_SEEDS_FOR_A_TEST
            else ("not_estimable")
        )

    if not strict_parity:
        parity_mode = "unchecked"
    elif missing_recorded:
        parity_mode = "vacuous"
    else:
        parity_mode = "checked"

    ids_digest = hashlib.sha256("\n".join(instance_ids).encode()).hexdigest()
    return ContrastResult(
        contrast=spec.name,
        dataset=spec.dataset,
        negative_label=spec.negative_label,
        condition_a=spec.arm_a[0],
        condition_b=spec.arm_b[0],
        root_a=str(spec.arm_a[1]),
        root_b=str(spec.arm_b[1]),
        seeds=tuple(spec.seeds),
        n_instances=len(instance_ids),
        n_seeds=len(spec.seeds),
        seed=seed_block,
        bootstrap=paired_instance_bootstrap(
            stack_a, stack_b, n_resamples=n_resamples, rng_seed=rng_seed
        ),
        per_seed=per_seed,
        parity_ok=parity_ok,
        parity_tol=parity_tol,
        seed_source=seed_source,
        seed_recomputed=seed_recomputed,
        parity_mode=parity_mode,
        parity_checked=parity_checked,
        note=spec.note,
        instance_ids_sha256=ids_digest,
        extra={"parity_breaks": breaks, "missing_recorded": missing_recorded},
    )
