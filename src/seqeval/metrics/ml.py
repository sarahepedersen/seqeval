"""ML/probability metrics for backtesting (04) — thin composition over the replicate engine (02b).

This module contains **no** probability-estimation statistics of its own: point estimates,
intervals, MC-error corrections, null bands, and bootstraps all come from
:mod:`seqeval.core.replicates`. Here we (a) run the probability pipeline (evaluator output ->
run-level probability table), (b) join model probabilities to observed truth, and (c) score them
(calibration/ECE, tie-corrected ROC-AUC, raw+corrected Brier, log-loss, timing coverage) plus the
generic ``aggregate_error`` comparator for any 03 metric table.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Literal

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata
from sklearn.metrics import roc_auc_score

from seqeval.core import replicates as rep
from seqeval.core.specs import ReplicateSpec
from seqeval.io.schema import RUN_KEYS
from seqeval.metrics._disclosure import MIN_CELL, suppress_small_cells
from seqeval.units import years_to_days

logger = logging.getLogger("seqeval")

__all__ = [
    "probability_table",
    "join_truth",
    "calibration_table",
    "ece",
    "roc_auc",
    "brier",
    "log_loss",
    "score_cis",
    "timing_coverage",
    "timing_error_distribution",
    "subgroup_rates",
    "aggregate_error",
]


# =================================================================================================
# 1.1 probability pipeline
# =================================================================================================
def probability_table(
    gen_eval: pd.DataFrame, spec: ReplicateSpec, *, run_keys: list[str] = RUN_KEYS
) -> pd.DataFrame:
    """Replicate-level evaluator output -> run-level probability table (via 02b).

    ``[*run_keys, k, n, p_hat, logit_emp, var_logit, ci_lo, ci_hi]``. Warns when the median
    replicate count falls below ``spec.min_replicates`` (the probability grid is then coarser than
    ``1/min_replicates``).
    """
    summary = rep.replicate_summary(gen_eval, run_keys=run_keys)
    if len(summary):
        median_n = float(summary["n"].median())
        if median_n < spec.min_replicates:
            logger.warning(
                "probability_table: median replicate count %.0f < min_replicates %d — probability "
                "estimates are coarse; consider generating more seeds",
                median_n,
                spec.min_replicates,
            )
    return rep.estimate_probability(summary, spec=spec)


# =================================================================================================
# 1.2 probability metrics vs truth
# =================================================================================================
def join_truth(probs: pd.DataFrame, obs_outcomes: pd.DataFrame) -> pd.DataFrame:
    """Inner-join run probabilities to observed truth on ``person_id`` (both sides evaluable).

    ``obs_outcomes`` is the *same* evaluator's output on the observed data (same spec, same
    jumpoff). Returns ``[*probs cols, y_true]`` for persons evaluable on both sides.
    """
    truth = obs_outcomes.loc[obs_outcomes["evaluable"], ["person_id", "occurred"]].rename(
        columns={"occurred": "y_true"}
    )
    truth["y_true"] = truth["y_true"].astype(int)
    return probs.merge(truth, on="person_id", how="inner")


def _bin_edges(p: np.ndarray, n_bins: int, strategy: str) -> np.ndarray:
    """Bin edges over ``[0, 1]``; quantile edges collapse where ``p̂`` ties.

    ``p̂ = k/n`` lives on a grid of ``n + 1`` points, so a quantile cut point can never split an
    atom: where a single grid value carries more than one bin's worth of mass, the edges either
    side of it coincide and ``np.unique`` drops the duplicate. The realized bin count is therefore
    at most ``min(n_bins, distinct p̂ values)`` — few seeds mean few bins, whatever is requested.
    """
    if strategy == "uniform":
        return np.linspace(0.0, 1.0, n_bins + 1)
    if strategy != "quantile":
        raise ValueError(f"unknown strategy {strategy!r}; use uniform | quantile")
    edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 2:
        # Every p̂ identical — one atom, so one bin spanning the whole probability range. Without
        # this the two assignments below would both write to the same element and leave a
        # single-element array, which yields no bins at all and a silently empty table.
        return np.array([0.0, 1.0])
    edges[0], edges[-1] = 0.0, 1.0
    return edges


def calibration_table(
    joined: pd.DataFrame,
    *,
    n_bins: int = 10,
    strategy: Literal["uniform", "quantile"] = "quantile",
) -> pd.DataFrame:
    """Reliability table binned by ``p_hat``: ``[bin, ..., p_mean, y_rate, n, n_persons]``.

    Pair with :func:`seqeval.core.replicates.null_calibration_band` (02b) downstream so
    miscalibration is only claimed where the curve exits the perfect-calibration envelope.
    """
    p = joined["p_hat"].to_numpy()
    y = joined["y_true"].to_numpy()
    pid = joined["person_id"].to_numpy() if "person_id" in joined.columns else None
    edges = _bin_edges(p, n_bins, strategy)
    idx = np.clip(np.digitize(p, edges) - 1, 0, len(edges) - 2)
    rows = []
    for b in range(len(edges) - 1):
        sel = idx == b
        if not sel.any():
            continue
        rows.append(
            {
                "bin": b,
                "bin_left": edges[b],
                "bin_right": edges[b + 1],
                "p_mean": float(p[sel].mean()),
                "y_rate": float(y[sel].mean()),
                "n": int(sel.sum()),
                "n_persons": int(sel.sum()) if pid is None else int(len(np.unique(pid[sel]))),
            }
        )
    return pd.DataFrame(rows)


def p_hat_distribution(joined: pd.DataFrame, *, min_cell: int = MIN_CELL) -> pd.DataFrame:
    """How many people sit on each value of ``p̂``: ``[p_hat, n_persons, n_total, suppressed]``.

    ``p̂ = k/n`` is atomic — with ``n`` replicates it can only take ``n + 1`` values — so its
    distribution is a set of spikes on that grid, not a density. Calibration *bins* are a different
    object: they are chosen to hold roughly equal numbers of people, so a single bin can span a
    wide stretch of the axis while nearly all of its mass sits on one atom at the edge. Reporting
    the bin counts as if they were the distribution puts the weight in the wrong place.

    Every attainable grid point gets a row, including the empty ones, so a gap in the distribution
    is visible as a true zero rather than as a value that was never possible. Cells are withheld per
    :func:`~seqeval.metrics._disclosure.suppress_small_cells`.
    """
    p = joined["p_hat"].to_numpy()
    if not len(p):
        return pd.DataFrame(
            {c: pd.Series(dtype="float64") for c in ("p_hat", "n_persons", "n_total", "suppressed")}
        )
    counts = pd.Series(p).value_counts()
    # The grid is set by the replicate count, so a value nobody landed on is still a real zero.
    n_rep = int(pd.Series(joined["n"]).median()) if "n" in joined.columns else 0
    grid = np.round(np.arange(n_rep + 1) / n_rep, 12) if n_rep > 0 else np.sort(counts.index)
    cells = pd.DataFrame({"p_hat": grid})
    cells["n_persons"] = (
        cells["p_hat"].map(lambda v: int(counts.get(v, 0))).astype(np.int64)
    )
    cells["n_total"] = int(len(p))
    return suppress_small_cells(cells, count_col="n_persons", by=[], min_cell=min_cell)


def ece(calibration: pd.DataFrame) -> float:
    """Expected calibration error: population-weighted mean ``|p_mean - y_rate|`` over bins."""
    if not len(calibration):
        return float("nan")
    weights = calibration["n"] / calibration["n"].sum()
    return float((weights * (calibration["p_mean"] - calibration["y_rate"]).abs()).sum())


def roc_auc(joined: pd.DataFrame) -> float:
    """Rank-based, tie-corrected ROC-AUC (``p_hat`` lives on a coarse ``1/n`` grid; see 02b).

    ``NaN`` when only one class is present. Record the grid resolution ``1/median_n`` alongside
    (the arm does).
    """
    y = joined["y_true"].to_numpy()
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, joined["p_hat"].to_numpy()))


def brier(joined: pd.DataFrame) -> dict[str, float]:
    """``{"raw", "corrected"}`` — corrected via :func:`replicates.brier_noise_correction` (02b)."""
    p = joined["p_hat"].to_numpy()
    y = joined["y_true"].to_numpy()
    raw = float(np.mean((p - y) ** 2))
    correction = rep.brier_noise_correction(joined[["k", "n"]])
    return {"raw": raw, "corrected": raw - correction}


def mse(joined: pd.DataFrame) -> float:
    """Mean squared error of the raw replicate rate ``k/n`` against the observed outcome.

    Computed from the counts alone, with no probability machinery in the path. Since ``p_hat`` is
    itself ``k/n``, this equals ``brier["raw"]``; the two differ only in that :func:`brier` also
    reports a finite-seed-corrected value. Kept as the plain, self-contained form and as the
    numerator :func:`r2` rescales.
    """
    rate = joined["k"].to_numpy() / joined["n"].to_numpy()
    y = joined["y_true"].to_numpy()
    return float(np.mean((rate - y) ** 2))


def r2(joined: pd.DataFrame) -> float:
    """Coefficient of determination of the raw rate ``k/n`` against the observed outcome.

    ``R² = 1 − Σ(y − k/n)² / Σ(y − ȳ)²`` — MSE rescaled by the outcome's own variance. 1 = perfect,
    0 = no better than predicting the base rate, negative = worse than the base rate. Uses the same
    unsmoothed rate as :func:`mse`. ``NaN`` when the outcome has no variance (``ȳ`` is 0 or 1).
    """
    rate = joined["k"].to_numpy() / joined["n"].to_numpy()
    y = joined["y_true"].to_numpy().astype(float)
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    if ss_tot == 0:
        return float("nan")
    ss_res = float(np.sum((y - rate) ** 2))
    return 1.0 - ss_res / ss_tot


def score_cis(joined: pd.DataFrame, *, level: float = 0.95) -> pd.DataFrame:
    """Analytic ``[metric, ci_lo, ci_hi]`` for the backtest scores, from per-person quantities.

    Every interval is ``estimate ± z·se``. Persons are the sampling unit:

    - ``mse``/``brier_raw`` are means over persons of the loss ``l_i = (p̂_i − y_i)²``, so the
      standard error is ``sd_i(l_i)/√n``.
    - ``brier_corrected`` subtracts a mean of per-run terms (:func:`~seqeval.core.replicates.
      brier_noise_correction`), so it is the mean of ``l_i − c_i`` and takes that quantity's sd.
    - ``r2`` is a ratio of two means; :func:`_ratio_se` gives its delta-method standard error.
    - ``roc_auc`` is a rank statistic, so its variance comes from DeLong's placement values
      (:func:`_delong_var`), and the interval is clipped to ``[0, 1]``.
    - ``ece`` gets **no** interval. Its bins are chosen from the data under quantile binning and the
      statistic is biased upward, so there is no honest closed form to report; it merges to NaN.

    These are sampling intervals over persons that *already carry* replicate noise: each ``l_i``
    is computed from that person's own ``p̂_i``, so the spread across persons contains the seed
    uncertainty the same way ``var_i(mu_i)`` does for CCF. They are the ``total_var`` analogue, not
    the replicate-only one. No decomposition is reported because ``p̂`` is defined *across* seeds —
    there is no per-seed loss to average.
    """
    p = joined["p_hat"].to_numpy().astype(float)
    y = joined["y_true"].to_numpy().astype(float)
    n = len(y)
    z = norm.ppf(1 - (1 - level) / 2)
    rows: list[dict] = []
    if n < 2:
        return pd.DataFrame(columns=["metric", "ci_lo", "ci_hi"])

    loss = (p - y) ** 2
    raw, se_raw = float(loss.mean()), float(loss.std(ddof=1) / np.sqrt(n))
    rows.append({"metric": "mse", "value": raw, "se": se_raw})
    rows.append({"metric": "brier_raw", "value": raw, "se": se_raw})

    k, n_rep = joined["k"].to_numpy().astype(float), joined["n"].to_numpy().astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        inflation = np.where(n_rep >= 2, p * (1 - p) / (n_rep - 1), 0.0)
    corrected = loss - inflation
    rows.append(
        {
            "metric": "brier_corrected",
            "value": float(corrected.mean()),
            "se": float(corrected.std(ddof=1) / np.sqrt(n)),
        }
    )

    resid = (y - k / n_rep) ** 2
    spread = (y - y.mean()) ** 2
    if spread.sum() > 0:
        ratio, se_ratio = _ratio_se(resid, spread)
        rows.append({"metric": "r2", "value": 1.0 - ratio, "se": se_ratio})

    if len(np.unique(y)) == 2:
        auc = float(roc_auc_score(y, p))
        var = _delong_var(y, p)
        if np.isfinite(var) and var > 0:
            rows.append({"metric": "roc_auc", "value": auc, "se": float(np.sqrt(var))})

    out = pd.DataFrame(rows)
    out["ci_lo"] = out["value"] - z * out["se"]
    out["ci_hi"] = out["value"] + z * out["se"]
    auc_rows = out["metric"] == "roc_auc"
    out.loc[auc_rows, ["ci_lo", "ci_hi"]] = out.loc[auc_rows, ["ci_lo", "ci_hi"]].clip(0.0, 1.0)
    return out[["metric", "ci_lo", "ci_hi"]]


def _ratio_se(num: np.ndarray, den: np.ndarray) -> tuple[float, float]:
    """``(A/B, se)`` for a ratio of two per-person means, by the delta method.

    The influence function of ``A/B`` at person ``i`` is ``(a_i − (A/B)·b_i)/B``, so the standard
    error is that quantity's sd over ``√n``. The plug-in mean inside ``b_i`` contributes at
    ``O(1/n)`` and is not corrected for.
    """
    n = len(num)
    a, b = float(num.mean()), float(den.mean())
    ratio = a / b
    infl = (num - ratio * den) / b
    return ratio, float(infl.std(ddof=1) / np.sqrt(n))


def _delong_var(y: np.ndarray, p: np.ndarray) -> float:
    """DeLong variance of the ROC-AUC, computed from midranks so ties count half.

    ``var = S10/n₊ + S01/n₋`` where ``S10``/``S01`` are the variances of the per-person placement
    values — the share of the opposite class each person is ranked above. Midranks
    (:func:`scipy.stats.rankdata`) are what make this exact on the coarse ``1/n`` grid ``p_hat``
    lives on, where ties are the rule rather than the exception (see :data:`AUC_TIE_NOTE`).
    """
    pos, neg = p[y == 1], p[y == 0]
    n_pos, n_neg = len(pos), len(neg)
    if n_pos < 2 or n_neg < 2:
        return float("nan")
    r_all = rankdata(np.concatenate([pos, neg]))
    v10 = (r_all[:n_pos] - rankdata(pos)) / n_neg  # placement of each positive among negatives
    v01 = 1.0 - (r_all[n_pos:] - rankdata(neg)) / n_pos
    return float(v10.var(ddof=1) / n_pos + v01.var(ddof=1) / n_neg)


def log_loss(joined: pd.DataFrame, *, eps: float = 1e-12) -> float:
    """Binary log-loss on ``p_hat``, clipped to ``[eps, 1-eps]``.

    ``p_hat`` is the unsmoothed ``k/n``, so runs at 0 or 1 that go the other way are pinned at
    ``-ln(eps)`` rather than infinite. The clip is what keeps the score finite; read a log-loss
    dominated by boundary runs as "too few replicates", not as a calibration result.
    """
    p = np.clip(joined["p_hat"].to_numpy(), eps, 1 - eps)
    y = joined["y_true"].to_numpy()
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def timing_coverage(
    timing_dist: pd.DataFrame, obs_tte: pd.DataFrame, *, q: tuple[float, float] = (0.10, 0.90)
) -> float:
    """Predictive-interval coverage: fraction of observed durations inside ``(q_lo, q_hi)`` (02b).

    Restricted to persons whose observed event occurred (their duration is known); the timing
    calibration headline (well-calibrated timing => coverage ~ q_hi - q_lo).
    """
    lo, hi = f"q{int(q[0] * 100)}", f"q{int(q[1] * 100)}"
    seen = obs_tte.loc[obs_tte["observed"], ["person_id", "duration"]]
    m = timing_dist.merge(seen, on="person_id", how="inner")
    if not len(m):
        return float("nan")
    inside = (m["duration"] >= m[lo]) & (m["duration"] <= m[hi])
    return float(inside.mean())


def timing_pairs(
    tte_gen: pd.DataFrame,
    obs_tte: pd.DataFrame,
    *,
    horizon_days: int | None,
    persons: Iterable | None = None,
    seed_col: str = "seed",
) -> pd.DataFrame:
    """One row per generated **trajectory**: ``[person_id, seed, pred, obs, predicted]`` in days.

    Each seed is its own synthetic population, so a person contributes K rows rather than one
    summary of their replicates — the timing error is a property of a trajectory, not of a person.

    Two rules decide which trajectories are comparable at all:

    - the person's event must have been **observed**, and observed *inside* the frame horizon;
      without a truth there is nothing to difference against;
    - membership in ``persons``, the arm's scored population (condition minus settled-at-jump-off);
      a settled person's event sits in the replayed observed prefix and would land exactly on the
      truth by construction.

    Every surviving trajectory is a candidate. ``predicted`` marks the ones where the model actually
    produced the outcome inside the horizon; the rest have no predicted time, so ``pred`` is NaN and
    :func:`timing_error_distribution` counts them as excluded rather than capping them at the
    horizon — a cap would pile invented mass at the frame edge and read as a confident late call.
    """
    seen = obs_tte.loc[obs_tte["observed"], ["person_id", "duration"]].rename(
        columns={"duration": "obs"}
    )
    m = tte_gen.merge(seen, on="person_id", how="inner")
    if persons is not None:
        m = m[m["person_id"].isin(set(persons))]
    if horizon_days is not None:
        m = m[m["obs"] <= horizon_days]

    predicted = m["observed"].to_numpy()
    if horizon_days is not None:
        predicted = predicted & (m["duration"].to_numpy() <= horizon_days)
    out = pd.DataFrame(
        {
            "person_id": m["person_id"].to_numpy(),
            "pred": np.where(predicted, m["duration"].to_numpy(), np.nan),
            "obs": m["obs"].to_numpy().astype(np.int64),
            "predicted": predicted,
        }
    )
    if seed_col in m.columns:
        out.insert(1, seed_col, m[seed_col].to_numpy())
    return out


_TIMING_ERROR_COLUMNS = [
    "pred_bin", "pred_lo", "pred_hi", "pred_median", "n_pred_bin",
    "error_lo", "error_hi", "n_persons", "suppressed", "n_trajectories", "n_excluded",
]


def _timing_cells(pairs: pd.DataFrame, *, error_bin_years: float, pred_bin_years: float) -> list:
    """Cell rows for one already-scoped set of trajectories (no suppression yet)."""
    n_trajectories = len(pairs)
    kept = pairs[pairs["predicted"].astype(bool)]
    n_excluded = n_trajectories - len(kept)
    if kept.empty:
        return []

    width = years_to_days(error_bin_years)
    pred = kept["pred"].to_numpy().astype(np.int64)
    error = kept["obs"].to_numpy().astype(np.int64) - pred

    pred_width = years_to_days(pred_bin_years)
    first = int(np.floor(pred.min() / pred_width))
    last = int(np.floor(pred.max() / pred_width))
    pred_edges = (np.arange(first, last + 2) * pred_width).astype(float)
    pred_idx = np.clip(np.digitize(pred, pred_edges) - 1, 0, len(pred_edges) - 2)

    lo = np.floor(error.min() / width) * width
    hi = np.ceil(error.max() / width) * width + width
    error_edges = np.arange(lo, hi + width / 2, width)
    error_idx = np.clip(np.digitize(error, error_edges) - 1, 0, len(error_edges) - 2)

    rows = []
    for b in range(len(pred_edges) - 1):
        in_bin = pred_idx == b
        if not in_bin.any():
            continue
        counts = np.bincount(error_idx[in_bin], minlength=len(error_edges) - 1)
        for e, n in enumerate(counts):
            rows.append(
                {
                    "pred_bin": first + b,
                    "pred_lo": float(pred_edges[b]),
                    "pred_hi": float(pred_edges[b + 1]),
                    "pred_median": float(np.median(pred[in_bin])),
                    "n_pred_bin": int(in_bin.sum()),
                    "error_lo": float(error_edges[e]),
                    "error_hi": float(error_edges[e + 1]),
                    "n_persons": int(n),
                    "n_trajectories": n_trajectories,
                    "n_excluded": n_excluded,
                }
            )
    return rows


def timing_error_distribution(
    pairs: pd.DataFrame,
    *,
    by: list[str] = (),
    error_bin_years: float = 1.0,
    pred_bin_years: float = 2.0,
    min_cell: int = MIN_CELL,
) -> pd.DataFrame:
    """Binned distribution of timing error, by how early or late the model predicted.

    ``pairs`` is :func:`timing_pairs` — one row per generated trajectory. Returns ``[*by, pred_bin,
    pred_lo, pred_hi, pred_median, n_pred_bin, error_lo, error_hi, n_persons, suppressed,
    n_trajectories, n_excluded]``: a count per (predicted-value bin × signed-error bin), day-valued,
    with no person identifier anywhere. The error is ``observed - predicted``: **positive means the
    event happened later than predicted**, so mass to the right of zero is a model that predicts too
    early.

    ``by=["seed"]`` bins each synthetic population separately; ``by=()`` pools every trajectory into
    one distribution. Both anchor their bins the same way, so the two tables line up cell for cell.

    ``n_trajectories`` and ``n_excluded`` ride on every row: the candidates the group started with,
    and how many of them the model never brought to the outcome inside the frame. The figure states
    the exclusion, because a distribution of timing error among *predicted* events says nothing
    about the events that were never predicted.

    Predicted bins are fixed ``pred_bin_years``-wide intervals anchored at a multiple of the width,
    so the same predicted-age range is the same bin in every figure and jump-offs can be read
    against each other. A later jump-off simply has fewer bins — the ones its people no longer
    reach. Bins nobody lands in are dropped rather than drawn empty, and ``pred_bin`` is the global
    index of the interval, so it stays comparable across tables. Error bins are shared across all
    rows and likewise anchored, which puts **zero on a bin edge** — no cell can mix early with late.
    Cells are then suppressed per :func:`~seqeval.metrics._disclosure.suppress_small_cells`.
    """
    by = list(by)
    empty = pd.DataFrame({c: pd.Series(dtype="float64") for c in _TIMING_ERROR_COLUMNS})
    if pairs.empty:
        return empty if not by else empty.assign(**{c: pd.Series(dtype="object") for c in by})

    groups = [((), pairs)] if not by else list(pairs.groupby(by, observed=True))
    blocks = []
    for key, grp in groups:
        rows = _timing_cells(
            grp, error_bin_years=error_bin_years, pred_bin_years=pred_bin_years
        )
        if not rows:
            continue
        block = pd.DataFrame(rows)
        key_tuple = key if isinstance(key, tuple) else (key,)
        for col, val in zip(by, key_tuple, strict=True):
            block.insert(0, col, val)
        blocks.append(block)
    if not blocks:
        return empty if not by else empty.assign(**{c: pd.Series(dtype="object") for c in by})

    cells = pd.concat(blocks, ignore_index=True)
    suppressed = suppress_small_cells(
        cells, count_col="n_persons", by=[*by, "pred_bin"], min_cell=min_cell
    )
    return suppressed[[*by, *_TIMING_ERROR_COLUMNS]]


def subgroup_rates(
    gen_outcomes: pd.DataFrame, obs_outcomes: pd.DataFrame, *, by: list[str]
) -> pd.DataFrame:
    """Population-level predicted vs observed event rates per subgroup — no seed replication needed.

    Both frames must already carry the ``by`` columns (the arm merges persons covariates/cohort).
    Predicted rate pools over all evaluable (person, seed) generated rows; observed rate over
    evaluable observed rows. Returns ``[*by, pred_rate, obs_rate, n_pred, n_obs]``.
    """
    by = list(by)
    g = gen_outcomes[gen_outcomes["evaluable"]] if "evaluable" in gen_outcomes else gen_outcomes
    o = obs_outcomes[obs_outcomes["evaluable"]] if "evaluable" in obs_outcomes else obs_outcomes
    gr = g.groupby(by, observed=True)["occurred"].agg(pred_rate="mean", n_pred="size")
    orr = o.groupby(by, observed=True)["occurred"].agg(obs_rate="mean", n_obs="size")
    out = gr.join(orr, how="outer").reset_index()
    return out.sort_values(by).reset_index(drop=True)


# =================================================================================================
# 1.3 aggregate-metric error
# =================================================================================================
def aggregate_error(
    gen_metric: pd.DataFrame,
    obs_metric: pd.DataFrame,
    *,
    value_col: str,
    on: list[str],
    window_keys: tuple[str, ...] = ("age_start", "age_stop"),
) -> pd.DataFrame:
    """Generic comparator for any 03 metric table (CCF/ASFR/PPR/KM-at-times), gen vs observed.

    Aligns on ``on``, computes per-seed error, then per-window summary
    ``[*window_keys, *on, obs, gen_mean, gen_sd_over_seeds, bias, mae, rmse]``. Mismatched ``on``
    cells between the two sides raise (silent misalignment would corrupt every error).
    ``gen_sd_over_seeds`` is the seed-to-seed spread of the generated cell, which is the dispersion
    a reader needs here; no interval is placed on ``bias``. ``n_persons`` carries the observed
    side's distinct-person count for the cell when the metric table reports one.
    """
    on = list(on)
    wkeys = [c for c in window_keys if c in gen_metric.columns]

    obs_cols = [*on, value_col] + (["n_persons"] if "n_persons" in obs_metric.columns else [])
    merged = gen_metric.merge(
        obs_metric[obs_cols].rename(columns={value_col: "_obs", "n_persons": "_n_persons"}),
        on=on,
        how="outer",
        indicator=True,
        suffixes=("", "_obs"),
    )
    if (merged["_merge"] != "both").any():
        bad = merged.loc[merged["_merge"] != "both", on].drop_duplicates().to_dict("records")
        raise ValueError(
            f"aggregate_error: generated and observed metric cells do not align on {on}; "
            f"offending cells: {bad[:10]}"
        )
    merged["error"] = merged[value_col] - merged["_obs"]

    grouped = merged.groupby([*wkeys, *on], observed=True)
    aggs = {"n_persons": ("_n_persons", "first")} if "_n_persons" in merged.columns else {}
    out = grouped.agg(
        obs=("_obs", "first"),
        gen_mean=(value_col, "mean"),
        gen_sd_over_seeds=(value_col, "std"),
        bias=("error", "mean"),
        mae=("error", lambda e: e.abs().mean()),
        rmse=("error", lambda e: float(np.sqrt(np.mean(e**2)))),
        **aggs,
    ).reset_index()

    return out.sort_values([*wkeys, *on]).reset_index(drop=True)
