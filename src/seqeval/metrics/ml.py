"""ML/probability metrics for backtesting (04) — thin composition over the replicate engine (02b).

This module contains **no** probability-estimation statistics of its own: point estimators,
intervals, MC-error corrections, null bands, and bootstraps all come from
:mod:`seqeval.core.replicates`. Here we (a) run the probability pipeline (evaluator output ->
run-level probability table), (b) join model probabilities to observed truth, and (c) score them
(calibration/ECE, tie-corrected ROC-AUC, raw+corrected Brier, log-loss, timing coverage) plus the
generic ``aggregate_error`` comparator for any 03 metric table.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from seqeval.core import replicates as rep
from seqeval.core.specs import ReplicateSpec
from seqeval.io.schema import RUN_KEYS

logger = logging.getLogger("seqeval")

__all__ = [
    "probability_table",
    "join_truth",
    "calibration_table",
    "ece",
    "roc_auc",
    "brier",
    "log_loss",
    "timing_coverage",
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
    if strategy == "quantile":
        edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
        edges[0], edges[-1] = 0.0, 1.0
        return edges
    if strategy == "uniform":
        return np.linspace(0.0, 1.0, n_bins + 1)
    raise ValueError(f"unknown strategy {strategy!r}; use uniform | quantile")


def calibration_table(
    joined: pd.DataFrame,
    *,
    n_bins: int = 10,
    strategy: Literal["uniform", "quantile"] = "quantile",
) -> pd.DataFrame:
    """Reliability table binned by ``p_hat``: ``[bin, bin_left, bin_right, p_mean, y_rate, n]``.

    Pair with :func:`seqeval.core.replicates.null_calibration_band` (02b) downstream so
    miscalibration is only claimed where the curve exits the perfect-calibration envelope.
    """
    p = joined["p_hat"].to_numpy()
    y = joined["y_true"].to_numpy()
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
            }
        )
    return pd.DataFrame(rows)


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


def log_loss(joined: pd.DataFrame, *, eps: float = 1e-12) -> float:
    """Binary log-loss on the smoothed ``p_hat`` (defined everywhere; never on raw ``k/n``)."""
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
    over_seeds: str = "seed",
    window_keys: tuple[str, ...] = ("age_start", "age_stop"),
    spec: ReplicateSpec | None = None,
) -> pd.DataFrame:
    """Generic comparator for any 03 metric table (CCF/ASFR/PPR/KM-at-times), gen vs observed.

    Aligns on ``on``, computes per-seed error, then per-window summary
    ``[*window_keys, *on, obs, gen_mean, gen_sd_over_seeds, bias, mae, rmse]``. Mismatched ``on``
    cells between the two sides raise (silent misalignment would corrupt every error). When
    ``spec.bootstrap_n > 0`` percentile CIs on ``bias`` are added via
    :func:`replicates.seed_bootstrap`.
    """
    on = list(on)
    wkeys = [c for c in window_keys if c in gen_metric.columns]

    merged = gen_metric.merge(
        obs_metric[[*on, value_col]].rename(columns={value_col: "_obs"}),
        on=on,
        how="outer",
        indicator=True,
    )
    if (merged["_merge"] != "both").any():
        bad = merged.loc[merged["_merge"] != "both", on].drop_duplicates().to_dict("records")
        raise ValueError(
            f"aggregate_error: generated and observed metric cells do not align on {on}; "
            f"offending cells: {bad[:10]}"
        )
    merged["error"] = merged[value_col] - merged["_obs"]

    grouped = merged.groupby([*wkeys, *on], observed=True)
    out = grouped.agg(
        obs=("_obs", "first"),
        gen_mean=(value_col, "mean"),
        gen_sd_over_seeds=(value_col, "std"),
        bias=("error", "mean"),
        mae=("error", lambda e: e.abs().mean()),
        rmse=("error", lambda e: float(np.sqrt(np.mean(e**2)))),
    ).reset_index()

    if spec is not None and spec.bootstrap_n > 0 and over_seeds in gen_metric.columns:
        out = out.merge(
            _bootstrap_bias(gen_metric, obs_metric, value_col, on, over_seeds, wkeys, spec),
            on=[*wkeys, *on],
            how="left",
        )
    return out.sort_values([*wkeys, *on]).reset_index(drop=True)


def _bootstrap_bias(gen_metric, obs_metric, value_col, on, over_seeds, wkeys, spec) -> pd.DataFrame:
    """Seed-bootstrap percentile CIs on the per-cell bias (resample seeds, recompute bias)."""
    obs_lookup = obs_metric.set_index(on)[value_col]

    def stat_fn(df: pd.DataFrame) -> pd.DataFrame:
        agg = df.groupby([*wkeys, *on], observed=True)[value_col].mean().reset_index()
        agg["bias"] = agg[value_col] - agg.set_index(on).index.map(obs_lookup).to_numpy()
        return agg[[*wkeys, *on, "bias"]]

    rng = np.random.default_rng(spec.bootstrap_seed)
    boot = rep.seed_bootstrap(
        gen_metric,
        seed_col=over_seeds,
        stat_fn=stat_fn,
        n_boot=spec.bootstrap_n,
        rng=rng,
        value_cols=["bias"],
    )
    boot = boot[boot["metric"] == "bias"]
    return boot[[*wkeys, *on, "ci_lo", "ci_hi"]].rename(
        columns={"ci_lo": "bias_ci_lo", "ci_hi": "bias_ci_hi"}
    )
