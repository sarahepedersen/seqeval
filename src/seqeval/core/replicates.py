"""Replicate engine: empirical probabilities from seed stochasticity (02b core).

Models expose no logits, so **probabilities are recovered empirically from the distribution of
outcomes across replicate runs** — multiple ``seed`` values per (person, window). Everything
probabilistic downstream (calibration, ROC-AUC, Brier, timing calibration, replicate variance,
uncertainty bands) flows through this one engine, and the recovered probability is the event
probability *under the inference-time sampling procedure* (temperature, top-k, …) — the generative
system as actually used, not any internal softmax.

Estimator specification (Gart & Zweifel 1967; reproduced from 02b appendix)
---------------------------------------------------------------------------
For ``k ~ Binomial(n, p)``:

- **MLE** ``p̂ = k/n`` is unbiased but lives on a grid of width ``1/n`` and hits exact 0 and 1,
  where log-loss and logits are undefined.
- **Empirical logit (Haldane–Anscombe)** ``logit_emp = ln((k + ½)/(n − k + ½))`` — the additive-c
  family ``ln((k+c)/(n−k+c))`` has expectation ``logit(p) + (c − ½)(1/(np) − 1/(n(1−p))) + O(n⁻²)``,
  so **c = ½ is the unique first-order-bias-cancelling choice for all p**. Gart's variance
  ``var_logit = 1/(k+½) + 1/(n−k+½)`` is emitted as a required column (users regressing on empirical
  logits need it as weights, especially under ragged n).
- **Point estimate:** the MLE ``k/n``, unsmoothed. ``p_hat`` is the replicate frequency and
  nothing else, so a table entry can always be read straight back as "the outcome happened in
  ``k`` of ``n`` runs". The cost is the boundaries: ``p_hat`` reaches exactly 0 and 1, where
  logits are undefined and log-loss is only finite because it clips.
- **Intervals:** Jeffreys (``Beta(k+½, n−k+½)`` quantiles; default) or Wilson. These are interval
  methods, not smoothing — the point estimate they surround stays ``k/n``. The normal
  approximation is intentionally unsupported (poor at small n / extreme p).

``logit_emp`` is the one deliberately smoothed quantity: the Haldane–Anscombe correction is what
makes a log-odds finite at ``k = 0`` and ``k = n`` at all, so it is defined independently of
``p_hat`` and ``logit_emp != logit(p_hat)`` in general.

Estimation is strictly per run
------------------------------
Never pool, shrink, or borrow strength across runs or persons when estimating a run's probability —
the estimand is across-replicate variance for an *individual* sequence. All cross-run computation
(calibration binning, bootstraps) happens downstream on the per-run table, never inside estimation.

Dynamic range
-------------
``|logit_emp| ≤ ln(2n+1)``, so replicate data cannot express probabilities more extreme than about
``1/(2n+2)`` from the boundary:

===== ================ =====================
n     ``|logit| ≤``    approx p range
===== ================ =====================
5     2.40             ~[0.08, 0.92]
50    4.62             ~[0.01, 0.99]
200   6.00             ~[0.002, 0.998]
===== ================ =====================

Rare outcomes at small n saturate ``logit_emp`` (heavy shrinkage toward zero log-odds); a p_hat
that has not stabilized as replicates accumulate is how a researcher discovers they need more.

Informative-censoring guard
---------------------------
Spans always derive from the last age in the data (00 section 4.2). On models with stochastic
sequence length, event-sparse replicates can end early, get dropped as non-evaluable for late
frames, and bias ``k/n`` upward. The remedy is upstream (emit a trailing "no event" row at the
generation horizon t4, which carries the horizon into the span for free); detection is this
engine's job — :func:`replicate_summary` warns loudly when within-window ``n`` varies.

AUC on a coarse grid
--------------------
With ``p_hat`` on a ``1/n`` grid, ties are massive. Downstream AUC must be rank-based with tie
correction and record the grid resolution alongside — see :data:`AUC_TIE_NOTE`.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy.stats import beta, norm

from seqeval.core.specs import ReplicateSpec
from seqeval.io.schema import RUN_KEYS

logger = logging.getLogger("seqeval")

__all__ = [
    "replicate_summary",
    "estimate_probability",
    "timing_distribution",
    "count_distribution",
    "count_moments",
    "brier_noise_correction",
    "null_calibration_band",
    "seed_bootstrap",
    "AUC_TIE_NOTE",
]

AUC_TIE_NOTE = (
    "p_hat lies on a 1/n grid, so ties are massive; compute AUC with a rank-based, tie-corrected "
    "estimator (e.g. sklearn.metrics.roc_auc_score) and always record the grid resolution 1/n "
    "alongside the value."
)


# =================================================================================================
# 1. replicate summaries -> probability estimates
# =================================================================================================
def replicate_summary(
    outcome_table: pd.DataFrame, *, run_keys: list[str] = RUN_KEYS, seed_col: str = "seed"
) -> pd.DataFrame:
    """Collapse a per-replicate binary outcome table to ``[*run_keys, k, n]`` (one row per run).

    ``outcome_table`` is evaluator output (02 section 2.4). If it carries an ``evaluable`` column it
    is filtered to evaluable rows first. Ragged ``n`` across runs is legal (some replicates
    non-evaluable); runs that end up with ``n == 0`` simply vanish (counted for 04's coverage
    table). When ``n`` varies across runs *within a window* (``age_start``, ``age_stop``), a loud
    warning fires — the informative-censoring signature (see module docstring).
    """
    tbl = outcome_table
    if "evaluable" in tbl.columns:
        tbl = tbl[tbl["evaluable"]]
    grouped = tbl.groupby(run_keys, observed=True)
    summary = grouped["occurred"].agg(k="sum", n="size").reset_index()
    summary["k"] = summary["k"].astype(np.int64)
    summary["n"] = summary["n"].astype(np.int64)

    _warn_ragged_n(summary)
    return summary.sort_values(run_keys).reset_index(drop=True)


def _warn_ragged_n(summary: pd.DataFrame) -> None:
    """Warn when replicate count ``n`` varies across runs in a window (informative censoring)."""
    win = [c for c in ("age_start", "age_stop") if c in summary.columns]
    if not win:
        return
    varies = summary.groupby(win, observed=True)["n"].nunique()
    bad = varies[varies > 1]
    if len(bad):
        cells = [tuple(idx) if isinstance(idx, tuple) else (idx,) for idx in bad.index]
        logger.warning(
            "replicate_summary: within-window replicate count n varies for %d window cell(s) %s — "
            "possible informative censoring (event-sparse replicates ending early). Emit a "
            "trailing no-event row at the generation horizon so every replicate's span reaches it.",
            len(bad),
            cells,
        )


def estimate_probability(summary: pd.DataFrame, *, spec: ReplicateSpec) -> pd.DataFrame:
    """Add probability columns to a run summary (strictly per run — no pooling).

    Returns ``[*run_keys, k, n, p_hat, logit_emp, var_logit, ci_lo, ci_hi]``: ``p_hat`` is the
    unsmoothed MLE ``k/n``; ``logit_emp``/``var_logit`` are Haldane–Anscombe; the CI is per
    ``spec.interval`` at ``spec.level``.
    """
    run_keys = [c for c in summary.columns if c not in ("k", "n")]
    k = summary["k"].to_numpy().astype(np.float64)
    n = summary["n"].to_numpy().astype(np.float64)

    p_hat = k / n
    logit_emp = np.log((k + 0.5) / (n - k + 0.5))
    var_logit = 1.0 / (k + 0.5) + 1.0 / (n - k + 0.5)
    ci_lo, ci_hi = _interval(k, n, spec.interval, spec.level)

    out = summary[run_keys].copy()
    out["k"] = summary["k"].to_numpy()
    out["n"] = summary["n"].to_numpy()
    out["p_hat"] = p_hat
    out["logit_emp"] = logit_emp
    out["var_logit"] = var_logit
    out["ci_lo"] = ci_lo
    out["ci_hi"] = ci_hi
    return out


def _interval(
    k: np.ndarray, n: np.ndarray, method: str, level: float
) -> tuple[np.ndarray, np.ndarray]:
    alpha = 1.0 - level
    if method == "jeffreys":
        lo = beta.ppf(alpha / 2, k + 0.5, n - k + 0.5)
        hi = beta.ppf(1 - alpha / 2, k + 0.5, n - k + 0.5)
        lo = np.where(k == 0, 0.0, lo)  # Jeffreys interval convention at the boundaries
        hi = np.where(k == n, 1.0, hi)
        return lo, hi
    if method == "wilson":
        z = norm.ppf(1 - alpha / 2)
        phat = k / n
        denom = 1 + z * z / n
        center = (phat + z * z / (2 * n)) / denom
        half = z * np.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
        return np.clip(center - half, 0, 1), np.clip(center + half, 0, 1)
    raise ValueError(f"unknown interval {method!r}; use jeffreys | wilson")


# =================================================================================================
# 2. predictive distributions across replicates
# =================================================================================================
def timing_distribution(
    tte_table: pd.DataFrame, *, run_keys: list[str], seed_col: str, horizon: int
) -> pd.DataFrame:
    """Per-run predictive summary of a duration (from per-replicate :func:`time_to_event`).

    The distribution is *defective* (some replicates never see the event); non-occurring replicates
    are right-censored at ``horizon`` and durations are capped there, making the quantile summaries
    well-defined. Returns
    ``[*run_keys, q10, q25, q50, q75, q90, p_within_horizon, n_occurred, n]``.
    """
    occurred_within = tte_table["observed"].to_numpy() & (
        tte_table["duration"].to_numpy() <= horizon
    )
    capped = np.where(occurred_within, tte_table["duration"].to_numpy(), horizon)
    work = tte_table[run_keys].copy()
    work["capped"] = capped
    work["occurred_within"] = occurred_within

    grouped = work.groupby(run_keys, observed=True)
    quant = grouped["capped"].quantile([0.10, 0.25, 0.50, 0.75, 0.90]).unstack()
    quant.columns = ["q10", "q25", "q50", "q75", "q90"]
    agg = grouped.agg(n_occurred=("occurred_within", "sum"), n=("occurred_within", "size"))
    out = quant.join(agg).reset_index()
    out["p_within_horizon"] = out["n_occurred"] / out["n"]
    out["n_occurred"] = out["n_occurred"].astype(np.int64)
    out["n"] = out["n"].astype(np.int64)
    cols = [*run_keys, "q10", "q25", "q50", "q75", "q90", "p_within_horizon", "n_occurred", "n"]
    return out[cols].sort_values(run_keys).reset_index(drop=True)


def count_distribution(
    count_table: pd.DataFrame, *, run_keys: list[str], seed_col: str
) -> pd.DataFrame:
    """Per-run empirical pmf over a per-replicate integer ``count`` column.

    Returns long format ``[*run_keys, count, prob]`` — feeds individual-level parity/CCF
    uncertainty. Use :func:`count_moments` for predictive mean/variance.
    """
    n = count_table.groupby(run_keys, observed=True).size().rename("n")
    cells = (
        count_table.groupby([*run_keys, "count"], observed=True).size().rename("_c").reset_index()
    )
    cells = cells.merge(n.reset_index(), on=run_keys, how="left")
    cells["prob"] = cells["_c"] / cells["n"]
    return (
        cells[[*run_keys, "count", "prob"]].sort_values([*run_keys, "count"]).reset_index(drop=True)
    )


def count_moments(count_table: pd.DataFrame, *, run_keys: list[str], seed_col: str) -> pd.DataFrame:
    """Per-run predictive ``mean``, ``var`` and replicate count ``k`` of the ``count`` column.
    """
    grouped = count_table.groupby(run_keys, observed=True)["count"]
    # ddof = 0 because this is a descriptive property of the distribution for the individual 
    out = grouped.agg(mean="mean", var=lambda s: s.var(ddof=0), k="size").reset_index()
    return out.sort_values(run_keys).reset_index(drop=True)


# =================================================================================================
# 3. Monte-Carlo error: quantify, correct, baseline
# =================================================================================================
def brier_noise_correction(summary: pd.DataFrame) -> float:
    """Mean per-run Brier inflation from finite replicates: ``E[p(1−p)/n]``.

    ``E[Brier(p_hat)] = Brier(p) + E[p(1−p)/n]``; the unbiased per-run estimate of the inflation is
    ``p̂_mle(1−p̂_mle)/(n−1)``. Corrected Brier (04) = raw − this value. Runs with ``n < 2`` carry no
    usable MC-error estimate and are excluded.
    """
    k = summary["k"].to_numpy().astype(np.float64)
    n = summary["n"].to_numpy().astype(np.float64)
    mask = n >= 2
    if not mask.any():
        return 0.0
    p = k[mask] / n[mask]
    inflation = p * (1 - p) / (n[mask] - 1)
    return float(np.mean(inflation))


def null_calibration_band(
    summary: pd.DataFrame,
    *,
    n_bins: int = 10,
    strategy: str = "uniform",
    n_sims: int = 200,
    rng: np.random.Generator,
    band_level: float = 0.95,
) -> pd.DataFrame:
    """Reliability-diagram scatter envelope expected under **perfect** calibration.

    Treating each run's own ``p_hat = k/n`` as the true probability, simulate
    ``k* ~ Binomial(n, p_true)`` (the model's noisy re-estimate) and an independent realized
    outcome ``y* ~ Bernoulli(p_true)``, bin runs by the re-estimated probability, and record the
    observed frequency per bin. Repeated ``n_sims`` times, this yields the per-bin ``(lo, hi)``
    envelope a perfectly calibrated model would produce at the observed ``n`` profile. A model is
    only demonstrably miscalibrated where its curve exits this band — the guard against
    over-reading MC noise as miscalibration. Returns ``[bin, bin_left, bin_right, lo, hi]``.

    Limitation: the only truth signal available is each run's own ``p_hat``, which at small n is a
    coarse view of the true probability that piles up on exactly 0 and 1. The outer bins are then
    anchored on runs whose ``p_true`` is degenerate, and the band there collapses toward the bin
    edge — so a perfectly-calibrated model's reliability curve can still exit it. At that point the
    reliability diagram itself is unreliable, and the fix is to generate more replicates before
    reading it.
    """
    k = summary["k"].to_numpy().astype(np.float64)
    n = summary["n"].to_numpy().astype(np.float64)
    p_true = k / n

    if strategy == "uniform":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    elif strategy == "quantile":
        edges = np.unique(np.quantile(p_true, np.linspace(0, 1, n_bins + 1)))
        edges[0], edges[-1] = 0.0, 1.0
    else:
        raise ValueError(f"unknown strategy {strategy!r}; use uniform | quantile")
    n_bins = len(edges) - 1

    freqs = np.full((n_sims, n_bins), np.nan)
    for s in range(n_sims):
        k_star = rng.binomial(n.astype(int), p_true)
        p_pred = k_star.astype(np.float64) / n
        y_star = (rng.random(len(n)) < p_true).astype(np.float64)
        idx = np.clip(np.digitize(p_pred, edges, right=False) - 1, 0, n_bins - 1)
        for b in range(n_bins):
            sel = idx == b
            if sel.any():
                freqs[s, b] = y_star[sel].mean()

    alpha = 1 - band_level
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # empty bins -> NaN envelope
        lo = np.nanpercentile(freqs, 100 * alpha / 2, axis=0)
        hi = np.nanpercentile(freqs, 100 * (1 - alpha / 2), axis=0)
    return pd.DataFrame(
        {
            "bin": np.arange(n_bins),
            "bin_left": edges[:-1],
            "bin_right": edges[1:],
            "lo": lo,
            "hi": hi,
        }
    )


# =================================================================================================
# 4. resampling over the replicate dimension
# =================================================================================================
def _numeric_and_key_cols(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    keys = [c for c in df.columns if c not in numeric]
    return keys, numeric


def _melt_stat(result: pd.DataFrame, value_cols: list[str] | None = None) -> pd.DataFrame:
    """Melt a stat_fn output to ``[*key_cols, metric, __value__]`` (value columns become rows).

    ``value_cols`` names the numeric statistic columns explicitly — required when grouping keys are
    themselves numeric (cohort year, parity, window ages), which the dtype heuristic would otherwise
    mistake for values. When omitted, numeric columns are treated as values. A sentinel value column
    name is used so it never collides with a key column a ``stat_fn`` happens to call ``value``.
    """
    if value_cols is not None:
        numeric = list(value_cols)
        keys = [c for c in result.columns if c not in numeric]
    else:
        keys, numeric = _numeric_and_key_cols(result)
    return result.melt(id_vars=keys, value_vars=numeric, var_name="metric", value_name="__value__")


def _window_seed_groups(df: pd.DataFrame, seed_col: str) -> tuple[list[str], dict]:
    win = [c for c in ("age_start", "age_stop") if c in df.columns]
    if win:
        groups = {key: sub for key, sub in df.groupby(win, observed=True)}
    else:
        groups = {(): df}
    return win, groups


def seed_bootstrap(
    df: pd.DataFrame,
    *,
    seed_col: str,
    stat_fn: Callable[[pd.DataFrame], pd.DataFrame],
    n_boot: int,
    rng: np.random.Generator,
    level: float = 0.95,
    value_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Percentile CIs for any aggregate ``stat_fn`` by resampling seed labels with replacement.

    Seeds are resampled *within each window* (``age_start``, ``age_stop``) so window structure is
    preserved. ``stat_fn`` maps a replicate-level frame to a tidy frame (key columns + numeric
    value columns); it is applied once per bootstrap draw — outcome evaluation is **never** re-run
    inside the loop (evaluate once per replicate, resample the rows). ``value_cols`` names the
    statistic columns explicitly (needed when grouping keys are numeric). Captures Monte-Carlo
    (replicate) uncertainty only; population sampling uncertainty would need a person-level cluster
    bootstrap (out of scope v1). Returns ``[*keys, metric, estimate, ci_lo, ci_hi]``.
    """
    _, groups = _window_seed_groups(df, seed_col)
    seed_index = {
        key: {s: sub for s, sub in g.groupby(seed_col, observed=True)} for key, g in groups.items()
    }

    draws = []
    for _ in range(n_boot):
        parts = []
        new_id = 0
        for per_seed in seed_index.values():
            seeds = np.array(list(per_seed.keys()))
            chosen = rng.choice(seeds, size=len(seeds), replace=True)
            # Relabel each resampled copy as a fresh replicate so duplicated seed labels are not
            # collapsed by a stat_fn that groups on seed (e.g. dividing by the number of seeds).
            for s in chosen:
                sub = per_seed[s].copy()
                sub[seed_col] = new_id
                new_id += 1
                parts.append(sub)
        resampled = pd.concat(parts, ignore_index=True)
        draws.append(_melt_stat(stat_fn(resampled), value_cols))

    stacked = pd.concat(draws, ignore_index=True)
    keys = [c for c in stacked.columns if c != "__value__"]
    alpha = 1 - level
    ci = (
        stacked.groupby(keys, observed=True)["__value__"]
        .agg(ci_lo=lambda s: s.quantile(alpha / 2), ci_hi=lambda s: s.quantile(1 - alpha / 2))
        .reset_index()
    )
    point = _melt_stat(stat_fn(df), value_cols).rename(columns={"__value__": "estimate"})
    merge_keys = [c for c in point.columns if c != "estimate"]
    return ci.merge(point, on=merge_keys, how="left")
