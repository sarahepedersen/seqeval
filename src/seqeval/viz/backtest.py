"""Backtest overlay figures: observed-vs-generated KM/CCF bands and timing calibration (04).

Scalar scores (AUC, Brier, ...) are deliberately *not* plotted here — they are reported as numbers
in the report's per-outcome metrics table, where the value and its bootstrap CI are legible.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from scipy.stats import norm

from seqeval.units import days_to_years
from seqeval.viz._style import new_fig, stratum_colors

DEFAULT_LEVEL = 0.95


def _seed_ci(values: np.ndarray, level: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Across-seed mean and its Monte-Carlo CI, from a ``(n_seeds, ...)`` stack.

    ``mean ± z·sd/√K`` with the population sd (``ddof=0``). This is the *same quantity* as
    ``replicate_variance_aggregate.se``: that table's analytic decomposition
    ``sqrt(Σ_i s²_i/K)/n`` equals the standard error of the across-seed mean whenever persons are
    independent within a seed, and ``ddof=0`` makes the two agree exactly rather than up to a
    ``(K-1)/K`` factor.

    What is plotted here is the uncertainty in the estimate actually being drawn — the across-seed
    mean — which does shrink as √K.
    """
    k = values.shape[0]
    mean = values.mean(axis=0)
    sem = values.std(axis=0, ddof=0) / np.sqrt(k)
    z = norm.ppf(1 - (1 - level) / 2)
    return mean, mean - z * sem, mean + z * sem


def plot_km_seed_band(
    obs_km: pd.DataFrame,
    gen_km: pd.DataFrame,
    *,
    title: str | None = None,
    level: float = DEFAULT_LEVEL,
) -> Figure:
    """Observed KM curve overlaid with the generated across-seed mean and its Monte-Carlo CI.

    ``gen_km`` carries a ``seed`` column (one KM curve per seed); its survival is sampled on a
    common day grid, then :func:`_seed_ci` gives the pointwise mean and ``±z·sd/√K`` band (years
    axes). The band is replicate uncertainty in the plotted curve, on the same footing as
    ``replicate_variance_aggregate.se`` — not the spread of an individual seed's curve.
    """
    grid = np.union1d(obs_km["time"].to_numpy(), gen_km["time"].to_numpy())
    per_seed = []
    for _, g in gen_km.groupby("seed", observed=True):
        per_seed.append(_step_sample(g, grid))
    stacked = np.vstack(per_seed) if per_seed else np.empty((0, len(grid)))

    fig, ax = new_fig()
    years = days_to_years(grid)
    if len(stacked):
        mean, lo, hi = _seed_ci(stacked, level)
        ax.fill_between(
            years, lo, hi, step="post", alpha=0.3, color="tab:orange",
            label=f"generated {level:.0%} CI",
        )
        ax.step(years, mean, where="post", color="tab:orange", label="generated mean")
    ax.step(
        days_to_years(obs_km["time"].to_numpy()),
        obs_km["survival"],
        where="post",
        color="black",
        label="observed",
    )
    ax.set_xlabel("age (years)")
    ax.set_ylabel("survival S(t)")
    ax.set_ylim(0, 1.02)
    if title:
        ax.set_title(title)
    ax.legend(fontsize=8)
    return fig


def plot_ccf_seed_band(
    obs_ccf: pd.DataFrame,
    gen_ccf: pd.DataFrame,
    *,
    title: str | None = None,
    level: float = DEFAULT_LEVEL,
) -> Figure:
    """Observed CCF by cohort overlaid with the generated across-seed mean and Monte-Carlo CI.

    ``gen_ccf`` carries a ``seed`` column (one CCF-by-cohort curve per seed); per cohort the mean
    and its ``±z·sd/√K`` band are shaded (see :func:`_seed_ci` — the same replicate uncertainty
    reported as ``replicate_variance_aggregate.se``). Both frames have ``[cohort, ccf]`` (plus seed
    on generated).
    """
    fig, ax = new_fig()
    _draw_ccf_band(
        ax, gen_ccf, level, color="tab:orange",
        line_label="generated mean", band_label=f"generated {level:.0%} CI",
    )
    _draw_observed_ccf(ax, obs_ccf)
    ax.set_xlabel("birth cohort")
    ax.set_ylabel("CCF (mean births/woman)")
    if title:
        ax.set_title(title)
    ax.legend(fontsize=8)
    return fig


def _ccf_band(gen_ccf: pd.DataFrame, level: float):
    """``(cohorts, mean, half_width, complete)`` for one seed-replicated CCF-by-cohort frame."""
    stats = (
        gen_ccf.groupby("cohort", observed=True)["ccf"]
        .agg(mean="mean", sd=lambda s: s.std(ddof=0), k="size")
        .sort_index()
    )
    z = norm.ppf(1 - (1 - level) / 2)
    half = z * stats["sd"] / np.sqrt(stats["k"])
    complete = _majority_complete(gen_ccf).reindex(stats.index, fill_value=True).to_numpy()
    return stats.index.to_numpy(), stats["mean"].to_numpy(), half.to_numpy(), complete


def _draw_ccf_band(
    ax, gen_ccf, level, *, color, line_label, band_label=None, alpha=0.3, incomplete_label=True
) -> None:
    """One generated CCF curve: mean line (dashed where incomplete) inside its replicate CI.

    ``band_label=None`` leaves the shaded band out of the legend — right on the multi-window panel,
    where the level is stated once in the legend title instead of once per curve.
    """
    cohorts, mean, half, complete = _ccf_band(gen_ccf, level)
    if not len(cohorts):
        return
    ax.fill_between(
        cohorts, mean - half, mean + half, alpha=alpha, color=color,
        label=band_label if band_label else "_nolegend_",
    )
    _plot_with_incomplete(
        ax, cohorts, mean, complete, color=color, label=line_label,
        incomplete_label=incomplete_label,
    )


def _draw_observed_ccf(ax, obs_ccf: pd.DataFrame, *, incomplete_label: bool = True) -> None:
    """The observed CCF-by-cohort curve in black, dashed over truncated cohorts."""
    o = obs_ccf.sort_values("cohort")
    complete = (
        o["complete"].astype(bool).to_numpy()
        if "complete" in o.columns
        else np.ones(len(o), dtype=bool)
    )
    _plot_with_incomplete(
        ax, o["cohort"].to_numpy(), o["ccf"].to_numpy(), complete, color="black",
        label="observed", incomplete_label=incomplete_label,
    )


def plot_km_jumpoff_panel(
    obs_km: pd.DataFrame,
    gen_by_jumpoff: dict[int, pd.DataFrame],
    *,
    title: str | None = None,
    level: float = DEFAULT_LEVEL,
) -> Figure:
    """Every jump-off's generated KM curve on one axes, against a single observed curve.
    """
    jumpoffs = sorted(gen_by_jumpoff)
    grid = np.union1d(
        obs_km["time"].to_numpy(),
        np.concatenate([g["time"].to_numpy() for g in gen_by_jumpoff.values()])
        if gen_by_jumpoff
        else np.empty(0),
    )
    years = days_to_years(grid)

    fig, ax = new_fig()
    for t2, color in zip(jumpoffs, stratum_colors(len(jumpoffs), lo=0.1, hi=0.85), strict=True):
        gen_km = gen_by_jumpoff[t2]
        per_seed = [_step_sample(g, grid) for _, g in gen_km.groupby("seed", observed=True)]
        if not per_seed:
            continue
        mean, lo, hi = _seed_ci(np.vstack(per_seed), level)
        jump_y = days_to_years(t2)
        ax.fill_between(years, lo, hi, step="post", alpha=0.2, color=color, label="_nolegend_")
        ax.step(years, mean, where="post", color=color, lw=2, label=f"jump-off {jump_y:.0f}y")
        ax.axvline(jump_y, color=color, lw=0.8, ls=":", alpha=0.7)
    ax.step(
        days_to_years(obs_km["time"].to_numpy()),
        obs_km["survival"],
        where="post",
        color="black",
        lw=2,
        label="observed",
    )
    ax.set_xlabel("age (years)")
    ax.set_ylabel("survival S(t)")
    ax.set_ylim(0, 1.02)
    if title:
        ax.set_title(title)
    ax.legend(fontsize=8, title=f"bands: {level:.0%} replicate CI", title_fontsize=7)
    return fig


def plot_ccf_jumpoff_panel(
    obs_ccf: pd.DataFrame,
    gen_by_jumpoff: dict[int, pd.DataFrame],
    *,
    title: str | None = None,
    level: float = DEFAULT_LEVEL,
) -> Figure:
    """Every jump-off's generated CCF-by-cohort curve on one axes, against one observed curve.
    """
    jumpoffs = sorted(gen_by_jumpoff)
    fig, ax = new_fig()
    for t2, color in zip(jumpoffs, stratum_colors(len(jumpoffs), lo=0.1, hi=0.85), strict=True):
        _draw_ccf_band(
            ax, gen_by_jumpoff[t2], level, color=color,
            line_label=f"jump-off {days_to_years(t2):.0f}y", alpha=0.2, incomplete_label=False,
        )
    _draw_observed_ccf(ax, obs_ccf)
    ax.set_xlabel("birth cohort")
    ax.set_ylabel("CCF (mean births/woman)")
    if title:
        ax.set_title(title)
    ax.legend(fontsize=8, title=f"bands: {level:.0%} replicate CI", title_fontsize=7)
    return fig


def _majority_complete(gen_ccf: pd.DataFrame) -> pd.Series:
    """Per cohort: is the cohort complete in the majority of seeds? (all-complete if unmarked)."""
    if "complete" not in gen_ccf.columns:
        return pd.Series(dtype=bool)
    return (
        gen_ccf.assign(_c=gen_ccf["complete"].astype(float))
        .groupby("cohort", observed=True)["_c"]
        .mean()
        .gt(0.5)
        .sort_index()
    )


def _plot_with_incomplete(
    ax, x, y, complete, *, color: str, label: str, incomplete_label: bool = True
) -> None:
    """One CCF curve: solid/filled where the cohort is complete, dashed/open where it is not.

    The dashed run starts at the last complete cohort so the two segments join up rather than
    leaving a visual gap at the transition. ``incomplete_label=False`` keeps the styling but drops
    the second legend entry — on the multi-jump-off panel one legend row per curve is already
    enough, and the dashed/open convention is explained in the caption.
    """
    complete = np.asarray(complete, dtype=bool)
    ax.plot(x[complete], y[complete], "o-", color=color, label=label)
    if complete.all():
        return
    first_incomplete = int(np.argmax(~complete))
    start = max(first_incomplete - 1, 0)  # bridge back to the last complete point
    seg = np.zeros(len(x), dtype=bool)
    seg[start:] = ~complete[start:]
    seg[start] = True
    ax.plot(
        x[seg], y[seg], "o--", color=color, mfc="white",
        label=f"{label} (incomplete cohorts)" if incomplete_label else "_nolegend_",
    )


def timing_pairs(
    td: pd.DataFrame,
    obs_tte: pd.DataFrame,
    *,
    horizon_days: int | None = None,
    persons: Iterable | None = None,
    drop_projected_beyond: bool = True,
) -> pd.DataFrame:
    """Per-person ``[person_id, pred, obs]`` waiting times in **years**, on the scored population.

    The single place the predicted (seed-median) and observed waiting times are paired up, shared
    by :func:`plot_timing_calibration` and by the arm's outlier accounting so both describe exactly
    the same set of persons. A person contributes only when the comparison is a real prediction:

    - the event was actually observed, and observed **within the frame horizon** — the model's
      predictive distribution is defective and capped at ``horizon_days``, so it can never predict
      a longer wait and comparing against unbounded observed times would be ill-posed;
    - the person is in ``persons`` when given — the arm passes the scored population (its condition
      minus those whose answer was already settled at the jump-off). Settled persons matter most:
      their event sits in the observed prefix, which every replicate replays verbatim, so they
      would land exactly on ``y = x`` and flatter the model for free;
    - ``drop_projected_beyond`` (default) drops persons whose predicted median has reached the
      horizon, i.e. the model projects the event *outside* the outcome's frame. Their ``q50`` is the
      cap itself, not an estimated date, so they form a spurious vertical stripe at the frame edge
      that drags the binned trend toward it. Whether the model gets those persons right is the
      *probability* question, answered by the reliability diagram next door.
    """
    seen = obs_tte.loc[obs_tte["observed"], ["person_id", "duration"]]
    m = td.merge(seen, on="person_id", how="inner")
    if persons is not None:
        m = m[m["person_id"].isin(set(persons))]
    if horizon_days is not None:
        m = m[m["duration"] <= horizon_days]
        if drop_projected_beyond:
            m = m[m["q50"] < horizon_days]
    return pd.DataFrame(
        {
            "person_id": m["person_id"].to_numpy(),
            "pred": days_to_years(m["q50"].to_numpy().astype(float)),
            "obs": days_to_years(m["duration"].to_numpy().astype(float)),
        }
    )


def plot_timing_calibration(
    td: pd.DataFrame,
    obs_tte: pd.DataFrame,
    *,
    horizon_days: int | None = None,
    floor_days: int = 0,
    persons: Iterable | None = None,
    xlabel: str = "predicted waiting time (years, median across seeds)",
    ylabel: str = "observed waiting time (years)",
    title: str | None = None,
    n_bins: int = 10,
) -> Figure:
    """Predicted vs observed waiting time per person: the y=x ideal plus a binned median trend.

    ``td`` is :func:`seqeval.core.replicates.timing_distribution` (per-person quantiles in days);
    ``obs_tte`` is the observed time-to-event table. Gray points are individuals; the red trend is
    the median observed time within equal-count bins of predicted time, with its inter-quartile
    ribbon. Departure from the dashed ``y = x`` line is timing bias — **above** the line the model
    predicts too early, **below** it predicts too late.

    Scope is :func:`timing_pairs` — persons whose event was observed inside the frame, who were
    still unsettled at the jump-off, and whom the model does not project past the frame. The axes
    span that same region: ``floor_days`` (the jump-off, where prediction begins, for an outcome
    whose duration is an age; 0 otherwise) to ``horizon_days`` (where the frame closes). Nothing
    can land outside it, so no part of the box is unreachable and there is nothing to clip.
    """
    pairs = timing_pairs(td, obs_tte, horizon_days=horizon_days, persons=persons)

    fig, ax = new_fig()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontsize=10)
    if pairs.empty:
        return fig

    pred, obs = pairs["pred"].to_numpy(), pairs["obs"].to_numpy()
    ax.scatter(pred, obs, s=8, alpha=0.15, color="tab:gray", linewidths=0)

    xlo = float(days_to_years(floor_days))
    if horizon_days is not None:
        lim = float(days_to_years(horizon_days))
    else:
        lim = float(max(pred.max(), obs.max())) * 1.05 or 1.0
    ax.plot([xlo, lim], [xlo, lim], "k--", lw=1, label="ideal (y = x)")

    edges = np.unique(np.quantile(pred, np.linspace(0, 1, n_bins + 1)))
    if len(edges) > 2:
        idx = np.clip(np.digitize(pred, edges) - 1, 0, len(edges) - 2)
        centers, med, lo, hi = [], [], [], []
        for b in range(len(edges) - 1):
            sel = idx == b
            if sel.sum() < 5:  # too few persons to summarise this bin
                continue
            centers.append(float(np.median(pred[sel])))
            med.append(float(np.median(obs[sel])))
            lo.append(float(np.percentile(obs[sel], 25)))
            hi.append(float(np.percentile(obs[sel], 75)))
        if centers:
            ax.fill_between(centers, lo, hi, alpha=0.25, color="tab:red", label="binned IQR")
            ax.plot(centers, med, "o-", color="tab:red", ms=5, label="binned median")

    ax.set_xlim(xlo, lim)
    ax.set_ylim(xlo, lim)
    ax.set_aspect("equal")  # equal axes: y = x reads as a true 45° diagonal
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.legend(fontsize=8)
    return fig


def _step_sample(km_one: pd.DataFrame, grid: np.ndarray) -> np.ndarray:
    """Survival of one KM curve sampled at ``grid`` times (step function; 1.0 before the first)."""
    g = km_one.sort_values("time")
    times = g["time"].to_numpy()
    surv = g["survival"].to_numpy()
    idx = np.searchsorted(times, grid, side="right") - 1
    return np.where(idx >= 0, surv[np.clip(idx, 0, len(surv) - 1)], 1.0)
