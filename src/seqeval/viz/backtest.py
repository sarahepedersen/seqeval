"""Backtest summary figures: metric vs jump-off age, and observed-vs-generated KM overlay (04)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from seqeval.units import days_to_years
from seqeval.viz._style import new_fig, stratum_colors


def plot_metric_vs_jumpoff(scores: pd.DataFrame, *, metric: str) -> Figure:
    """One line per (outcome, condition): metric value vs jump-off age (``age_stop`` in years).

    Answers "how does predictability change as the jump-off moves?" — the spec's motivating
    question. ``scores`` is the long backtesting scores table.
    """
    df = scores[scores["metric"] == metric]
    fig, ax = new_fig()
    groups = list(df.groupby(["outcome", "condition"], observed=True))
    for (key, g), color in zip(groups, stratum_colors(len(groups)), strict=True):
        g = g.sort_values("age_stop_years")
        outcome, condition = key
        label = outcome if condition == "-" else f"{outcome} | {condition}"
        ax.plot(g["age_stop_years"], g["value"], "o-", color=color, label=label)
    ax.set_xlabel("jump-off age (years)")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} vs jump-off age")
    ax.legend(fontsize=7)
    return fig


def plot_km_seed_band(
    obs_km: pd.DataFrame, gen_km: pd.DataFrame, *, title: str | None = None
) -> Figure:
    """Observed KM curve overlaid with the generated across-seed band (median + IQR).

    ``gen_km`` carries a ``seed`` column (one KM curve per seed); its survival is sampled on a
    common day grid, then the median and inter-quartile range across seeds are shaded (years axes).
    """
    grid = np.union1d(obs_km["time"].to_numpy(), gen_km["time"].to_numpy())
    per_seed = []
    for _, g in gen_km.groupby("seed", observed=True):
        per_seed.append(_step_sample(g, grid))
    stacked = np.vstack(per_seed) if per_seed else np.empty((0, len(grid)))

    fig, ax = new_fig()
    years = days_to_years(grid)
    if len(stacked):
        med = np.median(stacked, axis=0)
        lo, hi = np.percentile(stacked, [25, 75], axis=0)
        ax.fill_between(
            years, lo, hi, step="post", alpha=0.3, color="tab:orange", label="generated IQR"
        )
        ax.step(years, med, where="post", color="tab:orange", label="generated median")
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
    obs_ccf: pd.DataFrame, gen_ccf: pd.DataFrame, *, title: str | None = None
) -> Figure:
    """Observed CCF by cohort overlaid with the generated across-seed band (median + IQR).

    ``gen_ccf`` carries a ``seed`` column (one CCF-by-cohort curve per seed); the median and IQR
    across seeds are shaded per cohort. Both frames have ``[cohort, ccf]`` (plus seed on generated).
    """
    fig, ax = new_fig()
    g = gen_ccf.groupby("cohort", observed=True)["ccf"]
    med = g.median().sort_index()
    lo = g.quantile(0.25).reindex(med.index)
    hi = g.quantile(0.75).reindex(med.index)
    cohorts = med.index.to_numpy()
    if len(cohorts):
        ax.fill_between(
            cohorts, lo.to_numpy(), hi.to_numpy(), alpha=0.3, color="tab:orange",
            label="generated IQR",
        )
        ax.plot(cohorts, med.to_numpy(), "o-", color="tab:orange", label="generated median")
    o = obs_ccf.sort_values("cohort")
    ax.plot(o["cohort"].to_numpy(), o["ccf"].to_numpy(), "o-", color="black", label="observed")
    ax.set_xlabel("birth cohort")
    ax.set_ylabel("CCF (mean births/woman)")
    if title:
        ax.set_title(title)
    ax.legend(fontsize=8)
    return fig


def plot_timing_calibration(
    td: pd.DataFrame,
    obs_tte: pd.DataFrame,
    *,
    horizon_days: int | None = None,
    title: str | None = None,
    n_bins: int = 10,
) -> Figure:
    """Predicted vs observed waiting time per person: the y=x ideal plus a binned median trend.

    ``td`` is :func:`seqeval.core.replicates.timing_distribution` (per-person quantiles in days);
    ``obs_tte`` is the observed time-to-event table. Gray points are individuals; the red trend is
    the median observed time within equal-count bins of predicted time, with its inter-quartile
    ribbon. Departure from the dashed ``y = x`` line is timing bias — **above** the line the model
    predicts too early, **below** it predicts too late.

    Scope: only persons whose event was actually observed **within the frame horizon** contribute.
    The model's predictive distribution is defective and capped at ``horizon_days``, so it can never
    predict a wait beyond it; comparing against unbounded observed times would be ill-posed. Both
    axes are therefore clipped to the horizon, and predicted medians pile up at that edge for
    persons whose replicates mostly never see the event.
    """
    seen = obs_tte.loc[obs_tte["observed"], ["person_id", "duration"]]
    m = td.merge(seen, on="person_id", how="inner")
    if horizon_days is not None:
        m = m[m["duration"] <= horizon_days]  # inside the frame, where the comparison is well-posed

    fig, ax = new_fig()
    ax.set_xlabel("predicted waiting time (years, median across seeds)")
    ax.set_ylabel("observed waiting time (years)")
    if title:
        ax.set_title(title, fontsize=10)
    if m.empty:
        return fig

    pred = days_to_years(m["q50"].to_numpy().astype(float))
    obs = days_to_years(m["duration"].to_numpy().astype(float))
    ax.scatter(pred, obs, s=8, alpha=0.15, color="tab:gray", linewidths=0)

    if horizon_days is not None:
        lim = float(days_to_years(horizon_days))
    else:
        lim = float(max(pred.max(), obs.max())) * 1.05 or 1.0
    ax.plot([0, lim], [0, lim], "k--", lw=1, label="ideal (y = x)")

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

    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
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
