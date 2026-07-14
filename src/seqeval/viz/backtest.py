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


def _step_sample(km_one: pd.DataFrame, grid: np.ndarray) -> np.ndarray:
    """Survival of one KM curve sampled at ``grid`` times (step function; 1.0 before the first)."""
    g = km_one.sort_values("time")
    times = g["time"].to_numpy()
    surv = g["survival"].to_numpy()
    idx = np.searchsorted(times, grid, side="right") - 1
    return np.where(idx >= 0, surv[np.clip(idx, 0, len(surv) - 1)], 1.0)
