"""ASFR-baseline figures: reliability overlay, per-individual comparison, skill vs jump-off (04)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from seqeval.viz._style import FIGSIZE, new_fig, stratum_colors


def plot_reliability_overlay(
    cal_model: pd.DataFrame, cal_base: pd.DataFrame, *, title: str | None = None
) -> Figure:
    """Model and ASFR-baseline reliability curves on one axes — who tracks the diagonal better?

    Both frames are :func:`seqeval.metrics.ml.calibration_table` outputs (columns ``p_mean``,
    ``y_rate``). The baseline curve is typically short and bunched: a population schedule assigns
    nearly the same probability to everyone of the same age, so it spans little of the x-axis. That
    narrowness *is* the comparison — the model's spread along x is the individual-level information
    the schedule does not have.
    """
    fig, ax = new_fig((5.5, 5.5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="ideal")
    ax.plot(
        cal_base["p_mean"],
        cal_base["y_rate"],
        "s-",
        color="tab:gray",
        label="ASFR baseline",
        lw=1.2,
    )
    ax.plot(cal_model["p_mean"], cal_model["y_rate"], "o-", color="tab:red", label="model", lw=1.2)
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("observed frequency")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=8)
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    return fig


def plot_individual_comparison(
    individual: pd.DataFrame, *, title: str | None = None, rng_seed: int = 0
) -> Figure:
    """Per-person model probability vs ASFR-baseline probability, split by what actually happened.

    ``individual`` carries ``p_hat`` (model), ``p_base`` (baseline) and ``y_true``. Points above the
    dashed ``y = x`` line are people the model considers *more* likely to have the event than their
    age-and-year schedule implies; below it, less likely. Good individual-level discrimination shows
    as the event cases sitting above the line and the non-cases below it — vertical spread at a
    given x is exactly the lift the model adds over the population baseline; a flat horizontal cloud
    means the model is ignoring what the schedule knows.

    The model's probability lives on a ``k/n`` grid, so with few seeds every point would stack on a
    handful of rows: a deterministic jitter of half a grid step spreads them, and each class's mean
    is drawn as a large marker so the signal survives the overplotting.
    """
    fig, ax = new_fig((5.5, 5.0))
    df = individual.dropna(subset=["p_base", "p_hat", "y_true"])
    step = 1.0 / float(df["n"].median()) if "n" in df and len(df) and df["n"].median() else 0.0
    rng = np.random.default_rng(rng_seed)

    classes = [("no event", 0, "tab:blue"), ("event", 1, "tab:orange")]
    # Draw the larger class first so the smaller one is not buried under it.
    classes.sort(key=lambda c: -int((df["y_true"] == c[1]).sum()))
    for label, value, color in classes:
        sub = df[df["y_true"] == value]
        if sub.empty:
            continue
        y = sub["p_hat"].to_numpy(dtype=float)
        if step:
            y = np.clip(y + rng.uniform(-step / 2, step / 2, size=len(y)), 0.0, 1.0)
        ax.scatter(sub["p_base"], y, s=6, alpha=0.20, color=color, linewidths=0, label=None)
        ax.scatter(
            [sub["p_base"].mean()],
            [sub["p_hat"].mean()],
            s=110,
            color=color,
            edgecolor="black",
            zorder=5,
            label=f"{label} (n={len(sub)}, mean)",
        )

    lo = max(0.0, min(df["p_base"].min(), df["p_hat"].min()) - 0.05) if len(df) else 0.0
    hi = min(1.0, max(df["p_base"].max(), df["p_hat"].max()) + 0.05) if len(df) else 1.0
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="model = baseline")
    ax.set_xlabel("ASFR-baseline probability")
    ax.set_ylabel("model probability" + (" (jittered)" if step else ""))
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.legend(fontsize=8, loc="upper left", framealpha=0.9)
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    return fig


def plot_lift_by_baseline(individual: pd.DataFrame, *, n_bins: int = 10, title=None) -> Figure:
    """Observed event rate vs model and baseline predictions, within equal-count baseline bins.

    Groups people into ``n_bins`` equal-count bins of their *baseline* probability and, in each,
    plots the mean baseline prediction, the mean model prediction, and the observed rate. Where the
    model's line separates from the baseline's and moves *toward* the observed rate, the model is
    adding information; where the two lines coincide, it is reproducing the schedule.
    """
    df = individual.dropna(subset=["p_base", "p_hat", "y_true"]).copy()
    fig, ax = new_fig(FIGSIZE)
    if df.empty:
        ax.set_title(title or "baseline lift (no data)")
        return fig

    edges = np.unique(np.quantile(df["p_base"], np.linspace(0, 1, n_bins + 1)))
    idx = np.clip(np.digitize(df["p_base"], edges) - 1, 0, max(len(edges) - 2, 0))
    df["_bin"] = idx
    g = df.groupby("_bin", observed=True).agg(
        p_base=("p_base", "mean"),
        p_model=("p_hat", "mean"),
        observed=("y_true", "mean"),
        n=("y_true", "size"),
    )
    ax.plot(g["p_base"], g["observed"], "o-", color="black", label="observed rate")
    ax.plot(g["p_base"], g["p_base"], "s--", color="tab:gray", label="ASFR baseline")
    ax.plot(g["p_base"], g["p_model"], "^-", color="tab:red", label="model (mean)")
    ax.set_xlabel("ASFR-baseline probability (bin mean)")
    ax.set_ylabel("probability / observed rate")
    ax.legend(fontsize=8)
    if title:
        ax.set_title(title, fontsize=10)
    return fig


def plot_skill_vs_jumpoff(comparison: pd.DataFrame, *, metric: str = "brier") -> Figure:
    """Skill over the ASFR baseline vs jump-off age, one line per outcome (condition).

    ``comparison`` is the long ``baseline_scores`` table. The zero line is the baseline itself:
    above it the model beats the age-and-year schedule, below it the schedule wins.
    """
    df = comparison[comparison["metric"] == metric]
    fig, ax = new_fig()
    groups = list(df.groupby(["outcome", "condition"], observed=True))
    for (key, g), color in zip(groups, stratum_colors(len(groups)), strict=True):
        g = g.sort_values("age_stop_years")
        outcome, condition = key
        label = outcome if condition == "-" else f"{outcome} | {condition}"
        ax.plot(g["age_stop_years"], g["skill"], "o-", color=color, label=label)
    ax.axhline(0.0, color="black", lw=1, ls="--")
    ax.set_xlabel("jump-off age (years)")
    ax.set_ylabel(f"{metric} skill vs ASFR baseline")
    ax.set_title(f"{metric} skill over the ASFR baseline (0 = no better than the schedule)")
    ax.legend(fontsize=7)
    return fig
