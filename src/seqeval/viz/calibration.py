"""Calibration figures: reliability diagram with the null band, and convergence curves (04 viz)."""

from __future__ import annotations

import pandas as pd
from matplotlib.figure import Figure

from seqeval.viz._style import FIGSIZE, new_fig, stratum_colors


def plot_reliability(
    calibration: pd.DataFrame, *, probs: pd.DataFrame | None = None, title: str | None = None
) -> Figure:
    """Reliability diagram with the perfect-calibration null band shaded and the diagonal reference.

    ``calibration`` is :func:`seqeval.metrics.ml.calibration_table` merged with the null band
    (columns ``p_mean, y_rate, band_lo, band_hi`` per bin). When ``probs`` is given (the run-level
    probability table), a ``p_hat`` histogram is drawn in a lower panel so bin populations are
    visible. A model is only demonstrably miscalibrated where its curve exits the band.
    """
    import matplotlib.pyplot as plt

    cal = calibration.sort_values("bin")
    centers = (cal["bin_left"] + cal["bin_right"]) / 2

    if probs is None:
        fig, ax = new_fig((5.5, 5.5))
        axes = [ax]
    else:
        fig, axes = plt.subplots(2, 1, figsize=(5.5, 6.5), height_ratios=[3, 1], sharex=True)
        ax = axes[0]
        ax.grid(True, alpha=0.3, linewidth=0.5)

    if {"band_lo", "band_hi"} <= set(cal.columns):
        ax.fill_between(
            centers, cal["band_lo"], cal["band_hi"], alpha=0.3, color="tab:blue", label="null band"
        )
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="ideal")
    ax.plot(cal["p_mean"], cal["y_rate"], "o-", color="tab:red", label="model")
    ax.set_ylabel("observed frequency")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=8)
    if title:
        ax.set_title(title, fontsize=10)

    if probs is not None:
        axes[1].hist(probs["p_hat"].to_numpy(), bins=20, range=(0, 1), color="tab:gray")
        axes[1].set_ylabel("count")
        axes[1].set_xlabel("predicted p_hat")
        axes[1].grid(True, alpha=0.3, linewidth=0.5)
    else:
        ax.set_xlabel("predicted p_hat")
    fig.tight_layout()
    return fig


def plot_convergence(
    convergence: pd.DataFrame, *, metrics: tuple[str, ...] = ("auc", "brier", "ece")
) -> Figure:
    """Metric estimate vs number of seeds (mean +/- sd) — when has the estimate stabilized?"""
    present = [m for m in metrics if m in set(convergence["metric"])]
    fig, ax = new_fig(FIGSIZE)
    for metric, color in zip(present, stratum_colors(len(present)), strict=True):
        g = convergence[convergence["metric"] == metric].sort_values("m")
        ax.errorbar(
            g["m"], g["mean"], yerr=g["std"], marker="o", capsize=3, color=color, label=metric
        )
    ax.set_xlabel("number of seeds m")
    ax.set_ylabel("metric estimate")
    ax.set_title("Seed-convergence of backtest metrics")
    ax.legend(fontsize=8)
    return fig
