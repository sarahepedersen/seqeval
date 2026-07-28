"""Calibration figures: the reliability diagram (04 viz)."""

from __future__ import annotations

import pandas as pd
from matplotlib.figure import Figure


def plot_reliability(calibration: pd.DataFrame, *, title: str | None = None) -> Figure:
    """Reliability diagram: binned observed frequency vs predicted probability, plus the diagonal.

    ``calibration`` is :func:`seqeval.metrics.ml.calibration_table` (columns ``bin, bin_left,
    bin_right, p_mean, y_rate, n``); with quantile binning each point rests on roughly the same
    number of persons. The lower panel is that table's own ``n``, one bar per bin, so the curve and
    the counts under it are the same grouping by construction.

    Only bins that hold someone are drawn. ``p_hat`` lives on a coarse ``1/n_seeds`` grid, so
    quantile edges routinely produce empty deciles; drawn as zero-height bars they read as a real
    trough in the distribution rather than as a bin that could not exist.
    """
    import matplotlib.pyplot as plt

    cal = calibration.sort_values("bin")
    occupied = cal[cal["n"] > 0]

    fig, axes = plt.subplots(2, 1, figsize=(5.5, 6.5), height_ratios=[3, 1], sharex=True)
    ax = axes[0]
    ax.grid(True, alpha=0.3, linewidth=0.5)

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="ideal")
    ax.plot(occupied["p_mean"], occupied["y_rate"], "o-", color="tab:red", label="model")
    ax.set_ylabel("observed frequency")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=8)
    if title:
        ax.set_title(title, fontsize=10)

    left = occupied["bin_left"].to_numpy()
    width = occupied["bin_right"].to_numpy() - left
    axes[1].bar(
        left, occupied["n"].to_numpy(), width=width, align="edge",
        color="tab:gray", edgecolor="white",
    )
    axes[1].set_ylabel("count")
    axes[1].set_xlabel("predicted p_hat")
    axes[1].grid(True, alpha=0.3, linewidth=0.5)
    fig.tight_layout()
    return fig
