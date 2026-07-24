"""Calibration figures: the reliability diagram (04 viz).

Seed-convergence of the scalar metrics is written as the ``convergence`` table only — it is a
sufficiency check to read off, not a figure worth carrying in the report.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from seqeval.viz._style import new_fig


def plot_reliability(
    calibration: pd.DataFrame, *, probs: pd.DataFrame | None = None, title: str | None = None
) -> Figure:
    """Reliability diagram: binned observed frequency vs predicted probability, plus the diagonal.

    ``calibration`` is :func:`seqeval.metrics.ml.calibration_table` (columns ``bin, bin_left,
    bin_right, p_mean, y_rate, n``); with quantile binning each point rests on the same number of
    persons. When ``probs`` is given (the run-level probability table), a ``p_hat`` histogram is
    drawn in a lower panel on the *same* bin edges as the curve, so the grouping is visible.
    """
    import matplotlib.pyplot as plt

    cal = calibration.sort_values("bin")
    edges = np.append(cal["bin_left"].to_numpy(), cal["bin_right"].to_numpy()[-1])

    if probs is None:
        fig, ax = new_fig((5.5, 5.5))
        axes = [ax]
    else:
        fig, axes = plt.subplots(2, 1, figsize=(5.5, 6.5), height_ratios=[3, 1], sharex=True)
        ax = axes[0]
        ax.grid(True, alpha=0.3, linewidth=0.5)

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
        axes[1].hist(probs["p_hat"].to_numpy(), bins=edges, color="tab:gray")
        axes[1].set_ylabel("count")
        axes[1].set_xlabel("predicted p_hat")
        axes[1].grid(True, alpha=0.3, linewidth=0.5)
    else:
        ax.set_xlabel("predicted p_hat")
    fig.tight_layout()
    return fig
