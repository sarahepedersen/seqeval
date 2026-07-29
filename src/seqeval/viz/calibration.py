"""Calibration figures: the reliability diagram (04 viz)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from seqeval.metrics._disclosure import MIN_CELL
from seqeval.viz._style import SUPPRESSED_HATCH


def plot_reliability(
    calibration: pd.DataFrame,
    distribution: pd.DataFrame | None = None,
    *,
    title: str | None = None,
    min_cell: int = MIN_CELL,
) -> Figure:
    """Reliability diagram: binned observed frequency vs predicted probability, plus the diagonal.

    ``calibration`` is :func:`seqeval.metrics.ml.calibration_table` (columns ``bin, bin_left,
    bin_right, p_mean, y_rate, n``); with quantile binning each point rests on roughly the same
    number of persons. Only bins that hold someone are drawn.

    ``distribution`` is :func:`seqeval.metrics.ml.p_hat_distribution` — where the people actually
    are on the ``p̂`` grid. The lower panel draws that, one narrow bar per attainable value, rather
    than the calibration bins: a bin chosen to hold an equal share of people can span a wide stretch
    of the axis while nearly all its mass sits on one atom at the edge, and a full-width bar there
    reads as weight spread across a range that holds almost nobody. Without it the panel falls back
    to the bin counts, which is the old, misleading-but-better-than-nothing view.

    Withheld cells are hatched at the largest count their threshold allows rather than drawn as
    zero, so a suppressed spike is not read as an empty one.
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

    if distribution is not None and len(distribution):
        _draw_p_hat_spikes(axes[1], distribution, min_cell=min_cell)
    else:
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


def _draw_p_hat_spikes(ax, distribution: pd.DataFrame, *, min_cell: int) -> None:
    """One bar per attainable ``p̂``, centred on its own value and no wider than the grid step."""
    d = distribution.sort_values("p_hat")
    x = d["p_hat"].to_numpy().astype(float)
    step = float(np.median(np.diff(x))) if len(x) > 1 else 0.1
    width = 0.8 * step

    shown = d[~d["suppressed"].astype(bool)] if "suppressed" in d.columns else d
    ax.bar(
        shown["p_hat"].to_numpy().astype(float),
        shown["n_persons"].fillna(0).to_numpy().astype(float),
        width=width, align="center", color="tab:gray", edgecolor="white", linewidth=0.3,
    )
    if "suppressed" in d.columns:
        hidden = d[d["suppressed"].astype(bool)]
        if len(hidden):
            # "at most this many", never a false zero — the same convention as the ridge figures.
            ax.bar(
                hidden["p_hat"].to_numpy().astype(float),
                np.full(len(hidden), max(min_cell - 1, 0), dtype=float),
                width=width, align="center", facecolor="none", edgecolor="0.6",
                hatch=SUPPRESSED_HATCH, linewidth=0.4,
            )
