"""Sequence-descriptive figures: the age profile of a predicted event (05 viz).

Drawn from :func:`~seqeval.metrics.sequences.event_age_distribution` — binned counts over
person-years, never a per-person frame — so the figure carries no individual and survives a
restricted run.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from seqeval.units import days_to_years
from seqeval.viz._style import new_fig, stratum_colors

__all__ = ["plot_event_age_distribution", "plot_token_frequency"]


def plot_event_age_distribution(
    dist: pd.DataFrame,
    *,
    label: str,
    title: str | None = None,
    value: str = "rate",
) -> Figure:
    """One panel per jump-off: the observed age profile of an event under the generated one.

    ``dist`` is every row of :func:`~seqeval.metrics.sequences.event_age_distribution` for a single
    alias — both sources, every window. The panels come from the windows in the frame, so the
    figure describes whatever the run produced rather than anything passed alongside it.

    **Small multiples rather than one axes.** Both sources are restricted to the same post-jump-off
    window, so an observed curve belongs to *its* jump-off; drawing all of them together would put
    three differently-truncated observed curves on one axes and invite the reader to compare them
    with each other. One panel per jump-off keeps every comparison like-for-like.

    ``value`` defaults to ``rate`` — events per person-year — because observed records stop at the
    observation year while generated trajectories run to the end of the fertile range, so the two
    populations carry different exposure at older ages. Passing ``share`` draws the composition
    instead, which describes one profile's shape but is not comparable across the two.

    A withheld bin carries NA and so breaks the line, leaving a visible gap rather than a segment
    drawn through a cell that was suppressed — the same convention as the quantile fan.
    """
    windows = sorted(dist["age_stop"].dropna().unique())
    fig, axes = (
        new_fig() if not windows else _panels(len(windows))
    )
    if not windows:
        return fig

    colors = stratum_colors(len(windows), lo=0.1, hi=0.85)
    for ax, t2, color in zip(axes, windows, colors, strict=True):
        panel = dist[dist["age_stop"] == t2]
        jumpoff = days_to_years(int(t2))
        for source, style in (("observed", {"color": "black", "lw": 1.4}),
                              ("generated", {"color": color, "lw": 1.6})):
            rows = panel[panel["source"] == source].sort_values("age_bin")
            if rows.empty:
                continue
            ax.plot(rows["age_bin"], rows[value], label=source, **style)
        # The generated side cannot produce anything left of here, and neither side is counted
        # there — the rule says so rather than leaving the reader to infer it from where the lines
        # begin.
        ax.axvline(jumpoff, color="0.4", lw=0.8, ls=":", zorder=1)
        ax.set_ylabel(_YLABELS.get(value, value), fontsize=8)
        ax.set_title(f"jump-off {jumpoff:.0f}y", fontsize=9, loc="left")
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.set_ylim(bottom=0)

    axes[0].legend(fontsize=7, frameon=False)
    axes[-1].set_xlabel("age (years)")
    if title:
        fig.suptitle(f"{title} — {label}")
    fig.tight_layout()
    return fig


#: Y-axis wording per plotted column; anything else falls through to the column name.
_YLABELS = {
    "rate": "events per person-year",
    "share": "share of the event's occurrences",
}


def _panels(n: int):
    """``n`` stacked panels sharing an x axis, in the house style."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(n, 1, figsize=(7, 2.6 * n), sharex=True, squeeze=False)
    return fig, list(axes[:, 0])


def plot_token_frequency(freq: pd.DataFrame, *, label: str, title: str | None = None) -> Figure:
    """How many times the token is predicted in each birth cohort, one bar group per cohort.

    ``freq`` is every :func:`~seqeval.metrics.sequences.token_frequency` row for a single alias.
    The all-cohorts row (``cohort`` null) is the total and is left out — it would dwarf the bars it
    sits beside — so the figure is the breakdown and the table carries the total.

    Bars are counts across every trajectory, so a cohort with more people or more replicates has a
    taller bar for that reason alone. The share columns in the table are the like-for-like reading;
    this is the volume one.
    """
    rows = freq[freq["cohort"].notna()]
    fig, ax = new_fig()
    if rows.empty:
        return fig

    cohorts = sorted(rows["cohort"].unique())
    jumpoffs = sorted(rows["age_stop"].dropna().unique())
    colors = stratum_colors(len(jumpoffs), lo=0.1, hi=0.85)
    positions = np.arange(len(cohorts), dtype=float)
    width = 0.8 / max(len(jumpoffs), 1)

    for i, (t2, color) in enumerate(zip(jumpoffs, colors, strict=True)):
        counts = (
            rows[rows["age_stop"] == t2].set_index("cohort")["n_events"].reindex(cohorts)
        )
        ax.bar(
            positions + (i - (len(jumpoffs) - 1) / 2) * width,
            pd.to_numeric(counts, errors="coerce").to_numpy(dtype=float),
            width=width,
            color=color,
            label=f"jump-off {days_to_years(int(t2)):.0f}y",
        )

    ax.set_xticks(positions)
    ax.set_xticklabels([f"{int(c)}" for c in cohorts])
    ax.set_xlabel("birth cohort")
    ax.set_ylabel(f"{label} predicted (count over all trajectories)")
    ax.legend(fontsize=7, frameon=False)
    if title:
        ax.set_title(f"{title} — {label}", fontsize=10)
    fig.tight_layout()
    return fig
