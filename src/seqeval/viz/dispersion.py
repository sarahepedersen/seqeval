"""Replicate-variance figures: within-seed dispersion across the population (05 viz).

Drawn from :func:`~seqeval.metrics.dispersion.dispersion_distribution` — binned counts, never the
per-person frame — so these figures carry no individual and survive a restricted run.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from seqeval.metrics._disclosure import MIN_CELL
from seqeval.units import days_to_years
from seqeval.viz._ridge import draw_distribution_columns

_LABELS = {
    "within_seed_var": "within-seed variance of completed births",
    "within_seed_cv": "within-seed CV of completed births",
    "timing_spread": "within-seed timing spread (days)",
}

_AXIS_LABELS = {
    "cohort": "birth cohort",
    "age_stop": "jump-off (years)",
    "age_start": "window start (years)",
}


def _fmt(col: str, value) -> str:
    if col in ("age_start", "age_stop"):
        return f"{days_to_years(int(value)):.0f}y jump-off"
    if col == "cohort":
        return f"{int(value)} cohort"
    return str(value)


def plot_within_seed_variance(
    dist: pd.DataFrame,
    *,
    value: str = "within_seed_var",
    x: str = "cohort",
    facet_by: str | None = None,
    title: str | None = None,
    min_cell: int = MIN_CELL,
) -> Figure:
    """Distribution of a per-person within-seed dispersion column, one column per ``x`` group.

    ``dist`` is :func:`~seqeval.metrics.dispersion.dispersion_distribution` output, grouped by ``x``
    (and ``facet_by`` when given). Each column is one group's distribution: the dispersion runs up
    the y axis and bar width is the share of that group's people in the bin, so a small cohort is
    comparable to a large one and mass high on the axis means people whose replicates disagree.

    Laid out to match the outcome-uncertainty figure — group along x, the varying quantity along y —
    so the two can be read with the same habit. ``facet_by`` gives one panel per value, letting the
    same cohorts be compared as the jump-off moves.

    Bins are shared across every group and panel so the shapes are comparable, but each panel is
    scaled to the range its own people occupy — a jump-off where nobody exceeds a small variance
    would otherwise be squashed by the panel with the longest tail. Nothing is cut: the tail sets
    the top of its own panel. Read the shrinkage off the axis numbers, not the bar heights.

    Withheld bars are hatched at the widest their threshold allows rather than dropped, so a thin
    tail reads as thin instead of absent.
    """
    facets = sorted(dist[facet_by].unique()) if facet_by else [None]
    fig, axes = plt.subplots(
        len(facets), 1, figsize=(7, 3.6 * len(facets)), sharex=True, squeeze=False,
    )
    for ax, fv in zip(axes[:, 0], facets, strict=True):
        sub = dist if fv is None else dist[dist[facet_by] == fv]
        draw_distribution_columns(
            ax, sub, x=x, lo="bin_lo", hi="bin_hi", count="n_persons", total="n_group",
            min_cell=min_cell,
        )
        ax.set_ylim(*_occupied_range(sub))
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.set_ylabel(_LABELS.get(value, value), fontsize=8)
        if fv is not None:
            ax.set_title(_fmt(facet_by, fv), fontsize=9, loc="left")
    if x in ("age_start", "age_stop"):
        ticks = axes[-1, 0].get_xticks()
        axes[-1, 0].set_xticklabels([f"{days_to_years(int(t)):.0f}" for t in ticks])
    axes[-1, 0].set_xlabel(_AXIS_LABELS.get(x, x))
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


def _occupied_range(cells: pd.DataFrame) -> tuple[float, float]:
    """The y span the panel's people occupy: empty bins trimmed, every occupied one kept.

    Only bins nobody landed in are dropped. A long thin tail is part of the distribution — a few
    people whose replicates disagree wildly is a finding, not clutter — so it sets the axis rather
    than being cut off. A withheld cell counts as occupied; it holds people, just too few to
    publish.
    """
    used = cells[(cells["n_persons"].fillna(0) > 0) | cells["suppressed"]]
    if used.empty:
        return float(cells["bin_lo"].min()), float(cells["bin_hi"].max())
    lo, hi = float(used["bin_lo"].min()), float(used["bin_hi"].max())
    pad = max((hi - lo) * 0.06, 1e-9)
    return lo - pad, hi + pad
