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

#: Default noun for the quantity being counted. The dispersion is over a *count of one event*, and
#: which event is configurable (``forecasting.replicate_variance.event``), so the callers pass the
#: event's natural-language plural and these strings only supply the surrounding words.
DEFAULT_EVENT_LABEL = "events"

_LABELS = {
    "within_seed_var": "within-seed variance of completed {noun}",
    "within_seed_cv": "within-seed CV of completed {noun}",
    "timing_spread": "within-seed timing spread (days)",
    "count": "completed {noun} per person",
}


def _label(key: str, event_label: str) -> str:
    """A y-axis label for ``key``, with the counted event's name filled in where one belongs."""
    return _LABELS.get(key, key).format(noun=event_label)

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
    event_label: str = DEFAULT_EVENT_LABEL,
) -> Figure:
    """Distribution of a per-person within-seed dispersion column, one column per ``x`` group.

    ``event_label`` names what is being counted (``"births"``, ``"marriages"``) and appears in the
    y-axis label; the dispersion itself is over a count of whichever event the run configured.

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
        ax.set_ylabel(_label(value, event_label), fontsize=8)
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


def plot_within_seed_quantile_fan(
    summary: pd.DataFrame,
    *,
    x: str = "age_stop",
    facet_by: str | None = None,
    title: str | None = None,
    event_label: str = DEFAULT_EVENT_LABEL,
) -> Figure:
    """Group-mean five-number summary of a completed count as a fan, one point per ``x`` group.

    ``summary`` is :func:`~seqeval.metrics.dispersion.quantile_summary` output. The line is
    ``mean_q50``, the dark band ``mean_q25``–``mean_q75`` and the light band
    ``mean_q0``–``mean_q100``: the *typical person's* median, interquartile spread and full
    replicate range, averaged over the group. It is not the population's spread — a wide fan means
    individuals' replicates disagree, not that individuals differ from each other.

    Suppressed groups carry NA means and so break the line, leaving a visible gap rather than a
    straight segment drawn through a group that was withheld. ``mean_q0``/``mean_q100`` widen with
    the replicate count, so the outer band is only comparable across groups at equal ``mean_k``;
    the caption on the table carries that warning.
    """
    facets = sorted(summary[facet_by].dropna().unique()) if facet_by else [None]
    fig, axes = plt.subplots(
        len(facets), 1, figsize=(7, 3.2 * len(facets)), sharex=True, squeeze=False,
    )
    for ax, fv in zip(axes[:, 0], facets, strict=True):
        sub = summary if fv is None else summary[summary[facet_by] == fv]
        sub = sub.sort_values(x)
        pos = range(len(sub))
        ax.fill_between(
            pos, sub["mean_q0"], sub["mean_q100"], alpha=0.18, color="C0", linewidth=0,
            label="mean min–max",
        )
        ax.fill_between(
            pos, sub["mean_q25"], sub["mean_q75"], alpha=0.38, color="C0", linewidth=0,
            label="mean IQR",
        )
        ax.plot(pos, sub["mean_q50"], color="C0", marker="o", markersize=3, label="mean median")
        ax.set_xticks(list(pos))
        # the x axis is already labelled with the unit, so the ticks carry the bare value —
        # matching the ridge figure, which the reader sees directly above this one
        labels = (
            [f"{days_to_years(int(v)):.0f}" for v in sub[x]]
            if x in ("age_start", "age_stop")
            else [str(v) for v in sub[x]]
        )
        ax.set_xticklabels(labels, fontsize=8)
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.set_ylabel(_label("count", event_label), fontsize=8)
        ax.set_ylim(bottom=0)
        if fv is not None:
            ax.set_title(_fmt(facet_by, fv), fontsize=9, loc="left")
    axes[0, 0].legend(fontsize=7, frameon=False)
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
