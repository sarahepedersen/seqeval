"""Replicate-variance figures: within-seed dispersion across the population (05 viz)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from seqeval.units import days_to_years
from seqeval.viz._style import stratum_colors

_LABELS = {
    "within_seed_var": "within-seed variance of completed births",
    "within_seed_cv": "within-seed CV of completed births",
    "timing_spread": "within-seed timing spread (days)",
}


def _fmt(col: str, value) -> str:
    if col in ("age_start", "age_stop"):
        return f"{days_to_years(int(value)):.0f}y jump-off"
    return str(value)


def plot_within_seed_variance(
    ind: pd.DataFrame,
    *,
    value: str = "within_seed_var",
    color_by: str = "age_stop",
    facet_by: str | None = None,
    title: str | None = None,
) -> Figure:
    """Population histogram of a per-person within-seed dispersion column.

    One overlaid histogram per ``color_by`` value (default the jump-off ``age_stop``), so people
    with heavier within-seed variance show up as mass to the right. ``facet_by`` (e.g. ``age_stop``)
    draws one panel per value, letting a subgroup ``color_by`` be read separately at each jump-off.
    """
    data = ind[np.isfinite(ind[value].to_numpy())]
    bins = np.histogram_bin_edges(data[value].to_numpy(), bins=20) if len(data) else 10
    facets = sorted(data[facet_by].unique()) if facet_by else [None]

    fig, axes = plt.subplots(
        len(facets), 1, figsize=(7, 2.6 * len(facets)), sharex=True, squeeze=False
    )
    for ax, fv in zip(axes[:, 0], facets, strict=True):
        sub = data if fv is None else data[data[facet_by] == fv]
        groups = list(sub.groupby(color_by, observed=True))
        for (key, grp), color in zip(groups, stratum_colors(len(groups)), strict=True):
            ax.hist(
                grp[value].to_numpy(), bins=bins, histtype="step", color=color,
                label=_fmt(color_by, key),
            )
        ax.set_ylabel("people")
        if fv is not None:
            ax.set_title(_fmt(facet_by, fv), fontsize=9, loc="left")
        if len(groups):
            ax.legend(fontsize=8)
    axes[-1, 0].set_xlabel(_LABELS.get(value, value))
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig
