"""Fertility figures (03 viz): every stratified curve is identified, at any number of strata."""

from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

from seqeval.viz.fertility import plot_asfr  # noqa: E402


def _asfr(cohorts) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"cohort": c, "age_bin": a, "asfr": 0.2 * np.exp(-((a - 28) ** 2) / 40)}
            for c in cohorts
            for a in range(15, 50)
        ]
    )


def test_few_cohorts_are_keyed_by_legend():
    ax = plot_asfr(_asfr(range(1940, 1948)), dim="cohort").axes[0]
    legend = ax.get_legend()
    assert legend is not None
    assert [t.get_text() for t in legend.get_texts()] == [str(c) for c in range(1940, 1948)]


def test_many_cohorts_are_keyed_by_colorbar():
    """Past the legend limit the key becomes a colorbar — never nothing at all."""
    cohorts = list(range(1940, 1970))
    fig = plot_asfr(_asfr(cohorts), dim="cohort")
    ax = fig.axes[0]
    assert ax.get_legend() is None
    assert len(fig.axes) == 2  # the colorbar is the second axes
    labels = [t.get_text() for t in fig.axes[1].get_yticklabels()]
    assert set(labels) <= {str(c) for c in cohorts}
    assert labels[0] == "1940"  # ticks are labelled with cohort values, not ranks


def test_colorbar_ticks_track_the_line_colours():
    """Colour is assigned by rank, so tick n must carry the colour of the nth curve."""
    cohorts = list(range(1940, 1970))
    fig = plot_asfr(_asfr(cohorts), dim="cohort")
    ramp = fig.axes[1].collections[0].cmap
    for line, c in zip(fig.axes[0].lines, np.linspace(0, 1, len(cohorts)), strict=True):
        np.testing.assert_allclose(line.get_color(), ramp(c), atol=0.01)
