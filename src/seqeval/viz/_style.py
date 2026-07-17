"""Shared figure style: one place for figsize, fonts, and the colormap (03 viz).

All viz functions return a ``Figure`` and never save it (the arm saves via ``OutputWriter``). Axes
and labels are in **years** — viz is one of the three sanctioned unit-conversion sites (00 section
3); it applies :func:`seqeval.units.days_to_years` at plot time and never mutates result tables.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

FIGSIZE = (7.0, 4.5)
_CMAP = plt.get_cmap("viridis")


def new_fig(figsize: tuple[float, float] = FIGSIZE) -> tuple[Figure, plt.Axes]:
    """A single-axes figure with the house style applied."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    return fig, ax


def stratum_colors(n: int) -> list:
    """``n`` evenly-spaced colors from the shared colormap, for one line per stratum."""
    if n <= 1:
        return [_CMAP(0.5)]
    return [_CMAP(i / (n - 1)) for i in range(n)]
