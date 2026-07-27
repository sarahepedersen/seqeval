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


def stratum_colors(n: int, *, lo: float = 0.0, hi: float = 1.0) -> list:
    """``n`` evenly-spaced colors from the shared colormap, for one line per stratum.

    ``lo``/``hi`` restrict the span of the ramp. The default is the whole map; pass a narrowed
    range (e.g. ``lo=0.1, hi=0.85``) for line work on white, where viridis's pale yellow end is
    too low-contrast to read as a curve.
    """
    if n <= 1:
        return [_CMAP((lo + hi) / 2)]
    return [_CMAP(lo + (hi - lo) * i / (n - 1)) for i in range(n)]
