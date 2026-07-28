"""Shared figure style: one place for figsize, fonts, and the colormap (03 viz).

All viz functions return a ``Figure`` and never save it (the arm saves via ``OutputWriter``). Axes
and labels are in **years** — viz is one of the three sanctioned unit-conversion sites (00 section
3); it applies :func:`seqeval.units.days_to_years` at plot time and never mutates result tables.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.figure import Figure

FIGSIZE = (7.0, 4.5)
_CMAP = plt.get_cmap("viridis")

#: Above this many strata a per-line legend stops fitting and a colorbar is used instead.
MAX_LEGEND_ENTRIES = 12


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


def stratum_key(ax, keys: list, colors: list, *, label: str, max_ticks: int = 10) -> None:
    """Identify one-line-per-stratum curves: a legend when few strata, a colorbar when many.

    ``keys`` are the stratum values in the order :func:`stratum_colors` assigned ``colors`` to.
    Because colors track rank rather than value, the colorbar is a discrete band per stratum with
    ticks labelled by the values themselves, so unevenly spaced strata stay honest. At most
    ``max_ticks`` are labelled; the rest are read off the ramp's order.
    """
    n = len(keys)
    if not n:
        return
    if n <= MAX_LEGEND_ENTRIES:
        ax.legend(title=label, fontsize=7, ncol=2)
        return
    mappable = ScalarMappable(
        norm=BoundaryNorm(np.arange(n + 1) - 0.5, n), cmap=ListedColormap(colors)
    )
    ticks = np.arange(0, n, max(1, -(-n // max_ticks)))
    bar = ax.figure.colorbar(mappable, ax=ax, ticks=ticks, label=label)
    bar.ax.set_yticklabels([str(keys[i]) for i in ticks], fontsize=7)
