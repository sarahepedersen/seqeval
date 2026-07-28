"""Ridge (joyplot) drawing shared by the figures built on binned counts.

One convention, so a reader learns it once: each row is a group's own distribution drawn as a share
of that group's people, rows are stacked bottom-to-top in group order, and a cell withheld by
small-cell suppression is hatched at the largest count it could hold rather than drawn flat.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from seqeval.metrics._disclosure import MIN_CELL
from seqeval.viz._style import SUPPRESSED_HATCH, stratum_colors


def draw_ridges(
    ax,
    cells: pd.DataFrame,
    *,
    row: str,
    lo: str,
    hi: str,
    count: str,
    total: str,
    label_fn: Callable[[object, int], str],
    x_transform: Callable[[np.ndarray], np.ndarray] = lambda x: x,
    overlap: float = 0.6,
    ylabel: str | None = None,
    min_cell: int = MIN_CELL,
) -> list:
    """Stack one filled density per ``row`` group on ``ax``; return the baselines drawn.

    ``cells`` is a binned-counts table: one row per (group, bin) with edges ``lo``/``hi``, the cell
    count ``count``, the group total ``total``, and a boolean ``suppressed``. Heights are
    ``count / total``, so groups of different sizes stay comparable, scaled so the tallest ridge
    spans ``1/(1 - overlap)`` baselines. Ridges are drawn top-first, so each occludes the one above
    it and never the one below.
    """
    if cells.empty:
        return []
    groups = sorted(cells[row].dropna().unique())
    colors = stratum_colors(len(groups), lo=0.1, hi=0.85)
    step, scale = 1.0, 1.0 / (1.0 - overlap)
    ticks, labels = [], []

    for i, (key, color) in enumerate(zip(groups, colors, strict=True)):
        sub = cells[cells[row] == key].sort_values(lo)
        base = i * step
        n_group = int(sub[total].iloc[0])
        x = x_transform(np.append(sub[lo].to_numpy(), sub[hi].to_numpy()[-1]))
        share = sub[count].fillna(0).to_numpy().astype(float) / max(n_group, 1)
        height = np.append(share, share[-1]) * scale
        ax.fill_between(x, base, base + height, step="post", color=color, alpha=0.85, lw=0)
        ax.step(x, base + height, where="post", color=color, lw=0.8)
        _hatch_withheld(
            ax, sub, base, n_group, scale,
            lo=lo, hi=hi, x_transform=x_transform, min_cell=min_cell,
        )
        ticks.append(base)
        labels.append(label_fn(key, n_group))

    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=7)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.set_ylim(-0.1 * step, (len(groups) - 1) * step + scale * 1.05)
    return ticks


def _hatch_withheld(ax, sub, base, n_group, scale, *, lo, hi, x_transform, min_cell) -> None:
    """Withheld cells at their upper bound: 'at most this many', never a false zero."""
    hidden = sub[sub["suppressed"]]
    if hidden.empty:
        return
    cap = max(min_cell - 1, 0) / max(n_group, 1) * scale
    left = x_transform(hidden[lo].to_numpy())
    right = x_transform(hidden[hi].to_numpy())
    ax.bar(
        left, cap, width=right - left, bottom=base, align="edge",
        facecolor="none", edgecolor="0.6", hatch=SUPPRESSED_HATCH, lw=0.4,
    )


def draw_distribution_columns(
    ax,
    cells: pd.DataFrame,
    *,
    x: str,
    lo: str,
    hi: str,
    count: str,
    total: str,
    min_cell: int = MIN_CELL,
    color: str = "tab:blue",
) -> np.ndarray:
    """One vertical distribution per ``x`` group: bars along y, width proportional to share.

    The same binned-counts table :func:`draw_ridges` takes, drawn the other way round — the binned
    quantity runs up the y axis and each group occupies a column at its own x position. Widths are
    within-group shares scaled so the widest bar in the figure fills most of the gap between
    columns, so groups stay comparable and never overlap. Returns the x positions drawn.

    Every bar grows rightward from its group's tick rather than being centred on it, so within a
    column the bins share a baseline and their lengths compare directly; a centred bar makes two
    nearby shares look alike.
    """
    if cells.empty:
        return np.empty(0)
    groups = np.array(sorted(cells[x].dropna().unique()))
    spacing = float(np.min(np.diff(groups))) if len(groups) > 1 else 1.0
    shares = cells[count] / cells[total]
    unit = 0.8 * spacing / float(shares.max() or 1.0)

    for key in groups:
        sub = cells[cells[x] == key].sort_values(lo)
        n_group = int(sub[total].iloc[0])
        height = (sub[hi] - sub[lo]).to_numpy()
        width = (sub[count].fillna(0).to_numpy().astype(float) / max(n_group, 1)) * unit
        ax.bar(
            np.full(len(width), key), height, width=width, bottom=sub[lo].to_numpy(),
            align="edge", color=color, alpha=0.45, zorder=2,
        )
        hidden = sub[sub["suppressed"]]
        if len(hidden):
            cap = max(min_cell - 1, 0) / max(n_group, 1) * unit
            ax.bar(
                np.full(len(hidden), key),
                (hidden[hi] - hidden[lo]).to_numpy(),
                width=cap, bottom=hidden[lo].to_numpy(), align="edge",
                facecolor="none", edgecolor="0.6", hatch=SUPPRESSED_HATCH, lw=0.4, zorder=2,
            )
    ax.set_xticks(groups)
    ax.set_xlim(float(groups.min()) - 0.15 * spacing, float(groups.max()) + 1.0 * spacing)
    return groups
