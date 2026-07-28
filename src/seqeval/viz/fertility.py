"""Fertility figures: ASFR age profiles, CCF by cohort, PPR bars (03 viz)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from seqeval.viz._style import new_fig, stratum_colors, stratum_key


def plot_asfr(asfr: pd.DataFrame, *, dim: str) -> Figure:
    """Age profile of ASFR, one line per period year or birth cohort.

    ``dim`` is the cell dimension column (``"year"`` for period, ``"cohort"`` for cohort);
    ``age_bin`` labels are already in years. ``dim`` is ordered, so the lines run along a sequential
    ramp and :func:`stratum_key` keys them by legend or colorbar depending on how many there are.
    """
    fig, ax = new_fig()
    groups = list(asfr.groupby(dim, observed=True))
    colors = stratum_colors(len(groups))
    for (key, grp), color in zip(groups, colors, strict=True):
        grp = grp.sort_values("age_bin")
        ax.plot(grp["age_bin"], grp["asfr"], color=color, label=str(key), lw=1.2)
    ax.set_xlabel("age (years)")
    ax.set_ylabel("age-specific fertility rate")
    ax.set_title(f"ASFR by {dim}")
    stratum_key(ax, [k for k, _ in groups], colors, label=dim)
    return fig


def plot_asfr_surface(asfr: pd.DataFrame, *, dim: str = "year") -> Figure:
    """Period ASFR as a heatmap surface: calendar year (x) vs age (y), rate as color.

    Far more legible than one line per year when there are many periods — it shows the fertility
    hump moving through calendar time as a continuous surface, the standard demographic view.
    """
    grid = asfr.pivot_table(index="age_bin", columns=dim, values="asfr").sort_index()
    fig, ax = new_fig()
    ax.grid(False)
    mesh = ax.pcolormesh(
        grid.columns.to_numpy(),
        grid.index.to_numpy(),
        grid.to_numpy(),
        shading="auto",
        cmap="magma",
    )
    fig.colorbar(mesh, ax=ax, label="age-specific fertility rate")
    ax.set_xlabel("calendar year")
    ax.set_ylabel("age (years)")
    ax.set_title("Period ASFR surface")
    return fig


def plot_tfr(tfr: pd.DataFrame) -> Figure:
    """Period total fertility rate over calendar years — the one-number summary of period ASFR."""
    fig, ax = new_fig()
    df = tfr.sort_values("year")
    ax.plot(df["year"], df["tfr"], "-", color="tab:blue")
    ax.set_xlabel("calendar year")
    ax.set_ylabel("total fertility rate")
    ax.set_title("Period TFR by year")
    return fig


def plot_ccf(ccf: pd.DataFrame) -> Figure:
    """CCF by birth cohort; incomplete cohorts drawn dashed so truncated means are visible."""
    fig, ax = new_fig()
    df = ccf.sort_values("cohort")
    ax.plot(df["cohort"], df["ccf"], "o-", color="tab:blue", label="complete")
    incomplete = df[~df["complete"].astype(bool)]
    if len(incomplete):
        ax.plot(incomplete["cohort"], incomplete["ccf"], "o--", color="tab:red", label="incomplete")
    ax.set_xlabel("birth cohort")
    ax.set_ylabel("completed cohort fertility")
    ax.set_title("CCF by cohort")
    ax.legend(fontsize=8)
    return fig


def plot_ppr(ppr: pd.DataFrame) -> Figure:
    """Parity progression ratios as a bar chart, one bar per parity transition."""
    fig, ax = new_fig()
    df = ppr.sort_values("parity_from")
    labels = [f"{a}→{b}" for a, b in zip(df["parity_from"], df["parity_to"], strict=True)]
    ax.bar(np.arange(len(df)), df["ppr"], color="tab:green")
    ax.set_xticks(np.arange(len(df)))
    ax.set_xticklabels(labels)
    ax.set_xlabel("parity transition")
    ax.set_ylabel("progression ratio")
    ax.set_ylim(0, 1.02)
    ax.set_title("Parity progression ratios")
    return fig
