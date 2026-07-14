"""Fertility figures: ASFR age profiles, CCF by cohort, PPR bars (03 viz)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from seqeval.viz._style import new_fig, stratum_colors


def plot_asfr(asfr: pd.DataFrame, *, dim: str) -> Figure:
    """Age profile of ASFR, one line per period year or birth cohort.

    ``dim`` is the cell dimension column (``"year"`` for period, ``"cohort"`` for cohort);
    ``age_bin`` labels are already in years.
    """
    fig, ax = new_fig()
    groups = list(asfr.groupby(dim, observed=True))
    for (key, grp), color in zip(groups, stratum_colors(len(groups)), strict=True):
        grp = grp.sort_values("age_bin")
        ax.plot(grp["age_bin"], grp["asfr"], color=color, label=str(key), lw=1.2)
    ax.set_xlabel("age (years)")
    ax.set_ylabel("age-specific fertility rate")
    ax.set_title(f"ASFR by {dim}")
    if len(groups) <= 12:
        ax.legend(title=dim, fontsize=7, ncol=2)
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
