"""Kaplan-Meier curves with confidence bands (03 viz)."""

from __future__ import annotations

import pandas as pd
from matplotlib.figure import Figure

from seqeval.units import days_to_years
from seqeval.viz._style import new_fig, stratum_colors


def plot_km(
    km: pd.DataFrame,
    *,
    by: list[str] = (),
    title: str | None = None,
    xlabel: str = "age (years)",
) -> Figure:
    """Step-function KM curve(s) with CI bands — one line per stratum, x-axis in years.

    Parameters
    ----------
    km : pandas.DataFrame
        Output of :func:`seqeval.metrics.survival.kaplan_meier` (``time`` in days).
    by : list of str
        Stratum columns present in ``km`` (empty for a single curve).
    title : str, optional
        Axis title (e.g. the outcome label via ``bundle.label``).
    xlabel : str, default "age (years)"
        X-axis label. Durations measured from an ``origin`` event are elapsed time since that
        event, not age, so callers pass e.g. ``"years since first birth"``.
    """
    by = list(by)
    fig, ax = new_fig()
    strata = [((), km)] if not by else list(km.groupby(by, observed=True))
    colors = stratum_colors(len(strata))

    for (key, grp), color in zip(strata, colors, strict=True):
        grp = grp.sort_values("time")
        years = days_to_years(grp["time"].to_numpy())
        ax.step(years, grp["survival"], where="post", color=color, label=_label(by, key))
        if {"ci_lo", "ci_hi"} <= set(grp.columns):
            ax.fill_between(years, grp["ci_lo"], grp["ci_hi"], step="post", alpha=0.2, color=color)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("survival S(t)")
    ax.set_ylim(0, 1.02)
    if title:
        ax.set_title(title)
    if by:
        ax.legend(title=", ".join(by), fontsize=8)
    return fig


def _label(by: list[str], key) -> str:
    if not by:
        return "all"
    key_tuple = key if isinstance(key, tuple) else (key,)
    return ", ".join(f"{c}={v}" for c, v in zip(by, key_tuple, strict=True))
