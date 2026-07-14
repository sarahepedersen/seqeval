"""Lexis-surface figures: observed/forecast intensity heatmaps over (calendar year x age) (05)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from seqeval.viz._style import new_fig


def plot_lexis(surface: pd.DataFrame, *, value: str = "rate", mark_forecast: bool = True) -> Figure:
    """Heatmap of a Lexis surface: x = calendar year, y = age, color = intensity.

    When ``mark_forecast`` and the surface carries a ``source`` column, the forecast region
    (``source == "forecast"``) is delineated with a hatch overlay so the model-completed
    upper-right triangle is visually distinct from the observed data.
    """
    grid = surface.pivot_table(index="age_bin", columns="year", values=value).sort_index()
    fig, ax = new_fig()
    ax.grid(False)
    years = grid.columns.to_numpy()
    ages = grid.index.to_numpy()
    # robust upper limit so a few sparse-exposure boundary cells don't wash out the surface
    finite = grid.to_numpy()[np.isfinite(grid.to_numpy())]
    vmax = float(np.percentile(finite, 98)) if len(finite) else None
    mesh = ax.pcolormesh(years, ages, grid.to_numpy(), shading="auto", cmap="magma", vmax=vmax)
    fig.colorbar(mesh, ax=ax, label=value)

    if mark_forecast and "source" in surface.columns:
        fc = surface[surface["source"] == "forecast"].pivot_table(
            index="age_bin", columns="year", values=value
        )
        fc = fc.reindex(index=grid.index, columns=grid.columns)
        # hatch the forecast cells (where a forecast value exists)
        ax.pcolor(
            years,
            ages,
            np.where(np.isfinite(fc.to_numpy()), 1.0, np.nan),
            hatch="///",
            alpha=0.0,
            edgecolor="white",
            linewidth=0.0,
        )
        ax.set_title("Lexis surface (hatched = model forecast)")
    else:
        ax.set_title("Lexis surface")
    ax.set_xlabel("calendar year")
    ax.set_ylabel("age (years)")
    return fig


def plot_lexis_uncertainty(surfaces_by_seed: pd.DataFrame, *, value: str = "rate") -> Figure:
    """Heatmap of the across-seed IQR (q75 - q25) of a forecast Lexis surface per cell.

    ``surfaces_by_seed`` is the per-seed forecast surface (carries a ``seed`` column); the spread
    across seeds in each (year, age) cell is the forecast's seed uncertainty, surfaced spatially.
    """
    iqr = (
        surfaces_by_seed.groupby(["year", "age_bin"], observed=True)[value]
        .agg(lambda s: s.quantile(0.75) - s.quantile(0.25))
        .reset_index(name="iqr")
    )
    grid = iqr.pivot_table(index="age_bin", columns="year", values="iqr").sort_index()
    fig, ax = new_fig()
    ax.grid(False)
    mesh = ax.pcolormesh(
        grid.columns.to_numpy(),
        grid.index.to_numpy(),
        grid.to_numpy(),
        shading="auto",
        cmap="viridis",
    )
    fig.colorbar(mesh, ax=ax, label=f"{value} IQR across seeds")
    ax.set_xlabel("calendar year")
    ax.set_ylabel("age (years)")
    ax.set_title("Forecast seed uncertainty (IQR)")
    return fig
