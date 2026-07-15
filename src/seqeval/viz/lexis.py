"""Lexis-surface figures: observed/forecast intensity heatmaps, period or cohort basis (05)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from seqeval.viz._style import new_fig

_AXIS_LABEL = {"year": "calendar year", "cohort": "birth cohort"}


def plot_lexis(
    surface: pd.DataFrame, *, dim: str = "year", value: str = "rate", mark_forecast: bool = True
) -> Figure:
    """Heatmap of a Lexis surface: x = ``dim`` (calendar year or birth cohort), y = age, color rate.

    When ``mark_forecast`` and the surface carries a ``source`` column, a bold cyan **forecast
    frontier** line is drawn at the boundary between observed cells and model-forecast cells, so it
    is unmistakable where real data ends and forecasting begins.
    """
    grid = surface.pivot_table(index="age_bin", columns=dim, values=value).sort_index()
    fig, ax = new_fig()
    ax.grid(False)
    xs = grid.columns.to_numpy()
    ages = grid.index.to_numpy()
    finite = grid.to_numpy()[np.isfinite(grid.to_numpy())]
    vmax = float(np.percentile(finite, 98)) if len(finite) else None
    mesh = ax.pcolormesh(xs, ages, grid.to_numpy(), shading="auto", cmap="magma", vmax=vmax)
    fig.colorbar(mesh, ax=ax, label=value)

    if mark_forecast and "source" in surface.columns:
        is_fc = (
            surface.assign(_f=(surface["source"] == "forecast").astype(float))
            .pivot_table(index="age_bin", columns=dim, values="_f", fill_value=0.0)
            .reindex(index=grid.index, columns=grid.columns, fill_value=0.0)
        )
        if is_fc.to_numpy().any():
            ax.contour(xs, ages, is_fc.to_numpy(), levels=[0.5], colors="cyan", linewidths=2.0)
            ax.plot([], [], color="cyan", lw=2.0, label="forecast frontier (data ends here)")
            ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
        ax.set_title("Lexis surface: observed data + model forecast")
    else:
        ax.set_title("Lexis surface")
    ax.set_xlabel(_AXIS_LABEL.get(dim, dim))
    ax.set_ylabel("age (years)")
    return fig


def plot_lexis_uncertainty(
    surfaces_by_seed: pd.DataFrame, *, dim: str = "year", value: str = "rate"
) -> Figure:
    """Heatmap of the across-seed IQR (q75 - q25) of a forecast Lexis surface per cell.

    ``surfaces_by_seed`` is the per-seed forecast surface (carries a ``seed`` column); the spread
    across seeds in each cell is the forecast's seed uncertainty, surfaced spatially.
    """
    iqr = (
        surfaces_by_seed.groupby([dim, "age_bin"], observed=True)[value]
        .agg(lambda s: s.quantile(0.75) - s.quantile(0.25))
        .reset_index(name="iqr")
    )
    grid = iqr.pivot_table(index="age_bin", columns=dim, values="iqr").sort_index()
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
    ax.set_xlabel(_AXIS_LABEL.get(dim, dim))
    ax.set_ylabel("age (years)")
    ax.set_title("Forecast seed uncertainty (IQR)")
    return fig
