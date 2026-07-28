"""Lexis-surface figures: observed/forecast intensity heatmaps, period or cohort basis (05)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from seqeval.viz._style import new_fig

_AXIS_LABEL = {"year": "calendar year", "cohort": "birth cohort"}


def plot_lexis(
    surface: pd.DataFrame,
    *,
    dim: str = "year",
    value: str = "rate",
    mark_forecast: bool = True,
    outcome: str | None = None,
) -> Figure:
    """Heatmap of a Lexis surface: x = ``dim`` (calendar year or birth cohort), y = age, color rate.

    When ``mark_forecast`` and the surface carries a ``source`` column, a bold cyan **forecast
    frontier** line is drawn at the boundary between observed cells and model-forecast cells, so it
    is unmistakable where real data ends and forecasting begins.

    ``outcome`` names the event whose intensity is plotted (e.g. ``"first_birth"``); it is carried
    into the title and the colorbar so the surface is not just an anonymous "rate" — a Lexis
    surface of first births and one of second births look alike otherwise.
    """
    grid = surface.pivot_table(index="age_bin", columns=dim, values=value).sort_index()
    fig, ax = new_fig()
    ax.grid(False)
    xs = grid.columns.to_numpy()
    ages = grid.index.to_numpy()
    finite = grid.to_numpy()[np.isfinite(grid.to_numpy())]
    vmax = float(np.percentile(finite, 98)) if len(finite) else None
    mesh = ax.pcolormesh(xs, ages, grid.to_numpy(), shading="auto", cmap="magma", vmax=vmax)
    fig.colorbar(mesh, ax=ax, label=f"{outcome} {value}" if outcome else value)
    what = f" — {outcome}" if outcome else ""

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
        ax.set_title(f"Lexis surface{what}: observed data + model forecast")
    else:
        ax.set_title(f"Lexis surface{what}")
    ax.set_xlabel(_AXIS_LABEL.get(dim, dim))
    ax.set_ylabel("age (years)")
    return fig
