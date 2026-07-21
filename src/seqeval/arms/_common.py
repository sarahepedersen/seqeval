"""Shared arm orchestration helpers: the :class:`OutputWriter` (03).

Arms produce tidy result frames and figures; the writer resolves their paths under
``output.dir/<arm>/``, stamps the ``model`` column into every frame (00 section 5.1 — cross-model
comparison is then a ``pd.concat`` over tidy tables), saves matplotlib figures, and records
everything written for the 06 manifest.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

_GEN_COLS = ["person_id", "seed", "age_start", "age_stop", "age", "event"]


def combine_prefix(observed: pd.DataFrame, gen_w: pd.DataFrame, t1: int, t2: int) -> pd.DataFrame:
    """Full life course per replicate: observed prefix (age <= t2) x seeds + the generated future.

    The generated file holds only ``age > t2`` rows; a framed outcome (absolute ordinal) or any
    aggregate metric needs each replicate's whole sequence. Replicating the observed prefix across
    the window's seeds and concatenating the generated future lets 02's evaluators and 03's metrics
    run unchanged on generated runs (shared by the backtesting and forecasting arms).
    """
    persons = gen_w["person_id"].unique()
    prefix = observed.loc[
        observed["person_id"].isin(persons) & (observed["age"] <= t2),
        ["person_id", "age", "event"],
    ]
    seeds = pd.DataFrame({"seed": gen_w["seed"].unique().astype(np.int32)})
    pref = prefix.merge(seeds, how="cross")
    pref["age_start"] = np.int32(t1)
    pref["age_stop"] = np.int32(t2)
    combined = pd.concat([pref[_GEN_COLS], gen_w[_GEN_COLS]], ignore_index=True)
    combined["event"] = combined["event"].astype("category")
    return combined


@dataclass
class OutputWriter:
    """Resolves paths, stamps the model column, saves frames/figures, and records the writes."""

    base_dir: Path
    arm: str
    model: str
    figure_format: str = "png"
    written: list[Path] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.base_dir = Path(self.base_dir)
        self.dir = self.base_dir / self.arm
        self.dir.mkdir(parents=True, exist_ok=True)

    def frame(self, name: str, df: pd.DataFrame) -> Path:
        """Save ``df`` as ``<name>.parquet`` with a leading ``model`` column; record and return."""
        stamped = df.copy()
        if "model" not in stamped.columns:
            stamped.insert(0, "model", self.model)
        path = self.dir / f"{name}.parquet"
        stamped.to_parquet(path, engine="pyarrow", index=False)
        self.written.append(path)
        return path

    def figure(self, name: str, fig: Figure) -> Path:
        """Save a matplotlib ``fig`` as ``<name>.<figure_format>``; close it; record and return."""
        path = self.dir / f"{name}.{self.figure_format}"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        self.written.append(path)
        return path
