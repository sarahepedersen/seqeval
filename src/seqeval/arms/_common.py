"""Shared arm orchestration helpers: the :class:`OutputWriter` (03).

Arms produce tidy result frames and figures; the writer resolves their paths under
``output.dir/<arm>/``, stamps the ``model`` column into every frame (00 section 5.1 — cross-model
comparison is then a ``pd.concat`` over tidy tables), saves matplotlib figures, and records
everything written for the 06 manifest.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from seqeval.metrics._disclosure import MIN_CELL, apply_policy, assert_publishable

logger = logging.getLogger("seqeval")

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


def pool_seeds(
    frame: pd.DataFrame, persons: pd.DataFrame | None = None, *, seed_col: str = "seed"
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """One synthetic population of N×K trajectories: every ``(person_id, seed)`` becomes a person.

    Re-keys ``person_id`` to a dense integer id per trajectory, keeping the original as
    ``source_person_id``. Every metric that groups by ``person_id`` — Kaplan-Meier, PPR, ASFR — then
    treats each trajectory as its own individual with no change to the metric itself, which is what
    "pool the generated sequences" means: no per-person aggregate underneath the estimate.

    ``persons`` is expanded the same way when given, so the ``birth_year``/cohort merges inside the
    fertility metrics still resolve. The id map is built from the sorted pairs, so a rerun on the
    same inputs produces the same ids.
    """
    units = (
        frame[["person_id", seed_col]]
        .drop_duplicates()
        .sort_values(["person_id", seed_col], kind="stable")
        .reset_index(drop=True)
    )
    units["unit_id"] = np.arange(len(units), dtype=np.int64)

    pooled = frame.merge(units, on=["person_id", seed_col], how="left")
    pooled = pooled.rename(columns={"person_id": "source_person_id", "unit_id": "person_id"})

    if persons is None:
        return pooled, None
    persons_pooled = persons.merge(units, on="person_id", how="inner")
    persons_pooled = persons_pooled.rename(
        columns={"person_id": "source_person_id", "unit_id": "person_id"}
    )
    return pooled, persons_pooled


@dataclass
class OutputWriter:
    """Resolves paths, stamps the model column, saves frames/figures, and records the writes.
    """

    base_dir: Path
    arm: str
    model: str
    figure_format: str = "png"
    # Both default to the restrictive setting, matching `OutputConfig`: a writer built without an
    # explicit policy withholds per-person output and suppresses thin cells.
    individual_level: bool = False
    min_cell: int = MIN_CELL
    written: list[Path] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.base_dir = Path(self.base_dir)
        self.dir = self.base_dir / self.arm
        self.dir.mkdir(parents=True, exist_ok=True)

    def suppress(self, name: str, df: pd.DataFrame) -> pd.DataFrame:
        """Apply ``name``'s disclosure policy — call before drawing a figure from ``df``.
        """
        return apply_policy(name, df, min_cell=self.min_cell)

    def frame(self, name: str, df: pd.DataFrame, *, individual: bool = False) -> Path | None:
        """Save ``df`` as ``<name>.parquet`` with a leading ``model`` column; record and return.

        The single ``to_parquet`` in the package, and so where the disclosure policy is enforced:
        the frame is suppressed (again, harmlessly, if the arm already did it) and then checked.
        """
        if self._withheld(name, individual):
            return None
        stamped = self.suppress(name, df).copy()
        if "model" not in stamped.columns:
            stamped.insert(0, "model", self.model)
        assert_publishable(name, stamped, min_cell=self.min_cell)
        path = self.dir / f"{name}.parquet"
        stamped.to_parquet(path, engine="pyarrow", index=False)
        self.written.append(path)
        return path

    def figure(self, name: str, fig: Figure, *, individual: bool = False) -> Path | None:
        """Save a matplotlib ``fig`` as ``<name>.<figure_format>``; close it; record and return."""
        if self._withheld(name, individual):
            plt.close(fig)  # the figure was built before we knew; do not leak it
            return None
        path = self.dir / f"{name}.{self.figure_format}"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        self.written.append(path)
        return path

    def withhold(self, name: str) -> None:
        """Record an output the caller suppressed itself, so the manifest still names it.

        Used for suppressing individual-level output that is skipped before it is built.
        """
        if name not in self.skipped:
            # Debug, not a warning: withholding is what the run was asked to do. The manifest's
            # per-arm `withheld` list is the record.
            logger.debug("%s: %s withheld — output.individual_level is false", self.arm, name)
            self.skipped.append(name)

    def _withheld(self, name: str, individual: bool) -> bool:
        """Whether this output is per-person on a run that publishes none, and note it if so."""
        if not individual or self.individual_level:
            return False
        self.withhold(name)
        return True
