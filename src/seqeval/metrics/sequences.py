"""Basic sequence descriptives: what the model predicts, how often, and at what age (05).

The other forecasting metrics look at generated output through a demographic lens — a Lexis surface
for one outcome, progression ratios, replicate dispersion of one event's count. These three describe
the sequences *as sequences*, for every event the config declares:

- :func:`event_age_distribution` — the age profile of each token.
- :func:`token_frequency` — how often each token is predicted at all (generated only).

Each builder describes **one cell**: one source ("generated" / "observed") in one jump-off window.
The arm calls them per cell and stacks the results, which is why none takes a ``source`` or window
argument — the caller stamps those.

One property of the data drives the design, and it was measured rather than assumed:

- **The comparison is a rate, not a share.** Observed records stop at the observation year (demo
  median last record: age 40) while generated trajectories run to the end of the fertile range
  (median 50). Comparing the *composition* of the two age profiles would therefore show a
  difference that is pure censoring. Dividing by each side's own person-years puts the censoring in
  the denominator where it belongs — the same thing :func:`~seqeval.metrics.fertility.asfr` and
  :func:`~seqeval.metrics.fertility.lexis_surface` do. ``share`` is published too, but ``rate`` is
  the comparable quantity and the one the figure draws.

``unit_keys`` names what one row of the population is. After
:func:`seqeval.arms._common.pool_seeds` a generated unit is one *trajectory* (and the real person it
came from survives as ``source_person_id``); for observed data a unit is a person. Both counts are
reported wherever they differ, since seeds multiply trajectories but not people.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from seqeval.core.outcomes import exposure
from seqeval.core.slicing import AgeBins, bin_ages
from seqeval.metrics._disclosure import MIN_CELL, suppress_small_cells
from seqeval.units import DAYS_PER_YEAR

__all__ = ["age_bins_for", "event_age_distribution", "token_frequency"]


def age_bins_for(frames: list[pd.DataFrame], *, width: float = 1.0) -> AgeBins:
    """One-year age bins spanning every age present in ``frames``, so nothing is dropped.

    :func:`~seqeval.core.slicing.bin_ages` returns NaN outside its edges, and a descriptive that
    silently discards the events at its extremes is worse than no descriptive. The grid is built
    from the data rather than from a fertility window: floor of the minimum age to ceil of the
    maximum, plus one bin so the oldest event has somewhere to land.
    """
    ages = [f["age"].to_numpy() for f in frames if len(f)]
    if not ages:
        return AgeBins.from_years(0.0, width, width)
    flat = np.concatenate(ages) / DAYS_PER_YEAR
    lo = float(np.floor(flat.min() / width) * width)
    hi = float(np.ceil(flat.max() / width) * width) + width
    return AgeBins.from_years(lo, hi, width)


def _unit_and_people(frame: pd.DataFrame, unit_keys: list[str]) -> tuple[int, int]:
    """``(n_units, n_source_persons)``; a pooled frame keeps the real id as ``source_person_id``."""
    units = int(frame[unit_keys[0]].nunique())
    people = (
        int(frame["source_person_id"].nunique())
        if "source_person_id" in frame.columns
        else units
    )
    return units, people


def _person_years(spans: pd.DataFrame, bins: AgeBins, unit_keys: list[str]) -> pd.DataFrame:
    """``[age_bin, person_years]`` for the cell — the same denominator for every token.

    Built with :func:`~seqeval.core.outcomes.exposure`, so a trajectory contributes to a bin only
    for the part of it that the trajectory actually covers. This is what makes a censored observed
    cell comparable to an uncensored generated one.
    """
    keys = [c for c in spans.columns if c not in ("start_age", "end_age")]
    exp = exposure(spans[[*keys, "start_age", "end_age"]], bins=bins)
    out = exp.groupby("age_bin", observed=True)["person_days"].sum().reset_index()
    out["person_years"] = out["person_days"] / DAYS_PER_YEAR
    return out[["age_bin", "person_years"]]


# =================================================================================================
# 1. age profile per token
# =================================================================================================
def event_age_distribution(
    frame: pd.DataFrame,
    spans: pd.DataFrame,
    *,
    tokens: dict[str, object],
    unit_keys: list[str],
    bins: AgeBins,
    min_cell: int = MIN_CELL,
) -> pd.DataFrame:
    """Age profile of each declared token: ``[alias, token, age_bin, n_events, rate, share, …]``.

    One row per ``(alias, age_bin)``. ``rate`` is ``n_events / person_years`` — occurrences per
    person-year of exposure in that bin — and is the quantity to compare across sources. ``share``
    is the bin's fraction of that token's events, which describes the shape of one profile but is
    not comparable across two differently censored populations (see the module docstring).

    Every bin gets a row for every token, including the empty ones: a gap in an age profile is a
    real zero, and leaving it out would make suppression and absence look alike.
    """
    cols = [
        "alias", "token", "age_bin", "n_events", "person_years", "rate", "share",
        "n_units", "n_source_persons", "n_events_total", "suppressed",
    ]
    exposure_by_bin = _person_years(spans, bins, unit_keys)
    rows = []
    for alias, token in tokens.items():
        ev = frame[frame["event"] == token]
        cell = pd.DataFrame({"age_bin": bins.labels})
        cell["alias"], cell["token"] = alias, str(token)

        if len(ev):
            binned = ev.assign(age_bin=bin_ages(ev["age"], bins)).dropna(subset=["age_bin"])
            agg = {"n_events": (unit_keys[0], "size"), "n_units": (unit_keys[0], "nunique")}
            if "source_person_id" in binned.columns:
                agg["n_source_persons"] = ("source_person_id", "nunique")
            per_bin = binned.groupby("age_bin", observed=True).agg(**agg).reset_index()
            cell = cell.merge(per_bin, on="age_bin", how="left")
        for col in ("n_events", "n_units", "n_source_persons"):
            if col not in cell.columns:
                cell[col] = 0
        # A trajectory count doubles as the head count when the frame carries no real-person id.
        cell["n_source_persons"] = cell["n_source_persons"].fillna(cell["n_units"])
        cell[["n_events", "n_units", "n_source_persons"]] = (
            cell[["n_events", "n_units", "n_source_persons"]].fillna(0).astype(np.int64)
        )

        cell = cell.merge(exposure_by_bin, on="age_bin", how="left")
        cell["person_years"] = cell["person_years"].fillna(0.0)
        cell["rate"] = np.where(
            cell["person_years"] > 0, cell["n_events"] / cell["person_years"], np.nan
        )
        cell["n_events_total"] = int(cell["n_events"].sum())
        cell["share"] = np.where(
            cell["n_events_total"] > 0, cell["n_events"] / cell["n_events_total"], np.nan
        )
        rows.append(cell)

    if not rows:
        return pd.DataFrame({c: pd.Series(dtype="float64") for c in cols})

    out = pd.concat(rows, ignore_index=True)
    # The bins partition `n_events_total`, which this table publishes, so a lone withheld bin is
    # recoverable by subtraction — hence the complement rule, grouped per token.
    out = suppress_small_cells(
        out,
        count_cols=("n_events", "n_source_persons", "n_events_total"),
        by=["alias"],
        min_cell=min_cell,
        also_null=("n_units", "person_years", "rate", "share"),
    )
    return out[cols].sort_values(["alias", "age_bin"]).reset_index(drop=True)


# =================================================================================================
# 2. how often each token is predicted
# =================================================================================================
def token_frequency(
    frame: pd.DataFrame,
    units: pd.DataFrame,
    *,
    tokens: dict[str, object],
    unit_keys: list[str],
    min_cell: int = MIN_CELL,
) -> pd.DataFrame:
    """How often each declared token is predicted, on two denominators that disagree.

    Generated output only — there is no observed baseline here. "How many of the model's sequences
    contain this token" is a question about the model's sequences.

    - ``share_with_any`` pools every trajectory: of all N×K sequences, the fraction carrying at
      least one of the token.
    - ``mean_person_share`` averages *per person*: each person's own fraction of their K
      trajectories that carry it, then the mean of those fractions.

    They answer different questions and can diverge sharply. The pooled share weights a person by
    how many trajectories they have, so with unequal seed counts it is dominated by the
    best-replicated people; the per-person mean gives every person one vote. Even at equal K they
    part company in interpretation: the pooled share is "how often does a sequence contain this",
    the per-person mean is "for a typical person, how often does the model predict this for them".

    ``units`` is the cell's population — one row per eligible trajectory, carrying
    ``source_person_id`` where a trajectory and a person differ. It cannot be read off ``frame``: a
    trajectory with no event has no rows there, and counting only what appears would drop it from
    every denominator.

    Only declared tokens are described. Anything else the model emits — an end-of-sequence marker,
    a token the config has not caught up with — is simply not this table's subject.
    """
    cols = [
        "alias", "token", "n_events", "n_units", "n_units_with_any", "share_with_any",
        "n_source_persons", "n_persons_with_any", "mean_person_share", "suppressed",
    ]
    unit = unit_keys[0]
    n_units, n_people = _unit_and_people(units, unit_keys)

    rows = []
    for alias, token in tokens.items():
        ev = frame[frame["event"] == token]
        carrying = set(ev[unit].unique())
        with_any = len(carrying)

        # Per-person share: each person's own fraction of their trajectories, then the mean of
        # those. Degenerates to a 0/1 flag when a unit *is* a person, which is the right answer.
        held = units.assign(_has=units[unit].isin(carrying))
        by_person = (
            held.groupby("source_person_id", observed=True)["_has"].mean()
            if "source_person_id" in held.columns
            else held.set_index(unit)["_has"].astype(float)
        )
        rows.append(
            {
                "alias": alias,
                "token": str(token),
                "n_events": int(len(ev)),
                "n_units": n_units,
                "n_units_with_any": with_any,
                "share_with_any": (with_any / n_units) if n_units else np.nan,
                "n_source_persons": n_people,
                "n_persons_with_any": int((by_person > 0).sum()),
                "mean_person_share": float(by_person.mean()) if len(by_person) else np.nan,
            }
        )

    if not rows:
        return pd.DataFrame({c: pd.Series(dtype="float64") for c in cols})

    out = suppress_small_cells(
        pd.DataFrame(rows),
        count_cols=("n_events", "n_units_with_any", "n_persons_with_any", "n_source_persons"),
        min_cell=min_cell,
        also_null=("n_units", "share_with_any", "mean_person_share"),
        # The tokens partition nothing this table publishes, so a withheld one is not recoverable
        # by subtracting the others.
        complement=False,
    )
    return out[cols].sort_values("alias").reset_index(drop=True)
