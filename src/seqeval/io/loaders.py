"""Validated IO for the four artifacts, plus the :class:`Bundle` that ties a run together.

Loaders are deliberately *dumb*: they read parquet/csv, normalize the age columns to canonical
integer days, validate against the pandera schemas, and return single frames. There are no side
tables, no padding, and no span precomputation — the observation span is derived downstream by
``core.outcomes.observation_spans`` (02) as the last ``age`` per group (one derivation path,
00 section 4.2). ``event`` values are kept raw; the event-definitions mapping is cosmetic only.

Unit conversion here is one of the three sanctioned sites (00 section 3): when ``age_unit ==
"years"`` the age columns are converted with :data:`~seqeval.units.DAYS_PER_YEAR` immediately after
read; when ``"days"`` they are cast to ``int32``, failing loudly on non-integral values.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from seqeval import units
from seqeval.config import Config, EventConfig
from seqeval.io.schema import (
    EVENT_DEFINITIONS_SCHEMA,
    GENERATED_SCHEMA,
    OBSERVED_SCHEMA,
    PERSONS_SCHEMA,
    SchemaError,
    validate,
)
from seqeval.units import days_to_years

__all__ = [
    "Bundle",
    "load_observed",
    "load_generated",
    "load_persons",
    "load_event_definitions",
    "load_all",
]

logger = logging.getLogger("seqeval")

_AGE_UNIT = Literal["days", "years"]


# =================================================================================================
# unit normalization
# =================================================================================================
def _normalize_ages(df: pd.DataFrame, cols: list[str], age_unit: _AGE_UNIT, artifact: str) -> None:
    """Normalize day/year age columns to canonical ``int32`` days, in place.

    ``years`` -> ``round(value * DAYS_PER_YEAR)`` (vectorized, matching
    :func:`seqeval.units.years_to_days`). ``days`` -> cast to ``int32``, raising
    :class:`SchemaError` if any value is non-integral.
    """
    for col in cols:
        arr = df[col].to_numpy()
        if age_unit == "years":
            # Sanctioned conversion site: same round-to-nearest-day rule as units.years_to_days.
            df[col] = np.rint(arr.astype("float64") * units.DAYS_PER_YEAR).astype(np.int32)
        else:
            floats = arr.astype("float64")
            if np.any(floats != np.rint(floats)):
                bad = df.loc[floats != np.rint(floats), col].head(5).tolist()
                raise SchemaError(
                    f"{artifact}: column {col!r} has non-integral day values (e.g. {bad}); with "
                    "age_unit='days' every age must be a whole number of days"
                )
            df[col] = floats.astype(np.int32)


# =================================================================================================
# individual artifact loaders
# =================================================================================================
def load_observed(
    path: str | Path, *, age_unit: _AGE_UNIT, columns: list[str] | None = None
) -> pd.DataFrame:
    """Load and validate the observed-sequences artifact."""
    df = pd.read_parquet(path, engine="pyarrow", columns=columns)
    _normalize_ages(df, ["age"], age_unit, "observed")
    return validate(df, OBSERVED_SCHEMA, "observed")


def load_generated(
    path: str | Path,
    *,
    age_unit: _AGE_UNIT,
    windows: list[tuple[int, int]] | None = None,  # days
    seeds: list[int] | None = None,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Load and validate the generated-sequences artifact, pushing ``windows``/``seeds`` down.

    ``windows`` are day-valued ``(age_start, age_stop)`` pairs. Predicate pushdown lets arms read
    only the runs they need. When ``age_unit == "years"`` the pushdown predicate is expressed in
    the file's native (year) unit as a *bounding range* to prune reads, and the exact day-space
    window filter is re-applied after normalization so the result is always exact regardless of
    rounding (00 section 6 note).
    """
    filters = _pushdown_filters(windows, seeds, age_unit)
    df = pd.read_parquet(path, engine="pyarrow", columns=columns, filters=filters)
    _normalize_ages(df, ["age", "age_start", "age_stop"], age_unit, "generated")

    # Exact filters in day space (pushdown for years is only an approximate prune).
    if windows is not None:
        wanted = set(windows)
        pairs = list(zip(df["age_start"], df["age_stop"], strict=True))
        df = df[[p in wanted for p in pairs]]
    if seeds is not None:
        df = df[df["seed"].isin(seeds)]

    return validate(df.reset_index(drop=True), GENERATED_SCHEMA, "generated")


def _pushdown_filters(
    windows: list[tuple[int, int]] | None, seeds: list[int] | None, age_unit: _AGE_UNIT
):
    """Build a pyarrow DNF filter list for :func:`load_generated`, or ``None``."""
    base: list[tuple] = []
    if seeds is not None:
        base.append(("seed", "in", list(seeds)))

    if not windows:
        return [base] if base else None

    if age_unit == "days":
        # Exact OR-of-windows: each DNF conjunction repeats the shared seed predicate.
        return [[*base, ("age_start", "=", s), ("age_stop", "=", e)] for s, e in windows]

    # years: prune with a bounding range in native years; the exact filter runs post-load.
    starts = [s for s, _ in windows]
    stops = [e for _, e in windows]
    lo_s, hi_s = days_to_years(min(starts)), days_to_years(max(starts))
    lo_e, hi_e = days_to_years(min(stops)), days_to_years(max(stops))
    eps = days_to_years(1)  # one day of slack so rounding never excludes a real match
    return [
        [
            *base,
            ("age_start", ">=", lo_s - eps),
            ("age_start", "<=", hi_s + eps),
            ("age_stop", ">=", lo_e - eps),
            ("age_stop", "<=", hi_e + eps),
        ]
    ]


def load_persons(path: str | Path, *, covariates: list[str]) -> pd.DataFrame:
    """Load persons, reading only ``person_id``, ``birth_year``, ``sex`` (if present), covariates.

    A declared covariate absent from the file is a :class:`SchemaError` naming the available
    columns.
    """
    available = set(pq.ParquetFile(path).schema.names)
    missing = [c for c in covariates if c not in available]
    if missing:
        raise SchemaError(
            f"persons: declared covariate(s) {missing} not in file; available columns are "
            f"{', '.join(sorted(available))}"
        )
    cols = ["person_id", "birth_year"]
    if "sex" in available:
        cols.append("sex")
    cols.extend(covariates)
    df = pd.read_parquet(path, engine="pyarrow", columns=cols)
    return validate(df, PERSONS_SCHEMA, "persons")


def load_event_definitions(path: str | Path) -> pd.DataFrame:
    """Load and validate the optional event-definitions csv (cosmetic labels only)."""
    df = pd.read_csv(path, dtype=str)
    return validate(df, EVENT_DEFINITIONS_SCHEMA, "event_definitions")


# =================================================================================================
# Bundle
# =================================================================================================
@dataclass(frozen=True)
class Bundle:
    """All loaded artifacts plus the resolved event alias map — one object per run.

    Frozen product of :func:`load_all`. Observation spans are *not* carried here; they derive from
    the frames themselves via the last-age convention downstream. Methods are lookups only.
    """

    observed: pd.DataFrame
    generated: pd.DataFrame | None
    persons: pd.DataFrame | None
    event_defs: pd.DataFrame | None
    events: EventConfig

    def token(self, alias: str) -> int | str:
        """Resolve an event alias to its raw token; ``KeyError`` names the known aliases."""
        try:
            return self.events[alias]
        except KeyError:
            raise KeyError(
                f"unknown event alias {alias!r}; declared aliases are: "
                f"{', '.join(sorted(self.events.keys())) or '(none)'}"
            ) from None

    def label(self, raw_token) -> str:
        """Human label for a raw token, falling back to ``str(raw_token)`` when unmapped."""
        if self.event_defs is not None:
            hit = self.event_defs.loc[
                self.event_defs["model_representation"].astype(str) == str(raw_token),
                "event_definition",
            ]
            if len(hit):
                return str(hit.iloc[0])
        return str(raw_token)

    def require_persons(self, why: str) -> pd.DataFrame:
        """Return the persons frame, or raise an actionable error naming what needs it."""
        if self.persons is None:
            raise ValueError(
                f"{why} requires a persons file (birth_year), but none was provided; add "
                "data.persons to the config"
            )
        return self.persons

    def available_windows(self) -> pd.DataFrame:
        """Unique ``(age_start, age_stop, n_seeds, n_persons)`` present in the generated data."""
        if self.generated is None:
            return pd.DataFrame(columns=["age_start", "age_stop", "n_seeds", "n_persons"]).astype(
                {
                    "age_start": "int32",
                    "age_stop": "int32",
                    "n_seeds": "int64",
                    "n_persons": "int64",
                }
            )
        grouped = self.generated.groupby(["age_start", "age_stop"], observed=True)
        out = grouped.agg(
            n_seeds=("seed", "nunique"),
            n_persons=("person_id", "nunique"),
        ).reset_index()
        return out.sort_values(["age_start", "age_stop"]).reset_index(drop=True)

    def population_summary(self) -> dict:
        """Population composition for ``seqeval validate`` — n, sex breakdown, cohort range.

        The observed file *defines* the population (00 section 5 rule 3); there is no filtering.
        Sex/cohort fields are populated only when persons is present.
        """
        n_persons = int(self.observed["person_id"].nunique())
        summary: dict = {"n_persons": n_persons, "sex_breakdown": None, "cohort_range": None}
        if self.persons is not None:
            if "sex" in self.persons.columns:
                summary["sex_breakdown"] = self.persons["sex"].value_counts(dropna=False).to_dict()
            summary["cohort_range"] = (
                int(self.persons["birth_year"].min()),
                int(self.persons["birth_year"].max()),
            )
        return summary


# =================================================================================================
# load_all
# =================================================================================================
def load_all(cfg: Config) -> Bundle:
    """Load every configured artifact and run cross-artifact validation into a :class:`Bundle`."""
    observed = load_observed(cfg.observed_path, age_unit=cfg.data.age_unit)

    generated = None
    if cfg.generated_path is not None:
        generated = load_generated(cfg.generated_path, age_unit=cfg.data.age_unit)
        generated = _drop_unknown_persons(generated, observed)

    persons = None
    if cfg.persons_path is not None:
        persons = load_persons(cfg.persons_path, covariates=cfg.covariates)

    event_defs = None
    if cfg.event_definitions_path is not None:
        event_defs = load_event_definitions(cfg.event_definitions_path)

    _warn_unseen_aliases(cfg.events, observed)

    return Bundle(
        observed=observed,
        generated=generated,
        persons=persons,
        event_defs=event_defs,
        events=cfg.events,
    )


def _drop_unknown_persons(generated: pd.DataFrame, observed: pd.DataFrame) -> pd.DataFrame:
    """Warn about and drop generated rows whose ``person_id`` is absent from observed."""
    known = set(observed["person_id"].unique())
    mask = generated["person_id"].isin(known)
    n_dropped = int((~mask).sum())
    if n_dropped:
        n_ids = int(generated.loc[~mask, "person_id"].nunique())
        logger.warning(
            "load_all: dropped %d generated row(s) for %d person_id(s) absent from observed",
            n_dropped,
            n_ids,
        )
    return generated[mask].reset_index(drop=True)


def _warn_unseen_aliases(events: EventConfig, observed: pd.DataFrame) -> None:
    """Warn for any declared alias whose token never appears in the observed events."""
    seen = set(observed["event"].astype(str).unique())
    for alias, token in events.items():
        if str(token) not in seen:
            logger.warning(
                "load_all: event alias %r -> token %r never appears in observed events; arms that "
                "consume it will find nothing",
                alias,
                token,
            )
