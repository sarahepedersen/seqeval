"""Pure slicing helpers on canonical long-format frames (02 core).

Every function takes ``(df, keys)`` (or a frame with a known shape) and is oblivious to whether the
sequences are observed or generated — the reuse mechanism of 00 section 4.3. All age/duration
values are **integer days**; year-valued config was already resolved by ``config.resolve_*``. The
one place years appear is :class:`AgeBins` *labels*, which are for output/plots only — the *edges*
used in comparisons are days.

No function mutates its input; each returns a new frame/series.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from seqeval.core.specs import Condition
from seqeval.units import completed_years, years_to_days

__all__ = [
    "truncate",
    "restrict_window",
    "attach_persons",
    "cohort_bins",
    "AgeBins",
    "bin_ages",
    "calendar_year",
    "condition_on_count",
    "align_jumpoff_to_event",
]


def _check_keys(df: pd.DataFrame, keys: list[str]) -> None:
    missing = [k for k in keys if k not in df.columns]
    if missing:
        raise ValueError(
            f"keys {missing} not in frame columns {list(df.columns)}; expected keys ⊆ columns"
        )


def truncate(df: pd.DataFrame, keys: list[str], *, max_age: int) -> pd.DataFrame:
    """Drop rows with ``age > max_age`` (simulate censoring at a jump-off point)."""
    _check_keys(df, keys)
    return df[df["age"] <= max_age].copy()


def restrict_window(df: pd.DataFrame, keys: list[str], *, lo: int, hi: int) -> pd.DataFrame:
    """Keep rows with ``lo <= age < hi`` (both in days)."""
    _check_keys(df, keys)
    return df[(df["age"] >= lo) & (df["age"] < hi)].copy()


def attach_persons(
    df: pd.DataFrame, persons: pd.DataFrame, columns: tuple[str, ...] = ("birth_year", "sex")
) -> pd.DataFrame:
    """Left-merge persons ``columns`` onto ``df`` by ``person_id``; raise on any missing person.

    Only requested columns that actually exist in ``persons`` are merged (``sex`` is optional).
    """
    missing_ids = set(df["person_id"].unique()) - set(persons["person_id"].unique())
    if missing_ids:
        shown = sorted(missing_ids)[:20]
        raise ValueError(
            f"{len(missing_ids)} person_id(s) in the frame are absent from persons (e.g. {shown}); "
            "cannot attach covariates"
        )
    have = [c for c in columns if c in persons.columns]
    return df.merge(persons[["person_id", *have]], on="person_id", how="left")


def cohort_bins(
    persons: pd.DataFrame, *, width: int = 1, range: tuple[int, int] | None = None
) -> pd.Series:
    """Person-indexed cohort label from ``birth_year`` (calendar years; no unit conversion).

    The label is the lower edge of each cohort bin. With ``width=1`` the label is simply the birth
    year. ``range`` fixes the lower anchor of the binning (default: the minimum birth year).
    """
    by = persons.set_index("person_id")["birth_year"].astype("int64")
    start = range[0] if range is not None else int(by.min())
    label = start + ((by - start) // width) * width
    return label.rename("cohort")


@dataclass(frozen=True)
class AgeBins:
    """Paired day-valued bin edges and year-valued labels that cannot drift apart.

    ``edges_days`` has length ``n_bins + 1``; ``labels`` has length ``n_bins`` (one per bin, the
    lower edge in years). Built via :meth:`from_years`, the only sanctioned conversion site's client
    (00a) — it calls :mod:`seqeval.units` once.
    """

    edges_days: np.ndarray  # int day edges, length n_bins + 1
    labels: np.ndarray  # year-valued lower-edge labels, length n_bins

    @classmethod
    def from_years(cls, lo: float, hi: float, width: float) -> AgeBins:
        """Build bins over ``[lo, hi)`` (years) of the given ``width`` (years)."""
        n = int(round((hi - lo) / width))
        edges_years = lo + width * np.arange(n + 1)
        edges_days = np.array([years_to_days(e) for e in edges_years], dtype=np.int32)
        return cls(edges_days=edges_days, labels=edges_years[:-1])


def bin_ages(ages: pd.Series, bins: AgeBins) -> pd.Series:
    """Map day-valued ``ages`` to their bin label; ages outside the edges become NaN."""
    idx = np.digitize(ages.to_numpy(), bins.edges_days, right=False) - 1
    valid = (idx >= 0) & (idx < len(bins.labels))
    out = np.full(len(ages), np.nan)
    out[valid] = bins.labels[idx[valid]]
    return pd.Series(out, index=ages.index, name="age_bin")


def calendar_year(df: pd.DataFrame) -> pd.Series:
    """Calendar year of each row: ``birth_year + completed_years(age)``.

    Requires :func:`attach_persons` to have merged ``birth_year`` first (raises otherwise). This is
    an approximation — the birth date *within* the calendar year is unknown, so a row is attributed
    to the year of its completed-age birthday, which can be off by up to a year at the boundary.
    """
    if "birth_year" not in df.columns:
        raise ValueError("calendar_year requires a 'birth_year' column; call attach_persons first")
    years = df["birth_year"].to_numpy().astype(np.int64) + completed_years(df["age"].to_numpy())
    return pd.Series(years, index=df.index, name="year")


def condition_on_count(
    df: pd.DataFrame, keys: list[str], *, cond: Condition, anchor_age: int | None = None
) -> pd.DataFrame:
    """Keep sequence-groups whose count of ``cond.event`` at ages ≤ anchor lies in the bounds.

    The anchor is ``cond.before_age`` if set, else ``anchor_age`` (the caller's jump-off); raising
    if both are ``None``. Generic count predicate (00 section 5 rule 6): parity conditioning is just
    ``event=birth``. Operates per group defined by ``keys`` so the same call works on observed
    (person keys) and generated (run keys) frames.
    """
    _check_keys(df, keys)
    anchor = cond.before_age if cond.before_age is not None else anchor_age
    if anchor is None:
        raise ValueError(
            "condition_on_count needs an anchor: set cond.before_age or pass anchor_age"
        )
    in_scope = (df["event"] == cond.event) & (df["age"] <= anchor)
    # Per-group count broadcast back to every row (single groupby, vectorized).
    counts = in_scope.groupby([df[k] for k in keys], observed=True).transform("sum")
    keep = counts >= cond.min_count
    if cond.max_count is not None:
        keep &= counts <= cond.max_count
    return df[keep].copy()


def align_jumpoff_to_event(observed: pd.DataFrame, *, event, occurrence: int) -> pd.DataFrame:
    """Per person, the age of the ``occurrence``-th ``event`` — person-specific jump-off values.

    Returns ``[person_id, age]`` for persons who reach that occurrence (others are absent). Supports
    "truncate at the time of the first birth" style backtests.
    """
    ev = observed[observed["event"] == event].sort_values(["person_id", "age"], kind="stable")
    order = ev.groupby("person_id", observed=True).cumcount() + 1
    hit = ev[order == occurrence]
    return hit[["person_id", "age"]].reset_index(drop=True)
