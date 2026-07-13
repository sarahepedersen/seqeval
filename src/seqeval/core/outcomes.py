"""Outcome extractors: canonical sequences → the analysis tables every metric consumes (02 core).

Three workhorses — :func:`births`, :func:`observation_spans` (+ :func:`exposure`),
:func:`time_to_event` — plus the binary-outcome evaluators (:func:`evaluate_framed`,
:func:`evaluate_count`) that define what every calibration number in 04 means. All take
``(df, keys)`` and are agnostic to observed vs generated; the *only* permitted asymmetry is that a
generated frame carries ``age_stop`` in its keys, from which :func:`observation_spans` derives the
start of observation. Every age/duration is integer days.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from seqeval.core.slicing import AgeBins
from seqeval.core.specs import CountQuery, FramedOutcome, TTESpec
from seqeval.units import DAYS_PER_YEAR

__all__ = [
    "births",
    "observation_spans",
    "exposure",
    "time_to_event",
    "evaluate_framed",
    "evaluate_count",
]


def _check_keys(df: pd.DataFrame, keys: list[str]) -> None:
    missing = [k for k in keys if k not in df.columns]
    if missing:
        raise ValueError(
            f"keys {missing} not in frame columns {list(df.columns)}; expected keys ⊆ columns"
        )


def _nth_occurrence(df: pd.DataFrame, keys: list[str], event, n: int, col: str) -> pd.DataFrame:
    """Age of the ``n``-th ``event`` per group as ``[*keys, col]`` (groups without it absent)."""
    ev = df[df["event"] == event].sort_values([*keys, "age"], kind="stable")
    order = ev.groupby(keys, observed=True).cumcount() + 1
    hit = ev[order == n]
    return hit[[*keys, "age"]].rename(columns={"age": col}).reset_index(drop=True)


# =================================================================================================
# 2.1 births
# =================================================================================================
def births(df: pd.DataFrame, keys: list[str], *, birth_event) -> pd.DataFrame:
    """One row per birth event: ``[*keys, order, age]`` with ``order = 1..k`` per group.

    Ties in age receive distinct consecutive orders (stable sort by ``(keys, age)``).
    """
    _check_keys(df, keys)
    b = df[df["event"] == birth_event].sort_values([*keys, "age"], kind="stable")
    b = b.copy()
    b["order"] = b.groupby(keys, observed=True).cumcount() + 1
    return b[[*keys, "order", "age"]].reset_index(drop=True)


# =================================================================================================
# 2.2 observation spans + exposure
# =================================================================================================
def observation_spans(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """One row per group: ``[*keys, start_age, end_age]`` (int days).

    ``end_age`` is the last age in the data (``max(age)`` per group) — one derivation path, no
    overrides (00 section 4.2); "no event" rows extend the span simply by existing. ``start_age``
    is the jump-off (``age_stop``) for generated frames (detected by ``age_stop`` being one of
    ``keys``) and 0 for observed frames. Must be computed on the *full* loaded frame, before any
    event filtering an arm applies.
    """
    _check_keys(df, keys)
    end = df.groupby(keys, observed=True)["age"].max().reset_index(name="end_age")
    if "age_stop" in keys:
        end["start_age"] = end["age_stop"].astype(np.int32)
    else:
        end["start_age"] = np.int32(0)
    end["end_age"] = end["end_age"].astype(np.int32)
    return end[[*keys, "start_age", "end_age"]].sort_values(keys).reset_index(drop=True)


def exposure(
    spans: pd.DataFrame,
    *,
    bins: AgeBins,
    persons: pd.DataFrame | None = None,
    by_year: bool = False,
) -> pd.DataFrame:
    """Expand spans into integer person-days per age bin (and per calendar year if ``by_year``).

    Vectorized overlap of ``[start_age, end_age)`` with each bin (and, for ``by_year``, with each
    calendar-year age segment). Person-days stay integer; conversion to person-years happens in the
    metrics at the final rate computation (00 section 3).

    Returns ``[*keys, age_bin, (year,) person_days]``.
    """
    keys = [c for c in spans.columns if c not in ("start_age", "end_age")]
    starts = spans["start_age"].to_numpy().astype(np.int64)
    ends = spans["end_age"].to_numpy().astype(np.int64)
    edges = bins.edges_days.astype(np.int64)

    if not by_year:
        lo, hi = edges[:-1][None, :], edges[1:][None, :]
        overlap = np.clip(np.minimum(ends[:, None], hi) - np.maximum(starts[:, None], lo), 0, None)
        g, b = len(spans), len(bins.labels)
        out = pd.DataFrame(
            {
                **{k: np.repeat(spans[k].to_numpy(), b) for k in keys},
                "age_bin": np.tile(bins.labels, g),
                "person_days": overlap.reshape(-1).astype(np.int64),
            }
        )
        return out.sort_values([*keys, "age_bin"]).reset_index(drop=True)

    if persons is None:
        raise ValueError("exposure(by_year=True) needs persons for birth_year (calendar time)")
    merged = spans.merge(persons[["person_id", "birth_year"]], on="person_id", how="left")
    birth_year = merged["birth_year"].to_numpy().astype(np.int64)

    # Fine cells = union of age-bin edges and calendar-year age edges (ceil(k * DAYS_PER_YEAR)).
    # A calendar year increments at age = ceil(k * DAYS_PER_YEAR) for integer k, independent of
    # birth_year, so the segmentation is shared across persons and only the label shifts.
    max_end = int(ends.max()) if len(ends) else 0
    kmax = int(np.floor(max_end / DAYS_PER_YEAR)) + 2
    year_edges = np.ceil(np.arange(kmax + 1) * DAYS_PER_YEAR).astype(np.int64)
    fine = np.unique(np.concatenate([edges, year_edges]))
    flo, fhi = fine[:-1][None, :], fine[1:][None, :]

    overlap = np.clip(np.minimum(ends[:, None], fhi) - np.maximum(starts[:, None], flo), 0, None)
    cell_lo = fine[:-1]
    cell_bin_idx = np.digitize(cell_lo, edges, right=False) - 1
    cell_k = np.floor(cell_lo / DAYS_PER_YEAR).astype(np.int64)  # completed years at cell start
    valid = (cell_bin_idx >= 0) & (cell_bin_idx < len(bins.labels))
    safe_idx = np.clip(cell_bin_idx, 0, len(bins.labels) - 1)
    cell_label = np.where(valid, bins.labels[safe_idx], np.nan)

    g, f = len(merged), len(cell_lo)
    out = pd.DataFrame(
        {
            **{k: np.repeat(merged[k].to_numpy(), f) for k in keys},
            "age_bin": np.tile(cell_label, g),
            "year": np.repeat(birth_year, f) + np.tile(cell_k, g),
            "person_days": overlap.reshape(-1).astype(np.int64),
        }
    )
    out = out[(out["person_days"] > 0)].dropna(subset=["age_bin"])
    return out.sort_values([*keys, "age_bin", "year"]).reset_index(drop=True)


# =================================================================================================
# 2.3 time-to-event
# =================================================================================================
def time_to_event(
    df: pd.DataFrame, keys: list[str], spec: TTESpec, spans: pd.DataFrame | None = None
) -> pd.DataFrame:
    """``[*keys, duration, observed]`` (duration in int days).

    ``duration`` = age of the target occurrence − origin age (origin = birth when ``spec.origin`` is
    ``None``). Groups where the origin never occurs are dropped (conditioning). ``observed`` is True
    when the target occurred within the span; otherwise the group is censored at ``end_age`` with
    ``observed=False``. Horizon capping is a frame concern (see the evaluators), not a TTE concern.
    """
    _check_keys(df, keys)
    if spans is None:
        spans = observation_spans(df, keys)
    base = spans[[*keys, "end_age"]].copy()

    if spec.origin is not None:
        org = _nth_occurrence(df, keys, spec.origin.target, spec.origin.occurrence, "origin_age")
        base = base.merge(org, on=keys, how="inner")  # drop groups without the origin
        origin_age = base["origin_age"].to_numpy().astype(np.int64)
    else:
        origin_age = np.zeros(len(base), dtype=np.int64)

    tgt = _nth_occurrence(df, keys, spec.target, spec.occurrence, "target_age")
    base = base.merge(tgt, on=keys, how="left")

    target_age = base["target_age"].to_numpy()
    end_age = base["end_age"].to_numpy().astype(np.int64)
    observed = ~np.isnan(target_age)
    duration = np.where(observed, np.nan_to_num(target_age) - origin_age, end_age - origin_age)

    out = base[keys].copy()
    out["duration"] = duration.astype(np.int64)
    out["observed"] = observed
    return out.sort_values(keys).reset_index(drop=True)


# =================================================================================================
# 2.4 binary outcome evaluators
# =================================================================================================
def evaluate_framed(
    df: pd.DataFrame,
    keys: list[str],
    spec: FramedOutcome,
    spans: pd.DataFrame,
    *,
    jumpoff: int | None = None,
) -> pd.DataFrame:
    """``[*keys, occurred, evaluable]`` — does ``spec.tte`` land inside ``spec.frame``?

    Frame semantics (all days): ``by_age A`` → n-th occurrence at age ≤ A; ``within W`` → n-th
    occurrence in ``(jumpoff, jumpoff + W]``; ``within_origin W`` → n-th occurrence within W of the
    outcome's origin event. A group is non-evaluable when the span does not cover the frame
    (censoring) or when the outcome is **settled at the jump-off** — already determined by the
    observed prefix (age ≤ jumpoff), either positively (the occurrence is in the prompt) or
    negatively (the whole frame lies at or before the jump-off). For ``within_origin``, groups whose
    origin never occurs are dropped.
    """
    _check_keys(df, keys)
    tte = spec.tte
    frame = spec.frame
    base = spans[[*keys, "end_age"]].copy()

    if frame.kind == "within_origin":
        if tte.origin is None:
            raise ValueError("within_origin frame requires the outcome's tte to declare an origin")
        org = _nth_occurrence(df, keys, tte.origin.target, tte.origin.occurrence, "origin_age")
        base = base.merge(org, on=keys, how="inner")  # drop groups without the origin

    tgt = _nth_occurrence(df, keys, tte.target, tte.occurrence, "target_age")
    base = base.merge(tgt, on=keys, how="left")

    end_age = base["end_age"].to_numpy().astype(np.int64)
    target_age = base["target_age"].to_numpy()
    has_target = ~np.isnan(target_age)

    if frame.kind == "by_age":
        upper = np.full(len(base), frame.value, dtype=np.float64)
        lower = np.full(len(base), -1.0)
    elif frame.kind == "within":
        if jumpoff is None:
            raise ValueError("within frame requires a jumpoff")
        upper = np.full(len(base), jumpoff + frame.value, dtype=np.float64)
        lower = np.full(len(base), jumpoff, dtype=np.float64)
    else:  # within_origin
        origin_age = base["origin_age"].to_numpy().astype(np.float64)
        upper = origin_age + frame.value
        lower = origin_age

    occurred = has_target & (target_age > lower) & (target_age <= upper)

    if jumpoff is None:
        settled = np.zeros(len(base), dtype=bool)
    else:
        settled_negative = upper <= jumpoff  # whole frame in the observed region
        settled_positive = has_target & (target_age <= jumpoff)  # occurrence already in the prompt
        settled = settled_negative | settled_positive
    # Coverage gates negatives only: a positive occurrence inside the frame is known even if the
    # span ends at it; a negative is trustworthy only if the span reaches the frame's upper bound.
    evaluable = ~settled & (occurred | (end_age >= upper))

    out = base[keys].copy()
    out["occurred"] = occurred
    out["evaluable"] = evaluable
    return out.sort_values(keys).reset_index(drop=True)


def evaluate_count(
    df: pd.DataFrame,
    keys: list[str],
    spec: CountQuery,
    spans: pd.DataFrame,
    *,
    jumpoff: int | None = None,
) -> pd.DataFrame:
    """``[*keys, occurred, evaluable]`` — do ≥ ``min_events`` post-jump-off events fall in frame?

    Counts only occurrences of ``spec.event`` with ``age > jumpoff`` inside the frame, so a count
    query can never be settled at the jump-off; ``evaluable`` is span-coverage only.
    """
    _check_keys(df, keys)
    if jumpoff is None:
        raise ValueError("evaluate_count requires a jumpoff (events are counted strictly after t2)")
    frame = spec.frame
    if frame.kind == "by_age":
        upper = frame.value
    elif frame.kind == "within":
        upper = jumpoff + frame.value
    else:
        raise ValueError(f"count query cannot use frame kind {frame.kind!r}")

    base = spans[[*keys, "end_age"]].copy()
    in_frame = (df["event"] == spec.event) & (df["age"] > jumpoff) & (df["age"] <= upper)
    counts = in_frame.groupby([df[k] for k in keys], observed=True).sum().rename("n").reset_index()
    base = base.merge(counts, on=keys, how="left")
    n = base["n"].fillna(0).to_numpy()
    occurred = n >= spec.min_events

    out = base[keys].copy()
    out["occurred"] = occurred
    # A reached threshold is known regardless of coverage; a below-threshold negative needs the
    # span to cover the frame's upper bound (else it may just be censored).
    out["evaluable"] = occurred | (base["end_age"].to_numpy().astype(np.int64) >= upper)
    return out.sort_values(keys).reset_index(drop=True)
