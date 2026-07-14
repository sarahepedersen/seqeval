"""Native survival metrics: Kaplan-Meier, parity life table, median survival (03).

No lifelines. Inputs are the day-valued outcome tables from :mod:`seqeval.core.outcomes`; all times
stay in **integer days** (viz converts axes to years). Exact integer times mean tied event times
group cleanly with no float-tolerance handling. Every function is key-agnostic — ``by`` / the life
table's grouping lets 04/05 reuse these verbatim with ``seed``/window keys.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from seqeval.core.outcomes import exposure
from seqeval.core.slicing import AgeBins, bin_ages
from seqeval.units import DAYS_PER_YEAR

__all__ = ["kaplan_meier", "life_table", "median_survival"]


def _km_one(dur: np.ndarray, obs: np.ndarray, z: float) -> pd.DataFrame:
    """Product-limit estimator for one group; Greenwood variance and log-log CIs."""
    n = len(dur)
    sorted_dur = np.sort(dur)
    event_times, d = np.unique(dur[obs], return_counts=True)
    # n_at_risk(t) = #(duration >= t); vectorized via searchsorted, no per-subject loop.
    n_at_risk = n - np.searchsorted(sorted_dur, event_times, side="left")

    surv = np.cumprod(1.0 - d / n_at_risk)
    # Greenwood cumulative term; guard the n == d step (survival hits 0, term is infinite).
    with np.errstate(divide="ignore", invalid="ignore"):
        increments = np.where(n_at_risk > d, d / (n_at_risk * (n_at_risk - d)), np.nan)
        cum_v = np.nancumsum(increments)
        se_loglog = np.sqrt(cum_v) / np.abs(np.log(surv))
        ci_lo = surv ** np.exp(z * se_loglog)
        ci_hi = surv ** np.exp(-z * se_loglog)
    # CIs are only defined for 0 < S < 1.
    degenerate = (surv <= 0) | (surv >= 1) | ~np.isfinite(se_loglog)
    ci_lo = np.where(degenerate, np.nan, ci_lo)
    ci_hi = np.where(degenerate, np.nan, ci_hi)

    return pd.DataFrame(
        {
            "time": event_times.astype(np.int64),
            "n_at_risk": n_at_risk.astype(np.int64),
            "n_events": d.astype(np.int64),
            "survival": surv,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
        }
    )


def kaplan_meier(tte: pd.DataFrame, *, by: list[str] = ()) -> pd.DataFrame:
    """Product-limit survival curve from a :func:`time_to_event` table (durations in days).

    Returns ``[*by, time, n_at_risk, n_events, survival, ci_lo, ci_hi]`` with ``time`` in days.
    ``by`` stratifies (cohort bins, ``sex``, or ``seed``/window for generated data); with no ``by``
    the whole table is one curve. Confidence intervals use the Greenwood variance on the
    complementary log-log scale (valid for ``0 < S < 1``; ``NaN`` at the boundaries).
    """
    by = list(by)
    z = norm.ppf(0.975)
    if not by:
        return _km_one(tte["duration"].to_numpy(), tte["observed"].to_numpy(), z)

    parts = []
    # One vectorized KM per stratum; the loop is over strata (not subjects), which is unavoidable
    # for a per-group product-limit estimator.
    for key, grp in tte.groupby(by, observed=True):
        one = _km_one(grp["duration"].to_numpy(), grp["observed"].to_numpy(), z)
        key_tuple = key if isinstance(key, tuple) else (key,)
        for col, val in zip(by, key_tuple, strict=True):
            one.insert(0, col, val)
        parts.append(one)
    return pd.concat(parts, ignore_index=True).sort_values([*by, "time"]).reset_index(drop=True)


def median_survival(km: pd.DataFrame, *, by: list[str] = ()) -> pd.DataFrame:
    """Median survival time (days): the first ``time`` where ``survival <= 0.5`` per group.

    ``NaN`` when the curve never reaches 0.5 (survival stays above one half within observation).
    """
    by = list(by)

    def _median(g: pd.DataFrame) -> float:
        below = g.loc[g["survival"] <= 0.5, "time"]
        return float(below.min()) if len(below) else np.nan

    if not by:
        return pd.DataFrame({"median": [_median(km)]})
    out = km.groupby(by, observed=True).apply(_median, include_groups=False).rename("median")
    return out.reset_index()


def life_table(
    births: pd.DataFrame, spans: pd.DataFrame, *, max_parity: int, bins: AgeBins
) -> pd.DataFrame:
    """Conventional demographic parity life table: occurrence/exposure rates by age and parity.

    Time spent at each parity within the childbearing window (from :func:`exposure`) is the
    denominator; births moving a woman from parity ``k`` to ``k+1`` are the numerator. This is the
    sequence-format-to-life-table conversion the spec calls for. Returns
    ``[age_bin, parity, person_years, births, occ_exp_rate]``; ``person_years = person_days /
    DAYS_PER_YEAR`` at the final rate step (00 section 3).

    A woman is at parity ``p`` from her ``p``-th birth (or the span start for ``p = 0``) until her
    ``(p+1)``-th birth (or the span end). Parities ``0 .. max_parity - 1`` are tabulated.
    """
    end = spans.set_index("person_id")["end_age"]
    start = spans.set_index("person_id")["start_age"]
    # birth age of the p-th birth per person (order == p).
    birth_age = births.pivot_table(index="person_id", columns="order", values="age", aggfunc="min")

    interval_rows = []
    for parity in range(max_parity):
        lo = start if parity == 0 else birth_age.get(parity)
        hi = birth_age.get(parity + 1)
        if lo is None:
            continue  # nobody reached this parity
        lo = lo.reindex(end.index)
        hi = hi.reindex(end.index) if hi is not None else pd.Series(np.nan, index=end.index)
        hi = hi.fillna(end)  # censored at parity p -> exposed until the span end
        seg = pd.DataFrame(
            {"person_id": end.index, "parity": parity, "start_age": lo, "end_age": hi}
        ).dropna(subset=["start_age"])
        seg = seg[seg["end_age"] > seg["start_age"]]
        interval_rows.append(seg)

    segments = pd.concat(interval_rows, ignore_index=True)
    segments["start_age"] = segments["start_age"].astype(np.int64)
    segments["end_age"] = segments["end_age"].astype(np.int64)
    exp = exposure(segments, bins=bins)  # keeps person_id + parity keys
    person_days = (
        exp.groupby(["parity", "age_bin"], observed=True)["person_days"].sum().reset_index()
    )

    b = births.copy()
    b["parity"] = b["order"] - 1  # a birth of order o happens while at parity o-1
    b = b[b["parity"] < max_parity]
    b["age_bin"] = bin_ages(b["age"], bins)
    birth_counts = (
        b.dropna(subset=["age_bin"])
        .groupby(["parity", "age_bin"], observed=True)
        .size()
        .reset_index(name="births")
    )

    out = person_days.merge(birth_counts, on=["parity", "age_bin"], how="left")
    out["births"] = out["births"].fillna(0).astype(np.int64)
    out["person_years"] = out["person_days"] / DAYS_PER_YEAR
    out["occ_exp_rate"] = np.where(
        out["person_years"] > 0, out["births"] / out["person_years"], np.nan
    )
    return (
        out[["age_bin", "parity", "person_years", "births", "occ_exp_rate"]]
        .sort_values(["parity", "age_bin"])
        .reset_index(drop=True)
    )
