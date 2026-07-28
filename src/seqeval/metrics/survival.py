"""Native survival metrics: Kaplan-Meier, median survival (03).

No lifelines. Inputs are the day-valued outcome tables from :mod:`seqeval.core.outcomes`; all times
stay in **integer days** (viz converts axes to years). Exact integer times mean tied event times
group cleanly with no float-tolerance handling. Every function is key-agnostic — the ``by`` grouping
lets 04/05 reuse these verbatim with ``seed``/window keys.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

__all__ = ["kaplan_meier", "median_survival"]


def _n_persons(frame: pd.DataFrame) -> int:
    """Distinct people behind a frame; the row count when it carries no ``person_id``."""
    return int(frame["person_id"].nunique()) if "person_id" in frame.columns else len(frame)


def _km_one(dur: np.ndarray, obs: np.ndarray, z: float, n_persons: int) -> pd.DataFrame:
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
            # Greenwood variance of S(t) itself, on the survival scale rather than the log-log one
            # the CIs use — what a caller needs to combine this curve's sampling error with any
            # other variance component.
            "greenwood_var": np.where(np.isfinite(cum_v), surv**2 * cum_v, np.nan),
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "n_persons": n_persons,
        }
    )


def kaplan_meier(tte: pd.DataFrame, *, by: list[str] = ()) -> pd.DataFrame:
    """Product-limit survival curve from a :func:`time_to_event` table (durations in days).

    Returns ``[*by, time, n_at_risk, n_events, survival, greenwood_var, ci_lo, ci_hi, n_persons]``
    with ``time`` in days. ``by`` stratifies (cohort bins, ``sex``, or ``seed``/window for generated
    data); with no ``by`` the whole table is one curve. Confidence intervals use the Greenwood
    variance on the complementary log-log scale (valid for ``0 < S < 1``; ``NaN`` at the
    boundaries); ``greenwood_var`` is that variance on the survival scale, for callers combining it
    with other variance components. ``n_persons`` is the distinct people behind the curve, which
    ``n_at_risk`` (a per-time denominator) does not report.
    """
    by = list(by)
    z = norm.ppf(0.975)
    if not by:
        return _km_one(
            tte["duration"].to_numpy(), tte["observed"].to_numpy(), z, _n_persons(tte)
        )

    parts = []
    # One vectorized KM per stratum; the loop is over strata (not subjects), which is unavoidable
    # for a per-group product-limit estimator.
    for key, grp in tte.groupby(by, observed=True):
        one = _km_one(
            grp["duration"].to_numpy(), grp["observed"].to_numpy(), z, _n_persons(grp)
        )
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
