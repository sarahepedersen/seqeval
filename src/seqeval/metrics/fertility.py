"""Fertility metrics: CCF, ASFR (period & cohort), PPR, TFR (03).

All operate on the day-valued births/spans tables from :mod:`seqeval.core.outcomes` plus ``persons``
where cohort/period is involved. Every function accepts ``extra_by`` — the mechanism by which 04/05
reuse them with ``seed``/window keys: pass ``extra_by=["seed", "age_start", "age_stop"]`` and the
metric is computed independently per (seed, window) cell. Person-days become person-years only at
the final rate step (00 section 3).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from seqeval.core.outcomes import exposure
from seqeval.core.slicing import AgeBins, bin_ages, calendar_year, cohort_bins
from seqeval.units import DAYS_PER_YEAR, years_to_days

__all__ = ["ccf", "asfr", "ppr", "tfr", "lexis_surface", "FERTILE_UPPER_YEARS"]

#: Upper edge of the childbearing window (years); a cohort is "complete" once observed to here.
FERTILE_UPPER_YEARS = 50.0


def ccf(
    births: pd.DataFrame,
    spans: pd.DataFrame,
    persons: pd.DataFrame,
    *,
    by_cohort: bool = True,
    extra_by: tuple[str, ...] = (),
    fertile_upper_days: int | None = None,
    cohort_width: int = 1,
) -> pd.DataFrame:
    """Completed cohort fertility: mean births per woman, by birth cohort.

    Returns ``[*extra_by, (cohort,) n_women, ccf, complete]``. ``complete`` is ``False`` when the
    cohort's observation does not reach the fertile upper bound (its members' spans all end before
    :data:`FERTILE_UPPER_YEARS`), so callers can tell a true CCF from a truncated mean — important
    when the same function runs on censored/backtest data. ``cohort_width`` is the birth-cohort band
    width in years.
    """
    extra_by = list(extra_by)
    fertile_upper = (
        fertile_upper_days if fertile_upper_days is not None else years_to_days(FERTILE_UPPER_YEARS)
    )
    group = [*extra_by, *(["cohort"] if by_cohort else [])]

    pop, bt = spans, births
    if by_cohort:
        # Tag both the population (spans) and the births with each person's cohort.
        ch = cohort_bins(persons, width=cohort_width).reset_index()
        pop = spans.merge(ch, on="person_id", how="left")
        bt = births.merge(ch, on="person_id", how="left")

    if group:
        n_women = pop.groupby(group, observed=True)["person_id"].nunique().rename("n_women")
        # A cohort is complete iff some member was observed to the fertile upper bound.
        complete = (
            pop.groupby(group, observed=True)["end_age"].max().ge(fertile_upper).rename("complete")
        )
        total_births = bt.groupby(group, observed=True).size().rename("total_births")
        out = pd.concat([n_women, complete, total_births], axis=1).reset_index()
    else:
        out = pd.DataFrame(
            {
                "n_women": [pop["person_id"].nunique()],
                "complete": [bool(pop["end_age"].max() >= fertile_upper)],
                "total_births": [len(bt)],
            }
        )
    out["total_births"] = out["total_births"].fillna(0)
    out["ccf"] = out["total_births"] / out["n_women"]
    cols = [*group, "n_women", "ccf", "complete"]
    return out[cols].sort_values(group).reset_index(drop=True) if group else out[cols]


def asfr(
    births: pd.DataFrame,
    spans: pd.DataFrame,
    persons: pd.DataFrame,
    *,
    mode: Literal["period", "cohort"],
    bins: AgeBins,
    extra_by: tuple[str, ...] = (),
    cohort_width: int = 1,
) -> pd.DataFrame:
    """Age-specific fertility rate: births in a cell / person-years in the cell.

    ``period`` cells are ``(year, age_bin)`` (calendar time from :func:`exposure` with
    ``by_year=True``); ``cohort`` cells are ``(cohort, age_bin)`` with ``cohort_width``-year bands.
    Returns ``[*extra_by, year|cohort, age_bin, births, person_years, asfr]``.
    """
    extra_by = list(extra_by)
    dim = "year" if mode == "period" else "cohort"

    # Exposure denominator.
    exp = exposure(spans, bins=bins, persons=persons, by_year=(mode == "period"))
    if mode == "cohort":
        exp = exp.merge(
            cohort_bins(persons, width=cohort_width).reset_index(), on="person_id", how="left"
        )
    person_days = (
        exp.groupby([*extra_by, dim, "age_bin"], observed=True)["person_days"].sum().reset_index()
    )

    # Birth numerator, tagged with the same cell dimension.
    b = births.merge(persons[["person_id", "birth_year"]], on="person_id", how="left")
    b["age_bin"] = bin_ages(b["age"], bins)
    if mode == "period":
        b[dim] = calendar_year(b)
    else:
        b = b.merge(
            cohort_bins(persons, width=cohort_width).reset_index(), on="person_id", how="left"
        )
    birth_counts = (
        b.dropna(subset=["age_bin"])
        .groupby([*extra_by, dim, "age_bin"], observed=True)
        .size()
        .reset_index(name="births")
    )

    out = person_days.merge(birth_counts, on=[*extra_by, dim, "age_bin"], how="left")
    out["births"] = out["births"].fillna(0).astype(np.int64)
    out["person_years"] = out["person_days"] / DAYS_PER_YEAR
    out["asfr"] = np.where(out["person_years"] > 0, out["births"] / out["person_years"], np.nan)
    cols = [*extra_by, dim, "age_bin", "births", "person_years", "asfr"]
    return out[cols].sort_values([*extra_by, dim, "age_bin"]).reset_index(drop=True)


def ppr(
    births: pd.DataFrame,
    spans: pd.DataFrame,
    *,
    max_parity: int,
    extra_by: tuple[str, ...] = (),
    min_exposure_after_k: int | None = None,
) -> pd.DataFrame:
    """Parity progression ratios: of groups reaching parity ``k``, the fraction reaching ``k+1``.

    Returns ``[*extra_by, parity_from, parity_to, n_at_risk, n_progressed, ppr]``.
    ``min_exposure_after_k`` (days) drops from the denominator any group whose span ends within that
    much of reaching parity ``k`` — it was censored before a fair chance to progress. Default
    ``None`` applies no such exclusion (censoring already removes groups that never reach ``k``);
    set it to require, e.g., a few years of post-birth exposure.
    """
    extra_by = list(extra_by)
    keys = [*extra_by, "person_id"] if extra_by else ["person_id"]
    # parity per group = number of births; age at the k-th birth via a pivot.
    parity = births.groupby(keys, observed=True).size().rename("parity")
    kth_age = births.pivot_table(index=keys, columns="order", values="age", aggfunc="min")
    base = spans.set_index(keys)[["start_age", "end_age"]].join(parity).fillna({"parity": 0})

    rows = []
    for k in range(max_parity):
        reached = base["parity"] >= k
        age_k = base["start_age"] if k == 0 else kth_age.get(k)
        if age_k is not None and k > 0:
            age_k = age_k.reindex(base.index)
        at_risk = reached.copy()
        if min_exposure_after_k is not None and age_k is not None:
            at_risk &= (base["end_age"] - age_k) >= min_exposure_after_k
        progressed = at_risk & (base["parity"] >= k + 1)

        if extra_by:
            grp = base.reset_index()
            grp["_at_risk"] = at_risk.to_numpy()
            grp["_prog"] = progressed.to_numpy()
            agg = grp.groupby(extra_by, observed=True)[["_at_risk", "_prog"]].sum().reset_index()
            for _, r in agg.iterrows():
                rows.append(
                    {
                        **{c: r[c] for c in extra_by},
                        "parity_from": k,
                        "parity_to": k + 1,
                        "n_at_risk": int(r["_at_risk"]),
                        "n_progressed": int(r["_prog"]),
                    }
                )
        else:
            rows.append(
                {
                    "parity_from": k,
                    "parity_to": k + 1,
                    "n_at_risk": int(at_risk.sum()),
                    "n_progressed": int(progressed.sum()),
                }
            )

    out = pd.DataFrame(rows)
    out["ppr"] = np.where(out["n_at_risk"] > 0, out["n_progressed"] / out["n_at_risk"], np.nan)
    return out.sort_values([*extra_by, "parity_from"]).reset_index(drop=True)


def lexis_surface(
    births: pd.DataFrame,
    spans: pd.DataFrame,
    persons: pd.DataFrame,
    *,
    occurrence: int = 1,
    bins: AgeBins,
    year_range: tuple[int, int],
    extra_by: tuple[str, ...] = (),
    basis: Literal["period", "cohort"] = "period",
    cohort_width: int = 1,
) -> pd.DataFrame:
    """Occurrence-specific fertility intensity per cell — the Lexis surface, period or cohort basis.

    Numerator: the ``occurrence``-th births in the cell (``births`` must carry the ``order`` column
    from :func:`seqeval.core.outcomes.births`). Denominator: exposure in the cell
    (:func:`~seqeval.core.outcomes.exposure`), person-days converted to person-years at the rate
    step. ``basis="period"`` gives ``(year, age_bin)`` cells (calendar time, limited to
    ``year_range``); ``basis="cohort"`` gives ``(cohort, age_bin)`` cells (``cohort_width``-year
    bands), the view that shows each cohort's age profile and its forecasted completion. Returns
    ``[dim, age_bin, *extra_by, rate, n_events, person_years]`` where ``dim`` is ``year`` or
    ``cohort``. Reuses births/exposure — never re-derives them.
    """
    extra_by = list(extra_by)
    dim = "year" if basis == "period" else "cohort"

    b = births[births["order"] == occurrence].merge(
        persons[["person_id", "birth_year"]], on="person_id", how="left"
    )
    b["age_bin"] = bin_ages(b["age"], bins)
    if basis == "period":
        b[dim] = calendar_year(b)
    else:
        b = b.merge(cohort_bins(persons, width=cohort_width).reset_index(), on="person_id")
    num = (
        b.dropna(subset=["age_bin"])
        .groupby([*extra_by, dim, "age_bin"], observed=True)
        .size()
        .reset_index(name="n_events")
    )

    exp = exposure(spans, bins=bins, persons=persons, by_year=(basis == "period"))
    if basis == "cohort":
        exp = exp.merge(cohort_bins(persons, width=cohort_width).reset_index(), on="person_id")
    den = exp.groupby([*extra_by, dim, "age_bin"], observed=True)["person_days"].sum().reset_index()

    out = den.merge(num, on=[*extra_by, dim, "age_bin"], how="left")
    # A Lexis cell exists only where there is exposure — drop zero-exposure cells so that
    # observed/forecast completion is not blocked by empty observed cells (the period/by-year
    # exposure already drops these; the cohort/non-year path does not).
    out = out[out["person_days"] > 0]
    out["n_events"] = out["n_events"].fillna(0).astype(np.int64)
    out["person_years"] = out["person_days"] / DAYS_PER_YEAR
    out["rate"] = np.where(out["person_years"] > 0, out["n_events"] / out["person_years"], np.nan)
    if basis == "period":
        out = out[(out[dim] >= year_range[0]) & (out[dim] <= year_range[1])]
    cols = [dim, "age_bin", *extra_by, "rate", "n_events", "person_years"]
    return out[cols].sort_values([*extra_by, dim, "age_bin"]).reset_index(drop=True)


def tfr(asfr_period: pd.DataFrame, *, extra_by: tuple[str, ...] = ()) -> pd.DataFrame:
    """Period total fertility rate: sum of period ASFRs over age bins (times the bin width).

    Returns ``[*extra_by, year, tfr]``. The bin width (years) is inferred from the spacing of
    ``age_bin`` labels so the sum is a proper integral of the age-specific rate; with the default
    one-year bins this is simply the sum of the rates.
    """
    extra_by = list(extra_by)
    labels = np.sort(asfr_period["age_bin"].dropna().unique())
    width = float(np.median(np.diff(labels))) if len(labels) > 1 else 1.0
    grouped = asfr_period.groupby([*extra_by, "year"], observed=True)["asfr"].sum() * width
    return grouped.rename("tfr").reset_index()
