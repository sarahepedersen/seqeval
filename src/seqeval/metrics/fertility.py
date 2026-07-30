"""Fertility metrics: CCF, cohort ASFR, PPR, the Lexis surface (03).

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

from seqeval.core import replicates as rep
from seqeval.core.outcomes import exposure
from seqeval.core.slicing import AgeBins, bin_ages, calendar_year, cohort_bins
from seqeval.metrics._disclosure import MIN_CELL, suppress_small_cells
from seqeval.units import DAYS_PER_YEAR, years_to_days

__all__ = [
    "ccf",
    "ccf_variance",
    "parity_distribution",
    "asfr",
    "ppr",
    "lexis_surface",
    "FERTILE_UPPER_YEARS",
]

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

    Returns ``[*extra_by, (cohort,) ccf, complete, n_persons]``, where ``n_persons`` is the
    cohort's distinct women — the denominator of the mean. ``complete`` is ``False``
    when the cohort's observation does not reach the fertile upper bound (its members' spans all
    end before :data:`FERTILE_UPPER_YEARS`), so callers can tell a true CCF from a truncated mean —
    important
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
        n_persons = pop.groupby(group, observed=True)["person_id"].nunique().rename("n_persons")
        # A cohort is complete iff some member was observed to the fertile upper bound.
        complete = (
            pop.groupby(group, observed=True)["end_age"].max().ge(fertile_upper).rename("complete")
        )
        total_births = bt.groupby(group, observed=True).size().rename("total_births")
        out = pd.concat([n_persons, complete, total_births], axis=1).reset_index()
    else:
        out = pd.DataFrame(
            {
                "n_persons": [pop["person_id"].nunique()],
                "complete": [bool(pop["end_age"].max() >= fertile_upper)],
                "total_births": [len(bt)],
            }
        )
    out["total_births"] = out["total_births"].fillna(0)
    out["ccf"] = out["total_births"] / out["n_persons"]
    cols = [*group, "ccf", "complete", "n_persons"]
    return out[cols].sort_values(group).reset_index(drop=True) if group else out[cols]


def _with_replicates(frame: pd.DataFrame, seed_col: str) -> pd.DataFrame:
    """A frame guaranteed to carry ``seed_col``; observed data has one replicate per person.

    Real data is not replicated, so every person contributes a single realization. Treating that as
    one seed makes the replicate machinery give the right answer without a special case: the
    within-person variance is 0 and the whole spread is between people.
    """
    return frame if seed_col in frame.columns else frame.assign(**{seed_col: 0})


def ccf_variance(
    births: pd.DataFrame,
    spans: pd.DataFrame,
    persons: pd.DataFrame,
    *,
    seed_col: str = "seed",
    cohort_width: int = 1,
) -> pd.DataFrame:
    """Variance of the per-cohort :func:`ccf`, split into replicate and between-woman parts.

    Returns ``[cohort, ccf, within_var, between_var, total_var, n_persons]``, the
    decomposition of :func:`~seqeval.core.replicates.mean_variance_components` applied per cohort.
    ``births`` and ``spans`` may be seed-replicated (``spans`` is the population, so women with no
    births are counted in the denominator exactly as :func:`ccf` does); ``ccf`` here is the
    across-seed mean and agrees with :func:`ccf` averaged over seeds.

    On unreplicated data — the observed history — there is nothing for seeds to disagree about:
    ``within_var`` is 0 and ``total_var`` is the plain sampling variance of the cohort mean.
    """
    births, spans = _with_replicates(births, seed_col), _with_replicates(spans, seed_col)
    cohorts = cohort_bins(persons, width=cohort_width).reset_index()
    pop = spans[["person_id", seed_col]].drop_duplicates()
    got = births.groupby(["person_id", seed_col], observed=True).size().rename("count")
    counts = pop.merge(got, on=["person_id", seed_col], how="left")
    counts["count"] = counts["count"].fillna(0).astype(np.float64)
    moments = rep.count_moments(counts, run_keys=["person_id"], seed_col=seed_col).merge(
        cohorts, on="person_id", how="left"
    )
    rows = []
    for cohort, sub in moments.groupby("cohort", observed=True):
        comp = rep.mean_variance_components(sub["mean"], sub["var"], sub["k"])
        rows.append(
            {
                "cohort": cohort,
                "ccf": comp["mean"],
                "within_var": comp["within_var"],
                "between_var": comp["between_var"],
                "total_var": comp["total_var"],
                "n_persons": int(sub["person_id"].nunique()),
            }
        )
    return pd.DataFrame(rows).sort_values("cohort").reset_index(drop=True)


def parity_distribution(
    births: pd.DataFrame,
    spans: pd.DataFrame,
    persons: pd.DataFrame,
    *,
    seed_col: str = "seed",
    max_parity: int = 4,
    cohort_width: int = 1,
    min_cell: int = MIN_CELL,
) -> pd.DataFrame:
    """How completed parity is distributed across women in each cohort.

    Returns ``[cohort, parity, n_replicates, n_replicates_total, n_women_equiv, share,
    n_women_total, n_persons, suppressed]``, where ``n_persons`` is the distinct women with any
    replicate in the cell (the count suppression is judged on, withheld with the rest of the cell).

    Two readings of "how much mass is here" live side by side, and they are not the same number
    whenever women carry different replicate counts. ``n_replicates`` is the **direct count** of
    (woman, seed) trajectories that landed at this parity, out of ``n_replicates_total`` — the
    pooled synthetic population, which is what the figures draw. ``n_women_equiv``/``share`` weight
    each woman to a total of 1 regardless of how many seeds she has, so they answer "what fraction
    of *women*" instead. With balanced seeds the two are proportional and the picture is identical;
    with unbalanced ones they are not.
    Where :func:`ccf` gives a cohort's mean births per woman, this gives the spread that mean is
    averaging over — the outcome uncertainty a single woman faces, as opposed to the uncertainty in
    the estimate of the mean. ``Σ_k k·share_k`` reproduces that mean exactly whenever ``max_parity``
    is above the largest count present.

    ``parity`` stays an integer and its last value is **inclusive and above** (a woman with more
    than ``max_parity`` births lands there), so the column can still be summed and averaged; only
    the display renders it as ``4+``.

    Each woman carries total weight 1, split across her ``K_i`` seeds, so a woman with more
    replicates does not count for more of the population. The result is one pooled mixture over
    (woman, seed) — the model's marginal distribution for one woman's completed parity. Spread
    *across* seeds is inference uncertainty and belongs to the interval, not here.

    ``spans`` is the population, so childless women are in the denominator exactly as :func:`ccf`
    and :func:`ccf_variance` count them. Cells are withheld per
    :func:`~seqeval.metrics._disclosure.suppress_small_cells`, judged on how many distinct women
    land in them.
    """
    births, spans = _with_replicates(births, seed_col), _with_replicates(spans, seed_col)
    cohorts = cohort_bins(persons, width=cohort_width).reset_index()
    pop = spans[["person_id", seed_col]].drop_duplicates()
    got = births.groupby(["person_id", seed_col], observed=True).size().rename("count")
    counts = pop.merge(got, on=["person_id", seed_col], how="left")
    counts["count"] = counts["count"].fillna(0)
    counts["parity"] = counts["count"].clip(upper=max_parity).astype(np.int64)
    counts["weight"] = 1.0 / counts.groupby("person_id", observed=True)[seed_col].transform("size")
    counts = counts.merge(cohorts, on="person_id", how="left")

    cells = (
        counts.groupby(["cohort", "parity"], observed=True)
        .agg(
            n_replicates=("weight", "size"),
            n_women_equiv=("weight", "sum"),
            _n_women=("person_id", "nunique"),
        )
        .reset_index()
    )
    # Every parity gets a row in every cohort: a parity nobody reached is a true zero, and a
    # distribution with holes punched in it would be read as suppression.
    grid = pd.MultiIndex.from_product(
        [sorted(counts["cohort"].dropna().unique()), range(max_parity + 1)],
        names=["cohort", "parity"],
    )
    cells = (
        cells.set_index(["cohort", "parity"]).reindex(grid, fill_value=0).reset_index()
    )
    totals = counts.groupby("cohort", observed=True)["person_id"].nunique().rename("n_women_total")
    cells = cells.merge(totals, on="cohort", how="left")
    # Trajectories, not women: the denominator for `n_replicates`, so a figure drawing the raw
    # counts can normalise them without reaching for the woman-weighted total.
    traj = counts.groupby("cohort", observed=True).size().rename("n_replicates_total")
    cells = cells.merge(traj, on="cohort", how="left")
    cells["share"] = cells["n_women_equiv"] / cells["n_women_total"]

    cells = cells.rename(columns={"_n_women": "n_persons"})
    cells = suppress_small_cells(
        cells,
        count_cols=("n_persons", "n_replicates", "n_women_total"),
        by=["cohort"],
        min_cell=min_cell,
        also_null=("n_women_equiv", "share", "n_replicates_total"),
    )
    cols = [
        "cohort", "parity", "n_replicates", "n_replicates_total", "n_women_equiv", "share",
        "n_women_total", "n_persons",
    ]
    return cells[[*cols, "suppressed"]].sort_values(["cohort", "parity"]).reset_index(drop=True)


def asfr(
    births: pd.DataFrame,
    spans: pd.DataFrame,
    persons: pd.DataFrame,
    *,
    bins: AgeBins,
    extra_by: tuple[str, ...] = (),
    cohort_width: int = 1,
) -> pd.DataFrame:
    """Cohort age-specific fertility rate: births in a cell / person-years in the cell.

    Cells are ``(cohort, age_bin)`` with ``cohort_width``-year bands. Returns
    ``[*extra_by, cohort, age_bin, births, person_years, asfr, asfr_var, n_persons]``;
    ``n_persons`` is the distinct people exposed in the cell.

    ``asfr_var`` is the sampling variance of that one cell's rate under a Poisson count of births on
    fixed exposure — ``births / person_years²``, ``NaN`` where the cell has no exposure. It is the
    uncertainty *within* a cell, not the spread across replicates: with ``extra_by=("seed", ...)``
    each seed's cell carries its own, and it is the pooled cell's own value that becomes the
    reported interval (see :mod:`seqeval.metrics.pooling`).
    """
    extra_by = list(extra_by)
    dim = "cohort"

    # Exposure denominator.
    exp = exposure(spans, bins=bins, persons=persons).merge(
        cohort_bins(persons, width=cohort_width).reset_index(), on="person_id", how="left"
    )
    person_days = (
        exp.groupby([*extra_by, dim, "age_bin"], observed=True)
        .agg(person_days=("person_days", "sum"), n_persons=("person_id", "nunique"))
        .reset_index()
    )

    # Birth numerator, tagged with the same cell dimension.
    b = births.merge(persons[["person_id", "birth_year"]], on="person_id", how="left")
    b["age_bin"] = bin_ages(b["age"], bins)
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
    out["asfr_var"] = np.where(
        out["person_years"] > 0, out["births"] / out["person_years"] ** 2, np.nan
    )
    cols = [*extra_by, dim, "age_bin", "births", "person_years", "asfr", "asfr_var", "n_persons"]
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

    Returns ``[*extra_by, parity_from, parity_to, n_at_risk, n_progressed, ppr, ppr_var,
    n_persons]``, where ``n_persons`` is the distinct people at risk of the transition.

    ``ppr_var`` is the sampling variance of that one transition's ratio under a binomial count of
    progressions among those at risk — ``p(1−p)/n_at_risk``, ``NaN`` where nobody is at risk. It is
    the uncertainty *within* a cell, not the spread across replicates: with
    ``extra_by=("seed", ...)`` each seed's transition carries its own, and it is the pooled
    transition's own value that becomes the reported interval (see
    :mod:`seqeval.metrics.pooling`).

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
            agg = (
                grp.groupby(extra_by, observed=True)[["_at_risk", "_prog"]].sum().reset_index()
            )
            persons = (
                grp[grp["_at_risk"]].groupby(extra_by, observed=True)["person_id"].nunique()
            )
            agg["_persons"] = (
                agg.set_index(extra_by).index.map(persons).to_numpy(na_value=0)
            )
            for _, r in agg.iterrows():
                rows.append(
                    {
                        **{c: r[c] for c in extra_by},
                        "parity_from": k,
                        "parity_to": k + 1,
                        "n_at_risk": int(r["_at_risk"]),
                        "n_progressed": int(r["_prog"]),
                        "n_persons": int(r["_persons"]),
                    }
                )
        else:
            rows.append(
                {
                    "parity_from": k,
                    "parity_to": k + 1,
                    "n_at_risk": int(at_risk.sum()),
                    "n_progressed": int(progressed.sum()),
                    "n_persons": int(base.index[at_risk.to_numpy()].nunique()),
                }
            )

    out = pd.DataFrame(rows)
    out["ppr"] = np.where(out["n_at_risk"] > 0, out["n_progressed"] / out["n_at_risk"], np.nan)
    out["ppr_var"] = out["ppr"] * (1.0 - out["ppr"]) / out["n_at_risk"].where(out["n_at_risk"] > 0)
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
    ``[dim, age_bin, *extra_by, rate, rate_var, n_events, person_years]`` where ``dim`` is ``year``
    or ``cohort``. ``rate_var`` is the Poisson variance of the cell rate, ``n_events/person_years²``
    — the same within-cell quantity :func:`asfr` reports, and what a caller pooling several of these
    surfaces needs. Reuses births/exposure — never re-derives them.
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
    den = (
        exp.groupby([*extra_by, dim, "age_bin"], observed=True)
        .agg(person_days=("person_days", "sum"), n_persons=("person_id", "nunique"))
        .reset_index()
    )

    out = den.merge(num, on=[*extra_by, dim, "age_bin"], how="left")
    # A Lexis cell exists only where there is exposure — drop zero-exposure cells so that
    # observed/forecast completion is not blocked by empty observed cells (the period/by-year
    # exposure already drops these; the cohort/non-year path does not).
    out = out[out["person_days"] > 0]
    out["n_events"] = out["n_events"].fillna(0).astype(np.int64)
    out["person_years"] = out["person_days"] / DAYS_PER_YEAR
    out["rate"] = np.where(out["person_years"] > 0, out["n_events"] / out["person_years"], np.nan)
    out["rate_var"] = np.where(
        out["person_years"] > 0, out["n_events"] / out["person_years"] ** 2, np.nan
    )
    if basis == "period":
        out = out[(out[dim] >= year_range[0]) & (out[dim] <= year_range[1])]
    cols = [dim, "age_bin", *extra_by, "rate", "rate_var", "n_events", "person_years", "n_persons"]
    return out[cols].sort_values([*extra_by, dim, "age_bin"]).reset_index(drop=True)


