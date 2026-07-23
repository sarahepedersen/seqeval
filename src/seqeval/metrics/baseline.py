"""Age-specific fertility rate (ASFR) baseline for the backtesting arm (04).

The backtest scores a model's per-person probabilities against observed truth, but a score is only
interpretable against a reference. This module builds that reference: a **demographic baseline** —
the probability of the same binary outcome implied by nothing more than the observed age-specific
fertility rates. A model that cannot beat it has learned nothing an actuarial table does not
already say.

The construction, in three steps
--------------------------------

1. **Schedule** (:func:`asfr_schedule`) — period ASFR from the *observed* file alone: births in
   cell ``(age_bin, calendar year)`` over person-years of exposure in that cell. This is exactly
   :func:`seqeval.metrics.fertility.asfr` with ``mode="period"``; nothing new is estimated here.

2. **Freeze at the jump-off** (:func:`frozen_rates`) — person ``i`` reaching the jump-off ``t2`` in
   calendar year ``y_i`` may only use rates from years ``<= y_i``. Their schedule is the row of
   rates at ``y_i``, held constant forward (the classic *frozen period rates* projection). Age bins
   with no rate at ``y_i`` fall back to the most recent earlier year that has one; still no cell
   later than ``y_i`` is ever read, so the baseline leaks nothing about the outcome being scored.

3. **Integrate over the frame** (:func:`baseline_probability`) — the outcome's frame gives each
   person an age interval ``(lo, hi]``. Summing the frozen rate times exposure over the age bins in
   that interval gives a cumulative intensity ``Λ``; treating births as a Poisson process with that
   intensity, ``P(N >= m) = poisson.sf(m - 1, Λ)``, where ``m`` is how many further births the
   outcome needs (1 for a count query with ``min_events=1``; for an ordinal outcome such as "2nd
   birth", the person's parity in the observed prefix is subtracted, since the model sees that
   prefix too).

The rate schedule itself is deliberately *plain*: births over exposure by age and year, with no
parity split. Two 30-year-olds in 1990 get the same rate whether they have zero children or two.
That is the point — it is the population-average schedule, and any lift the model shows over it is
lift from knowing the individual.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.stats import poisson

from seqeval.core.outcomes import exposure
from seqeval.core.slicing import AgeBins, align_jumpoff_to_event
from seqeval.core.specs import CountQuery, FramedOutcome
from seqeval.metrics import fertility as fe
from seqeval.units import DAYS_PER_YEAR, completed_years

logger = logging.getLogger("seqeval")

__all__ = [
    "DEFAULT_AGE_RANGE",
    "asfr_schedule",
    "frozen_rates",
    "frame_intervals",
    "baseline_probability",
    "score",
    "compare",
]

#: Default childbearing age range (years) over which the schedule and every ``Λ`` are integrated.
DEFAULT_AGE_RANGE = (15.0, 50.0)

#: Loss-type metrics, where skill is ``1 - model / baseline`` (higher is better, 0 = no better).
_LOSS_METRICS = {"brier", "mse", "ece", "log_loss"}


# =================================================================================================
# 1. the schedule
# =================================================================================================
def asfr_schedule(
    observed: pd.DataFrame,
    persons: pd.DataFrame,
    *,
    birth_event,
    bins: AgeBins,
    min_person_years: float = 1.0,
) -> pd.DataFrame:
    """Period ASFR from the observed file: ``[year, age_bin, births, person_years, asfr]``.

    A thin wrapper over :func:`seqeval.metrics.fertility.asfr` (``mode="period"``) that additionally
    blanks cells resting on less than ``min_person_years`` of exposure — a rate built on a handful
    of person-years is noise, and blanking lets the frozen-rate lookup fall back to a better-
    supported earlier year rather than propagate it.
    """
    from seqeval.core.outcomes import births as births_of
    from seqeval.core.outcomes import observation_spans
    from seqeval.io.schema import OBS_KEYS

    b = births_of(observed, OBS_KEYS, birth_event=birth_event)
    spans = observation_spans(observed, OBS_KEYS)
    sched = fe.asfr(b, spans, persons, mode="period", bins=bins)

    thin = sched["person_years"] < min_person_years
    if thin.any():
        sched = sched.copy()
        sched.loc[thin, "asfr"] = np.nan
        logger.debug(
            "baseline: blanked %d ASFR cell(s) with < %.3g person-years of exposure",
            int(thin.sum()),
            min_person_years,
        )
    return sched


def frozen_rates(schedule: pd.DataFrame) -> pd.DataFrame:
    """Year × age-bin rate matrix, forward-filled along calendar time — the frozen-rate lookup.

    Returns a long frame ``[year, age_bin, asfr_frozen, is_fallback]`` covering every year from the
    schedule's first to its last, where ``asfr_frozen`` at ``(y, a)`` is the rate for age bin ``a``
    in the most recent year ``<= y`` that has one (``is_fallback`` marks the carried-forward cells).
    Filling runs strictly forward, so a lookup at year ``y`` can never read a cell from a later
    year — that is what keeps the baseline free of hindsight.
    """
    wide = schedule.pivot_table(index="year", columns="age_bin", values="asfr")
    years = np.arange(int(wide.index.min()), int(wide.index.max()) + 1)
    wide = wide.reindex(years)
    filled = wide.ffill()  # forward in calendar time only — never backfill.

    long = filled.stack(future_stack=True).rename("asfr_frozen").reset_index()
    raw = wide.stack(future_stack=True).rename("asfr_raw").reset_index()
    out = long.merge(raw, on=["year", "age_bin"], how="left")
    out["is_fallback"] = out["asfr_raw"].isna() & out["asfr_frozen"].notna()
    return out[["year", "age_bin", "asfr_frozen", "is_fallback"]]


# =================================================================================================
# 2. per-person frame intervals
# =================================================================================================
def frame_intervals(
    observed: pd.DataFrame,
    spec: FramedOutcome | CountQuery,
    *,
    jumpoff: int,
    person_ids: np.ndarray | pd.Index | None = None,
) -> pd.DataFrame:
    """Per-person age interval the outcome asks about: ``[person_id, lo, hi, n_needed]`` (days).

    Mirrors the frame semantics of :func:`seqeval.core.outcomes.evaluate_framed` /
    :func:`~seqeval.core.outcomes.evaluate_count`: ``by_age A`` -> ``(jumpoff, A]``; ``within W`` ->
    ``(jumpoff, jumpoff + W]``; ``within_origin W`` -> ``(max(jumpoff, origin), origin + W]``.

    ``n_needed`` is how many further occurrences the outcome requires. For a
    :class:`~seqeval.core.specs.CountQuery` that is ``min_events`` (it counts only post-jump-off
    events by construction). For a :class:`~seqeval.core.specs.FramedOutcome` naming the ``k``-th
    occurrence it is ``k`` minus the person's count of that event in the observed prefix, floored at
    1 — the prefix is information the model sees too, so the baseline is allowed it; only the
    *rates* are population-average.

    Persons whose ``within_origin`` origin has not occurred by the jump-off are dropped (their
    interval is not knowable at prediction time), as are those the origin never occurs for at all.
    """
    ids = observed["person_id"].unique() if person_ids is None else np.asarray(person_ids)
    out = pd.DataFrame({"person_id": ids})

    event = spec.event if isinstance(spec, CountQuery) else spec.tte.target
    prior = (
        observed.loc[(observed["event"] == event) & (observed["age"] <= jumpoff)]
        .groupby("person_id", observed=True)
        .size()
        .rename("prior")
    )
    out["prior"] = out["person_id"].map(prior).fillna(0).astype(np.int64)

    if isinstance(spec, CountQuery):
        out["n_needed"] = int(spec.min_events)
    else:
        out["n_needed"] = np.maximum(spec.tte.occurrence - out["prior"], 1)

    frame = spec.frame
    if frame.kind == "by_age":
        out["lo"], out["hi"] = jumpoff, float(frame.value)
    elif frame.kind == "within":
        out["lo"], out["hi"] = jumpoff, float(jumpoff + frame.value)
    else:  # within_origin
        tte = spec.tte  # type: ignore[union-attr]
        origin = align_jumpoff_to_event(
            observed, event=tte.origin.target, occurrence=tte.origin.occurrence
        ).set_index("person_id")["age"]
        origin_age = out["person_id"].map(origin)
        known = origin_age.notna() & (origin_age <= jumpoff)
        n_dropped = int((~known).sum())
        if n_dropped:
            logger.debug(
                "baseline: dropped %d person(s) whose '%s' origin is not observed by the jump-off",
                n_dropped,
                spec.name,
            )
        out = out[known].copy()
        origin_age = origin_age[known]
        out["lo"] = np.maximum(origin_age.to_numpy(), jumpoff).astype(float)
        out["hi"] = (origin_age + frame.value).to_numpy().astype(float)

    # An interval that closes at or before it opens contributes no exposure (Λ = 0, p = 0).
    out["hi"] = np.maximum(out["hi"], out["lo"])
    return out[["person_id", "lo", "hi", "n_needed"]].reset_index(drop=True)


# =================================================================================================
# 3. the baseline probability
# =================================================================================================
def baseline_probability(
    observed: pd.DataFrame,
    persons: pd.DataFrame,
    spec: FramedOutcome | CountQuery,
    *,
    schedule: pd.DataFrame,
    jumpoff: int,
    bins: AgeBins,
    person_ids: np.ndarray | pd.Index | None = None,
) -> pd.DataFrame:
    """Per-person ASFR-baseline probability of ``spec`` at ``jumpoff``.

    Returns ``[person_id, jumpoff_year, n_needed, lo, hi, exposure_years, exposure_years_fallback,
    exposure_years_unmatched, lambda_hat, p_base]`` — one row per evaluable person, in days for
    ``lo``/``hi`` and years for every exposure column.

    ``lambda_hat`` is the cumulative fertility intensity over the person's frame under the rates
    frozen at their own jump-off year; ``p_base = P(N >= n_needed)`` for
    ``N ~ Poisson(lambda_hat)``.
    ``exposure_years_fallback`` is the part of their exposure priced with a carried-forward earlier
    year, and ``exposure_years_unmatched`` (with its share ``unmatched_fraction``) the part with no
    rate at or before their jump-off year at all — the left-truncation of the panel, where the
    oldest cohorts have no observed history above the age they entered it. Unmatched exposure is
    priced at rate zero, which biases those persons' ``p_base`` low, so the share is reported per
    person rather than absorbed: callers drop the unpriceable (see
    ``arms.backtesting.baseline.max_unmatched_fraction``) instead of scoring a baseline that is
    mostly an artifact of missing history.
    """
    iv = frame_intervals(observed, spec, jumpoff=jumpoff, person_ids=person_ids)
    if iv.empty:
        return _empty_baseline()

    spans = pd.DataFrame(
        {
            "person_id": iv["person_id"].to_numpy(),
            "start_age": iv["lo"].to_numpy().astype(np.int64),
            "end_age": iv["hi"].to_numpy().astype(np.int64),
        }
    )
    exp = exposure(spans, bins=bins)  # [person_id, age_bin, person_days]
    exp = exp[exp["person_days"] > 0]
    if exp.empty:
        return _empty_baseline()

    # Each person's jump-off calendar year — the year their rate schedule is frozen at.
    birth_year = persons.set_index("person_id")["birth_year"]
    jumpoff_completed = int(completed_years(np.array([jumpoff]))[0])
    jumpoff_year = (iv["person_id"].map(birth_year).astype("float64") + jumpoff_completed).rename(
        "jumpoff_year"
    )
    jo_year = pd.DataFrame({"person_id": iv["person_id"], "jumpoff_year": jumpoff_year})

    frozen = frozen_rates(schedule)
    year_lo, year_hi = int(frozen["year"].min()), int(frozen["year"].max())
    # Beyond the schedule's last year the frozen rates are simply the last year's (the same
    # carry-forward rule); before its first year no rate exists at all and exposure is unmatched.
    lookup_year = jo_year["jumpoff_year"].clip(upper=year_hi)
    jo_year["_lookup_year"] = np.where(jo_year["jumpoff_year"] >= year_lo, lookup_year, np.nan)

    cells = exp.merge(jo_year, on="person_id", how="left")
    cells = cells.merge(
        frozen.rename(columns={"year": "_lookup_year"}),
        on=["_lookup_year", "age_bin"],
        how="left",
    )
    cells["person_years"] = cells["person_days"] / DAYS_PER_YEAR
    matched = cells["asfr_frozen"].notna()
    cells["_contrib"] = np.where(matched, cells["asfr_frozen"].fillna(0.0), 0.0)
    cells["_contrib"] *= cells["person_years"]

    agg = cells.groupby("person_id", observed=True).apply(
        lambda g: pd.Series(
            {
                "exposure_years": g["person_years"].sum(),
                "exposure_years_fallback": g.loc[
                    g["is_fallback"].fillna(False), "person_years"
                ].sum(),
                "exposure_years_unmatched": g.loc[g["asfr_frozen"].isna(), "person_years"].sum(),
                "lambda_hat": g["_contrib"].sum(),
            }
        ),
        include_groups=False,
    )

    out = iv.merge(agg.reset_index(), on="person_id", how="left").merge(
        jo_year[["person_id", "jumpoff_year"]], on="person_id", how="left"
    )
    for col in ("exposure_years", "exposure_years_fallback", "exposure_years_unmatched"):
        out[col] = out[col].fillna(0.0)
    out["lambda_hat"] = out["lambda_hat"].fillna(0.0)
    out["unmatched_fraction"] = np.where(
        out["exposure_years"] > 0, out["exposure_years_unmatched"] / out["exposure_years"], 1.0
    )
    # P(at least n_needed events) for a Poisson count with mean lambda_hat.
    out["p_base"] = poisson.sf(out["n_needed"].to_numpy() - 1, out["lambda_hat"].to_numpy())

    _warn_unmatched(out, spec)
    cols = [
        "person_id",
        "jumpoff_year",
        "n_needed",
        "lo",
        "hi",
        "exposure_years",
        "exposure_years_fallback",
        "exposure_years_unmatched",
        "unmatched_fraction",
        "lambda_hat",
        "p_base",
    ]
    return out[cols].sort_values("person_id").reset_index(drop=True)


def _empty_baseline() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "person_id",
            "jumpoff_year",
            "n_needed",
            "lo",
            "hi",
            "exposure_years",
            "exposure_years_fallback",
            "exposure_years_unmatched",
            "unmatched_fraction",
            "lambda_hat",
            "p_base",
        ]
    )


def _warn_unmatched(out: pd.DataFrame, spec) -> None:
    total = float(out["exposure_years"].sum())
    if total <= 0:
        return
    frac = float(out["exposure_years_unmatched"].sum()) / total
    if frac > 0.05:
        logger.warning(
            "baseline %s: %.0f%% of frame exposure falls in (age, year) cells with no observed "
            "ASFR at or before the jump-off year — those cells are priced at rate 0, so the "
            "baseline is biased low here",
            spec.name,
            100 * frac,
        )


# =================================================================================================
# 4. scoring and comparison
# =================================================================================================
def score(joined: pd.DataFrame, *, p_col: str = "p_base") -> dict[str, float]:
    """Score a deterministic probability column against ``y_true``.

    Returns ``{"brier", "mse", "r2", "ece", "roc_auc", "log_loss"}``. The baseline carries no
    replicate noise, so no finite-seed correction applies and ``brier == mse`` by construction (both
    are reported so each can be paired with the corresponding model-side statistic).
    """
    from seqeval.metrics import ml

    p = joined[p_col].to_numpy(dtype=float)
    y = joined["y_true"].to_numpy(dtype=float)
    if not len(p):
        return dict.fromkeys(("brier", "mse", "r2", "ece", "roc_auc", "log_loss"), float("nan"))

    brier = float(np.mean((p - y) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    frame = pd.DataFrame({"p_hat": p, "y_true": y})
    return {
        "brier": brier,
        "mse": brier,
        "r2": float("nan") if ss_tot == 0 else 1.0 - float(np.sum((y - p) ** 2)) / ss_tot,
        "ece": ml.ece(ml.calibration_table(frame, strategy="quantile")),
        "roc_auc": ml.roc_auc(frame),
        "log_loss": ml.log_loss(frame),
    }


#: Comparison rows: ``metric -> (model-side statistic, baseline-side statistic)``.
_PAIRS = (
    ("brier", "brier_corrected", "brier"),
    ("mse", "mse", "mse"),
    ("r2", "r2", "r2"),
    ("ece", "ece", "ece"),
    ("roc_auc", "roc_auc", "roc_auc"),
)


def compare(model_scores: dict[str, float], baseline_scores: dict[str, float]) -> pd.DataFrame:
    """Model vs baseline, one row per metric.

    Columns: ``[metric, model_metric, model, baseline, delta, skill]``.

    ``delta`` is ``model - baseline`` (read with the metric's own direction). ``skill`` is defined
    only for loss-type metrics as ``1 - model / baseline``: 0 means no better than the ASFR
    schedule, 1 means perfect, negative means worse than the schedule. ``model_metric`` names the
    model-side statistic used, since the model's Brier is the finite-seed-corrected one while its
    MSE is the raw ``k/n`` rate — both are compared against the same (noise-free) baseline value.
    """
    rows = []
    for metric, model_key, base_key in _PAIRS:
        m = model_scores.get(model_key, float("nan"))
        b = baseline_scores.get(base_key, float("nan"))
        skill = float("nan")
        if metric in _LOSS_METRICS and pd.notna(m) and pd.notna(b) and b > 0:
            skill = 1.0 - m / b
        rows.append(
            {
                "metric": metric,
                "model_metric": model_key,
                "model": m,
                "baseline": b,
                "delta": m - b if pd.notna(m) and pd.notna(b) else float("nan"),
                "skill": skill,
            }
        )
    return pd.DataFrame(rows)
