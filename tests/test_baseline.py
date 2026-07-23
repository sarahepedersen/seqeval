"""ASFR baseline (04): schedule estimation, frozen-rate lookup, frame intervals, Poisson pricing.

The baseline is the reference every backtest score is read against, so the properties that matter
are (a) it recovers a known rate schedule, (b) it never reads a rate cell later than the person's
own jump-off year — the no-hindsight guarantee — and (c) the intensity it integrates matches the
closed form for a constant hazard.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from seqeval.core.slicing import AgeBins
from seqeval.core.specs import CountQuery, Frame, FramedOutcome, TTESpec
from seqeval.metrics import baseline as bl
from seqeval.units import DAYS_PER_YEAR
from seqeval.units import years_to_days as yd

BINS = AgeBins.from_years(15, 50, 1.0)
BIRTH = "birth"


# =================================================================================================
# fixtures: a population with an exactly known, flat fertility schedule
# =================================================================================================
def _flat_population(rate: float, n: int = 4000, seed: int = 0, cohorts: int = 61):
    """``n`` women born across ``cohorts`` successive years, births at a constant hazard 15–50.

    A flat hazard makes every quantity closed-form: the ASFR in every age cell is ``rate``, and the
    probability of at least one birth over ``W`` years is ``1 - exp(-rate * W)``.

    The cohorts must overlap in calendar time for the baseline to be constructible at all: a rate
    for (age ``a``, year ``y``) can only be estimated if some cohort is aged ``a`` in ``y``, and
    the frozen lookup refuses to read years after the person's own jump-off. A single-cohort panel
    therefore has no history above each person's jump-off age — see
    :func:`test_single_cohort_panel_cannot_price_ages_above_the_jumpoff`.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for pid in range(n):
        age = 15.0
        while True:
            age += rng.exponential(1.0 / rate)
            if age >= 50.0:
                break
            rows.append((pid, yd(age), BIRTH))
        rows.append((pid, yd(50.0), "no_event"))  # trailing time marker -> span reaches 50
    observed = pd.DataFrame(rows, columns=["person_id", "age", "event"])
    observed["age"] = observed["age"].astype(np.int32)
    persons = pd.DataFrame(
        {"person_id": np.arange(n), "birth_year": 1900 + (np.arange(n) % cohorts)}
    )
    return observed, persons


@pytest.fixture(scope="module")
def flat():
    return _flat_population(rate=0.15)


# =================================================================================================
# 1. schedule
# =================================================================================================
def test_schedule_recovers_a_flat_hazard(flat):
    observed, persons = flat
    sched = bl.asfr_schedule(observed, persons, birth_event=BIRTH, bins=BINS)

    # Restrict to well-supported cells; a flat hazard means every one estimates the same rate.
    solid = sched[sched["person_years"] > 40]
    assert len(solid) > 20
    assert solid["asfr"].mean() == pytest.approx(0.15, rel=0.05)


def test_schedule_blanks_thin_cells(flat):
    observed, persons = flat
    sched = bl.asfr_schedule(observed, persons, birth_event=BIRTH, bins=BINS, min_person_years=1e9)
    assert sched["asfr"].isna().all()


# =================================================================================================
# 2. frozen rates — the no-hindsight guarantee
# =================================================================================================
def _schedule(cells: dict[tuple[int, float], float]) -> pd.DataFrame:
    return pd.DataFrame([{"year": y, "age_bin": a, "asfr": r} for (y, a), r in cells.items()])


def test_frozen_rates_carry_forward_never_backward():
    sched = _schedule({(2000, 30.0): 0.2, (2004, 30.0): 0.5})
    frozen = bl.frozen_rates(sched).set_index("year")

    assert frozen.loc[2000, "asfr_frozen"] == 0.2
    assert not frozen.loc[2000, "is_fallback"]
    # 2002 has no cell of its own: carried forward from 2000, and flagged as such.
    assert frozen.loc[2002, "asfr_frozen"] == 0.2
    assert bool(frozen.loc[2002, "is_fallback"])
    # 2004's higher rate must not leak backwards into 2002/2003.
    assert frozen.loc[2003, "asfr_frozen"] == 0.2
    assert frozen.loc[2004, "asfr_frozen"] == 0.5


def test_frozen_rates_leave_pre_history_missing():
    sched = _schedule({(2000, 30.0): 0.2, (2001, 30.0): 0.3, (2000, 31.0): 0.4})
    frozen = bl.frozen_rates(sched)
    # age 31 has no rate in 2000... it does (0.4); age 30 in the *first* year is present.
    # The property under test: nothing before the schedule's first year is invented.
    assert frozen["year"].min() == 2000


def test_baseline_ignores_rates_after_the_jumpoff_year(flat):
    """Inflating the schedule strictly *after* each person's jump-off year cannot move their p.

    This is the no-hindsight guarantee stated as a test: the earliest jump-off year in the panel is
    the earliest cell any person may read, so scaling every cell after the *latest* jump-off year
    must leave every probability bit-identical.
    """
    observed, persons = flat
    sched = bl.asfr_schedule(observed, persons, birth_event=BIRTH, bins=BINS)
    spec = CountQuery("b1w5", BIRTH, 1, Frame("within", yd(5)))
    jumpoff = yd(30)
    latest_jumpoff_year = int(persons["birth_year"].max()) + 30

    base = bl.baseline_probability(
        observed, persons, spec, schedule=sched, jumpoff=jumpoff, bins=BINS
    )
    tampered = sched.copy()
    tampered.loc[tampered["year"] > latest_jumpoff_year, "asfr"] *= 10.0
    after = bl.baseline_probability(
        observed, persons, spec, schedule=tampered, jumpoff=jumpoff, bins=BINS
    )

    pd.testing.assert_series_equal(base["p_base"], after["p_base"])


def test_single_cohort_panel_cannot_price_ages_above_the_jumpoff():
    """One birth cohort means no observed history above the jump-off age — and it is reported.

    With every person born the same year, the only (age, year) cells that exist at a person's
    jump-off year are their own age. Everything the frame asks about lies in later years, which the
    frozen lookup is forbidden from reading, so the exposure is unmatched rather than silently
    priced. The baseline needs cohorts that overlap in calendar time.
    """
    observed, persons = _flat_population(rate=0.15, n=400, cohorts=1)
    sched = bl.asfr_schedule(observed, persons, birth_event=BIRTH, bins=BINS)
    out = bl.baseline_probability(
        observed,
        persons,
        CountQuery("b1w5", BIRTH, 1, Frame("within", yd(5))),
        schedule=sched,
        jumpoff=yd(30),
        bins=BINS,
    )
    # Only the jump-off age itself is priceable: 1 of the 5 age-years in the frame.
    assert out["unmatched_fraction"].mean() == pytest.approx(0.8, abs=0.02)


# =================================================================================================
# 3. frame intervals
# =================================================================================================
def _tiny_observed() -> pd.DataFrame:
    """Two people: 0 has one birth at 22 and one at 26; 1 has none."""
    return pd.DataFrame(
        {
            "person_id": [0, 0, 0, 1],
            "age": [yd(22), yd(26), yd(40), yd(40)],
            "event": [BIRTH, BIRTH, "no_event", "no_event"],
        }
    )


def test_frame_intervals_within_and_by_age():
    obs = _tiny_observed()
    jumpoff = yd(30)

    within = bl.frame_intervals(
        obs, CountQuery("c", BIRTH, 2, Frame("within", yd(5))), jumpoff=jumpoff
    )
    assert set(within["lo"]) == {jumpoff}
    assert set(within["hi"]) == {jumpoff + yd(5)}
    # A count query asks only about post-jump-off events, so the prefix parity is irrelevant.
    assert set(within["n_needed"]) == {2}

    spec = FramedOutcome("third", TTESpec(BIRTH, occurrence=3), Frame("by_age", yd(40)))
    by_age = bl.frame_intervals(obs, spec, jumpoff=jumpoff).set_index("person_id")
    assert by_age.loc[0, "hi"] == yd(40)
    # Person 0 already has 2 births in the prefix -> needs 1 more; person 1 needs all 3.
    assert by_age.loc[0, "n_needed"] == 1
    assert by_age.loc[1, "n_needed"] == 3


def test_frame_intervals_within_origin_requires_a_known_origin():
    obs = _tiny_observed()
    spec = FramedOutcome(
        "second",
        TTESpec(BIRTH, occurrence=2, origin=TTESpec(BIRTH, occurrence=1)),
        Frame("within_origin", yd(5)),
    )
    iv = bl.frame_intervals(obs, spec, jumpoff=yd(24)).set_index("person_id")

    # Person 1 never has the origin, so no interval is knowable — dropped.
    assert list(iv.index) == [0]
    # Origin at 22, jump-off at 24: the window still open is (24, 27].
    assert iv.loc[0, "lo"] == yd(24)
    assert iv.loc[0, "hi"] == yd(22) + yd(5)


def test_frame_intervals_drop_origins_after_the_jumpoff():
    """An origin that has not happened by the jump-off is not knowable at prediction time."""
    obs = _tiny_observed()
    spec = FramedOutcome(
        "second",
        TTESpec(BIRTH, occurrence=2, origin=TTESpec(BIRTH, occurrence=1)),
        Frame("within_origin", yd(5)),
    )
    assert bl.frame_intervals(obs, spec, jumpoff=yd(20)).empty


# =================================================================================================
# 4. the probability itself
# =================================================================================================
def test_poisson_pricing_matches_the_closed_form(flat):
    """With a flat hazard r over a W-year frame, p_base must be 1 - exp(-rW)."""
    observed, persons = flat
    sched = bl.asfr_schedule(observed, persons, birth_event=BIRTH, bins=BINS)
    spec = CountQuery("b1w5", BIRTH, 1, Frame("within", yd(5)))

    out = bl.baseline_probability(
        observed, persons, spec, schedule=sched, jumpoff=yd(30), bins=BINS
    )
    assert len(out) == observed["person_id"].nunique()
    assert out["exposure_years"].mean() == pytest.approx(5.0, rel=1e-3)

    # Judge the closed form only where the panel actually has rate history for the whole frame;
    # cohorts at the edge of the calendar range are priced on part of it (and say so).
    priced = out[out["unmatched_fraction"] == 0]
    assert len(priced) > 0.5 * len(out)
    assert priced["lambda_hat"].mean() == pytest.approx(0.15 * 5.0, rel=0.05)
    assert priced["p_base"].mean() == pytest.approx(1 - np.exp(-0.15 * 5.0), rel=0.05)


def test_more_needed_events_lowers_the_probability(flat):
    observed, persons = flat
    sched = bl.asfr_schedule(observed, persons, birth_event=BIRTH, bins=BINS)
    one = bl.baseline_probability(
        observed,
        persons,
        CountQuery("b1", BIRTH, 1, Frame("within", yd(5))),
        schedule=sched,
        jumpoff=yd(30),
        bins=BINS,
    )
    two = bl.baseline_probability(
        observed,
        persons,
        CountQuery("b2", BIRTH, 2, Frame("within", yd(5))),
        schedule=sched,
        jumpoff=yd(30),
        bins=BINS,
    )
    assert two["p_base"].mean() < one["p_base"].mean()


def test_exposure_outside_the_schedule_age_range_is_not_priced(flat):
    """A frame lying entirely above the fertile upper bound gets no intensity and p_base = 0."""
    observed, persons = flat
    sched = bl.asfr_schedule(observed, persons, birth_event=BIRTH, bins=BINS)
    out = bl.baseline_probability(
        observed,
        persons,
        CountQuery("late", BIRTH, 1, Frame("within", yd(5))),
        schedule=sched,
        jumpoff=yd(55),
        bins=BINS,
    )
    assert (out["p_base"] == 0).all()


def test_unmatched_exposure_is_reported_not_hidden():
    """A jump-off year before the schedule's first year prices nothing and says so."""
    observed = pd.DataFrame(
        {"person_id": [0, 0], "age": [yd(20), yd(50)], "event": [BIRTH, "no_event"]}
    )
    persons = pd.DataFrame({"person_id": [0], "birth_year": [1900]})
    sched = _schedule({(2000, 30.0): 0.2, (2001, 30.0): 0.2})

    out = bl.baseline_probability(
        observed,
        persons,
        CountQuery("b1", BIRTH, 1, Frame("within", yd(5))),
        schedule=sched,
        jumpoff=yd(30),
        bins=BINS,
    )
    assert out["unmatched_fraction"].iloc[0] == pytest.approx(1.0)
    assert out["p_base"].iloc[0] == 0.0


# =================================================================================================
# 5. scoring and comparison
# =================================================================================================
def _joined(p_model, p_base, y):
    n = np.full(len(y), 10)
    return pd.DataFrame(
        {
            "p_hat": p_model,
            "p_base": p_base,
            "y_true": y,
            "k": np.round(np.asarray(p_model) * 10).astype(int),
            "n": n,
        }
    )


def test_score_of_a_perfect_baseline():
    y = np.array([0, 0, 1, 1])
    s = bl.score(_joined(y.astype(float), y.astype(float), y))
    assert s["brier"] == pytest.approx(0.0)
    assert s["mse"] == pytest.approx(0.0)
    assert s["r2"] == pytest.approx(1.0)
    assert s["roc_auc"] == pytest.approx(1.0)


def test_skill_is_zero_when_the_model_matches_the_baseline():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.2, 0.8, 200)
    y = rng.binomial(1, p)
    j = _joined(p, p, y)
    cmp_ = bl.compare(
        {"brier_corrected": bl.score(j, p_col="p_hat")["brier"], "roc_auc": 0.7}, bl.score(j)
    ).set_index("metric")
    assert cmp_.loc["brier", "skill"] == pytest.approx(0.0, abs=1e-12)


def test_skill_is_positive_when_the_model_beats_the_baseline():
    y = np.array([0, 0, 1, 1])
    cmp_ = bl.compare(
        {"brier_corrected": 0.05},
        {"brier": 0.25, "mse": 0.25, "r2": 0.0, "ece": 0.1, "roc_auc": 0.5},
    ).set_index("metric")
    assert cmp_.loc["brier", "skill"] == pytest.approx(0.8)
    assert cmp_.loc["brier", "delta"] == pytest.approx(-0.20)
    assert cmp_.loc["brier", "model_metric"] == "brier_corrected"
    assert len(y)  # keep the example concrete


def test_skill_is_negative_when_the_baseline_wins():
    cmp_ = bl.compare(
        {"brier_corrected": 0.30},
        {"brier": 0.25, "mse": 0.25, "r2": 0.0, "ece": 0.1, "roc_auc": 0.5},
    ).set_index("metric")
    assert cmp_.loc["brier", "skill"] < 0


def test_auc_row_reports_a_difference_not_a_skill():
    cmp_ = bl.compare(
        {"roc_auc": 0.75}, {"brier": 0.25, "mse": 0.25, "r2": 0.0, "ece": 0.1, "roc_auc": 0.60}
    ).set_index("metric")
    assert np.isnan(cmp_.loc["roc_auc", "skill"])
    assert cmp_.loc["roc_auc", "delta"] == pytest.approx(0.15)


def test_person_years_use_the_shared_day_conversion(flat):
    """Exposure is person-days / DAYS_PER_YEAR — the one sanctioned rate-step conversion."""
    observed, persons = flat
    sched = bl.asfr_schedule(observed, persons, birth_event=BIRTH, bins=BINS)
    out = bl.baseline_probability(
        observed,
        persons,
        CountQuery("b1", BIRTH, 1, Frame("within", yd(4))),
        schedule=sched,
        jumpoff=yd(30),
        bins=BINS,
    )
    assert out["exposure_years"].iloc[0] == pytest.approx(yd(4) / DAYS_PER_YEAR, rel=1e-9)
