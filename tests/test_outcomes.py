"""Outcome extractors: exact fixture values, symmetry, exposure properties, evaluator logic."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from seqeval.core import outcomes as O
from seqeval.core.slicing import AgeBins
from seqeval.core.specs import CountQuery, Frame, FramedOutcome, TTESpec
from seqeval.units import years_to_days as yd
from tests import synthetic as S
from tests.fixtures import tiny

OBS_KEYS = ["person_id"]
GEN_KEYS = ["person_id", "seed", "age_start", "age_stop"]


# --- births -------------------------------------------------------------------------------------
def test_births_orders_and_ages():
    obs = tiny.observed_fixture()
    b = O.births(obs, OBS_KEYS, birth_event="birth")
    assert b.groupby("person_id")["order"].max().to_dict() == {1: 1, 2: 2, 3: 3, 4: 1, 5: 2}
    # p3 birth orders map to ages 22, 26, 31
    p3 = b[b["person_id"] == 3].sort_values("order")["age"].tolist()
    assert p3 == [yd(22), yd(26), yd(31)]
    assert len(b) == (obs["event"] == "birth").sum()


# --- observation spans + exposure ---------------------------------------------------------------
def test_observation_spans_observed():
    obs = tiny.observed_fixture()
    sp = O.observation_spans(obs, OBS_KEYS).set_index("person_id")
    assert sp.loc[0, "start_age"] == 0
    assert sp.loc[0, "end_age"] == yd(28)  # childless, censored by the no-event marker
    assert sp.loc[3, "end_age"] == yd(31)


def test_observation_spans_generated_start_is_jumpoff():
    obs = tiny.observed_fixture()
    gen = obs.copy()
    gen["seed"] = np.int32(0)
    gen["age_start"] = np.int32(0)
    gen["age_stop"] = np.int32(yd(20))
    # keep only rows after the jump-off so the generated invariant holds
    gen = gen[gen["age"] > yd(20)]
    sp = O.observation_spans(gen, GEN_KEYS)
    assert (sp["start_age"] == yd(20)).all()


def test_exposure_tiny_band():
    obs = tiny.observed_fixture()
    sp = O.observation_spans(obs, OBS_KEYS)
    bins = AgeBins.from_years(25, 30, 5)
    exp = O.exposure(sp, bins=bins)
    assert int(exp["person_days"].sum()) == tiny.EXPECTED_PERSON_DAYS_25_30


def test_exposure_person_days_conserved():
    rng = np.random.default_rng(3)
    observed, _ = S.simulate_cohort(500, (1960, 1990), S.default_hazards(), None, rng)
    sp = O.observation_spans(observed, OBS_KEYS)
    bins = AgeBins.from_years(0, 60, 1)  # covers every age
    exp = O.exposure(sp, bins=bins)
    total_span = int((sp["end_age"] - sp["start_age"]).sum())
    assert int(exp["person_days"].sum()) == total_span  # exact integer arithmetic


def test_exposure_by_year_conserves_total():
    rng = np.random.default_rng(4)
    observed, persons = S.simulate_cohort(300, (1960, 1990), S.default_hazards(), None, rng)
    sp = O.observation_spans(observed, OBS_KEYS)
    bins = AgeBins.from_years(0, 60, 1)
    plain = O.exposure(sp, bins=bins)
    by_year = O.exposure(sp, bins=bins, persons=persons, by_year=True)
    assert int(by_year["person_days"].sum()) == int(plain["person_days"].sum())


# --- time to event ------------------------------------------------------------------------------
def test_time_to_first_birth_durations_and_censoring():
    obs = tiny.observed_fixture()
    tte = TTESpec(target="birth", occurrence=1)
    tt = O.time_to_event(obs, OBS_KEYS, tte).set_index("person_id")
    assert tt.loc[1, "duration"] == yd(25) and bool(tt.loc[1, "observed"])
    # childless p0: censored at end_age (age 28), observed False
    assert tt.loc[0, "duration"] == yd(28) and not bool(tt.loc[0, "observed"])


def test_time_to_second_birth_uses_origin():
    obs = tiny.observed_fixture()
    tte = TTESpec(target="birth", occurrence=2, origin=TTESpec("birth", 1))
    tt = O.time_to_event(obs, OBS_KEYS, tte).set_index("person_id")
    # origin conditioning: p0 (no first birth) and p1/p4 (only one birth) still appear only if they
    # have a first birth. p1/p4 have a first birth so they appear (censored second birth).
    assert 0 not in tt.index  # no first birth -> dropped
    # p2: births 24, 29 -> duration 29-24
    assert tt.loc[2, "duration"] == yd(29) - yd(24) and bool(tt.loc[2, "observed"])
    # p1: first birth 25, no second -> censored at end_age(25) - origin(25) = 0, observed False
    assert not bool(tt.loc[1, "observed"])


def test_symmetry_observed_vs_generated_zero_window():
    obs = tiny.observed_fixture()
    gen = obs.copy()
    gen["seed"] = np.int32(0)
    gen["age_start"] = np.int32(0)
    gen["age_stop"] = np.int32(0)  # jump-off at birth => start_age 0, mirrors observed
    tte = TTESpec(target="birth", occurrence=1)

    obs_tte = O.time_to_event(obs, OBS_KEYS, tte)[["person_id", "duration", "observed"]]
    gen_tte = O.time_to_event(gen, GEN_KEYS, tte)[["person_id", "duration", "observed"]]
    pd.testing.assert_frame_equal(obs_tte.reset_index(drop=True), gen_tte.reset_index(drop=True))

    obs_b = O.births(obs, OBS_KEYS, birth_event="birth")[["person_id", "order", "age"]]
    gen_b = O.births(gen, GEN_KEYS, birth_event="birth")[["person_id", "order", "age"]]
    pd.testing.assert_frame_equal(obs_b, gen_b)


# --- evaluators ---------------------------------------------------------------------------------
def _people(rows):
    df = pd.DataFrame(
        {
            "person_id": [r[0] for r in rows],
            "age": np.array([yd(r[1]) for r in rows], dtype=np.int32),
            "event": [r[2] for r in rows],
        }
    )
    df["event"] = df["event"].astype("category")
    return df


def test_framed_by_age_worked_example():
    # 00a worked example: jump-off 30, second_birth by_age 35.
    df = _people(
        [
            ("A", 27, "birth"),
            ("A", 33, "birth"),
            ("A", 40, "no_event"),
            ("B", 22, "birth"),
            ("B", 25, "birth"),
            ("B", 40, "no_event"),
            ("C", 40, "no_event"),
        ]
    )
    sp = O.observation_spans(df, OBS_KEYS)
    spec = FramedOutcome(
        "sb35", TTESpec("birth", 2, origin=TTESpec("birth", 1)), Frame("by_age", yd(35))
    )
    res = O.evaluate_framed(df, OBS_KEYS, spec, sp, jumpoff=yd(30)).set_index("person_id")
    assert res.loc["A"].to_dict() == {"occurred": True, "evaluable": True}
    assert res.loc["B"].to_dict() == {"occurred": True, "evaluable": False}  # settled at jump-off
    assert res.loc["C"].to_dict() == {"occurred": False, "evaluable": True}


def test_framed_negative_determination_settled():
    # by_age A entirely at/before jump-off -> settled negative (frame in observed region).
    df = _people([("X", 40, "no_event")])
    sp = O.observation_spans(df, OBS_KEYS)
    spec = FramedOutcome("fb28", TTESpec("birth", 1), Frame("by_age", yd(28)))
    res = O.evaluate_framed(df, OBS_KEYS, spec, sp, jumpoff=yd(30)).iloc[0]
    assert not bool(res["evaluable"])


def test_framed_non_evaluable_when_span_ends_inside_frame():
    # negative outcome but span ends before frame upper -> censored, non-evaluable.
    df = _people([("Y", 27, "no_event")])  # observed only to 27, frame wants coverage to 35
    sp = O.observation_spans(df, OBS_KEYS)
    spec = FramedOutcome("fb35", TTESpec("birth", 1), Frame("by_age", yd(35)))
    res = O.evaluate_framed(df, OBS_KEYS, spec, sp, jumpoff=yd(20)).iloc[0]
    assert not bool(res["occurred"]) and not bool(res["evaluable"])


def test_framed_within_origin_drops_missing_origin():
    df = _people(
        [
            ("M", 25, "birth"),
            ("M", 28, "birth"),
            ("M", 40, "no_event"),  # origin present
            ("N", 40, "no_event"),  # no first birth -> dropped
        ]
    )
    sp = O.observation_spans(df, OBS_KEYS)
    spec = FramedOutcome(
        "sb_wo5", TTESpec("birth", 2, origin=TTESpec("birth", 1)), Frame("within_origin", yd(5))
    )
    res = O.evaluate_framed(df, OBS_KEYS, spec, sp, jumpoff=yd(20))
    assert set(res["person_id"]) == {"M"}
    assert bool(res.set_index("person_id").loc["M", "occurred"])  # 28 within 5y of 25


def test_count_pre_jumpoff_events_not_counted():
    # 00a: births before the jump-off never count for a count query.
    df = _people([("B", 22, "birth"), ("B", 25, "birth"), ("B", 40, "no_event")])
    sp = O.observation_spans(df, OBS_KEYS)
    spec = CountQuery("b1w5", "birth", 1, Frame("within", yd(5)))
    res = O.evaluate_count(df, OBS_KEYS, spec, sp, jumpoff=yd(30)).iloc[0]
    assert not bool(res["occurred"]) and bool(res["evaluable"])


def test_count_post_jumpoff_event_counts():
    df = _people([("A", 33, "birth"), ("A", 40, "no_event")])
    sp = O.observation_spans(df, OBS_KEYS)
    spec = CountQuery("b1w5", "birth", 1, Frame("within", yd(5)))
    res = O.evaluate_count(df, OBS_KEYS, spec, sp, jumpoff=yd(30)).iloc[0]
    assert bool(res["occurred"]) and bool(res["evaluable"])


def test_count_requires_jumpoff():
    df = _people([("A", 33, "birth")])
    sp = O.observation_spans(df, OBS_KEYS)
    spec = CountQuery("b1w5", "birth", 1, Frame("within", yd(5)))
    with pytest.raises(ValueError, match="requires a jumpoff"):
        O.evaluate_count(df, OBS_KEYS, spec, sp)
