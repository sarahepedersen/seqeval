"""Slicing helpers: truncation, windows, cohort/age binning, calendar year, count conditioning."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from seqeval.core import slicing as SL
from seqeval.core.specs import Condition
from seqeval.units import completed_years, years_to_days
from tests.fixtures import tiny


def test_truncate_and_restrict_window():
    obs = tiny.observed_fixture()
    keys = ["person_id"]
    t = SL.truncate(obs, keys, max_age=years_to_days(28))
    assert (t["age"] <= years_to_days(28)).all()
    w = SL.restrict_window(obs, keys, lo=years_to_days(25), hi=years_to_days(30))
    assert (w["age"] >= years_to_days(25)).all() and (w["age"] < years_to_days(30)).all()


def test_bad_keys_raise():
    obs = tiny.observed_fixture()
    with pytest.raises(ValueError, match="not in frame columns"):
        SL.truncate(obs, ["nope"], max_age=1)


def test_attach_persons_and_missing_raise():
    obs = tiny.observed_fixture()
    persons = tiny.persons_fixture()
    merged = SL.attach_persons(obs, persons)
    assert "birth_year" in merged.columns and "sex" in merged.columns
    assert len(merged) == len(obs)

    short = persons[persons["person_id"] != 0]
    with pytest.raises(ValueError, match="absent from persons"):
        SL.attach_persons(obs, short)


def test_cohort_bins_width_one_is_birth_year():
    persons = tiny.persons_fixture()
    cb = SL.cohort_bins(persons, width=1)
    assert cb.loc[0] == 1970
    assert cb.name == "cohort"


def test_cohort_bins_width_five():
    persons = tiny.persons_fixture()
    cb = SL.cohort_bins(persons, width=5, range=(1960, 2000))
    # 1970 -> 1970, 1975 -> 1975, 1980 -> 1980 (all aligned to 5-year edges from 1960)
    assert cb.loc[0] == 1970
    assert cb.loc[2] == 1975


def test_agebins_and_bin_ages():
    bins = SL.AgeBins.from_years(15, 45, 5)
    assert len(bins.edges_days) == 7
    assert len(bins.labels) == 6
    ages = pd.Series([years_to_days(16), years_to_days(31), years_to_days(50)])
    labels = SL.bin_ages(ages, bins)
    assert labels.iloc[0] == 15
    assert labels.iloc[1] == 30
    assert np.isnan(labels.iloc[2])  # 50 is outside [15, 45)


def test_calendar_year_requires_birth_year():
    obs = tiny.observed_fixture()
    with pytest.raises(ValueError, match="birth_year"):
        SL.calendar_year(obs)


def test_calendar_year_value():
    obs = tiny.observed_fixture()
    persons = tiny.persons_fixture()
    merged = SL.attach_persons(obs, persons)
    year = SL.calendar_year(merged)
    expected = merged["birth_year"].to_numpy() + completed_years(merged["age"].to_numpy())
    np.testing.assert_array_equal(year.to_numpy(), expected)


# --- condition_on_count -------------------------------------------------------------------------
def test_condition_needs_an_anchor():
    obs = tiny.observed_fixture()
    cond = Condition(name="p1", event="birth", min_count=1)
    with pytest.raises(ValueError, match="needs an anchor"):
        SL.condition_on_count(obs, ["person_id"], cond=cond)


@pytest.mark.parametrize(
    "cond,expected_ids",
    [
        # parity exactly 1 by age 40 (anchor): p1, p4
        (Condition("p1", "birth", min_count=1, max_count=1), {1, 4}),
        # childless by age 40: p0
        (Condition("p0", "birth", min_count=0, max_count=0), {0}),
        # at least 2 births: p2, p3, p5
        (Condition("p2plus", "birth", min_count=2), {2, 3, 5}),
    ],
)
def test_condition_on_count_bounds(cond, expected_ids):
    obs = tiny.observed_fixture()
    kept = SL.condition_on_count(obs, ["person_id"], cond=cond, anchor_age=years_to_days(40))
    assert set(kept["person_id"].unique()) == expected_ids


def test_condition_before_age_overrides_anchor():
    obs = tiny.observed_fixture()
    # parity exactly 1 before age 26: p1 (birth 25) and p2 (birth 24 only, 29 excluded) and p3
    # (birth 22 only within <=26? 22 yes, 26 yes -> 2). Let's assert p1 qualifies, p3 does not.
    cond = Condition("p1_by26", "birth", min_count=1, max_count=1, before_age=years_to_days(26))
    kept = set(
        SL.condition_on_count(obs, ["person_id"], cond=cond, anchor_age=years_to_days(40))[
            "person_id"
        ].unique()
    )
    assert 1 in kept  # p1 birth at 25 -> exactly 1 by 26
    assert 3 not in kept  # p3 births at 22 and 26 -> 2 by 26


def test_condition_identical_on_observed_and_generated_keys():
    obs = tiny.observed_fixture()
    # Build a generated-shaped frame: one seed, one window (0,0) so runs mirror persons.
    gen = obs.copy()
    gen["seed"] = np.int32(0)
    gen["age_start"] = np.int32(0)
    gen["age_stop"] = np.int32(0)
    cond = Condition("p1", "birth", min_count=1, max_count=1)
    obs_kept = set(
        SL.condition_on_count(obs, ["person_id"], cond=cond, anchor_age=years_to_days(40))[
            "person_id"
        ].unique()
    )
    gen_kept = set(
        SL.condition_on_count(
            gen,
            ["person_id", "seed", "age_start", "age_stop"],
            cond=cond,
            anchor_age=years_to_days(40),
        )["person_id"].unique()
    )
    assert obs_kept == gen_kept


def test_align_jumpoff_to_event():
    obs = tiny.observed_fixture()
    first = SL.align_jumpoff_to_event(obs, event="birth", occurrence=1)
    # every mother appears with her first-birth age; childless p0 absent
    assert set(first["person_id"]) == {1, 2, 3, 4, 5}
    assert first.set_index("person_id").loc[2, "age"] == years_to_days(24)
