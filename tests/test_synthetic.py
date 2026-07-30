"""Synthetic generator: determinism, range/parity invariants, CCF convergence (01 section 9)."""

from __future__ import annotations

import numpy as np
import pytest

from seqeval.io.schema import (
    GENERATED_SCHEMA,
    OBSERVED_SCHEMA,
    PERSONS_SCHEMA,
    validate,
)
from seqeval.units import days_to_years
from tests import synthetic as S


def _cohort(seed=0, n=1500):
    rng = np.random.default_rng(seed)
    hazards = S.default_hazards()
    observed, persons = S.simulate_cohort(n, (1960, 1990), hazards, None, rng)
    return hazards, observed, persons, rng


def test_determinism_under_fixed_rng():
    _, obs1, pers1, _ = _cohort(seed=42)
    _, obs2, pers2, _ = _cohort(seed=42)
    assert obs1.equals(obs2)
    assert pers1.equals(pers2)


def test_frames_are_schema_conformant():
    hazards, observed, persons, rng = _cohort()
    validate(observed, OBSERVED_SCHEMA, "observed")
    validate(persons, PERSONS_SCHEMA, "persons")
    generated = S.simulate_generated(observed, persons, hazards, [(0.0, 25.0)], 3, rng)
    validate(generated, GENERATED_SCHEMA, "generated")


def test_every_person_appears():
    _, observed, persons, _ = _cohort()
    assert set(observed["person_id"].unique()) == set(persons["person_id"].unique())


def test_birth_ages_within_fertile_range():
    hazards, observed, _, _ = _cohort()
    lo, hi = hazards.fertile_ages
    births = observed.loc[observed["event"] == S.BIRTH_TOKEN, "age"].to_numpy()
    ages_yr = days_to_years(births)
    assert ages_yr.min() >= lo
    assert ages_yr.max() <= hi


def test_parity_never_exceeds_max():
    hazards, observed, _, _ = _cohort()
    parity = observed[observed["event"] == S.BIRTH_TOKEN].groupby("person_id").size()
    assert parity.max() <= hazards.max_parity


def test_ccf_converges_to_expected():
    hazards, observed, persons, _ = _cohort(n=4000)
    n = persons["person_id"].nunique()
    empirical = (observed["event"] == S.BIRTH_TOKEN).sum() / n
    expected = S.expected_ccf(hazards)
    assert empirical == pytest.approx(expected, abs=0.1)


def test_perturb_scales_rates():
    hazards = S.default_hazards()
    doubled = S.perturb(hazards, 2.0)
    for k, v in hazards.rates.items():
        assert doubled.rates[k] == pytest.approx(2.0 * v)


def _unfinished_cohort(seed=0, n=1500, observation_year=2025):
    rng = np.random.default_rng(seed)
    hazards = S.default_hazards()
    observed, persons = S.simulate_cohort(
        n, (1960, 2000), hazards, None, rng, observation_year=observation_year
    )
    return hazards, observed, persons, rng


def test_observation_year_censors_young_cohorts():
    """Each person's sequence stops at age = observation_year - birth_year (capped at fertile end)."""
    hazards, observed, persons, _ = _unfinished_cohort()
    validate(observed, OBSERVED_SCHEMA, "observed")
    last = observed.groupby("person_id")["age"].max()
    end_yr = days_to_years(last.reindex(persons["person_id"]).to_numpy())
    expected = np.minimum(2025 - persons["birth_year"].to_numpy(), hazards.fertile_ages[1])
    tol = 1.0 / 365.25  # ages are rounded to whole days
    assert (end_yr <= expected + tol).all()
    # Young cohorts end mid-life-course; the oldest run to the top of the fertile range.
    young = persons["birth_year"].to_numpy() >= 1995
    assert end_yr[young].max() <= 30 + tol
    assert end_yr[~young].max() == pytest.approx(hazards.fertile_ages[1], abs=0.01)


def test_unfinished_sequences_are_forecast_not_replayed():
    """With ``require_observed_prefix`` a run only exists at jump-offs the person's data reaches."""
    hazards, observed, persons, rng = _unfinished_cohort()
    gen = S.simulate_generated(
        observed, persons, hazards, [(0.0, 25.0), (0.0, 30.0), (0.0, 35.0)], 3, rng,
        require_observed_prefix=True,
    )
    validate(gen, GENERATED_SCHEMA, "generated")

    obs_end = persons.set_index("person_id")["observed_through"]
    runs = gen[["person_id", "age_stop"]].drop_duplicates()
    assert (obs_end.reindex(runs["person_id"]).to_numpy() >= runs["age_stop"].to_numpy()).all()

    # Someone censored before the latest jump-off is absent from that window but present earlier.
    stops = gen.groupby("person_id")["age_stop"].max()
    young = persons.loc[persons["birth_year"] >= 1998, "person_id"]
    assert stops.reindex(young).max() < gen["age_stop"].max()


def test_generated_futures_strictly_after_jumpoff():
    hazards, observed, persons, rng = _cohort()
    generated = S.simulate_generated(observed, persons, hazards, [(0.0, 25.0), (0.0, 30.0)], 3, rng)
    assert (generated["age"] > generated["age_stop"]).all()
