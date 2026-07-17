"""Survival metrics: exact KM on fixtures, life table, key-agnosticism, synthetic convergence."""

from __future__ import annotations

import numpy as np
import pytest

from seqeval.core import outcomes as O
from seqeval.core.slicing import AgeBins
from seqeval.core.specs import TTESpec
from seqeval.metrics import survival as SV
from seqeval.units import years_to_days as yd
from tests import synthetic as S
from tests.fixtures import tiny

OBS_KEYS = ["person_id"]
GEN_KEYS = ["person_id", "seed", "age_start", "age_stop"]


def test_km_exact_on_fixture():
    obs = tiny.observed_fixture()
    tte = O.time_to_event(obs, OBS_KEYS, TTESpec("birth", 1))
    km = SV.kaplan_meier(tte).set_index("time")["survival"]
    for age_years, expected in tiny.EXPECTED_KM_FIRST_BIRTH:
        assert km.loc[yd(age_years)] == pytest.approx(expected)


def test_km_confidence_band_present_and_ordered():
    obs = tiny.observed_fixture()
    tte = O.time_to_event(obs, OBS_KEYS, TTESpec("birth", 1))
    km = SV.kaplan_meier(tte)
    interior = km[(km["survival"] > 0) & (km["survival"] < 1)]
    assert (interior["ci_lo"] <= interior["survival"] + 1e-9).all()
    assert (interior["survival"] <= interior["ci_hi"] + 1e-9).all()


def test_median_survival():
    obs = tiny.observed_fixture()
    tte = O.time_to_event(obs, OBS_KEYS, TTESpec("birth", 1))
    km = SV.kaplan_meier(tte)
    med = SV.median_survival(km)
    assert med.iloc[0]["median"] == yd(25)  # survival first reaches 0.5 at age 25


def test_km_stratified_by_returns_one_curve_per_stratum():
    rng = np.random.default_rng(0)
    obs, pers = S.simulate_cohort(600, (1970, 1971), S.default_hazards(), None, rng)
    tte = O.time_to_event(obs, OBS_KEYS, TTESpec("birth", 1)).merge(
        pers[["person_id", "birth_year"]], on="person_id"
    )
    km = SV.kaplan_meier(tte, by=["birth_year"])
    assert set(km["birth_year"].unique()) == {1970, 1971}


def test_km_key_agnostic_on_generated():
    rng = np.random.default_rng(1)
    obs, pers = S.simulate_cohort(400, (1970, 1975), S.default_hazards(), None, rng)
    gen = S.simulate_generated(obs, pers, S.default_hazards(), [(0.0, 0.0)], 3, rng)
    tte = O.time_to_event(gen, GEN_KEYS, TTESpec("birth", 1))
    km = SV.kaplan_meier(tte, by=["seed", "age_start", "age_stop"])
    assert set(km["seed"].unique()) == {0, 1, 2}


def test_km_plateau_matches_never_birth_fraction():
    rng = np.random.default_rng(2)
    obs, pers = S.simulate_cohort(
        4000, (1960, 1970), S.default_hazards(), None, rng, no_event_fraction=1.0
    )
    tte = O.time_to_event(obs, OBS_KEYS, TTESpec("birth", 1))
    km = SV.kaplan_meier(tte)
    never = 1 - obs[obs["event"] == "birth"]["person_id"].nunique() / pers["person_id"].nunique()
    assert km["survival"].iloc[-1] == pytest.approx(never, abs=0.03)


def test_life_table_birth_counts_per_parity():
    obs = tiny.observed_fixture()
    births = O.births(obs, OBS_KEYS, birth_event="birth")
    spans = O.observation_spans(obs, OBS_KEYS)
    bins = AgeBins.from_years(15, 50, 1)
    lt = SV.life_table(births, spans, max_parity=4, bins=bins)
    per_parity = lt.groupby("parity")["births"].sum()
    assert per_parity.loc[0] == 5  # first births (p1..p5)
    assert per_parity.loc[1] == 3  # second births (p2, p3, p5)
    assert per_parity.loc[2] == 1  # third birth (p3)
    assert (lt["person_years"] > 0).any()
