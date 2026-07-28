"""Survival metrics: exact KM on fixtures, key-agnosticism, synthetic convergence."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from seqeval.core import outcomes as O
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


def test_greenwood_variance_is_exposed_and_consistent_with_the_log_log_ci():
    """The band builders need the variance on the survival scale, not just the interval."""
    rng = np.random.default_rng(4)
    dur = rng.integers(1, 100, 400)
    obs = rng.uniform(size=400) < 0.7
    km = SV.kaplan_meier(pd.DataFrame({"duration": dur, "observed": obs}))

    assert "greenwood_var" in km.columns
    assert (km["greenwood_var"].dropna() >= 0).all()
    # Greenwood's variance is S(t)^2 times the accumulated hazard term, so it grows as the curve
    # descends and information thins
    finite = km.dropna(subset=["greenwood_var"])
    assert finite["greenwood_var"].iloc[-1] > finite["greenwood_var"].iloc[0]
    # and it agrees with the log-log interval already reported: both come from the same cum_v
    row = km[km["ci_lo"].notna()].iloc[len(km[km["ci_lo"].notna()]) // 2]
    se_loglog = np.sqrt(row["greenwood_var"]) / (row["survival"] * abs(np.log(row["survival"])))
    expected = row["survival"] ** np.exp(1.959963985 * se_loglog)
    assert row["ci_lo"] == pytest.approx(expected, rel=1e-9)
