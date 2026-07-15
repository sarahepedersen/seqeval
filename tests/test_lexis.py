"""Lexis surface (05): cell values conserve events; combined surface marks the forecast region."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from seqeval.arms.forecasting import _combine_surfaces
from seqeval.core import outcomes as O
from seqeval.core.slicing import AgeBins
from seqeval.metrics import fertility as FE
from tests import synthetic as S

OBS_KEYS = ["person_id"]


def test_lexis_surface_conserves_and_rates():
    rng = np.random.default_rng(0)
    obs, pers = S.simulate_cohort(2000, (1960, 1980), S.default_hazards(), None, rng)
    b = O.births(obs, OBS_KEYS, birth_event="birth")
    sp = O.observation_spans(obs, OBS_KEYS)
    bins = AgeBins.from_years(12, 55, 1)
    surface = FE.lexis_surface(b, sp, pers, occurrence=1, bins=bins, year_range=(1975, 2035))

    # every first birth (whose calendar year is in range) lands in exactly one cell
    first = b[b["order"] == 1]
    assert surface["n_events"].sum() == len(first)
    # rate = events / person-years, exactly
    row = surface[surface["n_events"] > 0].iloc[0]
    assert row["rate"] == pytest.approx(row["n_events"] / row["person_years"])


def test_lexis_cohort_basis_bands_and_conserves():
    rng = np.random.default_rng(1)
    obs, pers = S.simulate_cohort(2000, (1960, 1974), S.default_hazards(), None, rng)
    b = O.births(obs, OBS_KEYS, birth_event="birth")
    sp = O.observation_spans(obs, OBS_KEYS)
    bins = AgeBins.from_years(12, 55, 1)
    surface = FE.lexis_surface(
        b,
        sp,
        pers,
        occurrence=1,
        bins=bins,
        year_range=(1975, 2035),
        basis="cohort",
        cohort_width=5,
    )
    assert "cohort" in surface.columns and "year" not in surface.columns
    assert set(surface["cohort"].unique()) == {1960, 1965, 1970}  # 5-year bands
    assert surface["n_events"].sum() == len(b[b["order"] == 1])  # cohort basis has no year filter


def _surface(cells, seed=None):
    df = pd.DataFrame(cells, columns=["year", "age_bin"])
    df["rate"] = 0.1
    df["n_events"] = 5
    df["person_years"] = 50.0
    if seed is not None:
        df["seed"] = seed
    return df


def test_combined_forecast_only_beyond_observed():
    observed = _surface([(2000, 25), (2001, 26)])
    forecast = pd.concat(
        [_surface([(2001, 26), (2002, 27)], seed=s) for s in (0, 1)], ignore_index=True
    )
    combined = _combine_surfaces(observed, forecast, "year", subgroup=[])
    src = combined.set_index(["year", "age_bin"])["source"]
    assert src.loc[(2000, 25)] == "observed"
    assert src.loc[(2001, 26)] == "observed"  # present in observed -> not overwritten by forecast
    assert src.loc[(2002, 27)] == "forecast"  # only cell beyond the observed surface
    assert (combined["source"] == "forecast").sum() == 1
