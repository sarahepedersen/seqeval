"""Loaders: dtype round-trip, unit normalization equivalence, pushdown, cross-artifact checks."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from seqeval.config import load_config
from seqeval.io.loaders import (
    Bundle,
    load_all,
    load_generated,
    load_observed,
    load_persons,
)
from seqeval.io.schema import SchemaError
from seqeval.units import years_to_days
from tests import synthetic as S


@pytest.fixture
def cohort():
    rng = np.random.default_rng(1)
    hazards = S.default_hazards()
    observed, persons = S.simulate_cohort(200, (1960, 1990), hazards, None, rng)
    generated = S.simulate_generated(observed, persons, hazards, [(0.0, 25.0), (0.0, 30.0)], 4, rng)
    return observed, persons, generated


def test_observed_round_trip_dtypes(tmp_path, cohort):
    observed = cohort[0]
    path = tmp_path / "observed.parquet"
    observed.to_parquet(path, index=False)
    loaded = load_observed(path, age_unit="days")
    assert loaded["age"].dtype == np.int32
    assert str(loaded["event"].dtype) == "category"


def test_years_and_days_normalize_identically(tmp_path):
    # Same data expressed in years vs days must load to identical canonical frames.
    ages_years = np.array([0.0, 25.0, 30.0], dtype=float)
    base = pd.DataFrame(
        {"person_id": np.array([1, 1, 1], dtype=np.int64), "event": ["no_event", "birth", "birth"]}
    )

    years_df = base.assign(age=ages_years)
    days_df = base.assign(age=np.array([years_to_days(y) for y in ages_years], dtype=np.int64))

    p_years = tmp_path / "years.parquet"
    p_days = tmp_path / "days.parquet"
    years_df.to_parquet(p_years, index=False)
    days_df.to_parquet(p_days, index=False)

    from_years = load_observed(p_years, age_unit="years")
    from_days = load_observed(p_days, age_unit="days")
    pd.testing.assert_frame_equal(from_years, from_days)


def test_days_unit_rejects_non_integral(tmp_path):
    df = pd.DataFrame({"person_id": [1], "age": [25.5], "event": ["birth"]})
    path = tmp_path / "bad.parquet"
    df.to_parquet(path, index=False)
    with pytest.raises(SchemaError, match="non-integral"):
        load_observed(path, age_unit="days")


def test_window_pushdown_returns_exactly_requested(tmp_path, cohort):
    generated = cohort[2]
    path = tmp_path / "generated.parquet"
    generated.to_parquet(path, index=False)

    want = [(0, years_to_days(25))]
    loaded = load_generated(path, age_unit="days", windows=want)
    got = set(map(tuple, loaded[["age_start", "age_stop"]].drop_duplicates().to_numpy()))
    assert got == set(want)


def test_seed_pushdown(tmp_path, cohort):
    generated = cohort[2]
    path = tmp_path / "generated.parquet"
    generated.to_parquet(path, index=False)
    loaded = load_generated(path, age_unit="days", seeds=[0, 1])
    assert set(loaded["seed"].unique()) <= {0, 1}


def test_trailing_no_event_rows_kept(tmp_path, cohort):
    observed = cohort[0]
    path = tmp_path / "observed.parquet"
    observed.to_parquet(path, index=False)
    loaded = load_observed(path, age_unit="days")
    assert (loaded["event"] == S.NO_EVENT_TOKEN).any()


def test_load_persons_missing_covariate_errors(tmp_path, cohort):
    persons = cohort[1]
    path = tmp_path / "persons.parquet"
    persons.to_parquet(path, index=False)
    with pytest.raises(SchemaError, match="not in file"):
        load_persons(path, covariates=["education", "nonexistent"])


def _write_bundle_files(tmp_path, cohort):
    observed, persons, generated = cohort
    observed.to_parquet(tmp_path / "observed.parquet", index=False)
    generated.to_parquet(tmp_path / "generated.parquet", index=False)
    persons.to_parquet(tmp_path / "persons.parquet", index=False)
    pd.DataFrame({"model_representation": ["birth"], "event_definition": ["live birth"]}).to_csv(
        tmp_path / "events.csv", index=False
    )


_CONFIG = """\
model: {name: t}
data:
  observed: observed.parquet
  generated: generated.parquet
  persons: persons.parquet
  event_definitions: events.csv
  age_unit: days
events: {birth: birth}
persons: {covariates: [education, region]}
"""


def test_load_all_end_to_end(tmp_path, cohort):
    _write_bundle_files(tmp_path, cohort)
    (tmp_path / "config.yaml").write_text(_CONFIG)
    bundle = load_all(load_config(tmp_path / "config.yaml"))
    assert isinstance(bundle, Bundle)
    assert bundle.population_summary()["n_persons"] == 200
    assert len(bundle.available_windows()) == 2
    assert bundle.token("birth") == "birth"
    assert bundle.label("birth") == "live birth"


def test_load_all_drops_unknown_generated_persons(tmp_path, cohort, caplog):
    observed, persons, generated = cohort
    # Inject a generated row for a person absent from observed.
    rogue = generated.iloc[[0]].copy()
    rogue["person_id"] = 999999
    generated = pd.concat([generated, rogue], ignore_index=True)
    _write_bundle_files(tmp_path, (observed, persons, generated))
    (tmp_path / "config.yaml").write_text(_CONFIG)

    import logging

    with caplog.at_level(logging.WARNING, logger="seqeval"):
        bundle = load_all(load_config(tmp_path / "config.yaml"))
    assert 999999 not in set(bundle.generated["person_id"].unique())
    assert any("absent from observed" in r.message for r in caplog.records)


def test_require_persons_raises_when_absent(cohort):
    observed = cohort[0]
    bundle = Bundle(observed=observed, generated=None, persons=None, event_defs=None, events=None)
    with pytest.raises(ValueError, match="requires a persons file"):
        bundle.require_persons("cohort ASFR")
