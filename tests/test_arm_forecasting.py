"""Forecasting arm (05): smoke test + replicate-variance definitional checks."""

from __future__ import annotations

import numpy as np
import pandas as pd
import yaml

from seqeval.arms import forecasting as FC
from seqeval.arms._common import OutputWriter
from seqeval.config import (
    Config,
    EventConfig,
    resolve_outcomes,
    resolve_replicates,
    resolve_rules,
)
from seqeval.io.loaders import Bundle
from seqeval.units import years_to_days as yd
from tests import synthetic as S

_CFG = """
model: {name: perfect}
data: {observed: o.parquet, age_unit: days}
events: {birth: birth}
persons: {cohort_width: 5}
replicates: {min_replicates: 5}
outcomes: {first_birth: {event: birth, n: 1}}
arms:
  forecasting:
    lexis: {outcome: first_birth, ages: [12, 55], years: [1975, 2035], subgroup_by: []}
    illegal_moves:
      - {event: birth, max_age: 50}
      - {event: birth, min_spacing: 0.6, severity: warn}
    replicate_variance: {individual: true, aggregate: [ccf], subgroup_by: [cohort]}
"""


def _run(tmp_path, seeds=12):
    cfg = Config.model_validate(yaml.safe_load(_CFG))
    rng = np.random.default_rng(0)
    h = S.default_hazards()
    obs, pers = S.simulate_cohort(1000, (1960, 1985), h, None, rng, no_event_fraction=1.0)
    gen = S.simulate_generated(obs, pers, h, [(0.0, 25.0), (0.0, 30.0)], seeds, rng)
    bundle = Bundle(
        observed=obs,
        generated=gen,
        persons=pers,
        event_defs=None,
        events=EventConfig(birth="birth"),
    )
    out = OutputWriter(base_dir=tmp_path, arm="forecasting", model="perfect", individual_level=True)
    FC.run(
        bundle,
        cfg.arms.forecasting,
        out,
        outcomes=resolve_outcomes(cfg),
        rules=resolve_rules(cfg),
        replicate_spec=resolve_replicates(cfg),
        cohort_width=5,
    )
    return out


def test_arm_writes_all_tables_and_figures(tmp_path):
    out = _run(tmp_path)
    names = {p.name for p in out.written}
    assert {
        "lexis_cohort_forecast.parquet",
        "lexis_cohort_pooled.parquet",
        "lexis_cohort_combined.parquet",
        "violations.parquet",
        "violation_rates.parquet",
        "replicate_variance_individual.parquet",
        "replicate_variance_aggregate.parquet",
    } <= names
    # cohort (birth-cohort x age) is the only Lexis basis: a period cell is part observed and part
    # forecast, since the jump-off is an age and lands in a different calendar year for each cohort
    assert "lexis_cohort_combined.png" in names
    assert not [n for n in names if n.startswith("lexis_") and "cohort" not in n]
    # within-seed variance histograms: population-wide + faceted by the requested subgroup
    assert {"within_seed_variance.png", "within_seed_variance_by_cohort.png"} <= names
    for p in out.written:
        assert p.exists() and p.stat().st_size > 0


def test_lexis_forecast_cells_are_pooled_over_every_trajectory(tmp_path):
    """The surface drawn is one estimate over all N×K trajectories, not a per-cell seed summary."""
    out = _run(tmp_path)
    by_seed = pd.read_parquet(out.dir / "lexis_cohort_forecast.parquet")
    pooled = pd.read_parquet(out.dir / "lexis_cohort_pooled.parquet")
    combined = pd.read_parquet(out.dir / "lexis_cohort_combined.parquet")

    assert by_seed["seed"].nunique() > 1
    assert "seed" not in pooled.columns
    cell = ["cohort", "age_bin"]
    assert not pooled.duplicated(cell).any()
    # trajectories counted as units, with the head-count they came from alongside
    assert {"n_units", "n_source_persons"} <= set(pooled.columns)
    assert (pooled["n_source_persons"] < pooled["n_units"].max()).all()

    # the interval is the cell's own Poisson variance, exactly as the backtest families do
    rows = pooled.dropna(subset=["rate_var"])
    assert len(rows)
    np.testing.assert_allclose(rows["pooled_var"], rows["rate_var"])
    assert {"k_seeds", "mean_var", "between_var"} <= set(pooled.columns)

    # every forecast cell in the combined surface is its pooled cell, taken verbatim. A fixture
    # whose observed surface already covers every cell has none, and then there is nothing to check
    fc = combined[combined["source"] == "forecast"].merge(
        pooled[[*cell, "rate"]], on=cell, how="left", suffixes=("", "_pooled")
    )
    if len(fc):
        np.testing.assert_allclose(fc["rate"], fc["rate_pooled"])


def test_violation_rates_report_observed_baseline(tmp_path):
    out = _run(tmp_path)
    vr = pd.read_parquet(out.dir / "violation_rates.parquet")
    # both the model's and the observed data's rates are reported (data-artifact contextualization)
    assert set(vr["source"]) == {"generated", "observed"}


def test_occurrence_probability_lives_in_its_own_outcome_labelled_table(tmp_path):
    out = _run(tmp_path)
    ind = pd.read_parquet(out.dir / "replicate_variance_individual.parquet")
    occ = pd.read_parquet(out.dir / "replicate_occurrence.parquet")
    # the dispersion table is about the birth count only — nothing outcome-specific rides along
    assert not {"p_hat", "p_within_horizon", "timing_spread"} & set(ind.columns)
    # the occurrence table names the outcome it is about, and the horizon the event must fall inside
    assert set(occ["outcome"]) == {"first_birth"}
    assert (occ["horizon"] == yd(50)).all()
    assert (occ["timing_spread"] >= 0).all()
    # p_hat is the raw replicate frequency, unsmoothed
    np.testing.assert_allclose(occ["p_hat"], occ["n_occurred"] / occ["n"])


def test_replicate_variance_aggregate_ccf_band_covers_truth(tmp_path):
    out = _run(tmp_path)
    agg = pd.read_parquet(out.dir / "replicate_variance_aggregate.parquet")
    truth = S.expected_ccf(S.default_hazards())
    row = agg[agg["cohort"].isna()].iloc[0]  # pooled (all-cohorts) row for the first window
    # the replicate-only band, rebuilt from within_var, brackets the truth
    half = 1.959963984540054 * np.sqrt(row["within_var"])
    assert row["ccf"] - half <= truth <= row["ccf"] + half


def test_replicate_variance_aggregate_splits_the_variance_of_the_estimate(tmp_path):
    """Seed noise and heterogeneity, in variance units of the CCF, adding to the total."""
    agg = pd.read_parquet(_run(tmp_path).dir / "replicate_variance_aggregate.parquet")
    pooled = agg[agg["cohort"].isna()]
    # the split is exact and neither component is negative
    np.testing.assert_allclose(pooled["within_var"] + pooled["between_var"], pooled["total_var"])
    assert (pooled["within_var"] > 0).all() and (pooled["between_var"] >= 0).all()
    # total_var is what the reported interval is built from
    np.testing.assert_allclose(pooled["se_total"] ** 2, pooled["total_var"])
    # the replicate-only uncertainty stays recoverable as sqrt(within_var), and is the smaller half
    assert (np.sqrt(pooled["within_var"]) < pooled["se_total"]).all()


def test_replicate_variance_aggregate_flags_forecast_provenance(tmp_path):
    out = _run(tmp_path)
    agg = pd.read_parquet(out.dir / "replicate_variance_aggregate.parquet")
    # provenance, not completeness: how much of each CCF is model output rather than history
    assert ((agg["forecast_share"] >= 0) & (agg["forecast_share"] <= 1)).all()
    # both windows jump off well inside the fertile ages, so every CCF leans on the forecast
    assert (agg["forecast_share"] > 0).all()
    # the later jump-off (age 30) has more observed history, so less of its CCF is forecast
    pooled = agg[agg["cohort"].isna()].set_index("age_stop")["forecast_share"]
    assert pooled.loc[pooled.index.max()] < pooled.loc[pooled.index.min()]
