"""Forecasting arm (05): smoke test + seed-stability definitional checks."""

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
from tests import synthetic as S

_CFG = """
model: {name: perfect}
data: {observed: o.parquet, age_unit: days}
events: {birth: birth}
persons: {cohort_width: 5}
replicates: {min_replicates: 5, bootstrap: {n: 0, seed: 7}}
outcomes: {first_birth: {event: birth, n: 1}}
arms:
  forecasting:
    lexis: {outcome: first_birth, ages: [12, 55], years: [1975, 2035], subgroup_by: []}
    illegal_moves:
      - {event: birth, max_age: 50}
      - {event: birth, min_spacing: 0.6, severity: warn}
    seed_stability: {individual: true, aggregate: [ccf]}
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
    out = OutputWriter(base_dir=tmp_path, arm="forecasting", model="perfect")
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
        "lexis_observed.parquet",
        "lexis_forecast.parquet",
        "lexis_combined.parquet",
        "lexis_cohort_observed.parquet",
        "lexis_cohort_combined.parquet",
        "violations.parquet",
        "violation_rates.parquet",
        "seed_stability_individual.parquet",
        "seed_stability_aggregate.parquet",
    } <= names
    # both period (year x age) and cohort (birth-cohort x age) Lexis heatmaps render
    assert {"lexis_combined.png", "lexis_cohort_combined.png"} <= names
    for p in out.written:
        assert p.exists() and p.stat().st_size > 0


def test_violation_rates_report_observed_baseline(tmp_path):
    out = _run(tmp_path)
    vr = pd.read_parquet(out.dir / "violation_rates.parquet")
    # both the model's and the observed data's rates are reported (data-artifact contextualization)
    assert set(vr["source"]) == {"generated", "observed"}


def test_seed_stability_disagreement_is_bernoulli_variance(tmp_path):
    out = _run(tmp_path)
    ss = pd.read_parquet(out.dir / "seed_stability_individual.parquet")
    # occurrence disagreement IS p_hat(1-p_hat) on the smoothed estimate, and lies in [0, 0.25]
    np.testing.assert_allclose(ss["disagreement"], ss["p_hat"] * (1 - ss["p_hat"]), atol=1e-12)
    assert (ss["disagreement"] >= 0).all() and (ss["disagreement"] <= 0.25 + 1e-9).all()


def test_seed_stability_aggregate_ccf_band_covers_truth(tmp_path):
    out = _run(tmp_path)
    agg = pd.read_parquet(out.dir / "seed_stability_aggregate.parquet")
    truth = S.expected_ccf(S.default_hazards())
    row = agg.iloc[0]
    assert row["ci_lo"] <= truth <= row["ci_hi"]  # forecast CCF band brackets the known truth
