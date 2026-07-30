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
        "replicate_variance_individual_first_birth.parquet",
        "replicate_variance_aggregate_first_birth.parquet",
    } <= names
    # cohort (birth-cohort x age) is the only Lexis basis: a period cell is part observed and part
    # forecast, since the jump-off is an age and lands in a different calendar year for each cohort
    assert "lexis_cohort_combined.png" in names
    assert not [n for n in names if n.startswith("lexis_") and "cohort" not in n]
    # within-seed variance histograms: population-wide + faceted by the requested subgroup, each
    # carrying the block's name so a second block lands beside it rather than on top of it
    assert {
        "within_seed_variance_first_birth.png",
        "within_seed_variance_first_birth_by_cohort.png",
    } <= names
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
    ind = pd.read_parquet(out.dir / "replicate_variance_individual_first_birth.parquet")
    occ = pd.read_parquet(out.dir / "replicate_occurrence_first_birth.parquet")
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
    agg = pd.read_parquet(out.dir / "replicate_variance_aggregate_first_birth.parquet")
    truth = S.expected_ccf(S.default_hazards())
    row = agg[agg["cohort"].isna()].iloc[0]  # pooled (all-cohorts) row for the first window
    # the replicate-only band, rebuilt from within_var, brackets the truth
    half = 1.959963984540054 * np.sqrt(row["within_var"])
    assert row["ccf"] - half <= truth <= row["ccf"] + half


def test_replicate_variance_aggregate_splits_the_variance_of_the_estimate(tmp_path):
    """Seed noise and heterogeneity, in variance units of the CCF, adding to the total."""
    agg = pd.read_parquet(_run(tmp_path).dir / "replicate_variance_aggregate_first_birth.parquet")
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
    agg = pd.read_parquet(out.dir / "replicate_variance_aggregate_first_birth.parquet")
    # provenance, not completeness: how much of each CCF is model output rather than history
    assert ((agg["forecast_share"] >= 0) & (agg["forecast_share"] <= 1)).all()
    # both windows jump off well inside the fertile ages, so every CCF leans on the forecast
    assert (agg["forecast_share"] > 0).all()
    # the later jump-off (age 30) has more observed history, so less of its CCF is forecast
    pooled = agg[agg["cohort"].isna()].set_index("age_stop")["forecast_share"]
    assert pooled.loc[pooled.index.max()] < pooled.loc[pooled.index.min()]


# =================================================================================================
# the within-seed spread is about a configurable event, not births specifically
# =================================================================================================
def _two_event_generated() -> pd.DataFrame:
    """Two people x two seeds, carrying both a `birth` and a `marriage` stream.

    Person 1's births are identical across seeds and their marriages are not; person 2 is the other
    way round. So whichever event the dispersion counts, exactly one person has non-zero variance —
    which is what makes the choice observable.
    """
    rows = []
    for seed, (p1_births, p1_marr, p2_births, p2_marr) in enumerate([(2, 1, 1, 3), (2, 4, 5, 3)]):
        for pid, (nb, nm) in enumerate([(p1_births, p1_marr), (p2_births, p2_marr)], start=1):
            for i in range(nb):
                rows.append((pid, seed, "birth", yd(20 + i)))
            for i in range(nm):
                rows.append((pid, seed, "marriage", yd(20 + i)))
    df = pd.DataFrame(rows, columns=["person_id", "seed", "event", "age"])
    df["age_start"], df["age_stop"] = 0, yd(19)
    return df


def test_within_seed_spread_counts_the_event_it_is_given():
    gen = _two_event_generated()

    births = FC._replicate_variance_individual(gen, "birth").set_index("person_id")
    marriages = FC._replicate_variance_individual(gen, "marriage").set_index("person_id")

    # person 1: births agree across seeds (2, 2), marriages do not (1, 4)
    assert births.loc[1, "within_seed_var"] == 0
    assert marriages.loc[1, "within_seed_var"] > 0
    # person 2: the mirror image — births disagree (1, 5), marriages agree (3, 3)
    assert births.loc[2, "within_seed_var"] > 0
    assert marriages.loc[2, "within_seed_var"] == 0

    # the mean is the count of that event, under the neutral name
    assert births.loc[1, "expected_count"] == 2
    assert marriages.loc[2, "expected_count"] == 3


def test_replicate_variance_event_defaults_to_the_lexis_target(tmp_path):
    """Unset, the spread is about the same event the lexis outcome is about — births here."""
    cfg = Config.model_validate(yaml.safe_load(_CFG))
    (block,) = cfg.arms.forecasting.replicate_variance
    assert block.event is None
    assert block.name == "first_birth"  # inherited from the lexis outcome


def test_replicate_variance_event_must_be_a_declared_alias():
    import pytest
    from pydantic import ValidationError

    raw = yaml.safe_load(_CFG)
    raw["arms"]["forecasting"]["replicate_variance"]["event"] = "marriage"
    with pytest.raises(ValidationError, match="marriage"):
        Config.model_validate(raw)

    raw["events"]["marriage"] = "marriage"
    (block,) = Config.model_validate(raw).arms.forecasting.replicate_variance
    assert block.event == "marriage"
    assert block.name == "marriage"  # the event wins over the lexis outcome


def test_plural_only_touches_the_caption():
    assert FC._plural("live birth") == "live births"
    assert FC._plural("marriage") == "marriages"
    assert FC._plural("divorces") == "divorces"  # already plural, left alone


# =================================================================================================
# several replicate_variance blocks in one run
# =================================================================================================
_TWO_BLOCK_CFG = _CFG.replace(
    "    replicate_variance: {individual: true, aggregate: [ccf], subgroup_by: [cohort]}",
    """    replicate_variance:
      - {individual: true, aggregate: [ccf], subgroup_by: [cohort]}
      - {individual: true, event: birth, name: quantum, aggregate: [], subgroup_by: []}""",
)


def test_a_single_block_may_still_be_written_as_a_mapping():
    """The old shape keeps working — it is wrapped into a one-element list."""
    cfg = Config.model_validate(yaml.safe_load(_CFG))
    assert len(cfg.arms.forecasting.replicate_variance) == 1


def test_two_blocks_write_side_by_side(tmp_path):
    cfg = Config.model_validate(yaml.safe_load(_TWO_BLOCK_CFG))
    rng = np.random.default_rng(0)
    h = S.default_hazards()
    obs, pers = S.simulate_cohort(600, (1960, 1985), h, None, rng, no_event_fraction=1.0)
    gen = S.simulate_generated(obs, pers, h, [(0.0, 25.0), (0.0, 30.0)], 8, rng)
    bundle = Bundle(
        observed=obs, generated=gen, persons=pers, event_defs=None,
        events=EventConfig(birth="birth"),
    )
    out = OutputWriter(base_dir=tmp_path, arm="forecasting", model="perfect", individual_level=True)
    FC.run(
        bundle, cfg.arms.forecasting, out,
        outcomes=resolve_outcomes(cfg),
        rules=resolve_rules(cfg),
        replicate_spec=resolve_replicates(cfg),
    )

    names = {p.name for p in out.written}
    # neither block overwrote the other: one stem per block, named by it
    assert {
        "replicate_variance_individual_first_birth.parquet",
        "replicate_variance_individual_quantum.parquet",
        "within_seed_variance_distribution_first_birth.parquet",
        "within_seed_variance_distribution_quantum.parquet",
        "within_seed_quantile_summary_first_birth.parquet",
        "within_seed_quantile_summary_quantum.parquet",
    } <= names
    # the CCF roll-up only ran for the block that asked for it
    assert "replicate_variance_aggregate_first_birth.parquet" in names
    assert "replicate_variance_aggregate_quantum.parquet" not in names
    # subgroup_by is per block, so only the first faceted
    assert "within_seed_variance_distribution_first_birth_by_cohort.parquet" in names
    assert "within_seed_variance_distribution_quantum_by_cohort.parquet" not in names


def test_every_block_output_still_finds_its_disclosure_policy(tmp_path):
    """A name appended to the stem must not shake off the suppression policy."""
    from seqeval.metrics._disclosure import policy_for

    for stem in (
        "within_seed_variance_distribution_quantum",
        "within_seed_variance_distribution_quantum_by_cohort",
        "within_seed_quantile_summary_quantum",
        "replicate_variance_aggregate_quantum",
    ):
        assert policy_for(stem) is not None, stem


def test_blocks_that_would_collide_are_rejected():
    import pytest
    from pydantic import ValidationError

    raw = yaml.safe_load(_CFG)
    raw["arms"]["forecasting"]["replicate_variance"] = [
        {"individual": True},
        {"individual": True},  # both fall back to the lexis outcome name
    ]
    with pytest.raises(ValidationError, match="overwrite each other"):
        Config.model_validate(raw)


def test_a_block_with_nothing_to_name_it_by_is_rejected():
    import pytest
    from pydantic import ValidationError

    raw = yaml.safe_load(_CFG)
    del raw["arms"]["forecasting"]["lexis"]
    raw["arms"]["forecasting"]["replicate_variance"] = [{"individual": True}]
    with pytest.raises(ValidationError, match="cannot name this block"):
        Config.model_validate(raw)
