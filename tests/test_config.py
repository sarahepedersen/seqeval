"""Config: reference parse, cross-reference violations, path resolution, hash, resolvers."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from seqeval import config as C
from seqeval.units import years_to_days

_REF = Path(__file__).parent / "data" / "reference_config.yaml"


def _ref_dict():
    return yaml.safe_load(_REF.read_text())


def test_reference_config_parses():
    cfg = C.load_config(_REF)
    assert cfg.model.name == "transformer_model_example"
    assert cfg.arms.backtesting is not None
    assert cfg.arms.forecasting is not None


def test_relative_paths_resolved_against_yaml_dir():
    cfg = C.load_config(_REF)
    assert cfg.observed_path == _REF.parent / "data" / "observed.parquet"
    assert cfg.generated_path == _REF.parent / "data" / "generated.parquet"


def test_cohort_width_is_shared_via_persons():
    # cohort_width lives on the persons block; Config.cohort_width exposes it to every arm.
    d = _ref_dict()
    d["persons"]["cohort_width"] = 10
    assert C.Config.model_validate(d).cohort_width == 10
    # falls back to the default when no persons block is present
    minimal = {"model": {"name": "m"}, "data": {"observed": "o.parquet"}, "events": {"birth": "01"}}
    assert C.Config.model_validate(minimal).cohort_width == C.DEFAULT_COHORT_WIDTH


def test_hash_stable_under_key_reordering():
    cfg1 = C.Config.model_validate(_ref_dict())
    reordered = dict(reversed(list(_ref_dict().items())))
    cfg2 = C.Config.model_validate(reordered)
    assert cfg1.hash() == cfg2.hash()


# --- cross-reference violations -----------------------------------------------------------------
def test_unknown_top_level_key():
    d = _ref_dict()
    d["nonsense"] = 1
    with pytest.raises(ValidationError, match="nonsense"):
        C.Config.model_validate(d)


def test_frame_key_on_registry_outcome_forbidden():
    d = _ref_dict()
    d["outcomes"]["first_birth"]["by_age"] = 35
    with pytest.raises(ValidationError, match="by_age"):
        C.Config.model_validate(d)


def test_unknown_event_alias():
    d = _ref_dict()
    d["outcomes"]["first_birth"]["event"] = "not_an_alias"
    with pytest.raises(ValidationError, match="unknown event alias"):
        C.Config.model_validate(d)


def test_unknown_outcome_reference():
    d = _ref_dict()
    d["arms"]["backtesting"]["probability_outcomes"][0]["outcome"] = "first_brith"
    with pytest.raises(ValidationError, match="unknown outcome 'first_brith'"):
        C.Config.model_validate(d)


def test_unknown_condition_reference():
    d = _ref_dict()
    d["arms"]["backtesting"]["probability_outcomes"][0]["given"] = "pX"
    with pytest.raises(ValidationError, match="unknown condition 'pX'"):
        C.Config.model_validate(d)


def test_duplicate_condition_name():
    d = _ref_dict()
    conds = d["arms"]["backtesting"]["conditions"]
    conds.append({"name": "p0", "event": "birth", "max_count": 2})
    with pytest.raises(ValidationError, match="duplicate condition name"):
        C.Config.model_validate(d)


def test_both_form_keys_on_probability_outcome():
    d = _ref_dict()
    d["arms"]["backtesting"]["probability_outcomes"].append(
        {"outcome": "first_birth", "event": "birth", "min_events": 1, "within": 5}
    )
    with pytest.raises(ValidationError, match="cannot mix"):
        C.Config.model_validate(d)


def test_neither_form_keys_on_probability_outcome():
    d = _ref_dict()
    d["arms"]["backtesting"]["probability_outcomes"].append({"by_age": 30})
    with pytest.raises(ValidationError, match="must be a framed reference"):
        C.Config.model_validate(d)


def test_calibration_binning_defaults_to_quantile():
    cfg = C.load_config(_REF)
    assert cfg.arms.backtesting.calibration_binning == "quantile"


def test_calibration_binning_rejects_unknown_value():
    d = _ref_dict()
    d["arms"]["backtesting"]["calibration_binning"] = "deciles"
    with pytest.raises(ValidationError):
        C.Config.model_validate(d)


def test_calibration_bins_is_configurable_and_bounded():
    """More replicates support more bins, so the count has to be a knob, not a constant."""
    assert C.load_config(_REF).arms.backtesting.calibration_bins == 10
    d = _ref_dict()
    d["arms"]["backtesting"]["calibration_bins"] = 20
    assert C.Config.model_validate(d).arms.backtesting.calibration_bins == 20
    d["arms"]["backtesting"]["calibration_bins"] = 1  # a single bin says nothing about calibration
    with pytest.raises(ValidationError):
        C.Config.model_validate(d)


def test_within_origin_on_outcome_without_origin():
    d = _ref_dict()
    d["arms"]["backtesting"]["probability_outcomes"].append(
        {"outcome": "first_birth", "within_origin": 5}
    )
    with pytest.raises(ValidationError, match="declares no 'origin'"):
        C.Config.model_validate(d)


def test_within_origin_on_count_query():
    d = _ref_dict()
    d["arms"]["backtesting"]["probability_outcomes"].append(
        {"event": "birth", "min_events": 1, "within_origin": 5}
    )
    with pytest.raises(ValidationError, match="illegal on a count"):
        C.Config.model_validate(d)


def test_resolve_rules_not_before_maps_to_raw_token():
    d = _ref_dict()
    d["events"]["death"] = "99"
    d["arms"]["forecasting"]["illegal_moves"].append(
        {"event": "birth", "not_before": "death", "name": "birth_before_death"}
    )
    cfg = C.Config.model_validate(d)
    rule = next(r for r in C.resolve_rules(cfg) if r.name == "birth_before_death")
    assert rule.not_before == "99"
    assert rule.not_after is None


def test_unknown_not_before_reference_rejected():
    """An anchor may be an alias or an outcome, so the error enumerates both."""
    d = _ref_dict()
    d["arms"]["forecasting"]["illegal_moves"].append({"event": "birth", "not_before": "wedding"})
    with pytest.raises(ValidationError, match="unknown reference"):
        C.Config.model_validate(d)


def test_two_frame_keys_rejected():
    d = _ref_dict()
    d["arms"]["backtesting"]["probability_outcomes"].append(
        {"outcome": "first_birth", "by_age": 35, "within": 5}
    )
    with pytest.raises(ValidationError, match="exactly one frame key"):
        C.Config.model_validate(d)


def test_undeclared_covariate_stratifier():
    d = _ref_dict()
    d["arms"]["descriptives"]["stratify_by"] = ["not_a_covariate"]
    with pytest.raises(ValidationError, match="not an allowed stratifier"):
        C.Config.model_validate(d)


def test_bare_sex_stratifier_rejected():
    # Decision: bare 'sex' is NOT allowed (00 section 5 rule 8) unless declared as a covariate.
    d = _ref_dict()
    d["arms"]["descriptives"]["stratify_by"] = ["sex"]
    with pytest.raises(ValidationError, match="not an allowed stratifier"):
        C.Config.model_validate(d)


def test_chained_origin_rejected():
    d = _ref_dict()
    d["outcomes"]["third_birth"] = {"event": "birth", "n": 3, "origin": "second_birth"}
    with pytest.raises(ValidationError, match="chained origins"):
        C.Config.model_validate(d)


def test_unknown_aggregate_target():
    d = _ref_dict()
    d["arms"]["backtesting"]["aggregate_targets"].append("not_a_metric")
    with pytest.raises(ValidationError, match="unknown aggregate target"):
        C.Config.model_validate(d)


def test_condition_needs_a_bound():
    d = _ref_dict()
    d["arms"]["backtesting"]["conditions"].append({"name": "pX", "event": "birth"})
    with pytest.raises(ValidationError, match="at least one of min_count"):
        C.Config.model_validate(d)


# --- resolver round-trips (years -> days) -------------------------------------------------------
def test_resolve_outcomes_tokens_and_origin():
    cfg = C.load_config(_REF)
    outs = C.resolve_outcomes(cfg)
    assert outs["first_birth"].target == "01"
    assert outs["first_birth"].occurrence == 1
    assert outs["second_birth"].occurrence == 2
    assert outs["second_birth"].origin == outs["first_birth"]


def test_resolve_conditions_before_age_to_days():
    d = _ref_dict()
    d["arms"]["backtesting"]["conditions"][0]["before_age"] = 30
    cfg = C.Config.model_validate(d)
    conds = C.resolve_conditions(cfg)
    assert conds["p0"].before_age == years_to_days(30)
    assert conds["p0"].event == "01"


def test_resolve_probability_outcomes_frames_in_days():
    cfg = C.load_config(_REF)
    outs = C.resolve_outcomes(cfg)
    pos = {p.name: p for p in C.resolve_probability_outcomes(cfg, outs)}
    framed = next(p for p in pos.values() if getattr(p, "frame", None) and p.frame.kind == "by_age")
    assert framed.frame.value == years_to_days(35)
    count = next(p for p in pos.values() if type(p).__name__ == "CountQuery")
    assert count.frame.value == years_to_days(5)
    assert count.event == "01"


def test_resolve_rules_days_and_severity():
    cfg = C.load_config(_REF)
    rules = C.resolve_rules(cfg)
    max_age_rule = next(r for r in rules if r.max_age is not None)
    assert max_age_rule.max_age == years_to_days(45)
    spacing_rule = next(r for r in rules if r.min_spacing is not None)
    assert spacing_rule.min_spacing == years_to_days(0.6)
    assert spacing_rule.severity == "warn"


def test_resolve_replicates_passthrough():
    cfg = C.load_config(_REF)
    spec = C.resolve_replicates(cfg)
    assert spec.interval == "jeffreys"


def test_resolve_rules_smoke():
    cfg = C.load_config(_REF)
    rules = C.resolve_rules(cfg)
    assert len(rules) == 4


def test_resolve_windows_subset_and_missing(caplog):
    import logging

    import pandas as pd

    available = pd.DataFrame(
        {"age_start": [0, 0], "age_stop": [years_to_days(25), years_to_days(30)]}
    )
    spec = [C.WindowConfig(age_start=0, age_stop=25), C.WindowConfig(age_start=0, age_stop=99)]
    with caplog.at_level(logging.WARNING, logger="seqeval"):
        got = C.resolve_windows(spec, available)
    assert got == [(0, years_to_days(25))]
    assert any("absent" in r.message for r in caplog.records)


def test_output_publication_policy_defaults_to_the_safe_side():
    """Naming a person is opt-in, and the suppression threshold applies unless asked otherwise."""
    cfg = C.Config.model_validate(_ref_dict())
    assert cfg.output.individual_level is False
    assert cfg.output.min_cell == 3


def test_output_publication_policy_is_configurable():
    d = _ref_dict()
    d["output"] = {"individual_level": False, "min_cell": 10}
    cfg = C.Config.model_validate(d)
    assert cfg.output.individual_level is False and cfg.output.min_cell == 10


def test_negative_min_cell_is_rejected():
    d = _ref_dict()
    d["output"] = {"min_cell": -1}
    with pytest.raises(ValidationError):
        C.Config.model_validate(d)


def test_publication_policy_changes_the_config_hash():
    """A restricted run is a different run, and its manifest has to say so."""
    base = C.Config.model_validate(_ref_dict()).hash()
    d = _ref_dict()
    d["output"] = {"individual_level": True}
    assert C.Config.model_validate(d).hash() != base


def test_period_asfr_is_not_a_backtest_target():
    """A (year, age) cell is part observed and part forecast, so it has no honest comparison.

    The jump-off is an age: cohort-indexed cells fall wholly on one side of it, calendar-year cells
    do not. Rejected at parse time rather than raising deep in the arm.
    """
    d = _ref_dict()
    d["arms"]["backtesting"]["aggregate_targets"] = ["asfr_period"]
    with pytest.raises(ValidationError, match="unknown aggregate target"):
        C.Config.model_validate(d)

    d["arms"]["backtesting"]["aggregate_targets"] = ["asfr_cohort"]
    assert C.Config.model_validate(d).arms.backtesting.aggregate_targets == ["asfr_cohort"]


def test_period_asfr_is_not_a_descriptive_either():
    """The same reason removes it from `fertility.asfr`: nothing in the report can draw it."""
    d = _ref_dict()
    d["arms"]["descriptives"]["fertility"]["asfr"] = ["period"]
    with pytest.raises(ValidationError):
        C.Config.model_validate(d)

    d["arms"]["descriptives"]["fertility"]["asfr"] = ["cohort"]
    assert C.Config.model_validate(d).arms.descriptives.fertility.asfr == ["cohort"]


def test_fertility_grid_follows_the_descriptives_settings():
    """Backtesting bins fertility the way descriptives does, so the two figures are comparable."""
    d = _ref_dict()
    d["arms"]["descriptives"]["fertility"] = {
        "ccf": True, "asfr": ["cohort"], "ppr": {"max_parity": 3}, "age_bin_width": 5.0,
    }
    grid = C.resolve_fertility_grid(C.Config.model_validate(d))
    assert (grid.max_parity, grid.age_bin_width) == (3, 5.0)


def test_fertility_grid_falls_back_when_descriptives_says_nothing():
    """No descriptives block (or no ppr block) is not an error; the defaults still bin."""
    d = _ref_dict()
    d["arms"].pop("descriptives", None)
    assert C.resolve_fertility_grid(C.Config.model_validate(d)) == C.FertilityGrid()

    d = _ref_dict()
    d["arms"]["descriptives"]["fertility"] = {"ccf": True, "asfr": ["cohort"]}
    grid = C.resolve_fertility_grid(C.Config.model_validate(d))
    assert grid.max_parity == C.FertilityGrid().max_parity


# =================================================================================================
# illegal moves keyed on outcomes
# =================================================================================================
def _rule_named(cfg, name):
    return next(r for r in C.resolve_rules(cfg) if r.name == name)


def test_an_outcome_keyed_rule_pins_the_occurrence():
    d = _ref_dict()
    d["arms"]["forecasting"]["illegal_moves"].append(
        {"outcome": "second_birth", "not_before": "first_birth", "name": "out_of_order"}
    )
    rule = _rule_named(C.Config.model_validate(d), "out_of_order")
    assert rule.occurrence == 2  # the subject is the 2nd birth, not every birth
    assert rule.not_before_occurrence == 1  # anchored on the 1st


def test_an_event_keyed_rule_still_means_every_occurrence():
    """The existing form is unchanged: no occurrence, and an alias anchor means the first one."""
    d = _ref_dict()
    d["arms"]["forecasting"]["illegal_moves"].append(
        {"event": "birth", "not_before": "birth", "name": "token_form"}
    )
    rule = _rule_named(C.Config.model_validate(d), "token_form")
    assert rule.occurrence is None
    assert rule.not_before_occurrence == 1


def test_an_outcome_anchor_carries_its_own_ordinal():
    d = _ref_dict()
    d["arms"]["forecasting"]["illegal_moves"].append(
        {"event": "birth", "not_before": "second_birth", "name": "anchored_on_second"}
    )
    rule = _rule_named(C.Config.model_validate(d), "anchored_on_second")
    assert rule.not_before_occurrence == 2


def test_a_rule_needs_exactly_one_subject():
    d = _ref_dict()
    d["arms"]["forecasting"]["illegal_moves"].append({"max_age": 45})
    with pytest.raises(ValidationError, match="exactly one of"):
        C.Config.model_validate(d)

    d = _ref_dict()
    d["arms"]["forecasting"]["illegal_moves"].append(
        {"event": "birth", "outcome": "first_birth", "max_age": 45}
    )
    with pytest.raises(ValidationError, match="exactly one of"):
        C.Config.model_validate(d)


def test_stream_wide_fields_need_an_event_not_an_outcome():
    """`min_spacing`/`max_count` count across occurrences, so pinning one makes them meaningless."""
    for field, value in (("min_spacing", 0.6), ("max_count", 3)):
        d = _ref_dict()
        d["arms"]["forecasting"]["illegal_moves"].append(
            {"outcome": "first_birth", field: value}
        )
        with pytest.raises(ValidationError, match="needs `event`"):
            C.Config.model_validate(d)


def test_unknown_outcome_subject_rejected():
    d = _ref_dict()
    d["arms"]["forecasting"]["illegal_moves"].append({"outcome": "first_divorce"})
    with pytest.raises(ValidationError, match="unknown outcome"):
        C.Config.model_validate(d)
