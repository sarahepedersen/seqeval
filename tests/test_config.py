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


def test_unknown_not_before_alias_rejected():
    d = _ref_dict()
    d["arms"]["forecasting"]["illegal_moves"].append({"event": "birth", "not_before": "wedding"})
    with pytest.raises(ValidationError, match="unknown event alias"):
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
    assert spec.bootstrap_n == 200
    assert spec.bootstrap_seed == 7


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


def test_output_publication_policy_defaults_to_current_behaviour():
    cfg = C.Config.model_validate(_ref_dict())
    assert cfg.output.individual_level is True
    assert cfg.output.min_cell == 5


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
    d["output"] = {"individual_level": False}
    assert C.Config.model_validate(d).hash() != base
