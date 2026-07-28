"""Backtesting arm (04): perfect-model calibration, miscalibration, semantics, conditioning."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import yaml

from seqeval.arms import backtesting as BT
from seqeval.arms._common import OutputWriter
from seqeval.config import (
    Config,
    EventConfig,
    resolve_conditions,
    resolve_outcomes,
    resolve_probability_outcomes,
    resolve_replicates,
)
from seqeval.core import outcomes as O
from seqeval.core import replicates as rep
from seqeval.core.slicing import condition_on_count
from seqeval.core.specs import (
    Condition,
    CountQuery,
    FertilityGrid,
    Frame,
    FramedOutcome,
    ReplicateSpec,
    TTESpec,
)
from seqeval.io.loaders import Bundle
from seqeval.metrics import ml
from seqeval.units import years_to_days as yd
from tests import synthetic as S

GK = ["person_id", "seed", "age_start", "age_stop"]
RK = ["person_id", "age_start", "age_stop"]
WIN, JO = (0.0, 28.0), yd(28)
SPEC = CountQuery("b1w12", "birth", 1, Frame("within", yd(12)))


def _pipeline(gen_hazards, n_seeds, *, seed=0, n=2500):
    rng = np.random.default_rng(seed)
    truth = S.default_hazards()
    obs, pers = S.simulate_cohort(n, (1960, 1990), truth, None, rng, no_event_fraction=1.0)
    gen = S.simulate_generated(obs, pers, gen_hazards, [WIN], n_seeds, rng)
    ev = O.evaluate_count(gen, GK, SPEC, O.observation_spans(gen, GK), jumpoff=JO)
    summ = rep.replicate_summary(ev, run_keys=RK)
    est = rep.estimate_probability(summ, spec=ReplicateSpec())
    oev = O.evaluate_count(
        obs, ["person_id"], SPEC, O.observation_spans(obs, ["person_id"]), jumpoff=JO
    )
    return summ, ml.join_truth(est, oev)


def _band_coverage(summ, joined, rng):
    band = rep.null_calibration_band(
        summ, n_bins=10, strategy="uniform", n_sims=300, rng=rng
    )
    cal = ml.calibration_table(joined, n_bins=10, strategy="uniform").merge(band, on="bin")
    valid = cal[cal["n"] >= 10]
    inside = valid["y_rate"].between(valid["lo"] - 1e-9, valid["hi"] + 1e-9)
    return float(inside.mean())


def test_perfect_model_calibrated_at_n50():
    summ, joined = _pipeline(S.default_hazards(), 50)
    cov = _band_coverage(summ, joined, np.random.default_rng(1))
    assert cov >= 0.7  # curve sits inside the null band for most bins
    b = ml.brier(joined)
    assert abs(b["corrected"] - b["raw"]) < 0.02  # MC correction vanishes at high n


def test_few_seeds_corrected_below_raw():
    summ, joined = _pipeline(S.default_hazards(), 5)
    b = ml.brier(joined)
    assert b["corrected"] < b["raw"]  # few seeds inflate raw Brier; correction pulls it down


def test_miscalibrated_model_over_predicts_and_exits_band():
    # generated from inflated hazards -> predicts more births than truth -> over-prediction
    summ, joined = _pipeline(S.perturb(S.default_hazards(), 1.6), 50)
    assert joined["p_hat"].mean() > joined["y_true"].mean() + 0.03  # systematic over-prediction
    perfect_summ, perfect_joined = _pipeline(S.default_hazards(), 50)
    assert _band_coverage(summ, joined, np.random.default_rng(2)) < _band_coverage(
        perfect_summ, perfect_joined, np.random.default_rng(2)
    )


def test_framed_and_count_agree_under_parity_cap():
    # {second_birth within 5 | p1} and {birth >=1 within 5 | p1} are the same question at parity 1.
    rng = np.random.default_rng(3)
    obs, _ = S.simulate_cohort(
        1500, (1960, 1985), S.default_hazards(), None, rng, no_event_fraction=1.0
    )
    spans = O.observation_spans(obs, ["person_id"])
    p1 = set(
        condition_on_count(obs, ["person_id"], cond=Condition("p1", "birth", 1, 1), anchor_age=JO)[
            "person_id"
        ]
    )
    framed = FramedOutcome(
        "sb5", TTESpec("birth", 2, origin=TTESpec("birth", 1)), Frame("within", yd(5))
    )
    count = CountQuery("b1w5", "birth", 1, Frame("within", yd(5)))
    fe = O.evaluate_framed(obs, ["person_id"], framed, spans, jumpoff=JO).set_index("person_id")
    ce = O.evaluate_count(obs, ["person_id"], count, spans, jumpoff=JO).set_index("person_id")

    shared = [p for p in p1 if bool(fe.loc[p, "evaluable"]) and bool(ce.loc[p, "evaluable"])]
    assert len(shared) > 50
    assert (fe.loc[shared, "occurred"].to_numpy() == ce.loc[shared, "occurred"].to_numpy()).all()


# --- arm end to end -----------------------------------------------------------------------------
_CFG_YAML = """
model: {name: perfect}
data: {observed: o.parquet, age_unit: days}
events: {birth: birth}
persons: {cohort_width: 5}
replicates: {min_replicates: 5}
outcomes:
  first_birth: {event: birth, n: 1}
arms:
  backtesting:
    probability_outcomes:
      - {outcome: first_birth, by_age: 40, given: p0}
      - {outcome: first_birth, by_age: 40}
      - {event: birth, min_events: 1, within: 10}
    conditions:
      - {name: p0, event: birth, max_count: 0}
    aggregate_targets: [ccf, ppr, asfr_cohort]
    min_seeds: 5
"""


_GRID = FertilityGrid()


def _run_arm(
    tmp_path, windows=((0.0, 25.0), (0.0, 30.0)), n_seeds=10, fertility_grid=_GRID
):
    cfg = Config.model_validate(yaml.safe_load(_CFG_YAML))
    rng = np.random.default_rng(0)
    h = S.default_hazards()
    obs, pers = S.simulate_cohort(1200, (1960, 1985), h, None, rng, no_event_fraction=1.0)
    gen = S.simulate_generated(obs, pers, h, list(windows), n_seeds, rng)
    bundle = Bundle(
        observed=obs,
        generated=gen,
        persons=pers,
        event_defs=None,
        events=EventConfig(birth="birth"),
    )
    out = OutputWriter(base_dir=tmp_path, arm="backtesting", model="perfect")
    BT.run(
        bundle,
        cfg.arms.backtesting,
        out,
        outcomes=resolve_outcomes(cfg),
        conditions=resolve_conditions(cfg),
        prob_outcomes=resolve_probability_outcomes(cfg, resolve_outcomes(cfg)),
        replicate_spec=resolve_replicates(cfg),
        cohort_width=5,
        fertility_grid=fertility_grid,
    )
    return out


def test_arm_writes_scores_and_tables(tmp_path):
    out = _run_arm(tmp_path)
    names = {p.name for p in out.written}
    assert {
        "scores.parquet",
        "probabilities.parquet",
        "calibration.parquet",
        "coverage.parquet",
        "aggregate_error.parquet",
    } <= names

    scores = pd.read_parquet(out.dir / "scores.parquet")
    assert (scores["model"] == "perfect").all()
    # one row per (window, outcome, condition, metric)
    assert not scores.duplicated(["age_stop", "outcome", "condition", "metric"]).any()
    assert {"roc_auc", "brier_corrected", "mse", "r2", "ece"} <= set(scores["metric"])
    assert "log_loss" not in set(scores["metric"])  # removed from the backtest score set


def test_scores_carry_analytic_cis(tmp_path):
    """CIs are analytic, so there is nothing to switch on — and every one brackets its estimate."""
    cfg = Config.model_validate(yaml.safe_load(_CFG_YAML))
    rng = np.random.default_rng(0)
    h = S.default_hazards()
    obs, pers = S.simulate_cohort(1200, (1960, 1985), h, None, rng, no_event_fraction=1.0)
    gen = S.simulate_generated(obs, pers, h, [(0.0, 25.0), (0.0, 30.0)], 10, rng)
    bundle = Bundle(
        observed=obs, generated=gen, persons=pers, event_defs=None,
        events=EventConfig(birth="birth"),
    )
    out = OutputWriter(base_dir=tmp_path, arm="backtesting", model="perfect")
    BT.run(
        bundle,
        cfg.arms.backtesting,
        out,
        outcomes=resolve_outcomes(cfg),
        conditions=resolve_conditions(cfg),
        prob_outcomes=resolve_probability_outcomes(cfg, resolve_outcomes(cfg)),
        replicate_spec=resolve_replicates(cfg),
        cohort_width=5,
    )
    scores = pd.read_parquet(out.dir / "scores.parquet")
    assert {"ci_lo", "ci_hi"} <= set(scores.columns)
    finite = scores.dropna(subset=["ci_lo", "ci_hi"])
    assert {"roc_auc", "brier_corrected", "mse", "r2"} <= set(finite["metric"])
    assert (finite["ci_lo"] <= finite["value"]).all()
    assert (finite["value"] <= finite["ci_hi"]).all()
    # ECE is reported without one: its bins are data-chosen and the statistic is biased upward
    assert scores[scores["metric"] == "ece"]["ci_lo"].isna().all()

    # reliability is emitted as one figure per (outcome, window) — names carry a `_w<age>` suffix.
    figs = {p.name for p in out.written if p.suffix == ".png"}
    reliab = [f for f in figs if f.startswith("reliability_")]
    assert reliab
    assert all(re.search(r"_w\d", f) for f in reliab)


def test_no_scalar_metric_figures_are_emitted(tmp_path):
    """AUC/Brier live in the report's metrics table; the arm draws no line charts for them."""
    out = _run_arm(tmp_path)
    figs = {p.name for p in out.written if p.suffix == ".png"}
    assert not [f for f in figs if f.startswith("metric_vs_jumpoff")]
    assert (out.dir / "scores.parquet").exists()  # the numbers themselves are still written


def test_timing_ridge_figures_emitted_per_framed_jumpoff(tmp_path):
    """One timing-error ridge per (framed outcome, jump-off), backed by a binned counts table."""
    out = _run_arm(tmp_path)
    figs = {p.name for p in out.written if p.suffix == ".png"}
    assert {
        "timing_ridge_first_birth_by_age_40y_given_p0_w25.png",
        "timing_ridge_first_birth_by_age_40y_given_p0_w30.png",
    } <= figs
    assert not [f for f in figs if f.startswith("timing_calibration")]

    err = pd.read_parquet(out.dir / "timing_error.parquet")
    assert "person_id" not in err.columns  # the table the figure is drawn from names nobody
    assert set(err["outcome"]) and (err["n_pred_bin"] > 0).all()


def test_coverage_reports_settled_for_framed(tmp_path):
    out = _run_arm(tmp_path)
    cov = pd.read_parquet(out.dir / "coverage.parquet")
    # the *unconditioned* first_birth by_age 40: persons whose first birth precedes the jump-off
    # are settled (answer already in the observed prefix) and excluded.
    unconditioned = cov[(cov["outcome"] == "first_birth_by_age_40y") & (cov["condition"] == "-")]
    assert unconditioned["n_settled"].sum() > 0


_KM_ONLY_YAML = """
model: {name: perfect}
data: {observed: o.parquet, age_unit: days}
events: {union: birth}
replicates: {min_replicates: 5}
outcomes:
  first_union: {event: union, n: 1}
arms:
  backtesting:
    probability_outcomes:
      - {outcome: first_union, by_age: 40}
    aggregate_targets: [km:first_union]
    min_seeds: 5
"""


def test_km_only_targets_need_neither_births_nor_persons(tmp_path, caplog):
    """A survival target is built from the outcome registry, not from fertility inputs.

    This bundle declares no ``birth`` alias and carries no persons table, so a run that resolved
    either would fail outright rather than degrade.
    """
    import logging

    cfg = Config.model_validate(yaml.safe_load(_KM_ONLY_YAML))
    rng = np.random.default_rng(0)
    h = S.default_hazards()
    obs, pers = S.simulate_cohort(800, (1960, 1985), h, None, rng, no_event_fraction=1.0)
    gen = S.simulate_generated(obs, pers, h, [(0.0, 25.0)], 8, rng)
    bundle = Bundle(
        observed=obs, generated=gen, persons=None, event_defs=None,
        events=EventConfig(union="birth"),
    )
    out = OutputWriter(base_dir=tmp_path, arm="backtesting", model="perfect")
    with caplog.at_level(logging.WARNING, logger="seqeval"):
        BT.run(
            bundle,
            cfg.arms.backtesting,
            out,
            outcomes=resolve_outcomes(cfg),
            conditions=resolve_conditions(cfg),
            prob_outcomes=resolve_probability_outcomes(cfg, resolve_outcomes(cfg)),
            replicate_spec=resolve_replicates(cfg),
            cohort_width=5,
        )
    err = pd.read_parquet(out.dir / "aggregate_error.parquet")
    assert set(err["target"]) == {"km:first_union"}
    assert not any("needs persons" in r.message for r in caplog.records)


def test_needs_births_ignores_km_targets():
    def cfg_for(targets):
        y = _CFG_YAML.replace(
            "aggregate_targets: [ccf, ppr, asfr_cohort]", f"aggregate_targets: {targets}"
        )
        return Config.model_validate(yaml.safe_load(y)).arms.backtesting

    assert not BT._needs_births(cfg_for("[km:first_birth]"))
    assert not BT._needs_births(cfg_for("[]"))
    assert BT._needs_births(cfg_for("[ppr]"))
    assert BT._needs_births(cfg_for("[km:first_birth, ccf]"))  # mixed: births still required


def test_ccf_uncertainty_figure_and_parity_table_emitted(tmp_path):
    """The CCF gets an inference-vs-outcome view, backed by a publishable parity table."""
    out = _run_arm(tmp_path)
    figs = {p.name for p in out.written if p.suffix == ".png"}
    assert any(f.startswith("uncertainty_ccf_w") for f in figs)

    par = pd.read_parquet(out.dir / "parity_distribution.parquet")
    assert "person_id" not in par.columns
    # every cohort's published shares are a distribution: bounded, and complete unless withheld
    shown = par[~par["suppressed"]]
    assert ((shown["share"] >= 0) & (shown["share"] <= 1)).all()
    assert (par.groupby(["age_stop", "cohort"], observed=True)["share"].sum() <= 1.0 + 1e-9).all()


def test_ppr_and_asfr_overlays_emitted_per_jumpoff_and_jointly(tmp_path):
    """Every scored aggregate fertility target is also drawn, per window and across windows."""
    out = _run_arm(tmp_path)
    figs = {p.name for p in out.written if p.suffix == ".png"}
    assert {"ppr_overlay_w25.png", "ppr_overlay_w30.png"} <= figs
    assert {"asfr_overlay_w25.png", "asfr_overlay_w30.png"} <= figs
    # two windows, so each family also gets the panel comparing them on one axes
    assert {"ppr_overlay_all_jumpoffs.png", "asfr_overlay_all_jumpoffs.png"} <= figs


def test_aggregate_error_covers_every_configured_target(tmp_path):
    out = _run_arm(tmp_path)
    err = pd.read_parquet(out.dir / "aggregate_error.parquet")
    assert set(err["target"]) == {"ccf", "ppr", "asfr_cohort"}
    assert "person_id" not in err.columns


def test_fertility_grid_sets_the_backtest_parity_ceiling(tmp_path):
    """The PPR grid is the resolved one, not a constant private to this arm."""
    out = _run_arm(tmp_path, fertility_grid=FertilityGrid(max_parity=3))
    err = pd.read_parquet(out.dir / "aggregate_error.parquet")
    ppr = err[err["target"] == "ppr"]
    assert sorted(ppr["parity_from"].unique()) == [0, 1, 2]
