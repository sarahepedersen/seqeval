"""Backtesting arm (04): perfect-model calibration, miscalibration, semantics, conditioning."""

from __future__ import annotations

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
from seqeval.core.specs import Condition, CountQuery, Frame, FramedOutcome, ReplicateSpec, TTESpec
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
    est = rep.estimate_probability(summ, spec=ReplicateSpec(estimator="jeffreys"))
    oev = O.evaluate_count(
        obs, ["person_id"], SPEC, O.observation_spans(obs, ["person_id"]), jumpoff=JO
    )
    return summ, ml.join_truth(est, oev)


def _band_coverage(summ, joined, rng):
    band = rep.null_calibration_band(
        summ, n_bins=10, strategy="uniform", n_sims=300, rng=rng, estimator="jeffreys"
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
replicates: {min_replicates: 5, bootstrap: {n: 0, seed: 7}, convergence_curve: false}
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
    aggregate_targets: [ccf]
    min_seeds: 5
"""


def _run_arm(tmp_path, windows=((0.0, 25.0), (0.0, 30.0)), n_seeds=10):
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
    assert {"roc_auc", "brier_corrected", "ece", "log_loss"} <= set(scores["metric"])


def test_coverage_reports_settled_for_framed(tmp_path):
    out = _run_arm(tmp_path)
    cov = pd.read_parquet(out.dir / "coverage.parquet")
    # the *unconditioned* first_birth by_age 40: persons whose first birth precedes the jump-off
    # are settled (answer already in the observed prefix) and excluded.
    unconditioned = cov[(cov["outcome"] == "first_birth_by_age_40y") & (cov["condition"] == "-")]
    assert unconditioned["n_settled"].sum() > 0
