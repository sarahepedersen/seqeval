"""Replicate engine: exact estimators, MC-error correction, null band, resampling diagnostics."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest
from scipy.special import logit

from seqeval.core import outcomes as O
from seqeval.core import replicates as R
from seqeval.core.specs import CountQuery, Frame, ReplicateSpec, TTESpec
from seqeval.units import years_to_days as yd
from tests import synthetic as S

RK = ["person_id", "age_start", "age_stop"]
GK = ["person_id", "seed", "age_start", "age_stop"]


def _summary(k, n):
    k = np.asarray(k)
    return pd.DataFrame(
        {
            "person_id": np.arange(len(k)),
            "age_start": 0,
            "age_stop": 100,
            "k": k,
            "n": n,
        }
    )


# --- exact estimator / interval / logit values --------------------------------------------------
def test_estimators_exact_including_edges():
    summ = _summary([0, 1, 3, 5], 5)
    est = R.estimate_probability(summ, spec=ReplicateSpec(estimator="jeffreys")).set_index("k")
    np.testing.assert_allclose(est["p_hat"], [0.5 / 6, 1.5 / 6, 3.5 / 6, 5.5 / 6])
    np.testing.assert_allclose(
        est["logit_emp"], [np.log(0.5 / 5.5), np.log(1 / 3), np.log(1.4), np.log(11)]
    )
    np.testing.assert_allclose(
        est["var_logit"],
        [2 + 1 / 5.5, 1 / 1.5 + 1 / 4.5, 1 / 3.5 + 1 / 2.5, 1 / 5.5 + 2],
    )


def test_estimator_variants():
    summ = _summary([1], 5)
    mle = R.estimate_probability(summ, spec=ReplicateSpec(estimator="mle")).iloc[0]
    lap = R.estimate_probability(summ, spec=ReplicateSpec(estimator="laplace")).iloc[0]
    assert mle["p_hat"] == pytest.approx(0.2)
    assert lap["p_hat"] == pytest.approx(2 / 7)
    # logit_emp is Haldane-Anscombe regardless of estimator
    assert mle["logit_emp"] == pytest.approx(np.log(1 / 3))
    assert lap["logit_emp"] == pytest.approx(np.log(1 / 3))


def test_coherence_logit_equals_logit_of_phat_for_jeffreys():
    summ = _summary([0, 1, 2, 3, 4, 5], 5)
    est = R.estimate_probability(summ, spec=ReplicateSpec(estimator="jeffreys"))
    np.testing.assert_allclose(est["logit_emp"], logit(est["p_hat"]), atol=1e-12)


def test_jeffreys_interval_boundaries():
    est = R.estimate_probability(
        _summary([0, 5], 5), spec=ReplicateSpec(interval="jeffreys", level=0.95)
    )
    assert est.iloc[0]["ci_lo"] == 0.0  # k == 0
    assert est.iloc[1]["ci_hi"] == 1.0  # k == n
    assert 0 < est.iloc[0]["ci_hi"] < 1
    assert 0 < est.iloc[1]["ci_lo"] < 1


def test_wilson_interval_in_unit_range():
    est = R.estimate_probability(_summary([0, 2, 5], 5), spec=ReplicateSpec(interval="wilson"))
    assert (est["ci_lo"] >= 0).all() and (est["ci_hi"] <= 1).all()
    assert (est["ci_lo"] <= est["ci_hi"]).all()


def test_dynamic_range_bound():
    # |logit_emp| <= ln(2n + 1)
    for n in (5, 50):
        est = R.estimate_probability(_summary([0, n], n), spec=ReplicateSpec(estimator="jeffreys"))
        assert np.abs(est["logit_emp"]).max() == pytest.approx(np.log(2 * n + 1))


# --- replicate_summary --------------------------------------------------------------------------
def test_replicate_summary_filters_evaluable():
    tbl = pd.DataFrame(
        {
            "person_id": [1, 1, 1],
            "age_start": 0,
            "age_stop": 100,
            "seed": [0, 1, 2],
            "occurred": [True, False, True],
            "evaluable": [True, True, False],  # last dropped
        }
    )
    summ = R.replicate_summary(tbl, run_keys=RK)
    assert summ.iloc[0]["k"] == 1 and summ.iloc[0]["n"] == 2


def test_ragged_n_warns(caplog):
    # two runs in the same window with different n -> informative-censoring warning
    tbl = pd.DataFrame(
        {
            "person_id": [1, 1, 2],
            "age_start": 0,
            "age_stop": 100,
            "seed": [0, 1, 0],
            "occurred": [True, False, True],
            "evaluable": True,
        }
    )
    with caplog.at_level(logging.WARNING, logger="seqeval"):
        R.replicate_summary(tbl, run_keys=RK)
    assert any("informative censoring" in r.message for r in caplog.records)


# --- MC error: Brier correction (the load-bearing test) -----------------------------------------
def _calibration_pipeline(
    hazards_gen, obs, pers, window, jumpoff, spec_cq, n_seeds, rng, estimator
):
    gen = S.simulate_generated(obs, pers, hazards_gen, [window], n_seeds, rng)
    sp = O.observation_spans(gen, GK)
    ev = O.evaluate_count(gen, GK, spec_cq, sp, jumpoff=jumpoff)
    summ = R.replicate_summary(ev, run_keys=RK)
    est = R.estimate_probability(summ, spec=ReplicateSpec(estimator=estimator))
    return summ, est


def _observed_y(obs, spec_cq, jumpoff):
    osp = O.observation_spans(obs, ["person_id"])
    return (
        O.evaluate_count(obs, ["person_id"], spec_cq, osp, jumpoff=jumpoff)
        .set_index("person_id")["occurred"]
        .astype(float)
    )


def test_perfect_model_brier_inflation_and_correction():
    rng = np.random.default_rng(7)
    h = S.default_hazards()
    obs, pers = S.simulate_cohort(3000, (1960, 1990), h, None, rng)
    window, jo = (0.0, 30.0), yd(30)
    spec = CountQuery("b1w10", "birth", 1, Frame("within", yd(10)))
    y = _observed_y(obs, spec, jo)

    s5, e5 = _calibration_pipeline(h, obs, pers, window, jo, spec, 5, rng, "mle")
    _, e50 = _calibration_pipeline(h, obs, pers, window, jo, spec, 50, rng, "mle")

    def brier(est):
        m = est.set_index("person_id")
        return float(np.mean((m["p_hat"] - y.reindex(m.index)) ** 2))

    raw5, near_truth = brier(e5), brier(e50)
    correction = R.brier_noise_correction(s5)
    corrected5 = raw5 - correction

    assert raw5 > near_truth  # few seeds inflate Brier
    assert correction > 0.02  # correction is large at n=5
    assert corrected5 == pytest.approx(near_truth, abs=0.02)  # correction recovers truth


# --- null calibration band ----------------------------------------------------------------------
def _band_coverage(summ, est, y, n_bins=10):
    rng = np.random.default_rng(0)
    band = R.null_calibration_band(summ, n_bins=n_bins, n_sims=300, rng=rng, estimator="jeffreys")
    p = est.set_index("person_id")["p_hat"].to_numpy()
    yy = y
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
    inside = []
    for b in range(n_bins):
        sel = idx == b
        if sel.sum() < 10 or np.isnan(band.iloc[b]["lo"]):
            continue
        freq = yy[sel].mean()
        inside.append(band.iloc[b]["lo"] - 1e-9 <= freq <= band.iloc[b]["hi"] + 1e-9)
    return float(np.mean(inside))


def test_null_band_covers_calibrated_data():
    # Data that is genuinely perfectly calibrated: y ~ Bernoulli(p_true), estimates from n seeds.
    covs = []
    for trial in range(3):
        r = np.random.default_rng(trial)
        p_true = r.uniform(0.05, 0.95, 3000)
        n = 20
        k = r.binomial(n, p_true)
        summ = _summary(k, n)
        est = R.estimate_probability(summ, spec=ReplicateSpec(estimator="jeffreys"))
        y = (r.random(3000) < p_true).astype(float)
        covs.append(_band_coverage(summ, est, y))
    assert np.mean(covs) >= 0.85  # ~95% nominal; loose for MC + finite bins


def test_null_band_flags_miscalibration():
    # Overconfident forecasts: reported k drawn from p_model = p_true + 0.25, truth stays at p_true.
    r = np.random.default_rng(1)
    p_true = r.uniform(0.05, 0.7, 3000)
    p_model = np.clip(p_true + 0.25, 0, 1)
    n = 20
    k = r.binomial(n, p_model)
    summ = _summary(k, n)
    est = R.estimate_probability(summ, spec=ReplicateSpec(estimator="jeffreys"))
    y = (r.random(3000) < p_true).astype(float)
    assert _band_coverage(summ, est, y) < 0.6  # curve exits the band


def test_null_band_widens_at_small_n():
    # Same predictions, fewer seeds -> wider per-prediction envelope for a fixed prediction value.
    r = np.random.default_rng(2)
    p_true = np.full(4000, 0.5)  # all runs at p=0.5 so one bin is densely populated
    widths = {}
    for n in (5, 50):
        k = r.binomial(n, p_true)
        band = R.null_calibration_band(
            _summary(k, n),
            n_bins=10,
            n_sims=400,
            rng=np.random.default_rng(3),
            estimator="jeffreys",
        )
        mid = band.iloc[5]  # bin containing 0.5
        widths[n] = mid["hi"] - mid["lo"]
    assert widths[5] > widths[50]


# --- timing distribution + coverage -------------------------------------------------------------
def test_timing_distribution_shape():
    tte = pd.DataFrame(
        {
            "person_id": [1, 1, 1, 1],
            "age_start": 0,
            "age_stop": 0,
            "seed": [0, 1, 2, 3],
            "duration": [100, 200, 5000, 300],  # last is beyond horizon
            "observed": [True, True, True, False],
        }
    )
    td = R.timing_distribution(tte, run_keys=RK, seed_col="seed", horizon=1000)
    row = td.iloc[0]
    assert row["n"] == 4
    assert row["n_occurred"] == 2  # durations 100, 200 within horizon (5000 capped, 300 unobserved)
    assert row["p_within_horizon"] == pytest.approx(0.5)


def test_timing_interval_coverage_perfect_model():
    rng = np.random.default_rng(5)
    h = S.default_hazards()
    obs, pers = S.simulate_cohort(2000, (1960, 1990), h, None, rng)
    gen = S.simulate_generated(obs, pers, h, [(0.0, 0.0)], 40, rng)  # full-life perfect model
    tte = TTESpec("birth", 1)
    td = R.timing_distribution(
        O.time_to_event(gen, GK, tte), run_keys=RK, seed_col="seed", horizon=yd(50)
    )
    obs_tte = O.time_to_event(obs, ["person_id"], tte).set_index("person_id")
    m = td.set_index("person_id").join(obs_tte[["duration", "observed"]])
    seen = m[m["observed"]]
    coverage = ((seen["duration"] >= seen["q10"]) & (seen["duration"] <= seen["q90"])).mean()
    assert 0.7 <= coverage <= 0.9  # nominal 0.8, loose


# --- count distribution -------------------------------------------------------------------------
def test_count_distribution_and_moments():
    ct = pd.DataFrame(
        {
            "person_id": [1, 1, 1, 1],
            "age_start": 0,
            "age_stop": 0,
            "seed": [0, 1, 2, 3],
            "count": [0, 1, 1, 2],
        }
    )
    pmf = R.count_distribution(ct, run_keys=RK, seed_col="seed").set_index("count")["prob"]
    assert pmf.loc[0] == pytest.approx(0.25)
    assert pmf.loc[1] == pytest.approx(0.5)
    assert pmf.sum() == pytest.approx(1.0)
    mom = R.count_moments(ct, run_keys=RK, seed_col="seed").iloc[0]
    assert mom["mean"] == pytest.approx(1.0)
    assert mom["var"] == pytest.approx(0.5)  # population variance of [0,1,1,2]


# --- resampling ---------------------------------------------------------------------------------
def _ccf_stat(df):
    b = df[df["event"] == "birth"]
    return pd.DataFrame({"ccf": [len(b) / df["seed"].nunique() / df["person_id"].nunique()]})


def test_seed_bootstrap_ci_covers_truth():
    rng = np.random.default_rng(5)
    h = S.default_hazards()
    obs, pers = S.simulate_cohort(1500, (1960, 1990), h, None, rng)
    gen = S.simulate_generated(obs, pers, h, [(0.0, 0.0)], 20, rng)
    bs = R.seed_bootstrap(
        gen, seed_col="seed", stat_fn=_ccf_stat, n_boot=300, rng=np.random.default_rng(1)
    ).iloc[0]
    assert bs["ci_lo"] <= bs["estimate"] <= bs["ci_hi"]
    assert bs["ci_lo"] <= S.expected_ccf(h) <= bs["ci_hi"]


def test_convergence_dispersion_decreases():
    rng = np.random.default_rng(5)
    h = S.default_hazards()
    obs, pers = S.simulate_cohort(1500, (1960, 1990), h, None, rng)
    gen = S.simulate_generated(obs, pers, h, [(0.0, 0.0)], 20, rng)
    cc = R.convergence_curve(
        gen,
        seed_col="seed",
        stat_fn=_ccf_stat,
        sizes=[2, 5, 10, 20],
        n_rep=15,
        rng=np.random.default_rng(2),
    ).set_index("m")
    assert cc.loc[2, "std"] > cc.loc[20, "std"]  # dispersion shrinks with more seeds
