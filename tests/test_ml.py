"""ML/probability metrics (04) — composition over 02b; the probability stats are tested in 02b."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from seqeval.core.specs import ReplicateSpec
from seqeval.metrics import ml
from seqeval.units import years_to_days as yd

RK = ["person_id", "age_start", "age_stop"]


def _gen_eval(k_by_person, n):
    """Build a replicate-level evaluator frame where person i has k_i occurrences out of n."""
    rows = []
    for pid, k in enumerate(k_by_person):
        for seed in range(n):
            rows.append(
                {
                    "person_id": pid,
                    "age_start": 0,
                    "age_stop": 100,
                    "seed": seed,
                    "occurred": seed < k,
                    "evaluable": True,
                }
            )
    return pd.DataFrame(rows)


def test_probability_table_columns_and_warning(caplog):
    gen = _gen_eval([0, 2, 5], n=5)
    with caplog.at_level(logging.WARNING, logger="seqeval"):
        probs = ml.probability_table(gen, ReplicateSpec(min_replicates=10))
    assert list(probs.columns[-6:]) == [
        "k",
        "n",
        "p_hat",
        "logit_emp",
        "var_logit",
        "ci_lo",
    ] or {"p_hat", "logit_emp", "ci_lo", "ci_hi"} <= set(probs.columns)
    assert any("min_replicates" in r.message for r in caplog.records)


def test_join_truth_keeps_evaluable_both_sides():
    probs = pd.DataFrame({"person_id": [1, 2, 3], "p_hat": [0.2, 0.8, 0.5], "k": [1, 4, 2], "n": 5})
    obs = pd.DataFrame(
        {"person_id": [1, 2, 3], "occurred": [False, True, True], "evaluable": [True, True, False]}
    )
    joined = ml.join_truth(probs, obs)
    assert set(joined["person_id"]) == {1, 2}  # person 3 not evaluable on truth
    assert joined.set_index("person_id").loc[2, "y_true"] == 1


def test_calibration_ece_and_perfect_case():
    # perfectly calibrated toy: p_hat == y_rate in each bin
    joined = pd.DataFrame(
        {
            "p_hat": [0.1] * 100 + [0.9] * 100,
            "y_true": [0] * 90 + [1] * 10 + [0] * 10 + [1] * 90,
            "k": 1,
            "n": 5,
        }
    )
    cal = ml.calibration_table(joined, n_bins=5, strategy="uniform")
    assert ml.ece(cal) == pytest.approx(0.0, abs=1e-9)


def test_roc_auc_perfect_and_degenerate():
    perfect = pd.DataFrame({"p_hat": [0.1, 0.2, 0.8, 0.9], "y_true": [0, 0, 1, 1], "k": 1, "n": 5})
    assert ml.roc_auc(perfect) == pytest.approx(1.0)
    one_class = pd.DataFrame({"p_hat": [0.1, 0.8], "y_true": [1, 1], "k": 1, "n": 5})
    assert np.isnan(ml.roc_auc(one_class))


def test_brier_raw_and_corrected():
    joined = pd.DataFrame({"p_hat": [0.5, 0.5], "y_true": [0, 1], "k": [2, 3], "n": [5, 5]})
    b = ml.brier(joined)
    assert b["raw"] == pytest.approx(0.25)
    assert b["corrected"] < b["raw"]  # correction subtracts the MC-error inflation


def test_log_loss_matches_manual():
    joined = pd.DataFrame({"p_hat": [0.25, 0.75], "y_true": [0, 1], "k": 1, "n": 5})
    expected = -np.mean([np.log(0.75), np.log(0.75)])
    assert ml.log_loss(joined) == pytest.approx(expected)


def test_mse_uses_raw_rate_not_smoothed():
    # rate = k/n = [0.4, 0.6]; MSE against y = [0, 1] ignores p_hat entirely
    joined = pd.DataFrame({"p_hat": [0.9, 0.1], "y_true": [0, 1], "k": [2, 3], "n": [5, 5]})
    expected = np.mean([(0.4 - 0) ** 2, (0.6 - 1) ** 2])
    assert ml.mse(joined) == pytest.approx(expected)


def test_r2_raw_rate_and_degenerate():
    # rate = [0.4, 0.6], y = [0, 1], ȳ = 0.5 -> SS_res=0.32, SS_tot=0.5 -> R² = 1 - 0.64
    joined = pd.DataFrame({"y_true": [0, 1], "k": [2, 3], "n": [5, 5]})
    assert ml.r2(joined) == pytest.approx(1 - 0.32 / 0.5)
    # no variance in the outcome -> R² undefined
    flat = pd.DataFrame({"y_true": [1, 1], "k": [2, 3], "n": [5, 5]})
    assert np.isnan(ml.r2(flat))


def test_timing_coverage():
    td = pd.DataFrame({"person_id": [1, 2, 3], "q10": [10, 10, 10], "q90": [20, 20, 20]})
    obs = pd.DataFrame(
        {"person_id": [1, 2, 3], "duration": [15, 25, 5], "observed": [True, True, True]}
    )
    assert ml.timing_coverage(td, obs) == pytest.approx(1 / 3)  # only person 1 inside [10, 20]


def test_subgroup_rates():
    gen = pd.DataFrame(
        {
            "cohort": [1960, 1960, 1970, 1970],
            "occurred": [True, False, True, True],
            "evaluable": True,
        }
    )
    obs = pd.DataFrame({"cohort": [1960, 1970], "occurred": [False, True], "evaluable": True})
    sr = ml.subgroup_rates(gen, obs, by=["cohort"]).set_index("cohort")
    assert sr.loc[1960, "pred_rate"] == pytest.approx(0.5)
    assert sr.loc[1970, "obs_rate"] == pytest.approx(1.0)


# --- aggregate_error ----------------------------------------------------------------------------
def _ccf_gen():
    return pd.DataFrame(
        {
            "age_start": 0,
            "age_stop": 30,
            "seed": [0, 0, 1, 1],
            "cohort": [1960, 1970, 1960, 1970],
            "ccf": [2.0, 1.8, 2.2, 2.0],
        }
    )


def test_aggregate_error_exact():
    obs = pd.DataFrame({"cohort": [1960, 1970], "ccf": [2.1, 1.9]})
    ae = ml.aggregate_error(_ccf_gen(), obs, value_col="ccf", on=["cohort"]).set_index("cohort")
    # cohort 1960: gen [2.0, 2.2] vs obs 2.1 -> errors [-0.1, 0.1] -> bias 0, rmse sqrt(0.01)
    assert ae.loc[1960, "gen_mean"] == pytest.approx(2.1)
    assert ae.loc[1960, "bias"] == pytest.approx(0.0)
    assert ae.loc[1960, "rmse"] == pytest.approx(0.1)


def test_aggregate_error_alignment_raises():
    obs = pd.DataFrame({"cohort": [1960, 1980], "ccf": [2.1, 1.9]})  # 1980 not in gen, 1970 missing
    with pytest.raises(ValueError, match="do not align"):
        ml.aggregate_error(_ccf_gen(), obs, value_col="ccf", on=["cohort"])


def _timing_frames(pred_years, obs_years, observed=None):
    """A (timing_distribution, observed tte) pair with one person per predicted/observed value."""
    n = len(pred_years)
    person = np.arange(n)
    td = pd.DataFrame({"person_id": person, "q50": [yd(v) for v in pred_years]})
    obs_tte = pd.DataFrame(
        {
            "person_id": person,
            "duration": [yd(v) for v in obs_years],
            "observed": np.ones(n, dtype=bool) if observed is None else observed,
        }
    )
    return td, obs_tte


def _scope(td, obs_tte, **kw):
    kw.setdefault("horizon_days", None)
    kw.setdefault("persons", None)
    kw.setdefault("drop_projected_beyond", True)
    return ml._timing_scope(td, obs_tte, **kw)


def test_scope_keeps_only_events_observed_inside_the_horizon():
    td, obs_tte = _timing_frames([20, 30, 45], [22, 33, 40], observed=[True, True, False])
    scoped = _scope(td, obs_tte, horizon_days=yd(50))
    assert list(scoped["person_id"]) == [0, 1]  # censored person dropped
    # a person whose observed wait exceeds the horizon has no predicted counterpart
    td, obs_tte = _timing_frames([20, 30], [22, 60])
    assert list(_scope(td, obs_tte, horizon_days=yd(50))["person_id"]) == [0]


def test_scope_drops_persons_projected_past_the_frame():
    """A predicted median sitting on the horizon is the cap, not a date — not timing signal."""
    td, obs_tte = _timing_frames([20, 50], [22, 44])
    assert list(_scope(td, obs_tte, horizon_days=yd(50))["person_id"]) == [0]
    kept = _scope(td, obs_tte, horizon_days=yd(50), drop_projected_beyond=False)
    assert list(kept["person_id"]) == [0, 1]


def test_scope_restricts_to_the_scored_population():
    """The arm passes its condition-minus-settled set, matching the reliability panel."""
    td, obs_tte = _timing_frames([20, 30, 35], [22, 33, 36])
    scoped = _scope(td, obs_tte, horizon_days=yd(50), persons={0, 2})
    assert list(scoped["person_id"]) == [0, 2]


def _uniform_pairs(n_per_bin=10, n_bins=6, offset=0.0):
    """``n_bins`` distinct predicted values, ``n_per_bin`` persons each, all off by ``offset``."""
    pred = np.repeat(np.arange(25, 25 + n_bins), n_per_bin).astype(float)
    return _timing_frames(pred, pred + offset)


def test_error_is_observed_minus_predicted():
    """Positive error means the event happened later than the model said."""
    pred = [30] * 5 + [31] * 5
    td, obs_tte = _timing_frames(pred, [p + 3 for p in pred])
    out = ml.timing_error_distribution(td, obs_tte, n_pred_bins=1)
    landed = out[out["n_persons"] == 10].iloc[0]
    assert landed["error_lo"] == pytest.approx(yd(3), abs=1)


def test_zero_is_always_a_bin_edge():
    """No cell may mix events that came early with events that came late."""
    for offset in (0.0, 0.4, -2.7, 1.5):
        td, obs_tte = _uniform_pairs(offset=offset)
        out = ml.timing_error_distribution(td, obs_tte)
        straddles = (out["error_lo"] < 0) & (out["error_hi"] > 0)
        assert not straddles.any()


def test_predicted_bins_rest_on_equal_numbers_of_people():
    td, obs_tte = _uniform_pairs(n_per_bin=10, n_bins=6)
    out = ml.timing_error_distribution(td, obs_tte, n_pred_bins=6)
    assert sorted(out.groupby("pred_bin")["n_pred_bin"].first()) == [10] * 6


def test_small_cells_are_withheld_and_unrecoverable():
    # one predicted bin: 20 people arrive a year late, 3 stragglers six years late
    pred = [30, 31] * 10 + [30, 31, 30]
    obs = [p + 1 for p in pred[:20]] + [p + 6 for p in pred[20:]]
    td, obs_tte = _timing_frames(pred, obs)
    out = ml.timing_error_distribution(td, obs_tte, n_pred_bins=1)
    hidden = out[out["suppressed"]]
    assert len(hidden) >= 2  # the lone small cell drags a second one with it
    assert hidden["n_persons"].isna().all()
    assert out["n_persons"].sum() < out["n_pred_bin"].iloc[0]


def test_output_carries_no_person_identifier():
    """The table the ridge is drawn from must name nobody."""
    td, obs_tte = _uniform_pairs()
    out = ml.timing_error_distribution(td, obs_tte)
    assert "person_id" not in out.columns
    assert (out["n_persons"].dropna() >= 0).all()


def test_too_few_distinct_predictions_returns_an_empty_frame():
    """One predicted value gives nothing to compare across, so no ridge is drawn."""
    td, obs_tte = _timing_frames([30] * 5, [31] * 5)
    out = ml.timing_error_distribution(td, obs_tte, n_pred_bins=6)
    assert out.empty


# --- analytic score intervals -------------------------------------------------------------------
def _scored(n=800, seed=0, k_seeds=5):
    """A joined frame with a real signal, on the coarse ``1/k_seeds`` probability grid."""
    rng = np.random.default_rng(seed)
    truth = rng.uniform(0.1, 0.9, n)
    y = (rng.uniform(size=n) < truth).astype(int)
    k = rng.binomial(k_seeds, truth)
    return pd.DataFrame({"p_hat": k / k_seeds, "y_true": y, "k": k, "n": k_seeds})


def test_every_interval_contains_its_own_point_estimate():
    """The defect that retired the seed bootstrap: its CIs sat off to one side of the estimate."""
    joined = _scored()
    cis = ml.score_cis(joined).set_index("metric")
    points = {
        "mse": ml.mse(joined),
        "brier_raw": ml.brier(joined)["raw"],
        "brier_corrected": ml.brier(joined)["corrected"],
        "r2": ml.r2(joined),
        "roc_auc": ml.roc_auc(joined),
    }
    for metric, value in points.items():
        lo, hi = cis.loc[metric, "ci_lo"], cis.loc[metric, "ci_hi"]
        assert lo <= value <= hi, metric


def test_squared_error_intervals_are_the_person_level_standard_error():
    joined = _scored()
    loss = (joined["p_hat"] - joined["y_true"]) ** 2
    half = 1.959963985 * loss.std(ddof=1) / np.sqrt(len(joined))
    row = ml.score_cis(joined).set_index("metric").loc["mse"]
    assert (row["ci_hi"] - row["ci_lo"]) / 2 == pytest.approx(half)


def test_ece_gets_no_interval():
    """Data-dependent bins and an upward-biased statistic: no honest closed form to report."""
    assert "ece" not in set(ml.score_cis(_scored())["metric"])


def test_auc_interval_stays_a_probability():
    """A strong classifier on few people pushes the symmetric interval past 1; it is clipped."""
    joined = pd.DataFrame(
        {"p_hat": [0.2] * 7 + [0.8] * 6, "y_true": [0] * 6 + [1] * 7, "k": 1, "n": 5}
    )
    row = ml.score_cis(joined).set_index("metric").loc["roc_auc"]
    assert row["ci_hi"] == 1.0 and row["ci_lo"] > 0.0


def test_delong_variance_matches_resampling_persons():
    """DeLong is the analytic form of what a person-level resample would estimate."""
    from sklearn.metrics import roc_auc_score

    joined = _scored(n=1500, seed=3)
    y, p = joined["y_true"].to_numpy(), joined["p_hat"].to_numpy()
    rng = np.random.default_rng(11)
    draws = [
        roc_auc_score(y[idx], p[idx])
        for idx in (rng.integers(0, len(y), len(y)) for _ in range(300))
        if len(np.unique(y[idx])) == 2
    ]
    assert np.sqrt(ml._delong_var(y, p)) == pytest.approx(np.std(draws, ddof=1), rel=0.15)


def test_no_auc_interval_where_delong_carries_no_information():
    """Every prediction tied, or perfectly separated, leaves DeLong with zero variance.

    A zero-width interval would claim certainty, so the metric is reported with no interval at all
    rather than a false one — the same treatment ECE gets.
    """
    tied = pd.DataFrame({"p_hat": [0.5] * 40, "y_true": [0] * 20 + [1] * 20, "k": 2, "n": 5})
    assert ml.roc_auc(tied) == pytest.approx(0.5)
    assert "roc_auc" not in set(ml.score_cis(tied)["metric"])

    separated = pd.DataFrame(
        {"p_hat": [0.0] * 20 + [1.0] * 20, "y_true": [0] * 20 + [1] * 20, "k": 0, "n": 5}
    )
    assert ml.roc_auc(separated) == pytest.approx(1.0)
    assert "roc_auc" not in set(ml.score_cis(separated)["metric"])


def test_level_widens_the_interval():
    joined = _scored()
    narrow = ml.score_cis(joined, level=0.80).set_index("metric").loc["mse"]
    wide = ml.score_cis(joined, level=0.99).set_index("metric").loc["mse"]
    assert (wide["ci_hi"] - wide["ci_lo"]) > (narrow["ci_hi"] - narrow["ci_lo"])
