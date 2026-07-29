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


def _grid(p_hat, y_true, n_seeds):
    """A joined frame whose p_hat sits on the `k/n` grid a run of `n_seeds` replicates produces."""
    return pd.DataFrame(
        {
            "p_hat": p_hat,
            "y_true": y_true,
            "k": np.round(np.asarray(p_hat) * n_seeds).astype(int),
            "n": n_seeds,
            "person_id": np.arange(len(p_hat)),
        }
    )


def test_quantile_bins_cannot_split_a_tie():
    """p_hat = k/n is atomic, so a coarse replicate grid caps how many bins can exist."""
    rng = np.random.default_rng(0)
    # 5 seeds -> p_hat has 6 possible values, so 10 bins are unreachable whatever the sample size
    p = rng.choice([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], size=4000, p=[0.01, 0.04, 0.05, 0.2, 0.4, 0.3])
    cal = ml.calibration_table(_grid(p, rng.random(4000) < p, 5), n_bins=10, strategy="quantile")
    assert 2 <= len(cal) <= 6  # at most one bin per distinct p_hat, never the 10 requested
    assert cal["n"].sum() == 4000  # nobody is dropped by the collapse

    # 50 seeds -> a finer grid, so the requested count is essentially reachable
    p50 = np.round(rng.beta(6, 2, size=4000) * 50) / 50
    joined50 = _grid(p50, rng.random(4000) < p50, 50)
    cal50 = ml.calibration_table(joined50, n_bins=10, strategy="quantile")
    assert len(cal50) >= 9
    # and the bins are near-equal-count, which is what quantile binning is for
    assert cal50["n"].max() / cal50["n"].min() < 4


def test_a_single_distinct_p_hat_still_yields_one_bin():
    """A degenerate p_hat must not silently produce an empty table and a NaN ECE."""
    joined = _grid(np.full(50, 0.4), np.r_[np.ones(20), np.zeros(30)], 5)
    cal = ml.calibration_table(joined, n_bins=10, strategy="quantile")
    assert len(cal) == 1
    assert cal["n"].iloc[0] == 50
    assert cal["p_mean"].iloc[0] == pytest.approx(0.4)
    assert cal["y_rate"].iloc[0] == pytest.approx(0.4)
    assert ml.ece(cal) == pytest.approx(0.0, abs=1e-12)


def test_every_person_lands_in_exactly_one_bin():
    rng = np.random.default_rng(1)
    p = np.round(rng.random(2000) * 50) / 50
    for strategy in ("quantile", "uniform"):
        cal = ml.calibration_table(_grid(p, rng.random(2000) < p, 50), strategy=strategy)
        assert cal["n"].sum() == 2000, strategy


def test_p_hat_distribution_is_the_grid_not_the_bins():
    """The distribution lives on the k/n atoms; calibration bins are a different grouping."""
    rng = np.random.default_rng(0)
    p = rng.choice([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], size=2000, p=[0.3, 0.35, 0.2, 0.1, 0.04, 0.01])
    joined = _grid(p, rng.random(2000) < p, 5)

    d = ml.p_hat_distribution(joined, min_cell=0)
    # every attainable value gets a row, so a gap reads as a true zero, not as an impossible value
    np.testing.assert_allclose(d["p_hat"], [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    assert d["n_persons"].sum() == 2000
    assert (d["n_total"] == 2000).all()

    # the top calibration bin spans a wide range, but the mass inside it sits on its left edge
    cal = ml.calibration_table(joined, n_bins=10, strategy="quantile")
    top = cal.iloc[-1]
    inside = d[(d["p_hat"] >= top["bin_left"]) & (d["p_hat"] <= top["bin_right"])]
    assert inside["n_persons"].sum() == top["n"]
    assert inside.iloc[0]["n_persons"] / top["n"] > 0.5  # most of the bar is its leftmost atom


def test_p_hat_distribution_withholds_thin_atoms():
    p = np.r_[np.full(300, 0.2), np.full(300, 0.4), np.full(3, 1.0)]
    d = ml.p_hat_distribution(_grid(p, np.zeros(len(p), dtype=bool), 5), min_cell=5)
    hidden = d[d["suppressed"]]
    assert len(hidden) >= 2  # the lone thin atom drags a second one with it
    assert hidden["n_persons"].isna().all()
    # a value nobody reached is a published zero, never a suppressed cell
    zeros = d[(d["n_persons"] == 0)]
    assert not zeros["suppressed"].any()


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


def _tte_frames(pred_years, obs_years, *, obs_observed=None, pred_observed=None, n_seeds=1):
    """A (generated tte, observed tte) pair — one generated row per ``(person, seed)`` trajectory.

    ``pred_observed`` marks trajectories where the model produced the outcome at all; the default
    is every one of them, so a test says nothing about exclusion unless it asks to.
    """
    n = len(pred_years)
    person = np.arange(n)
    gen = pd.DataFrame(
        {
            "person_id": np.tile(person, n_seeds),
            "seed": np.repeat(np.arange(n_seeds), n),
            "duration": np.tile(np.array([yd(v) for v in pred_years], dtype=np.int64), n_seeds),
            "observed": (
                np.ones(n * n_seeds, dtype=bool)
                if pred_observed is None
                else np.tile(np.asarray(pred_observed, dtype=bool), n_seeds)
            ),
        }
    )
    obs_tte = pd.DataFrame(
        {
            "person_id": person,
            "duration": np.array([yd(v) for v in obs_years], dtype=np.int64),
            "observed": np.ones(n, dtype=bool) if obs_observed is None else obs_observed,
        }
    )
    return gen, obs_tte


def _pairs(pred_years, obs_years, *, horizon_days=None, persons=None, **kw):
    gen, obs_tte = _tte_frames(pred_years, obs_years, **kw)
    return ml.timing_pairs(gen, obs_tte, horizon_days=horizon_days, persons=persons)


def test_scope_keeps_only_events_observed_inside_the_horizon():
    pairs = _pairs([20, 30, 45], [22, 33, 40], obs_observed=[True, True, False],
                   horizon_days=yd(50))
    assert list(pairs["person_id"]) == [0, 1]  # censored person dropped
    # a person whose observed wait exceeds the horizon has no predicted counterpart
    assert list(_pairs([20, 30], [22, 60], horizon_days=yd(50))["person_id"]) == [0]


def test_unpredicted_trajectories_are_excluded_not_capped():
    """A trajectory the model never brought to the outcome has no predicted time to difference."""
    # person 1's trajectory runs past the frame; person 2's never sees the event at all
    pairs = _pairs([20, 60, 30], [22, 44, 33], pred_observed=[True, True, False],
                   horizon_days=yd(50))
    assert list(pairs["person_id"]) == [0, 1, 2]  # every candidate is kept as a row
    assert list(pairs["predicted"]) == [True, False, False]
    assert pairs.loc[pairs["person_id"] != 0, "pred"].isna().all()

    out = ml.timing_error_distribution(pairs, pred_bin_years=50, min_cell=0)
    assert out["n_trajectories"].iloc[0] == 3
    assert out["n_excluded"].iloc[0] == 2
    # nothing is parked on the horizon: only the one real prediction is binned
    assert out["n_pred_bin"].iloc[0] == 1


def test_scope_restricts_to_the_scored_population():
    """The arm passes its condition-minus-settled set, matching the reliability panel."""
    pairs = _pairs([20, 30, 35], [22, 33, 36], horizon_days=yd(50), persons={0, 2})
    assert list(pairs["person_id"]) == [0, 2]


def test_every_seed_contributes_its_own_error():
    """No per-person median: K seeds mean K rows, and K counts in the table."""
    pairs = _pairs([30, 30], [33, 33], n_seeds=4)
    assert len(pairs) == 8
    out = ml.timing_error_distribution(pairs, pred_bin_years=50, min_cell=0)
    assert out["n_pred_bin"].iloc[0] == 8
    assert out["n_trajectories"].iloc[0] == 8

    per_seed = ml.timing_error_distribution(pairs, by=["seed"], pred_bin_years=50, min_cell=0)
    assert sorted(per_seed["seed"].unique()) == [0, 1, 2, 3]
    assert (per_seed.groupby("seed")["n_pred_bin"].first() == 2).all()


def _uniform_pairs(n_per_bin=10, n_bins=6, offset=0.0):
    """``n_bins`` distinct predicted values, ``n_per_bin`` persons each, all off by ``offset``."""
    pred = np.repeat(np.arange(25, 25 + n_bins), n_per_bin).astype(float)
    return _pairs(pred, pred + offset)


def test_error_is_observed_minus_predicted():
    """Positive error means the event happened later than the model said."""
    pred = [30] * 5 + [31] * 5
    pairs = _pairs(pred, [p + 3 for p in pred])
    out = ml.timing_error_distribution(pairs, pred_bin_years=50)
    landed = out[out["n_persons"] == 10].iloc[0]
    assert landed["error_lo"] == pytest.approx(yd(3), abs=1)


def test_zero_is_always_a_bin_edge():
    """No cell may mix events that came early with events that came late."""
    for offset in (0.0, 0.4, -2.7, 1.5):
        out = ml.timing_error_distribution(_uniform_pairs(offset=offset))
        straddles = (out["error_lo"] < 0) & (out["error_hi"] > 0)
        assert not straddles.any()


def test_predicted_bins_are_the_same_intervals_across_jumpoffs():
    """Fixed-width bins, so the same predicted-age range is the same bin in every figure."""
    early = _uniform_pairs(n_per_bin=10, n_bins=10)                      # predicted ages 25..34
    late = _pairs([32.0] * 40 + [33.0] * 40, [35.0] * 80)                # predicted ages 32..33

    a = ml.timing_error_distribution(early, pred_bin_years=2)
    b = ml.timing_error_distribution(late, pred_bin_years=2)
    def edges(out):
        return out.groupby("pred_bin")[["pred_lo", "pred_hi"]].first()

    shared = edges(a).index.intersection(edges(b).index)
    assert len(shared)
    pd.testing.assert_frame_equal(edges(a).loc[shared], edges(b).loc[shared])
    # the later jump-off simply covers fewer of them, and none of its own are new
    assert set(edges(b).index) < set(edges(a).index)


def test_per_seed_and_pooled_tables_share_their_bins():
    """The two views of one run must line up, or a seed cannot be read against the pool."""
    pairs = _pairs(np.repeat(np.arange(25.0, 31.0), 8), np.repeat(np.arange(26.0, 32.0), 8),
                   n_seeds=3)
    pooled = ml.timing_error_distribution(pairs, pred_bin_years=2, min_cell=0)
    per_seed = ml.timing_error_distribution(pairs, by=["seed"], pred_bin_years=2, min_cell=0)
    edges = ["pred_lo", "pred_hi"]
    shared = (
        per_seed.groupby("pred_bin")[edges].first()
        .join(pooled.groupby("pred_bin")[edges].first(), rsuffix="_pooled", how="inner")
    )
    assert len(shared)
    np.testing.assert_allclose(shared["pred_lo"], shared["pred_lo_pooled"])
    np.testing.assert_allclose(shared["pred_hi"], shared["pred_hi_pooled"])


def test_predicted_bins_are_anchored_to_the_bin_width():
    """Anchoring is absolute, not relative to the data, or two runs would not line up."""
    out = ml.timing_error_distribution(_uniform_pairs(n_per_bin=5, n_bins=10), pred_bin_years=2)
    lo = out.groupby("pred_bin")["pred_lo"].first() / yd(2)
    np.testing.assert_allclose(lo, np.round(lo), atol=1e-6)


def test_empty_predicted_bins_are_dropped_not_drawn():
    """A gap in predicted values leaves no blank ridge."""
    pairs = _pairs([26.0] * 20 + [40.0] * 20, [28.0] * 40)
    out = ml.timing_error_distribution(pairs, pred_bin_years=2)
    assert (out.groupby("pred_bin")["n_pred_bin"].first() > 0).all()
    assert out["pred_bin"].nunique() == 2


def test_small_cells_are_withheld_and_unrecoverable():
    # one predicted bin: 20 people arrive a year late, 3 stragglers six years late
    pred = [30, 31] * 10 + [30, 31, 30]
    obs = [p + 1 for p in pred[:20]] + [p + 6 for p in pred[20:]]
    out = ml.timing_error_distribution(_pairs(pred, obs), pred_bin_years=50)
    hidden = out[out["suppressed"]]
    assert len(hidden) >= 2  # the lone small cell drags a second one with it
    assert hidden["n_persons"].isna().all()
    assert out["n_persons"].sum() < out["n_pred_bin"].iloc[0]


def test_output_carries_no_person_identifier():
    """The table the ridge is drawn from must name nobody."""
    out = ml.timing_error_distribution(_uniform_pairs())
    assert "person_id" not in out.columns
    assert (out["n_persons"].dropna() >= 0).all()


def test_a_single_predicted_value_still_gets_its_own_fixed_bin():
    """Fixed bins need no spread in the predictions; the value falls where it falls."""
    out = ml.timing_error_distribution(_pairs([30] * 5, [31] * 5), pred_bin_years=2, min_cell=0)
    assert out["pred_bin"].nunique() == 1
    assert out["pred_lo"].iloc[0] <= yd(30) < out["pred_hi"].iloc[0]


def test_no_pairs_returns_an_empty_frame():
    assert ml.timing_error_distribution(_pairs([], [])).empty


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
