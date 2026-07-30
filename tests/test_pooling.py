"""Pooling K synthetic populations (05b): the plain sampling variance, and the recorded diagnostics.

The point estimate is the metric over every trajectory at once, and so is its interval: the textbook
formula on exactly those units, with no correction for the fact that the N×K pooled rows are not N×K
independent people. What these tests pin down is that the interval really is the pooled table's own
variance, and that the two inputs a downstream correction would need (``mean_var``, ``between_var``)
are still measured and emitted. ``design_effect_var`` is that correction, kept and tested here but
wired into nothing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from seqeval.arms._common import pool_seeds
from seqeval.metrics import pooling
from seqeval.metrics.survival import step_sample

Z = norm.ppf(0.975)


# =================================================================================================
# design_effect_var — the deferred correction: the formula and its limits
# =================================================================================================
def test_identical_seeds_keep_one_populations_width():
    """Seeds that agree exactly are K copies of one population, so K buys nothing."""
    var = pooling.design_effect_var(mean_var=0.04, between_var=0.0, k=10)
    assert var == pytest.approx(0.04)


def test_independent_seeds_reach_the_full_n_times_k_width():
    """When the seeds behave like independent samples the pooled table really is N·K people."""
    # between_var == mean_var is the independent case: the spread of the seed estimates is the
    # whole of one seed's sampling variance
    var = pooling.design_effect_var(mean_var=0.04, between_var=0.04, k=8)
    assert var == pytest.approx(0.04 / 8)


def test_partial_duplication_lands_between_the_limits():
    var = pooling.design_effect_var(mean_var=0.04, between_var=0.02, k=8)
    assert 0.04 / 8 < var < 0.04


def test_a_single_seed_keeps_its_own_variance():
    """One population is all the evidence there is; widening or narrowing it would be invention."""
    assert pooling.design_effect_var(0.04, np.nan, 1) == pytest.approx(0.04)
    assert pooling.design_effect_var(0.04, 0.0, 1) == pytest.approx(0.04)


def test_result_is_always_inside_the_clip_bounds():
    rng = np.random.default_rng(0)
    mean_var = rng.uniform(1e-6, 1.0, 500)
    between_var = rng.uniform(0.0, 3.0, 500)  # deliberately over-large, to exercise the floor
    k = rng.integers(1, 40, 500)
    var = pooling.design_effect_var(mean_var, between_var, k)
    assert (var >= mean_var / k - 1e-12).all()
    assert (var <= mean_var + 1e-12).all()


def test_width_is_monotone_in_the_seed_spread():
    """The more the seeds disagree, the more independent evidence there is, the tighter the band."""
    spreads = np.linspace(0.0, 0.04, 9)
    widths = pooling.design_effect_var(0.04, spreads, 6)
    assert (np.diff(widths) <= 1e-15).all()


def test_a_missing_sampling_variance_yields_no_interval():
    assert np.isnan(pooling.design_effect_var(np.nan, 0.01, 5))


# =================================================================================================
# attach_pooled_ci — the tidy-frame path (PPR, ASFR)
# =================================================================================================
def _by_seed(values, var, cell="a"):
    return pd.DataFrame(
        {"cell": cell, "seed": range(len(values)), "value": values, "var": var}
    )


def test_the_interval_is_the_pooled_cells_own_sampling_variance():
    """Estimate and width both come from the pooled pass; the seeds contribute neither."""
    by_seed = _by_seed([0.60, 0.66, 0.72], 0.0009)
    # deliberately neither the seed mean (0.66) nor the seed variance
    pooled = pd.DataFrame({"cell": ["a"], "value": [0.70], "var": [0.00012]})
    out = pooling.attach_pooled_ci(
        pooled, by_seed, value="value", var="var", on=["cell"], level=0.95
    )
    assert out["value"].iloc[0] == pytest.approx(0.70)
    assert out["pooled_var"].iloc[0] == pytest.approx(0.00012)
    assert out["ci_hi"].iloc[0] - out["value"].iloc[0] == pytest.approx(Z * np.sqrt(0.00012))


def test_the_seed_spread_is_recorded_without_touching_the_interval():
    """mean_var/between_var/k_seeds ride along so the correction can be applied downstream."""
    by_seed = _by_seed([0.60, 0.66, 0.72], 0.0009)
    pooled = pd.DataFrame({"cell": ["a"], "value": [0.70], "var": [0.00012]})
    out = pooling.attach_pooled_ci(pooled, by_seed, value="value", var="var", on=["cell"])
    assert out["k_seeds"].iloc[0] == 3
    assert out["mean_var"].iloc[0] == pytest.approx(0.0009)
    assert out["between_var"].iloc[0] == pytest.approx(np.var([0.60, 0.66, 0.72], ddof=1))
    # the correction those three feed is available, and is *not* what was reported
    corrected = pooling.design_effect_var(
        out["mean_var"], out["between_var"], out["k_seeds"]
    )
    assert corrected[0] != pytest.approx(out["pooled_var"].iloc[0])


def test_attach_pooled_ci_clips_a_proportion_to_the_unit_interval():
    by_seed = _by_seed([0.98, 0.99, 1.00], 0.05)
    pooled = pd.DataFrame({"cell": ["a"], "value": [0.99], "var": [0.05]})
    out = pooling.attach_pooled_ci(
        pooled, by_seed, value="value", var="var", on=["cell"], clip=(0.0, 1.0)
    )
    assert out["ci_hi"].iloc[0] <= 1.0
    assert out["ci_lo"].iloc[0] >= 0.0


def test_a_pooled_cell_with_no_seed_rows_keeps_its_interval_but_loses_its_diagnostics():
    """The band never depended on the seeds; only the correction inputs go missing."""
    pooled = pd.DataFrame({"cell": ["a", "b"], "value": [0.5, 0.6], "var": [0.01, 0.02]})
    out = pooling.attach_pooled_ci(
        pooled, _by_seed([0.5, 0.5], 0.01), value="value", var="var", on=["cell"]
    )
    assert out["ci_lo"].notna().all()
    assert out.loc[out["cell"] == "b", "mean_var"].isna().all()
    assert out.loc[out["cell"] == "a", "mean_var"].notna().all()


# =================================================================================================
# attach_km_pooled_ci — the step-function path
# =================================================================================================
def _km(times, survival, greenwood, seed=None):
    frame = pd.DataFrame(
        {"time": times, "survival": survival, "greenwood_var": greenwood}
    )
    if seed is not None:
        frame["seed"] = seed
    return frame


def test_km_seeds_are_sampled_onto_the_pooled_grid():
    """Seeds do not share event times, so each curve is read at the pooled curve's times."""
    by_seed = pd.concat(
        [
            _km([10, 30], [0.9, 0.5], [0.01, 0.02], seed=0),
            _km([20, 30], [0.8, 0.4], [0.03, 0.04], seed=1),
        ],
        ignore_index=True,
    )
    pooled = _km([10, 20, 30], [0.9, 0.85, 0.45], [0.001, 0.002, 0.003])
    out = pooling.attach_km_pooled_ci(pooled, by_seed)

    # at t=10 seed 1 has not reached its first event, so it contributes survival 1 / greenwood 0
    assert out["k_seeds"].iloc[0] == 2
    assert out.loc[out["time"] == 10, "mean_var"].iloc[0] == pytest.approx((0.01 + 0.0) / 2)
    assert out.loc[out["time"] == 10, "between_var"].iloc[0] == pytest.approx(
        np.var([0.9, 1.0], ddof=1)
    )


def test_km_keeps_the_product_limits_own_log_log_interval():
    """The traditional Greenwood interval survives untouched; the seed spread only rides along."""
    by_seed = pd.concat(
        [_km([10], [0.9 - 0.05 * s], [0.01], seed=s) for s in range(3)], ignore_index=True
    )
    pooled = _km([10], [0.85], [0.0001]).assign(ci_lo=0.849, ci_hi=0.851)
    out = pooling.attach_km_pooled_ci(pooled, by_seed)
    assert out["ci_lo"].iloc[0] == pytest.approx(0.849)
    assert out["ci_hi"].iloc[0] == pytest.approx(0.851)
    assert out["pooled_var"].iloc[0] == pytest.approx(0.0001)
    assert out["se"].iloc[0] == pytest.approx(np.sqrt(0.0001))
    # the seeds are measured but do not enter the band
    assert out["k_seeds"].iloc[0] == 3
    assert out["mean_var"].iloc[0] == pytest.approx(0.01)


def test_step_sample_holds_each_value_until_the_next_event():
    curve = _km([10, 20], [0.9, 0.5], [0.01, 0.02])
    got = step_sample(curve, np.array([5, 10, 15, 20, 25]))
    np.testing.assert_allclose(got, [1.0, 0.9, 0.9, 0.5, 0.5])
    gw = step_sample(curve, np.array([5, 10]), value="greenwood_var", before=0.0)
    np.testing.assert_allclose(gw, [0.0, 0.01])


# =================================================================================================
# pool_seeds — every trajectory becomes its own person
# =================================================================================================
def _sequences():
    return pd.DataFrame(
        {
            "person_id": [1, 1, 2, 2],
            "seed": [0, 1, 0, 1],
            "age": [100, 200, 300, 400],
            "event": ["birth"] * 4,
        }
    )


def test_pool_seeds_gives_every_trajectory_its_own_identity():
    pooled, _ = pool_seeds(_sequences())
    assert pooled["person_id"].nunique() == 4  # 2 people x 2 seeds
    assert sorted(pooled["source_person_id"].unique()) == [1, 2]
    # the same (person, seed) always maps to the same id
    pairs = pooled.groupby(["source_person_id", "seed"])["person_id"].nunique()
    assert (pairs == 1).all()


def test_pool_seeds_is_deterministic():
    a, _ = pool_seeds(_sequences())
    b, _ = pool_seeds(_sequences().iloc[::-1].reset_index(drop=True))
    key = ["source_person_id", "seed", "person_id"]
    pd.testing.assert_frame_equal(
        a[key].sort_values(key).reset_index(drop=True),
        b[key].sort_values(key).reset_index(drop=True),
    )


def test_pool_seeds_expands_persons_so_the_cohort_merges_still_resolve():
    persons = pd.DataFrame({"person_id": [1, 2], "birth_year": [1960, 1970]})
    pooled, persons_pooled = pool_seeds(_sequences(), persons)
    assert set(persons_pooled["person_id"]) == set(pooled["person_id"])
    # each trajectory keeps its source person's attributes
    merged = pooled.merge(persons_pooled[["person_id", "birth_year"]], on="person_id")
    assert (merged.loc[merged["source_person_id"] == 1, "birth_year"] == 1960).all()
