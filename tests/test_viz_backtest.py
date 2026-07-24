"""Backtest figures (04 viz): timing display windows, outlier accounting, incomplete-cohort marks.

These assert the *decisions* the figures encode — which persons count as off-axis, which cohorts
are drawn as truncated — not pixel appearance.
"""

from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from seqeval.units import years_to_days as yd  # noqa: E402
from seqeval.viz import backtest as B  # noqa: E402


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


def test_timing_pairs_keeps_observed_inside_horizon():
    td, obs_tte = _timing_frames([20, 30, 45], [22, 33, 40], observed=[True, True, False])
    pairs = B.timing_pairs(td, obs_tte, horizon_days=yd(50))
    assert list(pairs["person_id"]) == [0, 1]  # censored person dropped; both inside the horizon
    assert np.allclose(pairs["pred"], [20, 30], atol=0.01)

    # a person whose observed wait exceeds the horizon is out of scope for the comparison
    td, obs_tte = _timing_frames([20, 30], [22, 60])
    assert list(B.timing_pairs(td, obs_tte, horizon_days=yd(50))["person_id"]) == [0]


def test_timing_pairs_drops_persons_projected_past_the_frame():
    """A predicted median sitting on the horizon is the cap, not a date — not timing signal."""
    td, obs_tte = _timing_frames([20, 50], [22, 44])
    assert list(B.timing_pairs(td, obs_tte, horizon_days=yd(50))["person_id"]) == [0]
    kept = B.timing_pairs(td, obs_tte, horizon_days=yd(50), drop_projected_beyond=False)
    assert list(kept["person_id"]) == [0, 1]


def test_timing_pairs_restricts_to_the_scored_population():
    """The arm passes its condition-minus-settled set, matching the reliability panel."""
    td, obs_tte = _timing_frames([20, 30, 35], [22, 33, 36])
    pairs = B.timing_pairs(td, obs_tte, horizon_days=yd(50), persons={0, 2})
    assert list(pairs["person_id"]) == [0, 2]


def test_axes_span_jumpoff_to_frame_close():
    """Default axes run from the jump-off (prediction starts) to the horizon (frame closes)."""
    td, obs_tte = _timing_frames([28, 33], [30, 36])
    ax = B.plot_timing_calibration(td, obs_tte, horizon_days=yd(40), floor_days=yd(25)).axes[0]
    assert ax.get_xlim() == pytest.approx((25.0, 40.0), abs=0.01)
    assert ax.get_ylim() == pytest.approx((25.0, 40.0), abs=0.01)
    # both in-scope persons are plotted; nothing lands off the reachable box
    assert sum(c.get_offsets().shape[0] for c in ax.collections) == 2


def test_origin_based_outcome_keeps_a_zero_floor():
    td, obs_tte = _timing_frames([1, 3], [2, 4])
    ax = B.plot_timing_calibration(td, obs_tte, horizon_days=yd(5)).axes[0]
    assert ax.get_xlim() == pytest.approx((0.0, 5.0), abs=0.01)


def test_waiting_time_scatter_is_square():
    td, obs_tte = _timing_frames([28, 33], [30, 36])
    ax = B.plot_timing_calibration(td, obs_tte, horizon_days=yd(40), floor_days=yd(25)).axes[0]
    assert ax.get_aspect() == 1.0  # 1:1 aspect so y = x is a true 45° line


def _ccf(cohorts, values, complete, seeds=None):
    frame = pd.DataFrame({"cohort": cohorts, "ccf": values, "complete": complete})
    if seeds is not None:
        frame["seed"] = seeds
    return frame


def test_ccf_overlay_marks_incomplete_cohorts_on_both_curves():
    obs = _ccf([1950, 1960, 1970], [2.1, 2.0, 1.2], [True, True, False])
    # two seeds: 1970 is incomplete in both, so the majority rule marks it incomplete
    gen = _ccf(
        [1950, 1960, 1970] * 2,
        [2.0, 1.9, 1.1, 2.2, 2.1, 1.3],
        [True, True, False] * 2,
        seeds=[0, 0, 0, 1, 1, 1],
    )
    labels = [ln.get_label() for ln in B.plot_ccf_seed_band(obs, gen).axes[0].lines]
    assert "observed (incomplete cohorts)" in labels
    assert "generated median (incomplete cohorts)" in labels


def test_ccf_overlay_generated_incompleteness_is_by_majority_of_seeds():
    obs = _ccf([1950, 1960], [2.1, 2.0], [True, True])
    # 1960 incomplete in only one of three seeds -> the generated curve stays solid
    gen = _ccf(
        [1950, 1960] * 3,
        [2.0, 1.9, 2.1, 1.8, 2.2, 1.7],
        [True, False, True, True, True, True],
        seeds=[0, 0, 1, 1, 2, 2],
    )
    labels = [ln.get_label() for ln in B.plot_ccf_seed_band(obs, gen).axes[0].lines]
    assert "generated median (incomplete cohorts)" not in labels


def test_ccf_overlay_without_complete_column_draws_one_solid_curve():
    obs = pd.DataFrame({"cohort": [1950, 1960], "ccf": [2.1, 2.0]})
    gen = pd.DataFrame({"cohort": [1950, 1960], "ccf": [2.0, 1.9], "seed": [0, 0]})
    labels = [ln.get_label() for ln in B.plot_ccf_seed_band(obs, gen).axes[0].lines]
    assert not any("incomplete" in str(lb) for lb in labels)
