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


def test_seed_ci_is_the_standard_error_of_the_across_seed_mean():
    """The band is mean ± z·sd/√K on the population sd — it shrinks as seeds are added."""
    values = np.array([[1.0, 2.0, 3.0, 4.0], [2.0, 2.0, 2.0, 2.0], [3.0, 2.0, 1.0, 0.0]])
    mean, lo, hi = B._seed_ci(values, 0.95)
    np.testing.assert_allclose(mean, [2.0, 2.0, 2.0, 2.0])
    half = 1.959963985 * values.std(axis=0, ddof=0) / np.sqrt(3)
    np.testing.assert_allclose(hi - mean, half)
    np.testing.assert_allclose(mean - lo, half)


def test_seed_ci_matches_the_analytic_aggregate_ccf_standard_error():
    """The plotted band and ``replicate_variance_aggregate.se`` estimate the same quantity.

    The table computes ``sqrt(Σ_i s²_i/K)/n`` from per-person moments; the figure computes
    ``sd/√K`` from the K per-seed CCFs. They agree because persons are independent within a seed,
    which is what makes the two panels' uncertainty comparable at all.
    """
    rng = np.random.default_rng(11)
    n_persons, k_seeds = 500, 20
    counts = rng.poisson(2.1, size=(n_persons, k_seeds)).astype(float)  # person x seed

    ccf_per_seed = counts.mean(axis=0)  # what the figure sees
    _, lo, hi = B._seed_ci(ccf_per_seed, 0.95)
    figure_se = (hi - lo) / 2 / 1.959963985

    s2 = counts.var(axis=1, ddof=0)  # what the table sees: per-person moments
    table_se = np.sqrt(np.sum(s2 / k_seeds)) / n_persons

    assert abs(figure_se / table_se - 1) < 0.15


def test_ccf_jumpoff_panel_draws_one_labelled_curve_per_window():
    obs = _ccf([1950, 1960], [2.1, 2.0], [True, True])
    gen = {
        yd(25): _ccf([1950, 1960] * 2, [2.0, 1.9, 2.2, 2.1], [True] * 4, seeds=[0, 0, 1, 1]),
        yd(30): _ccf([1950, 1960] * 2, [2.05, 1.95, 2.15, 2.05], [True] * 4, seeds=[0, 0, 1, 1]),
    }
    labels = [ln.get_label() for ln in B.plot_ccf_jumpoff_panel(obs, gen).axes[0].lines]
    assert labels == ["jump-off 25y", "jump-off 30y", "observed"]  # ordered by jump-off


def test_jumpoff_panels_colour_windows_in_jumpoff_order():
    """Colour encodes the jump-off's rank, so it must not depend on insertion order."""
    obs = _ccf([1950, 1960], [2.1, 2.0], [True, True])

    def one(t2, v):
        return _ccf([1950, 1960] * 2, [v, v, v, v], [True] * 4, seeds=[0, 0, 1, 1])

    forward = {yd(25): one(25, 2.0), yd(30): one(30, 2.1)}
    reversed_ = {yd(30): one(30, 2.1), yd(25): one(25, 2.0)}
    colors = [
        [ln.get_color() for ln in B.plot_ccf_jumpoff_panel(obs, g).axes[0].lines]
        for g in (forward, reversed_)
    ]
    assert colors[0] == colors[1]


def test_km_jumpoff_panel_marks_each_windows_jumpoff():
    obs_km = pd.DataFrame({"time": [yd(20), yd(30)], "survival": [0.9, 0.5]})
    gen = {
        yd(25): pd.DataFrame(
            {"time": [yd(26), yd(30)] * 2, "survival": [0.8, 0.4, 0.85, 0.45], "seed": [0, 0, 1, 1]}
        ),
        yd(30): pd.DataFrame(
            {"time": [yd(31), yd(35)] * 2, "survival": [0.7, 0.3, 0.75, 0.35], "seed": [0, 0, 1, 1]}
        ),
    }
    ax = B.plot_km_jumpoff_panel(obs_km, gen).axes[0]
    # one vertical rule per window, at the jump-off age in years
    rules = sorted(ln.get_xdata()[0] for ln in ax.lines if len(set(ln.get_xdata())) == 1)
    assert rules == pytest.approx([25.0, 30.0], abs=0.02)
    assert "observed" in [ln.get_label() for ln in ax.lines]


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
    assert "generated mean (incomplete cohorts)" in labels


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
    assert "generated mean (incomplete cohorts)" not in labels


def test_ccf_overlay_without_complete_column_draws_one_solid_curve():
    obs = pd.DataFrame({"cohort": [1950, 1960], "ccf": [2.1, 2.0]})
    gen = pd.DataFrame({"cohort": [1950, 1960], "ccf": [2.0, 1.9], "seed": [0, 0]})
    labels = [ln.get_label() for ln in B.plot_ccf_seed_band(obs, gen).axes[0].lines]
    assert not any("incomplete" in str(lb) for lb in labels)
