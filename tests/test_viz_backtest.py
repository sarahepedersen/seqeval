"""Backtest figures (04 viz): band construction, incomplete-cohort marks, timing-error ridges.

These assert the *decisions* the figures encode — which uncertainty a band draws, which cohorts are
drawn as truncated, how a withheld cell appears — not pixel appearance.
"""

from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from seqeval.units import years_to_days as yd  # noqa: E402
from seqeval.viz import backtest as B  # noqa: E402


def _ccf(cohorts, values, complete, seeds=None):
    frame = pd.DataFrame({"cohort": cohorts, "ccf": values, "complete": complete})
    if seeds is not None:
        frame["seed"] = seeds
    return frame


def _pooled_km(times, survival, ci_lo=None, ci_hi=None):
    """A pooled KM frame as `attach_km_pooled_ci` leaves it: the curve and its own interval."""
    frame = pd.DataFrame({"time": [yd(t) for t in times], "survival": survival})
    if ci_lo is not None:
        frame["ci_lo"], frame["ci_hi"] = ci_lo, ci_hi
    return frame


def test_km_overlay_draws_the_interval_its_table_carries():
    """The figure computes no variance — it shades the ci_lo/ci_hi the pooled table hands it."""
    obs = _pooled_km([20, 30], [0.9, 0.5])
    gen = _pooled_km([20, 30], [0.8, 0.4], ci_lo=[0.75, 0.33], ci_hi=[0.85, 0.47])
    ax = B.plot_km_overlay(obs, gen).axes[0]

    (band,) = ax.collections
    ys = band.get_paths()[0].vertices[:, 1]
    assert ys.min() == pytest.approx(0.33)
    assert ys.max() == pytest.approx(0.85)
    labels = [ln.get_label() for ln in ax.lines]
    assert "generated (pooled)" in labels and "observed" in labels


def test_km_overlay_without_an_interval_still_draws_the_curve():
    """A frame with no CI columns loses its band, not its curve."""
    obs = _pooled_km([20, 30], [0.9, 0.5])
    ax = B.plot_km_overlay(obs, _pooled_km([20, 30], [0.8, 0.4])).axes[0]
    assert not ax.collections
    assert "generated (pooled)" in [ln.get_label() for ln in ax.lines]


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


def test_incomplete_cohorts_are_flagged_from_the_majority_of_seeds():
    """A cohort counts as truncated when most seeds say so, not when any one does."""
    gen = _ccf(
        [1950, 1960, 1970] * 2,
        [2.0, 1.9, 1.1, 2.2, 2.1, 1.3],
        [True, True, False] * 2,
        seeds=[0, 0, 0, 1, 1, 1],
    )
    assert list(B.majority_complete(gen)) == [True, True, False]

    # incomplete in only one of three seeds -> the cohort stays complete
    minority = _ccf(
        [1950, 1960] * 3,
        [2.0, 1.9, 2.1, 1.8, 2.2, 1.7],
        [True, False, True, True, True, True],
        seeds=[0, 0, 1, 1, 2, 2],
    )
    assert list(B.majority_complete(minority)) == [True, True]


def test_a_frame_without_a_complete_column_is_treated_as_finished():
    gen = pd.DataFrame({"cohort": [1950, 1960], "ccf": [2.0, 1.9], "seed": [0, 0]})
    _, _, _, complete = B._ccf_band(gen, 0.95)
    assert complete.all()


def test_ccf_band_uses_the_total_variance_when_it_is_supplied():
    """With a variance frame the band is ±z·sqrt(total_var), not the across-seed standard error."""
    gen = _ccf(
        [1950, 1960] * 3, [2.0, 1.9, 2.1, 1.8, 2.2, 1.7], [True] * 6, seeds=[0, 0, 1, 1, 2, 2]
    )
    var = pd.DataFrame({"cohort": [1950, 1960], "total_var": [0.04, 0.09]})
    _, _, half, _ = B._ccf_band(gen, 0.95, var)
    np.testing.assert_allclose(half, 1.959963985 * np.array([0.2, 0.3]))
    # the replicate-only width is the fallback, and is the narrower of the two here
    _, _, mc_half, _ = B._ccf_band(gen, 0.95)
    assert (mc_half < half).all()


def test_ccf_band_falls_back_to_replicate_width_for_cohorts_without_a_variance():
    """A cohort missing from the variance frame keeps its replicate band rather than vanishing."""
    gen = _ccf([1950, 1960] * 2, [2.0, 1.9, 2.2, 1.7], [True] * 4, seeds=[0, 0, 1, 1])
    partial = pd.DataFrame({"cohort": [1950], "total_var": [0.04]})
    _, _, half, _ = B._ccf_band(gen, 0.95, partial)
    _, _, mc_half, _ = B._ccf_band(gen, 0.95)
    assert half[0] == pytest.approx(1.959963985 * 0.2)
    assert half[1] == pytest.approx(mc_half[1])
    assert np.isfinite(half).all()


def _errors(pred_bins, error_edges, counts, n_pred_bin=None, suppressed=None):
    """A timing_error_distribution-shaped frame: one row per (pred_bin, error bin)."""
    rows = []
    for b in pred_bins:
        total = n_pred_bin[b] if n_pred_bin else sum(counts[b])
        for e, (lo, hi) in enumerate(zip(error_edges[:-1], error_edges[1:], strict=True)):
            rows.append(
                {
                    "pred_bin": b,
                    "pred_lo": yd(25 + 5 * b),
                    "pred_hi": yd(30 + 5 * b),
                    "pred_median": yd(27 + 5 * b),
                    "n_pred_bin": total,
                    "error_lo": yd(lo),
                    "error_hi": yd(hi),
                    "n_persons": counts[b][e],
                    "suppressed": bool(suppressed[b][e]) if suppressed else False,
                }
            )
    return pd.DataFrame(rows)


def test_ridge_draws_one_baseline_per_predicted_bin():
    counts = {0: [1, 9, 9, 1], 1: [2, 8, 8, 2], 2: [5, 5, 5, 5]}
    errs = _errors([0, 1, 2], [-2, -1, 0, 1, 2], counts)
    ax = B.plot_timing_ridge(errs).axes[0]
    assert len(ax.get_yticks()) == 3
    # each tick labels its predicted range and the people it rests on
    assert all("n=" in t.get_text() and "y" in t.get_text() for t in ax.get_yticklabels())


def test_ridge_marks_zero_error():
    """The zero line is the reference the whole figure is read against."""
    errs = _errors([0], [-2, -1, 0, 1, 2], {0: [1, 9, 9, 1]})
    ax = B.plot_timing_ridge(errs).axes[0]
    verticals = [ln for ln in ax.lines if len(set(ln.get_xdata())) == 1]
    assert any(ln.get_xdata()[0] == pytest.approx(0.0) for ln in verticals)


def test_ridge_draws_withheld_cells_at_their_upper_bound_not_zero():
    """A suppressed cell is 'at most 4 people', which must not read as an empty stretch."""
    errs = _errors(
        [0], [-2, -1, 0, 1, 2], {0: [0, 9, 9, 0]},
        n_pred_bin={0: 22}, suppressed={0: [False, False, False, True]},
    )
    patches = B.plot_timing_ridge(errs).axes[0].patches
    hatched = [p for p in patches if p.get_hatch()]
    assert len(hatched) == 1
    assert hatched[0].get_height() > 0


def test_ridge_needs_only_the_aggregate_table():
    """The figure is drawable from binned counts alone — nothing per-person reaches viz."""
    errs = _errors([0, 1], [-1, 0, 1], {0: [7, 7], 1: [6, 8]})
    assert "person_id" not in errs.columns
    assert B.plot_timing_ridge(errs).axes[0].get_yticks().size == 2


def test_ridge_heights_are_within_bin_proportions():
    """Rows resting on different numbers of people stay comparable."""
    errs = pd.concat(
        [
            _errors([0], [-1, 0, 1], {0: [10, 10]}, n_pred_bin={0: 20}),
            _errors([1], [-1, 0, 1], {1: [100, 100]}, n_pred_bin={1: 200}),
        ]
    ).reset_index(drop=True)
    ax = B.plot_timing_ridge(errs).axes[0]
    spans = [c.get_paths()[0].get_extents().height for c in ax.collections]
    assert spans[0] == pytest.approx(spans[1])  # same shape, 10x the people




# =================================================================================================
# PPR / ASFR overlays
# =================================================================================================
def _seeded(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _ppr_frames():
    obs = pd.DataFrame({
        "parity_from": [0, 1, 2], "parity_to": [1, 2, 3],
        "ppr": [0.9, 0.6, 0.3], "n_at_risk": [100, 90, 54],
        "ppr_var": [0.0009, 0.0027, 0.0039],
    })
    # the pooled frame as the arm hands it over: one row per transition, interval already attached
    gen = obs.assign(
        ppr=[0.88, 0.62, 0.31],
        ci_lo=[0.85, 0.55, 0.20],
        ci_hi=[0.91, 0.69, 0.42],
    )
    return obs, gen


def test_ppr_overlay_draws_the_interval_its_table_carries():
    """The figure computes no variance — the bars are the ci_lo/ci_hi on the pooled frame."""
    obs, gen = _ppr_frames()
    ax = B.plot_ppr_overlay(obs, gen, level=0.95).axes[0]
    lines = {tuple(np.round(ln.get_ydata(), 6)) for ln in ax.get_lines()}
    assert tuple(np.round(gen["ppr"].to_numpy(), 6)) in lines
    assert tuple(np.round(obs["ppr"].to_numpy(), 6)) in lines

    (bars,) = [c for c in ax.containers if hasattr(c, "errorbar")] or [ax.containers[0]]
    caps = bars.lines[2][0].get_segments()
    lo, hi = zip(*[(s[:, 1].min(), s[:, 1].max()) for s in caps], strict=True)
    np.testing.assert_allclose(sorted(lo), sorted(gen["ci_lo"]), atol=1e-9)
    np.testing.assert_allclose(sorted(hi), sorted(gen["ci_hi"]), atol=1e-9)
    # a thinner denominator at the later transition still reads as a wider interval
    widths = gen["ci_hi"] - gen["ci_lo"]
    assert widths.iloc[2] > widths.iloc[0]


def test_ppr_overlay_labels_transitions_not_bare_parities():
    obs, gen = _ppr_frames()
    fig = B.plot_ppr_overlay(obs, gen)
    labels = [t.get_text() for t in fig.axes[0].get_xticklabels()]
    assert labels == ["0→1", "1→2", "2→3"]


def _asfr_frames(cohorts=(1960, 1965)):
    ages = [20.0, 25.0, 30.0]
    obs = pd.DataFrame([
        {"cohort": c, "age_bin": a, "asfr": 0.1, "person_years": 500.0,
         "births": 50, "asfr_var": 0.0002}
        for c in cohorts for a in ages
    ])
    # the pooled frame: one row per (cohort, age_bin), interval already attached
    gen = obs.assign(ci_lo=obs["asfr"] - 0.01, ci_hi=obs["asfr"] + 0.01)
    return obs, gen


def test_asfr_overlay_shades_the_interval_its_table_carries():
    """Cohort ASFR now carries a band; it is the pooled table's, not one computed here."""
    obs, gen = _asfr_frames(cohorts=(1960,))
    ax = [a for a in B.plot_asfr_overlay(obs, gen).axes if a.get_visible()][0]
    (band,) = ax.collections
    ys = band.get_paths()[0].vertices[:, 1]
    assert ys.min() == pytest.approx(gen["ci_lo"].min())
    assert ys.max() == pytest.approx(gen["ci_hi"].max())


def test_asfr_jumpoff_panel_bands_every_jumpoff():
    """Each jump-off's profile carries its own interval, so the panel is read the same way."""
    obs, gen = _asfr_frames(cohorts=(1960,))
    fig = B.plot_asfr_jumpoff_panel(obs, {yd(25): gen, yd(30): gen})
    bands = [c for ax in fig.axes for c in ax.collections]
    assert len(bands) == 2


def test_asfr_overlay_draws_one_visible_panel_per_cohort():
    obs, gen = _asfr_frames(cohorts=(1960, 1965, 1970, 1975))
    fig = B.plot_asfr_overlay(obs, gen, jumpoff_days=yd(25))
    visible = [ax for ax in fig.axes if ax.get_visible()]
    assert len(visible) == 4
    assert [ax.get_title(loc="left") for ax in visible] == [
        "cohort 1960", "cohort 1965", "cohort 1970", "cohort 1975"
    ]


def test_asfr_overlay_marks_the_jumpoff_age_in_every_panel():
    """The jump-off is an age, so the same rule applies to every cohort's panel."""
    obs, gen = _asfr_frames()
    fig = B.plot_asfr_overlay(obs, gen, jumpoff_days=yd(30))
    for ax in [a for a in fig.axes if a.get_visible()]:
        rules = [ln for ln in ax.get_lines() if ln.get_linestyle() == ":"]
        assert [round(float(ln.get_xdata()[0])) for ln in rules] == [30]


def test_asfr_jumpoff_panel_marks_each_jumpoff_in_its_own_colour():
    obs, gen = _asfr_frames()
    fig = B.plot_asfr_jumpoff_panel(obs, {yd(25): gen, yd(30): gen})
    ax = [a for a in fig.axes if a.get_visible()][0]
    rules = [ln for ln in ax.get_lines() if ln.get_linestyle() == ":"]
    assert sorted(round(float(ln.get_xdata()[0])) for ln in rules) == [25, 30]
    assert len({tuple(ln.get_color()) for ln in rules}) == 2
