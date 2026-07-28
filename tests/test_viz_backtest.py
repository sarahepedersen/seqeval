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


def test_seed_ci_is_the_standard_error_of_the_across_seed_mean():
    """The band is mean ± z·sd/√K on the population sd — it shrinks as seeds are added."""
    values = np.array([[1.0, 2.0, 3.0, 4.0], [2.0, 2.0, 2.0, 2.0], [3.0, 2.0, 1.0, 0.0]])
    mean, lo, hi = B._seed_ci(values, 0.95)
    np.testing.assert_allclose(mean, [2.0, 2.0, 2.0, 2.0])
    half = 1.959963985 * values.std(axis=0, ddof=0) / np.sqrt(3)
    np.testing.assert_allclose(hi - mean, half)
    np.testing.assert_allclose(mean - lo, half)


def test_seed_ci_matches_the_analytic_aggregate_ccf_standard_error():
    """The plotted band and ``replicate_variance_aggregate.within_var`` are the same quantity.

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


def test_uncertainty_figure_hollows_truncated_cohorts():
    """A cohort whose life course is unfinished must not read as a finished CCF."""
    var = _variance([1960, 1965])
    par = _parity([1960, 1965], [0.1, 0.3, 0.35, 0.2, 0.05])
    complete = pd.Series([True, False], index=[1960, 1965])
    fig = B.plot_ccf_inference_vs_outcome(var, par, complete=complete)
    for ax in fig.axes[:2]:  # both panels carry the same marking
        _, labels = ax.get_legend_handles_labels()
        assert any("incomplete" in lb for lb in labels)


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


def _parity(cohorts, shares, n_women_total=500, suppressed=None):
    """A parity_distribution-shaped frame: one row per (cohort, parity)."""
    rows = []
    for c in cohorts:
        for k, share in enumerate(shares):
            rows.append(
                {
                    "cohort": c,
                    "parity": k,
                    "n_replicates": share * n_women_total,
                    "n_women_equiv": share * n_women_total,
                    "share": share,
                    "n_women_total": n_women_total,
                    "suppressed": bool(suppressed[k]) if suppressed else False,
                }
            )
    return pd.DataFrame(rows)


def _variance(cohorts, ccf=2.0, total_var=0.0004):
    return pd.DataFrame(
        {"cohort": cohorts, "n_women": 500, "ccf": ccf, "within_var": total_var / 2,
         "between_var": total_var / 2, "total_var": total_var}
    )


def test_inference_panel_is_a_magnification_of_the_outcome_panel():
    """The left panel zooms the band the right panel shades — same units, different range."""
    fig = B.plot_ccf_inference_vs_outcome(
        _variance([1960, 1965]), _parity([1960, 1965], [0.1, 0.3, 0.35, 0.2, 0.05])
    )
    inf_lo, inf_hi = fig.axes[0].get_ylim()
    out_lo, out_hi = fig.axes[1].get_ylim()
    assert out_lo < inf_lo and inf_hi < out_hi
    assert (inf_hi - inf_lo) < (out_hi - out_lo) / 5  # the point is that it is far narrower


def test_the_interval_is_never_rescaled_to_be_visible():
    """Both panels draw the true ±z·sqrt(total_var); its invisibility on the right is the point."""
    total_var = 0.0004
    fig = B.plot_ccf_inference_vs_outcome(
        _variance([1960], total_var=total_var), _parity([1960], [0.1, 0.3, 0.35, 0.2, 0.05])
    )
    expected = 1.959963985 * np.sqrt(total_var)
    for ax in fig.axes[:2]:
        bars = [c for c in ax.containers if hasattr(c, "errorbar") or hasattr(c, "lines")]
        segs = [s for c in bars for s in (c.lines[2] if getattr(c, "lines", None) else [])]
        spans = [seg.get_segments()[0] for seg in segs if seg.get_segments()]
        assert spans, "no error bar drawn"
        lo, hi = spans[0][0][1], spans[0][1][1]
        assert (hi - lo) / 2 == pytest.approx(expected, rel=1e-9)


def test_parity_bars_sit_at_integer_parities_with_a_top_coded_last_tick():
    fig = B.plot_ccf_inference_vs_outcome(
        _variance([1960]), _parity([1960], [0.1, 0.3, 0.35, 0.2, 0.05])
    )
    ax = fig.axes[1]
    assert list(ax.get_yticks()) == [0, 1, 2, 3, 4]
    assert [t.get_text() for t in ax.get_yticklabels()][-1] == "4+"


def test_bar_width_tracks_the_share():
    fig = B.plot_ccf_inference_vs_outcome(
        _variance([1960]), _parity([1960], [0.1, 0.2, 0.4, 0.2, 0.1])
    )
    widths = [p.get_width() for p in fig.axes[1].patches]
    assert widths[2] == pytest.approx(2 * widths[1])  # twice the share, twice the bar


def test_withheld_parities_are_hatched_rather_than_absent():
    fig = B.plot_ccf_inference_vs_outcome(
        _variance([1960]),
        _parity([1960], [0.1, 0.3, 0.35, 0.2, np.nan], suppressed=[0, 0, 0, 0, 1]),
    )
    hatched = [p for p in fig.axes[1].patches if p.get_hatch()]
    assert len(hatched) == 1 and hatched[0].get_width() > 0


def test_uncertainty_figure_needs_only_aggregate_frames():
    """Neither layer touches a per-person row, which is what makes the figure publishable."""
    var, par = _variance([1960, 1965]), _parity([1960, 1965], [0.2, 0.3, 0.3, 0.15, 0.05])
    assert "person_id" not in var.columns and "person_id" not in par.columns
    assert len(B.plot_ccf_inference_vs_outcome(var, par).axes) >= 2
