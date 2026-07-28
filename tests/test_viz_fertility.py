"""Fertility figures (03 viz): stratum identification, and the two-panel uncertainty contrast."""

from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from seqeval.viz import fertility as F  # noqa: E402
from seqeval.viz.fertility import plot_asfr  # noqa: E402


def _asfr(cohorts) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"cohort": c, "age_bin": a, "asfr": 0.2 * np.exp(-((a - 28) ** 2) / 40)}
            for c in cohorts
            for a in range(15, 50)
        ]
    )


def test_few_cohorts_are_keyed_by_legend():
    ax = plot_asfr(_asfr(range(1940, 1948)), dim="cohort").axes[0]
    legend = ax.get_legend()
    assert legend is not None
    assert [t.get_text() for t in legend.get_texts()] == [str(c) for c in range(1940, 1948)]


def test_many_cohorts_are_keyed_by_colorbar():
    """Past the legend limit the key becomes a colorbar — never nothing at all."""
    cohorts = list(range(1940, 1970))
    fig = plot_asfr(_asfr(cohorts), dim="cohort")
    ax = fig.axes[0]
    assert ax.get_legend() is None
    assert len(fig.axes) == 2  # the colorbar is the second axes
    labels = [t.get_text() for t in fig.axes[1].get_yticklabels()]
    assert set(labels) <= {str(c) for c in cohorts}
    assert labels[0] == "1940"  # ticks are labelled with cohort values, not ranks


def test_colorbar_ticks_track_the_line_colours():
    """Colour is assigned by rank, so tick n must carry the colour of the nth curve."""
    cohorts = list(range(1940, 1970))
    fig = plot_asfr(_asfr(cohorts), dim="cohort")
    ramp = fig.axes[1].collections[0].cmap
    for line, c in zip(fig.axes[0].lines, np.linspace(0, 1, len(cohorts)), strict=True):
        np.testing.assert_allclose(line.get_color(), ramp(c), atol=0.01)


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

def test_uncertainty_figure_hollows_truncated_cohorts():
    """A cohort whose life course is unfinished must not read as a finished CCF."""
    var = _variance([1960, 1965])
    par = _parity([1960, 1965], [0.1, 0.3, 0.35, 0.2, 0.05])
    complete = pd.Series([True, False], index=[1960, 1965])
    fig = F.plot_ccf_inference_vs_outcome(var, par, complete=complete)
    for ax in fig.axes[:2]:  # both panels carry the same marking
        _, labels = ax.get_legend_handles_labels()
        assert any("incomplete" in lb for lb in labels)

def test_inference_panel_is_a_magnification_of_the_outcome_panel():
    """The left panel zooms the band the right panel shades — same units, different range."""
    fig = F.plot_ccf_inference_vs_outcome(
        _variance([1960, 1965]), _parity([1960, 1965], [0.1, 0.3, 0.35, 0.2, 0.05])
    )
    inf_lo, inf_hi = fig.axes[0].get_ylim()
    out_lo, out_hi = fig.axes[1].get_ylim()
    assert out_lo < inf_lo and inf_hi < out_hi
    assert (inf_hi - inf_lo) < (out_hi - out_lo) / 5  # the point is that it is far narrower

def test_the_interval_is_never_rescaled_to_be_visible():
    """Both panels draw the true ±z·sqrt(total_var); its invisibility on the right is the point."""
    total_var = 0.0004
    fig = F.plot_ccf_inference_vs_outcome(
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
    fig = F.plot_ccf_inference_vs_outcome(
        _variance([1960]), _parity([1960], [0.1, 0.3, 0.35, 0.2, 0.05])
    )
    ax = fig.axes[1]
    assert list(ax.get_yticks()) == [0, 1, 2, 3, 4]
    assert [t.get_text() for t in ax.get_yticklabels()][-1] == "4+"

def test_bar_width_tracks_the_share():
    fig = F.plot_ccf_inference_vs_outcome(
        _variance([1960]), _parity([1960], [0.1, 0.2, 0.4, 0.2, 0.1])
    )
    widths = [p.get_width() for p in fig.axes[1].patches]
    assert widths[2] == pytest.approx(2 * widths[1])  # twice the share, twice the bar

def test_withheld_parities_are_hatched_rather_than_absent():
    fig = F.plot_ccf_inference_vs_outcome(
        _variance([1960]),
        _parity([1960], [0.1, 0.3, 0.35, 0.2, np.nan], suppressed=[0, 0, 0, 0, 1]),
    )
    hatched = [p for p in fig.axes[1].patches if p.get_hatch()]
    assert len(hatched) == 1 and hatched[0].get_width() > 0

def test_uncertainty_figure_needs_only_aggregate_frames():
    """Neither layer touches a per-person row, which is what makes the figure publishable."""
    var, par = _variance([1960, 1965]), _parity([1960, 1965], [0.2, 0.3, 0.3, 0.15, 0.05])
    assert "person_id" not in var.columns and "person_id" not in par.columns
    assert len(F.plot_ccf_inference_vs_outcome(var, par).axes) >= 2
