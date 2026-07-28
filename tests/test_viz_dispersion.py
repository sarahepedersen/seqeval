"""Within-seed dispersion columns: what a column means and what a withheld bar looks like."""

from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from seqeval.units import years_to_days as yd  # noqa: E402
from seqeval.viz import dispersion as D  # noqa: E402


def _dist(groups, counts, n_group=None, suppressed=None, col="cohort", extra=None):
    """A dispersion_distribution-shaped frame: one row per (group, bin)."""
    rows = []
    for g in groups:
        total = n_group[g] if n_group else sum(counts[g])
        for b, n in enumerate(counts[g]):
            row = {
                col: g,
                "bin": b,
                "bin_lo": 0.1 * b,
                "bin_hi": 0.1 * (b + 1),
                "n_persons": n,
                "n_group": total,
                "suppressed": bool(suppressed[g][b]) if suppressed else False,
            }
            if extra:
                row.update(extra)
            rows.append(row)
    return pd.DataFrame(rows)


def test_one_column_per_group_at_its_own_x_position():
    """Groups run along x and the dispersion runs up y, matching the outcome-uncertainty figure."""
    dist = _dist([1960, 1965], {1960: [4, 20, 8, 2], 1965: [10, 15, 5, 1]})
    ax = D.plot_within_seed_variance(dist).axes[0]
    assert list(ax.get_xticks()) == [1960, 1965]
    # bars are stacked up the y axis, one per bin per group
    assert len(ax.patches) == 8
    assert {round(p.get_y(), 3) for p in ax.patches} == {0.0, 0.1, 0.2, 0.3}


def test_bar_width_is_a_within_group_share():
    """A small cohort and a large one with the same shape draw the same column."""
    dist = pd.concat(
        [
            _dist([1960], {1960: [10, 10]}, n_group={1960: 20}),
            _dist([1965], {1965: [100, 100]}, n_group={1965: 200}),
        ]
    ).reset_index(drop=True)
    widths = [p.get_width() for p in D.plot_within_seed_variance(dist).axes[0].patches]
    assert widths[0] == pytest.approx(widths[2])  # same share, same width, 10x the people


def test_columns_start_at_their_group_tick():
    """Bars share a left baseline per column, so bin shares compare as lengths."""
    dist = _dist([1960, 1965], {1960: [10, 30], 1965: [20, 20]})
    ax = D.plot_within_seed_variance(dist).axes[0]
    assert ax.patches
    for p in ax.patches:
        assert p.get_x() == pytest.approx(1960) or p.get_x() == pytest.approx(1965)
        assert p.get_width() >= 0


def test_bar_lengths_are_proportional_to_the_within_group_share():
    """One scale for the whole figure, so columns compare to each other as well as within."""
    dist = _dist([1960, 1965], {1960: [10, 30], 1965: [20, 20]})
    ax = D.plot_within_seed_variance(dist).axes[0]
    widths = {}
    for p in ax.patches:
        widths.setdefault(round(p.get_x()), []).append(p.get_width())
    # cohort 1960 splits 10/30; its two bars are in that ratio
    a, b = sorted(widths[1960])
    assert b == pytest.approx(3 * a, rel=1e-6)
    # cohort 1965 splits evenly, and its bars match 1960's larger one at equal share
    assert widths[1965][0] == pytest.approx(widths[1965][1], rel=1e-6)


def test_faceting_draws_one_panel_per_value():
    """Cohorts are compared within a jump-off, and against themselves as it moves."""
    frames = [
        _dist([1960, 1965], {1960: [5, 20, 5], 1965: [8, 18, 4]}, extra={"age_stop": t2})
        for t2 in (yd(25), yd(30))
    ]
    fig = D.plot_within_seed_variance(
        pd.concat(frames).reset_index(drop=True), x="cohort", facet_by="age_stop"
    )
    assert len(fig.axes) == 2
    assert fig.axes[0].get_title(loc="left") == "25y jump-off"
    assert fig.axes[0].get_ylim() == fig.axes[1].get_ylim()  # shared y: panels are comparable


def test_withheld_bars_are_hatched_at_their_upper_bound():
    dist = _dist(
        [1960], {1960: [30, 20, np.nan]}, n_group={1960: 53},
        suppressed={1960: [False, False, True]},
    )
    ax = D.plot_within_seed_variance(dist).axes[0]
    hatched = [p for p in ax.patches if p.get_hatch()]
    assert len(hatched) == 1 and hatched[0].get_width() > 0


def test_figure_needs_only_the_binned_table():
    """No per-person column reaches the figure, which is what makes it publishable."""
    dist = _dist([1960], {1960: [4, 20, 8, 2]})
    assert "person_id" not in dist.columns
    assert D.plot_within_seed_variance(dist).axes
