"""Small-cell suppression: what is withheld, what survives, and what stays unrecoverable."""

from __future__ import annotations

import pandas as pd

from seqeval.metrics._disclosure import MIN_CELL, suppress_small_cells


def _cells(counts, group="a"):
    return pd.DataFrame({"g": [group] * len(counts), "bin": range(len(counts)), "n": counts})


def test_small_cells_are_nulled_not_dropped():
    out = suppress_small_cells(_cells([40, 3, 30, 20]), count_col="n", by=["g"])
    assert len(out) == 4  # the row and its bin survive; only the count goes
    assert out.loc[1, "suppressed"] and pd.isna(out.loc[1, "n"])


def test_true_zeros_are_published():
    """A zero names nobody, and a hole where a density is genuinely empty would misread."""
    out = suppress_small_cells(_cells([40, 0, 30, 20]), count_col="n", by=["g"])
    assert not out["suppressed"].any()
    assert out.loc[1, "n"] == 0


def test_a_lone_suppressed_cell_forces_a_second():
    """One withheld cell is just the group total minus the published ones."""
    out = suppress_small_cells(_cells([40, 3, 30, 20]), count_col="n", by=["g"])
    assert int(out["suppressed"].sum()) == 2
    assert out.loc[3, "suppressed"]  # the smallest survivor goes with it
    # what remains published cannot pin either withheld count
    assert out["n"].sum() < 40 + 3 + 30 + 20


def test_several_small_cells_need_no_extra_suppression():
    out = suppress_small_cells(_cells([40, 3, 2, 30]), count_col="n", by=["g"])
    assert list(out["suppressed"]) == [False, True, True, False]


def test_groups_are_suppressed_independently():
    frame = pd.concat([_cells([40, 3, 30, 20], "a"), _cells([50, 60, 70, 80], "b")])
    out = suppress_small_cells(frame.reset_index(drop=True), count_col="n", by=["g"])
    assert int(out[out["g"] == "a"]["suppressed"].sum()) == 2
    assert not out[out["g"] == "b"]["suppressed"].any()


def test_companion_columns_are_nulled_with_the_count():
    frame = _cells([40, 3, 30, 20]).assign(share=[0.4, 0.03, 0.3, 0.2])
    out = suppress_small_cells(frame, count_col="n", by=["g"], also_null=("share",))
    assert pd.isna(out.loc[1, "share"]) and pd.isna(out.loc[3, "share"])
    assert out.loc[0, "share"] == 0.4


def test_threshold_is_the_documented_minimum():
    """Exactly ``MIN_CELL`` is publishable; below it is not.

    Two cells fall below the threshold, so the lone-cell rule stays out of the way and the
    ``MIN_CELL`` cell is judged on the threshold alone.
    """
    counts = [40, MIN_CELL, MIN_CELL - 1, MIN_CELL - 2, 30]
    out = suppress_small_cells(_cells(counts), count_col="n", by=["g"])
    assert list(out["suppressed"]) == [False, False, True, True, False]
