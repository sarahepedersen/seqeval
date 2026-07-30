"""Small-cell suppression: what is withheld, what survives, and what stays unrecoverable."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from seqeval.metrics._disclosure import (
    MIN_CELL,
    POLICIES,
    DisclosureError,
    apply_policy,
    assert_publishable,
    policy_for,
    suppress_small_cells,
)


def _cells(counts, group="a"):
    return pd.DataFrame({"g": [group] * len(counts), "bin": range(len(counts)), "n": counts})


def test_small_cells_are_nulled_not_dropped():
    out = suppress_small_cells(_cells([40, 3, 30, 20]), count_cols="n", by=["g"])
    assert len(out) == 4  # the row and its bin survive; only the count goes
    assert out.loc[1, "suppressed"] and pd.isna(out.loc[1, "n"])


def test_true_zeros_are_published():
    """A zero names nobody, and a hole where a density is genuinely empty would misread."""
    out = suppress_small_cells(_cells([40, 0, 30, 20]), count_cols="n", by=["g"])
    assert not out["suppressed"].any()
    assert out.loc[1, "n"] == 0


def test_a_lone_suppressed_cell_forces_a_second():
    """One withheld cell is just the group total minus the published ones."""
    out = suppress_small_cells(_cells([40, 3, 30, 20]), count_cols="n", by=["g"])
    assert int(out["suppressed"].sum()) == 2
    assert out.loc[3, "suppressed"]  # the smallest survivor goes with it
    # what remains published cannot pin either withheld count
    assert out["n"].sum() < 40 + 3 + 30 + 20


def test_several_small_cells_need_no_extra_suppression():
    out = suppress_small_cells(_cells([40, 3, 2, 30]), count_cols="n", by=["g"])
    assert list(out["suppressed"]) == [False, True, True, False]


def test_groups_are_suppressed_independently():
    frame = pd.concat([_cells([40, 3, 30, 20], "a"), _cells([50, 60, 70, 80], "b")])
    out = suppress_small_cells(frame.reset_index(drop=True), count_cols="n", by=["g"])
    assert int(out[out["g"] == "a"]["suppressed"].sum()) == 2
    assert not out[out["g"] == "b"]["suppressed"].any()


def test_companion_columns_are_nulled_with_the_count():
    frame = _cells([40, 3, 30, 20]).assign(share=[0.4, 0.03, 0.3, 0.2])
    out = suppress_small_cells(frame, count_cols="n", by=["g"], also_null=("share",))
    assert pd.isna(out.loc[1, "share"]) and pd.isna(out.loc[3, "share"])
    assert out.loc[0, "share"] == 0.4


def test_the_threshold_is_inclusive():
    """``min_cell`` itself is withheld — the policy is "this many or fewer", not "fewer than"."""
    counts = [40, MIN_CELL + 1, MIN_CELL, MIN_CELL - 1, 30]
    out = suppress_small_cells(_cells(counts), count_cols="n", by=["g"], complement=False)
    assert list(out["suppressed"]) == [False, False, True, True, False]


def test_min_cell_zero_publishes_everything():
    out = suppress_small_cells(_cells([40, 1, 30]), count_cols="n", by=["g"], min_cell=0)
    assert not out["suppressed"].any()


def test_the_complement_can_be_turned_off():
    """Rows that partition nothing a reader can see need no complementary cell."""
    out = suppress_small_cells(_cells([40, 3, 30, 20]), count_cols="n", by=["g"], complement=False)
    assert list(out["suppressed"]) == [False, True, False, False]


# -------------------------------------------------------------------------------------------------
# any count, not just the denominator
# -------------------------------------------------------------------------------------------------
def test_a_thin_event_count_withholds_a_fat_person_count():
    """The whole point of the audit: 400 women holding one birth is a cell about one birth."""
    frame = pd.DataFrame({"g": ["a", "a"], "n_persons": [400, 380], "n_events": [1, 90]})
    out = suppress_small_cells(
        frame, count_cols=("n_persons", "n_events"), by=["g"], complement=False
    )
    assert list(out["suppressed"]) == [True, False]
    assert pd.isna(out.loc[0, "n_persons"]) and pd.isna(out.loc[0, "n_events"])
    assert out.loc[1, "n_persons"] == 380


def test_absent_columns_are_ignored():
    """One declaration covers schema variants — a by-seed table and its pooled twin."""
    frame = pd.DataFrame({"n_events": [40, 2]})
    out = suppress_small_cells(
        frame,
        count_cols=("n_events", "n_source_persons"),
        by=["nope"],
        also_null=("gone",),
        complement=False,
    )
    assert list(out["suppressed"]) == [False, True]


def test_suppression_is_idempotent_and_never_unsuppresses():
    once = suppress_small_cells(_cells([40, 3, 30, 20]), count_cols="n", by=["g"])
    twice = suppress_small_cells(once, count_cols="n", by=["g"])
    assert list(once["suppressed"]) == list(twice["suppressed"])
    assert twice["n"].isna().sum() == once["n"].isna().sum()


def test_the_suppressed_flag_stays_a_plain_bool():
    """Masked counts are NA, and ``NA <= min_cell`` is NA — the flag must not become nullable."""
    out = suppress_small_cells(_cells([40, 3, 30, 20]), count_cols="n", by=["g"])
    again = suppress_small_cells(out, count_cols="n", by=["g"])
    assert again["suppressed"].dtype == bool


# -------------------------------------------------------------------------------------------------
# the per-table registry
# -------------------------------------------------------------------------------------------------
def test_parameterised_stems_resolve_to_a_policy():
    assert policy_for("km_first_birth") is POLICIES["km_by_seed"]
    assert policy_for("km_pooled") is POLICIES["km_pooled"]  # exact wins over the `km_` prefix
    assert policy_for("within_seed_variance_distribution_by_cohort") is not None
    assert policy_for("within_seed_quantile_summary_by_cohort") is not None
    assert policy_for("violations") is None  # per-person, governed by `individual_level`


def test_every_policy_nulls_everything_it_inspects():
    for name, policy in POLICIES.items():
        assert policy.trip, name
        assert not set(policy.trip) & set(policy.also_null), name


def test_pooled_tables_are_judged_on_real_people_not_trajectories():
    """Seeds multiply trajectories, not persons, and cannot manufacture privacy."""
    frame = pd.DataFrame(
        {
            "time": [10, 20],
            "n_events": [90, 80],
            "n_at_risk": [250, 240],
            "n_units": [250, 240],
            "n_source_persons": [2, 50],
            "survival": [0.9, 0.8],
            "greenwood_var": [0.001, 0.002],
        }
    )
    out = apply_policy("km_pooled", frame, min_cell=3)
    assert list(out["suppressed"]) == [True, False]
    assert pd.isna(out.loc[0, "n_units"])  # nulled, though 250 would have passed on its own
    assert out.loc[0, "survival"] == 0.9  # the estimate survives


def test_km_suppression_closes_the_greenwood_inversion():
    """``greenwood_var`` and the survival ratio jointly solve for ``n_events`` and ``n_at_risk``."""
    frame = pd.DataFrame(
        {
            "time": [10, 20],
            "n_at_risk": [80, 79],
            "n_events": [1, 40],
            "survival": [0.9875, 0.4875],
            "greenwood_var": [0.00015, 0.003],
            "ci_lo": [0.95, 0.40],
            "ci_hi": [0.99, 0.55],
            "n_persons": [80, 80],
        }
    )
    out = apply_policy("km_first_birth", frame, min_cell=3)
    assert out.loc[0, "suppressed"]
    # The interval goes with the variance: `se = ln(log(ci_lo)/log(S))/z` returns the same Greenwood
    # sum, so keeping the CI would have withheld nothing.
    for col in ("n_at_risk", "n_events", "greenwood_var", "ci_lo", "ci_hi"):
        assert pd.isna(out.loc[0, col]), col
    # the curve still keeps its shape
    assert out.loc[0, "survival"] == 0.9875
    assert out.loc[1, "ci_lo"] == 0.40  # an untouched row keeps its interval


@pytest.mark.parametrize(
    ("name", "counts", "estimate", "variance"),
    [
        (
            "asfr_cohort",
            {"births": 2, "person_years": 81.9},
            ("asfr", 0.0244),
            ("asfr_var", 2.98e-4),
        ),
        ("ppr", {"n_at_risk": 3, "n_progressed": 2}, ("ppr", 0.667), ("ppr_var", 0.074)),
        (
            "lexis_cohort_observed",
            {"n_events": 1, "person_years": 50.0},
            ("rate", 0.02),
            ("rate_var", 4e-4),
        ),
        (
            "violation_rates",
            {"n_violations": 1, "n_events": 200},
            ("severity", np.nan),
            ("rate_per_event", 0.005),
        ),
    ],
)
def test_the_variance_that_inverts_to_the_count_falls_with_it(name, counts, estimate, variance):
    """Publishing ``asfr`` and ``asfr_var`` publishes ``births = asfr**2 / asfr_var`` exactly."""
    frame = pd.DataFrame({**{k: [v] for k, v in counts.items()},
                          estimate[0]: [estimate[1]], variance[0]: [variance[1]]})
    out = apply_policy(name, frame, min_cell=3)
    assert out.loc[0, "suppressed"]
    assert pd.isna(out.loc[0, variance[0]]), variance[0]
    for col in counts:
        assert pd.isna(out.loc[0, col]), col


def test_backtest_coverage_counts_go_but_the_score_stays():
    coverage = pd.DataFrame(
        {
            "n_condition": [120, 2],
            "n_evaluable": [100, 2],
            "n_settled": [10, 0],
            "n_uncovered": [3, 0],
            "n_seed_median": [12, 12],
            "n_persons": [100, 2],
        }
    )
    out = apply_policy("coverage", coverage, min_cell=3)
    # row 0 trips on `n_uncovered` alone, and takes the rest of its counts with it
    assert list(out["suppressed"]) == [True, True]
    assert pd.isna(out.loc[0, "n_condition"])
    assert out.loc[0, "n_seed_median"] == 12  # a replicate depth, not a head count

    scores = pd.DataFrame(
        {"metric": ["ece"], "value": [0.07], "ci_lo": [0.05], "ci_hi": [0.09], "n_persons": [2]}
    )
    scored = apply_policy("scores", scores, min_cell=3)
    assert pd.isna(scored.loc[0, "n_persons"])
    assert (scored.loc[0, "value"], scored.loc[0, "ci_lo"]) == (0.07, 0.05)


def test_apply_policy_is_a_no_op_without_one():
    frame = pd.DataFrame({"person_id": [1, 2], "age": [10, 20]})
    assert apply_policy("violations", frame, min_cell=3) is frame


# -------------------------------------------------------------------------------------------------
# the backstop
# -------------------------------------------------------------------------------------------------
def test_assert_publishable_accepts_a_suppressed_frame():
    frame = pd.DataFrame({"n_at_risk": [80, 2], "n_progressed": [40, 1], "ppr": [0.5, 0.5]})
    assert_publishable("ppr", apply_policy("ppr", frame, min_cell=3), min_cell=3)


def test_assert_publishable_catches_a_thin_cell():
    frame = pd.DataFrame({"n_at_risk": [80, 2], "n_progressed": [40, 1], "ppr": [0.5, 0.5]})
    with pytest.raises(DisclosureError, match="ppr.n_at_risk"):
        assert_publishable("ppr", frame, min_cell=3)


def test_assert_publishable_allows_zeros():
    frame = pd.DataFrame({"n_at_risk": [80, 0], "n_progressed": [40, 0], "ppr": [0.5, np.nan]})
    assert_publishable("ppr", frame, min_cell=3)
