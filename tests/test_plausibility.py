"""Illegal-move rules engine (05): per-rule violations and rates on constructed frames."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from seqeval.core.specs import Rule
from seqeval.metrics import plausibility as P
from seqeval.units import years_to_days as yd

KEYS = ["person_id"]


def _frame(rows):
    """rows: list of (person_id, event, age_years)."""
    return pd.DataFrame(
        {
            "person_id": [r[0] for r in rows],
            "event": pd.Categorical([r[1] for r in rows]),
            "age": np.array([yd(r[2]) for r in rows], dtype=np.int32),
        }
    )


def test_min_and_max_age():
    df = _frame([(1, "birth", 11), (1, "birth", 30), (2, "birth", 52)])
    v = P.check_rules(df, KEYS, [Rule("young", "birth", min_age=yd(12))])
    assert set(v["age"]) == {yd(11)}
    v = P.check_rules(df, KEYS, [Rule("old", "birth", max_age=yd(50))])
    assert set(v["age"]) == {yd(52)}


def test_clean_data_passes():
    df = _frame([(1, "birth", 25), (1, "birth", 30)])
    v = P.check_rules(df, KEYS, [Rule("old", "birth", max_age=yd(50), min_age=yd(15))])
    assert v.empty


def test_min_spacing_exact_days():
    # births 25.0 and 25.0+150 days apart; min_spacing 0.5y = 183 days -> flagged
    df = pd.DataFrame(
        {
            "person_id": [1, 1],
            "event": pd.Categorical(["birth", "birth"]),
            "age": np.array([yd(25), yd(25) + 150], dtype=np.int32),
        }
    )
    v = P.check_rules(df, KEYS, [Rule("space", "birth", min_spacing=yd(0.5))])
    assert list(v["age"]) == [yd(25) + 150]  # the later (too-close) birth is flagged
    # widen the gap: 200 days > 183 -> clean
    df.loc[1, "age"] = yd(25) + 200
    assert P.check_rules(df, KEYS, [Rule("space", "birth", min_spacing=yd(0.5))]).empty


def test_not_after_ordering():
    # birth after the first death in the same group is flagged; a birth before death is not
    df = _frame([(1, "birth", 20), (1, "death", 40), (1, "birth", 45), (2, "birth", 30)])
    v = P.check_rules(df, KEYS, [Rule("resurrection", "birth", not_after="death")])
    assert list(v["person_id"]) == [1]
    assert list(v["age"]) == [yd(45)]  # only the post-death birth


def test_not_before_ordering():
    # a divorce before the first marriage is flagged; one after it is not
    df = _frame([(1, "marriage", 25), (1, "divorce", 30), (2, "divorce", 20), (2, "marriage", 28)])
    v = P.check_rules(df, KEYS, [Rule("early_divorce", "divorce", not_before="marriage")])
    assert list(v["person_id"]) == [2]
    assert list(v["age"]) == [yd(20)]


def test_not_before_flags_missing_anchor():
    # a divorce with no marriage anywhere in the sequence is a violation too (unlike not_after)
    df = _frame([(1, "divorce", 30), (2, "marriage", 20), (2, "divorce", 25)])
    v = P.check_rules(df, KEYS, [Rule("early_divorce", "divorce", not_before="marriage")])
    assert list(v["person_id"]) == [1]
    # the mirror not_after rule leaves the anchorless group alone
    assert P.check_rules(df, KEYS, [Rule("m", "marriage", not_after="divorce")]).empty


def test_max_count_flags_excess():
    df = _frame([(1, "birth", 20), (1, "birth", 25), (1, "birth", 30), (1, "birth", 35)])
    v = P.check_rules(df, KEYS, [Rule("cap", "birth", max_count=2)])
    assert list(v["age"]) == [yd(30), yd(35)]  # 3rd and 4th births


def test_violation_rates_denominator_is_the_governed_event():
    """The rate is a share of the events the rule applies to, so it stays inside [0, 1]."""
    df = _frame([(1, "birth", 11), (1, "birth", 25), (2, "birth", 52), (2, "birth", 30)])
    rules = [Rule("young", "birth", min_age=yd(12)), Rule("old", "birth", max_age=yd(50))]
    v = P.check_rules(df, KEYS, rules)
    rates = P.violation_rates(v, df, KEYS, by=()).set_index("rule")
    # 4 birth events; each rule fires once
    assert rates.loc["young", "n_events"] == 4
    assert rates.loc["young", "rate_per_event"] == pytest.approx(0.25)
    assert "n_groups" not in rates.columns and "rate_per_group" not in rates.columns


def test_violation_rates_stay_bounded_when_one_sequence_offends_repeatedly():
    """Counting per event row means a rate per *sequence* could exceed 1; per event cannot."""
    df = _frame([(1, "birth", 11), (1, "birth", 11), (1, "birth", 11), (2, "birth", 30)])
    v = P.check_rules(df, KEYS, [Rule("young", "birth", min_age=yd(12))])
    rates = P.violation_rates(v, df, KEYS, by=()).set_index("rule")
    assert rates.loc["young", "n_violations"] == 3  # one sequence, three offending births
    assert rates.loc["young", "rate_per_event"] == pytest.approx(0.75)  # 3 of 4 births


def test_violation_rates_by_seed():
    df = pd.DataFrame(
        {
            "person_id": [1, 1, 2, 2],
            "seed": [0, 1, 0, 1],
            "event": pd.Categorical(["birth"] * 4),
            "age": np.array([yd(11), yd(30), yd(11), yd(30)], dtype=np.int32),
        }
    )
    v = P.check_rules(df, ["person_id", "seed"], [Rule("young", "birth", min_age=yd(12))])
    rates = P.violation_rates(v, df, ["person_id", "seed"], by=("seed",)).set_index("seed")
    assert rates.loc[0, "n_violations"] == 2  # both persons' seed-0 births at age 11
    assert rates.loc[1, "n_violations"] == 0  # seed-1 births at age 30 are clean


# =================================================================================================
# occurrence-scoped rules: the difference between a token and an outcome
# =================================================================================================
def test_occurrence_narrows_the_subject_to_one_ordinal():
    """`event: birth` constrains every birth; `outcome: second_birth` constrains only the second."""
    df = _frame([(1, "birth", 20), (1, "birth", 22), (1, "birth", 24)])
    every = P.check_rules(df, KEYS, [Rule("all", "birth", min_age=yd(23))])
    assert sorted(every["age"]) == [yd(20), yd(22)]

    second = P.check_rules(df, KEYS, [Rule("second", "birth", occurrence=2, min_age=yd(23))])
    assert list(second["age"]) == [yd(22)]  # the first birth at 20 is not this rule's business


def test_a_divorce_before_the_first_marriage_is_flagged():
    """The motivating case, keyed on outcomes rather than raw tokens."""
    df = _frame(
        [
            (1, "marriage", 25), (1, "divorce", 30),   # fine
            (2, "divorce", 24), (2, "marriage", 28),   # divorced before marrying
            (3, "divorce", 31),                        # divorced, never married
        ]
    )
    rule = Rule("divorce_before_marriage", "divorce", occurrence=1, not_before="marriage")
    v = P.check_rules(df, KEYS, [rule])
    assert sorted(v["person_id"]) == [2, 3]


def test_the_anchor_occurrence_moves_the_line():
    """Second marriage as the anchor: a divorce between the two marriages is now early."""
    df = _frame([(1, "marriage", 25), (1, "divorce", 30), (1, "marriage", 35)])

    first_anchor = Rule("vs_first", "divorce", occurrence=1, not_before="marriage")
    assert P.check_rules(df, KEYS, [first_anchor]).empty  # 30 is after the first marriage at 25

    second_anchor = Rule(
        "vs_second", "divorce", occurrence=1, not_before="marriage", not_before_occurrence=2
    )
    assert list(P.check_rules(df, KEYS, [second_anchor])["person_id"]) == [1]


def test_a_missing_anchor_occurrence_is_itself_a_violation():
    """`not_before` treats an anchor that never happens as early — including a missing 2nd one."""
    df = _frame([(1, "marriage", 25), (1, "divorce", 30)])
    rule = Rule(
        "needs_two", "divorce", occurrence=1, not_before="marriage", not_before_occurrence=2
    )
    assert list(P.check_rules(df, KEYS, [rule])["person_id"]) == [1]


def test_not_after_leaves_groups_without_the_anchor_alone():
    """The asymmetry survives occurrence scoping: no anchor, no `not_after` violation."""
    df = _frame([(1, "birth", 30), (2, "death", 28), (2, "birth", 30)])
    rule = Rule("birth_after_death", "birth", occurrence=1, not_after="death")
    assert list(P.check_rules(df, KEYS, [rule])["person_id"]) == [2]


def test_occurrence_ordering_is_by_age_not_row_order():
    df = _frame([(1, "birth", 24), (1, "birth", 20)])  # rows out of order on purpose
    rule = Rule("second", "birth", occurrence=2, min_age=yd(23))
    assert P.check_rules(df, KEYS, [rule]).empty  # the 2nd birth is the one at 24
