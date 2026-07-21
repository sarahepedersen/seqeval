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


def test_max_count_flags_excess():
    df = _frame([(1, "birth", 20), (1, "birth", 25), (1, "birth", 30), (1, "birth", 35)])
    v = P.check_rules(df, KEYS, [Rule("cap", "birth", max_count=2)])
    assert list(v["age"]) == [yd(30), yd(35)]  # 3rd and 4th births


def test_violation_rates_denominators():
    df = _frame([(1, "birth", 11), (1, "birth", 25), (2, "birth", 52), (2, "birth", 30)])
    rules = [Rule("young", "birth", min_age=yd(12)), Rule("old", "birth", max_age=yd(50))]
    v = P.check_rules(df, KEYS, rules)
    rates = P.violation_rates(v, df, KEYS, by=()).set_index("rule")
    # 2 groups, 4 birth events; each rule fires once
    assert rates.loc["young", "n_groups"] == 2
    assert rates.loc["young", "n_events"] == 4
    assert rates.loc["young", "rate_per_group"] == pytest.approx(0.5)
    assert rates.loc["young", "rate_per_event"] == pytest.approx(0.25)


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
