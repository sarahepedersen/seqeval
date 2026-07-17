"""Light script test for examples/make_persons.py (not part of the package)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from examples.make_persons import persons_from_sequences  # noqa: E402


def _observed_with_year_tokens(missing_person=False):
    rows = [
        (1, "YEAR_1987", 0),
        (1, "birth", 9131),
        (2, "YEAR_1990", 0),
        (2, "birth", 10000),
    ]
    if missing_person:
        rows.append((3, "birth", 8000))  # person 3 has no YEAR_ token
    return pd.DataFrame(
        {
            "person_id": np.array([r[0] for r in rows], dtype=np.int64),
            "event": [r[1] for r in rows],
            "age": np.array([r[2] for r in rows], dtype=np.int32),
        }
    )


def test_extracts_birth_years():
    persons = persons_from_sequences(_observed_with_year_tokens())
    assert dict(zip(persons["person_id"], persons["birth_year"], strict=True)) == {
        1: 1987,
        2: 1990,
    }
    assert persons["birth_year"].dtype == np.int16


def test_missing_token_lists_ids():
    with pytest.raises(ValueError, match=r"\[3\]"):
        persons_from_sequences(_observed_with_year_tokens(missing_person=True))


def test_allow_missing_drops_them():
    persons = persons_from_sequences(
        _observed_with_year_tokens(missing_person=True), allow_missing=True
    )
    assert set(persons["person_id"]) == {1, 2}
