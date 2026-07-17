"""Units: round-trip properties, integrality, vectorized paths (01 section 9)."""

from __future__ import annotations

import numpy as np
import pytest

from seqeval.units import DAYS_PER_YEAR, completed_years, days_to_years, years_to_days


@pytest.mark.parametrize("years", [0, 1, 15, 25.5, 30, 45.25, 110])
def test_years_to_days_is_integer(years):
    d = years_to_days(years)
    assert isinstance(d, int)
    assert d == round(years * DAYS_PER_YEAR)


def test_round_trip_within_half_day():
    # days -> years -> days recovers the original day (rounding error < half a day).
    for d in [0, 1, 9131, 10958, 40177]:
        assert years_to_days(days_to_years(d)) == d


def test_days_to_years_scalar_and_vector():
    assert days_to_years(0) == 0.0
    assert days_to_years(DAYS_PER_YEAR) == pytest.approx(1.0, abs=1e-9)

    arr = np.array([0, 365, 730], dtype=np.int32)
    out = days_to_years(arr)
    assert isinstance(out, np.ndarray)
    np.testing.assert_allclose(out, arr / DAYS_PER_YEAR)


def test_completed_years_floors():
    # completed_years counts *completed* years; note years_to_days(1.0)=365 rounds just under one
    # 365.25-day year, so it floors to 0 — the documented calendar-year approximation.
    ages = np.array([0, years_to_days(0.99), years_to_days(1.5), years_to_days(29.9)])
    np.testing.assert_array_equal(completed_years(ages), np.array([0, 0, 1, 29]))
