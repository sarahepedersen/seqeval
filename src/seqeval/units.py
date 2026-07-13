"""The single unit-conversion boundary for the whole codebase (00 architecture, section 3).

Canonical internal unit for age and duration is **integer days** (``int32``). Every user-facing
value — YAML config, figure axes, report tables — is in **years**. Conversions between the two are
allowed in exactly three families of call sites, and all of them go through this module:

1. loaders normalizing input data (``age_unit: years`` -> days),
2. ``config.resolve_*`` turning year-valued config into day-valued specs,
3. ``viz`` / reporting converting day-valued results back to years for display.

Nothing else in ``core/`` or ``metrics/`` converts units. The one documented exception is
demographic rates that are defined per person-*year* (ASFR, life-table rates): those divide
person-days by :data:`DAYS_PER_YEAR` at the final rate computation, with a comment at the site.
"""

from __future__ import annotations

import numpy as np

__all__ = ["DAYS_PER_YEAR", "years_to_days", "days_to_years", "completed_years"]

#: Days per year used for every age/duration conversion. Fixed (Julian year) so that the
#: conversion is deterministic and reproducible; we never key off an actual calendar.
DAYS_PER_YEAR = 365.25


def years_to_days(y: float) -> int:
    """Convert a year-valued quantity to integer days.

    Rounds to the nearest whole day. Used to turn year-valued config numbers (windows, horizons,
    ages, spacing) and year-valued input columns into the canonical integer-day representation.

    Parameters
    ----------
    y : float
        A duration or age in years.

    Returns
    -------
    int
        ``round(y * DAYS_PER_YEAR)``.
    """
    return int(round(y * DAYS_PER_YEAR))


def days_to_years(d: int | float | np.ndarray) -> float | np.ndarray:
    """Convert an integer-day quantity to years (scalar or vectorized).

    Parameters
    ----------
    d : int, float, or numpy.ndarray
        A duration or age in days.

    Returns
    -------
    float or numpy.ndarray
        ``d / DAYS_PER_YEAR``. Returns a float for scalar input and an ``ndarray`` (float) for
        array input, so callers can use the same helper on columns and on single values.
    """
    if isinstance(d, np.ndarray):
        return d / DAYS_PER_YEAR
    return float(d) / DAYS_PER_YEAR


def completed_years(age_days: np.ndarray) -> np.ndarray:
    """Completed (floored) integer years for an array of day-valued ages.

    This is the building block for calendar-year derivation
    (``year = birth_year + completed_years(age)``, 00 section 3): it counts *completed* years, i.e.
    the number of full birthdays reached, never rounding up. Because we do not know the birth date
    within the calendar year, the resulting calendar year is an approximation — that caveat is
    documented wherever this feeds a ``year`` column.

    Parameters
    ----------
    age_days : numpy.ndarray
        Ages in integer days.

    Returns
    -------
    numpy.ndarray
        ``floor(age_days / DAYS_PER_YEAR)`` as an ``int64`` array.
    """
    return np.floor(np.asarray(age_days) / DAYS_PER_YEAR).astype(np.int64)
