"""Human-readable descriptions of the day-valued question specs, for figure titles/captions (viz).

Turning a resolved :mod:`seqeval.core.specs` object back into a plain-English probability statement
is a display concern, so it lives in ``viz`` (one of the three sanctioned unit-conversion sites,
00 section 3) — it renders day-valued frames back to years and uses a caller-supplied ``label_fn``
(typically ``Bundle.label``) to name raw event tokens.
"""

from __future__ import annotations

from collections.abc import Callable

from seqeval.core.specs import CountQuery, FramedOutcome, TTESpec
from seqeval.units import days_to_years

__all__ = ["describe_outcome"]


def _ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _yr(days: int) -> str:
    return f"{days_to_years(days):.0f}"


def describe_outcome(
    spec: FramedOutcome | CountQuery,
    *,
    jumpoff_days: int | None = None,
    label_fn: Callable[[object], str] = str,
) -> str:
    """A one-line ``P(...)`` description of the binary outcome ``spec`` evaluates.

    Parameters
    ----------
    spec : FramedOutcome or CountQuery
        The resolved (day-valued, raw-token) outcome spec.
    jumpoff_days : int or None
        The jump-off age (days) the outcome is evaluated at. Always named, either inside the
        sentence or as a trailing clause: the same question asked at two jump-offs is two different
        figures, and a shared title would read as one.
    label_fn : callable, default str
        Maps a raw event token to a human label (pass ``Bundle.label`` for real names).
    """
    jo = f"age {_yr(jumpoff_days)}" if jumpoff_days is not None else "the jump-off"
    given = f" | {spec.given}" if spec.given else ""

    if isinstance(spec, CountQuery):
        event = label_fn(spec.event)
        w = _yr(spec.frame.value)
        if spec.frame.kind == "within":
            return f"P(≥{spec.min_events} {event} within {w}y after {jo}{given})"
        return f"P(≥{spec.min_events} {event} after {jo}, by age {w}{given})"

    if isinstance(spec, FramedOutcome):
        occurrence = _describe_tte(spec.tte, label_fn)
        v = _yr(spec.frame.value)
        if spec.frame.kind == "by_age":
            return f"P({occurrence} by age {v}{given}) from {jo}"
        if spec.frame.kind == "within":
            return f"P({occurrence} within {v}y after {jo}{given})"
        return f"P({occurrence} within {v}y of its origin{given}) from {jo}"

    raise TypeError(f"cannot describe {type(spec).__name__}")


def _describe_tte(tte: TTESpec, label_fn: Callable[[object], str]) -> str:
    """e.g. ``2nd live birth`` — the ordinal occurrence of the target event."""
    return f"{_ordinal(tte.occurrence)} {label_fn(tte.target)}"
