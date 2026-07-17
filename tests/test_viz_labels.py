"""Human-readable outcome descriptions for figure titles."""

from __future__ import annotations

from seqeval.core.specs import CountQuery, Frame, FramedOutcome, TTESpec
from seqeval.units import years_to_days as yd
from seqeval.viz._labels import describe_outcome

_LABEL = {"01": "live birth"}.get


def test_count_query_within():
    spec = CountQuery("q", "01", 1, Frame("within", yd(12)))
    assert (
        describe_outcome(spec, jumpoff_days=yd(28), label_fn=_LABEL)
        == "P(≥1 live birth within 12y after age 28)"
    )


def test_count_query_by_age():
    spec = CountQuery("q", "01", 2, Frame("by_age", yd(40)))
    assert (
        describe_outcome(spec, jumpoff_days=yd(30), label_fn=_LABEL)
        == "P(≥2 live birth after age 30, by age 40)"
    )


def test_framed_by_age_ordinal():
    spec = FramedOutcome("f", TTESpec("01", 2, origin=TTESpec("01", 1)), Frame("by_age", yd(35)))
    assert describe_outcome(spec, label_fn=_LABEL) == "P(2nd live birth by age 35)"


def test_framed_within_origin():
    spec = FramedOutcome(
        "f", TTESpec("01", 2, origin=TTESpec("01", 1)), Frame("within_origin", yd(5))
    )
    assert describe_outcome(spec, label_fn=_LABEL) == "P(2nd live birth within 5y of its origin)"


def test_default_label_fn_uses_raw_token():
    spec = CountQuery("q", 42, 1, Frame("within", yd(3)))
    assert describe_outcome(spec) == "P(≥1 42 within 3y after the jump-off)"
