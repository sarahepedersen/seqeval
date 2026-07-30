"""Sequence descriptives: what each table counts, and what it refuses to count."""

from __future__ import annotations

import numpy as np
import pandas as pd

from seqeval.metrics import sequences as SQ
from seqeval.units import years_to_days as yd

TOKENS = {"birth": "birth"}
UNITS = ["person_id"]


def _cell(events: dict[int, list[float]], *, end_age: float = 50.0, pad: bool = True):
    """``(frame, units, spans)`` for a hand-built cell: ages in years per unit.

    ``pad`` adds one undeclared end-of-sequence marker per unit, the shape the generator emits.
    """
    rows = []
    for unit, ages in events.items():
        rows += [(unit, "birth", yd(a)) for a in ages]
        if pad:
            rows.append((unit, "eos", yd(end_age)))
    frame = pd.DataFrame(rows, columns=["person_id", "event", "age"])
    ids = list(events)
    units = pd.DataFrame({"person_id": ids})
    spans = pd.DataFrame({"person_id": ids, "start_age": 0, "end_age": yd(end_age)})
    return frame, units, spans


# =================================================================================================
# the age grid
# =================================================================================================
def test_the_grid_spans_every_age_present():
    """`bin_ages` drops what falls outside its edges, so the grid is read off the data."""
    from seqeval.core.slicing import bin_ages

    frame, _, _ = _cell({1: [12.5, 47.9]}, end_age=48.0)
    bins = SQ.age_bins_for([frame])
    assert bin_ages(frame["age"], bins).notna().all()  # nothing silently dropped
    assert bins.labels.min() <= 12.0 and bins.labels.max() >= 48.0


def test_an_empty_grid_is_still_a_grid():
    assert len(SQ.age_bins_for([]).labels) >= 1


# =================================================================================================
# age distribution
# =================================================================================================
def test_share_sums_to_one_and_rate_uses_exposure():
    frame, _, spans = _cell({1: [20, 24], 2: [22], 3: []})
    bins = SQ.age_bins_for([frame])
    d = SQ.event_age_distribution(
        frame, spans, tokens=TOKENS, unit_keys=UNITS, bins=bins, min_cell=0
    )
    assert np.isclose(d["share"].sum(), 1.0)
    occupied = d[d["n_events"] > 0]
    np.testing.assert_allclose(occupied["rate"], occupied["n_events"] / occupied["person_years"])
    # three units alive through each one-year bin
    assert np.isclose(occupied["person_years"].iloc[0], 3.0, atol=0.05)


def test_every_bin_gets_a_row_so_a_zero_is_visible():
    """A gap in an age profile is a real zero; leaving it out would look like suppression."""
    frame, _, spans = _cell({1: [20, 30]})
    bins = SQ.age_bins_for([frame])
    d = SQ.event_age_distribution(
        frame, spans, tokens=TOKENS, unit_keys=UNITS, bins=bins, min_cell=0
    )
    assert len(d) == len(bins.labels)
    assert (d["n_events"] == 0).sum() == len(bins.labels) - 2


def test_a_pooled_cell_reports_people_as_well_as_trajectories():
    """Seeds multiply trajectories, not people, and both counts are published."""
    frame, _, spans = _cell({1: [20], 2: [20], 3: [20]})
    frame["source_person_id"] = frame["person_id"].map({1: 7, 2: 7, 3: 8})
    bins = SQ.age_bins_for([frame])
    d = SQ.event_age_distribution(
        frame, spans, tokens=TOKENS, unit_keys=UNITS, bins=bins, min_cell=0
    )
    row = d[d["n_events"] > 0].iloc[0]
    assert row["n_units"] == 3 and row["n_source_persons"] == 2


def test_an_undeclared_token_never_reaches_the_age_profile():
    frame, _, spans = _cell({1: [20]})
    bins = SQ.age_bins_for([frame])
    d = SQ.event_age_distribution(
        frame, spans, tokens=TOKENS, unit_keys=UNITS, bins=bins, min_cell=0
    )
    assert set(d["token"]) == {"birth"}
    assert d["n_events"].sum() == 1  # the `eos` marker is not counted


# =================================================================================================
# token frequency
# =================================================================================================
def _pooled(events: dict[int, list[float]], *, seeds_per_person: dict[int, int] | None = None):
    """A pooled generated cell: unit ids are trajectories, `source_person_id` the real person."""
    frame, units, spans = _cell(events)
    owner = seeds_per_person or dict.fromkeys(events, 1)
    frame["source_person_id"] = frame["person_id"].map(owner)
    units["source_person_id"] = units["person_id"].map(owner)
    return frame, units, spans


def test_the_pooled_share_counts_sequences_carrying_the_token():
    frame, units, _ = _pooled({1: [20, 24], 2: [22], 3: []})
    t = SQ.token_frequency(frame, units, tokens=TOKENS, unit_keys=UNITS, min_cell=0).iloc[0]
    assert t["n_events"] == 3
    assert t["n_units"] == 3 and t["n_units_with_any"] == 2
    assert np.isclose(t["share_with_any"], 2 / 3)


def test_the_per_person_share_gives_every_person_one_vote():
    """Person 1 has 4 trajectories and 1 hit; person 2 has 1 trajectory and 1 hit.

    Pooled, that is 2 of 5 sequences. Per person it is the mean of 1/4 and 1/1 — the pooled figure
    is dragged down by the better-replicated person, which is exactly the difference between the
    two columns.
    """
    frame, units, _ = _pooled(
        {1: [20], 2: [], 3: [], 4: [], 5: [22]},
        seeds_per_person={1: 100, 2: 100, 3: 100, 4: 100, 5: 200},
    )
    t = SQ.token_frequency(frame, units, tokens=TOKENS, unit_keys=UNITS, min_cell=0).iloc[0]
    assert np.isclose(t["share_with_any"], 2 / 5)
    assert np.isclose(t["mean_person_share"], (0.25 + 1.0) / 2)
    assert t["n_source_persons"] == 2 and t["n_persons_with_any"] == 2


def test_the_two_shares_agree_when_every_person_has_one_trajectory():
    frame, units, _ = _pooled({1: [20], 2: [], 3: [22]})
    t = SQ.token_frequency(frame, units, tokens=TOKENS, unit_keys=UNITS, min_cell=0).iloc[0]
    assert np.isclose(t["share_with_any"], t["mean_person_share"])


def test_the_denominator_counts_units_with_no_events():
    """A unit present in the population but absent from the event rows still belongs below."""
    frame, units, _ = _cell({1: [20]}, pad=False)
    units = pd.concat([units, pd.DataFrame({"person_id": [2, 3]})], ignore_index=True)
    t = SQ.token_frequency(frame, units, tokens=TOKENS, unit_keys=UNITS, min_cell=0).iloc[0]
    assert t["n_units"] == 3 and t["n_units_with_any"] == 1


def test_only_declared_tokens_are_described():
    """An undeclared end-of-sequence marker is not this table's subject and gets no row."""
    frame, units, _ = _cell({1: [20], 2: [22]})  # 2 births + 2 `eos` markers
    t = SQ.token_frequency(frame, units, tokens=TOKENS, unit_keys=UNITS, min_cell=0)
    assert list(t["alias"]) == ["birth"]
    assert t.iloc[0]["n_events"] == 2  # the markers are not counted


# =================================================================================================
# thin cells
# =================================================================================================
def test_a_thin_token_withholds_both_of_its_shares():
    frame, units, _ = _pooled({1: [20], 2: [22]})
    t = SQ.token_frequency(frame, units, tokens=TOKENS, unit_keys=UNITS, min_cell=3)
    assert t["suppressed"].all()
    assert t[["share_with_any", "mean_person_share", "n_units_with_any"]].isna().all().all()


def test_a_thin_age_bin_withholds_its_rate_and_share():
    frame, _, spans = _cell({1: [20], 2: [30], 3: [30], 4: [30], 5: [30]})
    bins = SQ.age_bins_for([frame])
    d = SQ.event_age_distribution(
        frame, spans, tokens=TOKENS, unit_keys=UNITS, bins=bins, min_cell=3
    )
    thin = d[d["suppressed"]]
    assert len(thin)
    assert thin[["n_events", "rate", "share", "person_years"]].isna().all().all()
    # a true zero is published, never suppressed
    assert not d[(d["n_events"] == 0)]["suppressed"].any()
