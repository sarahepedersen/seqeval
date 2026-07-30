"""Every parquet a real run writes, swept for cells the policy should have withheld.

The unit tests in ``test_disclosure.py`` prove the rule. These prove it is *reached*: a table can
only leak by being built somewhere the policy does not run, and no unit test of the policy can see
that. So this file runs the pipeline and reads what landed on disk.

The second test is the one that matters over time. A table added next year with a count column and
no registry entry fails it, which is the only mechanism that keeps the registry honest.
"""

from __future__ import annotations

import glob
from pathlib import Path

import pandas as pd
import pytest

from seqeval import cli
from seqeval.metrics._disclosure import policy_for

#: Count-shaped column names that must be governed by a policy wherever they are published. Kept
#: as a suffix/prefix rule rather than a list, so a new count column is caught by its own name.
_COUNT_PREFIXES = ("n_", "births", "k_")

#: Columns whose name looks like a count but which count neither people nor events: replicate depth,
#: bin indices, grid resolutions. None of them narrows down an individual.
_NOT_HEAD_COUNTS = frozenset(
    {
        "n_seed_min",
        "n_seed_median",
        "n_seed_max",
        "n_bins",
        "k_seeds",
        "n_units",  # N x K trajectories; the real head count is `n_source_persons`
    }
)

MIN_CELL = 3


@pytest.fixture(scope="module")
def swept_run(tmp_path_factory, request) -> Path:
    """One full pipeline run at ``min_cell: 3``, shared by every test in this module."""
    from tests.conftest import _write_demo

    root = tmp_path_factory.mktemp("sweep")
    config = _write_demo(root / "data")
    text = config.read_text().replace(
        "  figure_format: png", f"  figure_format: png\n  min_cell: {MIN_CELL}"
    )
    config.write_text(text)
    out = root / "results"
    assert cli.main(["run", str(config), "--out", str(out)]) == 0
    return out


def _tables(run: Path):
    for path in sorted(glob.glob(f"{run}/**/*.parquet", recursive=True)):
        yield Path(path).stem, pd.read_parquet(path)


def test_the_run_wrote_something_to_sweep(swept_run):
    """A sweep over an empty directory passes vacuously; make that impossible."""
    stems = [stem for stem, _ in _tables(swept_run)]
    assert len(stems) > 10
    assert "coverage" in stems and "scores" in stems


def test_no_published_count_rests_on_a_thin_cell(swept_run):
    """The whole audit in one assertion, read off the files a user would be handed."""
    leaks = []
    for stem, frame in _tables(swept_run):
        policy = policy_for(stem)
        if policy is None:
            continue
        for col in policy.trip:
            if col not in frame.columns:
                continue
            values = pd.to_numeric(frame[col], errors="coerce")
            thin = (values > 0) & (values <= MIN_CELL)
            if thin.any():
                leaks.append(f"{stem}.{col}: {int(thin.sum())} cell(s) <= {MIN_CELL}")
    assert leaks == []


def test_every_count_column_written_is_governed_by_a_policy(swept_run):
    """The registry's own regression test: a new count column with no policy fails here.

    Per-person tables are exempt — ``output.individual_level`` governs those, and this run does not
    write them — as are the columns in ``_NOT_HEAD_COUNTS``, which count seeds and bins.
    """
    ungoverned = []
    for stem, frame in _tables(swept_run):
        policy = policy_for(stem)
        governed = set(policy.trip) | set(policy.also_null) if policy else set()
        for col in frame.columns:
            if col in _NOT_HEAD_COUNTS or col in governed:
                continue
            if col.startswith(_COUNT_PREFIXES) or col == "n":
                ungoverned.append(f"{stem}.{col}")
    assert ungoverned == []


def test_suppression_nulls_cells_without_dropping_rows(swept_run):
    """A withheld cell keeps its keys, so a figure drawn from the table keeps its shape."""
    for stem, frame in _tables(swept_run):
        if "suppressed" not in frame.columns or not frame["suppressed"].any():
            continue
        hidden = frame[frame["suppressed"].astype(bool)]
        policy = policy_for(stem)
        assert policy is not None, stem
        for col in policy.trip:
            if col in frame.columns:
                assert hidden[col].isna().all(), f"{stem}.{col}"


def test_a_thin_km_row_keeps_only_its_time_and_its_survival(swept_run):
    """The blast radius on KM, read off real output.

    The log-log interval is a function of the same Greenwood sum as the variance, so it cannot be
    published on a row whose counts were withheld — see ``_KM_INVERTS``.
    """
    km = pd.read_parquet(swept_run / "descriptives/km_first_birth.parquet")
    hidden = km[km["suppressed"].astype(bool)]
    assert len(hidden), "the demo cohort is small enough that some event times must be withheld"
    withheld = ["n_events", "n_at_risk", "n_persons", "greenwood_var", "ci_lo", "ci_hi"]
    assert hidden[withheld].isna().all().all()
    assert hidden["survival"].notna().all()
    assert hidden["time"].notna().all()


def test_the_km_counts_cannot_be_rebuilt_from_what_km_publishes(swept_run):
    """The inversion this policy exists to close, run as an attack rather than asserted about."""
    km = pd.read_parquet(swept_run / "descriptives/km_first_birth.parquet")
    hidden = km[km["suppressed"].astype(bool)]
    # cum_v = (se * |log S|)**2 needs `se`, which needs an interval bound. Neither is published.
    assert not {"greenwood_var", "ci_lo", "ci_hi"} & set(hidden.dropna(axis=1, how="all").columns)


def test_min_cell_zero_turns_suppression_off(tmp_path):
    """The escape hatch, for a run inside a trusted enclave."""
    from tests.conftest import _write_demo

    config = _write_demo(tmp_path / "data")
    config.write_text(
        config.read_text().replace("  figure_format: png", "  figure_format: png\n  min_cell: 0")
    )
    out = tmp_path / "open"
    assert cli.main(["run", str(config), "--out", str(out)]) == 0
    km = pd.read_parquet(out / "descriptives/km_first_birth.parquet")
    # The policy never ran, so the frame carries no flag at all — not a flag that is all-false.
    assert "suppressed" not in km.columns
    assert km["n_events"].notna().all()
    assert (km["n_events"] == 1).any(), "the demo data has singleton event times to withhold"
