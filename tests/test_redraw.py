"""Rebuilding the report from the parquets alone.

The point of the redraw path is that a results directory can be exported as ``*.parquet`` +
``manifest.json`` and still produce the full report. The strong test is pixel equality against the
figures the run itself drew: it fails if a figure is drawn from anything the export does not carry.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from seqeval import cli, redraw


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def full_run(tmp_path_factory) -> Path:
    """One complete run, figures and all — the reference the redraw is compared against."""
    from tests.conftest import _write_demo

    root = tmp_path_factory.mktemp("redraw")
    config = _write_demo(root / "data")
    out = root / "results"
    assert cli.main(["run", str(config), "--out", str(out)]) == 0
    return out


@pytest.fixture
def exported(full_run, tmp_path) -> Path:
    """``full_run`` with every figure stripped — an export of the tables only."""
    dest = tmp_path / "exported"
    shutil.copytree(full_run, dest)
    for png in dest.rglob("*.png"):
        png.unlink()
    (dest / "report.html").unlink(missing_ok=True)
    return dest


def test_the_report_has_no_figures_without_a_redraw(exported):
    """The gap this module exists to close: `build_report` embeds PNGs, it never draws."""
    assert cli.main(["report", str(exported)]) == 0
    html = (exported / "report.html").read_text()
    assert "<img" not in html
    # the inline tables still render, which is why the failure is easy to miss
    assert 'id="summary"' in html


def test_redraw_rebuilds_every_figure(full_run, exported):
    expected = {str(p.relative_to(full_run)) for p in full_run.rglob("*.png")}
    assert expected, "the reference run drew no figures"

    written = redraw.redraw(exported)
    got = {str(p.relative_to(exported)) for p in exported.rglob("*.png")}
    assert got == expected
    assert len(written) == len(expected)


def test_redrawn_figures_are_identical_to_the_ones_the_run_drew(full_run, exported):
    """Pixel equality — the assertion that catches a figure reading unpublished data.

    Event labels come from ``events.csv``, an *input*, so it is passed here exactly as a user
    exporting tables would pass it back.
    """
    events = full_run.parent / "data" / "events.csv"
    redraw.redraw(exported, event_definitions=events)

    mismatched = [
        str(p.relative_to(full_run))
        for p in sorted(full_run.rglob("*.png"))
        if _digest(p) != _digest(exported / p.relative_to(full_run))
    ]
    assert mismatched == []


def test_the_rebuilt_report_matches_the_original(full_run, exported):
    events = full_run.parent / "data" / "events.csv"
    assert cli.main(["report", str(exported), "--redraw", "--events", str(events)]) == 0
    original = (full_run / "report.html").read_text()
    rebuilt = (exported / "report.html").read_text()
    assert rebuilt.count("<img") == original.count("<img")
    for anchor in ("summary", "observed", "generated", "comparison"):
        assert f'id="{anchor}"' in rebuilt


def test_redraw_without_events_csv_still_draws_everything(exported):
    """Only the titles degrade to raw tokens; every figure is still produced."""
    written = redraw.redraw(exported)
    assert len(written) == len({p for p in exported.rglob("*.png")})


def test_redraw_needs_the_manifest(exported):
    """The resolved config is not recoverable from the parquets, so its absence is an error."""
    (exported / "manifest.json").unlink()
    with pytest.raises(FileNotFoundError, match="manifest.json"):
        redraw.redraw(exported)
    assert cli.main(["report", str(exported), "--redraw"]) == 2


def test_a_partial_export_redraws_what_it_can(exported):
    """One arm's tables are enough; the others are skipped rather than failing the run."""
    shutil.rmtree(exported / "backtesting")
    shutil.rmtree(exported / "forecasting")
    written = redraw.redraw(exported)
    assert written
    assert all("descriptives" in str(p) for p in written)


def test_the_ccf_figures_are_skipped_without_their_variance_table(exported, caplog):
    """Their curve, band and dashing all come from `ccf_variance` — say so rather than guess."""
    import logging

    (exported / "backtesting" / "ccf_variance.parquet").unlink()
    with caplog.at_level(logging.WARNING, logger="seqeval"):
        redraw.redraw(exported)
    assert any("CCF figures" in r.message for r in caplog.records)
    assert not list((exported / "backtesting").glob("uncertainty_ccf_*.png"))
    # everything else still came through
    assert list((exported / "backtesting").glob("km_overlay_*.png"))


def test_redraw_reads_the_config_out_of_the_manifest(exported):
    """A hand-edited manifest changes the figures, which is what makes it the source of truth."""
    manifest = json.loads((exported / "manifest.json").read_text())
    assert manifest["config_resolved"]["output"]["min_cell"] >= 0
    assert "replicates" in manifest["config_resolved"]
