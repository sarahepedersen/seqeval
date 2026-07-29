"""Report + manifest builders (06): HTML assembly, section anchors, graceful gaps."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import pytest

from seqeval import cli, report


def _run(demo_config: Path, out: Path) -> Path:
    cli.main(["run", str(demo_config), "--out", str(out)])
    return out


def test_build_report_has_all_sections(demo_config, tmp_path):
    results = _run(demo_config, tmp_path / "results")
    html = (results / report.REPORT_NAME).read_text()

    for anchor in ("summary", "observed", "generated", "comparison"):
        assert f'id="{anchor}"' in html, anchor
    assert "demo_perfect_model" in html
    # figures embedded as base64, not linked to disk
    assert "data:image/png;base64," in html
    # tables link back to their parquet
    assert ".parquet" in html


def test_metrics_table_carries_the_coverage_counts(demo_config, tmp_path):
    """Evaluability rides on the score's own row, not in a table of its own."""
    results = _run(demo_config, tmp_path / "results")
    html = (results / report.REPORT_NAME).read_text()
    backtest = html.split('<h2 id="comparison"')[1]
    assert "Backtest metrics" in backtest
    assert "Backtest coverage (evaluability)" not in backtest
    header = backtest.split("</tr>")[0]
    for col in ("n_condition", "n_evaluable", "n_settled", "n_uncovered"):
        assert f"<th>{col}</th>" in header
    # the corrected Brier is gone; MSE carries both names, since p_hat is k/n either way
    assert "<th>MSE/Brier</th>" in header
    assert "<th>Brier</th>" not in header


def test_aggregate_target_error_table_is_not_reported(demo_config, tmp_path):
    results = _run(demo_config, tmp_path / "results")
    html = (results / report.REPORT_NAME).read_text()
    assert 'id="aggregate-error"' not in html
    assert (results / "backtesting" / "aggregate_error.parquet").exists()


def test_timing_error_section_renders_when_figures_present(demo_config, tmp_path):
    """The timed-outcome ridges appear in the backtesting section."""
    results = _run(demo_config, tmp_path / "results")
    assert list((results / "backtesting").glob("timing_ridge_*.png"))
    html = (results / report.REPORT_NAME).read_text()
    assert 'id="timing-error"' in html


def test_uncertainty_section_contrasts_the_two_uncertainties(demo_config, tmp_path):
    """The inference-vs-outcome figure gets its own section, with the distinction stated."""
    results = _run(demo_config, tmp_path / "results")
    assert list((results / "backtesting").glob("uncertainty_ccf_*.png"))
    html = (results / report.REPORT_NAME).read_text()
    assert 'id="uncertainty"' in html
    assert "inference vs outcome uncertainty" in html.lower()


def test_coverage_summary_absent_without_backtesting(demo_config, tmp_path):
    """No backtesting arm dir → no coverage summary, and the report still builds."""
    results = _run(demo_config, tmp_path / "results")
    import shutil

    shutil.rmtree(results / "backtesting")
    html = report.build_report(results).read_text()
    assert "Backtest coverage" not in html
    assert 'id="summary"' in html


def test_report_embeds_figures_and_samples_persons(demo_config, tmp_path):
    results = _run(demo_config, tmp_path / "results")
    html = (results / report.REPORT_NAME).read_text()
    assert "data:image/png;base64," in html
    # per-person tables (replicate variance, violations) are down-sampled, not fully dumped
    generated = html.split('<h2 id="generated"')[1].split('<h2 id="comparison"')[0]
    assert "sampled persons" in generated
    assert "replicate_variance_individual" in generated


def test_build_report_missing_arm_dir_is_graceful(demo_config, tmp_path):
    """A results dir with only one arm still builds; absent arms are simply omitted."""
    results = _run(
        demo_config,
        tmp_path / "results",
    )
    # drop two arm dirs, keep descriptives, rebuild
    import shutil

    shutil.rmtree(results / "backtesting")
    shutil.rmtree(results / "forecasting")
    path = report.build_report(results)
    html = path.read_text()
    assert 'id="observed"' in html
    assert 'id="comparison"' not in html
    assert 'id="generated"' not in html
    assert 'id="summary"' in html


def test_build_report_without_manifest(tmp_path):
    """build_report tolerates a results dir with no manifest.json."""
    results = tmp_path / "results"
    (results / "descriptives").mkdir(parents=True)
    # descriptives renders figures (with a parquet link) — write a figure so the section appears
    pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_parquet(
        results / "descriptives" / "t.parquet", index=False
    )
    (results / "descriptives" / "t.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    path = report.build_report(results)
    assert path.exists()
    html = path.read_text()
    assert 'id="observed"' in html
    assert "t.parquet" in html  # linked under the figure


def test_every_parquet_link_resolves_from_the_report(demo_config, tmp_path):
    """The report sits above the arm dirs, so a bare filename would 404 on every link."""
    results = _run(demo_config, tmp_path / "results")
    html = (results / report.REPORT_NAME).read_text()
    hrefs = {h for h in re.findall(r'<a href="([^"]+)"', html) if not h.startswith("#")}
    assert hrefs, "no parquet links rendered"
    broken = sorted(h for h in hrefs if not (results / h).exists())
    assert not broken, broken
    assert all("/" in h for h in hrefs)  # arm-qualified, not bare


def test_warnings_are_not_rendered_in_the_report(demo_config, tmp_path):
    """The manifest is the audit trail for warnings; the report is the read-through."""
    results = _run(demo_config, tmp_path / "results")
    html = (results / report.REPORT_NAME).read_text()
    assert 'id="warnings"' not in html
    assert "<h2>Warnings</h2>" not in html
    assert report.read_manifest(results) is not None  # still recorded where it belongs


def test_the_scored_quantity_and_the_calibration_figure_are_explained(demo_config, tmp_path):
    """Both sections define what they show before showing it, and nothing renders as a TODO."""
    results = _run(demo_config, tmp_path / "results")
    html = (results / report.REPORT_NAME).read_text()
    assert "TODO" not in html

    metrics = html.split('id="backtest-metrics"')[1].split("<table")[0]
    assert "p̂</code> is defined for a particular outcome" in metrics

    calibration = html.split('id="calibration"')[1].split('id="timing-error"')[0]
    assert "quantile bins" in calibration
    assert "perfect calibration" in calibration
    # the explanation sits above the figures it explains, as everywhere else in the report
    assert calibration.index("perfect calibration") < calibration.index("<div class='figrow'>")


def test_every_rendered_group_declares_its_replicate_basis(demo_config, tmp_path):
    """Averaged-across-replicates vs individual-trajectories is stated for every plot group."""
    results = _run(demo_config, tmp_path / "results")
    html = (results / report.REPORT_NAME).read_text()
    assert set(report.REPLICATE_BASIS.values()) <= {None, *report._BASIS_TEXT}
    # the groups whose construction the report commits to each state it once, in their own section
    stated = html.count("Replicate handling:")
    assert stated >= sum(v is not None for v in report.REPLICATE_BASIS.values()) - 3
    for section, anchor, nxt in (
        ("observed", '<h2 id="observed"', '<h2 id="generated"'),
        ("comparison", '<h2 id="comparison"', "</body>"),
    ):
        body = html.split(anchor)[1].split(nxt)[0]
        assert "Replicate handling:" in body, section


def test_replicate_basis_renders_the_chosen_wording(demo_config, tmp_path, monkeypatch):
    """Each vocabulary value renders its own sentence; an unset group draws no line at all."""
    results = _run(demo_config, tmp_path / "results")
    monkeypatch.setitem(report.REPLICATE_BASIS, "observed.km", "averaged")
    monkeypatch.setitem(report.REPLICATE_BASIS, "observed.ccf", "trajectories")
    monkeypatch.setitem(report.REPLICATE_BASIS, "observed.asfr", None)
    html = report.build_report(results).read_text()
    observed = html.split('<h2 id="observed"')[1].split('<h2 id="generated"')[0]
    assert report._BASIS_TEXT["averaged"] in observed
    assert report._BASIS_TEXT["trajectories"] in observed
    assert observed.count("Replicate handling:") == 2  # the unset group is silent, not blank


def test_an_unknown_basis_key_raises(monkeypatch):
    """A renamed group must fail loudly rather than lose its marker."""
    monkeypatch.delitem(report.REPLICATE_BASIS, "comparison.km")
    with pytest.raises(KeyError):
        report._basis_item("comparison.km")


def test_manifest_roundtrip(demo_config, tmp_path):
    results = _run(demo_config, tmp_path / "results")
    m = report.read_manifest(results)
    assert m is not None
    # config_resolved round-trips through the config hash
    assert m["config_hash"] == json.loads((results / "manifest.json").read_text())["config_hash"]
    assert m["coverage"]["n_persons"] > 0
    assert m["coverage"]["windows"], "expected window coverage rows"


def test_read_manifest_absent(tmp_path):
    assert report.read_manifest(tmp_path) is None


def test_sha256_file_stable(tmp_path):
    p = tmp_path / "x.bin"
    p.write_bytes(b"seqeval")
    assert report.sha256_file(p) == report.sha256_file(p)
    assert len(report.sha256_file(p)) == 64


def test_metrics_table_carries_ece_without_an_interval(demo_config, tmp_path):
    """ECE is the number the reliability diagrams draw, so it belongs in the headline table.

    It is the one metric with no defensible closed-form interval, so it renders bare while the
    others carry parentheses — and the note says why.
    """
    results = _run(demo_config, tmp_path / "results")
    html = (results / report.REPORT_NAME).read_text()
    section = html.split('<h2 id="comparison"')[1]
    assert "<th>ECE</th>" in section
    assert "ECE has no CI" in section

    scores = pd.read_parquet(results / "backtesting" / "scores.parquet")
    ece = scores[scores["metric"] == "ece"]
    assert len(ece) and ece["ci_lo"].isna().all()
    # the value itself is rendered somewhere in the table
    assert f"{ece['value'].iloc[0]:.3f}" in section

