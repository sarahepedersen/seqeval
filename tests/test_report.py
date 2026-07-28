"""Report + manifest builders (06): HTML assembly, section anchors, graceful gaps."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from seqeval import cli, report


def _run(demo_config: Path, out: Path) -> Path:
    cli.main(["run", str(demo_config), "--out", str(out)])
    return out


def test_build_report_has_all_sections(demo_config, tmp_path):
    results = _run(demo_config, tmp_path / "results")
    html = (results / report.REPORT_NAME).read_text()

    for anchor in ("summary", "descriptives", "backtesting", "forecasting", "warnings"):
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
    backtest = html.split('<h2 id="backtesting"')[1].split('<h2 id="forecasting"')[0]
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
    assert "Inference vs outcome uncertainty" in html


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
    forecasting = html.split('<h2 id="forecasting"')[1]
    assert "sampled persons" in forecasting
    assert "replicate_variance_individual" in forecasting


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
    assert 'id="descriptives"' in html
    assert 'id="backtesting"' not in html
    assert 'id="forecasting"' not in html
    assert 'id="summary"' in html and 'id="warnings"' in html


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
    assert 'id="descriptives"' in html
    assert "t.parquet" in html  # linked under the figure


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
    section = html.split('<h2 id="backtesting"')[1].split('<h2 id="forecasting"')[0]
    assert "<th>ECE</th>" in section
    assert "ECE has no CI" in section

    scores = pd.read_parquet(results / "backtesting" / "scores.parquet")
    ece = scores[scores["metric"] == "ece"]
    assert len(ece) and ece["ci_lo"].isna().all()
    # the value itself is rendered somewhere in the table
    assert f"{ece['value'].iloc[0]:.3f}" in section

