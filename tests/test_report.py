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


def test_report_embeds_figures_and_caps_tables(demo_config, tmp_path):
    results = _run(demo_config, tmp_path / "results")
    # a many-row table exists; report shows it capped with a "showing N" note
    html = (results / report.REPORT_NAME).read_text()
    assert f"showing {report._MAX_TABLE_ROWS}" in html


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
    pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_parquet(
        results / "descriptives" / "t.parquet", index=False
    )
    path = report.build_report(results)
    assert path.exists()
    assert 'id="descriptives"' in path.read_text()


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
