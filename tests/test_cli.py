"""CLI end-to-end (06): validate/run/report, arm selection, isolation, exit codes."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from seqeval import cli, report
from seqeval.arms import descriptives as descriptives_arm


def _manifest(results: Path) -> dict:
    return json.loads((results / report.MANIFEST_NAME).read_text())


def _strip_volatile(m: dict) -> dict:
    """Drop the two fields allowed to differ between identical runs (06 section 2)."""
    m = json.loads(json.dumps(m))  # deep copy
    m.pop("timestamp_utc", None)
    for arm in m["arms"].values():
        arm.pop("duration_s", None)
    return m


def test_validate_ok(demo_config, capsys):
    assert cli.main(["validate", str(demo_config)]) == 0
    out = capsys.readouterr().out
    assert "demo_perfect_model" in out
    assert "OK:" in out


def test_run_creates_manifest_and_outputs(demo_config, tmp_path):
    results = tmp_path / "results"
    assert cli.main(["run", str(demo_config), "--out", str(results)]) == 0

    assert (results / report.MANIFEST_NAME).exists()
    assert (results / report.REPORT_NAME).exists()

    m = _manifest(results)
    assert m["model"]["name"] == "demo_perfect_model"
    assert set(m["arms"]) == {"descriptives", "backtesting", "forecasting"}
    for arm, meta in m["arms"].items():
        assert meta["status"] == "ok", arm
        assert meta["outputs"], f"{arm} wrote nothing"
        assert (results / arm).is_dir()
    # every recorded output actually exists on disk
    for meta in m["arms"].values():
        for rel in meta["outputs"]:
            assert (results / rel).exists()


def test_inputs_hashed_in_manifest(demo_config, tmp_path):
    results = tmp_path / "results"
    cli.main(["run", str(demo_config), "--out", str(results)])
    inputs = _manifest(results)["inputs"]
    assert set(inputs) == {"observed", "generated", "persons", "event_definitions"}
    for role in ("observed", "generated", "persons"):
        assert len(inputs[role]["sha256"]) == 64
        assert inputs[role]["n_rows"] > 0


def test_manifest_deterministic(demo_config, tmp_path):
    r1, r2 = tmp_path / "r1", tmp_path / "r2"
    cli.main(["run", str(demo_config), "--out", str(r1)])
    cli.main(["run", str(demo_config), "--out", str(r2)])
    assert _strip_volatile(_manifest(r1)) == _strip_volatile(_manifest(r2))


def test_arm_selection_runs_only_named(demo_config, tmp_path):
    results = tmp_path / "results"
    assert cli.main(["run", str(demo_config), "--out", str(results), "--arm", "descriptives"]) == 0
    m = _manifest(results)
    assert set(m["arms"]) == {"descriptives"}
    assert (results / "descriptives").is_dir()
    assert not (results / "backtesting").exists()
    assert not (results / "forecasting").exists()


def test_missing_input_file_exit_code(demo_config, tmp_path):
    cfg = yaml.safe_load(demo_config.read_text())
    cfg["data"]["observed"] = "does_not_exist.parquet"
    demo_config.write_text(yaml.safe_dump(cfg))
    assert cli.main(["run", str(demo_config), "--out", str(tmp_path / "results")]) == 2


def test_bad_config_path_exit_code(tmp_path):
    assert cli.main(["validate", str(tmp_path / "nope.yaml")]) == 2


def test_arm_failure_is_isolated(demo_config, tmp_path, monkeypatch):
    """A failing arm records status=failed, others still run, exit code is nonzero."""

    def _boom(*args, **kwargs):
        raise RuntimeError("induced descriptives failure")

    monkeypatch.setattr(descriptives_arm, "run", _boom)

    results = tmp_path / "results"
    assert cli.main(["run", str(demo_config), "--out", str(results)]) == 1

    m = _manifest(results)
    assert m["arms"]["descriptives"]["status"] == "failed"
    assert m["arms"]["backtesting"]["status"] == "ok"
    assert m["arms"]["forecasting"]["status"] == "ok"
    # the report is still built despite the failure
    assert (results / report.REPORT_NAME).exists()


def test_force_required_to_overwrite(demo_config, tmp_path):
    results = tmp_path / "results"
    assert cli.main(["run", str(demo_config), "--out", str(results)]) == 0
    # second run without --force refuses
    assert cli.main(["run", str(demo_config), "--out", str(results)]) == 2
    # with --force it proceeds
    assert cli.main(["run", str(demo_config), "--out", str(results), "--force"]) == 0


def test_report_subcommand_rebuilds(demo_config, tmp_path):
    results = tmp_path / "results"
    cli.main(["run", str(demo_config), "--out", str(results)])
    (results / report.REPORT_NAME).unlink()
    assert cli.main(["report", str(results)]) == 0
    assert (results / report.REPORT_NAME).exists()


def test_report_subcommand_rejects_non_dir(tmp_path):
    assert cli.main(["report", str(tmp_path / "missing")]) == 2
