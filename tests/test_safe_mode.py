"""``output.individual_level: false``: what stops being written, and what has to survive it.

The point of the restriction is not that fewer files appear — it is that the results directory can
be handed on as it stands. So these tests assert both halves: no file anywhere describes a single
person, and the aggregate substitutes that replaced the per-person views are all still there.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import pandas as pd
import pytest

from seqeval import cli

_PER_PERSON_TABLES = [
    "backtesting/probabilities.parquet",
    "forecasting/replicate_variance_individual_first_birth.parquet",
    "forecasting/replicate_occurrence_first_birth.parquet",
    "forecasting/violations.parquet",
]

_AGGREGATE_SUBSTITUTES = [
    "backtesting/timing_error.parquet",
    "backtesting/parity_distribution.parquet",
    "backtesting/calibration.parquet",
    "backtesting/coverage.parquet",
    "backtesting/scores.parquet",
    "forecasting/replicate_variance_aggregate_first_birth.parquet",
    "forecasting/within_seed_variance_distribution_first_birth.parquet",
    "forecasting/violation_rates.parquet",
]


def _restricted_config(demo_config: Path) -> Path:
    text = demo_config.read_text().replace(
        "  figure_format: png", "  figure_format: png\n  individual_level: false"
    )
    path = demo_config.with_name("restricted.yaml")
    path.write_text(text)
    return path


@pytest.fixture
def restricted_run(demo_config, tmp_path) -> Path:
    out = tmp_path / "safe"
    cli.main(["run", str(_restricted_config(demo_config)), "--out", str(out)])
    return out


def test_no_written_table_describes_a_single_person(restricted_run):
    """The enumerating guard: it holds for tables nobody remembered to classify, too."""
    leaks = [
        p
        for p in glob.glob(f"{restricted_run}/**/*.parquet", recursive=True)
        if "person_id" in pd.read_parquet(p).columns
    ]
    assert leaks == []


def test_the_known_per_person_artifacts_are_absent(restricted_run):
    for rel in _PER_PERSON_TABLES:
        assert not (restricted_run / rel).exists(), rel
    assert not list(restricted_run.glob("backtesting/timing_calibration_*.png"))


def test_the_aggregate_substitutes_all_survive(restricted_run):
    """A restricted report still has to be worth reading — this is what makes it so."""
    for rel in _AGGREGATE_SUBSTITUTES:
        assert (restricted_run / rel).exists(), rel
    assert list(restricted_run.glob("backtesting/timing_ridge_*.png"))
    assert list(restricted_run.glob("backtesting/uncertainty_ccf_*.png"))
    assert list(restricted_run.glob("backtesting/ccf_overlay_*.png"))
    # the dispersion ridges read the binned distribution, so they survive where the histogram of
    # per-person values could not
    assert list(restricted_run.glob("forecasting/within_seed_variance*.png"))


def test_manifest_names_what_was_withheld(restricted_run):
    """Absence is not an audit trail; the withheld artifacts are listed by name."""
    manifest = json.loads((restricted_run / "manifest.json").read_text())
    assert manifest["config_resolved"]["output"]["individual_level"] is False
    withheld = {n for arm in manifest["arms"].values() for n in arm["withheld"]}
    assert {
        "probabilities",
        "replicate_variance_individual_first_birth",
        "violations",
    } <= withheld


def test_report_states_the_policy_and_still_renders_every_arm(restricted_run):
    """One row in the run summary, no banner: the manifest is where the audit trail lives."""
    html = (restricted_run / "report.html").read_text()
    assert "Individual-level output" in html and "Minimum cell size" in html
    assert "This run was written with" not in html  # no disclosure banner
    for anchor in ("summary", "observed", "generated", "comparison"):
        assert f'id="{anchor}"' in html
    # the views that replaced the per-person ones are the ones carrying the section
    assert 'id="timing-error"' in html and 'id="uncertainty"' in html
    assert "sampled persons" not in html


def test_a_run_that_does_not_ask_gets_no_per_person_output(demo_config, tmp_path):
    """The flag defaults to false: naming a person is opt-in, not the path of least resistance."""
    out = tmp_path / "default"
    cli.main(["run", str(demo_config), "--out", str(out)])
    for rel in _PER_PERSON_TABLES:
        assert not (out / rel).exists(), rel
    assert "Individual-level output" in (out / "report.html").read_text()


def test_asking_for_individual_level_output_gets_it(demo_config, tmp_path):
    """The opt-in still works, and is the only way to reach the per-person tables."""
    text = demo_config.read_text().replace(
        "  figure_format: png", "  figure_format: png\n  individual_level: true"
    )
    cfg = demo_config.with_name("permissive.yaml")
    cfg.write_text(text)
    out = tmp_path / "full"
    cli.main(["run", str(cfg), "--out", str(out)])
    for rel in _PER_PERSON_TABLES:
        assert (out / rel).exists(), rel


def test_min_cell_controls_how_much_is_withheld(demo_config, tmp_path):
    """Raising the threshold withholds strictly more; the tables keep their shape either way."""
    frames = {}
    for min_cell in (2, 25):
        text = demo_config.read_text().replace(
            "  figure_format: png", f"  figure_format: png\n  min_cell: {min_cell}"
        )
        cfg = demo_config.with_name(f"cell{min_cell}.yaml")
        cfg.write_text(text)
        out = tmp_path / f"cell{min_cell}"
        cli.main(["run", str(cfg), "--out", str(out)])
        frames[min_cell] = pd.read_parquet(out / "backtesting/parity_distribution.parquet")

    lax, strict = frames[2], frames[25]
    assert len(lax) == len(strict)  # suppression nulls cells, it never drops rows
    assert strict["suppressed"].sum() > lax["suppressed"].sum()
    assert strict["n_replicates"].isna().sum() > lax["n_replicates"].isna().sum()
