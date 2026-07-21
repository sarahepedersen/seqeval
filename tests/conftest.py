"""Shared fixtures for the pipeline (06) tests: a tiny on-disk demo dataset + config."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from tests import synthetic as S

# A compact config exercising all three arms; bootstrap off for speed.
_CONFIG = """\
model:
  name: demo_perfect_model
data:
  observed: observed.parquet
  generated: generated.parquet
  persons: persons.parquet
  event_definitions: events.csv
  age_unit: days
events:
  birth: birth
persons:
  covariates: [education, region]
  cohort_width: 5
replicates:
  min_replicates: 5
  bootstrap: {n: 0, seed: 7}
  convergence_curve: true
outcomes:
  first_birth: {event: birth, n: 1}
  second_birth: {event: birth, n: 2, origin: first_birth}
arms:
  descriptives:
    kaplan_meier: [first_birth, second_birth]
    fertility: {ccf: true, asfr: [period, cohort], ppr: {max_parity: 6}}
    life_table: {max_parity: 6}
    stratify_by: [cohort]
  backtesting:
    windows: all
    conditions:
      - {name: p0, event: birth, max_count: 0}
      - {name: p1, event: birth, min_count: 1, max_count: 1}
    probability_outcomes:
      - {outcome: first_birth, by_age: 35, given: p0}
      - {event: birth, min_events: 1, within: 5}
    aggregate_targets: [ccf, km:first_birth]
    min_seeds: 5
  forecasting:
    windows: all
    lexis: {outcome: first_birth, ages: [15, 45], years: [1960, 2035], subgroup_by: []}
    illegal_moves:
      - {event: birth, max_age: 45}
      - {event: birth, min_spacing: 0.6, severity: warn}
    seed_stability: {individual: true, aggregate: [ccf]}
output:
  dir: results/
  figure_format: png
"""


def _write_demo(data_dir: Path, *, n: int = 80, seeds: int = 5, rng_seed: int = 0) -> Path:
    """Write the demo artifacts + config.yaml into ``data_dir``; return the config path."""
    data_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(rng_seed)
    hazards = S.default_hazards()
    observed, persons = S.simulate_cohort(n, (1960, 1990), hazards, None, rng)
    generated = S.simulate_generated(
        observed, persons, hazards, [(0.0, 25.0), (0.0, 30.0), (0.0, 35.0)], seeds, rng
    )
    event_defs = pd.DataFrame(
        {"model_representation": [S.BIRTH_TOKEN], "event_definition": ["live birth"]}
    )
    observed.to_parquet(data_dir / "observed.parquet", engine="pyarrow", index=False)
    generated.to_parquet(data_dir / "generated.parquet", engine="pyarrow", index=False)
    persons.to_parquet(data_dir / "persons.parquet", engine="pyarrow", index=False)
    event_defs.to_csv(data_dir / "events.csv", index=False)
    config = data_dir / "config.yaml"
    config.write_text(_CONFIG)
    return config


@pytest.fixture
def demo_config(tmp_path: Path) -> Path:
    """Path to a config.yaml alongside a freshly written tiny demo dataset."""
    return _write_demo(tmp_path / "data")


@pytest.fixture
def demo_writer(tmp_path):
    """The ``_write_demo`` helper bound to ``tmp_path`` for tests needing multiple datasets."""

    def _make(**kwargs) -> Path:
        return _write_demo(tmp_path / kwargs.pop("subdir", "data"), **kwargs)

    return _make
