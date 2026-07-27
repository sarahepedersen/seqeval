"""Write a demo dataset + a matching config.yaml (mirroring 00 section 5.1) that ``load_all`` eats.

Run from the repo root:

    python examples/make_demo_data.py --out examples/data --n 500 --seeds 5

Produces ``observed.parquet``, ``generated.parquet``, ``persons.parquet``, ``events.csv`` and a
``config.yaml`` under ``--out``. The synthetic generator lives under ``tests/`` (it is test
scaffolding), so this script adds the repo root to ``sys.path`` to import it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tests import synthetic as S  # noqa: E402

CONFIG_TEMPLATE = """\
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

replicates:
  interval: jeffreys
  level: 0.95
  min_replicates: 5
  bootstrap: {n: 200, seed: 7}

outcomes:
  first_birth: {event: birth, n: 1}
  second_birth: {event: birth, n: 2, origin: first_birth}

arms:
  descriptives:
    kaplan_meier: [first_birth, second_birth]
    fertility:
      ccf: true
      asfr: [period, cohort]
      ppr: {max_parity: 6}
    stratify_by: [cohort]

  backtesting:
    windows: all
    conditions:
      - {name: p0, event: birth, max_count: 0}
      - {name: p1, event: birth, min_count: 1, max_count: 1}
    probability_outcomes:
      - {outcome: first_birth, by_age: 35, given: p0}
      - {outcome: second_birth, within_origin: 5, given: p1}
      - {event: birth, min_events: 1, within: 5}
      - {event: birth, min_events: 1, within: 5, given: p1}
    aggregate_targets: [ccf, asfr_cohort, ppr, km:first_birth, km:second_birth]
    min_seeds: 5

  forecasting:
    windows: all
    lexis:
      outcome: first_birth
      ages: [15, 45]
      years: [1960, 2035]
      subgroup_by: []
    illegal_moves:
      - {event: birth, max_age: 45}
      - {event: birth, min_age: 15}
      - {event: birth, min_spacing: 0.6, severity: warn}
      - {event: birth, max_count: 10, severity: warn}
    replicate_variance:
      individual: true
      aggregate: [ccf]

output:
  dir: results/
  figure_format: png
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="examples/data", help="output directory")
    parser.add_argument("--n", type=int, default=500, help="number of persons")
    parser.add_argument("--seeds", type=int, default=5, help="replicates per (person, window)")
    parser.add_argument("--rng", type=int, default=0, help="RNG seed")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.rng)

    hazards = S.default_hazards()
    observed, persons = S.simulate_cohort(args.n, (1960, 1995), hazards, None, rng)
    generated = S.simulate_generated(
        observed, persons, hazards, [(0.0, 25.0), (0.0, 30.0), (0.0, 35.0)], args.seeds, rng
    )
    event_defs = pd.DataFrame(
        {"model_representation": [S.BIRTH_TOKEN], "event_definition": ["live birth"]}
    )

    observed.to_parquet(out / "observed.parquet", engine="pyarrow", index=False)
    generated.to_parquet(out / "generated.parquet", engine="pyarrow", index=False)
    persons.to_parquet(out / "persons.parquet", engine="pyarrow", index=False)
    event_defs.to_csv(out / "events.csv", index=False)
    (out / "config.yaml").write_text(CONFIG_TEMPLATE)

    print(f"wrote demo dataset ({args.n} persons, {args.seeds} seeds) + config.yaml to {out}/")


if __name__ == "__main__":
    main()
