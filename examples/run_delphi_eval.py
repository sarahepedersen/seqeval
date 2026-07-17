"""Run the seqeval arms on Delphi export artifacts (until the `seqeval run` CLI lands).

The `seqeval` CLI (plan 06) isn't implemented yet, so this driver wires the config loader,
the data loader, and the three arms together the same way the eventual CLI will. Point it at a
config.yaml whose `data:` paths reference the parquet files produced by
`ferteval export-seqeval` (observed/generated/persons + event_definitions).

    python examples/run_delphi_eval.py --config /path/to/<out>/seqeval/delphi_config.yaml --out results/

Each arm is run independently and failures are isolated (one arm erroring doesn't abort the
rest), so you get partial results plus a clear traceback for anything that trips.
"""

from __future__ import annotations

import argparse
import logging
import sys
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from seqeval.arms import backtesting as backtesting_arm
from seqeval.arms import descriptives as descriptives_arm
from seqeval.arms import forecasting as forecasting_arm
from seqeval.arms._common import OutputWriter
from seqeval.config import (
    load_config,
    resolve_conditions,
    resolve_outcomes,
    resolve_probability_outcomes,
    resolve_replicates,
    resolve_rules,
)
from seqeval.io.loaders import load_all

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("run_delphi_eval")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="path to the seqeval config.yaml")
    ap.add_argument("--out", default=None, help="output dir (overrides output.dir in the config)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    bundle = load_all(cfg)

    # population summary — the sanity check `seqeval validate` will eventually print
    summary = bundle.population_summary()
    log.info("population: %s", summary)
    log.info("windows in generated: %s",
             [tuple(r) for r in bundle.available_windows()[["age_start", "age_stop"]].to_numpy()])

    out_dir = Path(args.out) if args.out else Path(cfg.output.dir)
    model = cfg.model.name
    fmt = cfg.output.figure_format
    cohort_width = cfg.cohort_width

    # resolve year-valued config into day-valued specs (the resolution boundary)
    outcomes = resolve_outcomes(cfg)
    written: list[Path] = []

    def run_arm(name, fn):
        if getattr(cfg.arms, name) is None:
            log.info("arm %s not configured — skipped", name)
            return
        writer = OutputWriter(base_dir=out_dir, arm=name, model=model, figure_format=fmt)
        try:
            fn(writer)
            written.extend(writer.written)
            log.info("arm %s: wrote %d files to %s", name, len(writer.written), writer.dir)
        except Exception:  # isolate arm failures
            log.error("arm %s FAILED:\n%s", name, traceback.format_exc())

    run_arm("descriptives", lambda w: descriptives_arm.run(
        bundle, cfg.arms.descriptives, w, outcomes=outcomes, cohort_width=cohort_width))

    run_arm("backtesting", lambda w: backtesting_arm.run(
        bundle, cfg.arms.backtesting, w,
        outcomes=outcomes,
        conditions=resolve_conditions(cfg),
        prob_outcomes=resolve_probability_outcomes(cfg, outcomes),
        replicate_spec=resolve_replicates(cfg),
        cohort_width=cohort_width))

    run_arm("forecasting", lambda w: forecasting_arm.run(
        bundle, cfg.arms.forecasting, w,
        outcomes=outcomes,
        rules=resolve_rules(cfg),
        replicate_spec=resolve_replicates(cfg),
        cohort_width=cohort_width))

    print(f"\nDone. {len(written)} files under {out_dir}/")
    for p in written:
        print(" ", p.relative_to(out_dir) if out_dir in p.parents else p)


if __name__ == "__main__":
    main()
