"""06 — the ``seqeval`` console script: ``validate`` / ``run`` / ``report``.

This ties the config-driven pipeline together. It owns no metric logic: it loads the config and
data (01), resolves year-valued config into day-valued specs (config resolvers), drives the three
arms (03–05) through their shared :class:`~seqeval.arms._common.OutputWriter`, and hands the
collected run metadata to :mod:`seqeval.report` for the manifest and HTML report.

Subcommands::

    seqeval validate config.yaml            # parse + validate + summarize, compute nothing
    seqeval run config.yaml [--arm ...] [--force] [--verbose]
    seqeval report results/                 # (re)build report.html from an existing results dir

``run`` behavior (00 section 5): ``validate`` runs implicitly first and aborts before heavy compute
on config/schema errors; present arm blocks are executed in order (descriptives -> backtesting ->
forecasting); each arm is isolated — a failure logs a full traceback, records ``status: "failed"``,
and continues, and the process exits nonzero. Every warning emitted by the lower layers (skipped
metrics, thin replicate counts, coverage gaps) is captured and written into the manifest.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: the CLI never opens a display

from seqeval import redraw, report
from seqeval.arms import backtesting as backtesting_arm
from seqeval.arms import descriptives as descriptives_arm
from seqeval.arms import forecasting as forecasting_arm
from seqeval.arms._common import OutputWriter
from seqeval.config import (
    Config,
    load_config,
    resolve_conditions,
    resolve_fertility_grid,
    resolve_outcomes,
    resolve_probability_outcomes,
    resolve_replicates,
    resolve_rules,
)
from seqeval.io.loaders import Bundle, load_all
from seqeval.io.schema import SchemaError

logger = logging.getLogger("seqeval")


# =================================================================================================
# warning capture — surface every lower-layer warning into the manifest (06 section 1)
# =================================================================================================
class _WarningCollector(logging.Handler):
    """Collects ``WARNING``+ records emitted on the ``seqeval`` logger during a run."""

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        if record.levelno >= logging.WARNING:
            self.messages.append(f"{record.name}: {record.getMessage()}")


# =================================================================================================
# validate — parse, load, cross-check, summarize; compute nothing
# =================================================================================================
def _summarize(bundle: Bundle, cfg: Config) -> list[str]:
    """Human-readable validation summary (00 section 5 rule 3 — catch denominator mistakes here)."""
    cov = report.coverage_block(bundle, cfg)
    lines = [
        f"model: {cfg.model.name}",
        f"population: {cov['n_persons']} persons"
        + (
            f", cohorts {cov['cohort_range'][0]}–{cov['cohort_range'][1]}"
            if cov["cohort_range"]
            else " (no persons file — cohort/period/Lexis will be skipped)"
        ),
    ]
    if cov["sex_breakdown"]:
        lines.append(f"  sex: {cov['sex_breakdown']}")

    if cov["windows"]:
        lines.append("windows (years) × replicates:")
        for w in cov["windows"]:
            lines.append(
                f"  ({w['age_start']}, {w['age_stop']}): {w['n_seeds']} seeds, "
                f"{w['n_persons']} persons"
            )
    else:
        lines.append("windows: none (no generated file — descriptives only)")

    seen = set(bundle.observed["event"].astype(str).unique())
    configured = {alias: str(tok) for alias, tok in cfg.events.items()}
    present = [a for a, t in configured.items() if t in seen]
    absent = [a for a, t in configured.items() if t not in seen]
    lines.append(
        f"events: configured {sorted(configured)}; seen in observed {sorted(present)}"
        + (f"; NEVER seen {sorted(absent)}" if absent else "")
    )
    return lines


def cmd_validate(args: argparse.Namespace) -> int:
    """``seqeval validate`` — parse, load, cross-validate, print a summary. No computation."""
    cfg = load_config(args.config)
    bundle = load_all(cfg)
    for line in _summarize(bundle, cfg):
        print(line)
    print("OK: config and artifacts valid.")
    return 0


# =================================================================================================
# run — execute present arms, write manifest + report
# =================================================================================================
def _arm_runners(bundle: Bundle, cfg: Config):
    """Map arm name -> (config block, thunk taking an OutputWriter). Wiring mirrors the driver."""
    outcomes = resolve_outcomes(cfg)
    cohort_width = cfg.cohort_width
    return {
        "descriptives": (
            cfg.arms.descriptives,
            lambda w: descriptives_arm.run(
                bundle, cfg.arms.descriptives, w, outcomes=outcomes, cohort_width=cohort_width
            ),
        ),
        "backtesting": (
            cfg.arms.backtesting,
            lambda w: backtesting_arm.run(
                bundle,
                cfg.arms.backtesting,
                w,
                outcomes=outcomes,
                conditions=resolve_conditions(cfg),
                prob_outcomes=resolve_probability_outcomes(cfg, outcomes),
                replicate_spec=resolve_replicates(cfg),
                cohort_width=cohort_width,
                fertility_grid=resolve_fertility_grid(cfg),
            ),
        ),
        "forecasting": (
            cfg.arms.forecasting,
            lambda w: forecasting_arm.run(
                bundle,
                cfg.arms.forecasting,
                w,
                outcomes=outcomes,
                rules=resolve_rules(cfg),
                replicate_spec=resolve_replicates(cfg),
                cohort_width=cohort_width,
            ),
        ),
    }


def cmd_run(args: argparse.Namespace) -> int:
    """``seqeval run`` — validate implicitly, run present arms in order, write manifest + report."""
    cfg = load_config(args.config)
    out_dir = Path(args.out) if args.out else Path(cfg.output.dir)

    if (out_dir / report.MANIFEST_NAME).exists() and not args.force:
        logger.error(
            "results dir %s already contains a manifest; pass --force to overwrite", out_dir
        )
        return 2

    # validate implicitly, before any heavy compute (00 section 5 / 06 §1).
    bundle = load_all(cfg)
    for line in _summarize(bundle, cfg):
        logger.info("%s", line)

    collector = _WarningCollector()
    collector.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(collector)

    only = set(args.arm) if args.arm else None
    runners = _arm_runners(bundle, cfg)
    arm_results: list[dict] = []
    failed = False

    try:
        for name in report.ARM_ORDER:
            cfg_block, thunk = runners[name]
            if cfg_block is None:
                logger.info("arm %s: not configured — skipped", name)
                continue
            if only is not None and name not in only:
                logger.info("arm %s: not selected by --arm — skipped", name)
                continue

            writer = OutputWriter(
                base_dir=out_dir,
                arm=name,
                model=cfg.model.name,
                figure_format=cfg.output.figure_format,
                individual_level=cfg.output.individual_level,
                min_cell=cfg.output.min_cell,
            )
            t0 = time.perf_counter()
            try:
                logger.info("arm %s: running…", name)
                thunk(writer)
                status = "ok"
            except Exception:  # isolate arm failures (06 §1): log, record, continue
                logger.error("arm %s FAILED:\n%s", name, traceback.format_exc())
                status = "failed"
                failed = True
            outputs = [str(p.relative_to(out_dir)) for p in writer.written]
            arm_results.append(
                {
                    "name": name,
                    "status": status,
                    "outputs": outputs,
                    "withheld": sorted(writer.skipped),
                    "duration_s": round(time.perf_counter() - t0, 3),
                }
            )
            logger.info("arm %s: %s, %d file(s)", name, status, len(outputs))
    finally:
        logger.removeHandler(collector)

    coverage = report.coverage_block(bundle, cfg)
    manifest = report.build_manifest(
        cfg=cfg, coverage=coverage, arm_results=arm_results, warnings=collector.messages
    )
    report.write_manifest(out_dir, manifest)
    report.build_report(out_dir)
    logger.info("done: results under %s (manifest.json, report.html)", out_dir)
    return 1 if failed else 0


# =================================================================================================
# report — rebuild the HTML from an existing results dir
# =================================================================================================
def cmd_report(args: argparse.Namespace) -> int:
    """``seqeval report`` — (re)build ``report.html`` from an existing results directory.

    ``--redraw`` rebuilds the figures from the parquets first. The report embeds the PNGs it finds
    on disk rather than drawing them, so a results directory exported without its figures needs
    this to produce anything but tables.
    """
    results_dir = Path(args.results)
    if not results_dir.is_dir():
        logger.error("not a directory: %s", results_dir)
        return 2
    if args.redraw:
        try:
            figures = redraw.redraw(results_dir, event_definitions=args.events)
        except FileNotFoundError as exc:
            logger.error("%s", exc)
            return 2
        logger.info("redraw: %d figure(s) rebuilt from the parquets", len(figures))
    path = report.build_report(results_dir)
    print(f"wrote {path}")
    return 0


# =================================================================================================
# argparse / entry point
# =================================================================================================
def build_parser() -> argparse.ArgumentParser:
    """Construct the ``seqeval`` argument parser."""
    parser = argparse.ArgumentParser(prog="seqeval", description=__doc__)
    parser.add_argument("--verbose", action="store_true", help="DEBUG-level logging")
    sub = parser.add_subparsers(dest="command", required=True)

    p_validate = sub.add_parser("validate", help="parse + validate + summarize; compute nothing")
    p_validate.add_argument("config", help="path to the config.yaml")
    p_validate.set_defaults(func=cmd_validate)

    p_run = sub.add_parser("run", help="run present arms, write manifest + report")
    p_run.add_argument("config", help="path to the config.yaml")
    p_run.add_argument("--out", default=None, help="output dir (overrides output.dir in config)")
    p_run.add_argument(
        "--arm",
        action="append",
        choices=report.ARM_ORDER,
        help="run only the named arm (repeatable); default runs every present arm",
    )
    p_run.add_argument(
        "--force", action="store_true", help="overwrite an existing results dir (has a manifest)"
    )
    p_run.set_defaults(func=cmd_run)

    p_report = sub.add_parser("report", help="(re)build report.html from a results dir")
    p_report.add_argument("results", help="path to an existing results directory")
    p_report.add_argument(
        "--redraw",
        action="store_true",
        help="rebuild every figure from the parquets before assembling the report — needed when "
        "the results dir was exported without its PNGs",
    )
    p_report.add_argument(
        "--events",
        default=None,
        help="path to the events.csv used for the run; only affects figure titles, which fall "
        "back to raw event tokens without it",
    )
    p_report.set_defaults(func=cmd_report)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Console-script entry point. Returns a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    try:
        return args.func(args)
    except (SchemaError, ValueError, FileNotFoundError) as exc:
        # actionable, expected failures (bad config path, schema mismatch, cross-ref error)
        logger.error("%s", exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
