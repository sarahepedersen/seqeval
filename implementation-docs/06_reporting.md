# 06 — Pipeline: CLI, Manifest, Report

> Context: read `00_architecture.md`; depends on 01–05. This task ties the arms into the
> config-driven pipeline: a CLI entry point, run manifest for reproducibility, and a lightweight
> report that assembles each run's tables and figures into one reviewable document.

## Deliverables

```
src/seqeval/cli.py
src/seqeval/report.py
tests/test_cli.py
tests/test_report.py
examples/config.yaml          # polished, fully-commented reference config
README.md                     # quickstart: make demo data → run → open report
```

## 1. `cli.py`

`seqeval` console script (pyproject `[project.scripts]`), three subcommands (argparse is fine):

```
seqeval validate config.yaml   # parse config, check files exist, validate schemas + cross-
                               # artifact checks WITHOUT computing anything; print a summary:
                               # model name; population composition (n persons; sex breakdown
                               # and cohort range when persons is present — the observed file
                               # DEFINES the population per 00 §5 rule 3, so this is where
                               # denominator mistakes get caught); the available window ×
                               # replicate grid (Bundle.available_windows, YEARS) with a
                               # plain-language note for windows below min_replicates ("5
                               # replicates → probability grid of 0.2; calibration bins finer
                               # than that are not meaningful"); events seen vs configured —
                               # this is the first thing a user runs
seqeval run config.yaml [--arm descriptives|backtesting|forecasting] [--force]
seqeval report results/        # (re)build report from an existing results dir
```

(`examples/make_persons.py` is a standalone dataset-specific script — deliberately NOT a
subcommand and NOT mentioned in the README.)

`run` behavior:

- `validate` implicitly first; abort with actionable errors before heavy compute.
- Execute present arms in order (descriptives → backtesting → forecasting); presence of the
  config block = enabled (00 §5). Each arm is independent — a failure in one logs a full
  traceback and continues to the next, with a nonzero final exit code.
- Logging: `logging` module, INFO progress per arm/step, `--verbose` for DEBUG; all warnings
  emitted by lower layers (skipped metrics, low seed counts, coverage gaps) must surface here.

## 2. Manifest (`report.py` or `arms/_common.py`)

`results/manifest.json`, written incrementally by `OutputWriter` (03) and finalized by the CLI:

```json
{
  "seqeval_version": "...", "timestamp_utc": "...",
  "model": {"name": "..."},
  "config_hash": "...", "config_resolved": { ... },
  "inputs": {"observed": {"path": ..., "sha256": ..., "n_rows": ...}, ...},
  "arms": {"descriptives": {"status": "ok", "outputs": ["descriptives/ccf.parquet", ...],
            "duration_s": ...}, ...},
  "warnings": ["backtesting: window (0,40) requested but absent from generated data", ...]
}
```

`OutputWriter` stamps `model.name` as a `model` column into **every** result table it writes —
this is what makes cross-model comparison `pd.concat` over tidy tables.

Hash inputs by file content (stream sha256). Two runs on identical inputs+config must produce
identical manifests except timestamp/duration.

## 3. `report.py`

Single self-contained HTML report (`results/report.html`) — no server, no JS build; embed figures
as base64 PNGs and tables as styled `DataFrame.to_html` (cap displayed rows, link the parquet).

Sections (only those with results present):

1. **Run summary** — model name, manifest highlights, data coverage (persons,
   cohorts, windows × replicates grid with evaluable counts and min_replicates flags).
2. **Descriptives** — KM curves, ASFR/CCF/PPR figures + headline numbers; note comparing CCF to
   user-supplied external reference values if provided (`report.reference_ccf` config key,
   optional).
3. **Backtesting** — the `scores` table pivoted (windows × outcomes × conditions) with bootstrap
   CIs, reliability diagrams (predicted probability grouped into decile / equal-count bins by
   default, `arms.backtesting.calibration_binning`), generated-vs-observed overlays, and the
   waiting-time scatter. 
4. **Forecasting** — combined Lexis heatmap + uncertainty map, illegal-move rates table
   (generated vs observed side by side), seed-stability summaries.
5. **Warnings** — verbatim list from manifest.

## 3b. Cross-model comparison (stub — spec only, do not build in v1)

Because every result table carries `model` and shares tidy schemas, a future
`seqeval compare results_a/ results_b/ ...` is a pure reporting exercise: concat the per-model
tables, render side-by-side scores (windows × outcomes × models), overlaid reliability diagrams,
and Lexis
difference maps. Write this section into the README roadmap; implement nothing beyond
ensuring the `model` column and manifest `model` block exist everywhere (they are acceptance
criteria of 03–05).

Keep the template simple: one Python function per section returning HTML strings; inline CSS.
This is a review artifact, not a product UI.

## 4. Tests

- CLI end-to-end on demo data (subprocess or direct main() call): `validate` passes, `run`
  creates manifest + expected outputs, exit codes correct on induced failure (e.g., missing
  file), `--arm` runs only the named arm.
- Manifest determinism: two runs, manifests equal modulo timestamp/duration fields.
- Report: builds from a completed demo results dir; contains expected section anchors; missing
  arm dirs handled gracefully.

## Acceptance criteria

- `pytest -q` green, `ruff` clean.
- Fresh-clone quickstart in README verified: `pip install -e .`, make demo data, `seqeval run
  examples/config.yaml`, open `results/report.html` — attach the demo report to the PR.
