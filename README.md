# seqeval

**Evaluation library for event/time sequence models.**

`seqeval` evaluates models that predict life events in time - not just in the machine learning / accuracy sense, but also analyzing heterogeneity, seed stability, and plausibility of predictions to learn more about the world through prediction. 

Models are treated as **black boxes**: evaluation consumes only a set of standardized output artifacts (observed and generated event sequences). Any model that can emit sampled event sequences (an LLM, a microsimulation, trajectories drawn from a fitted hazard model) enters the same framework.

## The framework

Our evaluation is targeted toward studying fertility events, so specific metrics like completed cohort fertility and Lexis surfaces for age/birth order are developed with that in mind. However, the framework can be applied to any similar social science question. Evaluation is organized around a 2×2 typology of sequences — past/future × observed/generated — which maps onto three evaluation **arms**:

| | past | future |
|---|---|---|
| **observed** | **descriptives**: life tables, Kaplan–Meier, CCF / ASFR / PPR | *(impossible)* |
| **generated** | **backtesting**: ML metrics (calibration, ROC-AUC, Brier) from Monte-Carlo empirical probabilities, with varying jump-off points and count-conditioning (parity) | **forecasting**: Lexis surfaces for incomplete cohorts, illegal-move detection, seed stability |

This is enabled by two design principles:  

- **One standardized sequence format.** Observed and generated sequences share a single long-format
  `DataFrame`: `(person_id, age, event)`. Each row is an age-time-stamped event in `person_id`'s sequence. Generated rows just carry extra key columns `(seed, age_start, age_stop)`. These describe the (1) seed used to generate the sequence, (2) the bounds of the input sequence utilized in prediction (e.g., if the model sees an observed sequence from the age of 0 to 30 versus 20 to 30). A sidecar `person` file provides relevant covariate or demographic information (birth year, etc.) for subgroups in downstream analysis. This enables us to compare models solely on their outputs. 
- **Probabilities are recovered empirically.** Because models are black boxes, per-outcome
  probabilities come from replicate runs, rather than any internal model architecture: for each `(person, window)`, inference is run under multiple `seed`s and the fraction of replicates in which an outcome occurs estimates its probability — evaluating the generative system *as actually used* (temperature, top-k, and all). The reported probability is the unsmoothed replicate frequency `k/n`, so a number always reads back as "it happened in k of n runs"; Monte-Carlo error is quantified and corrected rather than ignored.

## Data model

Three parquet artifacts (see `00_architecture.md` §4 for full schemas):

- **observed** — one real sequence per person: `person_id, age, event`.
- **generated** — many per person, keyed by run: `+ seed, age_start, age_stop` (generated rows
  have `age > age_stop`).
- **persons** (optional) — `person_id, birth_year, sex, ...covariates`; required for
  cohort/period/Lexis analyses. When absent, those are skipped with a logged warning; age-only metrics still run.

One option `.csv` file:
- **labels** (optional) -- `token_id`, `label`; if model does not represent events in natural language, provides a mapping for downstream analysis with event definitions (used for plots only, not computation)

## How `seqeval` represents time 

The canonical internal unit for age and duration is **integer days** (`int32`) — exact integer
arithmetic keeps event ordering and window membership out of float-equality traps. Everything
**in config files is in years**. Conversion is confined to three places (loaders, config resolvers, viz/report); `units.py` is the single boundary.

## Install `seqeval` package 

```bash
# from the repo root
pip install -e ".[dev]"      # runtime deps + pytest/ruff
```

- Requires Python ≥ 3.11. 
- Core dependencies: pandas, pyarrow, numpy, pydantic, pandera, matplotlib,
scikit-learn, scipy, pyyaml.

## Start-up Guide

The config file is the experiment specification: an arm runs if its block is present. See the fully-commented reference [`examples/config.yaml`](examples/config.yaml).


Example quickstart with synthetic data: 
```bash
# 1. Write a demo dataset (observed/generated/persons/events) next to the reference config.
#    Observation stops in --observation-year (default 2025), so the younger cohorts have
#    unfinished sequences and the forecasting arm predicts a genuine future.
python examples/make_demo_data.py --n 10000 --seeds 10 --out examples/data

# 2. Sanity-check config + artifacts WITHOUT computing anything: prints the population,
#    the window × replicate grid.
seqeval validate examples/config.yaml

# 3. Run descriptive, backtesting, and forecasting arms → results/manifest.json + results/report.html (openable in browser).
seqeval run examples/config.yaml

# Re-build the HTML report from an existing results dir.
seqeval report results/
```

`seqeval run` validates implicitly, executes present arms in order (descriptives → backtesting →
forecasting), isolates arm failures (one failing arm logs a traceback and the rest still run, with
a nonzero exit code), and writes a reproducibility **manifest** (seqeval version, content-hash of
every input, resolved config, per-arm status/outputs, and the verbatim warning list) plus a
self-contained HTML **report**. Use `--arm <name>` to run a single arm, `--force` to overwrite an
existing results dir, and `--verbose` for DEBUG logging.

Other example scripts:

```bash
# End-to-end walkthrough on a synthetic "perfect model" — writes every figure the
# implemented layers can produce, plus an INDEX.md to browse them.
python examples/walkthrough.py --out examples/walkthrough_output

# Read-only inspector for a directory of parquet artifacts (schemas, dtypes, window × seed grid).
python examples/inspect_data.py --data path/to/data_dir
```
