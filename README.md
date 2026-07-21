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
  `DataFrame`; generated rows just carry extra key columns `(seed, age_start, age_stop)`. These describe the (1) seed used to generate the sequence, (2) the bounds of the input sequence utilized in prediction (e.g., if the model sees an observed sequence from the age of 0 to 30 versus 20 to 30). A sidecar `person` file provides relevant covariate or demographic information (birth year, etc.) for subgroups in downstream analysis. This enables us to compare models solely on their outputs. 
- **Probabilities are recovered empirically.** Because models are black boxes, per-outcome
  probabilities come from replicate runs, rather than any internal model architecture: for each `(person, window)`, inference is run under multiple `seed`s and the fraction of replicates in which an outcome occurs estimates its probability — evaluating the generative system *as actually used* (temperature, top-k, and all). Smoothed estimators (Jeffreys) and empirical logits are the default; Monte-Carlo error is quantified and corrected rather than ignored.

## How `seqeval` represents time 

The canonical internal unit for age and duration is **integer days** (`int32`) — exact integer
arithmetic keeps event ordering and window membership out of float-equality traps. Everything
**user-facing is in years**: all YAML config values and every figure axis. Conversion is confined
to three places (loaders, config resolvers, viz/report) and `units.py` is the single boundary.

## Install

```bash
# from the repo root
pip install -e ".[dev]"      # runtime deps + pytest/ruff
```

Requires Python ≥ 3.11. Core dependencies: pandas, pyarrow, numpy, pydantic, pandera, matplotlib,
scikit-learn, scipy, pyyaml.

## Start-up Guide

The config file **is** the experiment specification: an arm runs if and only if its block is
present (presence = enabled). See a sample config: [`examples/delphi_config.yaml`](examples/delphi_config.yaml). 

```bash
# End-to-end walkthrough on a synthetic "perfect model" — writes every figure the
# implemented layers can produce, plus an INDEX.md to browse them.
python examples/walkthrough.py --out examples/walkthrough_output

# Run all three arms against real sequences.
python examples/run_delphi_eval.py --config examples/delphi_config.yaml --out results/

# Read-only inspector for a directory of parquet artifacts (schemas, dtypes, window × seed grid).
python examples/inspect_data.py --data path/to/data_dir
```

Programmatically, every arm exposes a `run(bundle, cfg, out, *, ...resolved specs...)` entry point
consuming resolved (day-valued) objects from `config.resolve_*`; arms never touch year-valued
config numbers directly.

## Data model

Three parquet artifacts (see `00_architecture.md` §4 for full schemas):

- **observed** — one real sequence per person: `person_id, age, event`.
- **generated** — many per person, keyed by run: `+ seed, age_start, age_stop` (generated rows
  have `age > age_stop`).
- **persons** (optional) — `person_id, birth_year, sex, ...covariates`; required for
  cohort/period/Lexis analyses. Absent → those are skipped with a logged warning; age-only metrics
  still run.

`event` values are kept raw (whatever the model emitted); an optional event-definitions CSV maps
them to human labels for plots only, never during computation.
