# 00 — Architecture & Conventions

> **Purpose of this document.** This file is shared context for every implementation task in this
> repo. Paste it (or reference it) at the start of each Claude Code session alongside the
> task-specific plan file (01–06). It defines the data model, units policy, config schema, repo
> layout, and conventions that every module must follow. It is the single source of truth; if a
> task file conflicts with this document, ask before proceeding.

## 1. What we are building

`seqeval` is a **model-agnostic evaluation library + pipeline** for generative sequence models that
predict life events in time (case study: fertility). Models are treated as black boxes: evaluation
consumes only their **output artifacts** (observed and generated event sequences), never logits,
embeddings, or hazard functions. Inference decisions (windows, seeds) are made upstream; the
config here is the formal specification of *what we want to test* on those artifacts. The pipeline
never re-runs inference — if a requested window/seed is absent from the data, it is skipped with a
clear warning.

Evaluation is organized around a 2x2 typology of sequences (past/future x observed/generated):

| | past | future |
|---|---|---|
| **observed** | descriptives: life tables, Kaplan-Meier, CCF/ASFR/PPR | (impossible) |
| **generated** | ML metrics via Monte-Carlo empirical probabilities; backtesting with varying jump-off points; count-conditioned backtesting (parity as the fertility instance) | forecasting incomplete cohorts (Lexis); illegal-move detection; seed stability |

**Core reuse principle:** observed and generated sequences share one canonical long format;
generated sequences simply carry extra key columns `(seed, age_start, age_stop)`. Every metric
function operates on the canonical format plus an optional list of groupby keys. The three
evaluation "arms" are thin orchestration layers over shared outcome-extraction and metric code.

## 2. Repo layout

```
fertility-eval/
├── pyproject.toml            # package name: seqeval; python >= 3.11
├── src/seqeval/
│   ├── config.py             # FULL typed config schema (pydantic) + outcome resolution
│   ├── units.py              # the ONLY place day/year conversion constants live
│   ├── io/
│   │   ├── schema.py         # pandera schemas for all artifacts
│   │   └── loaders.py        # parquet/csv readers, unit normalization, event mapping
│   ├── core/
│   │   ├── specs.py          # day-valued spec objects (TTESpec, FramedOutcome, CountQuery,
│   │   │                     #   Condition, Rule, ReplicateSpec) — the resolver target types
│   │   ├── slicing.py        # windowing, truncation, cohort binning, condition_on_count
│   │   ├── outcomes.py       # sequences → analysis tables (births, spans, exposure, TTE,
│   │   │                     #   framed/count outcome evaluation)
│   │   └── replicates.py     # empirical probabilities from seeds, MC-error correction,
│   │                         #   predictive distributions, variance components (plan 02b)
│   ├── metrics/
│   │   ├── survival.py       # life tables, Kaplan-Meier
│   │   ├── fertility.py      # CCF, ASFR (period & cohort), PPR
│   │   ├── ml.py             # empirical probabilities, calibration, ROC-AUC, Brier, MSE
│   │   └── plausibility.py   # illegal-move rules engine, seed-stability statistics
│   ├── arms/
│   │   ├── _common.py        # OutputWriter, shared orchestration helpers
│   │   ├── descriptives.py   # past/observed
│   │   ├── backtesting.py    # past/generated
│   │   └── forecasting.py    # future/generated
│   ├── viz/
│   │   ├── _style.py, km.py, calibration.py, lexis.py, fertility.py, backtest.py
│   └── cli.py                # validate / run / report
├── tests/
│   ├── synthetic.py          # synthetic cohort generator (see 01)
│   ├── fixtures/             # tiny hand-built sequences with known metric values
│   └── test_*.py
├── examples/                 # demo data script + make_persons.py (dataset-specific utility,
│   │                         #   deliberately outside the package and undocumented in README)
└── docs/plans/               # these markdown files
```

## 3. Units policy (read this twice)

- **Canonical internal unit for age and duration is integer days** (`int32`). All columns named
  `age`, `age_start`, `age_stop`, and all durations/spans in intermediate tables are integer
  days. Exact integer arithmetic means event ordering, spacing rules, and window membership never
  hit float-equality traps.
- **All user-facing values are years**: every number in the YAML config (windows, horizons,
  spacing, age bins, age ranges) and every axis/label in figures and reports is in years.
- `units.py` is the single conversion boundary:

```python
DAYS_PER_YEAR = 365.25
def years_to_days(y: float) -> int      # round(y * DAYS_PER_YEAR)
def days_to_years(d) -> float | np.ndarray
```

  Conversions are allowed in exactly three places: (1) loaders normalizing input data
  (`age_unit: years` → days), (2) `config.resolve_*` functions turning year-valued config into
  day-valued specs for `core/`, (3) `viz/` and the report converting day-valued results to years
  for display. Metric code in `core/` and `metrics/` works in days, except where demographic
  definitions require person-**years** (ASFR, life-table rates): there, divide person-days by
  `DAYS_PER_YEAR` at the final rate computation, with a comment.
- **Calendar year** is derived, never stored: `year = birth_year + floor(days_to_years(age))`
  (completed years). This is an approximation — we do not know the birth date within the year —
  and must be documented in the docstring of the helper that computes it.

## 3b. Replicates & empirical probabilities

Models are black boxes, so **probabilities are recovered empirically from
replicate runs**: for each (person, window), inference is run under multiple `seed`s, and the
fraction of replicates in which an outcome occurs estimates its probability under the
inference-time sampling procedure (temperature, top-k, and all) — evaluating the generative
system as actually used. Conventions that follow from this:

- **`seed` is a replicate identifier**, not necessarily a controlled RNG seed. LLM API samples,
  microsimulation draws, and trajectories simulated from a fitted hazard model all qualify; the
  only contract is uniqueness within a (person, window) run. A deterministic model has one
  replicate and the machinery degrades gracefully (p̂ ∈ {0,1}, calibration → accuracy, with
  loud `min_replicates` warnings).
- **Raw replicate fractions, not smoothed estimates**, are what get reported (`p̂ = k/n`), and the
  **empirical logit** `ln((k+½)/(n−k+½))` is always emitted — raw k/n hits exact 0/1 where
  logits and log-loss are undefined, and lives on a coarse 1/n grid.
- **Estimation is strictly per run — never pooled.** A run's probability estimate uses only
  that run's replicates; no shrinkage or borrowing strength across runs or persons. The
  purpose of empirical probabilities is to measure across-replicate variance for an individual
  sequence. Cross-run computation (calibration binning, scoring) happens downstream on the per-run 
  table, never inside estimation.
- **Monte-Carlo error is quantified and corrected, never ignored**: few seeds inflate Brier by
  a computable amount (corrected score reported alongside raw), and reliability diagrams carry
  a null band showing the scatter a *perfectly calibrated* model would produce at the observed
  replicate counts. Convergence diagnostics (metric vs number of seeds) tell researchers when
  to run more inference. All of this lives in `core/replicates.py` (plan 02b); 04 and 05 are
  consumers, never reimplementers.
- Trajectory sampling is the **deliberate universal interface**: models that natively output
  probabilities (e.g. proportional hazards) enter the framework by simulating trajectories from
  them — cheap for analytic models, so replicate counts can be driven high. We knowingly trade
  analytic probabilities for architecture independence.

## 4. Data model

### 4.1 Input artifacts

All tabular inputs are parquet unless noted. `event` values are kept **raw** (int tokens or
strings, whatever the model emitted); the optional event-definitions mapping is applied only for
plot labels and reports, never during computation.

**observed sequences** — one real sequence per person:

| column | dtype (post-load) | notes |
|---|---|---|
| `person_id` | int64 or string | |
| `age` | int32 (days) | normalized from `data.age_unit` |
| `event` | category | raw model representation |

**generated sequences** — many per person, keyed by run:

| column | dtype (post-load) | notes |
|---|---|---|
| `person_id` | int64 or string | must join to observed |
| `seed` | int32 | RNG seed of the inference run |
| `age_start` | int32 (days) | start of observation window shown to model (t1) |
| `age_stop` | int32 (days) | end of observation window / jump-off point (t2) |
| `age` | int32 (days) | generated rows have age > age_stop |
| `event` | category | raw model representation |

**persons** (companion file; required for cohort/period/Lexis analyses):

| column | dtype | notes |
|---|---|---|
| `person_id` | | unique |
| `birth_year` | int16 | enables calendar time |
| `sex` | category, optional | ordinary stratification covariate; never a filter |
| ...covariates | any | only columns declared in `persons.covariates` config are loaded |

**event definitions** (optional csv) — exactly the spec's two columns, cosmetic only:

| column | notes |
|---|---|
| `model_representation` | raw token as it appears in `event` |
| `event_definition` | human-readable label for plots/reports |

### 4.2 Conventions baked into the schema

- **Right-censoring:** a sequence-group's observation ends at its maximum observed `age`, period.
  There is no terminal-event concept; deaths/emigrations, if present in the vocabulary, are
  ordinary events (users can write illegal-move rules against them if desired).
- **Right-censoring = last age in the data:** a sequence-group's observation ends at the
  maximum `age` present in its rows — one derivation path, no user-supplied override. There is
  no padding concept and no terminal-event concept.
- **Missing persons file:** anything requiring `birth_year` (CCF/ASFR by cohort or period, Lexis,
  cohort stratification/filters) is skipped with a logged warning naming exactly what was skipped
  and why; age-only metrics still run.

### 4.3 Canonical in-memory representation

No wrapper class hierarchy. The canonical object is a validated `pd.DataFrame` in the long format
above. Two module-level constants define the key columns:

```python
OBS_KEYS = ["person_id"]
GEN_KEYS = ["person_id", "seed", "age_start", "age_stop"]
```

Every function in `core/` and `metrics/` takes `df: pd.DataFrame, keys: list[str]` and is
oblivious to which artifact it is processing. This is the reuse mechanism — do not special-case
observed vs generated inside metric code.

## 5. The config

The config is the experiment specification. Design rules:

1. **Presence = enabled.** An arm runs iff its block exists. No `enabled:` flags.
2. **Minimal required surface.** The smallest valid config is `model.name` + `data.observed` +
   one entry in `events`; that runs default descriptives. No specific alias (like
   `birth`) is required by the data layer — arms validate the aliases they consume, so the
   fertility module requires `birth` only when its block is present.
3. **The observed file defines the population.** There is no population/filter section: any
   restriction (sex, cohorts, subsamples) happens upstream in data preparation, because
   population definition is an analytic choice that sets every denominator (e.g. CCF is "per
   person in the file"). `seqeval validate` prints population composition (n, sex breakdown and
   cohort range when persons is present) so denominator mistakes are loud, not silent.
4. **Raw tokens appear exactly once**, in `events:`. Everywhere else, events are referenced by
   their alias (`birth`), keeping the experiment portable across tokenizations.
5. **Two outcome primitives, strictly separated.**
   - *Timing quantities* — "when does the n-th occurrence of E happen (from some origin)?" —
     are sequence-intrinsic and context-free. They live in the top-level `outcomes:` registry
     and are reusable by every arm (KM curves, aggregate backtest targets, Lexis outcomes).
   - *Count queries* — "do ≥ m occurrences of E fall inside frame F?" — cannot exist without a
     frame, and frames (jump-off horizons) only exist inside backtesting. Count queries are
     therefore defined arm-locally, where their anchor is well-defined. The registry never
     contains windows, horizons, or jump-off references.
   Arms attach *frames* (`by_age`, `within` = from jump-off, `within_origin`) and *conditions*
   (`given:`) to registry names; frames and conditions are only legal where their anchors exist.
6. **Conditions are generic count predicates**, `{event, min_count, max_count, before_age?}`,
   evaluated on the observed part of a sequence (default anchor: the jump-off). Parity
   conditioning is the fertility instance (`event: birth`); nothing in `core/` knows the word
   parity.
7. **Windows are discovered, not declared.** `windows: all` (default) consumes every
   `(age_start, age_stop)` pair present in the generated file; an explicit list only *subsets*.
   `seqeval validate` prints the available window × seed grid.
8. **Covariate allowlist.** `stratify_by`/`subgroup_by` may reference only `cohort` or 
   columns declared in `persons.covariates`; anything else fails validation.
9. **Strict parsing.** Unknown or misspelled keys anywhere are hard errors
   (`ConfigDict(extra="forbid")` on every model). Cross-references (outcome names, condition
   names, covariates, event aliases) are validated at parse time, before any data loads.

### 5.1 Reference config (full)

```yaml
model:
  name: transformer_model_example        # REQUIRED; stamped as a `model` column into every
                                         # result table and into the manifest — cross-model
                                         # comparison is then pd.concat over tidy tables

data:
  observed: data/observed.parquet        # DEFINES the analysis population
  generated: data/generated.parquet      # omit → descriptives-only run
  persons: data/persons.parquet          # omit → cohort/period/Lexis skipped with warning
  event_definitions: data/events.csv     # omit → raw tokens used as plot labels
  age_unit: days                         # days | years; ALL values below are in YEARS

events:                                  # alias → raw token map. GENERIC: no alias is required
  birth: 01                              # by the data layer; each arm validates the aliases it
                                         # needs (e.g., fertility metrics need `birth`; a disease
                                         # model would declare {mi: 117, death: 7} and never
                                         # define birth at all).

persons:
  covariates: [education, region]        # allowlist of extra persons columns to load

replicates:                              # how seed-stochasticity becomes probability (00 §3b,
  interval: jeffreys                     #   plan 02b). jeffreys | wilson
  level: 0.95
  min_replicates: 5                      # warn below this (probability grid coarser than 0.2)

outcomes:                                # TIMING registry: sequence-intrinsic, context-free
  first_birth:  {event: birth, n: 1}     # time of 1st occurrence, from birth of person
  second_birth: {event: birth, n: 2, origin: first_birth}
                                         # duration from origin; implicitly conditioned on
                                         # origin occurring.

arms:
  descriptives:
    kaplan_meier: [first_birth, second_birth]
    fertility:
      ccf: true
      asfr: [period, cohort]             # age_bin_width: 1 (years) by default
      ppr: {max_parity: 6}
    life_table: {max_parity: 6}
    stratify_by: [cohort]

  backtesting:
    windows: all                         # every unique (age_start, age_stop) pair present in
                                         # the generated file. An explicit YEAR-valued list
                                         # [{age_start: 0, age_stop: 25}, ...] subsets
                                         # what exists.
    conditions:                          # count predicates, evaluated at jump-off on the
                                         # OBSERVED part of the sequence (age <= t2)
      - {name: p0, event: birth, max_count: 0}
      - {name: p1, event: birth, min_count: 1, max_count: 1}
    probability_outcomes:                # example binary outcomes → calibration / ROC-AUC / Brier
      # (a) framed registry references: a timing outcome + a frame

        # tests: among women childless at jump-off, can the model predict WHO enters
        # motherhood by 35? (transition into parenthood)
      - {outcome: first_birth,  by_age: 35, given: p0}

        # tests: among parity-1 women, can the model predict birth SPACING — a 2nd birth
        # within 5y of the 1st? (spec §3.2's truncation-on-parity question)
      - {outcome: second_birth, within_origin: 5, given: p1}

      # (b) count queries: frame mandatory; arm-local by nature (needs a jump-off)

        # tests: will this person have any child in the 5y after jump-off?
      - {event: birth, min_events: 1, within: 5}

        # tests: same 5y question restricted to parity-1 women — parity progression as a
        # prediction task; compare against the pooled version to see what conditioning buys
      - {event: birth, min_events: 1, within: 5, given: p1}

    aggregate_targets: [ccf, asfr_cohort, ppr, km:first_birth, km:second_birth]
    min_seeds: 5

  forecasting:
    windows: all
    lexis:
      outcome: first_birth               # a registry name (timing quantity)
      ages: [15, 45]                    
      years: [1960, 2035]
      subgroup_by: []
    illegal_moves:
      - {event: birth, max_age: 45}
      - {event: birth, min_age: 15}
      - {event: birth, min_spacing: 0.6, severity: warn}
      - {event: birth, max_count: 10, severity: warn}
    seed_stability:
      individual: true
      aggregate: [ccf]

output:
  dir: results/
  figure_format: png
```

**Frame semantics** (day-resolved by `config.resolve_*`; t2 = the run's `age_stop`):

| frame | legal on | meaning |
|---|---|---|
| `by_age: A` | framed ref, count query | outcome/events occur at age ≤ A (count queries count only events with age > t2) |
| `within: W` | framed ref, count query | occurrence in (t2, t2 + W] |
| `within_origin: W` | framed ref only | target occurs within W of the outcome's `origin` event |

**Settled-at-jump-off rule:** a run where a framed outcome is already determined by the observed
part (age ≤ t2) — e.g. `first_birth by_age 35` when the first birth happened at 27 < t2 — is
**non-evaluable**: the answer was in the prompt, so it says nothing about the model. Such runs
are excluded from probability metrics and counted in the `coverage` table (04). Count queries
only count events strictly after t2, so they never trigger this; framed references require the
check.

### 5.2 Config → core boundary

Config models store **years** exactly as written. `config.py` exposes resolvers that produce
day-valued, raw-token spec objects (`core/specs.py`) for the rest of the system:

```python
def resolve_outcomes(cfg) -> dict[str, TTESpec]                    # timing registry
def resolve_conditions(cfg_bt) -> dict[str, Condition]
def resolve_probability_outcomes(cfg_bt, outcomes) -> list[FramedOutcome | CountQuery]
def resolve_rules(cfg_fc) -> list[Rule]
def resolve_windows(spec, available) -> list[tuple[int, int]]      # days
```

Arms consume only resolved objects; they never touch year-valued config numbers directly.

## 6. Engineering conventions

- **pandas** end to end, `pyarrow` engine for IO. Target scale: **low tens of millions of rows**
  in the generated file. This is fine in pandas provided:
  - `event` is `category`; ids int64 where possible; all age/duration columns int32 days;
    seed int32.
  - No per-person Python loops. All outcome extraction is `groupby` + vectorized ops
    (`cumulative_count`, `cumulative_sum`, `transform`, merges). `apply` with Python lambdas over groups is
    allowed only with a comment justifying why no vectorized equivalent exists.
  - Loaders support column pruning and (for generated) predicate pushdown on
    `age_start`/`age_stop`/`seed` via pyarrow filters, so arms load only the runs they need.
- **Dependencies:** pandas, pyarrow, numpy, pydantic, pandera, matplotlib, scikit-learn
  (ROC/Brier only), scipy (Beta quantiles for Jeffreys intervals; replicate engine, plan 02b),
  pyyaml. **No lifelines/scikit-survival** — KM and life tables are implemented
  natively (they are simple, and the spec calls for conversion rather than integration).
- **Single derivation path:** when a quantity can be derived from the data, it is derived from
  the data — no optional user-supplied overrides or alternate inputs (spans from last age,
  windows from the generated file, calendar time from persons). Every avoided branch between
  data sources and downstream code is one fewer way for two models' evaluations to diverge.
- **Determinism:** anything stochastic takes an explicit `rng: np.random.Generator`.
- **Errors:** loaders raise `SchemaError` with actionable messages (missing column, wrong dtype,
  generated rows with `age <= age_stop`, generated person_ids absent from observed). Config
  errors name the exact YAML path, e.g. `arms.backtesting.probability_outcomes[0].outcome:
  unknown outcome 'first_brith' (note the typo); declared outcomes are: first_birth,
  second_birth`.
- **Style:** type hints everywhere, numpy-style docstrings, ruff + ruff-format defaults,
  functions small and pure; IO only in `io/`, `arms/`, `cli.py`.
- **Testing:** pytest. Every metric must have (a) a hand-computable fixture test with exact
  expected values and (b) a property/consistency test on synthetic data (see 01). Target: tests
  run < 60s.

## 7. Definition of done (applies to every task file)

1. Code implemented per the task file, following conventions above.
2. Unit tests written and passing (`pytest -q`).
3. Public functions documented; any deviation from the plan noted in the PR/commit message.
4. `ruff check` and `ruff format --check` clean.
5. No changes to canonical schemas, units policy, or config keys without updating this document.

## 8. Implementation order

| file | contents | depends on |
|---|---|---|
| 01_data_layer.md | units, schemas, loaders, full config, synthetic generator | — |
| 02_core_outcomes.md | slicing + outcome extraction | 01 |
| 02b_replicate_engine.md | empirical probabilities/logits from seeds, MC-error tools | 02 |
| 03_descriptives.md | survival + fertility metrics, descriptives arm, plots | 02 |
| 04_model_performance.md | ML metrics, backtesting arm | 02b, 03 |
| 05_forecasting.md | Lexis, illegal moves, seed stability, forecasting arm | 02b, 03 |
| 06_reporting.md | CLI, manifest, HTML report, cross-model compare stub | 03–05 |
