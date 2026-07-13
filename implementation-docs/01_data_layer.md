# 01 — Data Layer: Units, Schemas, Loaders, Full Config, Synthetic Data

> Context: read `00_architecture.md` first. This task creates the foundation every other module
> builds on: the units boundary, validated IO for the standardized artifacts, the **complete
> typed config schema** (the config was co-designed and is final — implement it exactly as in
> 00 §5.1), the config resolvers, and a synthetic data generator that gives all
> later modules test fixtures with analytically known answers.

## Deliverables

```
src/seqeval/units.py
src/seqeval/io/schema.py
src/seqeval/io/loaders.py
src/seqeval/config.py
src/seqeval/core/specs.py             # spec dataclasses (resolver targets); evaluators come in 02
tests/synthetic.py                    # importable; used by later modules' tests
tests/fixtures/tiny.py                # hand-built micro-fixtures
tests/test_units.py, test_schema.py, test_loaders.py, test_config.py, test_synthetic.py
pyproject.toml                        # package skeleton, deps per 00 §6
examples/make_demo_data.py            # writes demo dataset + matching config.yaml
examples/make_persons.py              # dataset-specific utility (see §6) — NOT part of the
                                      #   package, NOT documented in README
```

## 1. `units.py`

Exactly as 00 §3: `DAYS_PER_YEAR = 365.25`, `years_to_days` (rounds to int), `days_to_years`
(scalar and ndarray). Also `completed_years(age_days) -> int array` used for calendar-year
derivation. Nothing else converts units anywhere in the codebase.

## 2. `io/schema.py`

Pandera schemas for the four artifacts in 00 §4.1: `OBSERVED_SCHEMA`, `GENERATED_SCHEMA`,
`PERSONS_SCHEMA`, `EVENT_DEFINITIONS_SCHEMA`. These validate the **post-load canonical form**
(ages already int32 days).

Checks beyond column/dtype presence:

- `0 <= age <= years_to_days(110)`, non-null.
- generated: `age > age_stop`; `0 <= age_start <= age_stop`; seed non-null.
- persons: `person_id` unique; `birth_year` within `[1850, 2100]`.
- event_definitions: exactly the two columns `(model_representation, event_definition)`;
  `model_representation` unique.

Also defined here:

```python
OBS_KEYS = ["person_id"]
GEN_KEYS = ["person_id", "seed", "age_start", "age_stop"]

class SchemaError(ValueError): ...   # wraps pandera errors with artifact name + fix hint
```

Coercion policy: coerce dtypes where safe (int → int64/int32, event str/int → category); raise
`SchemaError` otherwise with a message naming artifact, column, and expected dtype.

## 3. `io/loaders.py`

```python
def load_observed(path, *, age_unit: Literal["days","years"], columns=None) -> pd.DataFrame
def load_generated(path, *, age_unit, windows: list[tuple[int,int]] | None = None,   # days
                   seeds: list[int] | None = None, columns=None) -> pd.DataFrame
def load_persons(path, *, covariates: list[str]) -> pd.DataFrame
def load_event_definitions(path) -> pd.DataFrame
def load_all(cfg: Config) -> Bundle
```

- pyarrow engine; `load_generated` pushes `windows`/`seeds` down as pyarrow filters so arms never
  load unneeded runs. Note: when `age_unit == "years"`, pushdown predicates must be expressed in
  the file's native unit before conversion.
- **Unit normalization:** if `age_unit == "years"`, convert `age`/`age_start`/`age_stop` via
  `years_to_days` immediately after read; if `days`, cast to int32 (fail on non-integral values
  with a clear message). Schema validation runs on the normalized frame.
- `load_persons` reads only `person_id`, `birth_year`, `sex` (if present), and the declared
  `covariates`; missing declared covariate columns → `SchemaError` listing available columns.
- **No padding concept, no side tables:** loaders return single validated frames. "No event" /
  time-marker tokens are ordinary rows kept as-is; the observation span is derived downstream
  by `core.outcomes.observation_spans` (02) as the last `age` per group — one derivation path,
  no overrides (00 §4.2). This keeps the loaders dumb.

`Bundle` — small frozen dataclass tying the pipeline together:

```python
@dataclass(frozen=True)
class Bundle:
    observed: pd.DataFrame
    generated: pd.DataFrame | None
    persons: pd.DataFrame | None
    event_defs: pd.DataFrame | None
    events: EventConfig                       # resolved alias → raw token map

    def token(self, alias: str)               # 'birth' → raw token; KeyError w/ known aliases
    def label(self, raw_token) -> str         # raw token → human label (fallback str(token))
    def require_persons(self, why: str) -> pd.DataFrame   # raises with actionable message
    def available_windows(self) -> pd.DataFrame  # unique (age_start, age_stop, n_seeds,
                                                 # n_persons) from generated; used by
                                                 # resolve_windows and `seqeval validate`
```

Cross-artifact validation in `load_all`:

- warn (not fail) if generated contains person_ids absent from observed; drop them and count.
- warn (not fail) for any declared alias whose token never appears in observed events — the
  data layer requires no specific alias (00 §5 rule 2); **arms validate the aliases they
  consume** (fertility metrics fail fast if `birth` is undeclared *and* their block is present;
  outcomes/conditions/rules already validate their event references at config parse).
- compute a `population_summary()` on the Bundle (n persons, sex breakdown and cohort range when
  persons is present) for `seqeval validate` to print — there is **no filtering**: the observed
  file defines the population (00 §5 rule 3).

## 4. `config.py` — the full schema

Implement the **entire** config from 00 §5.1 now (arms included), strictly typed pydantic models
with `extra="forbid"` on every model. There are no dict-passthrough stubs; arms in 03–05 consume
these models as-is.

Models: `ModelConfig` (`name` required — stamped as a `model` column into every result table by
`OutputWriter` and into the manifest), `DataConfig`, `EventConfig` (`dict[str, int | str]` — generic, no
required keys — a flat alias map, e.g. `{birth: 42}`), `PersonsConfig`, `ReplicatesConfig` (`estimator`, `interval`,
`level`, `min_replicates`, `bootstrap {n, seed}`, `convergence_curve` — resolved to
`ReplicateSpec`, consumed by `core/replicates.py`, plan 02b), `TimingOutcomeConfig` (`event`,
`n`, optional `origin` — **no window/frame keys exist on this model**, per 00 §5 rule 5),
`ConditionConfig` (`name`, `event`, `min_count`/`max_count` at least one, optional `before_age`),
`ProbabilityOutcomeConfig` — a discriminated union of the two forms:

- framed reference: `outcome` (registry name) + exactly one frame key of
  `by_age` | `within` | `within_origin`, optional `given`
- count query: `event` + `min_events` + exactly one frame key of `by_age` | `within`,
  optional `given` (`within_origin` illegal here — validated)

plus `DescriptivesConfig`, `BacktestingConfig` (`windows`, `conditions`, `probability_outcomes`,
`aggregate_targets`, `min_seeds`), `ForecastingConfig` (incl. `RuleConfig`, `LexisConfig`,
`SeedStabilityConfig`), `OutputConfig`, top-level `Config` with `arms: ArmsConfig` where each arm
field is `| None` (presence = enabled).

Parse-time cross-reference validation (pydantic model validators):

- every event alias used anywhere (`outcomes`, `conditions`, count queries, `illegal_moves`)
  exists in `events`.
- every `outcome:` reference and `km:<name>` aggregate target names a declared registry entry;
  `aggregate_targets` validated against `{ccf, asfr_period, asfr_cohort, ppr}` ∪ `km:*`.
- every `given:` names a declared condition; condition names unique.
- `within_origin` only on framed references whose registry outcome declares an `origin`.
- `stratify_by`/`subgroup_by` ⊆ `{cohort, sex}` ∪ `persons.covariates`.
- `origin:` in a registry outcome references another declared outcome (no chained origins —
  reject depth > 1 for now).
- error messages name the exact YAML path and enumerate valid alternatives.

## 5. `core/specs.py` — resolver target types

Frozen dataclasses, **day-valued and raw-token-valued** (evaluator functions come in 02):

```python
@dataclass(frozen=True)
class TTESpec:      # timing registry entry, resolved
    target: Any; occurrence: int = 1
    origin: "TTESpec | None" = None

@dataclass(frozen=True)
class Frame:
    kind: Literal["by_age", "within", "within_origin"]
    value: int                                  # days

@dataclass(frozen=True)
class FramedOutcome:
    name: str; tte: TTESpec; frame: Frame; given: str | None = None

@dataclass(frozen=True)
class CountQuery:
    name: str; event: Any; min_events: int; frame: Frame; given: str | None = None
    # auto-name when unnamed: e.g. "birth_ge1_within_5y"

@dataclass(frozen=True)
class Condition:
    name: str; event: Any
    min_count: int = 0; max_count: int | None = None
    before_age: int | None = None               # days; None → anchor at jump-off

@dataclass(frozen=True)
class ReplicateSpec:                             # resolved from replicates: config (plan 02b)
    estimator: Literal["jeffreys", "mle", "laplace"] = "jeffreys"
    interval: Literal["jeffreys", "wilson"] = "jeffreys"
    level: float = 0.95
    min_replicates: int = 5
    bootstrap_n: int = 200; bootstrap_seed: int = 7
    convergence_curve: bool = True

@dataclass(frozen=True)
class Rule: ...                                  # as specified in 05
```

Also define `RUN_KEYS = ["person_id", "age_start", "age_stop"]` (GEN_KEYS minus seed) alongside
the other key constants in `io/schema.py` — the replicate engine groups by run.

Public config API:

```python
def load_config(path) -> Config          # paths resolved relative to the YAML's directory
Config.hash() -> str                     # sha256 of canonical JSON dump (manifest, 06)

# year → day resolution boundary (00 §5.2):
def resolve_outcomes(cfg) -> dict[str, TTESpec]
def resolve_conditions(cfg_bt) -> dict[str, Condition]
def resolve_probability_outcomes(cfg_bt, outcomes) -> list[FramedOutcome | CountQuery]
def resolve_rules(cfg_fc) -> list[Rule]
def resolve_replicates(cfg) -> ReplicateSpec           # pure passthrough with defaults
def resolve_windows(spec, available) -> list[tuple[int, int]]
```

`resolve_windows`: `all` → every pair in `Bundle.available_windows()`; explicit list → convert
years→days, intersect with available, warn listing any requested-but-absent windows.

## 6. `examples/make_persons.py`

Dataset-specific utility (NOT part of the `seqeval` package, no CLI subcommand, no README
mention): extracts `(person_id, birth_year)` for datasets where birth year happens to be encoded
in the sequences (e.g. a calendar token like `YEAR_1987` at age 0). The framework's contract is
simply "provide a persons file"; this script is one way to satisfy it for one family of models.

```python
def persons_from_sequences(observed, *, pattern: str = r"YEAR_(\d{4})",
                           token_map: dict | None = None,
                           allow_missing: bool = False) -> pd.DataFrame
    # returns a persons-schema frame; raises listing persons with no birth-year token unless
    # allow_missing (then drops them with a count). Runnable as a script with argparse.
```

## 7. `tests/synthetic.py` — synthetic cohort generator

Simulates fertility histories from a **known piecewise-constant hazard model** so later modules
can test metrics against converged truth and test calibration against a "perfect model". Hazards
are specified in years (human-readable); all emitted frames are canonical (ages in int days).

```python
@dataclass
class HazardSpec:
    # birth hazard by (age_band, parity): dict[(lo_age_yr, hi_age_yr, parity), rate_per_year]
    rates: dict[tuple[float, float, int], float]
    max_parity: int = 6
    fertile_ages: tuple[float, float] = (15.0, 50.0)

def default_hazards() -> HazardSpec   # roughly Denmark-like: peak late 20s/early 30s,
                                      # strong parity-1 → parity-2 progression

def simulate_cohort(n, birth_years: tuple[int, int], hazards, censor_age_yr: float | None,
                    rng) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
    # (observed, persons), all schema-conformant.
    # Piecewise-exponential sampling of waiting times, vectorized; event ages rounded to
    # integer days. censor_age_yr: right-censor everyone there (None = full histories to 50).
    # Emit trailing "no event" rows for a configurable fraction of persons so the last-age
    # span convention is exercised by realistic input in end-to-end tests.

def simulate_generated(observed, persons, hazards,
                       windows_yr: list[tuple[float, float]], n_seeds, rng) -> tuple[df, df]
    # A "perfect model": for each (person, window, seed), condition on the person's true
    # parity at age_stop and simulate the future from the SAME hazards. Gold standard for
    # calibration tests: empirical probabilities across many seeds must converge to true
    # event probabilities.

def perturb(hazards, factor: float) -> HazardSpec   # scale all rates → mis-calibrated model

def expected_ccf(hazards, censor_age_yr=None) -> float
    # large-n Monte Carlo with fixed seed, cached (functools + on-disk under tests/.cache)
```

## 8. `tests/fixtures/tiny.py`

Hand-built frames of ~6 persons where every downstream metric is computable on paper: women with
0/1/2/3 births, one censored at 28 (via a trailing "no event" row at 28 — the last-age
convention). Ages in days chosen as exact year multiples plus offsets so derivations
stay readable (constants like `A25 = years_to_days(25)`). Store expected values (CCF, PPRs, KM
survival points, person-days in chosen bins) as constants next to the fixtures with derivation
comments. Later modules import these.

## 9. Tests

- units: round-trip properties; `years_to_days` integrality; vectorized paths.
- schema: valid frames pass; each violation class raises `SchemaError` naming the problem
  (parametrized); non-integral day ages rejected.
- loaders: parquet round-trip preserves dtypes; **years-unit input file normalizes to identical
  frame as days-unit input**; window predicate pushdown returns exactly the requested runs;
  trailing "no event" rows are kept as ordinary rows; spans derived in 02 read the last age (asserted end to end in 02 tests).
- config: reference YAML (checked into tests/data) parses; each cross-reference violation class
  produces its actionable error (parametrized: unknown key, unknown outcome/condition/event
  reference, frame on a registry outcome, both/neither form keys on a probability outcome,
  `within_origin` on an outcome without `origin`, `within_origin` on a count query, undeclared
  covariate, chained origin); relative path resolution; hash stability under key reordering;
  resolver round-trips (year values in YAML → expected day values on specs).
- examples/make_persons.py: extracts birth years from a constructed frame; missing-token error lists ids (kept as a light script test, not a package test).
- synthetic: determinism under fixed rng; ages within fertile range; parity ≤ max_parity; CCF of
  a large cohort within tolerance of `expected_ccf`.

## Acceptance criteria

- `pytest -q` green; `ruff` clean.
- Generating a 100k-person cohort with 5 seeds × 3 windows takes under ~2 minutes on a laptop.
- `examples/make_demo_data.py` writes a demo dataset + matching `config.yaml` (mirroring 00 §5.1)
  that `load_all` ingests cleanly end to end.
