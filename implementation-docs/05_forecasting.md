# 05 — Future/Generated: Forecasting (Lexis, Illegal Moves, Seed Stability)

> Context: read `00_architecture.md`; depends on 01–03 and 02b (independent of 04; can be built in
> parallel with it). This arm evaluates generated futures with no ground truth: completing the
> Lexis surface for incomplete cohorts, screening output for demographically impossible or
> implausible "illegal moves", and quantifying seed stability of individual trajectories.

## Deliverables

```
src/seqeval/metrics/plausibility.py
src/seqeval/arms/forecasting.py
src/seqeval/viz/lexis.py
tests/test_plausibility.py
tests/test_lexis.py
tests/test_arm_forecasting.py
```

## 1. Lexis surfaces (`viz/lexis.py` + a builder in the arm)

```python
def lexis_surface(births, spans, persons, *, occurrence=1, bins: AgeBins, year_range,
                  extra_by=()) -> pd.DataFrame
    # occurrence-specific intensity m(x) per (calendar_year, age) cell:
    # numerator: k-th births in cell; denominator: exposure in cell (02's exposure() with
    # by_year=True), converted person_days → person_years at the rate step (00 §3).
    # Returns tidy [year, age_bin, *extra_by, rate, n_events, person_years].
    # Lives in metrics/fertility.py or arms/forecasting.py — implementer's choice, but it must
    # reuse births/exposure, not re-derive them.
```

**Completing the surface:** compute the observed surface from observed sequences, the forecasted
surface from generated sequences (rows with age > age_stop supply the missing upper-right
triangle), and produce:

- `lexis_observed.parquet`, `lexis_forecast.parquet` (per seed), `lexis_pooled.parquet`,
  `lexis_combined.parquet` (observed cells + pooled forecast cells, with a
  `source ∈ {observed, forecast}` column).

  The forecast side follows the same per-seed-population rule as the time-dependent backtest
  families: each seed is its own synthetic population (`lexis_forecast.parquet`), and the surface
  that gets drawn is one estimate over **all N×K trajectories at once** (`lexis_pooled.parquet`,
  built via `arms/_common.pool_seeds`) — not a per-cell summary across seeds. Each cell carries its
  Poisson `rate_var` and the design-effect interval from `metrics/pooling.py`
  (see `metrics_reference.md` §4.1). `_combine_surfaces` therefore aggregates nothing; it tags the
  pooled cells and stacks them under the observed ones.

`viz/lexis.py`:

```python
def plot_lexis(surface, *, value="rate", mark_forecast=True) -> Figure
    # heatmap, x=year, y=age; forecast region delineated (hatching or contour line at the
    # observed/forecast boundary). Optional cohort diagonals.
```

Subgroup Lexis: honor `extra_by` from persons covariates (education, region) when configured.

## 2. `metrics/plausibility.py` — illegal-move rules engine

Rules are **data, not code** (extensibility beyond fertility). Config values in years (typed as
`RuleConfig` in 01's `config.py`).

The subject of a rule is written **either** as an `event` alias — every occurrence of that token —
**or** as an `outcome` name from the top-level registry, which pins one ordinal occurrence. Exactly
one of the two. `not_before`/`not_after` take either kind: an alias anchors on the token's *first*
occurrence, an outcome name on the occurrence it names.

```yaml
forecasting:
  illegal_moves:
    # token form: constrains every occurrence
    - {name: birth_after_50,  event: birth, max_age: 50}
    - {name: birth_before_12, event: birth, min_age: 12}
    - {name: implausible_spacing, event: birth, min_spacing: 0.6, severity: warn}
    - {name: parity_cap, event: birth, max_count: 15, severity: warn}
    - {name: birth_after_death, event: birth, not_after: death}   # only valid if 'death' is a
                                                                 # declared event alias
    # outcome form: constrains one occurrence, which is what an ordering rule usually wants
    - {name: divorce_before_marriage, outcome: first_divorce, not_before: first_marriage}
    - {name: second_birth_unmarried, outcome: second_birth, not_before: first_marriage}
```

`not_before` flags the divorce itself, **including when no marriage exists at all**; `not_after`
only constrains groups where the anchor exists. That asymmetry is deliberate and unchanged.

An outcome's `origin` is ignored here: a rule constrains where an occurrence sits in the sequence,
not a duration measured from something else. `min_spacing` and `max_count` describe the whole stream,
so they require `event` and are rejected with `outcome`.

`config.resolve_rules` (01) converts to the day-valued, raw-token `Rule` in `core/specs.py`:

```python
@dataclass(frozen=True)
class Rule:
    name: str; event: Any                       # raw token
    occurrence: int | None = None               # None = every occurrence (the `event:` form);
                                                # an int pins one (the `outcome:` form)
    min_age: int | None = None; max_age: int | None = None    # days
    min_spacing: int | None = None                            # days
    not_after: Any | None = None                              # raw token
    not_before: Any | None = None                             # raw token
    not_after_occurrence: int = 1                # which occurrence of the anchor to measure against
    not_before_occurrence: int = 1
    max_count: int | None = None
    severity: Literal["illegal", "warn"] = "illegal"

def check_rules(df, keys, rules: list[Rule]) -> pd.DataFrame
    # row-level violations: [*keys, age, event, rule, severity]
def violation_rates(violations, df, keys, *, by=("seed",)) -> pd.DataFrame
    # rates per rule: violations / sequence-groups and / events, by seed (and window if present)
    # → the "rate of illegal moves" headline number per spec §3.3
```

Run `check_rules` on observed data too and report it: violations there indicate data problems or
mis-specified rules, which contextualizes model rates (spec's "isolate model learning from data
artifacts" logic).

## 3. Seed stability

Reframed as **views over the replicate engine (02b)** — 05 contains no statistics code of its
own:

```python
def seed_stability(births, spans, tte_tables, *, keys=GEN_KEYS,
                   level: Literal["individual","aggregate"], spec: ReplicateSpec)
    # individual, per (person_id, window):
    #   (a) occurrence disagreement: for "any target event in forecast horizon", this IS the
    #       Bernoulli variance p_hat(1−p_hat) from replicates.estimate_probability — report it
    #       as such, on p_hat = k/n;
    #   (b) timing dispersion: quantile spread (q90−q10) of age at first target occurrence,
    #       from replicates.timing_distribution;
    #   (c) count dispersion: predictive variance of completed event count, from
    #       replicates.count_distribution.
    #   Returns per-run table + summary distribution stats.
    # aggregate: analytic variance components on CCF cells via replicates.mean_variance_components
    #   metrics with extra_by=["seed"] — connects to spec's cross-model comparison of forecast
    #   variance; the Lexis IQR heatmap (§1) is the same computation surfaced spatially.
```

Baseline for interpretation: compute the same dispersion statistics on `simulate_generated`
synthetic data (where seed-to-seed variation is pure aleatoric noise from known hazards) and note
in docs that observed-cohort heterogeneity gives a reference band — the arm just reports numbers;
interpretation guidance goes in the report (06).

## 4. `arms/forecasting.py`

The config schema is already fully typed in `config.py` (01); this arm consumes
`ForecastingConfig` as-is. Its block, per 00 §5.1 (`lexis.outcome` is a **name from the
`outcomes:` registry**):

```yaml
forecasting:
  windows: all               # same resolve_windows semantics as backtesting (04)
  lexis:
    outcome: first_birth
    ages: [12, 55]             # years
    years: [1960, 2035]        # calendar years
    subgroup_by: []            # from declared persons.covariates (validated in 01)
  illegal_moves: [ ...rules as above... ]
  # one block, or a list of them; every stem a block writes ends in `_<name>`
  replicate_variance:
    - {individual: true, aggregate: [ccf], subgroup_by: [cohort]}
    - {individual: true, event: marriage, aggregate: []}   # `event` is what the spread counts
```

Behavior: uses ALL generated windows by default (forecasting wants the longest futures; document
that users typically point this arm at conditions-at-birth or late-jump-off runs via a
`windows:` filter identical to 04's). Writes
`results/forecasting/{lexis_*, violations, violation_rates, seed_stability_*}.parquet` + figures.

## 5. Tests

- Lexis: on fully-observed synthetic data, surface cells match direct computation on a couple of
  hand-checked cells; combined surface has `forecast` source only in cells beyond observed spans.
- Rules engine: parametrized unit tests per rule type on constructed frames (violations found,
  clean data passes; `not_after` flags event X occurring after the anchor occurrence of Y within the
  same sequence group; `not_before` flags X before it *and* X with that occurrence absent entirely;
  spacing computed in exact integer days; `occurrence` narrows the subject to one ordinal, ordered by
  age rather than row order).
- `violation_rates` denominators correct (per-group vs per-event).
- Seed stability: perfect-model synthetic data with many seeds → occurrence disagreement (p_hat(1−p_hat)) matches
  binomial expectation from empirical p (loose tolerance).
- Arm smoke test on demo data; figures written; Lexis heatmap renders with forecast region marked.

## Acceptance criteria

- `pytest -q` green, `ruff` clean.
- Demo run produces a combined Lexis heatmap where the synthetic model's forecast region visually
  continues observed trends (attach figure in PR).
