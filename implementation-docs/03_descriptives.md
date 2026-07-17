# 03 — Past/Observed: Descriptives (Survival + Fertility Metrics)

> Context: read `00_architecture.md`; depends on 01–02. This task implements the metric layer for
> time-to-event and fertility quantities, the past/observed "descriptives" arm that orchestrates
> them, and their plots. These metric functions are written once here and reused verbatim by the
> backtesting (04) and forecasting (05) arms — keep them key-agnostic per 00 §3.3.

## Deliverables

```
src/seqeval/metrics/survival.py
src/seqeval/metrics/fertility.py
src/seqeval/arms/descriptives.py
src/seqeval/viz/km.py
src/seqeval/viz/fertility.py
tests/test_survival.py
tests/test_fertility.py
tests/test_arm_descriptives.py
```

## 1. `metrics/survival.py`

Implemented natively (no lifelines). Inputs are the outcome tables from 02.

```python
def kaplan_meier(tte: pd.DataFrame, *, by: list[str] = ()) -> pd.DataFrame
    # tte: output of core.outcomes.time_to_event (durations in int days)
    # returns [*by, time, n_at_risk, n_events, survival, ci_lo, ci_hi]  — time stays in DAYS;
    # viz converts axes to years. Product-limit estimator; Greenwood variance + log-log CIs.
    # Exact integer times mean ties group cleanly with no float tolerance handling.
    # `by` lets callers stratify (cohort bins, or seed for generated data).

def life_table(births, spans, *, max_parity: int, bins: AgeBins) -> pd.DataFrame
    # conventional demographic parity life table: time spent at each parity within the
    # childbearing window. Returns [age_bin, parity, person_years, births, occ_exp_rate]
    # (occurrence/exposure; person_years = person_days / DAYS_PER_YEAR, converted here at the
    # final rate step per 00 §3). Built from core.outcomes.exposure + births — this IS the
    # "conversion between sequence format and life-table representation" the spec calls for.

def median_survival(km: pd.DataFrame, *, by=()) -> pd.DataFrame
```

## 2. `metrics/fertility.py`

All take births/spans tables plus persons where cohort/period is involved; all accept `by` keys.

```python
def ccf(births, spans, persons, *, by_cohort=True, extra_by=()) -> pd.DataFrame
    # completed cohort fertility: mean births per woman by birth cohort.
    # Include column n_women and a `complete` flag: cohort marked incomplete if its members'
    # spans end before fertile_ages upper bound (so callers can distinguish true CCF from
    # truncated means — important when the same function runs on censored/backtest data).

def asfr(births, spans, persons, *, mode: Literal["period", "cohort"], bins: AgeBins,
         extra_by=()) -> pd.DataFrame
    # births in cell / person-years in cell (person_days / DAYS_PER_YEAR at the rate step).
    # period: cells are (calendar_year, age_bin); cohort: (birth_cohort, age_bin).
    # Uses core.outcomes.exposure(by_year=mode=="period").

def ppr(births, spans, *, max_parity: int, extra_by=(),
        min_exposure_after_k: int | None = None) -> pd.DataFrame
    # parity progression ratios: of sequence-groups reaching parity k, fraction reaching k+1.
    # Denominator must exclude groups censored before a reasonable exposure —
    # min_exposure_after_k is in DAYS (resolved from a year-valued config knob if exposed);
    # document the default choice in the docstring.
    # Returns [*extra_by, parity_from, parity_to, n_at_risk, n_progressed, ppr]

def tfr(asfr_period: pd.DataFrame) -> pd.DataFrame
    # period total fertility rate = sum of period ASFRs over age bins; cheap and useful for
    # eyeballing against Human Fertility Database values.
```

`extra_by` is how 04/05 reuse these with `seed`/window keys — verify each function works when
`extra_by=["seed", "age_start", "age_stop"]`.

## 3. `arms/descriptives.py`

```python
def run(bundle: Bundle, cfg: DescriptivesConfig, out: OutputWriter) -> None
```

The config schema is already fully typed in `config.py` (01); this arm consumes
`DescriptivesConfig` as-is. Its block, per 00 §5.1 (presence = enabled; `kaplan_meier` entries
are **names from the top-level `outcomes:` registry**, resolved to `TTESpec`s by
`config.resolve_outcomes`):

```yaml
descriptives:
  kaplan_meier: [first_birth, second_birth]
  fertility:
    ccf: true
    asfr: [period, cohort]        # age_bin_width: 1 (years) by default
    ppr: {max_parity: 4}
  life_table: {max_parity: 4}
  stratify_by: [cohort]           # cohort | sex | declared covariates (validated in 01)
```

Behavior: build births/spans/tte tables once, compute each configured metric, write each result
to `results/descriptives/<name>.parquet`, and render figures. If `persons` is missing, skip
cohort/period metrics with a logged warning listing exactly what was skipped and why.

`OutputWriter` (small utility in `arms/_common.py`): resolves paths under `output.dir/<arm>/`,
saves frames + matplotlib figures, records everything written for the 06 manifest.

## 4. Viz

All axes and labels in **years** — viz is one of the three permitted conversion sites (00 §3);
apply `days_to_years` at plot time, never mutate result tables.

- `viz/km.py`: step-function KM curves with CI bands, one line per stratum; label events via
  `bundle.label`.
- `viz/fertility.py`: ASFR age profiles (line per period/cohort), CCF by cohort (line, with
  incomplete cohorts dashed), PPR bar chart by parity.
- Shared style helper (`viz/_style.py`): one place for figsize, fonts, colormap; all figures
  return `Figure` and are saved by the arm, not by viz functions.

## 5. Tests

- **Exact fixtures:** KM survival values, CCF, PPRs on `tests/fixtures/tiny.py` match
  hand-derived constants stored with the fixtures.
- **Synthetic convergence:** on a large `simulate_cohort` run, CCF within tolerance of
  `expected_ccf`; cohort ASFR summed over ages ≈ CCF for completed cohorts; KM for first birth at
  age 50 ≈ 1 - proportion ever having a child.
- **Censoring correctness:** simulate with `censor_age=30`; incomplete-cohort flag set; PPR
  denominators shrink accordingly.
- **Key-agnosticism:** every metric runs with `extra_by=GEN_KEYS` on a `simulate_generated` frame
  without error and returns one row-group per (seed, window) cell.
- Arm smoke test: run on the demo dataset from 01's `examples/`; assert expected files exist and
  are non-empty.

## Acceptance criteria

- `pytest -q` green, `ruff` clean.
- Running the demo config with only descriptives enabled produces a results folder whose figures
  are visually sane (KM monotone decreasing, ASFR hump-shaped) — include a screenshot or note in
  the PR.
