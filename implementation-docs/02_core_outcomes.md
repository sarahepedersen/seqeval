# 02 — Core: Slicing & Outcome Extraction

> Context: read `00_architecture.md`; depends on 01. This task builds the shared transformation
> layer: functions that turn canonical long-format sequences into the intermediate analysis
> tables that every metric (survival, fertility, ML) consumes. All functions take
> `(df, keys)` and never care whether sequences are observed or generated — this is where the
> cross-quadrant code reuse actually happens.

**Units reminder (00 §3):** every age/duration value entering or leaving these functions is
**integer days**. Year-valued config has already been converted by `config.resolve_*`. The one
exception: bin *labels* may be expressed in years for readability (see `AgeBins`), but bin
*edges* used in comparisons are days.

## Deliverables

```
src/seqeval/core/slicing.py
src/seqeval/core/outcomes.py      # incl. evaluators for the spec classes defined in 01
tests/test_slicing.py
tests/test_outcomes.py
```

## 1. `core/slicing.py`

Pure helpers on canonical frames. Signatures (all return new frames; no mutation; all age
arguments are int days):

```python
def truncate(df, keys, *, max_age: int) -> pd.DataFrame
    # drop rows with age > max_age (simulates censoring observed data at a jump-off point)

def restrict_window(df, keys, *, lo: int, hi: int) -> pd.DataFrame
    # keep rows with lo <= age < hi

def attach_persons(df, persons, columns=("birth_year", "sex")) -> pd.DataFrame
    # left-merge persons columns; raise if any person_id missing from persons

def cohort_bins(persons, *, width: int = 1, range: tuple[int, int] | None = None) -> pd.Series
    # person_id-indexed cohort label from birth_year (calendar years; no unit conversion here)

@dataclass(frozen=True)
class AgeBins:
    edges_days: np.ndarray        # int day edges
    labels: np.ndarray            # year-valued labels for output tables/plots
    @classmethod
    def from_years(cls, lo: float, hi: float, width: float) -> "AgeBins"   # uses units.py

def bin_ages(ages: pd.Series, bins: AgeBins) -> pd.Series

def calendar_year(df) -> pd.Series
    # birth_year + completed_years(age)  (units.py helper); requires attach_persons first,
    # raise otherwise. Document the approximation (birth date within year unknown).

def condition_on_count(df, keys, *, cond: Condition, anchor_age: int | None = None
                       ) -> pd.DataFrame
    # generic count-predicate filter (00 §5 rule 6): keep sequence-groups (per keys) where
    # min_count <= #occurrences(cond.event, age <= a) <= max_count, with
    # a = cond.before_age if set else anchor_age (the caller's jump-off).
    # Raise if both are None. Domain-free: parity conditioning is just event=birth.

def align_jumpoff_to_event(observed, *, event, occurrence: int) -> pd.DataFrame
    # per person: return (person_id, age_of_kth_occurrence) for use as person-specific t2 values
    # (supports "truncate at time of first birth" style backtests)
```

Note `condition_on_count` and window logic must operate per *sequence group* (`keys`), not per
person, so the same call works on generated frames where one person has many runs.

## 2. `core/outcomes.py`

Three workhorse extractors. Everything downstream (KM, CCF, ASFR, PPR, empirical probabilities)
is a small computation on one of these tables.

### 2.1 Births table

```python
def births(df, keys, *, birth_event) -> pd.DataFrame
    # one row per birth event:
    #   [*keys, order, age]   where order = 1..k via groupby(keys).cumcount()+1
    # ties in age get distinct consecutive orders; sort by (keys, age, stable)
```

### 2.2 Observation spans (exposure inputs)

```python
def observation_spans(df, keys) -> pd.DataFrame
    # one row per sequence group:
    #   [*keys, start_age, end_age]                      (int days)
    # start_age: 0 for observed sequences; age_stop for generated (detect via presence of
    #            age_stop in keys — the ONLY permitted observed/generated asymmetry, and it is
    #            derived from keys, not from a flag argument)
    # end_age: the LAST AGE IN THE DATA — max(age) within the group. One derivation path,
    #          no overrides (00 §4.2); no padding and no terminal-event concept. "No event"
    #          rows extend the span by existing.
    # MUST be computed from the full loaded frame, BEFORE any event filtering an arm applies.
```

```python
def exposure(spans, *, bins: AgeBins, persons=None, by_year=False) -> pd.DataFrame
    # expand spans into exposure per age bin (and per calendar year if by_year), vectorized:
    # overlap of [start_age, end_age) with each bin — no row-per-year Python loops; use
    # np.clip on broadcast bin edges or an interval-join on a prebuilt bin frame.
    # Returns [*keys or aggregated, age_bin, (year), person_days]   (integer person-days;
    # conversion to person-years happens inside fertility/survival metrics at the final rate
    # computation, per 00 §3)
```

### 2.3 Time-to-event table

Spec classes live in `core/specs.py` (defined in 01). They are **day-valued and
raw-token-valued**: `config.resolve_outcomes` (01) maps the year-valued, alias-valued YAML
registry onto them. `TTESpec.origin` is a nested `TTESpec` (depth ≤ 1, enforced at config
parse).

```python
def time_to_event(df, keys, spec: TTESpec, spans=None) -> pd.DataFrame
    # [*keys, duration, observed]        (duration in int days)
    # duration: age_of_target_occurrence - origin_age (drop groups where origin never occurs —
    #           conditioning, e.g. "time from 1st to 2nd birth" only for those with a 1st birth)
    # observed: True if target occurred before span end; else censored at end_age with
    #           observed=False. (Horizon capping is a frame concern — see §2.4 — not a TTE
    #           concern; TTESpec has no horizon.)
```

### 2.4 Binary outcome evaluation (for ML metrics in 04)

Spec dataclasses (`FramedOutcome`, `CountQuery`, `Frame`, `Condition`) are already defined in
`core/specs.py` (01). This task implements their evaluators. Both return the same shape so
`metrics/ml.py` (04) is agnostic to which primitive produced a table:

```python
def evaluate_framed(df, keys, spec: FramedOutcome, spans, *,
                    jumpoff: int | None = None) -> pd.DataFrame
    # [*keys, occurred: bool, evaluable: bool]
    # The framed outcome asks: does the timing quantity spec.tte land inside spec.frame?
    #   by_age A       → target's n-th occurrence at age <= A
    #   within W       → target's n-th occurrence in (jumpoff, jumpoff + W]
    #   within_origin W→ target occurs within W days of the outcome's origin occurrence
    # evaluable=False when:
    #   (a) the span does not fully cover the frame (censoring → biased negatives), OR
    #   (b) SETTLED-AT-JUMP-OFF (00 §5.1): the outcome is already determined by age <= jumpoff
    #       — e.g. first_birth by_age 35 with the first birth at 27 < t2, or (for negative
    #       determination) frames entirely inside the observed region. The answer was in the
    #       prompt; it says nothing about the model.
    # For within_origin, groups where origin occurs after span end are non-evaluable; groups
    # where origin never occurs are dropped (conditioning by origin, as in time_to_event).

def evaluate_count(df, keys, spec: CountQuery, spans, *,
                   jumpoff: int | None = None) -> pd.DataFrame
    # [*keys, occurred: bool, evaluable: bool]
    # occurred: #occurrences of spec.event with age > jumpoff inside the frame >= min_events.
    # Counting strictly after jumpoff means count queries can never be settled-at-jump-off;
    # evaluable is span-coverage only.
```

`jumpoff` is a scalar (int days) because backtesting iterates global `(age_start, age_stop)`
windows; when a `by_age` frame is evaluated on observed truth, the caller passes the same
jumpoff used for the generated side so the settled rule excludes identical persons on both
sides. Keep these functions symmetric and heavily unit-tested — they define what every
calibration number in 04 means.

## 3. Design rules

- All functions: single `groupby(keys, observed=True)` pass where possible; no `apply` with
  Python lambdas over groups when a vectorized equivalent exists (`cumcount`, `idxmin`,
  `first/last`, merges). `apply` is acceptable only with a comment justifying it.
- Every extractor validates that `keys ⊆ df.columns` and raises `ValueError` otherwise.
- Frames returned sorted by keys for reproducible downstream output.

## 4. Tests

- Run every extractor against `tests/fixtures/tiny.py` and assert exact expected values
  (durations, censor flags, person-years per bin).
- Symmetry test: for a generated frame with a single seed and window `(0, 0)`, births/TTE output
  must equal the observed-frame output for the same persons (proves observed/generated
  agnosticism).
- Evaluator logic, parametrized per frame kind: span ending inside the frame → non-evaluable;
  **settled-at-jump-off** cases for framed outcomes (positive determination: target's n-th
  occurrence before jumpoff; negative determination: frame entirely ≤ jumpoff) → non-evaluable;
  count queries with pre-jumpoff events → those events NOT counted, still evaluable.
- `condition_on_count`: parametrized (min only, max only, both, before_age vs anchor_age,
  neither anchor → raises); identical filtering on observed (person keys) and generated (run
  keys) frames.
- Property test on synthetic data: sum of `person_days` over bins == sum of (end_age −
  start_age) per span, **exactly** (integer arithmetic — a strength of the days convention);
  number of births rows == count of birth events in input.

## Acceptance criteria

- `pytest -q` green, `ruff` clean.
- A benchmark note in the PR: `births` + `time_to_event` on a 10M-row synthetic generated frame
  completes in seconds, not minutes (rough numbers fine; guard against accidental O(n²)).
