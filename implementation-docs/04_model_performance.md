# 04 — Past/Generated: Model Performance (ML Metrics + Backtesting)

> Context: read `00_architecture.md`; depends on 01–03 and **02b (replicate engine)**. This task
> implements the evaluation of generated sequences against observed truth: probability metrics
> built on the empirical probabilities/logits from `core/replicates.py`, and the backtesting
> harness that sweeps jump-off points, prediction horizons, and count conditioning, reusing the
> metric functions from 03 unchanged. This module contains NO probability-estimation statistics
> of its own — estimators, intervals, MC-error corrections, and bootstraps all come from 02b.

## Deliverables

```
src/seqeval/metrics/ml.py
src/seqeval/metrics/baseline.py
src/seqeval/arms/backtesting.py
src/seqeval/viz/calibration.py
src/seqeval/viz/backtest.py
src/seqeval/viz/baseline.py
tests/test_ml.py
tests/test_baseline.py
tests/test_arm_backtesting.py
```

## 1. `metrics/ml.py`

### 1.1 Probability pipeline (thin composition over 02b)

Per (window, outcome, condition): replicate-level evaluator outputs (02 §2.4) →
`replicates.replicate_summary` → `replicates.estimate_probability(spec)` → the run-level
probability table `[*RUN_KEYS, k, n, p_hat, logit_emp, var_logit, ci_lo, ci_hi]`. A logged warning fires
when median n < `spec.min_replicates`. This table (with model/window/outcome/condition columns
attached) is written as the first-class artifact `probabilities.parquet` — researchers will
regress on these, so it is an output, not an intermediate.

### 1.2 Probability metrics vs truth

```python
def join_truth(probs, obs_outcomes) -> pd.DataFrame
    # inner-join on person_id with the same evaluator's output computed on OBSERVED data (same
    # spec, same jumpoff) for the same window; keep rows evaluable on both sides.
    # returns [*RUN_KEYS, k, n, p_hat, logit_emp, y_true]

def calibration_table(joined, *, n_bins=10, strategy: Literal["uniform","quantile"]="quantile")
    # [bin, p_mean, y_rate, n] + ECE. Always computed alongside
    # replicates.null_calibration_band (02b §3) so miscalibration is only claimed where the
    # curve exits the perfect-calibration envelope at the observed replicate counts.
def roc_auc(joined) -> float          # rank-based with tie correction (p_hat lives on a 1/n
                                      # grid); record grid resolution (1/median_n) alongside
def brier(joined) -> dict             # {"raw": ..., "corrected": ...} — corrected via
                                      # replicates.brier_noise_correction (02b §3)
def log_loss(joined) -> float         # on smoothed p_hat (defined everywhere by construction;
                                      # NEVER on raw k/n)
def timing_coverage(timing_dist, obs_tte, *, q=(0.10, 0.90)) -> float
    # predictive-interval coverage of observed durations (02b §2) — the timing-calibration
    # headline; restricted to persons whose observed span covers the horizon
def subgroup_rates(gen_outcomes, obs_outcomes, *, by) -> pd.DataFrame
    # population-level comparison that needs NO seed replication (spec: bin by cohort etc.):
    # predicted vs observed event rates per subgroup with counts; by from persons covariates.
```

### 1.3 Aggregate-metric error

```python
def aggregate_error(gen_metric: pd.DataFrame, obs_metric: pd.DataFrame, *,
                    value_col: str, on: list[str], over_seeds="seed") -> pd.DataFrame
    # generic comparator for any 03 metric table (CCF by cohort, ASFR by cell, PPR by parity,
    # KM at fixed times): aligns on `on`, computes per-seed error, then per-window summary:
    # [window keys, *on, obs, gen_mean, gen_sd_over_seeds, bias, mae, rmse]
    # CIs on every cell via replicates.seed_bootstrap when spec.bootstrap_n > 0.
```

This single function is how "MSE of downstream fertility and time-to-event metrics" (spec §3.2)
is realized for every metric without metric-specific comparison code.

## 2. `arms/backtesting.py`

### 2.1 Config

The config schema is fully typed in `config.py` (01); this arm consumes `BacktestingConfig`
as-is. Its block, per 00 §5.1 — `conditions` are generic count predicates evaluated at jump-off,
and `probability_outcomes` take the two forms (framed registry reference / arm-local count
query) with frame semantics and the settled-at-jump-off rule defined in 00 §5.1 and implemented
by 02's evaluators:

```yaml
backtesting:
  windows: all                          # or explicit YEAR-valued subset
  conditions:
    - {name: p0, event: birth, max_count: 0}
    - {name: p1, event: birth, min_count: 1, max_count: 1}
  probability_outcomes:
    - {outcome: first_birth,  by_age: 35, given: p0}
    - {outcome: second_birth, within_origin: 5, given: p1}
    - {event: birth, min_events: 1, within: 5}
    - {event: birth, min_events: 1, within: 5, given: p1}
  aggregate_targets: [ccf, asfr_cohort, ppr, km:first_birth, km:second_birth]
  min_seeds: 5
```

Requested-but-absent windows are skipped with a warning naming them (00 §1: we evaluate what
inference produced; we never re-run models).

**Where conditions and the settled rule are evaluated — important.** The generated file contains
only rows with `age > age_stop`; the observed portion of each run is *not* in it. Therefore, for
each window `(t1, t2)`:

1. Evaluate every condition on the **observed** sequences with `anchor_age = t2`
   (`core.slicing.condition_on_count`) → a person_id set per condition.
2. Evaluate the settled-at-jump-off exclusion for framed outcomes on the **observed** sequences
   at `t2` → a person_id exclusion set per framed outcome.
3. Apply both sets identically to the generated runs and to the observed truth before computing
   anything. This guarantees the generated and truth sides describe the same population, and
   that conditioning reflects what the model was actually shown (the observed prefix up to t2).

### 2.2 Behavior

For each configured window:

1. Load only that window's generated runs (loader predicate pushdown from 01).
2. **Truth construction:** truncate nothing — observed sequences are the truth; evaluability
   (span coverage + settled-at-jump-off) is handled by 02's evaluators with the same `jumpoff`
   passed on both sides.
3. Resolve conditions and framed-outcome exclusions on observed data at t2 (§2.1) and apply to
   both sides.
4. For each probability outcome: run `evaluate_framed`/`evaluate_count` on generated (per
   replicate) and on observed truth; probability pipeline (§1.1) → `probabilities.parquet`;
   calibration table + null band, tie-corrected ROC-AUC, raw+corrected Brier, log-loss, and
   (for framed outcomes) timing interval coverage via `timing_distribution`, per
   (window, outcome, condition).
5. Compute configured aggregate metrics on generated (with `extra_by=GEN_KEYS` window/seed keys)
   and observed (03 functions unchanged); run `aggregate_error` with seed-bootstrap CIs.
   Aggregate targets may also be crossed with conditions (e.g. `km:second_birth` under `p1`
   answers "given parity 1 at t2, is the time to the next birth easier to predict than the
   time to a first?").
6. When `spec.convergence_curve`: run `replicates.convergence_curve` for ECE, AUC, and Brier
   per (window, outcome) → `convergence.parquet` (consumed by the report, 06).
7. Write per-window and pooled tables: `results/backtesting/{probabilities,calibration,scores,
   aggregate_error,convergence,coverage}.parquet` — every table carries the `model` column
   (OutputWriter stamps it). `coverage` records, per (window, outcome, condition) cell:
   n_persons evaluable, n excluded by condition, n excluded as settled-at-jump-off, and the
   replicate-count distribution (min/median/max n). Shrinking evaluable sets and thin replicate
   counts must be visible, never silent.

### 2.3 The headline output

A tidy frame `scores.parquet`: one row per (window, outcome, condition, metric) — this is the
table that answers the spec's motivating questions ("is a second birth easier to predict than a
first?", "how does predictability change moving the jump-off from 25 to 30?").

## 2.4 The ASFR baseline — what the scores are read *against*

A Brier score of 0.23 is not interpretable on its own. `metrics/baseline.py` supplies the reference
every score is judged against: the probability of the **same** binary outcome implied by nothing
but the observed age-specific fertility rates. Enabled by `arms.backtesting.baseline`.

**Construction (three steps, all reusing existing layers).**

1. **Schedule** — `asfr_schedule` is `fertility.asfr(mode="period")` on the observed file: births
   per person-year in each `(age_bin, calendar year)` cell. Cells resting on less than
   `min_person_years` of exposure are blanked (a rate on a handful of person-years is noise).
   Estimated **once per run**, written as `baseline_asfr_schedule.parquet`.
2. **Freeze at the jump-off** — `frozen_rates` forward-fills the schedule along calendar time only.
   Person *i* reaching the jump-off in year *yᵢ* is priced with the rates at *yᵢ*, held constant
   forward; an age bin with no rate at *yᵢ* falls back to the most recent **earlier** year that has
   one. Nothing after *yᵢ* is ever read — this is the no-hindsight guarantee, and it is what makes
   the baseline a genuine forecast-time reference rather than an in-sample oracle.
3. **Integrate over the frame** — the outcome's frame gives each person an age interval `(lo, hi]`
   (`frame_intervals`, mirroring `evaluate_framed` / `evaluate_count` semantics exactly). Summing
   frozen rate × exposure over the age bins in that interval gives a cumulative intensity `Λ`, and
   `p_base = P(N ≥ m)` for `N ~ Poisson(Λ)`, where `m` is how many further events the outcome needs
   — `min_events` for a count query; for an ordinal framed outcome, the occurrence minus the
   person's count of that event in the observed prefix (the prefix is information the model sees
   too; only the *rates* are population-average).

**The schedule is deliberately plain**: births over exposure by age and year, no parity split. Two
30-year-olds in 1990 get the same rate whether they have zero children or two. That is the point —
any lift the model shows over it is lift from knowing the individual.

**Structural requirement.** A rate for `(age a, year y)` exists only if some cohort is aged `a` in
year `y`, so the panel needs birth cohorts that overlap in calendar time. With a single cohort no
age above the jump-off has any history and every person is unpriceable. Exposure with no rate at or
before the person's jump-off year is reported per person as `unmatched_fraction`, warned about in
aggregate, and — above `max_unmatched_fraction` — dropped from **both** sides of the comparison
rather than scored against a rate-zero artifact.

**Outputs.**

- `baseline_individual.parquet` — one row per scored person: the model's `p_hat` beside `p_base`,
  `y_true`, `lambda_hat`, `n_needed`, and the exposure accounting. This is the individual-level
  comparison table.
- `baseline_scores.parquet` — one row per (window, outcome, condition, metric):
  `[model, baseline, delta, skill, n_compared, n_unpriceable]`. `skill = 1 − model/baseline` for
  loss metrics (0 = no better than the schedule, negative = worse); AUC and R² report `delta` only.
  Both columns are computed on the **same** persons, so the comparison has one denominator.

Note when reading Brier/ECE: the model's probabilities live on a `1/n_seeds` grid while the
baseline's are continuous, so with few seeds the baseline can look better calibrated purely from
granularity. `brier_corrected` is the fairer comparison.

## 3. Viz

Axes in years (`days_to_years` at plot time, per 00 §3).

- `viz/calibration.py`: reliability diagrams with the **null calibration band** (02b) shaded,
  diagonal reference, histogram of p_hat underneath (one panel per window/outcome); a
  convergence-curve panel (metric vs n_seeds) when computed.
- `viz/backtest.py`: (a) metric-vs-jump-off-age lines (x = age_stop in years, y =
  AUC/corrected-Brier/RMSE with bootstrap CI bands, one line per outcome); (b) observed vs
  generated overlay plots reusing 03 viz (KM curves with generated seed-band: median + IQR
  across seeds).
- `viz/baseline.py`: (a) reliability overlay — model and baseline curves on one axes; (b)
  per-individual scatter of model vs baseline probability, split by what actually happened
  (jittered onto the `k/n` grid, class means marked); (c) lift within equal-count baseline bins —
  observed rate vs both predictions; (d) skill vs jump-off age across windows.

## 4. Tests

(Estimator/interval/correction statistics are tested in 02b; here we test composition and the
arm.)

- Probability pipeline on constructed frames: exact (k, n) → expected p_hat/logit columns given
  a fixed ReplicateSpec; `min_replicates` warning fires; `probabilities.parquet` carries model/
  window/outcome/condition columns.
- **Perfect-model calibration:** on `simulate_generated` (same hazards as truth, ≥50 seeds,
  moderate n), calibration curve within the null band for ≈95% of bins and corrected Brier ≈
  raw Brier (corrections vanish at high n); with n=5 seeds, corrected Brier < raw Brier and
  the curve still sits inside the (now wide) null band. With `perturb(hazards, 1.5)`, the curve
  exits the band with systematic over-prediction (p_mean > y_rate in top bins) — assert the
  sign, not exact values.
- `aggregate_error` on tiny fixtures: exact bias/rmse; alignment failure (mismatched `on`
  values) raises; bootstrap CI columns present when enabled.
- Conditioning: constructed case where a condition evaluated on observed data at t2 changes the
  evaluable set identically on generated and truth sides; a run whose framed outcome is settled
  at t2 appears in `coverage` as excluded and in no metric.
- Semantics distinction test: on the same synthetic data, `{outcome: second_birth, within: 5,
  given: p1}` and `{event: birth, min_events: 1, within: 5, given: p1}` agree (with parity
  capped at 1 they are the same question), while the unconditioned count query differs from any
  framed outcome — guards against re-conflating the two primitives.
- **Baseline** (`tests/test_baseline.py`): the schedule recovers a known flat hazard; the frozen
  lookup carries rates forward and never backward; inflating every cell after the latest jump-off
  year leaves every `p_base` bit-identical (the no-hindsight guarantee); Poisson pricing matches
  the closed form `1 − exp(−rW)` for a constant hazard; frame intervals and `n_needed` match the
  evaluators' semantics including `within_origin` origin-unknown drops; a single-cohort panel is
  reported as unpriceable rather than silently priced; skill is 0 when model = baseline, positive
  when it wins, negative when it loses. Arm wiring is tested in `tests/test_arm_backtesting.py`
  (tables written, one population on both sides, graceful skip without persons).
- Arm smoke test on demo data: files exist; `scores.parquet` has one row per configured
  (window × outcome × condition × metric) with a `model` column.

## Acceptance criteria

- `pytest -q` green, `ruff` clean.
- Demo run: reliability diagram for the perfect synthetic model sits inside its null band at
  both n_seeds = 5 and 50; attach both figures in the PR (they are the framework's
  methodological pitch in two images).
