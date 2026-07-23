# Metrics reference — how every seqeval metric is calculated

This document explains, formula by formula, how each metric in the report is computed and where the
code lives. It is written for a reader who wants to trust (or audit) the numbers, not just read them
off a chart.

Three ideas recur and are worth reading first:

- **Probabilities are recovered from seeds, not read off logits.** The models seqeval evaluates
  expose no probabilities. For each (person, window) we run the model under several random `seed`s
  and count how often the event happens. That empirical frequency, lightly smoothed, *is* the
  predicted probability. Everything downstream flows from this. (`src/seqeval/core/replicates.py`)
- **Estimation is strictly per run.** A run's probability is estimated only from its own replicates
  — never pooled or shrunk toward other people. All cross-run work (calibration binning, bootstraps)
  happens afterwards on the per-run table.
- **The seed count `n` is a resolution limit.** With `n` seeds a probability lives on a grid of
  width `1/n` and cannot be more extreme than about `1/(2n+2)` from 0 or 1. Small `n` therefore
  makes several metrics coarse; the convergence curve is how you detect it.

---

## 1. From replicates to a per-run probability

For one (person, window) let `k` = number of replicate seeds in which the event occurred and `n` =
number of seeds. Code: `replicate_summary` → `estimate_probability` in `core/replicates.py`.

**Point estimate (default: Jeffreys posterior mean)**

```
p̂ = (k + ½) / (n + 1)
```

Alternatives selectable by config: Laplace `(k+1)/(n+2)`, or MLE `k/n`. The MLE is unbiased but hits
exactly 0 and 1, where log-loss and logits blow up — which is why a smoothed estimator is the
default.

**Empirical logit (Haldane–Anscombe) and its variance (Gart–Zweifel 1967)**

```
logit_emp = ln( (k + ½) / (n − k + ½) )
var_logit = 1/(k + ½) + 1/(n − k + ½)
```

The additive-½ is the unique first-order bias-cancelling continuity correction. `var_logit` is
emitted so anyone regressing on empirical logits has the correct weights (important under ragged
`n`). When the estimator is Jeffreys, `logit_emp == logit(p̂)` exactly.

**Grid resolution.** Because `p̂` sits on a `1/n` grid, the report records `auc_grid_resolution =
1/median_n` next to AUC. This is the tie granularity, not an error bar.

---

## 2. Classification metrics (backtesting)

All of these take the **joined** table: per-run `p̂` inner-joined to the observed binary outcome
`y_true` for persons who are evaluable on both sides (`join_truth` in `metrics/ml.py`).

### 2.1 ROC-AUC — `roc_auc` (`metrics/ml.py`)

*What it answers:* if you pick a random person who had the event and a random one who didn't, how
often does the model give the first a higher `p̂`?

*How:* rank-based, tie-corrected — `sklearn.metrics.roc_auc_score(y_true, p̂)`, which is the
Mann–Whitney U statistic normalised to [0, 1]:

```
AUC = P(p̂ | event  >  p̂ | no event) + ½·P(ties)
```

0.5 = no discrimination, 1.0 = perfect ranking. Returns `NaN` if only one class is present.

*Caveat:* `p̂` lives on the coarse `1/n` grid, so ties are common. A rank-based, tie-corrected
estimator (not a threshold sweep) is mandatory, and the grid resolution `1/median_n` is reported
alongside so you know how coarse the ranking was.

### 2.2 Brier score — `brier` → `brier_raw`, `brier_corrected` (`metrics/ml.py`)

*What it answers:* mean squared error of the probability forecast.

```
brier_raw = mean( (p̂ − y_true)² )
```

*Finite-seed correction.* With only `n` seeds, `p̂` is itself a noisy estimate of the run's true
probability, and that estimation noise **inflates** the raw Brier by the per-run sampling variance.
seqeval subtracts the expected inflation (`brier_noise_correction`, `core/replicates.py`):

```
correction     = mean_over_runs[ p̂·(1 − p̂) / n ]
brier_corrected = brier_raw − correction
```

`brier_corrected` is the headline. The gap between raw and corrected shrinks to ~0 as `n` grows;
a large gap is itself a signal that you have too few seeds.

### 2.2b Plain MSE of the raw rate — `mse` (`metrics/ml.py`)

*What it answers:* the same squared-error question as Brier, but with **no estimator machinery** — it
uses the unsmoothed empirical rate `k/n` directly, so it depends only on the replicate counts and
the observed outcome (no Jeffreys smoothing, no logit, no finite-seed correction).

```
mse = mean( (k/n − y_true)² )
```

Reported alongside Brier so you can see the estimator's effect explicitly: `mse` is the Brier score
of the MLE probability, `brier_raw` is the same on the smoothed `p̂`, and `brier_corrected` further
removes the finite-seed inflation. At large `n` all three converge; at small `n` `mse` will read
slightly higher because the raw rate saturates at 0/1 where the truth rarely is.

### 2.2c R² of the raw rate — `r2` (`metrics/ml.py`)

*What it answers:* how much of the outcome's variance the raw-rate predictions explain, relative to
just predicting the base rate.

```
R² = 1 − Σ(y − k/n)² / Σ(y − ȳ)²
```

Same unsmoothed rate as MSE, rescaled by the outcome variance: **1** = perfect, **0** = no better
than always predicting the base rate `ȳ`, **negative** = worse than the base rate. `NaN` when the
outcome has no variance (`ȳ` is 0 or 1, e.g. everyone or no one had the event). Use it as the
scale-free companion to MSE.

*(Log loss was removed from the backtest score set; the `log_loss` function remains in `metrics/ml.py`
if needed.)*

### 2.3 ASFR baseline probability and skill — `metrics/baseline.py`

*What it answers:* how much of the model's score is the model, and how much is just knowing the
person's age and the calendar year.

**The baseline probability.** For person *i* with jump-off age `t2` and an outcome frame spanning
ages `(lo, hi]`:

```
yᵢ  = birth_yearᵢ + completed_years(t2)          # the person's jump-off calendar year
Λᵢ  = Σ_{age bins a in (lo, hi]}  asfr(a, yᵢ) × exposure_years(a)
pᵢ  = P(N ≥ m),   N ~ Poisson(Λᵢ)                # m = further events the outcome needs
```

`asfr(a, y)` is the period ASFR estimated from the **observed** file (births / person-years in cell
`(a, y)`), forward-filled along calendar time so an age bin with no rate in year `yᵢ` uses the most
recent earlier year. No cell after `yᵢ` is ever read, so the baseline is a forecast-time reference,
not an in-sample one. With `m = 1` this reduces to the familiar `p = 1 − exp(−Λ)`.

`m` is `min_events` for a count query; for a framed ordinal outcome ("the k-th birth") it is `k`
minus the person's count of that event in the observed prefix, floored at 1 — the prefix is
information the model is given too.

**Skill.** For loss-type metrics (Brier, MSE, ECE):

```
skill = 1 − metric_model / metric_baseline
```

**0** = no better than the age-and-year schedule, **1** = perfect, **negative** = worse than the
schedule. AUC and R² are reported as a plain difference `model − baseline` instead. The model side
of the Brier row is `brier_corrected` (the baseline is deterministic and carries no finite-seed
noise to correct); the MSE row pairs the model's raw-rate MSE against the same baseline value.

*Denominator.* Both columns are computed on exactly the persons the baseline can price **and** the
model scored. Persons more than `max_unmatched_fraction` of whose frame exposure has no rate history
at or before their jump-off year (the panel's left truncation) are dropped from both sides and
counted in `n_unpriceable`.

*Caveat when reading Brier/ECE:* the model's `p̂` lives on a `1/n_seeds` grid, the baseline's on a
continuum, so with few seeds the baseline can appear better calibrated from granularity alone.

### 2.4 Calibration table + ECE — `calibration_table`, `ece` (`metrics/ml.py`)

*Reliability table.* Bin the runs by `p̂` (default 10 **quantile** bins; uniform bins available). For
each bin record the mean prediction `p_mean`, the observed event rate `y_rate`, and the population
`n`. A perfectly calibrated model has `y_rate ≈ p_mean` in every bin (points on the diagonal).

*Expected Calibration Error* — population-weighted mean gap between prediction and reality:

```
ECE = Σ_bins ( n_bin / N ) · | p_mean − y_rate |
```

0 = perfectly calibrated. **Reliability graphs in the report are the visual form of this table.**

### 2.5 Calibration null band — `null_calibration_band` (`core/replicates.py`)

The single most important thing to understand about the reliability graphs: **the shaded band is
how much wobble off the diagonal is explainable by seed noise alone.** A curve inside the band is
*not* demonstrably miscalibrated; a model is only miscalibrated where its curve **leaves** the band.

*How the band is built* — a **parametric bootstrap under the null hypothesis of perfect
calibration.** Treat each run's own `p̂` as if it were the true probability, then for `n_sims`
repetitions:

1. draw a noisy re-estimate `k* ~ Binomial(n, p̂)` and recompute a predicted probability from it;
2. draw an independent realised outcome `y* ~ Bernoulli(p̂)`;
3. bin by the re-estimated probability and record each bin's observed frequency.

The 2.5th–97.5th percentile envelope of those frequencies (per bin) is the band. Both noise sources
scale with `n`, so **more seeds ⇒ tighter band.**

*Limitation:* at very small `n` (≈5) `p̂` is a coarse, shrunk view of the truth, so even a perfectly
calibrated model's curve can exit the band. At that point the diagram is unreliable and you should
use the convergence curve (§3.2) to decide whether to generate more seeds.

### 2.6 Timing coverage — `timing_coverage` (`metrics/ml.py`)

For outcomes with a timing component: the fraction of persons whose *observed* event time falls
inside the model's predictive interval `(q10, q90)` of the event age. Restricted to persons whose
event actually occurred. Well-calibrated timing ⇒ coverage ≈ `q90 − q10 = 0.80`.

---

## 3. Seed stability

Seed stability asks a different question from the metrics above: **not "is the model right?" but "is
the model's answer stable across seeds, or an artefact of the particular random draws?"** It splits
into an individual level and an aggregate level (`arms/forecasting.py`, `core/replicates.py`).

### 3.1 Individual seed stability — `_seed_stability_individual` (`arms/forecasting.py`)

Per (person, window), three dispersion measures, one per outcome type. All are computed strictly
from that person's own replicates.

- **Occurrence — `disagreement`.** How much the seeds disagree on the yes/no question "does the
  target event occur within the horizon?" This is the Bernoulli variance of the smoothed occurrence
  probability:

  ```
  disagreement = p̂ · (1 − p̂)          # ranges [0, 0.25]
  ```

  0 = every seed agrees (fully stable); 0.25 = a coin flip at `p̂ = 0.5` (maximally unstable).

- **Timing — `timing_spread`.** Inter-quantile width of the predicted age at first occurrence across
  seeds: `q90 − q10`. Wider = the seeds disagree more about *when*.

- **Count — `count_var`.** Predictive variance across seeds of the completed event count for that
  person.

These are plug-in dispersions, **not** resampling procedures. In the report the individual table is
down-sampled to five randomly chosen persons (the full table is in the parquet).

### 3.2 Aggregate seed stability

Here we resample the **seed dimension** across runs to put uncertainty on a whole-cohort number.

- **Seed bootstrap — `seed_bootstrap` (`core/replicates.py`).** Resample seed labels *with
  replacement, within each window* (so window structure is preserved), recompute the aggregate
  statistic on each draw, and take percentile confidence intervals. Outcome evaluation is done once
  per replicate and the *rows* are resampled — the model is never re-run inside the loop. This is
  the source of the `(lo, hi)` interval shown next to each backtest metric. It captures Monte-Carlo
  (replicate) uncertainty only, **not** population sampling uncertainty.

- **Convergence curve — `convergence_curve` (`core/replicates.py`).** For each seed count `m` from 2
  up to `n`, subsample `m` seeds *without replacement* `n_rep` times, recompute the metric, and
  report its mean ± standard deviation across those subsamples. Reading it: where the curve flattens
  **and** its error bars go tight is the seed count at which the estimate has stabilised. If it is
  still sloping or the bars are still fat at your largest `m`, the reported metric is coarse and the
  action is *generate more seeds*. This is effectively a replicate-count power analysis.

> Note on the percentile interval: because the seed bootstrap resamples with replacement, the point
> estimate (computed on the full seed set) can occasionally sit just outside its own CI at small `n`.
> That is a known property of the percentile bootstrap, not a bug.

---

## 4. Aggregate-target error (backtesting) — `aggregate_error` (`metrics/ml.py`)

For demographic aggregates (CCF, ASFR, PPR, KM survival at fixed ages) the model's generated cohort
is compared to the observed cohort cell by cell:

```
bias = metric_generated − metric_observed
```

with seed-bootstrap percentile CIs on the bias (resample seeds, recompute the aggregate). The demo
report does not surface this table by default, but it is written to `aggregate_error.parquet`.

---

## 5. Coverage accounting (the shrinking denominator)

Every backtest score has a denominator that shrinks as the observation window changes. The coverage
table (`_coverage_row` in `arms/backtesting.py`) breaks the condition population into:

- **`n_condition`** — persons matching the condition at the jump-off.
- **`n_evaluable`** — persons who actually contribute a score (event resolvable within the frame).
- **`n_settled`** — persons whose framed outcome was already determined in the observed prefix at the
  jump-off (answer fixed, excluded).
- **`n_uncovered`** — the residual, `max(n_condition − n_evaluable − n_settled, 0)`: persons whose
  sequence ran out of observation before the frame closed. Because of the `max(·, 0)` floor, a
  constant `0` here just means everyone was either evaluable or settled.

A cell with `n_evaluable = 0` produces no score and is flagged in the report.

---

## File map

| Concern | Code |
|---|---|
| Per-run probability, estimators, logit/variance | `core/replicates.py` (`replicate_summary`, `estimate_probability`) |
| AUC, Brier, log-loss, ECE, calibration table, timing coverage | `metrics/ml.py` |
| ASFR baseline probability + skill | `metrics/baseline.py` |
| Brier finite-seed correction | `core/replicates.py` (`brier_noise_correction`) |
| Calibration null band | `core/replicates.py` (`null_calibration_band`) |
| Aggregate CIs / convergence | `core/replicates.py` (`seed_bootstrap`, `convergence_curve`) |
| Individual seed stability | `arms/forecasting.py` (`_seed_stability_individual`) |
| Coverage accounting | `arms/backtesting.py` (`_coverage_row`) |
