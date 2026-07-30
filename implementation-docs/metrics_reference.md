# Metrics reference — how every seqeval metric is calculated

This document explains, formula by formula, how each metric in the report is computed and where the
code lives. It is written for a reader who wants to trust (or audit) the numbers, not just read them
off a chart.

Three ideas recur and are worth reading first:

- **Probabilities are recovered from seeds, not read off logits.** The models seqeval evaluates
  expose no probabilities. For each (person, window) we run the model under several random `seed`s
  and count how often the event happens. That empirical frequency *is* the
  predicted probability. Everything downstream flows from this. (`src/seqeval/core/replicates.py`)
- **Estimation is strictly per run.** A run's probability is estimated only from its own replicates
  — never pooled or shrunk toward other people. All cross-run work (calibration binning, scoring)
  happens afterwards on the per-run table.
- **The seed count `n` is a resolution limit.** With `n` seeds a probability lives on a grid of
  width `1/n` and cannot be more extreme than about `1/(2n+2)` from 0 or 1. Small `n` therefore
  makes several metrics coarse; a p_hat that shifts as seeds accumulate is how you detect it.

---

## 1. From replicates to a per-run probability

For one (person, window) let `k` = number of replicate seeds in which the event occurred and `n` =
number of seeds. Code: `replicate_summary` → `estimate_probability` in `core/replicates.py`.

**Point estimate (MLE, unsmoothed)**

```
p̂ = k / n
```

`p̂` is the replicate frequency and nothing else, so it reads straight back as "the event happened
in `k` of `n` runs". It is unbiased, and it reaches exactly 0 and 1 — where `logit(p̂)` is undefined
and log-loss is finite only because it clips. That boundary behaviour is accepted deliberately: no
smoothing is applied anywhere to the point estimate.

**Empirical logit (Haldane–Anscombe) and its variance (Gart–Zweifel 1967)**

```
logit_emp = ln( (k + ½) / (n − k + ½) )
var_logit = 1/(k + ½) + 1/(n − k + ½)
```

The additive-½ is the unique first-order bias-cancelling continuity correction. `var_logit` is
emitted so anyone regressing on empirical logits has the correct weights (important under ragged
`n`). It is the one smoothed quantity here — the ½ is what makes a log-odds finite at `k = 0` and
`k = n` at all — so `logit_emp != logit(p̂)`.

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

*What it answers:* the same squared-error question as Brier, computed straight from the counts —
no probability machinery in the path, no finite-seed correction.

```
mse = mean( (k/n − y_true)² )
```

Since `p̂` is itself `k/n`, `mse == brier_raw` exactly; `brier_corrected` is the same value with the
finite-seed inflation removed, so the gap between them is the whole of the correction. `mse` is kept
as the self-contained form and as the numerator `r2` rescales.

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
watch whether p_hat still shifts as seeds accumulate to decide whether to generate more.

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

Per (person, window), computed strictly from that person's own replicates, and split across two
tables by *what the measure is about*.

**`replicate_occurrence`** — the named outcome. Labelled with `outcome` (the name from
`arms.forecasting.lexis.outcome`, else the first configured outcome) and `horizon`, the cut-off in
days the event must fall inside, so a row is readable on its own.

- **Occurrence — `p_hat`.** The probability the outcome happens within the horizon, as the raw
  replicate frequency `n_occurred/n`. `n` and `n_occurred` are carried alongside.

- **Timing — `timing_spread`.** Inter-quantile width of the predicted age at first occurrence across
  seeds: `q90 − q10`. Wider = the seeds disagree more about *when*.

**`replicate_variance_individual`** — the birth-event count, independent of any configured outcome.

- **Count — `expected_quantum`, `within_seed_var`, `within_seed_cv`.** Predictive mean, variance and
  coefficient of variation across seeds of that person's completed event count.

These are plug-in dispersions, **not** resampling procedures. In the report both tables are
down-sampled to five randomly chosen persons (the full tables are in the parquet).

### 3.2 Confidence intervals on the backtest scores

Every interval next to a backtest metric is **analytic and computed from per-person quantities** —
`score_cis` (`metrics/ml.py`). Persons are the sampling unit, and each interval is
`estimate ± z·se`, so it can never fail to contain its own point estimate:

| metric | standard error |
|---|---|
| `mse`, `brier_raw` | `sd_i(l_i)/√n` on the per-person loss `l_i = (p̂_i − y_i)²` |
| `brier_corrected` | the same, on `l_i − c_i` where `c_i` is that run's MC-inflation term |
| `r2` | delta method on the ratio `A/B`: `sd_i((a_i − (A/B)·b_i)/B)/√n` |
| `roc_auc` | DeLong — `S10/n₊ + S01/n₋` from the per-person placement values, computed by midranks so the coarse `1/n` grid's ties count half. Clipped to `[0, 1]` |
| `ece` | **none.** Its bins are chosen from the data and the statistic is biased upward; there is no honest closed form |

`roc_auc` also loses its interval where DeLong's variance is exactly zero — a fully tied `p̂` grid or
perfect separation — because a zero-width interval would claim certainty the data has not earned.

These are sampling intervals that *already carry* replicate noise: each `l_i` is computed from that
person's own `p̂_i`, so the spread across persons contains the seed uncertainty, exactly as
`var_i(mu_i)` does for CCF (§2). They are the `total_var` analogue, not the replicate-only one.
There is no within/between split to report because `p̂` is defined *across* seeds — no per-seed loss
exists to average.

> Historical note: these intervals were previously percentile CIs from a seed bootstrap. That
> procedure resampled seed labels with replacement, which makes `p̂ = k/n` noisier than in the real
> sample; since every one of these metrics is nonlinear in `p̂`, the bootstrap distribution was
> centred on a different estimand than the point estimate, and in the demo 46 of 55 intervals
> excluded their own point. The bootstrap has been removed, along with the `replicates.bootstrap`
> config block.

---

## 4. Aggregate-target error (backtesting) — `aggregate_error` (`metrics/ml.py`)

For demographic aggregates (CCF, ASFR, PPR, KM survival at fixed ages) the model's generated cohort
is compared to the observed cohort cell by cell:

```
bias = metric_generated − metric_observed
```

`gen_sd_over_seeds` reports the seed-to-seed spread of each generated cell alongside it; no interval
is placed on the bias. The demo report does not surface this table by default, but it is written to
`aggregate_error.parquet`.

### 4.1 Per-seed populations and the pooled estimate — `metrics/pooling.py`

For the **time-dependent** families — KM survival, PPR, cohort ASFR, the timing-error ridge, and the
forecasting arm's Lexis surface — each seed is treated as its own synthetic population. Nothing is
averaged within an individual, and no within-individual variance term enters any of their intervals.

Two tables per family are written (from the backtesting arm, except Lexis, which is 05):

| table | what it is |
|---|---|
| `<family>_by_seed.parquet` | the metric computed once per seed, each carrying its own ordinary between-person sampling variance (`greenwood_var` / `ppr_var` / `asfr_var`) |
| `<family>_pooled.parquet` | the metric computed once over **all N×K trajectories at once**, via `arms/_common.pool_seeds`, which re-keys every `(person_id, seed)` pair to its own `person_id`. This is the estimate every figure draws |

The pooled frames report `n_units` (trajectories in the cell) and `n_source_persons` (the people
they were generated for) rather than a single ambiguous `n_persons`.

**The interval on the pooled estimate.** It is the metric's textbook sampling variance evaluated on
exactly the units that produced the estimate — no combination across seeds, no correction:

```
pooled_var = greenwood_var                      # KM,   over the pooled product-limit table
           | p(1−p)/n_units                     # PPR,  over the pooled progression denominator
           | births/person_years²               # ASFR & Lexis, over the pooled person-years
```

For KM that means `attach_km_pooled_ci` **keeps** the product-limit table's own complementary
log-log `ci_lo`/`ci_hi` — the traditional Greenwood interval — and `kaplan_meier(..., level=)`
carries the run's `replicates.level` into it. PPR, ASFR and Lexis get a symmetric `estimate ± z·se`
clipped to the metric's natural range.

**These intervals are deliberately optimistic, by roughly a factor of `√K` at the worst.** The N×K
rows are not N×K independent people: `combine_prefix` replays the *same* observed prefix under every
seed, so below the jump-off a person's K trajectories are exact duplicates, while above it they
diverge. Correcting for that is a modelling choice made downstream, not in the metric.

So that the correction *can* be made downstream, every pooled table still records the two quantities
it needs, measured from the K per-seed curves:

```
k_seeds     = seeds behind the cell
mean_var    = mean_s(var_s)              # per-seed sampling variance, averaged
between_var = var_s(estimate_s, ddof=1)  # spread of the K per-seed estimates
```

`pooling.design_effect_var` is that correction, kept in the codebase and wired into nothing:
`clip(mean_var − (K−1)/K · between_var, mean_var/K, mean_var)`, which interpolates between one
population's worth of people (seeds identical) and the naive N·K answer (seeds independent). Applying
it is post-processing over the emitted columns. Note `between_var` is a sample variance over K
numbers, so at the handful of seeds a typical run has, it carries real noise.

KM needs one extra step for the diagnostics: seeds do not share event times, so each curve's
`survival` and `greenwood_var` are step-sampled onto the pooled curve's grid (`survival.step_sample`)
before `mean_var` and `between_var` are formed.

CCF is **not** in this set. It is a scalar per cohort, and keeps its `within_var`/`between_var`/
`total_var` decomposition (§2).

---

## 5. Coverage accounting (the shrinking denominator)

Every backtest score has a denominator that shrinks as the observation window changes. The coverage
table (`_coverage_row` in `arms/backtesting.py`) breaks the condition population into:

- **`n_condition`** — persons matching the condition at the jump-off.
- **`n_evaluable`** — persons who actually contribute a score (event resolvable within the frame).
  This is the table's head count; it carries no `n_persons` alias, since the residual below is
  written in these names.
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
| Per-run probability, logit/variance | `core/replicates.py` (`replicate_summary`, `estimate_probability`) |
| AUC, Brier, log-loss, ECE, calibration table, timing coverage | `metrics/ml.py` |
| Brier finite-seed correction | `core/replicates.py` (`brier_noise_correction`) |
| Calibration null band | `core/replicates.py` (`null_calibration_band`) |
| Backtest score CIs | `metrics/ml.py` (`score_cis`, `_delong_var`) |
| Individual seed stability | `arms/forecasting.py` (`_seed_stability_individual`) |
| Coverage accounting | `arms/backtesting.py` (`_coverage_row`) |
