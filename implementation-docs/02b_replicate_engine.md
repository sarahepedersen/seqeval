# 02b — Replicate Engine: Empirical Probabilities from Seed Stochasticity

> Context: read `00_architecture.md`; depends on 02; consumed by 04 and 05. This module is the
> methodological core of the framework: models expose no logits, so **probabilities are
> recovered empirically from the distribution of outcomes across replicate runs** (multiple
> seeds per (person, window)). Everything probabilistic downstream — calibration, ROC-AUC,
> Brier, timing calibration, seed stability, uncertainty bands on aggregate forecasts — flows
> through this one engine. Getting the statistics right here (smoothed estimators, empirical
> logits, Monte-Carlo error quantification and correction) is what makes few-seed evaluations
> honest and many-seed evaluations efficient.

## Deliverables

```
src/seqeval/core/replicates.py
tests/test_replicates.py
```

## 0. Semantics and notation

A **replicate** is one generated trajectory for a (person_id, age_start, age_stop) run,
identified by the `seed` column. `seed` is a replicate *identifier*, not necessarily an RNG
seed the user controlled (LLM API samples, microsimulation draws, and hazard-model simulations
all qualify); the only contract is uniqueness within the run (00 §4). For a binary outcome
evaluated per replicate (02's evaluators), each run yields `k` occurrences out of `n` evaluable
replicates. The engine's job: turn (k, n) — and, for timing questions, the set of occurrence
ages across replicates — into probability estimates with honest uncertainty.

Key statistical facts the implementation must encode (cite in docstrings):

- **MLE p̂ = k/n** is unbiased but lives on a grid of width 1/n (n=5 → {0, .2, .4, .6, .8, 1});
  it hits exact 0 and 1, where log-loss and logits are undefined.
- **Empirical logit** (Haldane–Anscombe): `logit_emp = ln((k + ½) / (n − k + ½))` — defined for
  all k, the standard smoothed logit for binomial data. This is the "empirical logit
  probability" of the proposal.
- **Smoothed estimators:** Jeffreys posterior mean `(k + ½)/(n + 1)` (default), Laplace
  `(k + 1)/(n + 2)`, MLE `k/n`. Configurable; all three columns cheap to emit.
- **Intervals:** Jeffreys (Beta(k+½, n−k+½) quantiles; default) or Wilson. Normal approximation
  intentionally unsupported (terrible at small n / extreme p).
- **MC error propagation:** Var(p̂ | p) = p(1−p)/n. This inflates downstream metrics in known
  ways; see §3.

The recovered probability is the event probability **under the inference-time sampling
procedure** (temperature, top-k, etc.), not any internal softmax — a feature, not a bug: we
evaluate the generative system as it is actually used, which is exactly the output-driven
philosophy.

## 1. Replicate summaries → probability estimates

```python
def replicate_summary(outcome_table, *, run_keys=RUN_KEYS, seed_col="seed") -> pd.DataFrame
    # outcome_table: evaluator output (02 §2.4) at replicate level, already filtered to
    # evaluable. Returns one row per run: [*run_keys, k, n].
    # Ragged n across runs is legal (some replicates non-evaluable); n==0 runs are dropped
    # and counted for the coverage table.

def estimate_probability(summary, *, spec: ReplicateSpec) -> pd.DataFrame
    # [*run_keys, k, n, p_hat, logit_emp, var_logit, ci_lo, ci_hi]
    # p_hat per spec.estimator; logit_emp always Haldane–Anscombe; CI per spec.interval at
    # spec.level. Vectorized (scipy.stats.beta for Jeffreys quantiles).
```

`ReplicateSpec` (frozen dataclass in `core/specs.py`, resolved from the top-level `replicates:`
config block; add to 00a): `estimator`, `interval`, `level`, `min_replicates`, `bootstrap_n`,
`bootstrap_seed`, `convergence_curve: bool`.

`RUN_KEYS = ["person_id", "age_start", "age_stop"]` — GEN_KEYS minus seed; define next to the
other key constants.

## 2. Beyond binary: predictive distributions across replicates

Binary outcomes answer "whether"; replicates also carry a full predictive distribution over
"when" and "how many". Two extractors, both per run:

```python
def timing_distribution(tte_table, *, run_keys, seed_col, horizon: int) -> pd.DataFrame
    # tte_table: 02's time_to_event computed per replicate for one TTESpec.
    # Per run: empirical quantiles of duration (capped at horizon), P(occurs <= horizon),
    # n_occurred, n. Occurrence may fail in some replicates → the distribution is defective;
    # horizon capping makes summaries well-defined. Returns
    # [*run_keys, q10, q25, q50, q75, q90, p_within_horizon, n_occurred, n]

def count_distribution(count_table, *, run_keys, seed_col) -> pd.DataFrame
    # count_table: per-replicate event counts in a frame (from a CountQuery generalization or
    # a births-per-run aggregation). Per run: predictive mean, var, and pmf support columns
    # (long format [*run_keys, count, prob]) — feeds individual-level parity/CCF uncertainty.
```

Headline timing-calibration metric (computed in 04, defined here): **predictive interval
coverage** — the fraction of persons whose *observed* duration falls inside the central
(q10, q90) predictive interval; well-calibrated timing ⇒ ≈ 80%, restricted to persons whose
observed spans cover the horizon. CRPS (horizon-capped) is provided as an optional secondary
score with its censoring caveat documented.

## 3. Monte-Carlo error: quantify, correct, and baseline

Few seeds do not merely add noise — they add *bias* to specific metrics, in directions and
magnitudes we can compute. The engine ships three tools:

```python
def brier_noise_correction(summary) -> float
    # E[Brier(p_hat)] = Brier(p) + E[p(1−p)/n].  Unbiased per-run estimate of the inflation:
    # p_hat_mle(1−p_hat_mle)/(n−1).  Corrected Brier = raw − mean(inflation).
    # On a perfectly calibrated model with n=5 seeds this correction is LARGE; report both
    # raw and corrected in 04's scores.

def null_calibration_band(summary, *, n_bins, strategy, n_sims, rng) -> pd.DataFrame
    # Expected reliability-diagram scatter under PERFECT calibration given the observed
    # (n per run) profile: simulate k* ~ Binomial(n, p_hat), rebuild the calibration table per
    # sim, return per-bin (lo, hi) envelope. A model is only demonstrably miscalibrated where
    # its curve exits this band — prevents over-reading MC noise as miscalibration, the
    # single most likely misuse of few-seed evaluation.

def auc_tie_note()
    # not a function — a documented behavior: with p_hat on a 1/n grid, ties are massive;
    # use rank-based AUC with tie correction (scipy) and record grid resolution alongside.
```

## 4. Resampling over the replicate dimension

Generic machinery that turns *any* aggregate metric into one with uncertainty:

```python
def seed_bootstrap(df, *, seed_col, stat_fn: Callable[[pd.DataFrame], pd.DataFrame],
                   n_boot, rng) -> pd.DataFrame
    # resample seed labels with replacement (within each window), apply stat_fn (e.g. 03's ccf
    # with extra_by, or 04's aggregate_error), return percentile CIs per output cell.
    # Captures MC (replicate) uncertainty ONLY — document that population sampling uncertainty
    # would need a person-level cluster bootstrap (out of scope v1, note the extension point).

def convergence_curve(df, *, seed_col, stat_fn, sizes: list[int] | None, n_rep, rng)
    # subsample m seeds without replacement for m in sizes (default: 2..n), recompute stat_fn,
    # return dispersion vs m. THE actionable diagnostic: since inference is upstream, "your
    # AUC estimate has not stabilized at 10 seeds" tells the researcher to go generate more
    # replicates — the framework's version of a power analysis.
```

Both must be implemented as groupby-over-precomputed-replicate-tables, never re-running outcome
evaluation per bootstrap draw (evaluate once per replicate; resample the rows).

## 5. Relationship to consumers

- **04 (backtesting)** replaces its ad-hoc `empirical_probability` with this engine: per
  (window, outcome, condition), emit `probabilities.parquet`
  `[model, window..., outcome, condition, person_id, k, n, p_hat, logit_emp, ci_lo, ci_hi]` as
  a first-class artifact (researchers will regress on these), then calibration (with null
  band), AUC (tie-corrected), Brier raw + corrected, log-loss on smoothed p̂, timing interval
  coverage; bootstrap CIs on all `aggregate_error` cells; convergence curves when configured.
- **05 (seed stability)** is reframed as views over this engine: individual-level occurrence
  disagreement IS `p_hat(1−p_hat)` (report as such); timing dispersion comes from
  `timing_distribution` quantile spreads; aggregate forecast uncertainty (CCF bands, Lexis
  IQR maps) comes from `seed_bootstrap`. No duplicate statistics code in 05.
- **06 (validate/report)**: validate prints replicates-per-window and flags windows below
  `min_replicates` with a plain-language note of what that implies (probability grid width,
  finest meaningful calibration bin); report shows convergence curves and the null band on
  every reliability diagram.

## 6. Tests

- Exact-value tests for all estimators/intervals/logit on constructed (k, n) tables, including
  k=0 and k=n.
- **Perfect-model few-seed test** (the load-bearing one): `simulate_generated` with n=5 seeds —
  raw Brier exceeds the true Brier (computable from known hazards) by ≈ mean p(1−p)/5;
  corrected Brier recovers truth within tolerance; the model's calibration curve stays inside
  `null_calibration_band` ≈ 95% of bins. Repeat with n=50: raw and corrected converge.
- Miscalibrated model (`perturb(hazards, 1.5)`): calibration curve exits the null band in the
  expected direction — the band flags real miscalibration, not noise.
- Timing coverage: perfect model → (q10, q90) coverage ≈ 0.8 (loose tolerance, documented).
- `seed_bootstrap` CI on synthetic CCF covers the converged `expected_ccf` at nominal rate
  across repeated simulations (small repetition count, sanity not certification).
- Convergence curve monotone-in-expectation dispersion decrease; ragged-n and n==0 handling.

## 7. Appendix: estimator specification (reproduce as module docstring material)

For k ~ Binomial(n, p), the additive-c family `ln((k+c)/(n−k+c))` has expectation
`logit(p) + (c − ½)(1/(np) − 1/(n(1−p))) + O(n⁻²)` (Gart & Zweifel 1967): **c = ½ (Haldane–
Anscombe) is the unique first-order-bias-cancelling choice for all p**, with Gart's variance
`1/(k+½) + 1/(n−k+½)` emitted as `var_logit` (a required column in the probability table:
users regressing on empirical logits need it as weights, especially with ragged n). Coherence
identity to preserve and test: `logit_emp == logit(p_hat)` exactly when `estimator: jeffreys`
(both reduce to (k+½) vs (n−k+½)); with `mle`/`laplace`, `logit_emp` stays Haldane–Anscombe by
definition — say so in the docstring.

**Estimation is strictly per run.** Never pool, shrink, or borrow strength across runs or
persons when estimating a run's probability: the purpose of replicate probabilities is to
measure across-replicate variance for an individual sequence, and cross-run pooling changes
the estimand. All cross-run computation happens downstream, on the per-run table (calibration
binning, bootstraps), never inside estimation.

**Dynamic range** (this table belongs in the module docstring): |logit_emp| ≤ ln(2n+1), so
replicate data cannot express probabilities more extreme than ≈ 1/(2n+2) from the boundary —
n=5 → |logit| ≤ 2.40 (p ∈ ~[0.08, 0.92]); n=50 → 4.62 (~[0.01, 0.99]); n=200 → 6.00. Rare
outcomes at small n saturate the estimator (heavy shrinkage toward zero log-odds); the
convergence curve (§4) is how a researcher discovers they need more replicates.

**Informative censoring guard:** spans always derive from the last age in the data (00 §4.2)
— one path, no overrides. On models with stochastic sequence length, event-sparse replicates
can therefore look shorter, get dropped as non-evaluable for late frames, and bias k/n upward
(informative censoring). The remedy is upstream and data-borne: inference generated to a fixed
horizon t4 should emit a trailing "no event" row at t4, which carries the horizon into the
span for free. Detection is the engine's job: when within-run n varies for a (window, outcome)
cell, `replicate_summary` must emit a loud warning naming the affected cells, and validate
(06) surfaces it. Add a test: construct a run where event-free replicates end early — the
warning fires and p_hat is biased up vs truth; with trailing no-event rows at t4, n is
constant and p_hat is unbiased.

## Acceptance criteria

- `pytest -q` green, `ruff` clean.
- A demo notebook/script figure: reliability diagram with null band at n_seeds ∈ {5, 50} on the
  same synthetic model, showing the band shrinking — attach to PR; this figure is the
  methodological pitch of the whole framework in one image.
