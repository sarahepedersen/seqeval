# 00a — Dataclass Reference

> Companion to `00_architecture.md`. This documents every dataclass in the system: what each one
> defines, who creates it, who consumes it, and why it exists as a separate type. The design goal
> is a *minimal* set — ten classes total, each answering one question. Claude Code: reproduce the 
> relevant parts of this document as the module docstring of `core/specs.py`.

## The mental model in one paragraph

Data flows through three shapes. **Carriers** hold validated data (`Bundle`, `AgeBins`).
**Question specs** (`core/specs.py`) are frozen, day-valued, raw-token-valued objects that state
a question precisely — they are what `config.resolve_*` produces from the year-valued,
alias-valued YAML, and what `core/` evaluator functions consume. **Test scaffolding**
(`HazardSpec`) exists only under `tests/`. No dataclass contains logic; all evaluation lives in
functions that take (DataFrame, keys, spec).

## Quick reference

| class | module | one-line definition | created by | consumed by |
|---|---|---|---|---|
| `TTESpec` | core/specs | *when* does the n-th occurrence of an event happen, measured from an origin | `resolve_outcomes` | `time_to_event`, KM, Lexis, `FramedOutcome` |
| `Frame` | core/specs | a time window that turns a question into a yes/no | resolvers | `evaluate_framed`, `evaluate_count` |
| `FramedOutcome` | core/specs | does a **specific ordinal occurrence** land inside a frame | `resolve_probability_outcomes` | `evaluate_framed` (02) |
| `CountQuery` | core/specs | do **≥ m post-jump-off occurrences** land inside a frame | `resolve_probability_outcomes` | `evaluate_count` (02) |
| `Condition` | core/specs | count predicate on the **observed prefix**, used as a population filter | `resolve_conditions` | `condition_on_count` (02) |
| `Rule` | core/specs | an impossible/implausible pattern to flag in output | `resolve_rules` | `check_rules` (05) |
| `ReplicateSpec` | core/specs | policy for turning replicate counts into probabilities | `resolve_replicates` | `core/replicates.py` (02b) |
| `Bundle` | io/loaders | all loaded artifacts + resolved event tokens, one object | `load_all` | arms, CLI |
| `AgeBins` | core/slicing | paired day-valued bin edges and year-valued labels | `AgeBins.from_years` | `exposure`, ASFR, life table, Lexis |
| `HazardSpec` | tests/synthetic | known ground-truth hazard model for synthetic data | test code | `simulate_*` |

## Question specs, one by one

### `TTESpec` — a timing quantity

```python
TTESpec(target, occurrence=1, origin: TTESpec | None = None)
```

States: *the time at which the `occurrence`-th `target` event happens, measured from `origin`
(person's birth when `origin is None`).* This is the registry primitive — sequence-intrinsic,
context-free, valid for observed and generated data alike. `origin` nests one level at most
(enforced at config parse); when set, the quantity is implicitly conditioned on the origin
occurring ("time from 1st to 2nd birth" only exists for people with a 1st birth).

YAML → object:

```yaml
second_birth: {event: birth, n: 2, origin: first_birth}
```
```python
TTESpec(target=1, occurrence=2, origin=TTESpec(target=42, occurrence=1))
```

Used directly (no frame) wherever a duration is the object of study: Kaplan-Meier curves,
`km:*` aggregate backtest targets, the Lexis outcome.

### `Frame` — a window that makes a question binary

```python
Frame(kind: Literal["by_age", "within", "within_origin"], value: int)  # value in days
```

A timing quantity answers "when"; attaching a `Frame` converts it to "does it happen inside
this window" — the binary form that calibration/ROC-AUC/Brier need. Frames are the *only* place
evaluation context enters: `within` means "within `value` of the jump-off (t2)", which is why
frames live in arm config, never in the registry. `by_age` is absolute; `within_origin` is
relative to a `TTESpec`'s own origin event and is therefore only legal on framed references
whose registry outcome declares an `origin`.

(Minimality note: `Frame` could be inlined as two fields on each of the next two classes; it is
kept as a shared type so frame validation and day-resolution exist in exactly one place. It is
the one class we'd fold away without regret if it ever bothers you.)

### `FramedOutcome` vs `CountQuery` — the distinction that matters

Both resolve a `probability_outcomes:` entry; both are evaluated per sequence-group to the same
output shape `[*keys, occurred, evaluable]`, so everything downstream (empirical probabilities,
calibration, AUC) is agnostic to which one produced the table. They are separate classes because
they ask **genuinely different scientific questions that are easy to conflate**:

```python
FramedOutcome(name, tte: TTESpec, frame: Frame, given: str | None)
CountQuery(name, event, min_events: int, frame: Frame, given: str | None)
```

- A **FramedOutcome** asks about a *specific ordinal occurrence in the whole life course*: "does
  the person's **2nd** birth happen by age 35?" The ordinal is absolute — counted from the start
  of life, including events the model was shown in its prompt.
- A **CountQuery** asks about *how many events happen after the jump-off*: "do **≥ 1** births
  occur in (t2, t2+5]?" — "will they have a child in the next five years", the spec's own
  motivating ROC-AUC example. Ordinals are irrelevant; only post-t2 events count.

Worked example — jump-off t2 = 30, frame horizon age 35:

| woman | births at | `FramedOutcome(second_birth, by_age 35)` | `CountQuery(birth, ≥1, within 5)` |
|---|---|---|---|
| A | 27, 33 | occurred = True (2nd at 33 ≤ 35) | occurred = True (33 ∈ (30, 35]) |
| B | 22, 25 | **non-evaluable** — 2nd birth at 25 < t2: settled at jump-off, the answer was in the prompt | occurred = False — pre-t2 births don't count; this is a real prediction about her future |
| C | none | occurred = False (if span covers to 35) | occurred = False |

Woman B is the whole argument: the framed question is *already answered* by the observed prefix
(so it must be excluded — it measures nothing about the model), while the count question is a
live prediction. One class with a mode flag would invite writing the framed form while meaning
the count form; two classes force the choice. The evaluators embody the asymmetry: only
`evaluate_framed` performs the settled-at-jump-off check; `evaluate_count` never needs it
because it counts strictly after t2 by construction.

`given` on either form names a `Condition` (below); resolution validates the reference.

### `Condition` — a population filter, the mirror image of `CountQuery`

```python
Condition(name, event, min_count=0, max_count=None, before_age: int | None = None)
```

States: *keep only sequence-groups where the count of `event` occurrences at ages ≤ anchor lies
in [min_count, max_count]*, where the anchor is `before_age` if set, else the caller's jump-off.
Parity conditioning is the fertility instance (`event=birth, min_count=1, max_count=1` = "parity
exactly 1 at t2"); nothing in `core/` knows the word parity.

`Condition` and `CountQuery` are deliberately near-twins — both count occurrences of one event
against thresholds — split across the jump-off:

| | `Condition` | `CountQuery` |
|---|---|---|
| counts events in | observed prefix, age ≤ anchor | generated future, age > t2 |
| role | filters *who is in* the evaluation (given:) | defines *what is predicted* (the outcome) |
| evaluated on | observed data (the prefix the model actually saw), then applied to both sides | generated runs, and observed truth for comparison |

They share the internal counting helper in implementation but stay separate types: merging them
would let a filter be used as an outcome (or vice versa) with silently opposite time regions.
The symmetry is a feature for *understanding*; the type split is a guard for *usage*.

### `Rule` — an illegal-move pattern

```python
Rule(name, event, min_age=None, max_age=None, min_spacing=None,
     not_after=None, not_before=None, max_count=None, severity="illegal")
```

States one impossible or implausible pattern to flag in sequences (all age/spacing values in
days). Purely declarative — the rules engine (`check_rules`, 05) interprets whichever fields are
set; adding a rule never means adding code. Runs against generated *and* observed data, since
violations in observed data indicate data problems rather than model problems. Not merged with
`Condition` despite both mentioning counts: a `Rule` flags *rows* (which events violate),
carries severity, and supports pattern fields (`min_spacing`, `not_after`/`not_before`) that make
no sense as population filters. The two ordering fields are not mirror images: `not_after` only
constrains groups where the anchor event exists, while `not_before` also flags the event when the
anchor never occurs (a divorce with no marriage anywhere).

## Carriers

### `Bundle` — everything loaded, once

```python
Bundle(observed, generated, persons, event_defs, events)
```

Frozen product of `load_all`: the validated canonical frames and the resolved event alias map.
Observation spans are *not* precomputed here — they derive from the frames themselves via the
last-age convention (`observation_spans`, 02), so there is nothing extra to carry. Methods are
lookups only: `token(alias)`, `label(raw_token)`, `require_persons(why)`,
`available_windows()`, `population_summary()`. It
exists so arm signatures are `run(bundle, cfg, out)` instead of seven parameters, and so
"persons is missing" errors are raised in one place with one message style.

### `AgeBins` — edges and labels that cannot drift apart

```python
AgeBins(edges_days: np.ndarray, labels: np.ndarray)   # built via AgeBins.from_years(lo, hi, w)
```

Pairs the day-valued edges used for comparisons with the year-valued labels used in output
tables and plots. Exists because passing bare edge arrays around is exactly how a
days-vs-years bug would slip past review; the constructor is one of the three sanctioned unit
conversion sites' clients (it calls `units.py` once).

## Test scaffolding

### `HazardSpec` — the known truth behind synthetic data

```python
HazardSpec(rates: dict[(lo_age_yr, hi_age_yr, parity), rate_per_year], max_parity, fertile_ages)
```

A piecewise-constant birth-hazard model, specified in years for readability, from which
`simulate_cohort` draws observed histories and `simulate_generated` draws "perfect model"
futures. Lives under `tests/`, never imported by `src/`. It is the reason every metric can be
tested against converged truth and every calibration test has a model that *should* hug the
diagonal.

## What is deliberately NOT a dataclass

- **Sequences, births tables, spans, TTE tables, metric results** — plain `pd.DataFrame`s in
  documented shapes. Wrapping frames in classes buys nothing at millions of rows and breaks the
  `(df, keys)` reuse mechanism.
- **Pipeline config** — pydantic models (validation, `extra="forbid"`, YAML-path errors), a
  different tool for a different job: config models mirror user-facing YAML in years/aliases;
  spec dataclasses are the resolved internal form in days/tokens. The pair per concept
  (`ConditionConfig` → `Condition`) is the units boundary made visible in the type system.
- **Windows** — plain `tuple[int, int]` (day-valued `(age_start, age_stop)`). They have no
  behavior and no invariants beyond `start ≤ stop`, which loaders already validate.

### `ReplicateSpec` — how seed-stochasticity becomes probability (added with plan 02b)

```python
ReplicateSpec(interval="jeffreys", level=0.95,
              min_replicates=5)
```

Resolved from the top-level `replicates:` config block; consumed by every function in
`core/replicates.py`. States the *policy* for turning per-run replicate counts (k of n) into
probability estimates: which interval, when to warn about thin
replicate counts, and how much resampling to spend on uncertainty bands. It is a policy object
rather than loose keyword arguments so that 04 and 05 provably apply identical estimation
settings — one spec, threaded everywhere. (`RUN_KEYS = GEN_KEYS minus seed` is a companion
constant, not a class.)
