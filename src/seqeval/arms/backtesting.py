"""Past/generated backtesting arm: sweep jump-off windows, score generated vs observed (04).

Orchestration only — every statistic comes from 02b (probabilities/bands/bootstraps) and 03
(fertility/survival metrics), reused unchanged. For each window ``(t1, t2)`` the arm evaluates the
configured probability outcomes and aggregate targets on the generated runs and on the observed
truth with the *same* ``jumpoff = t2``, so both sides describe the same population.

Full-life-course construction (00 section 5.1, 04 section 2.1): the generated file holds only rows
with ``age > t2``. A **framed** outcome (absolute ordinal) and every **aggregate** metric therefore
need each replicate's *full* sequence, so the arm concatenates the observed prefix (``age <= t2``)
with the generated future per replicate and runs 02's evaluators / 03's metrics on the combined
frame. This makes the settled-at-jump-off rule and the ordinal count fall out correctly and
identically on both sides. A **count** query counts only post-``t2`` events, so it is evaluated on
the generated rows directly (no prefix needed).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from seqeval.arms._common import OutputWriter, combine_prefix
from seqeval.config import (
    DEFAULT_COHORT_WIDTH,
    FERTILITY_TARGETS,
    FERTILITY_TARGETS_NEEDING_PERSONS,
    BacktestingConfig,
)
from seqeval.core import replicates as rep
from seqeval.core.outcomes import (
    births,
    evaluate_count,
    evaluate_framed,
    observation_spans,
    time_to_event,
)
from seqeval.core.slicing import AgeBins, align_jumpoff_to_event, condition_on_count
from seqeval.core.specs import (
    Condition,
    CountQuery,
    FertilityGrid,
    FramedOutcome,
    ReplicateSpec,
    TTESpec,
)
from seqeval.io.loaders import Bundle
from seqeval.io.schema import GEN_KEYS, OBS_KEYS
from seqeval.metrics import fertility as fe
from seqeval.metrics import ml
from seqeval.metrics import survival as sv
from seqeval.units import days_to_years
from seqeval.viz import backtest as viz_backtest
from seqeval.viz import calibration as viz_calibration
from seqeval.viz import fertility as viz_fertility
from seqeval.viz._labels import describe_outcome

logger = logging.getLogger("seqeval")

_RUN_KEYS = ["person_id", "age_start", "age_stop"]
_EXTRA_BY = ("seed", "age_start", "age_stop")
_FERTILE = (15.0, 50.0)
# Width of a timing-error bin on the ridge; one year is the resolution a reader of ages expects.
_ERROR_BIN_YEARS = 1.0
# Accumulated tables keyed to individuals. The rest are per-bin or per-cell aggregates:
# `calibration`/`coverage` count people in a bin, `timing_error`/`parity_distribution` in a cell.
_PER_PERSON = {"probabilities"}
# Age grid (years) at which generated vs observed KM survival is compared for `km:*` targets.
_KM_GRID_YEARS = list(range(16, 46, 2))
# Fertility cell geometry when the caller supplies none; the CLI always resolves one from the
# descriptives block (`config.resolve_fertility_grid`) so the two arms bin alike.
_DEFAULT_FERTILITY_GRID = FertilityGrid()


def run(
    bundle: Bundle,
    cfg: BacktestingConfig,
    out: OutputWriter,
    *,
    outcomes: dict[str, TTESpec],
    conditions: dict[str, Condition],
    prob_outcomes: list[FramedOutcome | CountQuery],
    replicate_spec: ReplicateSpec,
    cohort_width: int = DEFAULT_COHORT_WIDTH,
    fertility_grid: FertilityGrid = _DEFAULT_FERTILITY_GRID,
) -> None:
    """Run backtesting over every configured window; write the six result tables (04 section 2.2).

    ``outcomes``/``conditions``/``prob_outcomes``/``replicate_spec``/``fertility_grid`` are the
    resolved objects from ``config.resolve_*`` (passed in, like the descriptives registry, because
    they live at the top level of the config). Writes ``probabilities``, ``calibration``,
    ``scores``, ``aggregate_error`` and ``coverage`` parquet tables.
    """
    if bundle.generated is None:
        logger.warning("backtesting: no generated file; arm skipped")
        return

    from seqeval.config import resolve_windows

    windows = resolve_windows(cfg.windows, bundle.available_windows())
    observed = bundle.observed
    spans_obs = observation_spans(observed, OBS_KEYS)
    birth_token = bundle.token("birth") if _needs_births(cfg) else None

    acc: dict[str, list[pd.DataFrame]] = {
        k: []
        for k in (
            "probabilities",
            "calibration",
            "scores",
            "aggregate_error",
            "coverage",
            "timing_error",
            "parity_distribution",
        )
    }
    # Overlay curves kept across the window loop, keyed by family ("ccf" / "km:<outcome>"), so the
    # per-jump-off figures can be joined into one cross-window comparison panel at the end.
    panels: dict[str, dict] = {}

    for t1, t2 in windows:
        gen_w = bundle.generated[
            (bundle.generated["age_start"] == t1) & (bundle.generated["age_stop"] == t2)
        ]
        if gen_w.empty:
            continue
        cond_sets = {
            name: set(condition_on_count(observed, OBS_KEYS, cond=cond, anchor_age=t2)["person_id"])
            for name, cond in conditions.items()
        }
        all_persons = set(observed["person_id"].unique())

        for spec in prob_outcomes:
            _score_probability_outcome(
                spec,
                gen_w,
                observed,
                spans_obs,
                t1,
                t2,
                cond_sets,
                all_persons,
                replicate_spec,
                acc,
                out,
                bundle.label,
                cfg.calibration_binning,
            )

        for target in cfg.aggregate_targets:
            _score_aggregate_target(
                target,
                gen_w,
                observed,
                spans_obs,
                bundle.persons,
                birth_token,
                outcomes,
                cohort_width,
                t1,
                t2,
                replicate_spec,
                acc,
                out,
                panels,
                fertility_grid,
            )

    _emit_jumpoff_panels(panels, out, replicate_spec.level)

    for name, frames in acc.items():
        if frames:
            out.frame(name, pd.concat(frames, ignore_index=True), individual=name in _PER_PERSON)


# =================================================================================================
# per-outcome scoring
# =================================================================================================
def _score_probability_outcome(
    spec,
    gen_w,
    observed,
    spans_obs,
    t1,
    t2,
    cond_sets,
    all_persons,
    replicate_spec,
    acc,
    out,
    label_fn,
    binning,
) -> None:
    given = spec.given
    cond_persons = cond_sets[given] if given else all_persons

    obs_eval = _evaluate_observed(spec, observed, spans_obs, t2)
    gen_eval = _evaluate_generated(spec, gen_w, observed, t1, t2)

    obs_eval = obs_eval[obs_eval["person_id"].isin(cond_persons)]
    gen_eval = gen_eval[gen_eval["person_id"].isin(cond_persons)]
    settled = _settled_persons(spec, observed, t2) & set(cond_persons)

    label = _cell_label(t1, t2, spec.name, given)
    acc["coverage"].append(
        _coverage_row(obs_eval, gen_eval, cond_persons, all_persons, settled, label)
    )

    summary = rep.replicate_summary(gen_eval, run_keys=_RUN_KEYS)
    if summary.empty:
        return
    _warn_thin_replicates(summary, replicate_spec, label)
    probs = rep.estimate_probability(summary, spec=replicate_spec)
    joined = ml.join_truth(probs, obs_eval)
    if joined.empty or joined["y_true"].nunique() < 1:
        return

    if out.individual_level:
        acc["probabilities"].append(_stamp(probs, label))
    else:
        out.withhold("probabilities")  # never assembled, so the writer never sees it

    cal = ml.calibration_table(joined, n_bins=10, strategy=binning)
    acc["calibration"].append(_stamp(cal, label))

    desc = describe_outcome(spec, jumpoff_days=t2, label_fn=label_fn)
    # The curve is binned either way; its histogram panel is keyed to the per-person probability
    # table, so it goes when per-person output does. The bin counts survive in `calibration`.
    out.figure(
        f"reliability_{spec.name}_w{int(round(days_to_years(t2)))}",
        viz_calibration.plot_reliability(
            cal, probs=probs if out.individual_level else None, title=desc
        ),
    )
    # Timed outcomes also get a timing-error ridge, drawn on the same population this reliability
    # diagram scores: the condition minus the settled. The tables it needs also carry the timing
    # coverage in `scores`, so they are built once here and handed to both.
    tables = None
    if isinstance(spec, FramedOutcome):
        tables = _timing_tables(spec, gen_w, observed, t1, t2)
        _emit_timing_ridge(spec, tables, t2, out, desc, set(cond_persons) - settled, acc, label)

    scores = _score_row(joined, summary, tables, binning)
    cis = ml.score_cis(joined, level=replicate_spec.level)
    acc["scores"].append(_stamp_scores(scores, label, cis))


def _evaluate_observed(spec, observed, spans_obs, t2) -> pd.DataFrame:
    if isinstance(spec, CountQuery):
        return evaluate_count(observed, OBS_KEYS, spec, spans_obs, jumpoff=t2)
    return evaluate_framed(observed, OBS_KEYS, spec, spans_obs, jumpoff=t2)


def _evaluate_generated(spec, gen_w, observed, t1, t2) -> pd.DataFrame:
    if isinstance(spec, CountQuery):
        spans = observation_spans(gen_w, GEN_KEYS)
        return evaluate_count(gen_w, GEN_KEYS, spec, spans, jumpoff=t2)
    combined = combine_prefix(observed, gen_w, t1, t2)
    spans = observation_spans(combined, GEN_KEYS)
    return evaluate_framed(combined, GEN_KEYS, spec, spans, jumpoff=t2)


# =================================================================================================
# coverage + scores
# =================================================================================================
def _settled_persons(spec, observed, t2) -> set:
    """Persons whose *framed* outcome is already settled at t2 (answer in the observed prefix)."""
    if not isinstance(spec, FramedOutcome):
        return set()
    tte = spec.tte
    target = align_jumpoff_to_event(observed, event=tte.target, occurrence=tte.occurrence)
    target_age = target.set_index("person_id")["age"]
    if spec.frame.kind == "by_age":
        upper = pd.Series(spec.frame.value, index=target_age.index)
    elif spec.frame.kind == "within":
        upper = pd.Series(t2 + spec.frame.value, index=target_age.index)
    else:  # within_origin
        origin = align_jumpoff_to_event(
            observed, event=tte.origin.target, occurrence=tte.origin.occurrence
        ).set_index("person_id")["age"]
        upper = (origin + spec.frame.value).reindex(target_age.index)
    settled = (upper <= t2) | (target_age <= t2)
    return set(target_age.index[settled.fillna(False)])


def _warn_thin_replicates(summary, spec, label) -> None:
    median_n = float(summary["n"].median())
    if median_n < spec.min_replicates:
        logger.warning(
            "backtesting %s: median replicate count %.0f < min_replicates %d — probabilities are "
            "coarse; generate more seeds",
            label["outcome"],
            median_n,
            spec.min_replicates,
        )


def _coverage_row(obs_eval, gen_eval, cond_persons, all_persons, settled, label) -> pd.DataFrame:
    """Per-cell evaluability accounting — shrinking sets and thin replicate counts, never silent."""
    n_condition = len(cond_persons)
    n_evaluable = int(obs_eval["evaluable"].sum())
    n_settled = len(settled)
    ns = gen_eval.loc[gen_eval["evaluable"]].groupby(_RUN_KEYS, observed=True).size()
    return pd.DataFrame(
        [
            {
                **label,
                "n_condition": n_condition,
                "n_evaluable": n_evaluable,
                "n_excluded_condition": len(all_persons) - n_condition,
                "n_settled": n_settled,
                # everything in the condition that is neither evaluable nor settled = uncovered span
                "n_uncovered": max(n_condition - n_evaluable - n_settled, 0),
                "n_seed_min": int(ns.min()) if len(ns) else 0,
                "n_seed_median": float(ns.median()) if len(ns) else 0.0,
                "n_seed_max": int(ns.max()) if len(ns) else 0,
            }
        ]
    )


def _score_row(joined, summary, tables, binning) -> dict:
    brier = ml.brier(joined)
    median_n = float(summary["n"].median())
    scores = {
        "ece": ml.ece(ml.calibration_table(joined, strategy=binning)),
        "roc_auc": ml.roc_auc(joined),
        "auc_grid_resolution": 1.0 / median_n if median_n else np.nan,
        "brier_raw": brier["raw"],
        "brier_corrected": brier["corrected"],
        "mse": ml.mse(joined),
        "r2": ml.r2(joined),
    }
    if tables is not None:
        td, obs_tte, _ = tables
        scores["timing_coverage"] = ml.timing_coverage(td, obs_tte)
    return scores


def _timing_horizon(spec, t2) -> int:
    """Where the frame closes, expressed in the outcome's duration units (days from its origin).

    ``time_to_event`` measures from the outcome's origin, so an absolute (``by_age``) or
    jump-off-relative (``within``) frame has to be re-expressed before it can cap a duration. For an
    origin-less outcome the duration *is* the age, so the translation is exact. For an origin-based
    outcome with an absolute frame the exact close is person-specific (``value - origin_age``); the
    frame value is kept as a loose upper cap there rather than silently inventing a scalar.
    """
    kind, value = spec.frame.kind, spec.frame.value
    if kind == "within_origin" or spec.tte.origin is not None:
        return value
    return value if kind == "by_age" else t2 + value


def _timing_tables(spec, gen_w, observed, t1, t2):
    """``(timing_distribution, observed tte, horizon_days)`` for a framed (timed) outcome."""
    combined = combine_prefix(observed, gen_w, t1, t2)
    tte_gen = time_to_event(combined, GEN_KEYS, spec.tte)
    horizon = _timing_horizon(spec, t2)
    td = rep.timing_distribution(tte_gen, run_keys=_RUN_KEYS, seed_col="seed", horizon=horizon)
    obs_tte = time_to_event(observed, OBS_KEYS, spec.tte)
    return td, obs_tte, horizon


def _emit_timing_ridge(spec, tables, t2, out, desc, scored, acc, label) -> None:
    """Timing-error ridge + its binned table for a timed outcome (one figure per jump-off).

    The error is measured in the outcome's own duration units, so an origin-less outcome compares
    ages at the event and an origin-based one compares elapsed times from its origin. Both reduce to
    the same signed difference, which is why one figure serves both and only the label changes.
    """
    td, obs_tte, horizon = tables
    err = ml.timing_error_distribution(
        td, obs_tte, horizon_days=horizon, persons=scored, error_bin_years=_ERROR_BIN_YEARS,
        min_cell=out.min_cell,
    )
    if err.empty:
        return
    err.insert(0, "outcome", spec.name)
    acc["timing_error"].append(_stamp(err, label))
    is_age = spec.tte.origin is None
    unit = "age at event" if is_age else "waiting time"
    out.figure(
        f"timing_ridge_{spec.name}_w{int(round(days_to_years(t2)))}",
        viz_backtest.plot_timing_ridge(
            err,
            xlabel=f"observed − predicted {unit} (years)",
            title=f"Timing error — {desc}",
        ),
    )


# =================================================================================================
# aggregate targets
# =================================================================================================
def _score_aggregate_target(
    target,
    gen_w,
    observed,
    spans_obs,
    persons,
    birth_token,
    outcomes,
    cohort_width,
    t1,
    t2,
    replicate_spec,
    acc,
    out,
    panels,
    fertility_grid,
) -> None:
    if persons is None and target in FERTILITY_TARGETS_NEEDING_PERSONS:
        logger.warning("backtesting: skipping aggregate target %r — needs persons", target)
        return
    combined = combine_prefix(observed, gen_w, t1, t2)
    fertility = target in FERTILITY_TARGETS
    gen_births = births(combined, GEN_KEYS, birth_event=birth_token) if fertility else None
    obs_births = births(observed, OBS_KEYS, birth_event=birth_token) if fertility else None
    gen_spans = observation_spans(combined, GEN_KEYS) if fertility else None

    try:
        gen_m, obs_m, value_col, on = _aggregate_tables(
            target,
            gen_births,
            gen_spans,
            obs_births,
            spans_obs,
            persons,
            outcomes,
            cohort_width,
            combined,
            observed,
            fertility_grid,
        )
    except _UnsupportedTarget:
        logger.warning("backtesting: aggregate target %r not supported; skipped", target)
        return

    err = ml.aggregate_error(gen_m, obs_m, value_col=value_col, on=on)
    err.insert(0, "target", target)
    acc["aggregate_error"].append(_stamp(err, _cell_label(t1, t2, target, None)))

    # Generated-vs-observed overlays: the observed "truth" under the generated replicate-CI band.
    jumpoff_y = round(days_to_years(t2))
    if target.startswith("km:"):
        _emit_km_overlay(
            target[len("km:") :], combined, observed, outcomes, t2, out,
            replicate_spec.level, panels,
        )
    elif target == "ccf":
        var_m = fe.ccf_variance(gen_births, gen_spans, persons, cohort_width=cohort_width)
        # This window's own CCF view is the inference-vs-outcome figure below; the overlay survives
        # only as the cross-jump-off panel, which compares windows on one axes.
        _stash_panel(panels, "ccf", obs_m, gen_m, t2, variance=var_m)

        # The same population, so the distribution and the mean drawn over it describe one quantity.
        par_m = fe.parity_distribution(
            gen_births, gen_spans, persons, cohort_width=cohort_width, min_cell=out.min_cell
        )
        acc["parity_distribution"].append(_stamp(par_m, _cell_label(t1, t2, "ccf", None)))
        out.figure(
            f"uncertainty_ccf_w{jumpoff_y}",
            viz_fertility.plot_ccf_inference_vs_outcome(
                var_m, par_m, observed=obs_m,
                complete=viz_backtest.majority_complete(gen_m),
                level=replicate_spec.level,
                title=f"Inference vs outcome uncertainty — jump-off {jumpoff_y}y",
            ),
        )
    elif target == "ppr":
        _stash_panel(panels, "ppr", obs_m, gen_m, t2)
        out.figure(
            f"ppr_overlay_w{jumpoff_y}",
            viz_backtest.plot_ppr_overlay(
                obs_m, gen_m, level=replicate_spec.level,
                title=f"Parity progression — jump-off {jumpoff_y}y",
            ),
        )
    elif target == "asfr_cohort":
        _stash_panel(panels, "asfr_cohort", obs_m, gen_m, t2)
        out.figure(
            f"asfr_overlay_w{jumpoff_y}",
            viz_backtest.plot_asfr_overlay(
                obs_m, gen_m, jumpoff_days=t2, level=replicate_spec.level,
                title=f"Cohort ASFR — jump-off {jumpoff_y}y",
            ),
        )


def _emit_km_overlay(name, combined, observed, outcomes, t2, out, level, panels) -> None:
    """Emit the observed KM curve under the generated across-seed mean + Monte-Carlo CI band."""
    spec = outcomes[name]
    obs_km = sv.kaplan_meier(time_to_event(observed, OBS_KEYS, spec), by=[])
    gen_km = sv.kaplan_meier(time_to_event(combined, GEN_KEYS, spec), by=["seed"])
    jumpoff_y = round(days_to_years(t2))
    out.figure(
        f"km_overlay_{name}_w{jumpoff_y}",
        viz_backtest.plot_km_seed_band(
            obs_km, gen_km, title=f"{name} survival — jump-off {jumpoff_y}y", level=level
        ),
    )
    _stash_panel(panels, f"km:{name}", obs_km, gen_km, t2)


def _stash_panel(
    panels: dict, key: str, obs: pd.DataFrame, gen: pd.DataFrame, t2: int, *, variance=None
) -> None:
    """Keep one window's curves for the cross-jump-off panel emitted after the window loop.
    """
    entry = panels.setdefault(key, {"obs": obs, "gen": {}, "variance": {}})
    entry["gen"][int(t2)] = gen
    if variance is not None:
        entry["variance"][int(t2)] = variance


#: Cross-jump-off panel per stashed overlay family: key -> (figure name, plot fn, title). ``km:*``
#: is the one dynamic family (the outcome name rides in both the figure name and the title), so it
#: is handled separately below rather than bent into this table.
_JUMPOFF_PANELS = {
    "ccf": (
        "ccf_overlay_all_jumpoffs",
        viz_backtest.plot_ccf_jumpoff_panel,
        "CCF by cohort — all jump-offs",
    ),
    "ppr": (
        "ppr_overlay_all_jumpoffs",
        viz_backtest.plot_ppr_jumpoff_panel,
        "Parity progression — all jump-offs",
    ),
    "asfr_cohort": (
        "asfr_overlay_all_jumpoffs",
        viz_backtest.plot_asfr_jumpoff_panel,
        "Cohort ASFR — all jump-offs",
    ),
}


def _emit_jumpoff_panels(panels: dict, out, level: float) -> None:
    """One all-jump-offs comparison figure per overlay family, when there are ≥2 windows to compare.

    With a single window the panel would duplicate that window's own figure, so it is skipped.
    """
    for key, entry in sorted(panels.items()):
        if len(entry["gen"]) < 2:
            continue
        if key.startswith("km:"):
            name = key[len("km:") :]
            out.figure(
                f"km_overlay_{name}_all_jumpoffs",
                viz_backtest.plot_km_jumpoff_panel(
                    entry["obs"], entry["gen"],
                    title=f"{name} survival — all jump-offs", level=level,
                ),
            )
            continue
        figure_name, plot_fn, title = _JUMPOFF_PANELS[key]
        # Only CCF carries a variance frame; the others compute their band from the metric table.
        extra = {"variance_by_jumpoff": entry["variance"]} if key == "ccf" else {}
        out.figure(
            figure_name,
            plot_fn(entry["obs"], entry["gen"], title=title, level=level, **extra),
        )


class _UnsupportedTarget(Exception):
    pass


def _aggregate_tables(
    target,
    gen_births,
    gen_spans,
    obs_births,
    spans_obs,
    persons,
    outcomes,
    cohort_width,
    combined,
    observed,
    fertility_grid,
):
    bins = AgeBins.from_years(*_FERTILE, fertility_grid.age_bin_width)
    if target == "ccf":
        gen = fe.ccf(
            gen_births,
            gen_spans,
            persons,
            by_cohort=True,
            extra_by=_EXTRA_BY,
            cohort_width=cohort_width,
        )
        obs = fe.ccf(obs_births, spans_obs, persons, by_cohort=True, cohort_width=cohort_width)
        return gen, obs, "ccf", ["cohort"]
    if target == "asfr_cohort":
        # Cohort only: the jump-off is an age, so a (cohort, age) cell is wholly forecast or wholly
        # replayed. A (year, age) cell is neither — the generated side has no calendar years before
        # the earliest cohort reaches the jump-off, so the two sides cannot be aligned.
        gen = fe.asfr(
            gen_births,
            gen_spans,
            persons,
            mode="cohort",
            bins=bins,
            extra_by=_EXTRA_BY,
            cohort_width=cohort_width,
        )
        obs = fe.asfr(
            obs_births, spans_obs, persons, mode="cohort", bins=bins, cohort_width=cohort_width
        )
        return gen, obs, "asfr", ["cohort", "age_bin"]
    if target == "ppr":
        gen = fe.ppr(
            gen_births, gen_spans, max_parity=fertility_grid.max_parity, extra_by=_EXTRA_BY
        )
        obs = fe.ppr(obs_births, spans_obs, max_parity=fertility_grid.max_parity)
        return gen, obs, "ppr", ["parity_from"]
    if target.startswith("km:"):
        name = target[len("km:") :]
        gen = _km_at_grid(combined, GEN_KEYS, outcomes[name], by=list(_EXTRA_BY))
        obs = _km_at_grid(observed, OBS_KEYS, outcomes[name], by=[])
        return gen, obs, "survival", ["time"]
    raise _UnsupportedTarget(target)


def _km_at_grid(df, keys, spec, by) -> pd.DataFrame:
    """KM survival at a fixed age grid; tidy ``[*by, time, survival]`` for aggregate_error."""
    from seqeval.units import years_to_days

    tte = time_to_event(df, keys, spec)
    km = sv.kaplan_meier(tte, by=by)
    grid = np.array([years_to_days(y) for y in _KM_GRID_YEARS], dtype=np.int64)
    rows = []
    groups = [((), km)] if not by else list(km.groupby(by, observed=True))
    for key, g in groups:
        g = g.sort_values("time")
        # step function: survival at time = last recorded value at or before that time (1.0 before).
        idx = np.searchsorted(g["time"].to_numpy(), grid, side="right") - 1
        surv = np.where(idx >= 0, g["survival"].to_numpy()[np.clip(idx, 0, len(g) - 1)], 1.0)
        block = pd.DataFrame({"time": grid, "survival": surv})
        if by:
            key_tuple = key if isinstance(key, tuple) else (key,)
            for col, val in zip(by, key_tuple, strict=True):
                block[col] = val
        rows.append(block)
    return pd.concat(rows, ignore_index=True)


# =================================================================================================
# small helpers
# =================================================================================================
def _needs_births(cfg: BacktestingConfig) -> bool:
    """Whether any configured aggregate target requires `birth`s in the model.
    """
    return any(t in FERTILITY_TARGETS for t in cfg.aggregate_targets)


def _cell_label(t1, t2, outcome, condition) -> dict:
    return {
        "age_start": int(t1),
        "age_stop": int(t2),
        "age_start_years": round(days_to_years(t1), 2),
        "age_stop_years": round(days_to_years(t2), 2),
        "outcome": outcome,
        "condition": condition if condition is not None else "-",
    }


def _stamp(df: pd.DataFrame, label: dict) -> pd.DataFrame:
    out = df.copy()
    for col in reversed(list(label)):
        if col not in out.columns:
            out.insert(0, col, label[col])
    return out


def _stamp_scores(scores: dict, label: dict, cis: pd.DataFrame | None = None) -> pd.DataFrame:
    """Long scores rows ``[*label, metric, value, ci_lo, ci_hi]`` (CIs NaN if not bootstrapped)."""
    df = pd.DataFrame([{**label, "metric": k, "value": v} for k, v in scores.items()])
    if cis is not None:
        return df.merge(cis, on="metric", how="left")
    df["ci_lo"] = np.nan
    df["ci_hi"] = np.nan
    return df
