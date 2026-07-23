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
from seqeval.config import DEFAULT_COHORT_WIDTH, BacktestingConfig
from seqeval.core import replicates as rep
from seqeval.core.outcomes import (
    births,
    evaluate_count,
    evaluate_framed,
    observation_spans,
    time_to_event,
)
from seqeval.core.slicing import AgeBins, align_jumpoff_to_event, condition_on_count
from seqeval.core.specs import Condition, CountQuery, FramedOutcome, ReplicateSpec, TTESpec
from seqeval.io.loaders import Bundle
from seqeval.io.schema import GEN_KEYS, OBS_KEYS
from seqeval.metrics import baseline as bl
from seqeval.metrics import fertility as fe
from seqeval.metrics import ml
from seqeval.metrics import survival as sv
from seqeval.units import days_to_years
from seqeval.viz import backtest as viz_backtest
from seqeval.viz import baseline as viz_baseline
from seqeval.viz import calibration as viz_calibration
from seqeval.viz import fertility as viz_fertility
from seqeval.viz._labels import describe_outcome

logger = logging.getLogger("seqeval")

_RUN_KEYS = ["person_id", "age_start", "age_stop"]
_EXTRA_BY = ("seed", "age_start", "age_stop")
_FERTILE = (15.0, 50.0)
# Age grid (years) at which generated vs observed KM survival is compared for `km:*` targets.
_KM_GRID_YEARS = list(range(16, 46, 2))


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
) -> None:
    """Run backtesting over every configured window; write the six result tables (04 section 2.2).

    ``outcomes``/``conditions``/``prob_outcomes``/``replicate_spec`` are the resolved objects from
    ``config.resolve_*`` (passed in, like the descriptives registry, because they live at the top
    level of the config). Writes ``probabilities``, ``calibration``, ``scores``,
    ``aggregate_error``, ``coverage`` and (when configured) ``convergence`` parquet tables.
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
            "convergence",
            "baseline_individual",
            "baseline_scores",
        )
    }

    baseline_ctx = _build_baseline(bundle, cfg, out)

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
                baseline_ctx,
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
            )

    tables = {}
    for name, frames in acc.items():
        if frames:
            tables[name] = pd.concat(frames, ignore_index=True)
            out.frame(name, tables[name])

    # Summary figures: how does each metric move as the jump-off shifts across windows?
    if "scores" in tables and tables["scores"]["age_stop"].nunique() > 1:
        for metric in ("roc_auc", "brier_corrected"):
            out.figure(
                f"metric_vs_jumpoff_{metric}",
                viz_backtest.plot_metric_vs_jumpoff(tables["scores"], metric=metric),
            )
    if "baseline_scores" in tables and tables["baseline_scores"]["age_stop"].nunique() > 1:
        out.figure(
            "baseline_skill_vs_jumpoff",
            viz_baseline.plot_skill_vs_jumpoff(tables["baseline_scores"], metric="brier"),
        )


# =================================================================================================
# ASFR baseline (04 section 2.3)
# =================================================================================================
class _Baseline:
    """What the per-outcome scorer needs to price a person under the observed ASFR schedule."""

    def __init__(
        self,
        schedule: pd.DataFrame,
        persons: pd.DataFrame,
        bins: AgeBins,
        max_unmatched_fraction: float,
    ) -> None:
        self.schedule = schedule
        self.persons = persons
        self.bins = bins
        self.max_unmatched_fraction = max_unmatched_fraction


def _build_baseline(bundle: Bundle, cfg: BacktestingConfig, out: OutputWriter) -> _Baseline | None:
    """Estimate the observed ASFR schedule once and write it (table + surface figure), or skip.

    The schedule is period ASFR over the configured age range, computed from the observed file only.
    It is estimated once for the whole run: freezing happens per person at scoring time (each person
    reads the row of rates at their own jump-off year), so no per-window refit is needed.
    """
    bcfg = cfg.baseline
    if bcfg is None or not bcfg.asfr:
        return None
    if bundle.persons is None:
        logger.warning("backtesting: ASFR baseline skipped — needs a persons file (birth_year)")
        return None
    try:
        birth_token = bundle.token("birth")
    except KeyError:
        logger.warning(
            "backtesting: ASFR baseline skipped — no 'birth' event alias is declared under events:"
        )
        return None

    bins = AgeBins.from_years(*bcfg.age_range, bcfg.age_bin_width)
    schedule = bl.asfr_schedule(
        bundle.observed,
        bundle.persons,
        birth_event=birth_token,
        bins=bins,
        min_person_years=bcfg.min_person_years,
    )
    if schedule.empty or schedule["asfr"].notna().sum() == 0:
        logger.warning(
            "backtesting: ASFR baseline skipped — the observed data yields no usable rate cells "
            "in the age range %s",
            bcfg.age_range,
        )
        return None

    out.frame("baseline_asfr_schedule", schedule)
    out.figure("baseline_asfr_surface", viz_fertility.plot_asfr_surface(schedule, dim="year"))
    logger.info(
        "backtesting: ASFR baseline schedule — %d cells over years %d–%d, ages %g–%g",
        int(schedule["asfr"].notna().sum()),
        int(schedule["year"].min()),
        int(schedule["year"].max()),
        *bcfg.age_range,
    )
    return _Baseline(schedule, bundle.persons, bins, bcfg.max_unmatched_fraction)


def _model_scores(joined: pd.DataFrame) -> dict[str, float]:
    """The model-side statistics the baseline comparison pairs against, on one given population."""
    return {
        "brier_corrected": ml.brier(joined)["corrected"],
        "mse": ml.mse(joined),
        "r2": ml.r2(joined),
        "ece": ml.ece(ml.calibration_table(joined, strategy="quantile")),
        "roc_auc": ml.roc_auc(joined),
    }


def _score_baseline(spec, observed, joined, t2, ctx, label, acc, out, desc) -> None:
    """Score the ASFR baseline on the same persons as the model and record the comparison.

    Both sides are restricted to the persons the baseline can price *and* the model scored, so the
    two columns of every comparison row describe the same population — a skill number computed
    across different denominators would be meaningless.
    """
    probs = bl.baseline_probability(
        observed,
        ctx.persons,
        spec,
        schedule=ctx.schedule,
        jumpoff=t2,
        bins=ctx.bins,
        person_ids=joined["person_id"].unique(),
    )
    if probs.empty:
        return
    ind = joined.merge(probs, on="person_id", how="inner")
    # Persons whose frame is mostly un-priceable (no rate history at or before their jump-off year)
    # are dropped from *both* sides rather than scored against a rate-zero artifact.
    priceable = ind["unmatched_fraction"] <= ctx.max_unmatched_fraction
    n_unpriceable = int((~priceable).sum())
    ind = ind[priceable]
    if ind.empty or ind["y_true"].nunique() < 2:
        logger.debug("baseline %s: no comparable population at jump-off %d", spec.name, t2)
        return

    acc["baseline_individual"].append(_stamp(ind, label))

    comparison = bl.compare(_model_scores(ind), bl.score(ind))
    comparison["n_compared"] = len(ind)
    comparison["n_unpriceable"] = n_unpriceable
    acc["baseline_scores"].append(_stamp(comparison, label))

    jumpoff_y = int(round(days_to_years(t2)))
    cal_model = ml.calibration_table(ind, n_bins=10, strategy="uniform")
    cal_base = ml.calibration_table(
        ind[["p_base", "y_true"]].rename(columns={"p_base": "p_hat"}),
        n_bins=10,
        strategy="uniform",
    )
    out.figure(
        f"baseline_reliability_{spec.name}_w{jumpoff_y}",
        viz_baseline.plot_reliability_overlay(cal_model, cal_base, title=desc),
    )
    out.figure(
        f"baseline_individual_{spec.name}_w{jumpoff_y}",
        viz_baseline.plot_individual_comparison(ind, title=f"Model vs ASFR baseline — {desc}"),
    )
    out.figure(
        f"baseline_lift_{spec.name}_w{jumpoff_y}",
        viz_baseline.plot_lift_by_baseline(ind, title=f"Lift over the ASFR baseline — {desc}"),
    )


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
    baseline_ctx=None,
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

    acc["probabilities"].append(_stamp(probs, label))

    cal = ml.calibration_table(joined, n_bins=10, strategy="uniform")
    band = rep.null_calibration_band(
        summary,
        n_bins=10,
        strategy="uniform",
        n_sims=200,
        rng=np.random.default_rng(replicate_spec.bootstrap_seed),
        estimator=replicate_spec.estimator,
    )
    cal = cal.merge(
        band[["bin", "lo", "hi"]].rename(columns={"lo": "band_lo", "hi": "band_hi"}),
        on="bin",
        how="left",
    )
    acc["calibration"].append(_stamp(cal, label))

    desc = describe_outcome(spec, jumpoff_days=t2, label_fn=label_fn)
    out.figure(
        f"reliability_{spec.name}_w{int(round(days_to_years(t2)))}",
        viz_calibration.plot_reliability(cal, probs=probs, title=desc),
    )
    # Timed outcomes also get a waiting-time calibration scatter (predicted vs observed duration).
    if isinstance(spec, FramedOutcome):
        _emit_timing_calibration(spec, gen_w, observed, t1, t2, out, desc)

    if baseline_ctx is not None:
        _score_baseline(spec, observed, joined, t2, baseline_ctx, label, acc, out, desc)

    scores = _score_row(spec, joined, summary, gen_w, observed, t1, t2)
    cis = _score_cis(gen_eval, obs_eval, replicate_spec)
    acc["scores"].append(_stamp_scores(scores, label, cis))

    if replicate_spec.convergence_curve:
        conv = _convergence(gen_eval, obs_eval, replicate_spec)
        if conv is not None:
            acc["convergence"].append(_stamp(conv, label))


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


def _score_row(spec, joined, summary, gen_w, observed, t1, t2) -> dict:
    brier = ml.brier(joined)
    median_n = float(summary["n"].median())
    scores = {
        "ece": ml.ece(ml.calibration_table(joined, strategy="quantile")),
        "roc_auc": ml.roc_auc(joined),
        "auc_grid_resolution": 1.0 / median_n if median_n else np.nan,
        "brier_raw": brier["raw"],
        "brier_corrected": brier["corrected"],
        "mse": ml.mse(joined),
        "r2": ml.r2(joined),
    }
    if isinstance(spec, FramedOutcome):
        scores["timing_coverage"] = _timing_coverage(spec, gen_w, observed, t1, t2)
    return scores


def _score_cis(gen_eval, obs_eval, spec) -> pd.DataFrame | None:
    """95% seed-bootstrap CIs for the ML metrics using **all** seeds, as ``[metric, ci_lo, ci_hi]``.

    Resamples seed labels with replacement (``spec.bootstrap_n`` draws) and recomputes each metric,
    giving the Monte-Carlo (replicate) uncertainty on the headline scores. Returns ``None`` when
    bootstrapping is disabled or the truth column is single-valued (no metric is defined).
    """
    if spec.bootstrap_n <= 0:
        return None
    truth = obs_eval.loc[obs_eval["evaluable"], ["person_id", "occurred"]].rename(
        columns={"occurred": "y_true"}
    )
    truth["y_true"] = truth["y_true"].astype(int)
    if truth["y_true"].nunique() < 2:
        return None

    metrics = ["roc_auc", "brier_corrected", "mse", "r2", "ece"]

    def stat_fn(df: pd.DataFrame) -> pd.DataFrame:
        summ = rep.replicate_summary(df, run_keys=_RUN_KEYS)
        est = rep.estimate_probability(summ, spec=spec)
        j = est.merge(truth, on="person_id", how="inner")
        if j.empty or j["y_true"].nunique() < 2:
            return pd.DataFrame({m: [np.nan] for m in metrics})
        return pd.DataFrame(
            {
                "roc_auc": [ml.roc_auc(j)],
                "brier_corrected": [ml.brier(j)["corrected"]],
                "mse": [ml.mse(j)],
                "r2": [ml.r2(j)],
                # match _score_row's estimator exactly (quantile bins) so the point sits on the
                # same statistic as its CI.
                "ece": [ml.ece(ml.calibration_table(j, strategy="quantile"))],
            }
        )

    boot = rep.seed_bootstrap(
        gen_eval,
        seed_col="seed",
        stat_fn=stat_fn,
        n_boot=spec.bootstrap_n,
        rng=np.random.default_rng(spec.bootstrap_seed),
        value_cols=metrics,
    )
    return boot[["metric", "ci_lo", "ci_hi"]]


def _timing_tables(spec, gen_w, observed, t1, t2):
    """``(timing_distribution, observed tte, horizon_days)`` for a framed (timed) outcome."""
    combined = combine_prefix(observed, gen_w, t1, t2)
    tte_gen = time_to_event(combined, GEN_KEYS, spec.tte)
    horizon = spec.frame.value
    td = rep.timing_distribution(tte_gen, run_keys=_RUN_KEYS, seed_col="seed", horizon=horizon)
    obs_tte = time_to_event(observed, OBS_KEYS, spec.tte)
    return td, obs_tte, horizon


def _timing_coverage(spec, gen_w, observed, t1, t2) -> float:
    td, obs_tte, _ = _timing_tables(spec, gen_w, observed, t1, t2)
    return ml.timing_coverage(td, obs_tte)


def _emit_timing_calibration(spec, gen_w, observed, t1, t2, out, desc) -> None:
    """Predicted-vs-observed waiting-time scatter for a timed outcome (one per jump-off)."""
    td, obs_tte, horizon = _timing_tables(spec, gen_w, observed, t1, t2)
    out.figure(
        f"timing_calibration_{spec.name}_w{int(round(days_to_years(t2)))}",
        viz_backtest.plot_timing_calibration(
            td, obs_tte, horizon_days=horizon, title=f"Waiting time — {desc}"
        ),
    )


def _convergence(gen_eval, obs_eval, spec) -> pd.DataFrame | None:
    truth = obs_eval.loc[obs_eval["evaluable"], ["person_id", "occurred"]].rename(
        columns={"occurred": "y_true"}
    )
    truth["y_true"] = truth["y_true"].astype(int)

    def stat_fn(df: pd.DataFrame) -> pd.DataFrame:
        summ = rep.replicate_summary(df, run_keys=_RUN_KEYS)
        est = rep.estimate_probability(summ, spec=spec)
        j = est.merge(truth, on="person_id", how="inner")
        if j.empty or j["y_true"].nunique() < 2:
            return pd.DataFrame({"auc": [np.nan], "brier": [np.nan], "ece": [np.nan]})
        return pd.DataFrame(
            {
                "auc": [ml.roc_auc(j)],
                "brier": [ml.brier(j)["corrected"]],
                "ece": [ml.ece(ml.calibration_table(j, strategy="uniform"))],
            }
        )

    n_seeds = int(gen_eval["seed"].nunique())
    if n_seeds < 3:
        return None
    sizes = sorted({2, 3, 5, 10, n_seeds} & set(range(2, n_seeds + 1)))
    return rep.convergence_curve(
        gen_eval,
        seed_col="seed",
        stat_fn=stat_fn,
        sizes=sizes,
        n_rep=10,
        rng=np.random.default_rng(spec.bootstrap_seed),
        value_cols=["auc", "brier", "ece"],
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
) -> None:
    if persons is None and target != "ppr":
        logger.warning("backtesting: skipping aggregate target %r — needs persons", target)
        return
    combined = combine_prefix(observed, gen_w, t1, t2)
    gen_births = births(combined, GEN_KEYS, birth_event=birth_token)
    gen_spans = observation_spans(combined, GEN_KEYS)
    obs_births = births(observed, OBS_KEYS, birth_event=birth_token)

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
        )
    except _UnsupportedTarget:
        logger.warning("backtesting: aggregate target %r not supported; skipped", target)
        return

    err = ml.aggregate_error(gen_m, obs_m, value_col=value_col, on=on, spec=replicate_spec)
    err.insert(0, "target", target)
    acc["aggregate_error"].append(_stamp(err, _cell_label(t1, t2, target, None)))

    # Generated-vs-observed overlays: the observed "truth" under the generated across-seed band.
    jumpoff_y = round(days_to_years(t2))
    if target.startswith("km:"):
        _emit_km_overlay(target[len("km:") :], combined, observed, outcomes, t2, out)
    elif target == "ccf":
        out.figure(
            f"ccf_overlay_w{jumpoff_y}",
            viz_backtest.plot_ccf_seed_band(
                obs_m, gen_m, title=f"CCF by cohort — jump-off {jumpoff_y}y"
            ),
        )


def _emit_km_overlay(name, combined, observed, outcomes, t2, out) -> None:
    """Emit the observed KM curve overlaid with the generated across-seed median + IQR band."""
    spec = outcomes[name]
    obs_km = sv.kaplan_meier(time_to_event(observed, OBS_KEYS, spec), by=[])
    gen_km = sv.kaplan_meier(time_to_event(combined, GEN_KEYS, spec), by=["seed"])
    jumpoff_y = round(days_to_years(t2))
    out.figure(
        f"km_overlay_{name}_w{jumpoff_y}",
        viz_backtest.plot_km_seed_band(
            obs_km, gen_km, title=f"{name} survival — jump-off {jumpoff_y}y"
        ),
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
):
    bins = AgeBins.from_years(*_FERTILE, 1.0)
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
    if target in ("asfr_cohort", "asfr_period"):
        mode = "cohort" if target == "asfr_cohort" else "period"
        gen = fe.asfr(
            gen_births,
            gen_spans,
            persons,
            mode=mode,
            bins=bins,
            extra_by=_EXTRA_BY,
            cohort_width=cohort_width,
        )
        obs = fe.asfr(
            obs_births, spans_obs, persons, mode=mode, bins=bins, cohort_width=cohort_width
        )
        dim = "cohort" if mode == "cohort" else "year"
        return gen, obs, "asfr", [dim, "age_bin"]
    if target == "ppr":
        gen = fe.ppr(gen_births, gen_spans, max_parity=6, extra_by=_EXTRA_BY)
        obs = fe.ppr(obs_births, spans_obs, max_parity=6)
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
    return bool(cfg.aggregate_targets)


def _baseline_enabled(cfg: BacktestingConfig) -> bool:
    return cfg.baseline is not None and cfg.baseline.asfr


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
