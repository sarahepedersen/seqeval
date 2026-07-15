"""Future/generated forecasting arm: Lexis surfaces, illegal moves, seed stability (05).

Evaluates generated futures with **no ground truth**: it completes the Lexis surface for incomplete
cohorts (observed cells + model-forecast cells), screens output for demographically impossible or
implausible "illegal moves" (a data-driven rules engine), and quantifies seed-to-seed stability of
trajectories as views over the replicate engine (02b) — this arm holds no statistics of its own.

Forecasting wants the longest futures, so it uses every generated window by default; point it at
conditions-at-birth / late-jump-off runs with a ``windows:`` filter (same semantics as 04). The
Lexis forecast is built from the earliest jump-off available (the longest future). Seed stability
and illegal-move screening run across all resolved windows.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from seqeval.arms._common import OutputWriter, combine_prefix
from seqeval.config import DEFAULT_COHORT_WIDTH, ForecastingConfig
from seqeval.core import replicates as rep
from seqeval.core.outcomes import births, observation_spans, time_to_event
from seqeval.core.slicing import AgeBins
from seqeval.core.specs import ReplicateSpec, Rule, TTESpec
from seqeval.io.loaders import Bundle
from seqeval.io.schema import GEN_KEYS, OBS_KEYS
from seqeval.metrics import fertility as fe
from seqeval.metrics import plausibility as pl
from seqeval.viz import lexis as viz_lexis

logger = logging.getLogger("seqeval")

_RUN_KEYS = ["person_id", "age_start", "age_stop"]


def run(
    bundle: Bundle,
    cfg: ForecastingConfig,
    out: OutputWriter,
    *,
    outcomes: dict[str, TTESpec],
    rules: list[Rule],
    replicate_spec: ReplicateSpec,
    cohort_width: int = DEFAULT_COHORT_WIDTH,
) -> None:
    """Run the forecasting arm; write Lexis surfaces, violations, and seed-stability tables.

    ``outcomes``/``rules``/``replicate_spec`` are the resolved objects from ``config.resolve_*``
    (passed in, like the other arms). Writes both period and cohort Lexis surfaces
    (``lexis_{observed,forecast,combined}`` and ``lexis_cohort_{...}``), ``violations``,
    ``violation_rates`` and ``seed_stability_{individual,aggregate}`` parquet tables plus figures.
    """
    if bundle.generated is None:
        logger.warning("forecasting: no generated file; arm skipped")
        return

    from seqeval.config import resolve_windows

    windows = resolve_windows(cfg.windows, bundle.available_windows())
    wanted = set(windows)
    generated = bundle.generated[
        [
            (int(s), int(e)) in wanted
            for s, e in zip(
                bundle.generated["age_start"], bundle.generated["age_stop"], strict=True
            )
        ]
    ]
    observed = bundle.observed

    if cfg.lexis is not None:
        _run_lexis(bundle, cfg, generated, windows, out, outcomes, cohort_width)
    if cfg.illegal_moves:
        _run_illegal_moves(observed, generated, rules, out)
    if cfg.seed_stability is not None:
        _run_seed_stability(bundle, cfg, generated, windows, outcomes, replicate_spec, out)


# =================================================================================================
# 1. Lexis
# =================================================================================================
def _run_lexis(bundle, cfg, generated, windows, out, outcomes, cohort_width) -> None:
    if bundle.persons is None:
        logger.warning("forecasting: skipping Lexis — no persons file (birth_year needed)")
        return
    lex = cfg.lexis
    spec = outcomes[lex.outcome]
    occurrence, target = spec.occurrence, spec.target
    bins = AgeBins.from_years(lex.ages[0], lex.ages[1], 1.0)
    year_range = (lex.years[0], lex.years[1])
    subgroup = list(lex.subgroup_by)

    persons, observed = bundle.persons, bundle.observed
    obs_b = births(observed, OBS_KEYS, birth_event=target)
    obs_spans = observation_spans(observed, OBS_KEYS)

    # Forecast from the earliest jump-off (longest future). The occurrence is absolute, so the
    # forecast surface is built on the full life course (observed prefix + generated future) — a
    # first birth is a *first* birth, not the first post-jump-off birth.
    t1, t2 = min(windows, key=lambda w: w[1])
    gen_w = generated[(generated["age_start"] == t1) & (generated["age_stop"] == t2)]
    combined_seq = combine_prefix(observed, gen_w, t1, t2)
    gen_b = births(combined_seq, GEN_KEYS, birth_event=target)
    gen_spans = observation_spans(combined_seq, GEN_KEYS)

    # Both bases: period (year x age) and cohort (birth-cohort x age).
    for basis, dim in (("period", "year"), ("cohort", "cohort")):
        prefix = "lexis" if basis == "period" else "lexis_cohort"
        obs_surface = fe.lexis_surface(
            obs_b,
            obs_spans,
            persons,
            occurrence=occurrence,
            bins=bins,
            year_range=year_range,
            extra_by=tuple(subgroup),
            basis=basis,
            cohort_width=cohort_width,
        )
        fc_by_seed = fe.lexis_surface(
            gen_b,
            gen_spans,
            persons,
            occurrence=occurrence,
            bins=bins,
            year_range=year_range,
            extra_by=("seed", *subgroup),
            basis=basis,
            cohort_width=cohort_width,
        )
        combined = _combine_surfaces(obs_surface, fc_by_seed, dim, subgroup)

        out.frame(f"{prefix}_observed", obs_surface)
        out.frame(f"{prefix}_forecast", fc_by_seed)
        out.frame(f"{prefix}_combined", combined)
        out.figure(
            f"{prefix}_combined", viz_lexis.plot_lexis(combined, dim=dim, mark_forecast=True)
        )
        if fc_by_seed["seed"].nunique() > 1:
            out.figure(
                f"{prefix}_uncertainty", viz_lexis.plot_lexis_uncertainty(fc_by_seed, dim=dim)
            )


def _combine_surfaces(observed_surface, forecast_by_seed, dim, subgroup) -> pd.DataFrame:
    """Observed cells (source=observed) + seed-median forecast cells absent from observed."""
    cell = [dim, "age_bin", *subgroup]
    fc_median = (
        forecast_by_seed.groupby(cell, observed=True)
        .agg(
            rate=("rate", "median"),
            n_events=("n_events", "median"),
            person_years=("person_years", "median"),
        )
        .reset_index()
    )
    obs = observed_surface.copy()
    obs["source"] = "observed"
    observed_cells = set(map(tuple, observed_surface[cell].to_numpy()))
    fc_new = fc_median[[tuple(r) not in observed_cells for r in fc_median[cell].to_numpy()]].copy()
    fc_new["source"] = "forecast"
    return pd.concat([obs, fc_new], ignore_index=True).sort_values(cell).reset_index(drop=True)


# =================================================================================================
# 2. illegal moves
# =================================================================================================
def _run_illegal_moves(observed, generated, rules, out) -> None:
    gen_viol = pl.check_rules(generated, GEN_KEYS, rules)
    obs_viol = pl.check_rules(observed, OBS_KEYS, rules)

    gen_viol = gen_viol.assign(source="generated")
    obs_viol = obs_viol.assign(source="observed")
    # align columns (observed has no seed/window keys) before stacking
    out.frame("violations", pd.concat([gen_viol, obs_viol], ignore_index=True))

    gen_rates = pl.violation_rates(gen_viol, generated, GEN_KEYS, by=("seed",)).assign(
        source="generated"
    )
    obs_rates = pl.violation_rates(obs_viol, observed, OBS_KEYS, by=()).assign(source="observed")
    out.frame("violation_rates", pd.concat([gen_rates, obs_rates], ignore_index=True))


# =================================================================================================
# 3. seed stability (views over 02b)
# =================================================================================================
def _run_seed_stability(bundle, cfg, generated, windows, outcomes, spec, out) -> None:
    scfg = cfg.seed_stability
    target_name = cfg.lexis.outcome if cfg.lexis is not None else next(iter(outcomes))
    tte_spec = outcomes[target_name]
    birth_token = tte_spec.target
    horizon = _fertile_upper_days()

    if scfg.individual:
        ind = _seed_stability_individual(generated, tte_spec, birth_token, horizon, spec)
        out.frame("seed_stability_individual", ind)

    if scfg.aggregate:
        agg = _seed_stability_aggregate(
            bundle, generated, windows, scfg.aggregate, birth_token, spec
        )
        if agg is not None:
            out.frame("seed_stability_aggregate", agg)


def _seed_stability_individual(generated, tte_spec, birth_token, horizon, spec) -> pd.DataFrame:
    """Per (person, window): occurrence disagreement p_hat(1-p_hat), timing IQR, count variance."""
    tte = time_to_event(generated, GEN_KEYS, tte_spec)

    # (a) occurrence disagreement — Bernoulli variance of "target occurs within horizon".
    occ = tte[_RUN_KEYS].copy()
    occ["occurred"] = tte["observed"].to_numpy() & (tte["duration"].to_numpy() <= horizon)
    summary = rep.replicate_summary(occ, run_keys=_RUN_KEYS)
    est = rep.estimate_probability(summary, spec=spec)
    est["disagreement"] = est["p_hat"] * (1 - est["p_hat"])

    # (b) timing dispersion — q10-q90 spread of the age at first occurrence.
    td = rep.timing_distribution(tte, run_keys=_RUN_KEYS, seed_col="seed", horizon=horizon)
    td["timing_spread"] = td["q90"] - td["q10"]

    # (c) count dispersion — predictive variance of the completed event count.
    counts = _counts_per_run(generated, birth_token)
    cm = rep.count_moments(counts, run_keys=_RUN_KEYS, seed_col="seed").rename(
        columns={"mean": "count_mean", "var": "count_var"}
    )

    out = (
        est[[*_RUN_KEYS, "p_hat", "disagreement"]]
        .merge(td[[*_RUN_KEYS, "timing_spread", "p_within_horizon"]], on=_RUN_KEYS, how="left")
        .merge(cm, on=_RUN_KEYS, how="left")
    )
    return out.sort_values(_RUN_KEYS).reset_index(drop=True)


def _counts_per_run(generated, birth_token) -> pd.DataFrame:
    """Number of target events per replicate ``[*RUN_KEYS, seed, count]`` (0 for empty runs)."""
    runs = generated[[*GEN_KEYS]].drop_duplicates()
    got = (
        generated[generated["event"] == birth_token]
        .groupby(GEN_KEYS, observed=True)
        .size()
        .rename("count")
        .reset_index()
    )
    counts = runs.merge(got, on=GEN_KEYS, how="left")
    counts["count"] = counts["count"].fillna(0).astype(np.int64)
    return counts


def _seed_stability_aggregate(bundle, generated, windows, targets, birth_token, spec):
    """Seed-bootstrap uncertainty bands on aggregate metrics (currently CCF) per window."""
    if "ccf" not in targets or bundle.persons is None:
        return None
    persons = bundle.persons
    frames = []
    for t1, t2 in windows:
        gen_w = generated[(generated["age_start"] == t1) & (generated["age_stop"] == t2)]
        if gen_w.empty:
            continue
        combined = combine_prefix(bundle.observed, gen_w, t1, t2)

        def ccf_stat(df, _persons=persons, _token=birth_token):
            b = df[df["event"] == _token]
            n_seeds = df["seed"].nunique()
            n_persons = df["person_id"].nunique()
            return pd.DataFrame({"ccf": [len(b) / n_seeds / n_persons]})

        boot = rep.seed_bootstrap(
            combined,
            seed_col="seed",
            stat_fn=ccf_stat,
            n_boot=max(spec.bootstrap_n, 100),
            rng=np.random.default_rng(spec.bootstrap_seed),
            value_cols=["ccf"],
        )
        boot.insert(0, "age_stop", int(t2))
        boot.insert(0, "age_start", int(t1))
        frames.append(boot)
    return pd.concat(frames, ignore_index=True) if frames else None


def _fertile_upper_days() -> int:
    from seqeval.units import years_to_days

    return years_to_days(fe.FERTILE_UPPER_YEARS)
