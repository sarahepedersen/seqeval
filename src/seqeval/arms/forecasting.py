"""Future/generated forecasting arm: Lexis surfaces, illegal moves, replicate variance (05).

Evaluates generated futures with **no ground truth**: it completes the Lexis surface for incomplete
cohorts (observed cells + model-forecast cells), screens output for demographically impossible or
implausible "illegal moves" (a data-driven rules engine), and quantifies replicate-to-replicate variance of trajectories.

Forecasting wants the longest futures, so it uses every generated window by default; point it at
conditions-at-birth / late-jump-off runs with a ``windows:`` filter (same semantics as 04). The
Lexis forecast is built from the earliest jump-off available (the longest future). Replicate
variance and illegal-move screening run across all resolved windows.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.stats import norm

from seqeval.arms._common import OutputWriter, combine_prefix
from seqeval.config import DEFAULT_COHORT_WIDTH, ForecastingConfig
from seqeval.core import replicates as rep
from seqeval.core.outcomes import births, observation_spans, time_to_event
from seqeval.core.slicing import AgeBins, cohort_bins
from seqeval.core.specs import ReplicateSpec, Rule, TTESpec
from seqeval.io.loaders import Bundle
from seqeval.io.schema import GEN_KEYS, OBS_KEYS
from seqeval.metrics import dispersion as md
from seqeval.metrics import fertility as fe
from seqeval.metrics import plausibility as pl
from seqeval.viz import dispersion as viz_dispersion
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
    """Run the forecasting arm; write Lexis surfaces, violations, and replicate-variance tables.

    ``outcomes``/``rules``/``replicate_spec`` are the resolved objects from ``config.resolve_*``
    (passed in, like the other arms). Writes both period and cohort Lexis surfaces
    (``lexis_{observed,forecast,combined}`` and ``lexis_cohort_{...}``), ``violations``,
    ``violation_rates`` and ``replicate_variance_{individual,aggregate}`` tables plus figures.
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
    if cfg.replicate_variance is not None:
        _run_replicate_variance(
            bundle, cfg, generated, windows, outcomes, replicate_spec, out, cohort_width
        )


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
            f"{prefix}_combined",
            viz_lexis.plot_lexis(combined, dim=dim, mark_forecast=True, outcome=lex.outcome),
        )
        if fc_by_seed["seed"].nunique() > 1:
            out.figure(
                f"{prefix}_uncertainty",
                viz_lexis.plot_lexis_uncertainty(fc_by_seed, dim=dim, outcome=lex.outcome),
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
    # violation_rates below is the aggregate this table backs; only the row-level examples go.
    out.frame("violations", pd.concat([gen_viol, obs_viol], ignore_index=True), individual=True)

    gen_rates = pl.violation_rates(gen_viol, generated, GEN_KEYS, by=("seed",)).assign(
        source="generated"
    )
    obs_rates = pl.violation_rates(obs_viol, observed, OBS_KEYS, by=()).assign(source="observed")
    out.frame("violation_rates", pd.concat([gen_rates, obs_rates], ignore_index=True))


# =================================================================================================
# 3. replicate variance (per-individual dispersion + upstream-metric roll-up)
# =================================================================================================
def _run_replicate_variance(
    bundle, cfg, generated, windows, outcomes, spec, out, cohort_width
) -> None:
    scfg = cfg.replicate_variance
    target_name = cfg.lexis.outcome if cfg.lexis is not None else next(iter(outcomes))
    tte_spec = outcomes[target_name]
    birth_token = tte_spec.target
    horizon = _fertile_upper_days()

    if scfg.individual:
        subgroups = _person_subgroups(bundle.persons, scfg.subgroup_by, cohort_width)
        ind = _replicate_variance_individual(generated, birth_token, subgroups)
        out.frame("replicate_variance_individual", ind, individual=True)
        out.frame(
            "replicate_occurrence",
            _replicate_occurrence(generated, tte_spec, target_name, horizon, spec),
            individual=True,
        )
        _emit_dispersion_ridges(ind, scfg.subgroup_by, out)

    if scfg.aggregate:
        agg = _replicate_variance_aggregate(
            bundle, generated, windows, scfg.aggregate, birth_token, spec, cohort_width
        )
        if agg is not None:
            out.frame("replicate_variance_aggregate", agg)


def _replicate_occurrence(generated, tte_spec, outcome_name, horizon, spec) -> pd.DataFrame:
    """Per-(person, jump-off) whether and when ``outcome_name`` occurs within ``horizon``.

    Everything the replicates say about one *named* outcome: ``[outcome, *_RUN_KEYS, horizon, n,
    n_occurred, p_hat, timing_spread]``. ``outcome`` is the configured outcome name and ``horizon``
    the cut-off the event must fall inside (days), so a row states which event it is about without
    reference to the calling context.

    ``p_hat = n_occurred/n`` is the raw replicate frequency. ``timing_spread`` is the ``q90 - q10``
    width of the predicted age at occurrence — how much the replicates disagree about *when*, given
    that it happens.
    """
    tte = time_to_event(generated, GEN_KEYS, tte_spec)

    occ = tte[_RUN_KEYS].copy()
    occ["occurred"] = tte["observed"].to_numpy() & (tte["duration"].to_numpy() <= horizon)
    est = rep.estimate_probability(rep.replicate_summary(occ, run_keys=_RUN_KEYS), spec=spec)

    td = rep.timing_distribution(tte, run_keys=_RUN_KEYS, seed_col="seed", horizon=horizon)
    td["timing_spread"] = td["q90"] - td["q10"]

    out = est[[*_RUN_KEYS, "n", "k", "p_hat"]].rename(columns={"k": "n_occurred"})
    out["horizon"] = int(horizon)
    out.insert(0, "outcome", outcome_name)
    out = out.merge(td[[*_RUN_KEYS, "timing_spread"]], on=_RUN_KEYS, how="left")
    out = _restrict_to_common_windows(out)  # same population as the dispersion table
    cols = ["outcome", *_RUN_KEYS, "horizon", "n", "n_occurred", "p_hat", "timing_spread"]
    return out[cols].sort_values(_RUN_KEYS).reset_index(drop=True)


def _emit_dispersion_ridges(ind: pd.DataFrame, subgroup_by, out) -> None:
    """Within-seed dispersion as binned distributions, plus one ridge figure per split.

    The population figure stacks one ridge per jump-off; each requested subgroup adds a figure whose
    ridges are that subgroup's values, one panel per jump-off — so a cohort's dispersion can be read
    against the other cohorts at a jump-off, and against itself as the jump-off moves later.
    """
    pop = md.dispersion_distribution(ind, by=["age_stop"], min_cell=out.min_cell)
    out.frame("within_seed_variance_distribution", pop)
    out.figure(
        "within_seed_variance",
        viz_dispersion.plot_within_seed_variance(pop, x="age_stop", min_cell=out.min_cell),
    )
    for col in subgroup_by:
        if col not in ind.columns:
            continue
        dist = md.dispersion_distribution(ind, by=[col, "age_stop"], min_cell=out.min_cell)
        out.frame(f"within_seed_variance_distribution_by_{col}", dist)
        out.figure(
            f"within_seed_variance_by_{col}",
            viz_dispersion.plot_within_seed_variance(
                dist, x=col, facet_by="age_stop", min_cell=out.min_cell
            ),
        )


def _replicate_variance_individual(generated, birth_token, subgroups=None) -> pd.DataFrame:
    """Per-(person, jump-off) replicate dispersion of the completed ``birth_token`` count.

    ``within_seed_var`` / ``within_seed_cv`` are the variance and coefficient of variation across a
    person's replicates for quantum completed fertility.
    """
    ind_counts = _counts_per_run(generated, birth_token)
    ind_cm = rep.count_moments(ind_counts, run_keys=_RUN_KEYS, seed_col="seed").rename(
        columns={"mean": "expected_quantum", "var": "within_seed_var"}
    )
    mu = ind_cm["expected_quantum"].to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        cv = np.sqrt(ind_cm["within_seed_var"].to_numpy()) / mu
    ind_cm["within_seed_cv"] = np.where(mu > 0, cv, np.nan)

    out = ind_cm[[*_RUN_KEYS, "expected_quantum", "within_seed_var", "within_seed_cv"]]
    out = _restrict_to_common_windows(out)
    if subgroups is not None:
        out = out.merge(subgroups, on="person_id", how="left")
    return out.sort_values(_RUN_KEYS).reset_index(drop=True)


def _person_subgroups(persons, subgroup_by, cohort_width) -> pd.DataFrame | None:
    """``[person_id, *subgroup_by]`` map; ``cohort`` via :func:`cohort_bins`, else from persons."""
    if not subgroup_by or persons is None:
        return None
    out = persons[["person_id"]].copy()
    for col in subgroup_by:
        if col == "cohort":
            out = out.merge(cohort_bins(persons, width=cohort_width).reset_index(), on="person_id")
        else:
            out = out.merge(persons[["person_id", col]], on="person_id")
    return out


def _restrict_to_common_windows(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only persons that appear in every (age_start, age_stop) window."""
    n_windows = df[["age_start", "age_stop"]].drop_duplicates().shape[0]
    per_person = df.groupby("person_id").size()
    return df[df["person_id"].isin(per_person[per_person == n_windows].index)]


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


def _replicate_variance_aggregate(
    bundle, generated, windows, targets, birth_token, spec, cohort_width
):
    """Analytic replicate uncertainty for CCF (completed cohort fertility), per cohort per prediction window.

    Utilizes each person's expected completed fertility and replicate variance to estimate
    ``CCF = mean_i mu_i`` for each cohort, splitting the variance of that estimate into
    ``within_var`` (inference uncertainty) and ``between_var`` (outcome uncertainty), which add to
    ``total_var`` — the variance ``se_total`` and ``ci_total`` report. See :func:`_ccf_row`.

    ``forecast_share`` is the fraction of each estimate contributed by post-jump-off generated
    events: 0.0 rests entirely on observed history, 1.0 entirely on model output.
    """
    if "ccf" not in targets or bundle.persons is None:
        return None
    cohorts = cohort_bins(bundle.persons, width=cohort_width).reset_index()  # [person_id, cohort]
    z = norm.ppf(1 - (1 - spec.level) / 2)

    frames = []
    for t1, t2 in windows:
        gen_w = generated[(generated["age_start"] == t1) & (generated["age_stop"] == t2)]
        if gen_w.empty:
            continue
        combined = combine_prefix(bundle.observed, gen_w, t1, t2)
        counts = _counts_per_run(combined, birth_token)
        moments = rep.count_moments(counts, run_keys=["person_id"], seed_col="seed").rename(
            columns={"mean": "mu", "var": "s2", "k": "K"}
        )
        # mu_gen: the post-jump-off (model-generated) part of each person's expected count. The
        # prefix is the same in every seed, so mu - mu_gen is that person's observed births.
        mu_gen = (
            _counts_per_run(gen_w, birth_token)
            .groupby("person_id", observed=True)["count"]
            .mean()
            .rename("mu_gen")
        )
        moments = moments.merge(cohorts, on="person_id").merge(mu_gen, on="person_id")

        rows = [
            {"cohort": int(c), **_ccf_row(sub, z)}
            for c, sub in moments.groupby("cohort", observed=True)
        ]
        rows.append({"cohort": pd.NA, **_ccf_row(moments, z)})  # pooled
        frame = pd.DataFrame(rows)
        frame["cohort"] = frame["cohort"].astype("Int64")
        frame.insert(0, "age_stop", int(t2))
        frame.insert(0, "age_start", int(t1))
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else None


def _ccf_row(sub: pd.DataFrame, z: float) -> dict:
    """One CCF point estimate, the variance behind it split by source, and its forecast provenance.

    Each woman contributes ``mu_i`` (her expected completed fertility over ``K_i`` seeds) and
    ``s2_i`` (her variance across those seeds). :func:`~seqeval.core.replicates.
    mean_variance_components` splits the variance of ``CCF = mean_i mu_i`` into ``within_var``
    (seeds) and ``between_var`` (population heterogeneity), both in variance units *of the CCF
    itself* — the same scale as the interval, not the scale of one woman's count. They add to
    ``total_var``, which is what ``se_total``, ``ci_total`` and the CCF figure band all report; the
    replicate-only interval stays available as ``ccf ± z*sqrt(within_var)``.
    """
    comp = rep.mean_variance_components(sub["mu"], sub["s2"], sub["K"])
    ccf = comp["mean"]
    ccf_forecast = float(sub["mu_gen"].to_numpy().mean())  # model-contributed births per woman
    se_total = float(np.sqrt(comp["total_var"]))
    return {
        "n_women": comp["n"],
        "ccf": ccf,
        "within_var": comp["within_var"],
        "between_var": comp["between_var"],
        "total_var": comp["total_var"],
        "se_total": se_total,
        "ci_total_lo": ccf - z * se_total,
        "ci_total_hi": ccf + z * se_total,
        "forecast_share": float(ccf_forecast / ccf) if ccf > 0 else np.nan,
    }


def _fertile_upper_days() -> int:
    from seqeval.units import years_to_days

    return years_to_days(fe.FERTILE_UPPER_YEARS)
