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

from seqeval.arms._common import OutputWriter, combine_prefix, pool_seeds
from seqeval.config import DEFAULT_COHORT_WIDTH, ForecastingConfig
from seqeval.core import replicates as rep
from seqeval.core.outcomes import births, observation_spans, time_to_event
from seqeval.core.slicing import AgeBins, cohort_bins
from seqeval.core.specs import ReplicateSpec, Rule, TTESpec
from seqeval.io.loaders import Bundle
from seqeval.io.schema import GEN_KEYS, OBS_KEYS, RUN_KEYS
from seqeval.metrics import dispersion as md
from seqeval.metrics import fertility as fe
from seqeval.metrics import plausibility as pl
from seqeval.metrics import pooling
from seqeval.metrics import sequences as sq
from seqeval.viz import dispersion as viz_dispersion
from seqeval.viz import lexis as viz_lexis
from seqeval.viz import sequences as viz_sequences

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
    (passed in, like the other arms). Writes the cohort Lexis surface
    (``lexis_cohort_{observed,forecast,combined}``), ``violations``,
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
        _run_lexis(
            bundle, cfg, generated, windows, out, outcomes, cohort_width, replicate_spec.level
        )
    if cfg.illegal_moves:
        _run_illegal_moves(observed, generated, rules, out)
    # One analysis per configured block, each about its own event and each writing stems suffixed
    # with the block's name, so several can coexist in one results directory.
    for block in cfg.replicate_variance:
        _run_replicate_variance(
            bundle, block, cfg, generated, windows, outcomes, replicate_spec, out, cohort_width
        )
    if cfg.sequence_descriptives:
        _run_sequence_descriptives(bundle, generated, windows, out, cohort_width)


# =================================================================================================
# 1. Lexis
# =================================================================================================
def _run_lexis(bundle, cfg, generated, windows, out, outcomes, cohort_width, level) -> None:
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

    # Cohort basis only (birth-cohort x age). The jump-off is an age, so a cohort-indexed cell is
    # wholly observed or wholly forecast; a calendar-year cell is neither, and the forecast region
    # in period space starts at a different year for every cohort.
    prefix, basis, dim = "lexis_cohort", "cohort", "cohort"
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

    # Each seed is its own synthetic population; the surface drawn is the one estimate over every
    # trajectory at once, with no per-cell aggregate across seeds underneath it (05b).
    pooled_seq, persons_pooled = pool_seeds(combined_seq, persons)
    fc_pooled = fe.lexis_surface(
        births(pooled_seq, RUN_KEYS, birth_event=target),
        observation_spans(pooled_seq, RUN_KEYS),
        persons_pooled,
        occurrence=occurrence,
        bins=bins,
        year_range=year_range,
        extra_by=tuple(subgroup),
        basis=basis,
        cohort_width=cohort_width,
    )
    fc_pooled = fc_pooled.rename(columns={"n_persons": "n_units"})
    fc_pooled["n_source_persons"] = int(combined_seq["person_id"].nunique())
    fc_pooled = pooling.attach_pooled_ci(
        fc_pooled,
        fc_by_seed,
        value="rate",
        var="rate_var",
        on=[dim, "age_bin", *subgroup],
        level=level,
        clip=(0.0, None),
    )
    # Suppress each surface before they are stacked, so the combined table inherits both sides'
    # withheld cells and the figure below is drawn from the same cells the parquet publishes.
    obs_surface = out.suppress(f"{prefix}_observed", obs_surface)
    fc_by_seed = out.suppress(f"{prefix}_forecast", fc_by_seed)
    fc_pooled = out.suppress(f"{prefix}_pooled", fc_pooled)
    combined = out.suppress(
        f"{prefix}_combined", _combine_surfaces(obs_surface, fc_pooled, dim, subgroup)
    )

    # `_observed` is not written on its own: its cells reach the reader inside `_combined`, tagged
    # `source == "observed"`, and that is the surface the figure draws. It is still computed — and
    # still suppressed above — because `_combine_surfaces` is built from it.
    out.frame(f"{prefix}_forecast", fc_by_seed)
    out.frame(f"{prefix}_pooled", fc_pooled)
    out.frame(f"{prefix}_combined", combined)
    out.figure(
        f"{prefix}_combined",
        viz_lexis.plot_lexis(combined, dim=dim, mark_forecast=True, outcome=lex.outcome),
    )


def _combine_surfaces(observed_surface, forecast_pooled, dim, subgroup) -> pd.DataFrame:
    """Observed cells (source=observed) + pooled forecast cells absent from observed.

    ``forecast_pooled`` is already one estimate per cell over all N×K trajectories, so there is
    nothing to aggregate here — the cells are taken as they are and merely tagged and stacked.
    """
    cell = [dim, "age_bin", *subgroup]
    obs = observed_surface.copy()
    obs["source"] = "observed"
    observed_cells = set(map(tuple, observed_surface[cell].to_numpy()))
    keep = [tuple(r) not in observed_cells for r in forecast_pooled[cell].to_numpy()]
    fc_new = forecast_pooled[keep].copy()
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
    bundle, scfg, cfg, generated, windows, outcomes, spec, out, cohort_width
) -> None:
    """One ``replicate_variance`` block: per-person dispersion of its event, plus the roll-up.

    ``scfg.name`` (resolved at config parse) suffixes every stem this writes, so two blocks — births
    and marriages, say — land side by side instead of overwriting each other.
    """
    target_name = cfg.lexis.outcome if cfg.lexis is not None else next(iter(outcomes))
    tte_spec = outcomes[target_name]
    birth_token = tte_spec.target
    # The within-seed spread is a dispersion of *some* event count, configurable independently
    # of the CCF roll-up below — that one is births by definition.
    spread_token = bundle.token(scfg.event) if scfg.event else birth_token
    spread_label = _plural(bundle.label(spread_token))
    suffix = f"_{scfg.name}"
    horizon = _fertile_upper_days()

    if scfg.individual:
        subgroups = _person_subgroups(bundle.persons, scfg.subgroup_by, cohort_width)
        ind = _replicate_variance_individual(generated, spread_token, subgroups)
        out.frame(f"replicate_variance_individual{suffix}", ind, individual=True)
        out.frame(
            f"replicate_occurrence{suffix}",
            _replicate_occurrence(generated, tte_spec, target_name, horizon, spec),
            individual=True,
        )
        _emit_dispersion_ridges(
            ind, scfg.subgroup_by, out, event_label=spread_label, suffix=suffix
        )

    if scfg.aggregate:
        agg = _replicate_variance_aggregate(
            bundle, generated, windows, scfg.aggregate, birth_token, spec, cohort_width
        )
        if agg is not None:
            out.frame(f"replicate_variance_aggregate{suffix}", agg)


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


def _emit_dispersion_ridges(
    ind: pd.DataFrame, subgroup_by, out, *, event_label: str, suffix: str = ""
) -> None:
    """Within-seed dispersion as binned distributions and quantile summaries, plus their figures.

    Two aggregate views of the same individual-level frame, neither carrying a person. The ridges
    stack one binned ``within_seed_var`` distribution per jump-off; the fan draws the group-mean
    five-number summary of the underlying completed counts, so the variance ridge can be read
    against the outcome spread it summarises. Each requested subgroup adds a figure whose groups are
    that subgroup's values, one panel per jump-off — so a cohort can be read against the other
    cohorts at a jump-off, and against itself as the jump-off moves later.

    ``event_label`` is the counted event's name, used in the figure titles and axis labels — the
    quantity is a count of whichever event the block configured, not of births specifically.
    ``suffix`` names the block, and goes on the end of every stem so the disclosure registry's
    prefix matching still resolves them.
    """
    pop = md.dispersion_distribution(ind, by=["age_stop"], min_cell=out.min_cell)
    out.frame(f"within_seed_variance_distribution{suffix}", pop)
    out.figure(
        f"within_seed_variance{suffix}",
        viz_dispersion.plot_within_seed_variance(
            pop, x="age_stop", min_cell=out.min_cell, event_label=event_label,
            title=f"Within-person replicate variance of {event_label} by jump-off",
        ),
    )
    pop_q = md.quantile_summary(ind, by=["age_stop"], min_cell=out.min_cell)
    out.frame(f"within_seed_quantile_summary{suffix}", pop_q)
    out.figure(
        f"within_seed_quantile_fan{suffix}",
        viz_dispersion.plot_within_seed_quantile_fan(
            pop_q, x="age_stop", event_label=event_label,
            title=f"Within-person spread of completed {event_label} by jump-off",
        ),
    )

    for col in subgroup_by:
        if col not in ind.columns:
            continue
        dist = md.dispersion_distribution(ind, by=[col, "age_stop"], min_cell=out.min_cell)
        out.frame(f"within_seed_variance_distribution{suffix}_by_{col}", dist)
        out.figure(
            f"within_seed_variance{suffix}_by_{col}",
            viz_dispersion.plot_within_seed_variance(
                dist, x=col, facet_by="age_stop", min_cell=out.min_cell, event_label=event_label,
                title=f"Within-person replicate variance of {event_label} by {col}, per jump-off",
            ),
        )

        summary = md.quantile_summary(ind, by=[col, "age_stop"], min_cell=out.min_cell)
        out.frame(f"within_seed_quantile_summary{suffix}_by_{col}", summary)
        out.figure(
            f"within_seed_quantile_fan{suffix}_by_{col}",
            viz_dispersion.plot_within_seed_quantile_fan(
                summary, x=col, facet_by="age_stop", event_label=event_label,
                title=f"Within-person spread of completed {event_label} by {col}, per jump-off",
            ),
        )


def _plural(label: str) -> str:
    """A crude plural for a figure caption: ``birth`` -> ``births``, ``marriage`` -> ``marriages``.

    Event labels come from ``events.csv`` and are written in the singular ("live birth"). Only the
    caption needs the plural, so an ``-s``/``-es`` rule is enough; a label already ending in ``s``
    is left alone.
    """
    low = label.lower()
    if low.endswith("s"):
        return label
    if low.endswith(("ch", "sh", "x", "z")):
        return f"{label}es"
    return f"{label}s"


def _replicate_variance_individual(generated, event_token, subgroups=None) -> pd.DataFrame:
    """Per-(person, jump-off) replicate dispersion of the completed ``event_token`` count.

    ``within_seed_var`` / ``within_seed_cv`` are the variance and coefficient of variation across a
    person's replicates of how many times the event happens; ``expected_count`` is its mean.
    ``q0``–``q100`` are the five-number summary of the same counts
    (:func:`~seqeval.core.replicates.count_quantiles`) — the shape the single variance number
    compresses — and ``k`` the replicates behind both. Which event is counted is the run's choice
    (``forecasting.replicate_variance.event``); on a fertility run it is births, and
    ``expected_count`` is then completed quantum fertility.
    """
    ind_counts = _counts_per_run(generated, event_token)
    ind_cm = rep.count_moments(ind_counts, run_keys=_RUN_KEYS, seed_col="seed").rename(
        columns={"mean": "expected_count", "var": "within_seed_var"}
    )
    mu = ind_cm["expected_count"].to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        cv = np.sqrt(ind_cm["within_seed_var"].to_numpy()) / mu
    ind_cm["within_seed_cv"] = np.where(mu > 0, cv, np.nan)

    quant = rep.count_quantiles(ind_counts, run_keys=_RUN_KEYS, seed_col="seed")
    ind_cm = ind_cm.drop(columns=["k"]).merge(quant, on=_RUN_KEYS, how="left")

    out = ind_cm[
        [
            *_RUN_KEYS,
            "expected_count",
            "within_seed_var",
            "within_seed_cv",
            *md.QUANTILE_COLS,
            "k",
        ]
    ]
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
        "ccf": ccf,
        "within_var": comp["within_var"],
        "between_var": comp["between_var"],
        "total_var": comp["total_var"],
        "se_total": se_total,
        "ci_total_lo": ccf - z * se_total,
        "ci_total_hi": ccf + z * se_total,
        "forecast_share": float(ccf_forecast / ccf) if ccf > 0 else np.nan,
        "n_persons": int(sub["person_id"].nunique()) if "person_id" in sub.columns else comp["n"],
    }


def _fertile_upper_days() -> int:
    from seqeval.units import years_to_days

    return years_to_days(fe.FERTILE_UPPER_YEARS)


# =================================================================================================
# 4. sequence descriptives
# =================================================================================================
def _run_sequence_descriptives(
    bundle: Bundle, generated, windows, out: OutputWriter, cohort_width: int
) -> None:
    """Age profile, frequency and shape of every declared event, generated against observed.

    Three tables, each stacking one row block per (source, window) cell. Both sources are restricted
    to the *same* support — the window's people, after its jump-off — because generated rows only
    exist after the jump-off while observed ones span a whole life. Without that restriction the
    two sides would not be describing the same thing, and the difference the reader saw would be
    the censoring rather than the model.

    `token_frequency` is the exception: it describes the generated sequences alone, broken down by
    jump-off. A share of trajectories carrying a token has no observed counterpart — a person has
    one observed history, not K of them.

    Only `event_age_distribution` is drawn. `token_frequency` and `sequence_shape` are summaries a
    table states better than a figure, and the report renders them inline.
    """
    tokens = {alias: bundle.token(alias) for alias in bundle.events.keys()}
    if not tokens:
        logger.warning(
            "forecasting: skipping sequence descriptives — no `events:` aliases declared"
        )
        return
    observed = bundle.observed
    # One grid over both sources and every window, so every cell's bins line up and the figure can
    # put them on one axis.
    bins = sq.age_bins_for([generated, observed])
    # `person_id -> cohort` for the frequency breakdown; absent without a persons file, and the
    # table then carries the all-cohorts row alone.
    cohorts = _person_subgroups(bundle.persons, ["cohort"], cohort_width)
    if cohorts is None:
        logger.warning(
            "forecasting: sequence descriptives without a cohort breakdown — no persons file"
        )

    dist, freq = [], []
    for t1, t2 in windows:
        gen_w = generated[(generated["age_start"] == t1) & (generated["age_stop"] == t2)]
        if gen_w.empty:
            continue
        cells = {
            "generated": _pooled_cell(gen_w, t2),
            "observed": _observed_cell(observed, gen_w, t2),
        }
        for source, (frame, units, spans) in cells.items():
            if units.empty:
                continue
            window = {"age_start": int(t1), "age_stop": int(t2)}
            for cohort, c_frame, c_units, c_spans in _cohort_blocks(frame, units, spans, cohorts):
                label = {"source": source, **window, "cohort": cohort}
                dist.append(
                    _stamp_cell(
                        sq.event_age_distribution(
                            c_frame, c_spans, tokens=tokens, unit_keys=_UNIT_KEYS, bins=bins,
                            min_cell=out.min_cell,
                        ),
                        label,
                    )
                )
                if source == "generated":
                    # Generated only: "how many of the model's sequences contain this token" is a
                    # question about the model's sequences, and there is no observed counterpart to
                    # a per-person share over replicates.
                    freq.append(
                        _stamp_cell(
                            sq.token_frequency(
                                c_frame, c_units, tokens=tokens, unit_keys=_UNIT_KEYS,
                                min_cell=out.min_cell,
                            ),
                            {**window, "cohort": cohort},
                        )
                    )

    if not dist:
        logger.warning("forecasting: sequence descriptives produced nothing — no usable windows")
        return

    # Suppressed once on the assembled frames, then written and drawn from the same object: the
    # writer's policy groups the complement rule across the whole table, which a per-block call
    # cannot see, and a figure drawn before that would show what the parquet withholds.
    distribution = out.suppress("event_age_distribution", pd.concat(dist, ignore_index=True))
    frequency = out.suppress("token_frequency", pd.concat(freq, ignore_index=True))
    out.frame("event_age_distribution", distribution)
    out.frame("token_frequency", frequency)

    for alias, token in tokens.items():
        label = bundle.label(token)
        # The age profile draws the all-cohorts rows: cohort x jump-off x source would be a panel
        # grid nobody can read. The per-cohort rows are in the parquet.
        rows = distribution[
            (distribution["alias"] == alias) & (distribution["cohort"].isna())
        ]
        if not rows.empty:
            out.figure(
                f"event_age_distribution_{alias}",
                viz_sequences.plot_event_age_distribution(
                    rows, label=label, title="Age profile of predicted events"
                ),
            )
        bars = frequency[frequency["alias"] == alias]
        if not bars.empty:
            out.figure(
                f"token_frequency_{alias}",
                viz_sequences.plot_token_frequency(
                    bars, label=label, title="Predicted events by cohort"
                ),
            )


#: A cell's population is keyed by one column: the trajectory after pooling, the person otherwise.
_UNIT_KEYS = ["person_id"]


def _cohort_blocks(frame, units, spans, cohorts):
    """``(cohort, frame, units, spans)`` for the whole cell, then once per birth cohort.

    The whole-cell block carries ``cohort = NA`` — the convention
    :func:`_replicate_variance_aggregate` already uses — so a total and its breakdown live in one
    table and a reader cannot mistake a cohort for everybody.

    A cohort's block restricts the *population* as well as the events: units, spans and therefore
    exposure all shrink with it. Restricting only the events would measure a cohort's rate against
    everybody's person-years.
    """
    blocks = [(pd.NA, frame, units, spans)]
    if cohorts is None:
        return blocks

    key = "source_person_id" if "source_person_id" in units.columns else "person_id"
    tagged = units.merge(
        cohorts.rename(columns={"person_id": key}) if key != "person_id" else cohorts,
        on=key,
        how="left",
    )
    for cohort, grp in tagged.dropna(subset=["cohort"]).groupby("cohort", observed=True):
        ids = set(grp["person_id"])
        blocks.append(
            (
                cohort,
                frame[frame["person_id"].isin(ids)],
                grp,
                spans[spans["person_id"].isin(ids)],
            )
        )
    return blocks


def _stamp_cell(frame: pd.DataFrame, label: dict) -> pd.DataFrame:
    """Prefix a builder's output with the cell it describes."""
    out = frame.copy()
    for i, (col, value) in enumerate(label.items()):
        out.insert(i, col, value)
    return out


def _pooled_cell(gen_w: pd.DataFrame, t2: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """``(events, units, spans)`` for the generated side: every trajectory as its own unit.

    Exposure runs from the jump-off to the trajectory's last record — which includes any
    end-of-sequence padding, correctly: the trajectory *was* carried to that age, whether or not
    anything happened.
    """
    pooled, _ = pool_seeds(gen_w)
    units = pooled[["person_id", "source_person_id"]].drop_duplicates()
    spans = pooled.groupby("person_id", observed=True)["age"].max().reset_index(name="end_age")
    spans["start_age"] = np.int32(t2)
    return pooled, units, spans[["person_id", "start_age", "end_age"]]


def _observed_cell(
    observed: pd.DataFrame, gen_w: pd.DataFrame, t2: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """``(events, units, spans)`` for the observed side, on the generated cell's own terms.

    Restricted twice over: to the people the window actually generated for, and to what happened
    after its jump-off. A person whose record ends at or before the jump-off contributes no
    exposure and is dropped — they are not a zero, they are absent.
    """
    people = set(gen_w["person_id"].unique())
    mine = observed[observed["person_id"].isin(people)]
    spans = mine.groupby("person_id", observed=True)["age"].max().reset_index(name="end_age")
    spans = spans[spans["end_age"] > t2].copy()
    spans["start_age"] = np.int32(t2)
    units = spans[["person_id"]].copy()
    events = mine[mine["person_id"].isin(set(units["person_id"])) & (mine["age"] > t2)]
    return events, units, spans[["person_id", "start_age", "end_age"]]
