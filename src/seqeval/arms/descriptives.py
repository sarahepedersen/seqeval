"""Past/observed descriptives arm: orchestrates survival + fertility metrics and their plots (03).

Thin orchestration over the shared metric functions (survival, fertility) and the outcome
extractors (02). Presence of a config block enables the metric (00 section 5 rule 1). When
``persons`` is missing, cohort/period metrics are skipped with a logged warning naming exactly what
was skipped; age-only metrics (KM, PPR) still run.
"""

from __future__ import annotations

import logging
from dataclasses import replace

import pandas as pd

from seqeval.arms._common import OutputWriter
from seqeval.config import DEFAULT_COHORT_WIDTH, DescriptivesConfig
from seqeval.core.outcomes import births as births_table
from seqeval.core.outcomes import observation_spans, time_to_event
from seqeval.core.slicing import AgeBins, cohort_bins
from seqeval.core.specs import TTESpec
from seqeval.io.loaders import Bundle
from seqeval.io.schema import OBS_KEYS
from seqeval.metrics import fertility as fe
from seqeval.metrics import survival as sv
from seqeval.viz import fertility as viz_fertility
from seqeval.viz import km as viz_km

logger = logging.getLogger("seqeval")

_FERTILE_LO_YEARS = 15.0
_FERTILE_HI_YEARS = 50.0


def run(
    bundle: Bundle,
    cfg: DescriptivesConfig,
    out: OutputWriter,
    *,
    outcomes: dict[str, TTESpec],
    cohort_width: int = DEFAULT_COHORT_WIDTH,
) -> None:
    """Compute every configured descriptive metric, write result frames, and render figures.

    ``outcomes`` is the resolved timing registry (``config.resolve_outcomes``); ``kaplan_meier``
    config entries are names into it. ``cohort_width`` is the shared birth-cohort band width
    (``Config.cohort_width``, from the top-level ``persons`` block) — passed in rather than read
    from ``DescriptivesConfig`` so every arm uses one population-wide value. (Both parameters extend
    01's ``run(bundle, cfg, out)`` signature because they live at the top level of the config.)
    """
    bundle = _restrict_cohorts(bundle, cfg.max_cohort_year)
    observed = bundle.observed
    spans = observation_spans(observed, OBS_KEYS)
    strata = _strata_frame(bundle, cfg.stratify_by, cohort_width)

    _run_kaplan_meier(observed, cfg, out, outcomes, strata)

    if cfg.fertility is not None:
        births = births_table(observed, OBS_KEYS, birth_event=bundle.token("birth"))
        _run_fertility(bundle, cfg, out, births, spans, cohort_width)


def _restrict_cohorts(bundle: Bundle, max_cohort_year: int | None) -> Bundle:
    """Drop people born after ``max_cohort_year`` from the arm's whole population.

    The people go, not just their rows in cohort-indexed tables: a period rate computed on everyone
    and a cohort rate computed on a subset would describe different populations and could not be
    read against each other. Needs ``persons`` for ``birth_year``; without it the cutoff cannot be
    applied and says so rather than silently describing everybody.
    """
    if max_cohort_year is None:
        return bundle
    if bundle.persons is None:
        logger.warning(
            "descriptives: max_cohort_year=%d ignored — no persons file, so birth_year is unknown",
            max_cohort_year,
        )
        return bundle

    persons = bundle.persons
    keep = persons[persons["birth_year"] <= max_cohort_year]
    dropped = len(persons) - len(keep)
    if not dropped:
        return bundle
    logger.info(
        "descriptives: max_cohort_year=%d drops %d of %d people (born %d–%d)",
        max_cohort_year, dropped, len(persons),
        int(persons["birth_year"].max()), max_cohort_year + 1,
    )
    kept_ids = set(keep["person_id"])
    return replace(
        bundle,
        persons=keep.reset_index(drop=True),
        observed=bundle.observed[bundle.observed["person_id"].isin(kept_ids)],
    )


# =================================================================================================
# stratification
# =================================================================================================
def _strata_frame(bundle: Bundle, stratify_by: list[str], cohort_width: int) -> pd.DataFrame | None:
    """A ``[person_id, *stratify_by]`` frame, or ``None`` if stratification cannot be honored."""
    if not stratify_by:
        return None
    if bundle.persons is None:
        logger.warning(
            "descriptives: stratify_by=%s requires persons; running unstratified", stratify_by
        )
        return None
    frame = bundle.persons[["person_id"]].copy()
    for col in stratify_by:
        if col == "cohort":
            cohorts = cohort_bins(bundle.persons, width=cohort_width).reset_index()
            frame = frame.merge(cohorts, on="person_id")
        else:
            frame = frame.merge(bundle.persons[["person_id", col]], on="person_id")
    return frame


# =================================================================================================
# metric groups
# =================================================================================================
def _run_kaplan_meier(observed, cfg, out, outcomes, strata) -> None:
    by = list(cfg.stratify_by) if strata is not None else []
    for name in cfg.kaplan_meier:
        spec = outcomes[name]
        tte = time_to_event(observed, OBS_KEYS, spec)
        if strata is not None:
            tte = tte.merge(strata, on="person_id", how="left")
        km = sv.kaplan_meier(tte, by=by)
        out.frame(f"km_{name}", km)
        out.figure(
            f"km_{name}",
            viz_km.plot_km(
                km, by=by, title=name.replace("_", " "), xlabel=_km_xlabel(spec, outcomes)
            ),
        )


def _km_xlabel(spec: TTESpec, outcomes: dict[str, TTESpec]) -> str:
    """X-axis label for a KM curve: age from birth, or duration since the outcome's origin event."""
    if spec.origin is None:
        return "age (years)"
    # Duration is measured from the origin occurrence; name it if the origin is a registry entry.
    for name, other in outcomes.items():
        if other == spec.origin:
            return f"years since {name.replace('_', ' ')}"
    return "years since origin event"


def _run_fertility(bundle: Bundle, cfg, out, births, spans, cohort_width: int) -> None:
    fcfg = cfg.fertility
    persons = bundle.persons

    if fcfg.ppr is not None:
        out.frame("ppr", fe.ppr(births, spans, max_parity=fcfg.ppr.max_parity))

    if persons is None and (fcfg.ccf or fcfg.asfr):
        logger.warning(
            "descriptives: skipping CCF/ASFR — no persons file (birth_year needed for "
            "cohort/period metrics)"
        )
        return

    if fcfg.ccf:
        ccf = fe.ccf(births, spans, persons, by_cohort=True, cohort_width=cohort_width)
        variance = fe.ccf_variance(births, spans, persons, cohort_width=cohort_width)
        out.frame(
            "ccf",
            ccf.merge(variance.drop(columns=["n_women", "ccf", "n_persons"]), on="cohort"),
        )

        # The same two-panel contrast the backtesting arm draws, on the observed history: the
        # interval is pure sampling error here (nothing is replicated), and the parity distribution
        # is the realized one rather than a model's predictive mixture.
        parity = fe.parity_distribution(
            births, spans, persons, cohort_width=cohort_width, min_cell=out.min_cell
        )
        out.frame("parity_distribution", parity)
        out.figure(
            "ccf_uncertainty",
            viz_fertility.plot_ccf_inference_vs_outcome(
                variance,
                parity,
                complete=ccf.set_index("cohort")["complete"],
                left_title="sampling uncertainty",
                title="Sampling vs outcome uncertainty — observed",
            ),
        )

    bins = AgeBins.from_years(_FERTILE_LO_YEARS, _FERTILE_HI_YEARS, fcfg.age_bin_width)
    for mode in fcfg.asfr:
        table = fe.asfr(births, spans, persons, mode=mode, bins=bins, cohort_width=cohort_width)
        out.frame(f"asfr_{mode}", table)
        if mode != "period":
            # Cohort ASFR has few groups; one age-profile line per cohort stays legible.
            out.figure("asfr_cohort", viz_fertility.plot_asfr(table, dim="cohort"))
        # Period ASFR is written but not drawn: neither the year x age surface nor its TFR summary
        # is reported, since the jump-off is an age and a calendar-year cell is part observed and
        # part forecast, so nothing in the other arms can be read against it.


