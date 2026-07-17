"""Past/observed descriptives arm: orchestrates survival + fertility metrics and their plots (03).

Thin orchestration over the shared metric functions (survival, fertility) and the outcome
extractors (02). Presence of a config block enables the metric (00 section 5 rule 1). When
``persons`` is missing, cohort/period metrics are skipped with a logged warning naming exactly what
was skipped; age-only metrics (KM, PPR, life table) still run.
"""

from __future__ import annotations

import logging

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
    observed = bundle.observed
    spans = observation_spans(observed, OBS_KEYS)
    strata = _strata_frame(bundle, cfg.stratify_by, cohort_width)

    _run_kaplan_meier(observed, cfg, out, outcomes, strata)

    needs_births = cfg.fertility is not None or cfg.life_table is not None
    births = None
    if needs_births:
        births = births_table(observed, OBS_KEYS, birth_event=bundle.token("birth"))

    if cfg.fertility is not None:
        _run_fertility(bundle, cfg, out, births, spans, cohort_width)
    if cfg.life_table is not None:
        _run_life_table(cfg, out, births, spans)


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
        ppr = fe.ppr(births, spans, max_parity=fcfg.ppr.max_parity)
        out.frame("ppr", ppr)
        out.figure("ppr", viz_fertility.plot_ppr(ppr))

    if persons is None and (fcfg.ccf or fcfg.asfr):
        logger.warning(
            "descriptives: skipping CCF/ASFR — no persons file (birth_year needed for "
            "cohort/period metrics)"
        )
        return

    if fcfg.ccf:
        ccf = fe.ccf(births, spans, persons, by_cohort=True, cohort_width=cohort_width)
        out.frame("ccf", ccf)
        out.figure("ccf", viz_fertility.plot_ccf(ccf))

    bins = AgeBins.from_years(_FERTILE_LO_YEARS, _FERTILE_HI_YEARS, fcfg.age_bin_width)
    for mode in fcfg.asfr:
        table = fe.asfr(births, spans, persons, mode=mode, bins=bins, cohort_width=cohort_width)
        out.frame(f"asfr_{mode}", table)
        if mode == "period":
            # Period ASFR is a (year x age) surface — a heatmap plus a TFR-over-time summary line
            # read far better than one age-profile line per calendar year.
            out.figure("asfr_period", viz_fertility.plot_asfr_surface(table))
            tfr = fe.tfr(table)
            out.frame("tfr", tfr)
            out.figure("tfr", viz_fertility.plot_tfr(tfr))
        else:
            # Cohort ASFR has few groups; one age-profile line per cohort stays legible.
            out.figure("asfr_cohort", viz_fertility.plot_asfr(table, dim="cohort"))


def _run_life_table(cfg, out, births, spans) -> None:
    bins = AgeBins.from_years(_FERTILE_LO_YEARS, _FERTILE_HI_YEARS, 1.0)
    lt = sv.life_table(births, spans, max_parity=cfg.life_table.max_parity, bins=bins)
    out.frame("life_table", lt)
