"""Rebuild every figure from the published parquets alone (06).

:func:`seqeval.report.build_report` embeds the PNGs it finds on disk; it never draws. So a results
directory carrying only ``*.parquet`` — the shape an export takes when the figures are left behind —
renders a report with no figures in it. This module closes that gap: it reads the parquets and the
resolved config out of ``manifest.json``, redraws every figure, and writes the PNGs back beside the
tables so ``build_report`` finds them.

Two things follow from drawing off the *published* tables rather than the in-memory ones, and both
are deliberate:

- **A redrawn figure shows exactly what the parquet publishes.** Where small-cell suppression
  withheld a variance or an interval, the redrawn figure has no band — see
  :mod:`seqeval.metrics._disclosure`. The arms suppress before drawing for the same reason, so the
  two agree.
- **Titles fall back to raw event tokens.** The pipeline names events through
  ``Bundle.label`` (``events.csv``), which is an *input* and not part of a results export. Pass
  ``event_definitions`` to recover the natural-language names.

Every stem this module writes matches what the arm wrote, because the report finds figures by
globbing those names.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from seqeval.arms._common import OutputWriter
from seqeval.config import Config, resolve_outcomes, resolve_probability_outcomes
from seqeval.report import read_manifest
from seqeval.units import days_to_years
from seqeval.viz import backtest as viz_backtest
from seqeval.viz import calibration as viz_calibration
from seqeval.viz import dispersion as viz_dispersion
from seqeval.viz import fertility as viz_fertility
from seqeval.viz import km as viz_km
from seqeval.viz import lexis as viz_lexis
from seqeval.viz import sequences as viz_sequences
from seqeval.viz._labels import describe_outcome

logger = logging.getLogger("seqeval")

__all__ = ["redraw"]

#: Columns every backtesting long table carries to say which cell a row belongs to.
_LABEL_COLS = ("outcome", "condition", "age_start", "age_stop")


def redraw(results_dir: str | Path, *, event_definitions: str | Path | None = None) -> list[Path]:
    """Redraw every figure in ``results_dir`` from its parquets; return the paths written.

    Needs ``manifest.json`` for the resolved config — the figure titles, the interval level, the
    suppression threshold and the stratification are all config, not columns. Arms whose directory
    is absent are skipped, so a partial export redraws whatever it can.
    """
    results_dir = Path(results_dir)
    manifest = read_manifest(results_dir)
    if manifest is None:
        raise FileNotFoundError(
            f"{results_dir / 'manifest.json'} is missing; redraw needs the resolved config it "
            "carries (figure titles, interval level, min_cell)"
        )
    cfg = Config.model_validate(manifest["config_resolved"])
    labeller = _labeller(event_definitions)

    written: list[Path] = []
    for arm, fn in (
        ("descriptives", _redraw_descriptives),
        ("backtesting", _redraw_backtesting),
        ("forecasting", _redraw_forecasting),
    ):
        arm_dir = results_dir / arm
        if not arm_dir.is_dir():
            continue
        out = OutputWriter(
            base_dir=results_dir,
            arm=arm,
            model=cfg.model.name,
            figure_format=cfg.output.figure_format,
            individual_level=True,  # nothing drawn here is per-person
            min_cell=cfg.output.min_cell,
        )
        fn(arm_dir, cfg, out, labeller)
        written.extend(out.written)
        logger.info("redraw %s: %d figure(s)", arm, len(out.written))
    return written


def _labeller(event_definitions: str | Path | None):
    """A raw-token -> natural-language function, or ``str`` when no definitions are available."""
    if event_definitions is None:
        return str
    defs = pd.read_csv(event_definitions)
    mapping = dict(
        zip(
            defs["model_representation"].astype(str),
            defs["event_definition"].astype(str),
            strict=True,
        )
    )
    return lambda token: mapping.get(str(token), str(token))


def _read(arm_dir: Path, stem: str) -> pd.DataFrame | None:
    """A published table, or ``None`` when this run did not write it."""
    path = arm_dir / f"{stem}.parquet"
    if not path.exists():
        return None
    frame = pd.read_parquet(path, engine="pyarrow")
    return None if frame.empty else frame


def _cells(frame: pd.DataFrame, *, outcome: str | None = None) -> list[tuple]:
    """The distinct label cells present, in a stable order.

    A cell is ``(outcome, condition, age_start, age_stop)`` — one figure's worth of rows out of a
    long table that holds every outcome and window together.
    """
    cols = [c for c in _LABEL_COLS if c in frame.columns]
    sub = frame if outcome is None else frame[frame["outcome"] == outcome]
    if not cols or sub.empty:
        return []
    return [tuple(r) for r in sub[cols].drop_duplicates().sort_values(cols).to_numpy()]


def _slice(frame: pd.DataFrame, cell: tuple, *, cols=_LABEL_COLS) -> pd.DataFrame:
    """The rows of ``frame`` belonging to one label cell."""
    present = [c for c in cols if c in frame.columns]
    mask = pd.Series(True, index=frame.index)
    for col, value in zip(present, cell, strict=False):
        mask &= (frame[col].isna() if pd.isna(value) else frame[col] == value)
    return frame[mask]


def _jumpoff_label(age_stop_days) -> int:
    """The ``w<N>`` suffix the arms use: the jump-off age rounded to whole years."""
    return int(round(days_to_years(int(age_stop_days))))


# =================================================================================================
# descriptives
# =================================================================================================
def _redraw_descriptives(arm_dir: Path, cfg: Config, out: OutputWriter, labeller) -> None:
    dcfg = cfg.arms.descriptives
    if dcfg is None:
        return
    outcomes = resolve_outcomes(cfg)

    for name in dcfg.kaplan_meier:
        km = _read(arm_dir, f"km_{name}")
        if km is None:
            continue
        by = [c for c in dcfg.stratify_by if c in km.columns]
        out.figure(
            f"km_{name}",
            viz_km.plot_km(
                km, by=by, title=name.replace("_", " "),
                xlabel=_km_xlabel(outcomes, name),
            ),
        )

    if dcfg.fertility is None:
        return
    asfr = _read(arm_dir, "asfr_cohort")
    if asfr is not None:
        out.figure("asfr_cohort", viz_fertility.plot_asfr(asfr, dim="cohort"))

    ccf, parity = _read(arm_dir, "ccf"), _read(arm_dir, "parity_distribution")
    if ccf is not None and parity is not None:
        out.figure(
            "ccf_uncertainty",
            viz_fertility.plot_ccf_inference_vs_outcome(
                ccf,
                parity,
                complete=ccf.set_index("cohort")["complete"]
                if "complete" in ccf.columns
                else None,
                level=cfg.replicates.level,
                left_title="sampling uncertainty",
                title="Sampling vs outcome uncertainty — observed",
                min_cell=cfg.output.min_cell,
            ),
        )


def _km_xlabel(outcomes: dict, name: str) -> str:
    """The arm's own x-axis rule: age from birth, or duration since the outcome's origin."""
    spec = outcomes.get(name)
    if spec is None or spec.origin is None:
        return "age (years)"
    for other_name, other in outcomes.items():
        if other == spec.origin:
            return f"years since {other_name.replace('_', ' ')}"
    return "years since origin event"


# =================================================================================================
# forecasting
# =================================================================================================
def _redraw_forecasting(arm_dir: Path, cfg: Config, out: OutputWriter, labeller) -> None:
    fcfg = cfg.arms.forecasting
    if fcfg is None:
        return

    if fcfg.lexis is not None:
        combined = _read(arm_dir, "lexis_cohort_combined")
        if combined is not None:
            out.figure(
                "lexis_cohort_combined",
                viz_lexis.plot_lexis(
                    combined, dim="cohort", mark_forecast=True, outcome=fcfg.lexis.outcome
                ),
            )

    outcomes = resolve_outcomes(cfg)
    default_target = (
        outcomes[fcfg.lexis.outcome].target
        if fcfg.lexis is not None and fcfg.lexis.outcome in outcomes
        else None
    )
    _redraw_sequence_descriptives(arm_dir, out, labeller)

    for block in fcfg.replicate_variance:
        if not block.individual:
            continue
        # The same fallback chain the arm walks: the block's own event, else the target event of the
        # lexis outcome. Only then the block name, which is a label of last resort.
        token = cfg.events[block.event] if block.event else default_target
        label = _plural(labeller(token) if token is not None else block.name)
        suffix = f"_{block.name}"
        _redraw_dispersion(arm_dir, out, suffix, label, subgroup_by=block.subgroup_by)


def _redraw_sequence_descriptives(arm_dir: Path, out: OutputWriter, labeller) -> None:
    """One age-profile figure per alias, taken from the table's own `alias`/`token` columns.

    Driven by the data rather than by the config: an export whose config has since changed still
    redraws exactly the figures its tables describe.
    """
    dist = _read(arm_dir, "event_age_distribution")
    freq = _read(arm_dir, "token_frequency")
    aliases = sorted(
        set()
        | (set(dist["alias"].dropna().unique()) if dist is not None else set())
        | (set(freq["alias"].dropna().unique()) if freq is not None else set())
    )
    for alias in aliases:
        # The arm draws the age profile from the all-cohorts rows only; the per-cohort ones are in
        # the table. Mirror that exactly or the pixel-parity guard fails.
        rows = (
            dist[(dist["alias"] == alias) & (dist["cohort"].isna())]
            if dist is not None
            else None
        )
        if rows is not None and not rows.empty:
            out.figure(
                f"event_age_distribution_{alias}",
                viz_sequences.plot_event_age_distribution(
                    rows, label=labeller(rows["token"].iloc[0]),
                    title="Age profile of predicted events",
                ),
            )
        bars = freq[freq["alias"] == alias] if freq is not None else None
        if bars is not None and not bars.empty:
            out.figure(
                f"token_frequency_{alias}",
                viz_sequences.plot_token_frequency(
                    bars, label=labeller(bars["token"].iloc[0]),
                    title="Predicted events by cohort",
                ),
            )


def _redraw_dispersion(
    arm_dir: Path, out: OutputWriter, suffix: str, label: str, *, subgroup_by
) -> None:
    """One block's four figure families, mirroring ``forecasting._emit_dispersion_ridges``."""
    pop = _read(arm_dir, f"within_seed_variance_distribution{suffix}")
    if pop is not None:
        out.figure(
            f"within_seed_variance{suffix}",
            viz_dispersion.plot_within_seed_variance(
                pop, x="age_stop", min_cell=out.min_cell, event_label=label,
                title=f"Within-person replicate variance of {label} by jump-off",
            ),
        )
    pop_q = _read(arm_dir, f"within_seed_quantile_summary{suffix}")
    if pop_q is not None:
        out.figure(
            f"within_seed_quantile_fan{suffix}",
            viz_dispersion.plot_within_seed_quantile_fan(
                pop_q, x="age_stop", event_label=label,
                title=f"Within-person spread of completed {label} by jump-off",
            ),
        )

    for col in subgroup_by:
        dist = _read(arm_dir, f"within_seed_variance_distribution{suffix}_by_{col}")
        if dist is not None:
            out.figure(
                f"within_seed_variance{suffix}_by_{col}",
                viz_dispersion.plot_within_seed_variance(
                    dist, x=col, facet_by="age_stop", min_cell=out.min_cell, event_label=label,
                    title=f"Within-person replicate variance of {label} by {col}, per jump-off",
                ),
            )
        summary = _read(arm_dir, f"within_seed_quantile_summary{suffix}_by_{col}")
        if summary is not None:
            out.figure(
                f"within_seed_quantile_fan{suffix}_by_{col}",
                viz_dispersion.plot_within_seed_quantile_fan(
                    summary, x=col, facet_by="age_stop", event_label=label,
                    title=f"Within-person spread of completed {label} by {col}, per jump-off",
                ),
            )


def _plural(label: str) -> str:
    """Same crude caption plural the forecasting arm uses."""
    low = label.lower()
    if low.endswith("s"):
        return label
    return f"{label}es" if low.endswith(("ch", "sh", "x", "z")) else f"{label}s"


# =================================================================================================
# backtesting
# =================================================================================================
def _redraw_backtesting(arm_dir: Path, cfg: Config, out: OutputWriter, labeller) -> None:
    bcfg = cfg.arms.backtesting
    if bcfg is None:
        return
    level = cfg.replicates.level
    _redraw_reliability(arm_dir, cfg, out, labeller, level)
    _redraw_timing(arm_dir, cfg, out, labeller)
    _redraw_overlays(arm_dir, out, level)


def _redraw_reliability(arm_dir, cfg, out, labeller, level) -> None:
    cal = _read(arm_dir, "calibration")
    if cal is None:
        return
    dist = _read(arm_dir, "p_hat_distribution")
    specs = {s.name: s for s in resolve_probability_outcomes(cfg, resolve_outcomes(cfg))}
    for cell in _cells(cal):
        outcome, _condition, _t1, t2 = cell
        spec = specs.get(outcome)
        desc = (
            describe_outcome(spec, jumpoff_days=int(t2), label_fn=labeller)
            if spec is not None
            else outcome
        )
        out.figure(
            f"reliability_{outcome}_w{_jumpoff_label(t2)}",
            viz_calibration.plot_reliability(
                _slice(cal, cell),
                _slice(dist, cell) if dist is not None else None,
                title=desc,
                min_cell=out.min_cell,
            ),
        )


def _redraw_timing(arm_dir, cfg, out, labeller) -> None:
    err = _read(arm_dir, "timing_error")
    if err is None:
        return
    outcomes = resolve_outcomes(cfg)
    specs = {s.name: s for s in resolve_probability_outcomes(cfg, outcomes)}
    for cell in _cells(err):
        outcome, _condition, _t1, t2 = cell
        spec = specs.get(outcome)
        desc = (
            describe_outcome(spec, jumpoff_days=int(t2), label_fn=labeller)
            if spec is not None
            else outcome
        )
        is_age = spec is None or getattr(spec, "tte", None) is None or spec.tte.origin is None
        unit = "age at event" if is_age else "waiting time"
        out.figure(
            f"timing_ridge_{outcome}_w{_jumpoff_label(t2)}",
            viz_backtest.plot_timing_ridge(
                _slice(err, cell),
                xlabel=f"observed − predicted {unit} (years)",
                title=f"Timing error — {desc}",
                min_cell=out.min_cell,
            ),
        )


def _observed_side(err: pd.DataFrame, target: str, keys: list[str], value: str) -> pd.DataFrame:
    """The observed curve for one aggregate target, read out of ``aggregate_error``.

    ``aggregate_error`` is the only published home of the observed aggregate: the arms compute it,
    compare it, and keep just the comparison. Its ``obs`` column is that curve, so renaming ``obs``
    to the metric's own value column reproduces the frame the overlay expects.
    """
    rows = err[err["target"] == target]
    cols = [c for c in keys if c in rows.columns]
    return (
        rows[[*cols, "obs"]]
        .dropna(subset=["obs"])
        .drop_duplicates(subset=cols)
        .rename(columns={"obs": value})
        .sort_values(cols)
    )


def _redraw_overlays(arm_dir: Path, out: OutputWriter, level: float) -> None:
    err = _read(arm_dir, "aggregate_error")
    if err is None:
        return

    # -- Kaplan-Meier, one family per outcome -------------------------------------------------
    for target in sorted({t for t in err["target"].unique() if str(t).startswith("km:")}):
        name = target[len("km:") :]
        pooled = _read(arm_dir, "km_pooled")
        if pooled is None:
            continue
        # `km_observed` is the arm's own observed curve on its full event-time grid; the
        # `aggregate_error` fallback is the same curve sampled onto the coarser comparison grid,
        # which is what an export predating that table can offer.
        obs_all = _read(arm_dir, "km_observed")
        if obs_all is not None and "outcome" in obs_all.columns:
            obs = obs_all[obs_all["outcome"] == target].sort_values("time")
        else:
            obs = _observed_side(err, target, ["time"], "survival")
        gen_by_jumpoff = {}
        for cell in _cells(pooled, outcome=target):
            t2 = int(cell[3])
            gen = _slice(pooled, cell)
            gen_by_jumpoff[t2] = gen
            out.figure(
                f"km_overlay_{name}_w{_jumpoff_label(t2)}",
                viz_backtest.plot_km_overlay(
                    obs, gen, title=f"{name} survival — jump-off {_jumpoff_label(t2)}y",
                    level=level,
                ),
            )
        if len(gen_by_jumpoff) > 1:
            out.figure(
                f"km_overlay_{name}_all_jumpoffs",
                viz_backtest.plot_km_jumpoff_panel(
                    obs, gen_by_jumpoff, title=f"{name} survival — all jump-offs", level=level
                ),
            )

    # -- parity progression ------------------------------------------------------------------
    pooled = _read(arm_dir, "ppr_pooled")
    if pooled is not None and (err["target"] == "ppr").any():
        obs = _observed_side(err, "ppr", ["parity_from"], "ppr")
        # `aggregate_error` keys PPR on `parity_from` alone, but the transition *labels* need the
        # destination parity too; the pooled table is where that lives.
        obs = obs.merge(
            pooled[["parity_from", "parity_to"]].drop_duplicates(), on="parity_from", how="left"
        )
        gen_by_jumpoff = {}
        for cell in _cells(pooled, outcome="ppr"):
            t2 = int(cell[3])
            gen_by_jumpoff[t2] = _slice(pooled, cell)
            out.figure(
                f"ppr_overlay_w{_jumpoff_label(t2)}",
                viz_backtest.plot_ppr_overlay(
                    obs, gen_by_jumpoff[t2], level=level,
                    title=f"Parity progression — jump-off {_jumpoff_label(t2)}y",
                ),
            )
        if len(gen_by_jumpoff) > 1:
            out.figure(
                "ppr_overlay_all_jumpoffs",
                viz_backtest.plot_ppr_jumpoff_panel(
                    obs, gen_by_jumpoff, title="Parity progression — all jump-offs", level=level
                ),
            )

    # -- cohort ASFR -------------------------------------------------------------------------
    pooled = _read(arm_dir, "asfr_pooled")
    if pooled is not None and (err["target"] == "asfr_cohort").any():
        obs = _observed_side(err, "asfr_cohort", ["cohort", "age_bin"], "asfr")
        gen_by_jumpoff = {}
        for cell in _cells(pooled, outcome="asfr_cohort"):
            t2 = int(cell[3])
            gen_by_jumpoff[t2] = _slice(pooled, cell)
            out.figure(
                f"asfr_overlay_w{_jumpoff_label(t2)}",
                viz_backtest.plot_asfr_overlay(
                    obs, gen_by_jumpoff[t2], jumpoff_days=t2, level=level,
                    title=f"Cohort ASFR — jump-off {_jumpoff_label(t2)}y",
                ),
            )
        if len(gen_by_jumpoff) > 1:
            out.figure(
                "asfr_overlay_all_jumpoffs",
                viz_backtest.plot_asfr_jumpoff_panel(
                    obs, gen_by_jumpoff, title="Cohort ASFR — all jump-offs", level=level
                ),
            )

    _redraw_ccf(arm_dir, out, err, level)


def _redraw_ccf(arm_dir: Path, out: OutputWriter, err: pd.DataFrame, level: float) -> None:
    """The two CCF figures, both drawn from ``ccf_variance`` on the generated side.

    ``ccf_variance`` carries, per (window x cohort), the generated CCF, its variance decomposition
    and whether the cohort completes in most seeds — the three things the band, the line and the
    hollow markers need. ``aggregate_error`` keeps only the comparison, so without this table there
    is nothing to draw and both figures are skipped.
    """
    if not (err["target"] == "ccf").any():
        return
    variance = _read(arm_dir, "ccf_variance")
    if variance is None:
        logger.warning(
            "redraw backtesting: skipping the CCF figures — no ccf_variance table to take the "
            "generated curve and its band from"
        )
        return

    # The observed CCF: `descriptives/ccf.parquet` is the same quantity and carries `complete`, so
    # truncated cohorts stay hollow. Falls back to the comparison table when that arm did not run.
    obs = _read(arm_dir.parent / "descriptives", "ccf")
    if obs is None or "ccf" not in obs.columns:
        obs = _observed_side(err, "ccf", ["cohort"], "ccf")
    parity = _read(arm_dir, "parity_distribution")
    gen_by_jumpoff, var_by_jumpoff = {}, {}

    for cell in _cells(variance):
        t2 = int(cell[3])
        var = _slice(variance, cell).dropna(subset=["cohort"])
        if var.empty:
            continue
        # `_ccf_band` takes the per-cohort mean of `ccf`, so one row per cohort is the frame it
        # wants; `total_var` sets the width and `complete` the dashing.
        gen_by_jumpoff[t2] = var[
            [c for c in ("cohort", "ccf", "complete") if c in var.columns]
        ]
        var_by_jumpoff[t2] = var[["cohort", "total_var"]]

        if parity is not None:
            par = _slice(parity, cell)
            if not par.empty:
                out.figure(
                    f"uncertainty_ccf_w{_jumpoff_label(t2)}",
                    viz_fertility.plot_ccf_inference_vs_outcome(
                        var,
                        par,
                        observed=obs,
                        complete=var.set_index("cohort")["complete"]
                        if "complete" in var.columns
                        else None,
                        level=level,
                        title=(
                            "Inference vs outcome uncertainty — jump-off "
                            f"{_jumpoff_label(t2)}y"
                        ),
                        min_cell=out.min_cell,
                    ),
                )

    if gen_by_jumpoff:
        out.figure(
            "ccf_overlay_all_jumpoffs",
            viz_backtest.plot_ccf_jumpoff_panel(
                obs, gen_by_jumpoff, variance_by_jumpoff=var_by_jumpoff,
                title="CCF by cohort — all jump-offs", level=level,
            ),
        )
