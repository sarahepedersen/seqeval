"""Backtest overlay figures: observed-vs-generated KM/CCF bands and the timing-error ridge (04).

Scalar scores (AUC, Brier, ...) are deliberately *not* plotted here — they are reported as numbers
in the report's per-outcome metrics table, where the value and its bootstrap CI are legible.

Every figure here draws an aggregate: a band over seeds, a curve over cohorts, or a table of binned
counts. None of them takes a per-person frame, which is what lets the whole set be published.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from scipy.stats import norm

from seqeval.metrics._disclosure import MIN_CELL
from seqeval.units import days_to_years
from seqeval.viz._ridge import draw_ridges
from seqeval.viz._style import DEFAULT_LEVEL, new_fig, stratum_colors


def _draw_km_curve(ax, gen_km: pd.DataFrame, *, color, line_label, band_label) -> None:
    """One pooled generated KM curve inside the interval its own table carries."""
    g = gen_km.sort_values("time")
    if g.empty:
        return
    years = days_to_years(g["time"].to_numpy())
    if {"ci_lo", "ci_hi"} <= set(g.columns):
        ax.fill_between(
            years, g["ci_lo"], g["ci_hi"], step="post", alpha=0.3, color=color,
            label=band_label if band_label else "_nolegend_",
        )
    ax.step(years, g["survival"], where="post", color=color, label=line_label)


def plot_km_overlay(
    obs_km: pd.DataFrame,
    gen_km: pd.DataFrame,
    *,
    title: str | None = None,
    level: float = DEFAULT_LEVEL,
) -> Figure:
    """Observed KM curve overlaid with the pooled generated curve and its interval.

    ``gen_km`` is the *pooled* product-limit estimate over every generated trajectory at once, with
    the ``ci_lo``/``ci_hi`` computed upstream by
    :func:`seqeval.metrics.pooling.attach_km_pooled_ci`. Nothing is averaged per person and nothing
    is computed here — the figure draws the columns the parquet beside it carries, so the band can
    be checked against the table.
    """
    fig, ax = new_fig()
    _draw_km_curve(
        ax, gen_km, color="tab:orange", line_label="generated (pooled)",
        band_label=f"{level:.0%} CI",
    )
    ax.step(
        days_to_years(obs_km["time"].to_numpy()),
        obs_km["survival"],
        where="post",
        color="black",
        label="observed",
    )
    ax.set_xlabel("age (years)")
    ax.set_ylabel("survival S(t)")
    ax.set_ylim(0, 1.02)
    if title:
        ax.set_title(title)
    ax.legend(fontsize=8)
    return fig


def _ccf_band(gen_ccf: pd.DataFrame, level: float, variance: pd.DataFrame | None = None):
    """``(cohorts, mean, half_width, complete)`` for one seed-replicated CCF-by-cohort frame.

    Without ``variance`` the half-width is the replicate-only ``z·sd/√K`` across seed curves. With
    it, the half-width is ``z·sqrt(total_var)``: replicate noise *plus* the sampling error of a
    finite cohort, which is the uncertainty in the CCF being estimated rather than in the average of
    these particular seeds. Cohorts absent from ``variance`` (or too small for a sample variance)
    keep the replicate-only width rather than losing their band.
    """
    stats = (
        gen_ccf.groupby("cohort", observed=True)["ccf"]
        .agg(mean="mean", sd=lambda s: s.std(ddof=0), k="size")
        .sort_index()
    )
    z = norm.ppf(1 - (1 - level) / 2)
    half = z * stats["sd"] / np.sqrt(stats["k"])
    if variance is not None:
        total = variance.set_index("cohort")["total_var"].reindex(stats.index)
        half = (z * np.sqrt(total)).fillna(half)
    complete = majority_complete(gen_ccf).reindex(stats.index, fill_value=True).to_numpy()
    return stats.index.to_numpy(), stats["mean"].to_numpy(), half.to_numpy(), complete


def _draw_ccf_band(
    ax, gen_ccf, level, *, color, line_label, variance=None, band_label=None, alpha=0.3,
    incomplete_label=True,
) -> None:
    """One generated CCF curve: mean line (dashed where incomplete) inside its CI.

    ``band_label=None`` leaves the shaded band out of the legend — right on the multi-window panel,
    where the level is stated once in the legend title instead of once per curve.
    """
    cohorts, mean, half, complete = _ccf_band(gen_ccf, level, variance)
    if not len(cohorts):
        return
    ax.fill_between(
        cohorts, mean - half, mean + half, alpha=alpha, color=color,
        label=band_label if band_label else "_nolegend_",
    )
    _plot_with_incomplete(
        ax, cohorts, mean, complete, color=color, label=line_label,
        incomplete_label=incomplete_label,
    )


def _draw_observed_ccf(ax, obs_ccf: pd.DataFrame, *, incomplete_label: bool = True) -> None:
    """The observed CCF-by-cohort curve in black, dashed over truncated cohorts."""
    o = obs_ccf.sort_values("cohort")
    complete = (
        o["complete"].astype(bool).to_numpy()
        if "complete" in o.columns
        else np.ones(len(o), dtype=bool)
    )
    _plot_with_incomplete(
        ax, o["cohort"].to_numpy(), o["ccf"].to_numpy(), complete, color="black",
        label="observed", incomplete_label=incomplete_label,
    )


def plot_km_jumpoff_panel(
    obs_km: pd.DataFrame,
    gen_by_jumpoff: dict[int, pd.DataFrame],
    *,
    title: str | None = None,
    level: float = DEFAULT_LEVEL,
) -> Figure:
    """Every jump-off's pooled generated KM curve on one axes, against a single observed curve.
    """
    jumpoffs = sorted(gen_by_jumpoff)
    fig, ax = new_fig()
    for t2, color in zip(jumpoffs, stratum_colors(len(jumpoffs), lo=0.1, hi=0.85), strict=True):
        jump_y = days_to_years(t2)
        _draw_km_curve(
            ax, gen_by_jumpoff[t2], color=color,
            line_label=f"jump-off {jump_y:.0f}y", band_label=None,
        )
        ax.axvline(jump_y, color=color, lw=0.8, ls=":", alpha=0.7)
    ax.step(
        days_to_years(obs_km["time"].to_numpy()),
        obs_km["survival"],
        where="post",
        color="black",
        lw=2,
        label="observed",
    )
    ax.set_xlabel("age (years)")
    ax.set_ylabel("survival S(t)")
    ax.set_ylim(0, 1.02)
    if title:
        ax.set_title(title)
    ax.legend(fontsize=8, title=f"bands: {level:.0%} CI (pooled)", title_fontsize=7)
    return fig


def plot_ccf_jumpoff_panel(
    obs_ccf: pd.DataFrame,
    gen_by_jumpoff: dict[int, pd.DataFrame],
    *,
    variance_by_jumpoff: dict[int, pd.DataFrame] | None = None,
    title: str | None = None,
    level: float = DEFAULT_LEVEL,
) -> Figure:
    """Every jump-off's generated CCF-by-cohort curve on one axes, against one observed curve.
    """
    jumpoffs = sorted(gen_by_jumpoff)
    variances = variance_by_jumpoff or {}
    fig, ax = new_fig()
    for t2, color in zip(jumpoffs, stratum_colors(len(jumpoffs), lo=0.1, hi=0.85), strict=True):
        _draw_ccf_band(
            ax, gen_by_jumpoff[t2], level, variance=variances.get(t2), color=color,
            line_label=f"jump-off {days_to_years(t2):.0f}y", alpha=0.2, incomplete_label=False,
        )
    _draw_observed_ccf(ax, obs_ccf)
    ax.set_xlabel("birth cohort")
    ax.set_ylabel("CCF (mean births/woman)")
    if title:
        ax.set_title(title)
    kind = "CI" if variances else "replicate CI"
    ax.legend(fontsize=8, title=f"bands: {level:.0%} {kind}", title_fontsize=7)
    return fig


# =================================================================================================
# parity progression ratios
# =================================================================================================
def _ppr_labels(ppr: pd.DataFrame) -> dict[int, str]:
    """``parity_from`` -> ``"k→k+1"`` display label, read off the frame's own transitions."""
    pairs = ppr[["parity_from", "parity_to"]].drop_duplicates()
    return {
        int(r.parity_from): f"{int(r.parity_from)}→{int(r.parity_to)}" for r in pairs.itertuples()
    }


def _draw_ppr_series(ax, gen_ppr, level, *, color, label, dodge: float = 0.0) -> None:
    """One pooled PPR series with the interval its own table carries, one x per transition."""
    g = gen_ppr.sort_values("parity_from")
    if g.empty:
        return
    x = g["parity_from"].to_numpy().astype(float) + dodge
    value = g["ppr"].to_numpy()
    yerr = (
        np.vstack([value - g["ci_lo"].to_numpy(), g["ci_hi"].to_numpy() - value])
        if {"ci_lo", "ci_hi"} <= set(g.columns)
        else None
    )
    ax.errorbar(x, value, yerr=yerr, fmt="o-", color=color, capsize=3, lw=1.6, label=label)


def _draw_observed_ppr(ax, obs_ppr: pd.DataFrame) -> None:
    """The observed progression ratios in black — a point estimate, drawn without a band."""
    o = obs_ppr.sort_values("parity_from")
    ax.plot(
        o["parity_from"].to_numpy().astype(float), o["ppr"].to_numpy(),
        "s-", color="black", lw=1.2, label="observed",
    )


def _finish_ppr_axes(ax, obs_ppr, gen_ppr, *, title, legend_title) -> None:
    labels = {**_ppr_labels(gen_ppr), **_ppr_labels(obs_ppr)}
    ticks = sorted(labels)
    ax.set_xticks(ticks)
    ax.set_xticklabels([labels[t] for t in ticks])
    ax.set_xlabel("parity transition")
    ax.set_ylabel("progression ratio")
    ax.set_ylim(0, 1.02)
    ax.grid(True, axis="y", alpha=0.3, linewidth=0.5)
    if title:
        ax.set_title(title)
    ax.legend(fontsize=8, title=legend_title, title_fontsize=7)


def plot_ppr_overlay(
    obs_ppr: pd.DataFrame,
    gen_ppr: pd.DataFrame,
    *,
    title: str | None = None,
    level: float = DEFAULT_LEVEL,
) -> Figure:
    """Observed parity progression ratios under the pooled generated ratios and their CI.

    ``gen_ppr`` is the *pooled* estimate over every generated trajectory at once, carrying the
    ``ci_lo``/``ci_hi`` computed upstream by
    :func:`seqeval.metrics.pooling.attach_pooled_ci`.

    Later transitions rest on the women who reached that parity, so the bars widen to the right on
    their own; a wide interval at parity 4 is a thin denominator, not a worse model.
    """
    fig, ax = new_fig()
    _draw_ppr_series(ax, gen_ppr, level, color="tab:orange", label="generated (pooled)")
    _draw_observed_ppr(ax, obs_ppr)
    _finish_ppr_axes(
        ax, obs_ppr, gen_ppr, title=title, legend_title=f"bars: {level:.0%} CI"
    )
    return fig


def plot_ppr_jumpoff_panel(
    obs_ppr: pd.DataFrame,
    gen_by_jumpoff: dict[int, pd.DataFrame],
    *,
    title: str | None = None,
    level: float = DEFAULT_LEVEL,
) -> Figure:
    """Every jump-off's generated PPR series on one axes, against one observed series.

    Series are nudged apart along x by a fraction of a transition so their caps stay legible; the
    integer tick under each cluster is the transition they all describe.
    """
    jumpoffs = sorted(gen_by_jumpoff)
    offsets = np.linspace(-0.12, 0.12, len(jumpoffs)) if len(jumpoffs) > 1 else np.zeros(1)
    fig, ax = new_fig()
    colors = stratum_colors(len(jumpoffs), lo=0.1, hi=0.85)
    for t2, color, dodge in zip(jumpoffs, colors, offsets, strict=True):
        _draw_ppr_series(
            ax, gen_by_jumpoff[t2], level, color=color,
            label=f"jump-off {days_to_years(t2):.0f}y", dodge=float(dodge),
        )
    _draw_observed_ppr(ax, obs_ppr)
    any_gen = next(iter(gen_by_jumpoff.values()), obs_ppr)
    _finish_ppr_axes(ax, obs_ppr, any_gen, title=title, legend_title=f"bars: {level:.0%} CI")
    return fig


# =================================================================================================
# cohort ASFR
# =================================================================================================
#: Panel size for the cohort-ASFR small multiples (inches). These carry a per-cell band now, and a
#: one-year age grid under a band needs room to stay readable.
_ASFR_PANEL = (5.0, 4.0)


def _asfr_grid(cohorts: list) -> tuple[Figure, np.ndarray]:
    """A shared-axes small-multiple grid, one panel per cohort, with the spares hidden."""
    ncols = min(3, max(len(cohorts), 1))
    nrows = int(np.ceil(max(len(cohorts), 1) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(_ASFR_PANEL[0] * ncols, _ASFR_PANEL[1] * nrows),
        sharex=True, sharey=True, squeeze=False,
    )
    flat = axes.ravel()
    for ax in flat[len(cohorts) :]:
        ax.set_visible(False)
    return fig, flat


def _draw_asfr_panel(ax, gen_cohort, *, color, label, band: bool = True) -> None:
    """One cohort's pooled generated age profile, inside the interval its own table carries."""
    if gen_cohort.empty:
        return
    g = gen_cohort.sort_values("age_bin")
    ages = g["age_bin"].to_numpy().astype(float)
    if band and {"ci_lo", "ci_hi"} <= set(g.columns):
        ax.fill_between(ages, g["ci_lo"], g["ci_hi"], color=color, alpha=0.2, lw=0)
    ax.plot(ages, g["asfr"].to_numpy(), color=color, lw=1.6, label=label)


def _finish_asfr_grid(fig, axes, cohorts, *, title, level: float | None = None) -> Figure:
    for ax in axes[: len(cohorts)]:
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("age (years)", fontsize=9)
        ax.tick_params(labelsize=8)
    axes[0].set_ylabel("ASFR (births/woman-year)", fontsize=9)
    legend_title = f"bands: {level:.0%} CI (pooled)" if level is not None else None
    axes[0].legend(fontsize=8, title=legend_title, title_fontsize=7)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_asfr_overlay(
    obs_asfr: pd.DataFrame,
    gen_asfr: pd.DataFrame,
    *,
    jumpoff_days: int | None = None,
    title: str | None = None,
    level: float = DEFAULT_LEVEL,
) -> Figure:
    """Observed cohort age-fertility profiles under the pooled generated profile, one panel each.

    A cohort ASFR is a ``(cohort, age)`` surface, so it is drawn as small multiples: one panel per
    birth cohort, age along x, the observed profile in black beneath the pooled generated one.
    Panels share both axes so the cohorts can be compared by eye. The shaded band is the interval
    the pooled table carries; it is drawn flat rather than as error bars so the shape of the profile
    stays the thing you read.

    ``jumpoff_days`` marks the jump-off **age** in every panel. The jump-off is an age, not a date,
    so the rule falls in the same place for every cohort: everything to its left is replayed
    history, everything to its right is model output, and only the right-hand side is a forecast
    being scored.
    """
    cohorts = sorted(gen_asfr["cohort"].dropna().unique())
    fig, axes = _asfr_grid(cohorts)
    obs_by = dict(list(obs_asfr.groupby("cohort", observed=True)))
    for ax, cohort in zip(axes, cohorts, strict=False):
        sub = gen_asfr[gen_asfr["cohort"] == cohort]
        _draw_asfr_panel(
            ax, sub, color="tab:orange", label=f"generated (pooled, {level:.0%} CI)"
        )
        o = obs_by.get(cohort)
        if o is not None:
            o = o.sort_values("age_bin")
            ax.plot(o["age_bin"], o["asfr"], color="black", lw=1.4, label="observed")
        if jumpoff_days is not None:
            ax.axvline(days_to_years(jumpoff_days), color="0.4", lw=0.9, ls=":")
        ax.set_title(f"cohort {cohort}", fontsize=10, loc="left")
    return _finish_asfr_grid(fig, axes, cohorts, title=title, level=level)


def plot_asfr_jumpoff_panel(
    obs_asfr: pd.DataFrame,
    gen_by_jumpoff: dict[int, pd.DataFrame],
    *,
    title: str | None = None,
    level: float = DEFAULT_LEVEL,
) -> Figure:
    """The same cohort grid with every jump-off's pooled generated profile drawn in each panel.

    Each jump-off gets a color, and a dotted rule of its own color marks the age it starts
    forecasting from — so a panel shows directly how much of a cohort's profile the model is being
    asked to produce, and whether the fit degrades as that share grows. Each profile carries its own
    interval; because a later jump-off's profile begins where the earlier ones are already well
    under way, the bands overlap far less than the curves do.
    """
    jumpoffs = sorted(gen_by_jumpoff)
    cohorts = sorted(
        {c for g in gen_by_jumpoff.values() for c in g["cohort"].dropna().unique()}
    )
    fig, axes = _asfr_grid(cohorts)
    obs_by = dict(list(obs_asfr.groupby("cohort", observed=True)))
    colors = stratum_colors(len(jumpoffs), lo=0.1, hi=0.85)
    for ax, cohort in zip(axes, cohorts, strict=False):
        for t2, color in zip(jumpoffs, colors, strict=True):
            gen = gen_by_jumpoff[t2]
            _draw_asfr_panel(
                ax, gen[gen["cohort"] == cohort], color=color,
                label=f"jump-off {days_to_years(t2):.0f}y",
            )
            ax.axvline(days_to_years(t2), color=color, lw=0.9, ls=":", alpha=0.7)
        o = obs_by.get(cohort)
        if o is not None:
            o = o.sort_values("age_bin")
            ax.plot(o["age_bin"], o["asfr"], color="black", lw=1.4, label="observed")
        ax.set_title(f"cohort {cohort}", fontsize=10, loc="left")
    return _finish_asfr_grid(fig, axes, cohorts, title=title, level=level)


def majority_complete(gen_ccf: pd.DataFrame) -> pd.Series:
    """Per cohort: is the cohort complete in the majority of seeds? (all-complete if unmarked)."""
    if "complete" not in gen_ccf.columns:
        return pd.Series(dtype=bool)
    return (
        gen_ccf.assign(_c=gen_ccf["complete"].astype(float))
        .groupby("cohort", observed=True)["_c"]
        .mean()
        .gt(0.5)
        .sort_index()
    )


def _plot_with_incomplete(
    ax, x, y, complete, *, color: str, label: str, incomplete_label: bool = True
) -> None:
    """One CCF curve: solid/filled where the cohort is complete, dashed/open where it is not.

    The dashed run starts at the last complete cohort so the two segments join up rather than
    leaving a visual gap at the transition. ``incomplete_label=False`` keeps the styling but drops
    the second legend entry — on the multi-jump-off panel one legend row per curve is already
    enough, and the dashed/open convention is explained in the caption.
    """
    complete = np.asarray(complete, dtype=bool)
    ax.plot(x[complete], y[complete], "o-", color=color, label=label)
    if complete.all():
        return
    first_incomplete = int(np.argmax(~complete))
    start = max(first_incomplete - 1, 0)  # bridge back to the last complete point
    seg = np.zeros(len(x), dtype=bool)
    seg[start:] = ~complete[start:]
    seg[start] = True
    ax.plot(
        x[seg], y[seg], "o--", color=color, mfc="white",
        label=f"{label} (incomplete cohorts)" if incomplete_label else "_nolegend_",
    )


def plot_timing_ridge(
    errors: pd.DataFrame,
    *,
    xlabel: str = "observed − predicted (years)",
    title: str | None = None,
    min_cell: int = MIN_CELL,
) -> Figure:
    """Distribution of timing error, one ridge per bin of predicted value.

    ``errors`` is :func:`seqeval.metrics.ml.timing_error_distribution` — binned counts and nothing
    else, which is what makes this figure publishable. Each ridge is one equal-count bin of
    predicted value drawn as within-bin proportions (:func:`~seqeval.viz._ridge.draw_ridges`); the
    dashed line at zero is a perfectly timed prediction, and mass to its right is an event that
    happened later than predicted.

    Each ridge carries a tick at the bin holding its cumulative half — the bias reading the
    ``y = x`` diagonal gives on a scatter, at bin resolution rather than interpolated.
    """
    fig, ax = new_fig()
    ax.set_xlabel(xlabel)
    if title:
        ax.set_title(title, fontsize=10)
    if errors.empty:
        return fig

    def _label(key, n_bin: int | None) -> str:
        sub = errors[errors["pred_bin"] == key]
        lo, hi = days_to_years(sub["pred_lo"].iloc[0]), days_to_years(sub["pred_hi"].iloc[0])
        shown = "withheld" if n_bin is None else str(n_bin)
        return f"{lo:.1f}–{hi:.1f}y (n={shown})"

    bases = draw_ridges(
        ax, errors, row="pred_bin", lo="error_lo", hi="error_hi",
        count="n_persons", total="n_pred_bin",
        label_fn=_label, x_transform=days_to_years, ylabel="predicted value", min_cell=min_cell,
    )
    colors = stratum_colors(len(bases), lo=0.1, hi=0.85)
    for base, b, color in zip(bases, sorted(errors["pred_bin"].unique()), colors, strict=True):
        _draw_bin_median(ax, errors[errors["pred_bin"] == b].sort_values("error_lo"), base, color)

    ax.axvline(0, color="black", lw=1, ls="--", zorder=5)
    ax.annotate(
        "← predicts too late          predicts too early →",
        xy=(0.5, -0.16), xycoords="axes fraction", ha="center", fontsize=7, color="0.35",
    )
    note = _exclusion_note(errors)
    if note:
        ax.annotate(
            note, xy=(0.5, -0.23), xycoords="axes fraction", ha="center", fontsize=7,
            color="0.35",
        )
    return fig


def _exclusion_note(errors: pd.DataFrame) -> str:
    """What the ridge leaves out: trajectories the model never brought to the outcome.

    A distribution of timing error among predicted events says nothing about the events that were
    never predicted, so the figure has to say how many of those there were.
    """
    if not {"n_trajectories", "n_excluded"} <= set(errors.columns):
        return ""
    total = pd.to_numeric(errors["n_trajectories"], errors="coerce").dropna()
    excluded = pd.to_numeric(errors["n_excluded"], errors="coerce").dropna()
    if total.empty or excluded.empty:
        return ""
    total, excluded = float(total.iloc[0]), float(excluded.iloc[0])
    if not total or excluded <= 0:
        return ""
    return (
        f"excludes {excluded:,.0f} of {total:,.0f} trajectories ({excluded / total:.1%}) "
        "with no predicted event inside the frame"
    )


def _draw_bin_median(ax, sub: pd.DataFrame, base: float, color) -> None:
    """Tick at the error bin holding the row's cumulative half — bin resolution, uninterpolated."""
    counts = sub["n_persons"].fillna(0).to_numpy().astype(float)
    total = counts.sum()
    if total <= 0:
        return
    idx = int(np.searchsorted(np.cumsum(counts), total / 2.0))
    idx = min(idx, len(sub) - 1)
    mid = days_to_years((sub["error_lo"].iloc[idx] + sub["error_hi"].iloc[idx]) / 2)
    ax.plot([mid, mid], [base, base + 0.12], color=color, lw=1.6, solid_capstyle="butt", zorder=4)
