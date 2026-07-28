"""Backtest overlay figures: observed-vs-generated KM/CCF bands and the timing-error ridge (04).

Scalar scores (AUC, Brier, ...) are deliberately *not* plotted here — they are reported as numbers
in the report's per-outcome metrics table, where the value and its bootstrap CI are legible.

Every figure here draws an aggregate: a band over seeds, a curve over cohorts, or a table of binned
counts. None of them takes a per-person frame, which is what lets the whole set be published.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from scipy.stats import norm

from seqeval.metrics._disclosure import MIN_CELL
from seqeval.units import days_to_years
from seqeval.viz._ridge import draw_ridges
from seqeval.viz._style import SUPPRESSED_HATCH, new_fig, stratum_colors

DEFAULT_LEVEL = 0.95


def _seed_ci(values: np.ndarray, level: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Across-seed mean and its Monte-Carlo CI, from a ``(n_seeds, ...)`` stack.

    ``mean ± z·sd/√K`` with the population sd (``ddof=0``). This is the *same quantity* as
    ``replicate_variance_aggregate.within_var``: that table's analytic decomposition
    ``sqrt(Σ_i s²_i/K)/n`` equals the standard error of the across-seed mean whenever persons are
    independent within a seed, and ``ddof=0`` makes the two agree exactly rather than up to a
    ``(K-1)/K`` factor.
    """
    k = values.shape[0]
    mean = values.mean(axis=0)
    sem = values.std(axis=0, ddof=0) / np.sqrt(k)
    z = norm.ppf(1 - (1 - level) / 2)
    return mean, mean - z * sem, mean + z * sem


def plot_km_seed_band(
    obs_km: pd.DataFrame,
    gen_km: pd.DataFrame,
    *,
    title: str | None = None,
    level: float = DEFAULT_LEVEL,
) -> Figure:
    """Observed KM curve overlaid with the generated across-seed mean and its Monte-Carlo CI.

    ``gen_km`` carries a ``seed`` column (one KM curve per seed); its survival is sampled on a
    common day grid, then :func:`_seed_ci` gives the pointwise mean and ``±z·sd/√K`` band (years
    axes). The band is replicate uncertainty in the plotted curve, on the same footing as
    ``replicate_variance_aggregate.within_var`` — not the spread of an individual seed's curve.
    """
    grid = np.union1d(obs_km["time"].to_numpy(), gen_km["time"].to_numpy())
    per_seed = []
    for _, g in gen_km.groupby("seed", observed=True):
        per_seed.append(_step_sample(g, grid))
    stacked = np.vstack(per_seed) if per_seed else np.empty((0, len(grid)))

    fig, ax = new_fig()
    years = days_to_years(grid)
    if len(stacked):
        mean, lo, hi = _seed_ci(stacked, level)
        ax.fill_between(
            years, lo, hi, step="post", alpha=0.3, color="tab:orange",
            label=f"generated {level:.0%} CI",
        )
        ax.step(years, mean, where="post", color="tab:orange", label="generated mean")
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


def plot_ccf_inference_vs_outcome(
    variance: pd.DataFrame,
    parity: pd.DataFrame,
    *,
    observed: pd.DataFrame | None = None,
    complete: pd.Series | None = None,
    level: float = DEFAULT_LEVEL,
    title: str | None = None,
) -> Figure:
    """The uncertainty in the cohort mean beside the uncertainty a woman actually faces.

    ``variance`` is :func:`~seqeval.metrics.fertility.ccf_variance` and ``parity`` is
    :func:`~seqeval.metrics.fertility.parity_distribution`. Both panels carry the same estimate,
    the same interval, and the same y unit — births per woman — so the only difference between them
    is what is being counted: the *left* is the CCF and its ``±z·sqrt(total_var)`` interval, the
    *right* adds the distribution of individual completed parity that the CCF averages over.

    The interval is not rescaled to make it visible against the distribution. It is roughly thirty
    times narrower, and that is the point of putting them side by side: a confident estimate of a
    mean says almost nothing about how confidently one woman's outcome can be predicted. The left
    panel is a magnification of the shaded band drawn on the right, and the annotation names the
    ratio so the comparison does not rest on the reader eyeballing two axis scales.

    ``complete`` (cohort -> bool, from :func:`majority_complete`) hollows the marker of any cohort
    not observed to the end of the fertile window, on both panels: a truncated cohort's "CCF" is a
    mean over an unfinished life course and must not read as a finished one. ``observed`` carries
    its own ``complete`` column and is marked the same way.
    """
    import matplotlib.pyplot as plt

    z = norm.ppf(1 - (1 - level) / 2)
    var = variance.sort_values("cohort")
    cohorts = var["cohort"].to_numpy()
    ccf = var["ccf"].to_numpy()
    half = z * np.sqrt(var["total_var"].to_numpy())
    done = (
        np.ones(len(cohorts), dtype=bool)
        if complete is None
        else complete.reindex(cohorts).fillna(True).to_numpy().astype(bool)
    )

    fig, (ax_inf, ax_out) = plt.subplots(1, 2, figsize=(9.5, 4.5), width_ratios=[1, 1.4])
    for ax in (ax_inf, ax_out):
        ax.grid(True, alpha=0.3, linewidth=0.5)
        _errorbar_split(
            ax, cohorts, ccf, half, done, color="tab:orange", label=f"CCF ± {level:.0%} CI"
        )
        if observed is not None:
            _observed_split(ax, observed)
        ax.set_xlabel("birth cohort")
        ax.set_xticks(cohorts)  # cohorts are labels, not a continuous scale to interpolate

    lo, hi = _padded_range(ccf - half, ccf + half)
    ax_inf.set_ylim(lo, hi)
    ax_inf.set_ylabel("CCF (mean births/woman) — magnified")
    ax_inf.set_title("inference uncertainty", fontsize=9)
    ax_inf.legend(fontsize=7, loc="best")

    _draw_parity_columns(ax_out, parity, cohorts)
    ax_out.axhspan(lo, hi, color="tab:orange", alpha=0.12, zorder=0)
    ax_out.annotate(
        "← left panel",
        xy=(0.995, hi), xycoords=("axes fraction", "data"),
        fontsize=6.5, color="0.45", ha="right", va="bottom",
    )
    ax_out.set_ylabel("births per woman (individual)")
    ax_out.set_title("outcome uncertainty", fontsize=9)
    ax_out.annotate(
        _uncertainty_ratio(half, parity),
        xy=(0.5, -0.19), xycoords="axes fraction", ha="center", fontsize=7, color="0.35",
    )
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    return fig


def _errorbar_split(ax, x, y, half, complete, *, color: str, label: str) -> None:
    """Estimate + interval per cohort, hollow where the cohort's life course is unfinished."""
    for sel, filled, suffix in ((complete, color, ""), (~complete, "white", " (incomplete)")):
        if not sel.any():
            continue
        ax.errorbar(
            x[sel], y[sel], yerr=half[sel], fmt="o", color=color, mfc=filled, ms=4, lw=1.4,
            capsize=3, label=f"{label}{suffix}", zorder=6,
        )


def _observed_split(ax, observed: pd.DataFrame) -> None:
    """The observed CCF per cohort, marked truncated on the same convention as the estimate."""
    o = observed.sort_values("cohort")
    done = (
        o["complete"].astype(bool).to_numpy()
        if "complete" in o.columns
        else np.ones(len(o), dtype=bool)
    )
    x, y = o["cohort"].to_numpy(), o["ccf"].to_numpy()
    for sel, marker, suffix in ((done, "o", ""), (~done, "s", " (incomplete)")):
        if not sel.any():
            continue
        ax.plot(
            x[sel], y[sel], marker, mfc="white", mec="black", ms=5,
            label=f"observed{suffix}", zorder=7,
        )


def _padded_range(lo: np.ndarray, hi: np.ndarray) -> tuple[float, float]:
    """A y range around the intervals with a little air, never a zero-height one."""
    bottom, top = float(np.min(lo)), float(np.max(hi))
    pad = max((top - bottom) * 0.25, 0.01)
    return bottom - pad, top + pad


def _draw_parity_columns(ax, parity: pd.DataFrame, cohorts: np.ndarray) -> None:
    """Per cohort, the share of women at each completed parity, as bars centred on the cohort."""
    if parity.empty:
        return
    spacing = float(np.min(np.diff(np.sort(cohorts)))) if len(cohorts) > 1 else 1.0
    widest = float(parity["share"].max() or 1.0)
    unit = 0.42 * spacing / widest
    for cohort, sub in parity.groupby("cohort", observed=True):
        shown = sub[~sub["suppressed"]]
        width = shown["share"] * unit
        ax.barh(
            shown["parity"], width, height=0.72, left=cohort - width / 2,
            color="tab:blue", alpha=0.35, zorder=2,
        )
        _draw_suppressed_parity(ax, sub, cohort, unit)
    ax.set_yticks(sorted(parity["parity"].unique()))
    labels = [str(p) for p in sorted(parity["parity"].unique())]
    labels[-1] = f"{labels[-1]}+"  # the top category is inclusive-and-above
    ax.set_yticklabels(labels)
    ax.set_ylim(parity["parity"].min() - 0.7, parity["parity"].max() + 0.7)


def _draw_suppressed_parity(ax, sub: pd.DataFrame, cohort, unit: float) -> None:
    """Withheld parities hatched at the widest bar their threshold allows."""
    hidden = sub[sub["suppressed"]]
    if hidden.empty:
        return
    total = float(sub["n_women_total"].iloc[0]) or 1.0
    cap = (MIN_CELL - 1) / total * unit
    ax.barh(
        hidden["parity"], cap, height=0.72, left=cohort - cap / 2,
        facecolor="none", edgecolor="0.6", hatch=SUPPRESSED_HATCH, lw=0.4, zorder=2,
    )


def _uncertainty_ratio(half: np.ndarray, parity: pd.DataFrame) -> str:
    """One line naming how much wider the outcome spread is than the interval on the mean."""
    ci = float(np.median(half))
    sds = []
    for _, sub in parity.groupby("cohort", observed=True):
        share = sub["share"].fillna(0).to_numpy()
        k = sub["parity"].to_numpy().astype(float)
        mass = share.sum()
        if mass <= 0:
            continue
        mean = float(np.sum(k * share) / mass)
        sds.append(float(np.sqrt(np.sum(share * (k - mean) ** 2) / mass)))
    if not sds or ci <= 0:
        return ""
    sd = float(np.median(sds))
    return f"CI half-width {ci:.3f} births · individual sd {sd:.2f} births ({sd / ci:.0f}×)"


def plot_km_jumpoff_panel(
    obs_km: pd.DataFrame,
    gen_by_jumpoff: dict[int, pd.DataFrame],
    *,
    title: str | None = None,
    level: float = DEFAULT_LEVEL,
) -> Figure:
    """Every jump-off's generated KM curve on one axes, against a single observed curve.
    """
    jumpoffs = sorted(gen_by_jumpoff)
    grid = np.union1d(
        obs_km["time"].to_numpy(),
        np.concatenate([g["time"].to_numpy() for g in gen_by_jumpoff.values()])
        if gen_by_jumpoff
        else np.empty(0),
    )
    years = days_to_years(grid)

    fig, ax = new_fig()
    for t2, color in zip(jumpoffs, stratum_colors(len(jumpoffs), lo=0.1, hi=0.85), strict=True):
        gen_km = gen_by_jumpoff[t2]
        per_seed = [_step_sample(g, grid) for _, g in gen_km.groupby("seed", observed=True)]
        if not per_seed:
            continue
        mean, lo, hi = _seed_ci(np.vstack(per_seed), level)
        jump_y = days_to_years(t2)
        ax.fill_between(years, lo, hi, step="post", alpha=0.2, color=color, label="_nolegend_")
        ax.step(years, mean, where="post", color=color, lw=2, label=f"jump-off {jump_y:.0f}y")
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
    ax.legend(fontsize=8, title=f"bands: {level:.0%} replicate CI", title_fontsize=7)
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

    def _label(key, n_bin: int) -> str:
        sub = errors[errors["pred_bin"] == key]
        lo, hi = days_to_years(sub["pred_lo"].iloc[0]), days_to_years(sub["pred_hi"].iloc[0])
        return f"{lo:.1f}–{hi:.1f}y (n={n_bin})"

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
    return fig


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


def _step_sample(km_one: pd.DataFrame, grid: np.ndarray) -> np.ndarray:
    """Survival of one KM curve sampled at ``grid`` times (step function; 1.0 before the first)."""
    g = km_one.sort_values("time")
    times = g["time"].to_numpy()
    surv = g["survival"].to_numpy()
    idx = np.searchsorted(times, grid, side="right") - 1
    return np.where(idx >= 0, surv[np.clip(idx, 0, len(surv) - 1)], 1.0)
