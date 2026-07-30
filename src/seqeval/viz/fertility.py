"""Fertility figures: cohort ASFR age profiles and the CCF inference/outcome panel (03 viz)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from scipy.stats import norm

from seqeval.metrics._disclosure import MIN_CELL
from seqeval.viz._style import (
    DEFAULT_LEVEL,
    SUPPRESSED_HATCH,
    new_fig,
    stratum_colors,
    stratum_key,
)


def plot_asfr(asfr: pd.DataFrame, *, dim: str) -> Figure:
    """Age profile of ASFR, one line per period year or birth cohort.

    ``dim`` is the cell dimension column (``"year"`` for period, ``"cohort"`` for cohort);
    ``age_bin`` labels are already in years. ``dim`` is ordered, so the lines run along a sequential
    ramp and :func:`stratum_key` keys them by legend or colorbar depending on how many there are.
    """
    fig, ax = new_fig()
    groups = list(asfr.groupby(dim, observed=True))
    colors = stratum_colors(len(groups))
    for (key, grp), color in zip(groups, colors, strict=True):
        grp = grp.sort_values("age_bin")
        ax.plot(grp["age_bin"], grp["asfr"], color=color, label=str(key), lw=1.2)
    ax.set_xlabel("age (years)")
    ax.set_ylabel("age-specific fertility rate")
    ax.set_title(f"ASFR by {dim}")
    stratum_key(ax, [k for k, _ in groups], colors, label=dim)
    return fig


def plot_ccf_inference_vs_outcome(
    variance: pd.DataFrame,
    parity: pd.DataFrame,
    *,
    observed: pd.DataFrame | None = None,
    complete: pd.Series | None = None,
    level: float = DEFAULT_LEVEL,
    left_title: str = "inference uncertainty",
    title: str | None = None,
    min_cell: int = MIN_CELL,
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
    panel is a magnification of the right one's y range, and the annotation names the ratio so the
    comparison does not rest on the reader eyeballing two axis scales.

    ``left_title`` names what the interval is on the left panel. On generated data it is inference
    uncertainty; on the observed history there are no replicates to disagree, ``within_var`` is 0
    and the interval is pure sampling error — so the caller says which.

    ``complete`` (cohort -> bool) hollows the marker of any cohort
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
        # Cohorts are labels, not a continuous scale to interpolate — but one label per yearly
        # cohort is unreadable, so only round ones are named.
        ax.set_xticks(_cohort_ticks(cohorts))

    # A suppressed cohort has no half-width; it still pins the axis through its estimate.
    reach = np.nan_to_num(half, nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = _padded_range(ccf - reach, ccf + reach)
    ax_inf.set_ylim(lo, hi)
    ax_inf.set_ylabel("CCF (mean births/woman) — magnified")
    ax_inf.set_title(left_title, fontsize=9)
    ax_inf.legend(fontsize=7, loc="best")

    _draw_parity_columns(ax_out, parity, cohorts, min_cell=min_cell)
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


def _cohort_ticks(cohorts: np.ndarray, *, max_labels: int = 10) -> np.ndarray:
    """The cohorts to label: every one when they are few, else round ones at a readable stride.

    Yearly cohorts over a few decades give dozens of ticks that overprint into a smear. Preferring
    multiples of 5 (then 10, then 25) keeps the labels on values a reader of birth cohorts expects,
    rather than an arbitrary every-nth subset that starts wherever the data happens to.
    """
    cohorts = np.asarray(cohorts)
    if len(cohorts) <= max_labels:
        return cohorts
    for step in (5, 10, 25, 50):
        round_ones = cohorts[cohorts % step == 0]
        if 0 < len(round_ones) <= max_labels:
            return round_ones
    stride = int(np.ceil(len(cohorts) / max_labels))
    return cohorts[::stride]


def _errorbar_split(ax, x, y, half, complete, *, color: str, label: str) -> None:
    """Estimate + interval per cohort, hollow where the cohort's life course is unfinished.

    A suppressed cohort keeps its estimate and loses its interval, which reaches here as a NaN
    half-width; matplotlib draws the marker with no whisker.
    """
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
    """A y range around the intervals with a little air, never a zero-height one.

    Non-finite endpoints are ignored, so a cohort whose variance was suppressed does not drag the
    axis to NaN; it simply contributes its estimate instead of an interval.
    """
    finite_lo, finite_hi = lo[np.isfinite(lo)], hi[np.isfinite(hi)]
    if not len(finite_lo) or not len(finite_hi):
        return 0.0, 1.0
    bottom, top = float(np.min(finite_lo)), float(np.max(finite_hi))
    pad = max((top - bottom) * 0.25, 0.01)
    return bottom - pad, top + pad


def _parity_fraction(parity: pd.DataFrame) -> pd.Series:
    """Each cell's direct trajectory count over its cohort's total — what the bars are drawn from.

    ``n_replicates`` counts (woman, seed) trajectories, so this is the pooled synthetic
    population's own histogram. The alternative, ``share``, weights every woman to 1 regardless of
    her replicate count; the two coincide with balanced seeds and diverge without them, and the
    figure should be drawing the counts it claims to.
    """
    total = pd.to_numeric(parity["n_replicates_total"], errors="coerce").replace(0, np.nan)
    return pd.to_numeric(parity["n_replicates"], errors="coerce") / total


def _draw_parity_columns(
    ax, parity: pd.DataFrame, cohorts: np.ndarray, *, min_cell: int = MIN_CELL
) -> None:
    """Per cohort, the trajectory counts at each completed parity, as bars from the cohort tick.

    Every bar starts at its cohort's x position rather than being centred on it, so within a cohort
    the parities share a baseline and their lengths can be compared directly — which is the whole
    question the panel is asked (how the mass splits across parities). A centred bar makes two
    counts of 0.3n and 0.4n look nearly alike; against a common left edge the difference is a
    length.

    Bar length is the cell's count over its cohort's total, so cohorts of different sizes stay
    comparable; one scale is used for every cohort.
    """
    if parity.empty or "n_replicates_total" not in parity.columns:
        return
    spacing = float(np.min(np.diff(np.sort(cohorts)))) if len(cohorts) > 1 else 1.0
    fraction = _parity_fraction(parity)
    # Every cohort may be withheld, leaving no fraction to scale by; fall back to a full-width unit.
    widest = fraction.max()
    unit = 0.8 * spacing / (float(widest) if pd.notna(widest) and widest else 1.0)
    for cohort, sub in parity.groupby("cohort", observed=True):
        shown = sub[~sub["suppressed"]]
        ax.barh(
            shown["parity"], _parity_fraction(shown) * unit, height=0.72, left=cohort,
            color="tab:blue", alpha=0.35, zorder=2,
        )
        _draw_suppressed_parity(ax, sub, cohort, unit, min_cell=min_cell)
    # Bars grow to the right of their cohort tick, so the last cohort needs room past the last tick.
    ax.set_xlim(float(np.min(cohorts)) - 0.15 * spacing, float(np.max(cohorts)) + 1.0 * spacing)
    ax.set_yticks(sorted(parity["parity"].unique()))
    labels = [str(p) for p in sorted(parity["parity"].unique())]
    labels[-1] = f"{labels[-1]}+"  # the top category is inclusive-and-above
    ax.set_yticklabels(labels)
    ax.set_ylim(parity["parity"].min() - 0.7, parity["parity"].max() + 0.7)


def _draw_suppressed_parity(
    ax, sub: pd.DataFrame, cohort, unit: float, *, min_cell: int = MIN_CELL
) -> None:
    """Withheld parities hatched at the widest bar their threshold allows."""
    hidden = sub[sub["suppressed"]]
    if hidden.empty:
        return
    # Same denominator the drawn bars use, so the cap is on their scale rather than the women one.
    # A withheld total means the cohort itself is too thin to describe: there is no honest upper
    # bound to hatch at, so nothing is drawn for it.
    totals = pd.to_numeric(sub["n_replicates_total"], errors="coerce").dropna()
    if totals.empty or not totals.iloc[0]:
        return
    cap = max(min_cell, 0) / float(totals.iloc[0]) * unit
    ax.barh(
        hidden["parity"], cap, height=0.72, left=cohort,
        facecolor="none", edgecolor="0.6", hatch=SUPPRESSED_HATCH, lw=0.4, zorder=2,
    )


def _uncertainty_ratio(half: np.ndarray, parity: pd.DataFrame) -> str:
    """One line naming how much wider the outcome spread is than the interval on the mean.

    Weighted by the same trajectory counts the bars are, so the sd quoted describes the
    distribution the reader is looking at.
    """
    # Cohorts whose variance was suppressed have no half-width to contribute to the median.
    finite = half[np.isfinite(half)]
    ci = float(np.median(finite)) if len(finite) else 0.0
    sds = []
    for _, sub in parity.groupby("cohort", observed=True):
        share = _parity_fraction(sub).fillna(0).to_numpy()
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
