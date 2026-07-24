"""Backtest overlay figures: observed-vs-generated KM/CCF bands and timing calibration (04).

Scalar scores (AUC, Brier, ...) are deliberately *not* plotted here — they are reported as numbers
in the report's per-outcome metrics table, where the value and its bootstrap CI are legible.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from seqeval.units import days_to_years
from seqeval.viz._style import new_fig


def plot_km_seed_band(
    obs_km: pd.DataFrame, gen_km: pd.DataFrame, *, title: str | None = None
) -> Figure:
    """Observed KM curve overlaid with the generated across-seed band (median + IQR).

    ``gen_km`` carries a ``seed`` column (one KM curve per seed); its survival is sampled on a
    common day grid, then the median and inter-quartile range across seeds are shaded (years axes).
    """
    grid = np.union1d(obs_km["time"].to_numpy(), gen_km["time"].to_numpy())
    per_seed = []
    for _, g in gen_km.groupby("seed", observed=True):
        per_seed.append(_step_sample(g, grid))
    stacked = np.vstack(per_seed) if per_seed else np.empty((0, len(grid)))

    fig, ax = new_fig()
    years = days_to_years(grid)
    if len(stacked):
        med = np.median(stacked, axis=0)
        lo, hi = np.percentile(stacked, [25, 75], axis=0)
        ax.fill_between(
            years, lo, hi, step="post", alpha=0.3, color="tab:orange", label="generated IQR"
        )
        ax.step(years, med, where="post", color="tab:orange", label="generated median")
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


def plot_ccf_seed_band(
    obs_ccf: pd.DataFrame, gen_ccf: pd.DataFrame, *, title: str | None = None
) -> Figure:
    """Observed CCF by cohort overlaid with the generated across-seed band (median + IQR).

    ``gen_ccf`` carries a ``seed`` column (one CCF-by-cohort curve per seed); the median and IQR
    across seeds are shaded per cohort. Both frames have ``[cohort, ccf]`` (plus seed on generated).

    Incomplete cohorts — those never observed to the fertile upper bound, whose mean is truncated
    rather than completed — are drawn with open markers on a dashed segment on **both** curves, the
    same convention as the descriptive :func:`seqeval.viz.fertility.plot_ccf`, so a falling tail is
    not misread as a fertility decline. A generated cohort counts as incomplete when the majority
    of seeds report it incomplete. Frames without a ``complete`` column are drawn as all-complete.
    """
    fig, ax = new_fig()
    g = gen_ccf.groupby("cohort", observed=True)["ccf"]
    med = g.median().sort_index()
    lo = g.quantile(0.25).reindex(med.index)
    hi = g.quantile(0.75).reindex(med.index)
    cohorts = med.index.to_numpy()
    if len(cohorts):
        ax.fill_between(
            cohorts, lo.to_numpy(), hi.to_numpy(), alpha=0.3, color="tab:orange",
            label="generated IQR",
        )
        gen_complete = _majority_complete(gen_ccf).reindex(med.index, fill_value=True).to_numpy()
        _plot_with_incomplete(
            ax, cohorts, med.to_numpy(), gen_complete, color="tab:orange", label="generated median"
        )
    o = obs_ccf.sort_values("cohort")
    obs_complete = (
        o["complete"].astype(bool).to_numpy()
        if "complete" in o.columns
        else np.ones(len(o), dtype=bool)
    )
    _plot_with_incomplete(
        ax, o["cohort"].to_numpy(), o["ccf"].to_numpy(), obs_complete, color="black",
        label="observed",
    )
    ax.set_xlabel("birth cohort")
    ax.set_ylabel("CCF (mean births/woman)")
    if title:
        ax.set_title(title)
    ax.legend(fontsize=8)
    return fig


def _majority_complete(gen_ccf: pd.DataFrame) -> pd.Series:
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


def _plot_with_incomplete(ax, x, y, complete, *, color: str, label: str) -> None:
    """One CCF curve: solid/filled where the cohort is complete, dashed/open where it is not.

    The dashed run starts at the last complete cohort so the two segments join up rather than
    leaving a visual gap at the transition.
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
        x[seg], y[seg], "o--", color=color, mfc="white", label=f"{label} (incomplete cohorts)"
    )


def timing_pairs(
    td: pd.DataFrame,
    obs_tte: pd.DataFrame,
    *,
    horizon_days: int | None = None,
    persons: Iterable | None = None,
    drop_projected_beyond: bool = True,
) -> pd.DataFrame:
    """Per-person ``[person_id, pred, obs]`` waiting times in **years**, on the scored population.

    The single place the predicted (seed-median) and observed waiting times are paired up, shared
    by :func:`plot_timing_calibration` and by the arm's outlier accounting so both describe exactly
    the same set of persons. A person contributes only when the comparison is a real prediction:

    - the event was actually observed, and observed **within the frame horizon** — the model's
      predictive distribution is defective and capped at ``horizon_days``, so it can never predict
      a longer wait and comparing against unbounded observed times would be ill-posed;
    - the person is in ``persons`` when given — the arm passes the scored population (its condition
      minus those whose answer was already settled at the jump-off). Settled persons matter most:
      their event sits in the observed prefix, which every replicate replays verbatim, so they
      would land exactly on ``y = x`` and flatter the model for free;
    - ``drop_projected_beyond`` (default) drops persons whose predicted median has reached the
      horizon, i.e. the model projects the event *outside* the outcome's frame. Their ``q50`` is the
      cap itself, not an estimated date, so they form a spurious vertical stripe at the frame edge
      that drags the binned trend toward it. Whether the model gets those persons right is the
      *probability* question, answered by the reliability diagram next door.
    """
    seen = obs_tte.loc[obs_tte["observed"], ["person_id", "duration"]]
    m = td.merge(seen, on="person_id", how="inner")
    if persons is not None:
        m = m[m["person_id"].isin(set(persons))]
    if horizon_days is not None:
        m = m[m["duration"] <= horizon_days]
        if drop_projected_beyond:
            m = m[m["q50"] < horizon_days]
    return pd.DataFrame(
        {
            "person_id": m["person_id"].to_numpy(),
            "pred": days_to_years(m["q50"].to_numpy().astype(float)),
            "obs": days_to_years(m["duration"].to_numpy().astype(float)),
        }
    )


def plot_timing_calibration(
    td: pd.DataFrame,
    obs_tte: pd.DataFrame,
    *,
    horizon_days: int | None = None,
    floor_days: int = 0,
    persons: Iterable | None = None,
    xlabel: str = "predicted waiting time (years, median across seeds)",
    ylabel: str = "observed waiting time (years)",
    title: str | None = None,
    n_bins: int = 10,
) -> Figure:
    """Predicted vs observed waiting time per person: the y=x ideal plus a binned median trend.

    ``td`` is :func:`seqeval.core.replicates.timing_distribution` (per-person quantiles in days);
    ``obs_tte`` is the observed time-to-event table. Gray points are individuals; the red trend is
    the median observed time within equal-count bins of predicted time, with its inter-quartile
    ribbon. Departure from the dashed ``y = x`` line is timing bias — **above** the line the model
    predicts too early, **below** it predicts too late.

    Scope is :func:`timing_pairs` — persons whose event was observed inside the frame, who were
    still unsettled at the jump-off, and whom the model does not project past the frame. The axes
    span that same region: ``floor_days`` (the jump-off, where prediction begins, for an outcome
    whose duration is an age; 0 otherwise) to ``horizon_days`` (where the frame closes). Nothing
    can land outside it, so no part of the box is unreachable and there is nothing to clip.
    """
    pairs = timing_pairs(td, obs_tte, horizon_days=horizon_days, persons=persons)

    fig, ax = new_fig()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontsize=10)
    if pairs.empty:
        return fig

    pred, obs = pairs["pred"].to_numpy(), pairs["obs"].to_numpy()
    ax.scatter(pred, obs, s=8, alpha=0.15, color="tab:gray", linewidths=0)

    xlo = float(days_to_years(floor_days))
    if horizon_days is not None:
        lim = float(days_to_years(horizon_days))
    else:
        lim = float(max(pred.max(), obs.max())) * 1.05 or 1.0
    ax.plot([xlo, lim], [xlo, lim], "k--", lw=1, label="ideal (y = x)")

    edges = np.unique(np.quantile(pred, np.linspace(0, 1, n_bins + 1)))
    if len(edges) > 2:
        idx = np.clip(np.digitize(pred, edges) - 1, 0, len(edges) - 2)
        centers, med, lo, hi = [], [], [], []
        for b in range(len(edges) - 1):
            sel = idx == b
            if sel.sum() < 5:  # too few persons to summarise this bin
                continue
            centers.append(float(np.median(pred[sel])))
            med.append(float(np.median(obs[sel])))
            lo.append(float(np.percentile(obs[sel], 25)))
            hi.append(float(np.percentile(obs[sel], 75)))
        if centers:
            ax.fill_between(centers, lo, hi, alpha=0.25, color="tab:red", label="binned IQR")
            ax.plot(centers, med, "o-", color="tab:red", ms=5, label="binned median")

    ax.set_xlim(xlo, lim)
    ax.set_ylim(xlo, lim)
    ax.set_aspect("equal")  # equal axes: y = x reads as a true 45° diagonal
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.legend(fontsize=8)
    return fig


def _step_sample(km_one: pd.DataFrame, grid: np.ndarray) -> np.ndarray:
    """Survival of one KM curve sampled at ``grid`` times (step function; 1.0 before the first)."""
    g = km_one.sort_values("time")
    times = g["time"].to_numpy()
    surv = g["survival"].to_numpy()
    idx = np.searchsorted(times, grid, side="right") - 1
    return np.where(idx >= 0, surv[np.clip(idx, 0, len(surv) - 1)], 1.0)
