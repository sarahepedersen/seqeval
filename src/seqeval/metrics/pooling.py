"""Pooling K per-seed synthetic populations into one estimate, with a measured design effect.

Each seed of a generated run is its own synthetic population: the same N people, re-run. A metric
computed per seed therefore carries an ordinary **between-person** sampling variance (Greenwood for
a survival curve, binomial for a progression ratio, Poisson for an age-specific rate) and nothing
else — no within-individual term is added anywhere in this module.

The estimate the report draws is the **pooled** one, computed once over all N×K trajectories at
once, with no per-person averaging beneath it. Its interval is the open question, because the N×K
rows are not N×K independent people: :func:`seqeval.arms._common.combine_prefix` replays the *same*
observed prefix under every seed, so below the jump-off a person's K trajectories are exact
duplicates, while above it they genuinely diverge. Running the sampling formula on the pooled table
would be up to ``√K`` too narrow.

:func:`design_effect_var` reads that duplication off the per-seed values instead of assuming it::

    Var(pooled) = clip( mean_var - (K-1)/K * between_var,  mean_var/K,  mean_var )

with ``mean_var`` the average per-seed sampling variance and ``between_var`` the spread of the K
per-seed estimates. Seeds identical (``between_var = 0``) gives ``mean_var`` — one population's
worth, N people. Seeds independent (``between_var = mean_var``) gives ``mean_var/K`` — the naive
N·K answer. Everything real lands between the two, and the clip keeps it there.

**Read the width as a per-cell estimate, not a constant.** ``between_var`` is a sample variance over
K numbers, so at the handful of seeds a typical run has it is noisy: near the independent limit its
own error is larger than the target it is estimating, and cells scatter toward whichever bound the
noise pushed them to. The clip bounds the damage in both directions and more seeds tighten it. On
the demo run the KM curves sit near ``mean_var`` — the observed prefix is shared, so the seeds
really are near-duplicates — while PPR and ASFR cells, where the seeds diverge, land much closer to
the floor.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from seqeval.metrics.survival import step_sample

__all__ = ["DEFAULT_LEVEL", "attach_km_pooled_ci", "attach_pooled_ci", "design_effect_var"]

#: Interval level used when a caller does not pass the run's ``replicates.level``.
DEFAULT_LEVEL = 0.95

#: Columns every ``attach_*`` helper adds, in order.
POOLED_COLUMNS = ("k_seeds", "mean_var", "between_var", "pooled_var", "se", "ci_lo", "ci_hi")


def design_effect_var(mean_var, between_var, k) -> np.ndarray:
    """Variance of a pooled estimate, corrected by the design effect the seeds actually show.

    ``mean_var`` is the per-seed sampling variance averaged over seeds, ``between_var`` the sample
    variance (``ddof=1``) of the K per-seed estimates, ``k`` the number of seeds. All three
    broadcast, so this works cell-by-cell over an array.

    A single seed, or a cell where ``between_var`` could not be formed, keeps ``mean_var``: one
    population is all the evidence there is, and widening or narrowing it would be invention.
    ``NaN`` in ``mean_var`` propagates — a cell with no sampling variance gets no interval.
    """
    mean_var = np.asarray(mean_var, dtype=float)
    between_var = np.asarray(between_var, dtype=float)
    k = np.asarray(k, dtype=float)

    shrink = np.where(k > 1, (k - 1) / np.where(k > 0, k, 1.0), 0.0)
    corrected = mean_var - shrink * np.nan_to_num(between_var)
    floor = mean_var / np.where(k > 0, k, 1.0)
    # np.clip propagates NaN from `mean_var` through both bounds, which is what we want.
    return np.clip(corrected, floor, mean_var)


def _components(by_seed: pd.DataFrame, *, value: str, var: str, on: list[str]) -> pd.DataFrame:
    """``[*on, k_seeds, mean_var, between_var]`` from a per-seed metric table."""
    grouped = by_seed.groupby(on, observed=True)
    parts = pd.DataFrame(
        {
            "k_seeds": grouped[value].size(),
            "mean_var": (
                grouped[var].mean() if var in by_seed.columns else np.nan
            ),
            "between_var": grouped[value].var(ddof=1),
        }
    )
    return parts.reset_index()


def _finish(
    out: pd.DataFrame, *, value: str, level: float, clip: tuple[float | None, float | None]
) -> pd.DataFrame:
    """Add ``pooled_var``/``se``/``ci_lo``/``ci_hi`` to a frame already carrying the components."""
    out["pooled_var"] = design_effect_var(out["mean_var"], out["between_var"], out["k_seeds"])
    out["se"] = np.sqrt(out["pooled_var"])
    half = norm.ppf(1 - (1 - level) / 2) * out["se"]
    lo, hi = clip
    out["ci_lo"] = (out[value] - half).clip(lower=lo, upper=hi)
    out["ci_hi"] = (out[value] + half).clip(lower=lo, upper=hi)
    return out


def attach_pooled_ci(
    pooled: pd.DataFrame,
    by_seed: pd.DataFrame,
    *,
    value: str,
    var: str,
    on: list[str],
    level: float = DEFAULT_LEVEL,
    clip: tuple[float | None, float | None] = (None, None),
) -> pd.DataFrame:
    """Merge the design-effect interval onto a pooled metric frame, cell by cell.

    ``pooled`` holds the estimate computed over all trajectories at once; ``by_seed`` holds the same
    metric computed once per seed, carrying its own sampling variance in ``var``. Both are keyed by
    ``on``. Cells present in ``pooled`` but not in ``by_seed`` get no interval rather than a wrong
    one. ``clip`` bounds the endpoints for a metric with a natural range (``(0, 1)`` for a
    proportion, ``(0, None)`` for a rate).
    """
    on = list(on)
    out = pooled.merge(_components(by_seed, value=value, var=var, on=on), on=on, how="left")
    return _finish(out, value=value, level=level, clip=clip)


def attach_km_pooled_ci(
    pooled_km: pd.DataFrame,
    by_seed_km: pd.DataFrame,
    *,
    by: list[str] = (),
    seed_col: str = "seed",
    level: float = DEFAULT_LEVEL,
) -> pd.DataFrame:
    """The same correction for survival curves, where the seeds do not share a time grid.

    A KM curve only has rows at its own event times, so each seed's ``survival`` and
    ``greenwood_var`` are step-sampled onto the pooled curve's times before the components are
    formed (:func:`seqeval.metrics.survival.step_sample`). ``by`` are the grouping columns the two
    frames share — the window keys, and the outcome when several curves ride in one table.
    """
    by = list(by)
    # The product-limit table carries its own log-log interval, computed as though the pooled
    # trajectories were independent people. That is the interval this correction exists to replace.
    pooled_km = pooled_km.drop(columns=["ci_lo", "ci_hi"], errors="ignore")
    if pooled_km.empty:
        return pooled_km.assign(**{c: pd.Series(dtype="float64") for c in POOLED_COLUMNS})

    seed_groups = dict(list(by_seed_km.groupby(by, observed=True))) if by else {(): by_seed_km}
    parts = []
    for key, curve in pooled_km.groupby(by, observed=True) if by else [((), pooled_km)]:
        curve = curve.sort_values("time")
        grid = curve["time"].to_numpy()
        seeds = seed_groups.get(key)
        surv, gw = [], []
        if seeds is not None:
            for _, one in seeds.groupby(seed_col, observed=True):
                surv.append(step_sample(one, grid))
                gw.append(step_sample(one, grid, value="greenwood_var", before=0.0))
        block = curve.copy()
        if surv:
            surv, gw = np.vstack(surv), np.vstack(gw)
            block["k_seeds"] = surv.shape[0]
            with np.errstate(invalid="ignore"):
                block["mean_var"] = np.nanmean(gw, axis=0)
                block["between_var"] = (
                    surv.var(axis=0, ddof=1) if surv.shape[0] > 1 else np.nan
                )
        else:
            block["k_seeds"] = 0
            block["mean_var"] = np.nan
            block["between_var"] = np.nan
        parts.append(block)

    out = pd.concat(parts, ignore_index=True)
    return _finish(out, value="survival", level=level, clip=(0.0, 1.0))
