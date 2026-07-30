"""Pooling K per-seed synthetic populations into one estimate, with the plain sampling variance.

Each seed of a generated run is its own synthetic population: the same N people, re-run. A metric
computed per seed therefore carries an ordinary **between-person** sampling variance (Greenwood for
a survival curve, binomial for a progression ratio, Poisson for an age-specific rate) and nothing
else — no within-individual term is added anywhere in this module.

The estimate the report draws is the **pooled** one, computed once over all N×K trajectories at
once, with no per-person averaging beneath it. Its interval is the textbook formula applied to
exactly the units that produced it: Greenwood over the pooled product-limit table, ``p(1-p)/n`` over
the pooled progression denominator, ``births/PY²`` over the pooled person-years. Nothing is combined
across seeds and nothing is corrected — a row's variance is the variance of the sample the row's
estimate was computed from.

**That interval is deliberately optimistic, and by a known amount.** The N×K pooled rows are not
N×K independent people: :func:`seqeval.arms._common.combine_prefix` replays the *same* observed
prefix under every seed, so below the jump-off a person's K trajectories are exact duplicates, while
above it they genuinely diverge. Treating them as N·K units makes the band up to ``√K`` too narrow
where the duplication is total. Correcting for that is a modelling choice made downstream, not here.

So that the correction *can* be made downstream, every pooled table still records the two quantities
it needs, measured from the K per-seed curves::

    k_seeds     = number of seeds behind the cell
    mean_var    = mean_s(var_s)              # per-seed sampling variance, averaged
    between_var = var_s(estimate_s, ddof=1)  # spread of the K per-seed estimates

:func:`design_effect_var` is the correction those two feed, kept here and **not** wired into any
interval: ``clip(mean_var - (K-1)/K · between_var, mean_var/K, mean_var)``, which interpolates
between one population's worth of people (seeds identical) and the naive N·K answer (seeds
independent). Applying it is a post-processing step over the emitted columns.
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

    **Not used by any interval this module attaches** — the reported variance is the plain sampling
    variance of the pooled sample (see the module docstring). This is the correction the emitted
    ``mean_var`` / ``between_var`` / ``k_seeds`` columns exist to make possible downstream.

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
    out: pd.DataFrame,
    *,
    value: str,
    var: str,
    level: float,
    clip: tuple[float | None, float | None],
) -> pd.DataFrame:
    """Add ``pooled_var``/``se``/``ci_lo``/``ci_hi`` to a frame already carrying the components.

    ``pooled_var`` is the pooled frame's own sampling variance column ``var`` — the metric's
    textbook formula evaluated on the pooled trajectories — not a function of the seed spread.
    """
    out["pooled_var"] = out[var]
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
    """Merge the interval and the seed diagnostics onto a pooled metric frame, cell by cell.

    ``pooled`` holds the estimate computed over all trajectories at once, carrying its own sampling
    variance in ``var``; that column is the interval. ``by_seed`` holds the same metric computed
    once per seed and supplies the recorded ``k_seeds``/``mean_var``/``between_var`` only. Both are
    keyed by ``on``; cells absent from ``by_seed`` still get an interval, just no diagnostics.
    ``clip`` bounds the endpoints for a metric with a natural range (``(0, 1)`` for a proportion,
    ``(0, None)`` for a rate).
    """
    on = list(on)
    out = pooled.merge(_components(by_seed, value=value, var=var, on=on), on=on, how="left")
    return _finish(out, value=value, var=var, level=level, clip=clip)


def attach_km_pooled_ci(
    pooled_km: pd.DataFrame,
    by_seed_km: pd.DataFrame,
    *,
    by: list[str] = (),
    seed_col: str = "seed",
) -> pd.DataFrame:
    """The same treatment for survival curves, where the seeds do not share a time grid.

    The pooled product-limit table already carries the traditional Greenwood interval on the
    complementary log-log scale, so ``ci_lo`` and ``ci_hi`` are kept as computed and ``pooled_var``
    is simply ``greenwood_var``. There is no ``level`` here: the endpoints were formed at the level
    the caller passed to :func:`seqeval.metrics.survival.kaplan_meier`.

    What this adds is the seed diagnostics. A KM curve only has rows at its own event times, so each
    seed's ``survival`` and ``greenwood_var`` are step-sampled onto the pooled curve's times
    (:func:`seqeval.metrics.survival.step_sample`) before ``mean_var`` and ``between_var`` are
    formed. ``by`` are the grouping columns the two frames share — the window keys, and the outcome
    when several curves ride in one table.
    """
    by = list(by)
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
    out["pooled_var"] = out["greenwood_var"]
    out["se"] = np.sqrt(out["pooled_var"])
    return out
