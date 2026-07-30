"""Replicate-dispersion distributions: how a per-person spread is distributed across a population.

The per-person dispersion columns (``within_seed_var``, ``within_seed_cv``, and the five-number
summary ``q0``–``q100`` of a person's completed-birth counts) describe individuals, so nothing built
directly from them can be published. This module reduces them to group-level shapes with no row
belonging to anybody: :func:`dispersion_distribution` bins one column into counts per (group, bin),
and :func:`quantile_summary` averages the per-person quantiles within a group. Both are what the
figures draw and what a restricted run is able to write.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from seqeval.metrics._disclosure import MIN_CELL, suppress_small_cells

__all__ = ["QUANTILE_COLS", "dispersion_distribution", "quantile_summary"]

#: Per-person five-number-summary columns, as produced by
#: :func:`~seqeval.core.replicates.count_quantiles`.
QUANTILE_COLS = ("q0", "q25", "q50", "q75", "q100")


def dispersion_distribution(
    ind: pd.DataFrame,
    *,
    value: str = "within_seed_var",
    by: list[str],
    n_bins: int = 20,
    min_cell: int = MIN_CELL,
) -> pd.DataFrame:
    """Distribution of ``value`` within each ``by`` group, as counts per shared bin.

    Returns ``[*by, bin, bin_lo, bin_hi, n_persons, n_group, suppressed]``. Bin edges are computed
    once over the whole population and reused for every group, so the groups are stacked on one
    axis and their shapes are directly comparable; ``n_group`` is each group's own total, which is
    what turns a count into a within-group proportion.

    Rows where ``value`` is not finite are dropped — a person with no replicate spread to speak of
    (an undefined CV at zero expected births, say) is absent from the distribution rather than
    piled at an edge. Cells are withheld per
    :func:`~seqeval.metrics._disclosure.suppress_small_cells`.
    """
    cols = [*by, "bin", "bin_lo", "bin_hi", "n_persons", "n_group", "suppressed"]
    data = ind[np.isfinite(ind[value].to_numpy())]
    if data.empty:
        return pd.DataFrame({c: pd.Series(dtype="float64") for c in cols})

    edges = np.histogram_bin_edges(data[value].to_numpy(), bins=n_bins)
    idx = np.clip(np.digitize(data[value].to_numpy(), edges) - 1, 0, len(edges) - 2)
    tagged = data.assign(bin=idx)

    rows = []
    for key, grp in tagged.groupby(by, observed=True):
        key = key if isinstance(key, tuple) else (key,)
        counts = np.bincount(grp["bin"].to_numpy(), minlength=len(edges) - 1)
        for b, n in enumerate(counts):
            rows.append(
                {
                    **dict(zip(by, key, strict=True)),
                    "bin": b,
                    "bin_lo": float(edges[b]),
                    "bin_hi": float(edges[b + 1]),
                    "n_persons": int(n),
                    "n_group": int(len(grp)),
                }
            )
    cells = suppress_small_cells(
        pd.DataFrame(rows), count_col="n_persons", by=by, min_cell=min_cell
    )
    return cells[cols].sort_values([*by, "bin"]).reset_index(drop=True)


def quantile_summary(
    ind: pd.DataFrame,
    *,
    cols: tuple[str, ...] = QUANTILE_COLS,
    by: list[str],
    min_cell: int = MIN_CELL,
) -> pd.DataFrame:
    """Group-mean five-number summary: each person's quantiles averaged over the group.

    A person's own ``q0…q100`` is the spread of *their* completed-birth count across their
    replicates, and is individual-level. Averaging each quantile within a group gives the aggregate
    stand-in: ``mean_q50`` is the typical person's median outcome, ``mean_q75 - mean_q25`` the
    typical person's interquartile spread. It is the mean of the quantiles, not the quantile of the
    population — the group's own spread across people is not what this table is about.

    Returns ``[*by, n_persons, mean_k, mean_q0, mean_q25, mean_q50, mean_q75, mean_q100,
    suppressed]``. ``mean_k`` is there because ``q0`` and ``q100`` are extremes of ``k`` draws and
    drift outward as ``k`` grows: two runs' whiskers are only comparable at equal ``mean_k``.
    Persons with any non-finite quantile are dropped. Groups under ``min_cell`` are withheld per
    :func:`~seqeval.metrics._disclosure.suppress_small_cells`; each row here *is* its own group, so
    only the direct rule applies.
    """
    mean_cols = [f"mean_{c}" for c in cols]
    out_cols = [*by, "n_persons", "mean_k", *mean_cols, "suppressed"]
    have = [c for c in cols if c in ind.columns]
    if len(have) != len(cols) or ind.empty:
        return pd.DataFrame({c: pd.Series(dtype="float64") for c in out_cols})

    data = ind[np.isfinite(ind[list(cols)].to_numpy()).all(axis=1)]
    if data.empty:
        return pd.DataFrame({c: pd.Series(dtype="float64") for c in out_cols})

    grouped = data.groupby(by, observed=True)
    summary = grouped[list(cols)].mean().rename(columns=dict(zip(cols, mean_cols, strict=True)))
    summary["n_persons"] = grouped.size()
    summary["mean_k"] = grouped["k"].mean() if "k" in data.columns else np.nan
    summary = summary.reset_index()

    cells = suppress_small_cells(
        summary, count_col="n_persons", by=by, min_cell=min_cell, also_null=tuple(mean_cols)
    )
    return cells[out_cols].sort_values(by).reset_index(drop=True)
