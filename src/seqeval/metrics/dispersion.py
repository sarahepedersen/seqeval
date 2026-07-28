"""Replicate-dispersion distributions: how a per-person spread is distributed across a population.

The per-person dispersion columns (``within_seed_var``, ``within_seed_cv``) describe individuals,
so nothing built directly from them can be published. This module turns one of those columns into
counts per (group, bin) — the shape of the distribution, with no row belonging to anybody — which is
what the ridge figure draws and what a restricted run is able to write.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from seqeval.metrics._disclosure import MIN_CELL, suppress_small_cells

__all__ = ["dispersion_distribution"]


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
