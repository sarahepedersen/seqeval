"""Small-cell suppression for published aggregates.

Every table that is safe to publish is a table of binned counts, and a binned count is only safe
while no cell is thin enough to be about one identifiable person. This module holds that rule, once,
so the ridge (:func:`seqeval.metrics.ml.timing_error_distribution`) and the parity distribution
(:func:`seqeval.metrics.fertility.parity_distribution`) suppress on identical terms — a reader who
learns the convention in one figure reads the other.
"""

from __future__ import annotations

import pandas as pd

#: Cells resting on fewer than this many people are withheld.
MIN_CELL = 5

__all__ = ["MIN_CELL", "suppress_small_cells"]


def suppress_small_cells(
    df: pd.DataFrame,
    *,
    count_col: str,
    by: list[str],
    min_cell: int = MIN_CELL,
    also_null: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Withhold cells resting on too few people; return ``df`` plus a ``suppressed`` flag.

    A suppressed cell keeps its row and its bin edges — only ``count_col`` and the columns named in
    ``also_null`` become NA, so the shape of the table (and of the figure drawn from it) survives.
    Three rules:

    - ``0 < count < min_cell`` is suppressed.
    - ``count == 0`` is published. A true zero names nobody, and a density with holes punched where
      the count happened to be zero would misread as suppression.
    - Within each ``by`` group, a *lone* suppressed cell forces a second: the smallest non-zero cell
      still standing is suppressed too. One withheld cell is recoverable by subtracting the
      published cells from the group total, so suppressing it alone withholds nothing.
    """
    out = df.copy()
    counts = out[count_col]
    out["suppressed"] = (counts > 0) & (counts < min_cell)

    groups = out.groupby(by, observed=True).indices.values() if by else [out.index.to_numpy()]
    for idx in groups:
        grp = out.loc[idx]
        if int(grp["suppressed"].sum()) != 1:
            continue
        eligible = grp[~grp["suppressed"] & (grp[count_col] > 0)]
        if len(eligible):
            out.loc[eligible[count_col].idxmin(), "suppressed"] = True

    hidden = out["suppressed"].to_numpy()
    out[count_col] = out[count_col].astype("Int64").mask(hidden)
    for col in also_null:
        out[col] = out[col].mask(hidden)
    return out
