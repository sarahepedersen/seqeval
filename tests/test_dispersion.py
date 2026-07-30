"""Dispersion aggregates: binning a per-person spread, and averaging per-person quantiles."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from seqeval.metrics import dispersion as MD


def _ind(values, groups, col="age_stop"):
    return pd.DataFrame(
        {"person_id": range(len(values)), "within_seed_var": values, col: groups}
    )


def test_counts_are_binned_on_shared_edges_so_groups_are_comparable():
    ind = _ind([*np.linspace(0, 1, 40), *np.linspace(0, 1, 40)], [0] * 40 + [1] * 40)
    dist = MD.dispersion_distribution(ind, by=["age_stop"], n_bins=5, min_cell=0)
    edges = dist.groupby("age_stop")[["bin_lo", "bin_hi"]].apply(lambda d: d.to_numpy().tolist())
    assert edges.iloc[0] == edges.iloc[1]  # one set of edges, reused per group


def test_every_persons_value_lands_in_exactly_one_cell():
    ind = _ind(np.linspace(0, 2, 60), [0] * 30 + [1] * 30)
    dist = MD.dispersion_distribution(ind, by=["age_stop"], n_bins=8, min_cell=0)
    assert dist["n_persons"].sum() == 60
    assert (dist.groupby("age_stop")["n_group"].first() == 30).all()


def test_non_finite_values_are_dropped_not_piled_at_an_edge():
    """An undefined CV is absent from the distribution rather than counted at zero."""
    ind = _ind([0.1, 0.2, np.nan, np.inf, 0.3], [0] * 5)
    dist = MD.dispersion_distribution(ind, by=["age_stop"], n_bins=4, min_cell=0)
    assert dist["n_persons"].sum() == 3
    assert dist["n_group"].iloc[0] == 3


def test_output_names_nobody():
    ind = _ind(np.linspace(0, 1, 50), [0] * 25 + [1] * 25)
    dist = MD.dispersion_distribution(ind, by=["age_stop"], n_bins=6)
    assert "person_id" not in dist.columns


def test_thin_cells_are_withheld():
    # a long right tail: a handful of people far from the mass
    ind = _ind([*np.full(60, 0.05), 0.9, 0.95, 1.0], [0] * 63)
    dist = MD.dispersion_distribution(ind, by=["age_stop"], n_bins=10)
    assert dist["suppressed"].any()
    assert dist.loc[dist["suppressed"], "n_persons"].isna().all()


def _quantiles(rows, col="age_stop"):
    """Individual-table shaped: one person per (q0, q25, q50, q75, q100, k, group)."""
    return pd.DataFrame(
        [
            {
                "person_id": i,
                **dict(zip(MD.QUANTILE_COLS, q, strict=True)),
                "k": k,
                col: g,
            }
            for i, (q, k, g) in enumerate(rows)
        ]
    )


def test_quantile_summary_averages_each_quantile_within_the_group():
    ind = _quantiles(
        [
            ([0, 1, 1, 2, 3], 5, 30),
            ([0, 1, 2, 3, 4], 5, 30),
            ([1, 1, 1, 1, 1], 5, 40),
        ]
    )
    out = MD.quantile_summary(ind, by=["age_stop"], min_cell=0).set_index("age_stop")
    assert out.loc[30, "mean_q0"] == pytest.approx(0.0)
    assert out.loc[30, "mean_q50"] == pytest.approx(1.5)
    assert out.loc[30, "mean_q100"] == pytest.approx(3.5)
    assert out.loc[30, "n_persons"] == 2
    assert out.loc[30, "mean_k"] == pytest.approx(5.0)
    assert (out.loc[40, ["mean_q0", "mean_q50", "mean_q100"]] == 1.0).all()


def test_quantile_summary_withholds_a_thin_group():
    """Under the threshold the means are NA — the row survives so the gap is visible."""
    ind = _quantiles([([0, 1, 1, 2, 3], 5, 30)] * 8 + [([0, 0, 1, 1, 2], 5, 40)] * 2)
    out = MD.quantile_summary(ind, by=["age_stop"], min_cell=5).set_index("age_stop")
    assert not out.loc[30, "suppressed"] and out.loc[30, "mean_q50"] == pytest.approx(1.0)
    assert out.loc[40, "suppressed"]
    assert pd.isna(out.loc[40, "mean_q50"]) and pd.isna(out.loc[40, "n_persons"])


def test_quantile_summary_drops_persons_with_a_non_finite_quantile():
    ind = _quantiles([([0, 1, 1, 2, 3], 5, 30), ([np.nan, 1, 1, 2, 3], 5, 30)])
    out = MD.quantile_summary(ind, by=["age_stop"], min_cell=0)
    assert out["n_persons"].iloc[0] == 1


def test_quantile_summary_names_nobody_and_keeps_its_column_order():
    ind = _quantiles([([0, 1, 1, 2, 3], 5, 30)] * 6)
    out = MD.quantile_summary(ind, by=["age_stop"], min_cell=0)
    assert list(out.columns) == [
        "age_stop", "n_persons", "mean_k",
        "mean_q0", "mean_q25", "mean_q50", "mean_q75", "mean_q100", "suppressed",
    ]


def test_quantile_summary_without_the_quantile_columns_is_empty_not_an_error():
    """A run whose individual table predates the five-number summary emits an empty frame."""
    out = MD.quantile_summary(_ind([0.1, 0.2], [30, 30]), by=["age_stop"], min_cell=0)
    assert out.empty and "mean_q50" in out.columns


def test_grouping_by_two_columns_gives_a_row_per_pair():
    ind = pd.DataFrame(
        {
            "person_id": range(40),
            "within_seed_var": np.linspace(0, 1, 40),
            "cohort": [1960, 1965] * 20,
            "age_stop": [9131] * 20 + [10958] * 20,
        }
    )
    dist = MD.dispersion_distribution(ind, by=["cohort", "age_stop"], n_bins=4, min_cell=0)
    assert dist.groupby(["cohort", "age_stop"]).ngroups == 4
    assert dist["n_persons"].sum() == 40
