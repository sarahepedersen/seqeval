"""Dispersion distributions: binning a per-person spread into publishable counts."""

from __future__ import annotations

import numpy as np
import pandas as pd

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
