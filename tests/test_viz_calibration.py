"""Reliability diagram (04 viz): no null band, counts share the curve's own bins."""

from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

from seqeval.metrics import ml  # noqa: E402
from seqeval.viz import calibration as C  # noqa: E402


def _joined(n=400, seed=0):
    rng = np.random.default_rng(seed)
    p = rng.uniform(0, 1, n)
    y = (rng.uniform(0, 1, n) < p).astype(int)
    return pd.DataFrame({"p_hat": p, "y_true": y})


def test_reliability_draws_no_null_band():
    cal = ml.calibration_table(_joined(), n_bins=10, strategy="quantile")
    ax = C.plot_reliability(cal).axes[0]
    labels = {ln.get_label() for ln in ax.lines} | {c.get_label() for c in ax.collections}
    assert "null band" not in labels
    assert {"ideal", "model"} <= labels  # the curve and its diagonal are still there


def test_reliability_is_square():
    cal = ml.calibration_table(_joined(), n_bins=10, strategy="quantile")
    assert C.plot_reliability(cal).axes[0].get_aspect() == 1.0


def test_counts_fall_back_to_the_curves_own_bins():
    """With no distribution to draw, the panel is the calibration bins — the old behaviour."""
    cal = ml.calibration_table(_joined(), n_bins=10, strategy="quantile")
    fig = C.plot_reliability(cal)
    assert len(fig.axes) == 2
    bars = sorted(fig.axes[1].patches, key=lambda p: p.get_x())
    assert [p.get_x() for p in bars] == list(cal["bin_left"])
    assert [p.get_height() for p in bars] == list(cal["n"])


def _atomic(n_seeds=5, weights=(0.3, 0.35, 0.2, 0.1, 0.04, 0.01), n=2000):
    """A joined frame whose p_hat sits on the k/n grid, weighted toward the low end."""
    rng = np.random.default_rng(0)
    grid = np.arange(n_seeds + 1) / n_seeds
    p = rng.choice(grid, size=n, p=weights)
    return pd.DataFrame(
        {"p_hat": p, "y_true": (rng.random(n) < p).astype(int), "n": n_seeds}
    )


def test_the_count_panel_draws_the_p_hat_grid_not_the_bins():
    """A wide equal-count bin must not read as weight spread across its whole range."""
    joined = _atomic()
    cal = ml.calibration_table(joined, n_bins=10, strategy="quantile")
    dist = ml.p_hat_distribution(joined, min_cell=0)
    bars = sorted(C.plot_reliability(cal, dist).axes[1].patches, key=lambda p: p.get_x())

    # one bar per attainable p_hat, centred on its own value
    assert len(bars) == len(dist)
    centres = [p.get_x() + p.get_width() / 2 for p in bars]
    np.testing.assert_allclose(centres, dist["p_hat"], atol=1e-9)
    assert [p.get_height() for p in bars] == list(dist["n_persons"])
    # and no bar spans more than one grid step, unlike the calibration bins
    step = 1 / 5
    assert max(p.get_width() for p in bars) <= step
    assert (cal["bin_right"] - cal["bin_left"]).max() > step


def test_withheld_atoms_are_hatched_rather_than_drawn_as_zero():
    joined = _atomic(weights=(0.3, 0.35, 0.2, 0.1489, 0.0001, 0.001), n=20000)
    cal = ml.calibration_table(joined, n_bins=10, strategy="quantile")
    dist = ml.p_hat_distribution(joined, min_cell=5)
    assert dist["suppressed"].any(), "fixture should produce at least one thin atom"
    hatched = [p for p in C.plot_reliability(cal, dist).axes[1].patches if p.get_hatch()]
    assert len(hatched) == int(dist["suppressed"].sum())
    assert all(p.get_height() > 0 for p in hatched)  # "at most this many", never a false zero


def test_empty_bins_are_not_drawn_as_zero_height_bars():
    """p_hat on a coarse grid leaves deciles empty; a zero bar would read as a real trough."""
    cal = pd.DataFrame({
        "bin": [0, 1, 2],
        "bin_left": [0.0, 0.2, 0.6],
        "bin_right": [0.2, 0.6, 1.0],
        "p_mean": [0.1, 0.4, 0.8],
        "y_rate": [0.1, 0.4, 0.8],
        "n": [30, 0, 25],
    })
    fig = C.plot_reliability(cal)
    assert len(fig.axes[1].patches) == 2
    assert all(p.get_height() > 0 for p in fig.axes[1].patches)
    # the curve drops the empty bin too, so both panels describe the same bins
    model = next(ln for ln in fig.axes[0].lines if ln.get_label() == "model")
    assert len(model.get_xdata()) == 2


def test_counts_are_drawn_without_any_per_person_frame():
    """The panel reads the aggregate table, so it survives a run that publishes no individuals."""
    cal = ml.calibration_table(_joined(), n_bins=10, strategy="quantile")
    assert len(C.plot_reliability(cal).axes) == 2
