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


def test_counts_come_from_the_curves_own_bins():
    """One bar per calibration row, so the curve and the counts cannot disagree."""
    cal = ml.calibration_table(_joined(), n_bins=10, strategy="quantile")
    fig = C.plot_reliability(cal)
    assert len(fig.axes) == 2
    bars = sorted(fig.axes[1].patches, key=lambda p: p.get_x())
    assert [p.get_x() for p in bars] == list(cal["bin_left"])
    assert [p.get_height() for p in bars] == list(cal["n"])


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
