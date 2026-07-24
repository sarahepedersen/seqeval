"""Reliability diagram (04 viz): no null band, histogram shares the curve's bin edges."""

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


def test_histogram_uses_the_curve_bin_edges():
    joined = _joined()
    cal = ml.calibration_table(joined, n_bins=10, strategy="quantile")
    probs = pd.DataFrame({"p_hat": joined["p_hat"].to_numpy()})
    fig = C.plot_reliability(cal, probs=probs)
    # the lower panel exists and its bars align to the decile edges from the calibration table
    assert len(fig.axes) == 2
    edges = np.append(cal["bin_left"].to_numpy(), cal["bin_right"].to_numpy()[-1])
    bar_lefts = sorted(patch.get_x() for patch in fig.axes[1].patches)
    assert np.allclose(bar_lefts, edges[:-1], atol=1e-9)
