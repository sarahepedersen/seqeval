"""Reliability diagram with the null calibration band at n_seeds in {5, 50} (02b PR figure).

The methodological pitch of the whole framework in one image: on the *same* synthetic perfect
model, the reliability curve with a few seeds is coarse and its null band wide; with more seeds the
grid refines and the band tightens. A model is only demonstrably miscalibrated where its curve
exits the band — so the shrinking band shows how many replicates a claim requires.

Run from the repo root::

    python examples/reliability_band_demo.py --out examples/data/reliability_band.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tests import synthetic as S  # noqa: E402

from seqeval.core import outcomes as O  # noqa: E402
from seqeval.core import replicates as R  # noqa: E402
from seqeval.core.specs import CountQuery, Frame, ReplicateSpec  # noqa: E402
from seqeval.units import years_to_days as yd  # noqa: E402
from seqeval.viz._labels import describe_outcome  # noqa: E402

GK = ["person_id", "seed", "age_start", "age_stop"]
RK = ["person_id", "age_start", "age_stop"]


def _pipeline(hazards, obs, pers, window, jumpoff, spec, n_seeds, rng):
    gen = S.simulate_generated(obs, pers, hazards, [window], n_seeds, rng)
    ev = O.evaluate_count(gen, GK, spec, O.observation_spans(gen, GK), jumpoff=jumpoff)
    summ = R.replicate_summary(ev, run_keys=RK)
    est = R.estimate_probability(summ, spec=ReplicateSpec(estimator="jeffreys"))
    return summ, est


def _reliability_points(est, y, n_bins):
    p = est.set_index("person_id")["p_hat"].to_numpy()
    yy = y.reindex(est.set_index("person_id").index).to_numpy()
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
    xs, ys = [], []
    for b in range(n_bins):
        sel = idx == b
        if sel.sum() >= 10:
            xs.append(p[sel].mean())
            ys.append(yy[sel].mean())
    return np.array(xs), np.array(ys)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="examples/data/reliability_band.png")
    parser.add_argument("--n", type=int, default=4000, help="cohort size")
    args = parser.parse_args()

    rng = np.random.default_rng(0)
    h = S.default_hazards()
    obs, pers = S.simulate_cohort(args.n, (1960, 1990), h, None, rng)
    window, jo = (0.0, 28.0), yd(28)
    spec = CountQuery("b1w12", "birth", 1, Frame("within", yd(12)))
    osp = O.observation_spans(obs, ["person_id"])
    y = (
        O.evaluate_count(obs, ["person_id"], spec, osp, jumpoff=jo)
        .set_index("person_id")["occurred"]
        .astype(float)
    )

    n_bins = 10
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    for ax, n_seeds in zip(axes, (5, 50), strict=True):
        summ, est = _pipeline(h, obs, pers, window, jo, spec, n_seeds, rng)
        band = R.null_calibration_band(
            summ, n_bins=n_bins, n_sims=500, rng=np.random.default_rng(1), estimator="jeffreys"
        )
        centers = (band["bin_left"] + band["bin_right"]) / 2
        ax.fill_between(
            centers,
            band["lo"],
            band["hi"],
            alpha=0.3,
            color="tab:blue",
            label="null band (perfect calibration)",
        )
        xs, ys = _reliability_points(est, y, n_bins)
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="ideal")
        ax.plot(xs, ys, "o-", color="tab:red", label="perfect model")
        ax.set_title(f"n_seeds = {n_seeds}")
        ax.set_xlabel("predicted probability (p_hat)")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.legend(loc="upper left", fontsize=8)
    axes[0].set_ylabel("observed frequency")
    outcome_desc = describe_outcome(spec, jumpoff_days=jo, label_fn=str)
    fig.suptitle(
        f"Reliability + null band  —  outcome: {outcome_desc}\n"
        "at n=5 the p̂ grid is coarse and estimation noise dominates; at n=50 the curve hugs the "
        "diagonal within a tight band",
        fontsize=10,
    )
    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
