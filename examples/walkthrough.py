"""End-to-end walkthrough of everything implemented so far (plans 01, 02, 02b, 03).

Runs the whole pipeline on a synthetic "perfect model" and writes every figure the implemented
layers can produce, so the implementation can be evaluated visually and numerically in one place:

  1. Data layer (01)      — generate demo artifacts, load + validate, print population summary
  2. Core outcomes (02)   — births / spans / time-to-event / binary-outcome evaluators (preview)
  3. Descriptives (03)     — KM curves, ASFR profiles, CCF, PPR, life table (arm + figures)
  4. Replicate engine (02b)— empirical probabilities, reliability + null band, Brier correction,
                              seed-convergence curve

Run from the repo root::

    python examples/walkthrough.py --out examples/walkthrough_output

Then open ``<out>/INDEX.md`` and the ``*.png`` files. The backtesting (04), forecasting (05), and
reporting/CLI (06) arms are not implemented yet, so this walkthrough drives their building blocks
directly instead of through a `seqeval run` command.
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tests import synthetic as S  # noqa: E402

from seqeval.arms import descriptives as descriptives_arm  # noqa: E402
from seqeval.arms._common import OutputWriter  # noqa: E402
from seqeval.config import load_config, resolve_outcomes  # noqa: E402
from seqeval.core import outcomes as O  # noqa: E402
from seqeval.core import replicates as R  # noqa: E402
from seqeval.core.specs import CountQuery, Frame, ReplicateSpec  # noqa: E402
from seqeval.io.loaders import load_all  # noqa: E402
from seqeval.units import years_to_days as yd  # noqa: E402
from seqeval.viz._labels import describe_outcome  # noqa: E402

GK = ["person_id", "seed", "age_start", "age_stop"]
RK = ["person_id", "age_start", "age_stop"]

_CONFIG_YAML = """\
model:
  name: synthetic_perfect_model

data:
  observed: observed.parquet
  generated: generated.parquet
  persons: persons.parquet
  event_definitions: events.csv
  age_unit: days

events:
  birth: birth

persons:
  covariates: [education, region]
  cohort_width: 5          # birth-cohort band width (years) — shared by every arm

replicates:
  estimator: jeffreys
  interval: jeffreys
  level: 0.95
  min_replicates: 5
  bootstrap: {n: 200, seed: 7}
  convergence_curve: true

outcomes:
  first_birth: {event: birth, n: 1}
  second_birth: {event: birth, n: 2, origin: first_birth}

arms:
  descriptives:
    kaplan_meier: [first_birth, second_birth]
    fertility:
      ccf: true
      asfr: [period, cohort]
      ppr: {max_parity: 6}
    life_table: {max_parity: 6}
    stratify_by: [cohort]
"""


def banner(title: str) -> None:
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


# =================================================================================================
# 1. data layer
# =================================================================================================
def build_demo_data(data_dir: Path, n: int, seeds: int, rng) -> Path:
    """Generate synthetic artifacts + a matching config.yaml; return the config path."""
    data_dir.mkdir(parents=True, exist_ok=True)
    hazards = S.default_hazards()
    # Full exposure to the fertile upper bound (marker at 50 for everyone) so the descriptive
    # identities (cohort ASFR sum == CCF) hold cleanly for evaluation.
    observed, persons = S.simulate_cohort(
        n, (1960, 1990), hazards, None, rng, no_event_fraction=1.0
    )
    generated = S.simulate_generated(
        observed, persons, hazards, [(0.0, 25.0), (0.0, 30.0), (0.0, 35.0)], seeds, rng
    )
    observed.to_parquet(data_dir / "observed.parquet", index=False)
    generated.to_parquet(data_dir / "generated.parquet", index=False)
    persons.to_parquet(data_dir / "persons.parquet", index=False)
    pd.DataFrame(
        {"model_representation": [S.BIRTH_TOKEN], "event_definition": ["live birth"]}
    ).to_csv(data_dir / "events.csv", index=False)
    (data_dir / "config.yaml").write_text(_CONFIG_YAML)
    return data_dir / "config.yaml"


# =================================================================================================
# 4. replicate-engine showcase (a manual preview of the not-yet-built backtesting arm)
# =================================================================================================
def replicate_showcase(bundle, hazards, out_dir: Path) -> list[tuple[str, str]]:
    """Reliability + null band at two seed counts, Brier correction, and a convergence curve."""
    figures: list[tuple[str, str]] = []
    obs, persons = bundle.observed, bundle.persons
    window, jo = (0.0, 28.0), yd(28)
    spec = CountQuery("b1w12", "birth", 1, Frame("within", yd(12)))

    # Observed truth label per person for this window.
    osp = O.observation_spans(obs, ["person_id"])
    y = (
        O.evaluate_count(obs, ["person_id"], spec, osp, jumpoff=jo)
        .set_index("person_id")["occurred"]
        .astype(float)
    )

    def pipeline(n_seeds, rng):
        gen = S.simulate_generated(obs, persons, hazards, [window], n_seeds, rng)
        ev = O.evaluate_count(gen, GK, spec, O.observation_spans(gen, GK), jumpoff=jo)
        summ = R.replicate_summary(ev, run_keys=RK)
        est = R.estimate_probability(summ, spec=ReplicateSpec(estimator="jeffreys"))
        return gen, summ, est

    # --- reliability diagram with null band, n_seeds in {5, 50} ---------------------------------
    n_bins = 10
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    brier_rows = []
    for ax, n_seeds in zip(axes, (5, 50), strict=True):
        rng = np.random.default_rng(100 + n_seeds)
        _, summ, est = pipeline(n_seeds, rng)
        band = R.null_calibration_band(
            summ, n_bins=n_bins, n_sims=400, rng=np.random.default_rng(1), estimator="jeffreys"
        )
        centers = (band["bin_left"] + band["bin_right"]) / 2
        ax.fill_between(
            centers, band["lo"], band["hi"], alpha=0.3, color="tab:blue", label="null band"
        )
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
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="ideal")
        ax.plot(xs, ys, "o-", color="tab:red", label="model")
        ax.set(title=f"n_seeds = {n_seeds}", xlabel="predicted p_hat", xlim=(0, 1), ylim=(0, 1))
        ax.set_aspect("equal")
        ax.legend(loc="upper left", fontsize=8)

        raw = float(
            np.mean(
                (est.set_index("person_id")["p_hat"] - y.reindex(est.set_index("person_id").index))
                ** 2
            )
        )
        corr = R.brier_noise_correction(summ)
        brier_rows.append(
            {
                "n_seeds": n_seeds,
                "brier_raw": raw,
                "correction": corr,
                "brier_corrected": raw - corr,
            }
        )
    axes[0].set_ylabel("observed frequency")
    outcome_desc = describe_outcome(spec, jumpoff_days=jo, label_fn=bundle.label)
    fig.suptitle(
        f"Reliability + null band  —  outcome: {outcome_desc}\n"
        "coarse and wide at n=5, tight at n=50",
        fontsize=11,
    )
    fig.tight_layout()
    path = out_dir / "reliability_band.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    figures.append(
        (
            path.name,
            f"Reliability diagram + perfect-calibration null band for {outcome_desc} "
            "(n_seeds 5 vs 50)",
        )
    )

    print("\nBrier score: few seeds inflate it; the MC-error correction recovers the truth")
    print(pd.DataFrame(brier_rows).to_string(index=False))

    # --- seed-convergence curve for CCF ---------------------------------------------------------
    rng = np.random.default_rng(7)
    gen50 = S.simulate_generated(obs, persons, hazards, [(0.0, 0.0)], 50, rng)

    def ccf_stat(df):
        b = df[df["event"] == "birth"]
        return pd.DataFrame({"ccf": [len(b) / df["seed"].nunique() / df["person_id"].nunique()]})

    cc = R.convergence_curve(
        gen50,
        seed_col="seed",
        stat_fn=ccf_stat,
        sizes=[2, 3, 5, 8, 12, 20, 30, 50],
        n_rep=25,
        rng=np.random.default_rng(2),
    )
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(cc["m"], cc["mean"], yerr=cc["std"], marker="o", capsize=3, color="tab:purple")
    ax.axhline(S.expected_ccf(hazards), color="k", ls="--", lw=1, label="converged truth")
    ax.set(
        xlabel="number of seeds m",
        ylabel="CCF estimate",
        title="Seed-convergence of CCF (mean +/- sd)",
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    path = out_dir / "convergence_ccf.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    figures.append(
        (
            path.name,
            "Dispersion of the CCF estimate shrinking as seeds increase (replicate power analysis)",
        )
    )
    return figures


# =================================================================================================
# driver
# =================================================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="examples/walkthrough_output")
    parser.add_argument("--n", type=int, default=3000, help="cohort size")
    parser.add_argument(
        "--seeds", type=int, default=5, help="replicates in the demo generated file"
    )
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    hazards = S.default_hazards()

    banner("1. DATA LAYER (01): generate + load + validate")
    cfg_path = build_demo_data(out / "data", args.n, args.seeds, rng)
    cfg = load_config(cfg_path)
    bundle = load_all(cfg)
    print(f"model: {cfg.model.name}")
    print(f"population summary: {bundle.population_summary()}")
    print("available (window x seed) grid:")
    print(bundle.available_windows().to_string(index=False))

    banner("2. CORE OUTCOMES (02): extraction preview")
    births = O.births(bundle.observed, ["person_id"], birth_event=bundle.token("birth"))
    spans = O.observation_spans(bundle.observed, ["person_id"])
    print(f"births rows: {len(births):,} | example:\n{births.head(3).to_string(index=False)}")
    print(f"\nobservation spans (days) example:\n{spans.head(3).to_string(index=False)}")

    banner("3. DESCRIPTIVES ARM (03): metrics + figures")
    writer = OutputWriter(base_dir=out, arm="descriptives", model=cfg.model.name)
    descriptives_arm.run(
        bundle,
        cfg.arms.descriptives,
        writer,
        outcomes=resolve_outcomes(cfg),
        cohort_width=cfg.cohort_width,
    )
    desc_figs = [(p.name, p) for p in writer.written if p.suffix == ".png"]
    for _name, path in desc_figs:
        print(f"  wrote {path.relative_to(out)}")
    # A couple of headline numbers to eyeball.
    ccf = pd.read_parquet(writer.dir / "ccf.parquet")
    print(
        f"\nCCF by cohort (mean over cohorts): {ccf['ccf'].mean():.3f}  "
        f"(converged truth ~ {S.expected_ccf(hazards):.3f})"
    )

    banner("4. REPLICATE ENGINE (02b): empirical probabilities + calibration")
    rep_dir = out / "replicates"
    rep_dir.mkdir(exist_ok=True)
    rep_figs = replicate_showcase(bundle, hazards, rep_dir)

    # --- INDEX.md -------------------------------------------------------------------------------
    lines = [
        "# seqeval walkthrough output\n",
        f"Generated on a synthetic perfect-model cohort of {args.n} persons.\n",
        "## Descriptive figures (`descriptives/`)\n",
    ]
    for name, _path in desc_figs:
        lines.append(f"- `descriptives/{name}`")
    lines.append("\n## Replicate-engine figures (`replicates/`)\n")
    for name, caption in rep_figs:
        lines.append(f"- `replicates/{name}` — {caption}")
    lines.append("\n## Result tables\n")
    for path in sorted(writer.written):
        if path.suffix == ".parquet":
            lines.append(f"- `descriptives/{path.name}`")
    (out / "INDEX.md").write_text("\n".join(lines) + "\n")

    banner("DONE")
    print(
        textwrap.dedent(f"""
        Wrote everything under: {out}/
          data/          demo artifacts + config.yaml
          descriptives/  KM / ASFR / CCF / PPR / life-table tables + figures
          replicates/    reliability_band.png, convergence_ccf.png
          INDEX.md       a listing of all figures and tables

        Implemented and exercised here: 01 data layer, 02 core outcomes,
        02b replicate engine, 03 descriptives. Not yet built: 04 backtesting,
        05 forecasting, 06 reporting/CLI.
    """)
    )


if __name__ == "__main__":
    main()
