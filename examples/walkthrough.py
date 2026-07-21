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

from seqeval.arms import backtesting as backtesting_arm  # noqa: E402
from seqeval.arms import descriptives as descriptives_arm  # noqa: E402
from seqeval.arms import forecasting as forecasting_arm  # noqa: E402
from seqeval.arms._common import OutputWriter  # noqa: E402
from seqeval.config import (  # noqa: E402
    load_config,
    resolve_conditions,
    resolve_outcomes,
    resolve_probability_outcomes,
    resolve_replicates,
    resolve_rules,
)
from seqeval.core import outcomes as O  # noqa: E402
from seqeval.core import replicates as R  # noqa: E402
from seqeval.core.specs import CountQuery, Frame, ReplicateSpec  # noqa: E402
from seqeval.io.loaders import Bundle, load_all  # noqa: E402
from seqeval.units import DAYS_PER_YEAR, completed_years  # noqa: E402
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

  backtesting:
    windows: all
    conditions:
      - {name: p0, event: birth, max_count: 0}
      - {name: p1, event: birth, min_count: 1, max_count: 1}
    probability_outcomes:
      - {outcome: first_birth, by_age: 35, given: p0}
      - {event: birth, min_events: 1, within: 5}
      - {event: birth, min_events: 1, within: 5, given: p1}
    aggregate_targets: [ccf, ppr]
    min_seeds: 5

  forecasting:
    windows: all
    lexis:
      outcome: first_birth
      ages: [12, 55]
      years: [1975, 2040]
      subgroup_by: []
    illegal_moves:
      - {event: birth, max_age: 50}
      - {event: birth, min_age: 15}
      - {event: birth, min_spacing: 0.6, severity: warn}
    seed_stability: {individual: true, aggregate: [ccf]}
"""

# For the forecasting demo we pretend data is only available up to this calendar year, so recent
# cohorts are incomplete and the model's futures fill the upper-right Lexis triangle.
_FORECAST_CUTOFF_YEAR = 2015


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
    # (0, 15) is an early jump-off so the forecasting Lexis has a full future triangle.
    generated = S.simulate_generated(
        observed, persons, hazards, [(0.0, 15.0), (0.0, 25.0), (0.0, 30.0), (0.0, 35.0)], seeds, rng
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
# forecasting: pretend "today" is a cutoff year so recent cohorts are incomplete
# =================================================================================================
def _calendar_truncate(observed, persons, cutoff_year):
    """Keep observed rows in calendar years <= cutoff and add a no-event marker at the cutoff age.

    Simulates a real data snapshot: cohorts young enough that the cutoff falls mid-life are
    incomplete, so the forecasting arm's model-generated futures fill the rest of the Lexis surface.
    """
    birth_year = observed["person_id"].map(persons.set_index("person_id")["birth_year"])
    year = birth_year.to_numpy() + completed_years(observed["age"].to_numpy())
    kept = observed[year <= cutoff_year].copy()

    # a marker at each person's age when the cutoff is reached (capped at the fertile upper bound)
    cutoff_age = np.clip(
        (cutoff_year - persons["birth_year"].to_numpy()) * DAYS_PER_YEAR, 0, yd(50)
    )
    markers = pd.DataFrame(
        {
            "person_id": persons["person_id"].to_numpy(),
            "age": np.rint(cutoff_age).astype(np.int32),
            "event": "no_event",
        }
    )
    markers = markers[markers["age"] > 0]
    out = pd.concat([kept, markers], ignore_index=True)
    out["event"] = out["event"].astype("category")
    return out.sort_values(["person_id", "age"]).reset_index(drop=True)


def forecasting_section(cfg, bundle, out) -> list[str]:
    """Run the forecasting arm on a calendar-truncated view so the Lexis forecast region shows."""
    truncated = _calendar_truncate(bundle.observed, bundle.persons, _FORECAST_CUTOFF_YEAR)
    fc_bundle = Bundle(
        observed=truncated,
        generated=bundle.generated,
        persons=bundle.persons,
        event_defs=bundle.event_defs,
        events=bundle.events,
    )
    writer = OutputWriter(base_dir=out, arm="forecasting", model=cfg.model.name)
    forecasting_arm.run(
        fc_bundle,
        cfg.arms.forecasting,
        writer,
        outcomes=resolve_outcomes(cfg),
        rules=resolve_rules(cfg),
        replicate_spec=resolve_replicates(cfg),
        cohort_width=cfg.cohort_width,
    )
    return [p.name for p in writer.written if p.suffix == ".png"], writer


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

    banner("4. BACKTESTING ARM (04): probability metrics vs observed truth")
    bt_writer = OutputWriter(base_dir=out, arm="backtesting", model=cfg.model.name)
    resolved = resolve_outcomes(cfg)
    backtesting_arm.run(
        bundle,
        cfg.arms.backtesting,
        bt_writer,
        outcomes=resolved,
        conditions=resolve_conditions(cfg),
        prob_outcomes=resolve_probability_outcomes(cfg, resolved),
        replicate_spec=resolve_replicates(cfg),
        cohort_width=cfg.cohort_width,
    )
    bt_figs = [p.name for p in bt_writer.written if p.suffix == ".png"]
    scores = pd.read_parquet(bt_writer.dir / "scores.parquet")
    headline = (
        scores[scores["metric"].isin(["roc_auc", "brier_corrected", "ece"])]
        .pivot_table(
            index=["age_stop_years", "outcome", "condition"], columns="metric", values="value"
        )
        .round(3)
    )
    print(f"wrote {len(bt_figs)} figures + 6 result tables to backtesting/")
    print("\nheadline scores (one row per window x outcome x condition):")
    print(headline.to_string())

    banner("5. FORECASTING ARM (05): Lexis completion, illegal moves, seed stability")
    fc_figs, fc_writer = forecasting_section(cfg, bundle, out)
    vr = pd.read_parquet(fc_writer.dir / "violation_rates.parquet")
    gen_rate = vr.loc[vr["source"] == "generated", "rate_per_event"].mean()
    obs_rate = vr.loc[vr["source"] == "observed", "rate_per_event"].mean()
    print(f"wrote {len(fc_figs)} figures + Lexis/violations/seed-stability tables to forecasting/")
    print(
        f"illegal-move rate (per event): model {gen_rate:.4f} vs observed baseline {obs_rate:.4f} "
        "(observed rate contextualizes data artifacts)"
    )
    print(f"Lexis forecast region fills calendar years > {_FORECAST_CUTOFF_YEAR}")

    banner("6. REPLICATE ENGINE (02b): reliability band at n=5 vs n=50 + convergence")
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
    lines.append("\n## Backtesting figures (`backtesting/`)\n")
    for name in bt_figs:
        lines.append(f"- `backtesting/{name}`")
    lines.append("\n## Forecasting figures (`forecasting/`)\n")
    for name in fc_figs:
        lines.append(f"- `forecasting/{name}`")
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
          backtesting/   probabilities / calibration / scores / coverage tables + reliability figs
          forecasting/   Lexis surfaces + violations + seed-stability tables + Lexis heatmap
          replicates/    reliability_band.png, convergence_ccf.png
          INDEX.md       a listing of all figures and tables

        Implemented and exercised here: 01 data layer, 02 core outcomes,
        02b replicate engine, 03 descriptives, 04 backtesting, 05 forecasting.
        Not yet built: 06 reporting/CLI.
    """)
    )


if __name__ == "__main__":
    main()
