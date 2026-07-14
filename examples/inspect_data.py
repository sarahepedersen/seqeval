"""Glance at seqeval parquet artifacts: sample sequences, distributions, descriptives.

A read-only inspector for a demo dataset (and, optionally, a results folder). It prints:

  1. artifact overview      — shape / dtypes / memory of each parquet
  2. sample sequences        — a few persons' full observed histories + generated replicates
  3. distributions           — parity, birth ages, events-per-person, window x seed grid, cohorts
  4. result tables (optional)— shape + head of each results/*.parquet

Run from the repo root::

    python examples/inspect_data.py --data examples/walkthrough_output/data
    python examples/inspect_data.py --data examples/walkthrough_output/data \\
        --results examples/walkthrough_output

Nothing is written; ages are shown in years for readability (data on disk is canonical days).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from seqeval.units import days_to_years  # noqa: E402

pd.set_option("display.width", 100)
pd.set_option("display.max_columns", 30)


def banner(title: str) -> None:
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


def yr(days) -> np.ndarray:
    return np.round(days_to_years(np.asarray(days, dtype=float)), 1)


def ascii_hist(values: np.ndarray, *, bins: int = 20, width: int = 46, label: str = "") -> None:
    """A compact text histogram (years on the left, bar + count on the right)."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        print(f"  ({label}: no data)")
        return
    counts, edges = np.histogram(values, bins=bins)
    peak = counts.max() or 1
    for i, c in enumerate(counts):
        bar = "█" * int(round(width * c / peak))
        print(f"  {edges[i]:6.1f}–{edges[i + 1]:<6.1f} | {bar} {c}")


def _birth_token(data_dir: Path) -> tuple[object, str]:
    """(raw birth token, human label) from events.csv if present, else defaults."""
    ev = data_dir / "events.csv"
    if ev.exists():
        defs = pd.read_csv(ev, dtype=str)
        row = defs.iloc[0]
        return row["model_representation"], row["event_definition"]
    return "birth", "birth"


# =================================================================================================
# 1. overview
# =================================================================================================
def overview(frames: dict[str, pd.DataFrame]) -> None:
    banner("1. ARTIFACT OVERVIEW")
    for name, df in frames.items():
        if df is None:
            print(f"{name:10s}: (absent)")
            continue
        mem = df.memory_usage(deep=True).sum() / 1e6
        print(f"\n{name} — {len(df):,} rows x {df.shape[1]} cols, {mem:.1f} MB")
        dtypes = ", ".join(f"{c}:{t}" for c, t in df.dtypes.astype(str).items())
        print(f"  dtypes: {dtypes}")


# =================================================================================================
# 2. sample sequences
# =================================================================================================
def sample_sequences(observed, generated, persons, birth_token, label, n_examples=4) -> None:
    banner("2. SAMPLE SEQUENCES")
    parity = (
        observed[observed["event"] == birth_token]
        .groupby("person_id")
        .size()
        .reindex(observed["person_id"].unique(), fill_value=0)
    )
    # one example person at each of a few parities
    picks = []
    for target in (0, 1, 2, 3):
        cands = parity[parity == target].index
        if len(cands):
            picks.append(int(cands[0]))
    pinfo = persons.set_index("person_id") if persons is not None else None

    for pid in picks[:n_examples]:
        seq = observed[observed["person_id"] == pid].sort_values("age")
        events = ", ".join(
            f"{label if e == birth_token else e}@{a}"
            for e, a in zip(seq["event"].astype(str), yr(seq["age"]), strict=True)
        )
        meta = ""
        if pinfo is not None and pid in pinfo.index:
            row = pinfo.loc[pid]
            meta = " | " + ", ".join(f"{c}={row[c]}" for c in pinfo.columns)
        print(f"\nperson {pid} (parity {int(parity[pid])}{meta})")
        print(f"  observed: [{events}]")

        if generated is not None:
            runs = generated[generated["person_id"] == pid]
            if len(runs):
                start, stop = runs[["age_start", "age_stop"]].iloc[0]
                w = runs[(runs["age_start"] == start) & (runs["age_stop"] == stop)]
                n_show = min(3, w["seed"].nunique())
                print(f"  generated future (window jump-off age {yr(stop)}, first {n_show} seeds):")
                for seed in sorted(w["seed"].unique())[:3]:
                    s = w[w["seed"] == seed].sort_values("age")
                    fut = ", ".join(
                        f"{label if e == birth_token else e}@{a}"
                        for e, a in zip(s["event"].astype(str), yr(s["age"]), strict=True)
                    )
                    print(f"    seed {seed}: [{fut}]")


# =================================================================================================
# 3. distributions
# =================================================================================================
def distributions(observed, generated, persons, birth_token, label) -> None:
    banner("3. DISTRIBUTIONS & DESCRIPTIVES")
    n_persons = observed["person_id"].nunique()
    print(f"population: {n_persons:,} persons")

    if persons is not None:
        if "sex" in persons.columns:
            print(f"  sex: {persons['sex'].value_counts(dropna=False).to_dict()}")
        lo, hi = int(persons["birth_year"].min()), int(persons["birth_year"].max())
        print(f"  birth_year range: {lo}–{hi}")
        for cov in [c for c in persons.columns if c not in ("person_id", "birth_year", "sex")]:
            print(f"  {cov}: {persons[cov].value_counts(dropna=False).to_dict()}")

    # parity distribution (0 for persons with no births)
    births = observed[observed["event"] == birth_token]
    parity = (
        births.groupby("person_id").size().reindex(observed["person_id"].unique(), fill_value=0)
    )
    pcounts = parity.value_counts().sort_index()
    print("\nparity distribution (# births per person):")
    for k, c in pcounts.items():
        print(f"  parity {k}: {c:5d}  ({100 * c / n_persons:4.1f}%)")
    print(f"  mean births per person (CCF-like): {parity.mean():.3f}")

    # events per person
    epp = observed.groupby("person_id").size()
    print(f"\nrows per person: min={epp.min()} median={epp.median():.0f} max={epp.max()}")

    # birth-age distribution
    if len(births):
        print(f"\nbirth ages (years) — {label}: describe")
        print(pd.Series(yr(births["age"])).describe().round(1).to_string())
        print("\nbirth-age histogram (years):")
        ascii_hist(yr(births["age"]), bins=18, label="birth ages")

        first = births.sort_values(["person_id", "age"]).groupby("person_id").first()
        print("\nage at first birth (years): describe")
        print(pd.Series(yr(first["age"])).describe().round(1).to_string())

    # generated
    if generated is not None:
        banner("3b. GENERATED (window x seed grid + future births)")
        grid = (
            generated.groupby(["age_start", "age_stop"], observed=True)
            .agg(
                n_seeds=("seed", "nunique"),
                n_persons=("person_id", "nunique"),
                n_rows=("age", "size"),
            )
            .reset_index()
        )
        grid["jumpoff_yr"] = yr(grid["age_stop"])
        print(grid[["jumpoff_yr", "n_seeds", "n_persons", "n_rows"]].to_string(index=False))

        gb = generated[generated["event"] == birth_token]
        per_run = gb.groupby(["person_id", "seed", "age_start", "age_stop"], observed=True).size()
        print(
            f"\nfuture births per (person, window, seed): "
            f"mean={per_run.mean():.2f} max={per_run.max()} "
            f"(runs with >=1 future birth: {len(per_run):,})"
        )
        print("\nfuture-birth ages (years) histogram:")
        ascii_hist(yr(gb["age"]), bins=18, label="future births")


# =================================================================================================
# 4. result tables
# =================================================================================================
def result_tables(results_dir: Path) -> None:
    banner("4. RESULT TABLES")
    for arm_dir in sorted(p for p in results_dir.iterdir() if p.is_dir()):
        parquets = sorted(arm_dir.glob("*.parquet"))
        if not parquets:
            continue
        print(f"\n--- {arm_dir.name}/ ---")
        for pq in parquets:
            df = pd.read_parquet(pq)
            print(f"\n{pq.name}  ({len(df):,} rows x {df.shape[1]} cols)")
            print(df.head(4).to_string(index=False))


# =================================================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        default="examples/walkthrough_output/data",
        help="directory with observed/generated/persons parquet",
    )
    parser.add_argument("--results", default=None, help="optional results directory to glance at")
    args = parser.parse_args()

    data_dir = Path(args.data)
    birth_token, label = _birth_token(data_dir)

    def _load(name):
        p = data_dir / f"{name}.parquet"
        return pd.read_parquet(p) if p.exists() else None

    observed = _load("observed")
    if observed is None:
        raise SystemExit(f"no observed.parquet under {data_dir}; run examples/walkthrough.py first")
    generated = _load("generated")
    persons = _load("persons")

    overview({"observed": observed, "generated": generated, "persons": persons})
    sample_sequences(observed, generated, persons, birth_token, label)
    distributions(observed, generated, persons, birth_token, label)
    if args.results:
        result_tables(Path(args.results))


if __name__ == "__main__":
    main()
