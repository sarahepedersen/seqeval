"""06 — the run manifest and the self-contained HTML report.

Two concerns live here, both pure functions over already-written results:

- **Manifest** (:func:`build_manifest` / :func:`write_manifest`): ``results/manifest.json`` records
  everything needed to reproduce and audit a run — seqeval model, the SHA-256 of every
  input file, the fully-resolved config, per-arm status/outputs/timing, data coverage, and the
  verbatim warning list surfaced by the lower layers. Two runs on identical inputs+config produce
  byte-identical manifests except for the ``timestamp_utc`` and ``duration_s`` fields (00 rule:
  content-hashed inputs, canonical JSON).
- **Report** (:func:`build_report`): ``results/report.html`` — one static, self-contained document
  (no server, no JS build) assembling each arm's figures (embedded as base64 PNGs) and tables
  (styled ``DataFrame.to_html``, capped and linked to the parquet) into one reviewable artifact.
  One function per section; only sections whose results are present are rendered.

The CLI (:mod:`seqeval.cli`) orchestrates the run and feeds this module; nothing here loads data or
computes metrics.
"""

from __future__ import annotations

import base64
import hashlib
import html
import json
import logging
import re
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from seqeval.metrics._disclosure import MIN_CELL
from seqeval.units import days_to_years

logger = logging.getLogger("seqeval")

MANIFEST_NAME = "manifest.json"
REPORT_NAME = "report.html"

#: Arms in execution order (00 section 5: descriptives -> backtesting -> forecasting). The report
#: presents them in a different order and under different names — see :data:`SECTIONS`.
ARM_ORDER = ["descriptives", "backtesting", "forecasting"]

#: Image suffixes embedded inline in the report; anything else is linked, not embedded.
_EMBED_SUFFIXES = {".png", ".jpg", ".jpeg", ".svg"}

#: Rows shown per table in the HTML report before truncation (full data stays in the parquet).
_MAX_TABLE_ROWS = 50
#: Rows of the backing table shown under each figure.
_PEEK_ROWS = 5


# =================================================================================================
# hashing / input records
# =================================================================================================
def sha256_file(path: str | Path) -> str:
    """Stream a file through SHA-256 and return the hex digest (content, not path)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _parquet_rows(path: Path) -> int | None:
    """Row count from parquet metadata (cheap; no full read), or ``None`` for non-parquet."""
    if path.suffix != ".parquet":
        return None
    return int(pq.ParquetFile(path).metadata.num_rows)


def input_records(cfg) -> dict[str, dict]:
    """SHA-256 + row count for every present input artifact, keyed by role.

    Paths are recorded as written in the config (relative to the YAML), so the manifest is stable
    across machines; identity comes from the content hash, not the path.
    """
    roles = {
        "observed": (cfg.data.observed, cfg.observed_path),
        "generated": (cfg.data.generated, cfg.generated_path),
        "persons": (cfg.data.persons, cfg.persons_path),
        "event_definitions": (cfg.data.event_definitions, cfg.event_definitions_path),
    }
    out: dict[str, dict] = {}
    for role, (rel, abs_path) in roles.items():
        if abs_path is None:
            continue
        out[role] = {
            "path": rel,
            "sha256": sha256_file(abs_path),
            "n_rows": _parquet_rows(Path(abs_path)),
        }
    return out


# =================================================================================================
# coverage (data summary shared by validate output and the report)
# =================================================================================================
def coverage_block(bundle, cfg) -> dict:
    """Population + window×replicate grid for the manifest and report run-summary.

    Windows are emitted in **years**.
    """
    summary = bundle.population_summary()
    windows = []
    for row in bundle.available_windows().itertuples(index=False):
        windows.append(
            {
                "age_start": round(days_to_years(int(row.age_start)), 2),
                "age_stop": round(days_to_years(int(row.age_stop)), 2),
                "n_seeds": int(row.n_seeds),
                "n_persons": int(row.n_persons),
            }
        )
    return {
        "n_persons": summary["n_persons"],
        "sex_breakdown": summary["sex_breakdown"],
        "cohort_range": summary["cohort_range"],
        "windows": windows,
    }


# =================================================================================================
# manifest
# =================================================================================================
def build_manifest(*, cfg, coverage: dict, arm_results: list[dict], warnings: list[str]) -> dict:
    """Assemble the manifest dict from a completed run.

    ``arm_results`` is a list of ``{"name", "status", "outputs", "duration_s"}`` (``outputs`` are
    paths relative to the results dir). Everything is deterministic given the inputs except
    ``timestamp_utc`` and each arm's ``duration_s``.
    """
    arms = {}
    for res in arm_results:
        arms[res["name"]] = {
            "status": res["status"],
            "outputs": sorted(res["outputs"]),
            # Named rather than merely absent: a reader can tell a withheld artifact from one the
            # run never produced.
            "withheld": sorted(res.get("withheld", [])),
            "duration_s": res["duration_s"],
        }
    return {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "model": {"name": cfg.model.name},
        "config_hash": cfg.hash(),
        "config_resolved": cfg.model_dump(mode="json"),
        "inputs": input_records(cfg),
        "coverage": coverage,
        "arms": arms,
        "warnings": list(warnings),
    }


def write_manifest(out_dir: str | Path, manifest: dict) -> Path:
    """Write ``manifest.json`` (canonical, sorted keys) under ``out_dir`` and return its path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / MANIFEST_NAME
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return path


def read_manifest(results_dir: str | Path) -> dict | None:
    """Load ``manifest.json`` from a results dir, or ``None`` if absent."""
    path = Path(results_dir) / MANIFEST_NAME
    if not path.exists():
        return None
    return json.loads(path.read_text())


# =================================================================================================
# HTML report
# =================================================================================================
_CSS = """
body { font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
       max-width: 1100px; margin: 2rem auto; padding: 0 1.5rem; color: #1a1a1a;
       line-height: 1.5; }
h1 { border-bottom: 2px solid #333; padding-bottom: .3rem; }
h2 { margin-top: 2.5rem; border-bottom: 1px solid #ccc; padding-bottom: .2rem; }
h3 { margin-top: 1.5rem; color: #333; }
nav a { margin-right: 1rem; }
h4 { margin: 1rem 0 .3rem; color: #444; }
figure { margin: 1rem 0; }
img { max-width: 100%; height: auto; border: 1px solid #eee; }
table { border-collapse: collapse; margin: .5rem 0 1rem; font-size: .85rem; }
th, td { border: 1px solid #ddd; padding: .3rem .5rem; text-align: right; }
th { background: #f5f5f5; }
.meta td, .meta th { text-align: left; }
td.rowhdr { text-align: left; font-weight: 600; vertical-align: top; background: #fafafa; }
.figrow { display: flex; flex-wrap: wrap; gap: 1rem; align-items: flex-start; margin: .5rem 0; }
.figrow figure { flex: 1 1 340px; max-width: 460px; margin: 0; }
.figrow img { width: 100%; }
.peek { margin: .2rem 0 .6rem; }
.peek summary { cursor: pointer; }
.peek table { font-size: .72rem; margin: .3rem 0 0; display: block; overflow-x: auto; }
ul.muted { margin: .3rem 0 1rem; padding-left: 1.2rem; }
ul.muted li { margin: .15rem 0; }
.subnav { margin: .2rem 0 .8rem; }
.subnav a { margin-right: .8rem; }
.basis { margin: .2rem 0 .6rem; }
.warn { background: #fff8e1; border-left: 4px solid #f0ad4e; padding: .5rem 1rem; }
.flag { color: #b34700; font-weight: 600; }
.muted { color: #777; font-size: .85rem; }
code { background: #f2f2f2; padding: .1rem .3rem; border-radius: 3px; }
"""

#: How each plot/table group treats replicates — the distinction that governs how its interval
#: should be read. Each entry is one of the keys of :data:`_BASIS_TEXT` (``"averaged"``,
#: ``"trajectories"``, ``"observed"``), or ``None`` for a group where the distinction does not
#: apply and no line is drawn.
REPLICATE_BASIS: dict[str, str | None] = {
    # Observed Sequences
    "observed.km": "observed",
    "observed.ccf": "observed",
    "observed.asfr": "observed",
    # Generated Sequences
    "generated.lexis": "trajectories",
    "generated.dispersion": "averaged",
    "generated.replicate_variance_individual": "averaged",
    "generated.replicate_occurrence": None,
    "generated.violations": None,
    "generated.replicate_variance_aggregate": "averaged",
    "generated.violation_rates": None,
    # Observed and Generated Comparison
    "comparison.metrics": "averaged",
    "comparison.km": "trajectories",
    "comparison.ccf": "averaged",
    "comparison.ppr": "trajectories",
    "comparison.asfr": "trajectories",
    "comparison.uncertainty": None,
    "comparison.calibration": None,
    "comparison.timing": "trajectories",
}

#: The two ways a group can be built, spelled out for the reader.
_BASIS_TEXT = {
    "averaged": (
        "values are computed bottom-up by first averaging across within-individual replicates; CIs for means are "
        "computed analytically the sum of per-individual variances and between individual differences (see 'Within seed variance' and 'replicate_variance_aggregate')."
    ),
    "trajectories": (
        "values are computed using every generated trajectory; in other words, a synthetic "
        "population is created by pooling the `n` replicates from each individual. CIs are the "
        "metric's own sampling variance over that pooled population. The same metrics are saved for each of the K per-seed populations, and the "
        "pooled tables carry <code>k_seeds</code>, <code>mean_var</code> and "
        "<code>between_var</code>."
    ),
    "observed": (
        "observed sequences only, so no within-individual variation exists. CIs reflect inference uncertainty from the sample.")
}


def _basis_item(key: str) -> str:
    """The replicate-basis bullet, or ``""`` for a group where the distinction does not apply.

    Raises ``KeyError`` on an unknown key, so renaming a group cannot quietly drop its marker.
    """
    value = REPLICATE_BASIS[key]
    if value is None:
        return ""
    return f"<li><strong>Replicate handling:</strong> {_BASIS_TEXT[value]}</li>"


def _with_basis(note: str, basis_key: str | None) -> str:
    """``note`` with the replicate-basis bullet appended to it.

    When the note is already a bullet list the basis joins it as one more item, rather than opening
    a second list a hair below the first — one list is what a reader sees anyway, and it keeps the
    spacing from depending on whether a group happens to have a note.
    """
    item = _basis_item(basis_key) if basis_key else ""
    if not item:
        return note
    if note.rstrip().endswith("</ul>"):
        note = note.rstrip()
        return f'{note[: -len("</ul>")]}{item}</ul>'
    return f'{note}<ul class="basis muted">{item}</ul>'


def _rel_link(path: Path) -> str:
    """An href to a result file, relative to ``report.html``.

    Every table and figure is written by :class:`~seqeval.arms._common.OutputWriter` into
    ``<results>/<arm>/``, while the report itself sits at ``<results>/report.html`` — so the link
    needs the arm directory in front of it. A bare filename resolves to ``<results>/<name>``, which
    does not exist, and the link silently 404s.
    """
    return f"{path.parent.name}/{path.name}"


#: Extracts ``(outcome, jump-off age)`` from a ``reliability_<outcome>_w<age>`` figure stem.
_RELIABILITY_RE = re.compile(r"^reliability_(.+)_w(\d+)$")

#: Day-valued columns that the forecasting tables display converted to years.
_DAY_COLUMNS = (
    "age", "age_start", "age_stop", "duration", "horizon", "timing_spread", "q10", "q90",
)


def _to_years_display(df: pd.DataFrame) -> pd.DataFrame:
    """Convert known day-valued columns to years, renaming each to ``<col> (y)`` for display."""
    df = df.copy()
    for col in _DAY_COLUMNS:
        if col in df.columns:
            df[col] = df[col].map(
                lambda d: round(days_to_years(int(d)), 1) if pd.notna(d) else d
            )
            df = df.rename(columns={col: f"{col} (y)"})
    return df


def _b64_img(path: Path) -> str:
    """Return an ``<img>``/inline-SVG tag for an image file (base64-embedded)."""
    data = path.read_bytes()
    if path.suffix == ".svg":
        return data.decode("utf-8")
    mime = "image/jpeg" if path.suffix in {".jpg", ".jpeg"} else "image/png"
    b64 = base64.b64encode(data).decode("ascii")
    return f'<img alt="{html.escape(path.stem)}" src="data:{mime};base64,{b64}">'


#: Figure-stem prefix -> the parquet it is drawn from, for figures whose name is not the table's.
#: Longest prefix wins, so a more specific entry can override a general one.
_FIGURE_SOURCES = (
    ("reliability_", "calibration"),
    ("timing_ridge_", "timing_error"),
    ("uncertainty_ccf_", "parity_distribution"),
    ("ccf_uncertainty", "parity_distribution"),
    ("ccf_overlay_", "aggregate_error"),
    # These three draw the pooled estimate and its interval, so the peek under each figure should
    # be the table it was drawn from rather than the per-seed error summary.
    ("ppr_overlay_", "ppr_pooled"),
    ("asfr_overlay_", "asfr_pooled"),
    ("km_overlay_", "km_pooled"),
    ("within_seed_variance", "within_seed_variance_distribution"),
    ("within_seed_variance_by_cohort", "within_seed_variance_distribution_by_cohort"),
    ("quantum_quantile_fan", "quantum_quantile_summary"),
    ("quantum_quantile_fan_by_cohort", "quantum_quantile_summary_by_cohort"),
)


def _bullets(*items: str, lead: str = "") -> str:
    """A caption written as bullets, rendered as one — ``lead`` above an unordered list."""
    lis = "".join(f"<li>{item}</li>" for item in items if item)
    head = f'<p class="muted">{lead}</p>' if lead else ""
    return f'{head}<ul class="muted">{lis}</ul>'


def _figure_source(fig: Path) -> Path | None:
    """The parquet a figure is drawn from: its own stem when that exists, else the mapped table."""
    same_stem = fig.with_suffix(".parquet")
    if same_stem.exists():
        return same_stem
    matches = [name for prefix, name in _FIGURE_SOURCES if fig.stem.startswith(prefix)]
    for name in sorted(matches, key=len, reverse=True):
        candidate = fig.parent / f"{name}.parquet"
        if candidate.exists():
            return candidate
    return None


def _figure_html(fig: Path, *, link_parquet: bool = True) -> str:
    """Embed a figure over a link to the parquet it is drawn from and a peek at its first rows."""
    caption = html.escape(fig.stem)
    peek = ""
    source = _figure_source(fig) if link_parquet else None
    if source is not None:
        caption += (
            f' · <a href="{html.escape(_rel_link(source))}">{html.escape(source.name)}</a>'
        )
        peek = _peek_html(source)
    return (
        f"<figure>{_b64_img(fig)}"
        f"<figcaption class='muted'>{caption}</figcaption>{peek}</figure>"
    )


def _peek_html(path: Path) -> str:
    """The first :data:`_PEEK_ROWS` rows of a parquet: the figure's own numbers, visible."""
    try:
        df = pd.read_parquet(path, engine="pyarrow")
    except (OSError, ValueError):
        return ""
    if df.empty:
        return ""
    table = df.head(_PEEK_ROWS).to_html(index=False, border=0, na_rep="")
    return (
        f"<details class='peek'><summary class='muted'>first {min(_PEEK_ROWS, len(df))} of "
        f"{len(df)} rows</summary>{table}</details>"
    )


def _table_html(
    path: Path, *, to_years: bool = False, note: str = "", basis_key: str | None = None
) -> str:
    """Render a parquet table as a capped HTML table with a link to the full file.

    ``note`` and the replicate-basis line sit between the caption and the table, so the reader
    meets the explanation before the numbers.
    """
    df = pd.read_parquet(path, engine="pyarrow")
    if to_years:
        df = _to_years_display(df)
    n = len(df)
    shown = df.head(_MAX_TABLE_ROWS)
    table = shown.to_html(index=False, border=0, na_rep="")
    caption = f'<h3>{html.escape(path.stem)} <span class="muted">'
    caption += f"({n} rows" + (f", showing {_MAX_TABLE_ROWS}" if n > _MAX_TABLE_ROWS else "")
    caption += (
        f' · <a href="{html.escape(_rel_link(path))}">{html.escape(path.name)}</a>)</span></h3>'
    )
    return caption + _with_basis(note, basis_key) + table


def _figure_group(
    label: str,
    figs: list[Path],
    *,
    basis_key: str | None = None,
    note: str = "",
    level: int = 4,
    anchor: str | None = None,
) -> str:
    """One plot group, rendered the same way everywhere: heading, explanation, basis, figures.

    ``note`` is pre-rendered HTML (typically :func:`_bullets`). Returns ``""`` for an empty group,
    so an artifact the run never produced simply drops out of the report.
    """
    if not figs:
        return ""
    ident = f' id="{html.escape(anchor)}"' if anchor else ""
    cells = "".join(_figure_html(f) for f in figs)
    return (
        f"<h{level}{ident}>{html.escape(label)}</h{level}>"
        f"{_with_basis(note, basis_key)}<div class='figrow'>{cells}</div>"
    )


def _grouped_figures(
    arm_dir: Path, groups: tuple[tuple[str, str, str, str], ...], *, level: int = 3
) -> str:
    """Render an arm's figures as ``(label, glob, basis_key, note)`` groups, in the given order.

    Whatever no group claims is emitted last under "Other figures", so a newly-added figure is
    never silently dropped from the report just because no pattern matches it yet.
    """
    all_figs = sorted(p for p in arm_dir.iterdir() if p.suffix in _EMBED_SUFFIXES)
    claimed: set[Path] = set()
    parts = []
    for label, pattern, basis_key, note in groups:
        figs = [f for f in all_figs if f.match(pattern)]
        claimed.update(figs)
        parts.append(_figure_group(label, figs, basis_key=basis_key, note=note, level=level))
    leftover = [f for f in all_figs if f not in claimed]
    if leftover:
        parts.append(_figure_group("Other figures", leftover, level=level))
    return "\n".join(p for p in parts if p)


#: Evaluability counts folded into the backtest metrics table, in display order.
_COVERAGE_COLUMNS = ("n_condition", "n_evaluable", "n_settled", "n_uncovered")


def _coverage_columns(arm_dir: Path) -> pd.DataFrame | None:
    """``[outcome, age_stop_years, *_COVERAGE_COLUMNS]`` from coverage.parquet, or ``None``.

    The shrinking denominator behind every backtest score: how many persons actually contribute
    (``n_evaluable``) versus how many were excluded because the answer was fixed at jump-off
    (``n_settled``) or the sequence ran out before the frame closed (``n_uncovered``). It rides on
    the score's own row rather than in a table of its own, so the two are read together.
    """
    path = arm_dir / "coverage.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path, engine="pyarrow")
    keys = ["outcome", "age_stop_years"]
    cols = [c for c in _COVERAGE_COLUMNS if c in df.columns]
    if df.empty or not set(keys) <= set(df.columns) or not cols:
        return None
    return df[[*keys, *cols]].drop_duplicates(subset=keys)


def _publishes_individuals(manifest: dict) -> bool:
    """Whether the run wrote per-person artifacts (the default; ``output.individual_level``)."""
    return bool(
        manifest.get("config_resolved", {}).get("output", {}).get("individual_level", True)
    )


def _min_cell(manifest: dict) -> int:
    """The run's minimum publishable cell size (``output.min_cell``)."""
    return int(
        manifest.get("config_resolved", {}).get("output", {}).get("min_cell", MIN_CELL)
    )


def _run_summary_section(manifest: dict, results_dir: Path) -> str:
    """Run summary: model, timestamp, data coverage, and backtest evaluability."""
    cov = manifest.get("coverage", {})
    rows = [
        ("Model", manifest.get("model", {}).get("name", "")),
        ("Run (UTC)", manifest.get("timestamp_utc", "")),
        ("Config hash", manifest.get("config_hash", "")[:16]),
        ("Persons", cov.get("n_persons", "")),
        ("Cohort range", cov.get("cohort_range", "")),
        ("Sex breakdown", cov.get("sex_breakdown", "")),
        ("Individual-level output", "written" if _publishes_individuals(manifest) else "withheld"),
        ("Minimum cell size", _min_cell(manifest) or "no suppression"),
    ]
    body = "".join(
        f"<tr><th>{html.escape(str(k))}</th><td>{html.escape(str(v))}</td></tr>" for k, v in rows
    )
    out = [f'<h2 id="summary">Run summary</h2><table class="meta">{body}</table>']

    windows = cov.get("windows", [])
    if windows:
        wr = ["<tr><th>age_start</th><th>age_stop</th><th>n_seeds</th><th>n_persons</th></tr>"]
        for w in windows:
            wr.append(
                f"<tr><td>{w['age_start']}</td><td>{w['age_stop']}</td>"
                f"<td>{w['n_seeds']}</td><td>{w['n_persons']}</td></tr>"
            )
        out.append(f"<h3>Windows × replicates</h3><table>{''.join(wr)}</table>")

    return "\n".join(out)


#: Observed-only figure groups, in display order: ``(label, glob, basis key, note)``.
_OBSERVED_GROUPS = (
    (
        "Kaplan-Meier Curves",
        "km_*.png",
        "observed.km",
        _bullets(
            "Time to each outcome in the observed registry sequences by cohort."
        ),
    ),
    (
        "Completed cohort fertility (CCF) by parity",
        "ccf_*.png",
        "observed.ccf",
        _bullets(
            "Estimated mean completed cohort fertility with 95% confidence intervals beside the observed distribution of completed "
            "parity (outcome variation)."
        ),
    ),
    (
        "Cohort ASFR",
        "asfr_*.png",
        "observed.asfr",
        _bullets("Observed age-specific fertility rates by cohort."),
    ),
)


def _observed_section(arm_dir: Path) -> str:
    """Observed sequences: figures computed from observed history alone, grouped by measure."""
    body = _grouped_figures(arm_dir, _OBSERVED_GROUPS)
    if not body:
        return ""
    return '<h2 id="observed">Observed Sequences</h2>\n' + body


def _backtest_metrics_table(arm_dir: Path) -> str:
    """The headline scores per outcome × jump-off age, with their intervals.

    The outcome (its condition encoded in the name) spans its jump-off rows; each row carries the
    raw-rate MSE and R², the rank-based AUC, and the calibration error the reliability diagrams
    below display — each with its interval in parentheses where one exists. ECE has none, so it
    shows as a bare number. The evaluability counts behind the score follow on the same row, so the
    denominator is never a separate lookup.
    """
    path = arm_dir / "scores.parquet"
    if not path.exists():
        return ""
    df = pd.read_parquet(path, engine="pyarrow")
    metric_cols = [
        ("mse", "MSE/Brier"),
        ("r2", "R²"),
        ("roc_auc", "AUC"),
        ("ece", "ECE"),
    ]
    keys = [m for m, _ in metric_cols]
    keep = df[df["metric"].isin(keys)].copy()
    if keep.empty:
        return ""
    has_ci = {"ci_lo", "ci_hi"} <= set(keep.columns)

    def disp(row) -> str:
        if pd.isna(row["value"]):
            return ""
        text = f"{row['value']:.3f}"
        if has_ci and pd.notna(row["ci_lo"]) and pd.notna(row["ci_hi"]):
            text += f" ({row['ci_lo']:.3f}, {row['ci_hi']:.3f})"
        return text

    keep["disp"] = keep.apply(disp, axis=1)
    wide = keep.pivot_table(
        index=["outcome", "age_stop_years"], columns="metric", values="disp", aggfunc="first"
    ).reset_index()

    cover = _coverage_columns(arm_dir)
    if cover is not None:
        wide = wide.merge(cover, on=["outcome", "age_stop_years"], how="left")
    count_cols = [c for c in _COVERAGE_COLUMNS if c in wide.columns]

    def cell(row, metric: str) -> str:
        val = getattr(row, metric, None)
        if isinstance(val, str):
            return val
        return "" if val is None or pd.isna(val) else f"{int(val):d}"

    rows = []
    for outcome, grp in wide.groupby("outcome", sort=True):
        grp = grp.sort_values("age_stop_years")
        span = len(grp)
        for i, r in enumerate(grp.itertuples(index=False)):
            head = (
                f'<td class="rowhdr" rowspan="{span}">{html.escape(str(outcome))}</td>'
                if i == 0
                else ""
            )
            cells = "".join(f"<td>{cell(r, m)}</td>" for m in [*keys, *count_cols])
            rows.append(f"<tr>{head}<td>{r.age_stop_years:g}</td>{cells}</tr>")
    header = "<tr><th>outcome (condition)</th><th>jump-off (y)</th>" + "".join(
        f"<th>{label}</th>" for _, label in metric_cols
    ) + "".join(f"<th>{html.escape(c)}</th>" for c in count_cols) + "</tr>"
    ci_note = (
        "CIs are analytically built up from individual-level values to account for replication "
        "variance: the sd of the per-person "
        "loss for MSE, the delta method for R², and DeLong's for AUC. Each already "
        "carries replicate variation, since every person's loss is computed from "
        "their own <code>p̂</code> (see 'Within-seed variance'). ECE has no CI, since we "
        "use quantile binning."
        if has_ci
        else ""
    )
    p_hat = (
        "<code>p̂</code> is defined for a particular outcome (e.g., first birth by age 35) and "
        "jump-off age (e.g., 20, 25, 30). It is the raw MLE estimate of the number of trajectory "
        "replicates for that individual that reach the outcome by the jump-off age, divided by the "
        "total number of replicates for that individual."
    )
    note = _bullets(
        p_hat,
        "MSE is the squared-error using <code>p̂</code> (from the replicate mean) versus the "
        "observed outcome. This is equal to the Brier score for binary outcomes.",
        "R² is the MSE rescaled by the outcome variance (1 = perfect, 0 = base rate).",
        "AUC is rank-based, computed from <code>p̂</code>. It checks whether "
        "P(p̂ | event) > P(p̂ | no event) for a given outcome.",
        "ECE is the expected calibration error, measuring the average deviation between "
        "predicted probabilities and actual outcomes among binned p̂ values. We use quantile "
        "binning for ECE and the reliability diagrams (see below).",
        ci_note,
        "<code>n_evaluable</code> persons are included in the analysis; "
        "<code>n_settled</code> already had the outcome in the observed prefix (before jump-off) and "
        "<code>n_uncovered</code> had a sequence that stopped before the outcome period finished.",
    )
    return (
        f"{p_hat}"
        '<h3 id="backtest-metrics">Backtest metrics</h3>'
        f"{_with_basis(note, 'comparison.metrics')}"
        f"<table>{header}{''.join(rows)}</table>"
    )


def _calibration_subsections(arm_dir: Path) -> str:
    """Reliability graphs grouped into a navigable subsection per outcome (condition)."""
    figs = sorted(arm_dir.glob("reliability_*.png"))
    if not figs:
        return ""
    groups: dict[str, list[Path]] = {}
    for f in figs:
        m = _RELIABILITY_RE.match(f.stem)
        if not m:
            continue  # skip any legacy/gridded reliability_<outcome>.png — only per-jump-off panels
        groups.setdefault(m.group(1), []).append(f)
    if not groups:
        return ""

    def jumpoff(p: Path) -> int:
        m = _RELIABILITY_RE.match(p.stem)
        return int(m.group(2)) if m else 0

    note = _bullets(
        "The calibration/reliability diagram bins individuals by their <code>p̂</code> for a "
        "particular outcome and jump-off age. We use a fixed number of quantile bins rather than " \
        "fixed-width ones. Bin counts are in <code>calibration.parquet</code>.",
        "The x-axis is the mean <code>p̂</code> for the bin, and the y-axis is the proportion of "
        "observed individuals for which that outcome was actually reached after the jump-off age. "
        "A diagonal line is perfect calibration.",
        "The histogram shows the distribution of <code>p̂</code> values from "
        "<code>p_hat_distribution.parquet</code>.",
    )
    parts = [
        '<h3 id="calibration">Calibration</h3>',
        _with_basis(note, "comparison.calibration"),
    ]
    links = " ".join(
        f'<a href="#cal-{html.escape(o)}">{html.escape(o)}</a>' for o in sorted(groups)
    )
    parts.append(f'<p class="subnav muted">Jump to: {links}</p>')
    for outcome in sorted(groups):
        parts.append(
            _figure_group(
                outcome,
                sorted(groups[outcome], key=jumpoff),
                anchor=f"cal-{outcome}",
            )
        )
    return "\n".join(parts)


#: Generated-vs-observed overlay figure families, in display order, each carrying the explanation
#: that belongs above it: ``(label, glob, basis key, note)``.
_OVERLAY_GROUPS = (
    (
        "Kaplan-Meier Curves",
        "km_overlay_*.png",
        "comparison.km",
        _bullets(
            "Comparison of the time to outcome in the observed registry sequences versus the "
            "generated sequences. The model begins generating from each jumpoff year, so the "
            "observed history is known up to t_2 (i.e., for jumpoff 35, the shape of the generated "
            "curve deviates from the observed based on births predicted after the age of 35).",
            "The generated curve is fit over every trajectory sequence — AKA all replicates "
            "pooled into a single synthetic population. The results are broken down into K separate" \
            " seed populations in <code>km_by_seed.parquet</code>, the full synthetic population is in"
            "<code>km_pooled.parquet</code>.",
            "The band is the traditional Greenwood interval of the pooled curve, on the "
            "complementary log-log scale, computed as though each trajectory were its own person.",
        ),
    ),
    (
        "Parity progression ratios (PPR)",
        "ppr_overlay_*.png",
        "comparison.ppr",
        _bullets(
            "PPR is the proportion of individuals at parity X who ultimately move to the next "
            "parity. The generated sequences are grouped by jump-off; for example, if an "
            "individual is at parity 1 before the jumpoff, the model can predict the transition into parity "
            "2 or not. Only individuals at the given parity are included (i.e., smaller sample "
            "for 5->6 transition).",
            "Each ratio is computed over all replicates pooled into one synthetic population "
            "(<code>ppr_pooled.parquet</code>); the K separate seed populations are in "
            "<code>ppr_by_seed.parquet</code>.",
            "The bars are the binomial variance <code>p(1-p)/n_units</code> of the pooled "
            "population, where <code>n_units</code> counts trajectories rather than people.",
        ),
    ),
    (
        "Cohort ASFR",
        "asfr_overlay_all_jumpoffs.png",
        "comparison.asfr",
        _bullets(
            "Generated age-specific fertility rates by cohort. The generated ASFRs begin at their "
            "specific jump-off point (to the left of that is observed history, to the right is "
            "entirely computed from model-generated sequences).",
            "Each rate pools every replicate's trajectories into one synthetic population "
            "(<code>asfr_pooled.parquet</code>); the K separate seed populations are in "
            "<code>asfr_by_seed.parquet</code>.",
            "The band is the Poisson variance <code>births/person_years²</code> of the pooled "
            "population, whose person-years are those of N×K trajectories rather than N people.",
        ),
    ),
    (
        "Analytic completed cohort fertility (CCF)",
        "ccf_overlay_all_jumpoffs.png",
        "comparison.ccf",
        _bullets(
            "CCF by cohort computed from generated sequences at different jump-off years. The model sees the observed sequence up to t_2, and then all generated births are considered alongside the number of births pre-jumpoff." \
            "The CCF CIs are "
            "<code>±z·&radic;total_var</code>, a sum of the within-individual inference variation "
            "and the between-people variation divided by the number of women. (see `replicate_variance_aggregate`)"
        ),
    ),
)


def _gen_vs_obs_section(arm_dir: Path) -> str:
    """Observed-vs-generated overlays: observed 'truth' under the across-seed band, if present."""
    blocks = [
        _figure_group(label, sorted(arm_dir.glob(pattern)), basis_key=basis_key, note=note)
        for label, pattern, basis_key, note in _OVERLAY_GROUPS
    ]
    blocks = [b for b in blocks if b]
    if not blocks:
        return ""
    return '<h3 id="gen-vs-obs">Generated vs observed sequences</h3>' + "".join(blocks)


def _uncertainty_section(arm_dir: Path) -> str:
    """Inference uncertainty beside outcome uncertainty, per jump-off, if present."""
    figs = sorted(arm_dir.glob("uncertainty_ccf_*.png"))
    if not figs:
        return ""
    cells = "".join(_figure_html(f) for f in figs)
    return (
        '<h3 id="uncertainty">CCF by parity: inference vs outcome uncertainty</h3>'
        + _with_basis(
            _bullets(
            "Estimated mean completed cohort fertility beside the distribution of completed parity from the generated sequences.",
            "Inference (left): The 95% CI is the "
            "uncertainty in the CCF estimate mean — it is "
            "computed using all sequences (pooling a theoretical population made up of n replicates of each individual).",
            "Outcome (right): The histograms are "
            "distributions of estimated completed parity across the cohort. "
            "Counts are in <code>parity_distribution.parquet</code>",
            "Like the mean, each individual replicate is counted in the outcome distribution (synthetic pooled population).",
            ),
            "comparison.uncertainty",
        )
        + f'<div class="figrow">{cells}</div>'
    )


def _timing_error_section(arm_dir: Path) -> str:
    """Timing-error ridges (how early or late, by predicted value), if present."""
    figs = sorted(arm_dir.glob("timing_ridge_*.png"))
    if not figs:
        return ""
    cells = "".join(_figure_html(f) for f in figs)
    return (
        '<h3 id="timing-error">Timing error</h3>'
        + _with_basis(
            _bullets(
            "For each outcome, the distribution of "
            "<code>observed − predicted</code> timing (in years). The bins are fixed-width "
            "intervals of predicted value, the same in every figure, so jump-offs can be read "
            "against each other.",
            "Mass to the right "
            "of the dashed zero line is an event that happened later than predicted. Mass to the "
            "left is an event that happened earlier than predicted.",
            "One observation is one generated trajectory, not one person: a woman's seeds each "
            "carry their own predicted time and each contribute their own error, with no median "
            "taken across her replicates first.",
            "Only trajectories where the model actually produced the outcome are counted. Each figure states how many trajectories did not show the outcome (for individuals where at least one did); the counts are on "
            "every row of the table as <code>n_excluded</code> of <code>n_trajectories</code>. ",
            "The counts are in <code>timing_error.parquet</code>, and per seed population in "
            "<code>timing_error_by_seed.parquet</code>.",
            ),
            "comparison.timing",
        )
        + f'<div class="figrow">{cells}</div>'
    )


def _comparison_section(arm_dir: Path) -> str:
    """Observed against generated: metrics table, overlays, calibration, and timing error."""
    parts = ['<h2 id="comparison">Observed and Generated Comparison</h2>']
    for block in (
        _backtest_metrics_table(arm_dir),
        _gen_vs_obs_section(arm_dir),
        _uncertainty_section(arm_dir),
        _calibration_subsections(arm_dir),
        _timing_error_section(arm_dir),
    ):
        if block:
            parts.append(block)
    return "\n".join(parts) if len(parts) > 1 else ""


def _sample_persons_html(
    path: Path,
    n_people: int = 5,
    *,
    to_years: bool = False,
    note: str = "",
    basis_key: str | None = None,
    n_seeds: int | None = None,
) -> str:
    """A per-person table down-sampled to a few deterministically-chosen individuals (no full dump).

    Used for tables with one-or-more rows per ``person_id`` (individual violations, individual seed
    stability) where an aggregate or a 50-row head is unhelpful — five sampled persons show the
    shape of a record. The full table stays in the linked parquet.

    ``n_seeds`` samples the replicate axis too. A table with one row per (person, seed, event) runs
    to K rows per person before it shows a second person, so a 50-row window would otherwise be a
    couple of people seen through every one of their seeds; sampling seeds as well as people trades
    replicate depth for a wider view of the population, which is what these tables are shown for.
    """
    df = pd.read_parquet(path, engine="pyarrow")
    if df.empty or "person_id" not in df.columns:
        return _table_html(path, to_years=to_years, note=note, basis_key=basis_key)
    people = df["person_id"].drop_duplicates()
    k = min(n_people, len(people))
    chosen = set(people.sample(n=k, random_state=0))
    sub = df[df["person_id"].isin(chosen)]

    seed_note = ""
    if n_seeds is not None and "seed" in sub.columns:
        seeds = sub["seed"].dropna().drop_duplicates()
        k_seeds = min(n_seeds, len(seeds))
        if k_seeds:
            keep = set(seeds.sample(n=k_seeds, random_state=0))
            # A null seed is an observed row, not a replicate — these tables stack the observed
            # baseline under the generated rows, and dropping it would leave nothing to compare to.
            sub = sub[sub["seed"].isin(keep) | sub["seed"].isna()]
            seed_note = f" × {k_seeds} of {len(seeds)} seeds"

    sort_cols = ["person_id"] + (["seed"] if "seed" in sub.columns else [])
    sub = sub.sort_values(sort_cols, kind="stable")
    if to_years:
        sub = _to_years_display(sub)
    n = len(sub)
    table = sub.head(_MAX_TABLE_ROWS).to_html(index=False, border=0, na_rep="")
    caption = (
        f'<h3>{html.escape(path.stem)} <span class="muted">'
        f"({k} sampled persons{seed_note}, {n} rows"
        + (f", showing {_MAX_TABLE_ROWS}" if n > _MAX_TABLE_ROWS else "")
        + f' · <a href="{html.escape(_rel_link(path))}">{html.escape(path.name)}</a>)</span></h3>'
    )
    return caption + _with_basis(note, basis_key) + table


#: Generated-only figure groups, in display order: ``(label, glob, basis key, note)``.
_GENERATED_GROUPS = (
    (
        "Lexis surface",
        "lexis_*.png",
        "generated.lexis",
        _bullets(
            "Outcome heat map over cohort and age. The blue line marks where the observed data "
            "ends and the rate is computed solely from model forecasts.",
            "Each forecast cell pools every replicate's trajectories into one synthetic population "
            "(<code>lexis_cohort_pooled.parquet</code>). The K separate seed surfaces are in "
            "<code>lexis_cohort_forecast.parquet</code>. In the parquet, each pooled cell carries its own "
            "<code>ci_lo</code>/<code>ci_hi</code> (uncertainty not shown in the surface).",
        ),
    ),
    (
        "Within-seed variance",
        "within_seed_variance*.png",
        "generated.dispersion",
        _bullets(
            "How much variance there is in quantum fertility across one individual's trajectories, equal to the " \
            "inference uncertainty that the <code>within_var</code> term below is built from."
        ),
    ),
    (
        "Within-seed spread of completed births",
        "quantum_quantile_fan*.png",
        "generated.dispersion",
        _bullets(
            "The distribution behind the within_seed_variance. We compute summaries of individual's completed quantum fertility across "
            "replicates — min, quartiles, max. The plot shows "
            "the average of each of those quantities over all individuals: the line is the typical "
            "person's median outcome, the dark band their interquartile spread, the light band "
            "their full replicate range.",
            "It is the mean of the per-person quantiles, not the population's quantiles.",
        ),
    ),
)

#: Per-person tables (down-sampled to a few individuals), in display order:
#: ``(stem, basis key, note, n_seeds)``. The note is rendered above the table. ``n_seeds`` further
#: samples the replicate axis for tables that carry one row per (person, seed, violation) and would
#: otherwise show a handful of people entirely through one seed's worth of rows; ``None`` keeps
#: every seed of the sampled people.
_GENERATED_PERSON_TABLES = (
    ("replicate_variance_individual", "generated.replicate_variance_individual", "", None),
    (
        "replicate_occurrence",
        "generated.replicate_occurrence",
        '<p class="muted">Per-person replicate summary of the named '
        "<code>outcome</code>: whether it occurs inside <code>horizon</code>, and how "
        "much the seeds disagree about when (<code>timing_spread</code>, the q90–q10 "
        "width). <code>p_hat</code> is the raw replicate frequency "
        "<code>n_occurred/n</code>.</p>",
        None,
    ),
    ("violations", "generated.violations", "", 3),
)

#: Aggregate tables, in display order: ``(stem, basis key, note)``.
_GENERATED_AGGREGATE_TABLES = (
    (
        "replicate_variance_aggregate",
        "generated.replicate_variance_aggregate",
        _bullets(
            "The variance of each CCF, split by within-individual "
            "and between-person.",
            "<code>within_var</code> is within-individual replicate variance — "
            "rerunning inference on the same person to get a different trajectory;",
            "<code>between_var</code> is how much individuals differ (due to sampling), divided by "
            "<code>n</code>.",
            "Both terms are inference error in the mean rather than the "
            "spread of individual outcomes (see the backtesting "
            "outcome-uncertainty figure).",
            "The terms sum to "
            "<code>total_var</code>, which is what <code>se_total</code>, "
            "<code>ci_total</code> and the CCF figure band all report.",
            "<code>forecast_share</code> is the fraction of each estimate contributed "
            "by post-jump-off generated events: 0 rests entirely on observed history, "
            "1 entirely on model output.",
        ),
    ),
    (
        "illegal_moves",
        "generated.violation_rates",
        _bullets(
            "<code>rate_per_event</code> is the share of the events that "
            "break a given rule — <code>n_events</code> counts only that event "
            "kind.",
            "e.g.: the rate reads directly as \"x% of generated births came too soon "
            "after the last one\".",
            "Violations are counted per event, so one sequence "
            "offending three times contributes three.",
        ),
    ),
)


def _generated_section(arm_dir: Path) -> str:
    """Generated sequences: figures, per-person tables sampled to 5 individuals, and aggregates."""
    parts = ['<h2 id="generated">Generated Sequences</h2>']
    figures = _grouped_figures(arm_dir, _GENERATED_GROUPS)
    if figures:
        parts.append(figures)
    # per-person tables → sample 5 individuals; aggregate tables → full. Ages/times shown in years.
    for name, basis_key, note, n_seeds in _GENERATED_PERSON_TABLES:
        p = arm_dir / f"{name}.parquet"
        if p.exists():
            parts.append(
                _sample_persons_html(
                    p, to_years=True, note=note, basis_key=basis_key, n_seeds=n_seeds
                )
            )
    for name, basis_key, note in _GENERATED_AGGREGATE_TABLES:
        p = arm_dir / f"{name}.parquet"
        if p.exists():
            parts.append(_table_html(p, to_years=True, note=note, basis_key=basis_key))
    return "\n".join(parts) if len(parts) > 1 else ""


#: Report sections in display order, grouped by where the analysis comes from rather than by which
#: arm produced it: ``(anchor, title, arm directory, renderer)``. The arm directory names on disk
#: are unchanged; only the presentation is regrouped.
SECTIONS = (
    ("observed", "Observed Sequences", "descriptives", _observed_section),
    ("generated", "Generated Sequences", "forecasting", _generated_section),
    ("comparison", "Observed and Generated Comparison", "backtesting", _comparison_section),
)


def build_report(results_dir: str | Path) -> Path:
    """Build ``results/report.html`` from an existing results dir; return its path.

    Reads ``manifest.json`` for the run summary, then embeds the figures and tables found under each
    present arm directory. Missing arm dirs are skipped gracefully. The run's warnings are not
    rendered here — ``manifest.json`` carries the verbatim list, and the CLI logs them as they are
    raised, which is where they are actionable.
    """
    results_dir = Path(results_dir)
    manifest = read_manifest(results_dir) or {}

    sections = [_run_summary_section(manifest, results_dir)]
    present = []
    for anchor, _title, arm, renderer in SECTIONS:
        arm_dir = results_dir / arm
        if arm_dir.is_dir():
            section = renderer(arm_dir)
            if section:
                sections.append(section)
                present.append(anchor)
    model = manifest.get("model", {}).get("name", "")
    nav_targets = ["summary", *present]
    nav_labels = {
        "summary": "Run summary",
        **{anchor: title for anchor, title, _, _ in SECTIONS},
    }
    nav = " ".join(
        f'<a href="#{t}">{html.escape(nav_labels[t])}</a>' for t in nav_targets
    )
    doc = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>seqeval report — {html.escape(str(model))}</title>"
        f"<style>{_CSS}</style></head><body>"
        f"<h1>seqeval report — {html.escape(str(model))}</h1>"
        f"<nav>{nav}</nav>" + "\n".join(sections) + "</body></html>"
    )
    path = results_dir / REPORT_NAME
    path.write_text(doc)
    logger.info("report: wrote %s (%d section(s))", path, len(present))
    return path
