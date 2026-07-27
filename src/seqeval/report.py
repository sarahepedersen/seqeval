"""06 — the run manifest and the self-contained HTML report.

Two concerns live here, both pure functions over already-written results:

- **Manifest** (:func:`build_manifest` / :func:`write_manifest`): ``results/manifest.json`` records
  everything needed to reproduce and audit a run — seqeval version, model, the SHA-256 of every
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

from seqeval import __version__
from seqeval.units import days_to_years

logger = logging.getLogger("seqeval")

MANIFEST_NAME = "manifest.json"
REPORT_NAME = "report.html"

#: Arms in execution / display order (00 section 5: descriptives -> backtesting -> forecasting).
ARM_ORDER = ["descriptives", "backtesting", "forecasting"]

#: Image suffixes embedded inline in the report; anything else is linked, not embedded.
_EMBED_SUFFIXES = {".png", ".jpg", ".jpeg", ".svg"}

#: Rows shown per table in the HTML report before truncation (full data stays in the parquet).
_MAX_TABLE_ROWS = 50


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
            "duration_s": res["duration_s"],
        }
    return {
        "seqeval_version": __version__,
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
.subnav { margin: .2rem 0 .8rem; }
.subnav a { margin-right: .8rem; }
.warn { background: #fff8e1; border-left: 4px solid #f0ad4e; padding: .5rem 1rem; }
.flag { color: #b34700; font-weight: 600; }
.muted { color: #777; font-size: .85rem; }
code { background: #f2f2f2; padding: .1rem .3rem; border-radius: 3px; }
"""

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


def _figure_html(fig: Path, *, link_parquet: bool = False) -> str:
    """Embed a figure; optionally link the same-stem parquet under its caption."""
    caption = html.escape(fig.stem)
    if link_parquet:
        parquet = fig.with_suffix(".parquet")
        if parquet.exists():
            caption += f' · <a href="{html.escape(parquet.name)}">{html.escape(parquet.name)}</a>'
    return f"<figure>{_b64_img(fig)}<figcaption class='muted'>{caption}</figcaption></figure>"


def _table_html(path: Path, *, to_years: bool = False) -> str:
    """Render a parquet table as a capped HTML table with a link to the full file."""
    df = pd.read_parquet(path, engine="pyarrow")
    if to_years:
        df = _to_years_display(df)
    n = len(df)
    shown = df.head(_MAX_TABLE_ROWS)
    table = shown.to_html(index=False, border=0, na_rep="")
    caption = f'<h3>{html.escape(path.stem)} <span class="muted">'
    caption += f"({n} rows" + (f", showing {_MAX_TABLE_ROWS}" if n > _MAX_TABLE_ROWS else "")
    caption += f' · <a href="{html.escape(path.name)}">{html.escape(path.name)}</a>)</span></h3>'
    return caption + table


def _coverage_summary(results_dir: Path) -> str:
    """Compact backtest-evaluability table for the run summary (read from coverage.parquet).

    Surfaces the shrinking denominator behind every backtest score: per outcome × window ×
    condition, how many persons actually contribute a score (``n_evaluable``) versus how many were
    excluded because the answer was fixed at jump-off (``n_settled``) or the sequence ran out before
    the frame closed (``n_uncovered``). Cells with zero evaluable persons are flagged — they produce
    no score or reliability figure. Returns ``""`` when the arm did not run.
    """
    path = results_dir / "backtesting" / "coverage.parquet"
    if not path.exists():
        return ""
    df = pd.read_parquet(path, engine="pyarrow")
    if df.empty:
        return ""

    if {"age_start_years", "age_stop_years"} <= set(df.columns):
        window = df["age_start_years"].astype(str) + "–" + df["age_stop_years"].astype(str)
    else:  # fall back to converting the day-valued columns
        to_y = lambda d: round(days_to_years(int(d)), 1)  # noqa: E731
        window = df["age_start"].map(to_y).astype(str) + "–" + df["age_stop"].map(to_y).astype(str)

    view = pd.DataFrame(
        {
            "outcome": df.get("outcome", ""),
            "window (y)": window,
            "given": df.get("condition", "-"),
            "n_condition": df.get("n_condition"),
            "n_evaluable": df.get("n_evaluable"),
            "n_settled": df.get("n_settled"),
            "n_uncovered": df.get("n_uncovered"),
            "seeds (med)": df.get("n_seed_median"),
        }
    ).sort_values(["outcome", "window (y)", "given"], kind="stable")

    n_total = len(view)
    n_empty = int((view["n_evaluable"] == 0).sum())
    shown = view.head(_MAX_TABLE_ROWS)

    header = "".join(f"<th>{html.escape(c)}</th>" for c in shown.columns)
    body_rows = []
    for row in shown.itertuples(index=False):
        cells = []
        for col, val in zip(shown.columns, row, strict=True):
            empty = col == "n_evaluable" and (pd.isna(val) or val == 0)
            klass = ' class="flag"' if empty else ""
            cells.append(f"<td{klass}>{html.escape(str(val))}</td>")
        body_rows.append(f"<tr>{''.join(cells)}</tr>")

    note = (
        '<p class="muted"><code>n_evaluable</code> persons actually contribute a score; '
        "<code>n_settled</code> were already determined in the observed prefix and "
        "<code>n_uncovered</code> ran out of observation before the frame closed. "
    )
    if n_empty:
        note += (
            f'<span class="flag">{n_empty} cell(s)</span> have no evaluable persons — no score is '
            "produced there. "
        )
    if n_total > _MAX_TABLE_ROWS:
        note += (
            f"Showing {_MAX_TABLE_ROWS} of {n_total} cells; full table in the Backtesting section. "
        )
    note += "</p>"
    return (
        "<h3>Backtest coverage (evaluability)</h3>"
        f"<table><tr>{header}</tr>{''.join(body_rows)}</table>{note}"
    )


def _run_summary_section(manifest: dict, results_dir: Path) -> str:
    """Run summary: model, version, timestamp, data coverage, and backtest evaluability."""
    cov = manifest.get("coverage", {})
    rows = [
        ("Model", manifest.get("model", {}).get("name", "")),
        ("seqeval version", manifest.get("seqeval_version", "")),
        ("Run (UTC)", manifest.get("timestamp_utc", "")),
        ("Config hash", manifest.get("config_hash", "")[:16]),
        ("Persons", cov.get("n_persons", "")),
        ("Cohort range", cov.get("cohort_range", "")),
        ("Sex breakdown", cov.get("sex_breakdown", "")),
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


def _descriptives_section(arm_dir: Path) -> str:
    """Descriptives: figures only, each with a link to its backing parquet (no tables)."""
    figures = sorted(p for p in arm_dir.iterdir() if p.suffix in _EMBED_SUFFIXES)
    if not figures:
        return ""
    parts = ['<h2 id="descriptives">Descriptives</h2>']
    parts.extend(_figure_html(fig, link_parquet=True) for fig in figures)
    return "\n".join(parts)


def _backtest_metrics_table(path: Path, *, bootstrap_n: int | None = None) -> str:
    """Brier + AUC per outcome × jump-off age — replaces the metric-vs-jump-off line graphs.

    The outcome (its condition encoded in the name) spans its jump-off rows; each row carries the
    finite-seed-corrected Brier score, the raw-rate MSE and R², and the rank-based AUC — each with
    its 95% seed-bootstrap CI in parentheses where available.
    """
    if not path.exists():
        return ""
    df = pd.read_parquet(path, engine="pyarrow")
    metric_cols = [("brier_corrected", "Brier"), ("mse", "MSE"), ("r2", "R²"), ("roc_auc", "AUC")]
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

    def cell(row, metric: str) -> str:
        val = getattr(row, metric, None)
        return val if isinstance(val, str) else ""

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
            cells = "".join(f"<td>{cell(r, m)}</td>" for m in keys)
            rows.append(f"<tr>{head}<td>{r.age_stop_years:g}</td>{cells}</tr>")
    header = "<tr><th>outcome (condition)</th><th>jump-off (y)</th>" + "".join(
        f"<th>{label}</th>" for _, label in metric_cols
    ) + "</tr>"
    ci_note = ""
    if has_ci:
        reps = (
            f" from {bootstrap_n} seed-bootstrap resamples" if bootstrap_n else " (seed-bootstrap)"
        )
        ci_note = f" Parentheses are 95% CIs{reps}."
    note = (
        '<p class="muted">Brier is the finite-seed-corrected Brier score; MSE is the same '
        "squared error of <code>p̂ = k/n</code> against the observed outcome without that "
        "correction; R² rescales MSE by the outcome variance (1 = perfect, 0 = base "
        f"rate); AUC is rank-based (tie-corrected).{ci_note} One row per outcome × jump-off.</p>"
    )
    return f"<h3>Backtest metrics</h3><table>{header}{''.join(rows)}</table>{note}"


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

    parts = ['<h3 id="calibration">Calibration</h3>']
    links = " ".join(
        f'<a href="#cal-{html.escape(o)}">{html.escape(o)}</a>' for o in sorted(groups)
    )
    parts.append(f'<p class="subnav muted">Jump to: {links}</p>')
    for outcome in sorted(groups):
        parts.append(f'<h4 id="cal-{html.escape(outcome)}">{html.escape(outcome)}</h4>')
        cells = "".join(_figure_html(f) for f in sorted(groups[outcome], key=jumpoff))
        parts.append(f'<div class="figrow">{cells}</div>')
    return "\n".join(parts)


#: Generated-vs-observed overlay figure families rendered in the backtesting section, in order.
_OVERLAY_GROUPS = (
    ("Survival (KM)", "km_overlay_*.png"),
    ("Completed cohort fertility (CCF)", "ccf_overlay_*.png"),
)


def _gen_vs_obs_section(arm_dir: Path) -> str:
    """Observed-vs-generated overlays: observed 'truth' under the across-seed band, if present."""
    blocks = []
    for label, pattern in _OVERLAY_GROUPS:
        figs = sorted(arm_dir.glob(pattern))
        if figs:
            cells = "".join(_figure_html(f) for f in figs)
            blocks.append(f"<h4>{html.escape(label)}</h4><div class='figrow'>{cells}</div>")
    if not blocks:
        return ""
    return (
        '<h3 id="gen-vs-obs">Generated vs observed</h3>'
        '<p class="muted">Observed "truth" (black) under the generated across-seed mean and its '
        "replicate-uncertainty band (orange), per jump-off. The band is the Monte-Carlo error of "
        "the plotted mean (<code>±z·sd/&radic;K</code> across seeds), the same quantity reported "
        "as <code>se</code> in the forecasting arm's aggregate CCF table.</p>" + "".join(blocks)
    )


def _timing_calibration_section(arm_dir: Path) -> str:
    """Waiting-time scatters (predicted vs observed duration), if present."""
    figs = sorted(arm_dir.glob("timing_calibration_*.png"))
    if not figs:
        return ""
    cells = "".join(_figure_html(f) for f in figs)
    return (
        '<h3 id="timing-calibration">Waiting time scatter</h3>'
        '<p class="muted">Predicted vs observed timing per person, for timed outcomes. Gray '
        "points are individuals; the red trend is the median observed time within equal-count bins "
        "of predicted time (IQR ribbon). Above the dashed <code>y = x</code> line the model "
        "predicts events too early; below it, too late.</p>"
        f'<div class="figrow">{cells}</div>'
    )


def _backtesting_section(arm_dir: Path, *, bootstrap_n: int | None = None) -> str:
    """Backtesting: metrics table, coverage, gen-vs-obs overlays, and calibration subsections."""
    parts = ['<h2 id="backtesting">Backtesting</h2>']
    for block in (
        _backtest_metrics_table(arm_dir / "scores.parquet", bootstrap_n=bootstrap_n),
        _coverage_summary(arm_dir.parent),
        _gen_vs_obs_section(arm_dir),
        _calibration_subsections(arm_dir),
        _timing_calibration_section(arm_dir),
    ):
        if block:
            parts.append(block)
    return "\n".join(parts) if len(parts) > 1 else ""


def _sample_persons_html(path: Path, n_people: int = 5, *, to_years: bool = False) -> str:
    """A per-person table down-sampled to a few deterministically-chosen individuals (no full dump).

    Used for tables with one-or-more rows per ``person_id`` (individual violations, individual seed
    stability) where an aggregate or a 50-row head is unhelpful — five sampled persons show the
    shape of a record. The full table stays in the linked parquet.
    """
    df = pd.read_parquet(path, engine="pyarrow")
    if df.empty or "person_id" not in df.columns:
        return _table_html(path, to_years=to_years)
    people = df["person_id"].drop_duplicates()
    k = min(n_people, len(people))
    chosen = set(people.sample(n=k, random_state=0))
    sub = df[df["person_id"].isin(chosen)].sort_values("person_id", kind="stable")
    if to_years:
        sub = _to_years_display(sub)
    n = len(sub)
    table = sub.head(_MAX_TABLE_ROWS).to_html(index=False, border=0, na_rep="")
    caption = (
        f'<h3>{html.escape(path.stem)} <span class="muted">({k} sampled persons, {n} rows'
        + (f", showing {_MAX_TABLE_ROWS}" if n > _MAX_TABLE_ROWS else "")
        + f' · <a href="{html.escape(path.name)}">{html.escape(path.name)}</a>)</span></h3>'
    )
    return caption + table


def _forecasting_section(arm_dir: Path) -> str:
    """Forecasting: figures, per-person tables sampled to 5 individuals, and aggregate tables."""
    figures = sorted(p for p in arm_dir.iterdir() if p.suffix in _EMBED_SUFFIXES)
    parts = ['<h2 id="forecasting">Forecasting</h2>']
    parts.extend(_figure_html(fig) for fig in figures)
    # per-person tables → sample 5 individuals; aggregate tables → full. Ages/times shown in years.
    for name in ("replicate_variance_individual", "replicate_occurrence", "violations"):
        p = arm_dir / f"{name}.parquet"
        if p.exists():
            parts.append(_sample_persons_html(p, to_years=True))
            if name == "replicate_occurrence":
                parts.append(
                    '<p class="muted">Per-person replicate summary of the named '
                    "<code>outcome</code>: whether it occurs inside <code>horizon</code>, and how "
                    "much the seeds disagree about when (<code>timing_spread</code>, the q90–q10 "
                    "width). <code>p_hat</code> is the raw replicate frequency "
                    "<code>n_occurred/n</code>.</p>"
                )
    for name in ("replicate_variance_aggregate", "violation_rates"):
        p = arm_dir / f"{name}.parquet"
        if p.exists():
            parts.append(_table_html(p, to_years=True))
            if name == "replicate_variance_aggregate":
                parts.append(
                    '<p class="muted">Aggregate CIs come from the replicate-variance '
                    "decomposition (between-person plus within-person seed variance).</p>"
                )
    return "\n".join(parts) if len(parts) > 1 else ""


def _warnings_section(manifest: dict) -> str:
    """Verbatim warning list from the manifest."""
    warnings = manifest.get("warnings", [])
    if not warnings:
        body = '<p class="muted">No warnings emitted.</p>'
    else:
        items = "".join(f"<li>{html.escape(str(w))}</li>" for w in warnings)
        body = f'<div class="warn"><ul>{items}</ul></div>'
    return f'<h2 id="warnings">Warnings</h2>{body}'


def build_report(results_dir: str | Path) -> Path:
    """Build ``results/report.html`` from an existing results dir; return its path.

    Reads ``manifest.json`` for the run summary and warnings, then embeds the figures and tables
    found under each present arm directory. Missing arm dirs are skipped gracefully.
    """
    results_dir = Path(results_dir)
    manifest = read_manifest(results_dir) or {}

    # Bootstrap-resample count backs the backtest-metric CIs.
    boot_n = (
        manifest.get("config_resolved", {})
        .get("replicates", {})
        .get("bootstrap", {})
        .get("n")
    )
    arm_renderers = {
        "descriptives": _descriptives_section,
        "backtesting": lambda d: _backtesting_section(d, bootstrap_n=boot_n),
        "forecasting": lambda d: _forecasting_section(d),
    }
    sections = [_run_summary_section(manifest, results_dir)]
    present_arms = []
    for arm in ARM_ORDER:
        arm_dir = results_dir / arm
        if arm_dir.is_dir():
            section = arm_renderers[arm](arm_dir)
            if section:
                sections.append(section)
                present_arms.append(arm)
    sections.append(_warnings_section(manifest))

    model = manifest.get("model", {}).get("name", "")
    nav_targets = ["summary", *present_arms, "warnings"]
    nav = " ".join(f'<a href="#{t}">{t}</a>' for t in nav_targets)
    doc = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>seqeval report — {html.escape(str(model))}</title>"
        f"<style>{_CSS}</style></head><body>"
        f"<h1>seqeval report — {html.escape(str(model))}</h1>"
        f"<nav>{nav}</nav>" + "\n".join(sections) + "</body></html>"
    )
    path = results_dir / REPORT_NAME
    path.write_text(doc)
    logger.info("report: wrote %s (%d arm section(s))", path, len(present_arms))
    return path
