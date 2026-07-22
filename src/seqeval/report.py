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

    Windows are emitted in **years** (user-facing) with an ``under_min_replicates`` flag per window
    (00 section 3b: too few replicates make the probability grid coarser than calibration bins).
    """
    summary = bundle.population_summary()
    min_reps = cfg.replicates.min_replicates
    windows = []
    for row in bundle.available_windows().itertuples(index=False):
        windows.append(
            {
                "age_start": round(days_to_years(int(row.age_start)), 2),
                "age_stop": round(days_to_years(int(row.age_stop)), 2),
                "n_seeds": int(row.n_seeds),
                "n_persons": int(row.n_persons),
                "under_min_replicates": bool(int(row.n_seeds) < min_reps),
            }
        )
    return {
        "n_persons": summary["n_persons"],
        "sex_breakdown": summary["sex_breakdown"],
        "cohort_range": summary["cohort_range"],
        "min_replicates": min_reps,
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
figure { margin: 1rem 0; }
img { max-width: 100%; height: auto; border: 1px solid #eee; }
table { border-collapse: collapse; margin: .5rem 0 1rem; font-size: .85rem; }
th, td { border: 1px solid #ddd; padding: .3rem .5rem; text-align: right; }
th { background: #f5f5f5; }
.meta td, .meta th { text-align: left; }
.warn { background: #fff8e1; border-left: 4px solid #f0ad4e; padding: .5rem 1rem; }
.flag { color: #b34700; font-weight: 600; }
.muted { color: #777; font-size: .85rem; }
code { background: #f2f2f2; padding: .1rem .3rem; border-radius: 3px; }
"""


def _b64_img(path: Path) -> str:
    """Return an ``<img>``/inline-SVG tag for an image file (base64-embedded)."""
    data = path.read_bytes()
    if path.suffix == ".svg":
        return data.decode("utf-8")
    mime = "image/jpeg" if path.suffix in {".jpg", ".jpeg"} else "image/png"
    b64 = base64.b64encode(data).decode("ascii")
    return f'<img alt="{html.escape(path.stem)}" src="data:{mime};base64,{b64}">'


def _table_html(path: Path) -> str:
    """Render a parquet table as a capped HTML table with a link to the full file."""
    df = pd.read_parquet(path, engine="pyarrow")
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
    no score, reliability, or convergence figure. Returns ``""`` when the arm did not run.
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
        wr = [
            "<tr><th>age_start</th><th>age_stop</th><th>n_seeds</th><th>n_persons</th>"
            "<th>replicates</th></tr>"
        ]
        for w in windows:
            flag = '<span class="flag">below min</span>' if w.get("under_min_replicates") else "ok"
            wr.append(
                f"<tr><td>{w['age_start']}</td><td>{w['age_stop']}</td>"
                f"<td>{w['n_seeds']}</td><td>{w['n_persons']}</td><td>{flag}</td></tr>"
            )
        note = (
            f'<p class="muted">Windows below {cov.get("min_replicates")} replicates give a '
            "probability grid coarser than 1/n; calibration bins finer than that are not "
            "meaningful.</p>"
        )
        out.append(f"<h3>Windows × replicates</h3><table>{''.join(wr)}</table>{note}")

    out.append(_coverage_summary(results_dir))
    return "\n".join(out)


def _arm_section(arm: str, arm_dir: Path) -> str:
    """One arm section: every figure (embedded) followed by every table (capped)."""
    figures = sorted(p for p in arm_dir.iterdir() if p.suffix in _EMBED_SUFFIXES)
    tables = sorted(arm_dir.glob("*.parquet"))
    if not figures and not tables:
        return ""
    parts = [f'<h2 id="{html.escape(arm)}">{html.escape(arm.capitalize())}</h2>']
    for fig in figures:
        parts.append(
            f"<figure>{_b64_img(fig)}<figcaption class='muted'>{html.escape(fig.stem)}"
            "</figcaption></figure>"
        )
    for tbl in tables:
        parts.append(_table_html(tbl))
    return "\n".join(parts)


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

    sections = [_run_summary_section(manifest, results_dir)]
    present_arms = []
    for arm in ARM_ORDER:
        arm_dir = results_dir / arm
        if arm_dir.is_dir():
            section = _arm_section(arm, arm_dir)
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
