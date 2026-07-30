"""Illegal-move rules engine: flag demographically impossible/implausible patterns (05).

Rules are **data, not code** (:class:`~seqeval.core.specs.Rule`, resolved from year-valued config by
``config.resolve_rules``): adding a rule never means adding code. Each :func:`check_rules` pass is
key-agnostic and runs the same on generated *and* observed data — violations in observed data
indicate data problems or mis-specified rules rather than model problems, which contextualizes the
model's rates (isolating model learning from data artifacts).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from seqeval.core.specs import Rule

__all__ = ["check_rules", "violation_rates"]

_VIOLATION_COLS = ["age", "event", "rule", "severity"]


def check_rules(df: pd.DataFrame, keys: list[str], rules: list[Rule]) -> pd.DataFrame:
    """Row-level rule violations: ``[*keys, age, event, rule, severity]`` (row per flagged event).

    Each field set on a :class:`Rule` is interpreted independently (all in integer days):
    ``min_age``/``max_age`` bound the event's age; ``min_spacing`` flags consecutive occurrences
    closer than the gap; ``not_after`` flags the subject occurring strictly after the anchor
    occurrence; ``not_before`` flags it occurring strictly *before* the anchor occurrence (or with
    that occurrence absent entirely); ``max_count`` flags occurrences beyond the cap.

    ``not_after`` and ``not_before`` are not mirror images: ``not_after`` only constrains groups
    where the anchor exists, while ``not_before`` treats a missing anchor as a violation (a divorce
    is illegal both before the first marriage and with no marriage at all).

    ``Rule.occurrence`` narrows the subject to one ordinal occurrence, and
    ``not_*_occurrence`` picks which occurrence of the anchor to measure against — which is how an
    outcome-keyed rule ("the second birth may not precede the first marriage") differs from a
    token-keyed one ("no birth may precede any marriage").
    """
    _check_keys(df, keys)
    parts = [_check_one(df, keys, rule) for rule in rules]
    parts = [p for p in parts if len(p)]
    if not parts:
        return pd.DataFrame(columns=[*keys, *_VIOLATION_COLS])
    return pd.concat(parts, ignore_index=True).sort_values([*keys, "age"]).reset_index(drop=True)


def _occurrences(
    df: pd.DataFrame, keys: list[str], token, occurrence: int | None
) -> pd.DataFrame:
    """Rows of ``token``, narrowed to its ``occurrence``-th per group when one is named.

    ``occurrence=None`` returns every occurrence — the token-keyed reading, where a rule constrains
    the whole stream. Ordering is by age within the group, so occurrence 2 is the second-earliest.
    """
    ev = df[df["event"] == token]
    if occurrence is None or not len(ev):
        return ev
    ordered = ev.sort_values([*keys, "age"], kind="stable")
    order = ordered.groupby(keys, observed=True).cumcount() + 1
    return ordered[order == occurrence]


def _anchor_age(df: pd.DataFrame, keys: list[str], token, occurrence: int) -> pd.Series:
    """Age of the ``occurrence``-th ``token`` per group; groups never reaching it are absent."""
    return _occurrences(df, keys, token, occurrence).groupby(keys, observed=True)["age"].min()


def _check_one(df: pd.DataFrame, keys: list[str], rule: Rule) -> pd.DataFrame:
    """Rows of ``df`` violating a single rule (found by original index), as a violations frame."""
    ev = _occurrences(df, keys, rule.event, rule.occurrence)
    hits: list[np.ndarray] = []

    if rule.min_age is not None:
        hits.append(ev.index[ev["age"] < rule.min_age].to_numpy())
    if rule.max_age is not None:
        hits.append(ev.index[ev["age"] > rule.max_age].to_numpy())
    if rule.min_spacing is not None and len(ev):
        ordered = ev.sort_values([*keys, "age"])
        gap = ordered["age"] - ordered.groupby(keys, observed=True)["age"].shift(1)
        hits.append(ordered.index[gap.notna() & (gap < rule.min_spacing)].to_numpy())
    if rule.not_after is not None and len(ev):
        anchor = _anchor_age(df, keys, rule.not_after, rule.not_after_occurrence)
        # carry the original df index through the join so we flag the right rows
        merged = ev.reset_index(names="_idx").merge(
            anchor.rename("_after").reset_index(), on=keys, how="inner"
        )
        hits.append(merged.loc[merged["age"] > merged["_after"], "_idx"].to_numpy())
    if rule.not_before is not None and len(ev):
        anchor = _anchor_age(df, keys, rule.not_before, rule.not_before_occurrence)
        # left join: an absent anchor leaves _before NaN, which is itself a violation
        merged = ev.reset_index(names="_idx").merge(
            anchor.rename("_before").reset_index(), on=keys, how="left"
        )
        early = merged["_before"].isna() | (merged["age"] < merged["_before"])
        hits.append(merged.loc[early, "_idx"].to_numpy())
    if rule.max_count is not None and len(ev):
        ordered = ev.sort_values([*keys, "age"])
        order = ordered.groupby(keys, observed=True).cumcount() + 1
        hits.append(ordered.index[order > rule.max_count].to_numpy())

    if not hits:
        return pd.DataFrame(columns=[*keys, *_VIOLATION_COLS])
    hit_index = np.unique(np.concatenate(hits))
    out = df.loc[hit_index, [*keys, "age", "event"]].copy()
    out["rule"] = rule.name
    out["severity"] = rule.severity
    return out


def violation_rates(
    violations: pd.DataFrame, df: pd.DataFrame, keys: list[str], *, by: tuple[str, ...] = ("seed",)
) -> pd.DataFrame:
    """Per-rule violation rates, per event of the kind the rule governs, by seed (and window).

    Returns ``[*by, rule, severity, n_violations, n_events, rate_per_event, n_persons]``, where
    ``n_persons`` is the distinct people in the cell.
    """
    _check_keys(df, keys)
    by = [c for c in by if c in df.columns]
    window = [c for c in ("age_start", "age_stop") if c in df.columns and c not in by]
    by = by + window

    cells = _group_size(df[keys].drop_duplicates(), by)  # one row per `by` cell
    n_persons = _group_nunique(df, by, "person_id").reindex(cells.index, fill_value=0)

    rows = []
    for rule in violations["rule"].unique():
        rv = violations[violations["rule"] == rule]
        severity, event = rv["severity"].iloc[0], rv["event"].iloc[0]
        n_viol = _group_size(rv, by).reindex(cells.index, fill_value=0)
        n_events = _group_size(df[df["event"] == event], by).reindex(cells.index, fill_value=0)
        cell = pd.DataFrame(
            {
                "rule": rule,
                "severity": severity,
                "n_violations": n_viol.to_numpy(),
                "n_events": n_events.to_numpy(),
                "n_persons": n_persons.to_numpy(),
            }
        )
        for i, col in enumerate(by):
            cell.insert(i, col, [k[i] if isinstance(k, tuple) else k for k in cells.index])
        cell["rate_per_event"] = cell["n_violations"] / cell["n_events"].replace(0, np.nan)
        rows.append(cell)

    cols = [*by, "rule", "severity", "n_violations", "n_events", "rate_per_event", "n_persons"]
    if not rows:
        return pd.DataFrame(columns=cols)
    out = pd.concat(rows, ignore_index=True)[cols]
    return out.sort_values([*by, "rule"]).reset_index(drop=True) if by else out


def _group_size(frame: pd.DataFrame, by: list[str]) -> pd.Series:
    """Group size per ``by`` cell (index = cell tuples), or a single total when ``by`` is empty."""
    if by:
        s = frame.groupby(by, observed=True).size()
        return s
    return pd.Series([len(frame)], index=[None])


def _group_nunique(frame: pd.DataFrame, by: list[str], col: str) -> pd.Series:
    """Distinct ``col`` per ``by`` cell, on the same index convention as :func:`_group_size`."""
    if col not in frame.columns:
        return _group_size(frame, by)
    if by:
        return frame.groupby(by, observed=True)[col].nunique()
    return pd.Series([frame[col].nunique()], index=[None])


def _check_keys(df: pd.DataFrame, keys: list[str]) -> None:
    missing = [k for k in keys if k not in df.columns]
    if missing:
        raise ValueError(f"keys {missing} not in frame columns {list(df.columns)}")
