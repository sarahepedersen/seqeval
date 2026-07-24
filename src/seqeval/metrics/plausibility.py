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
    closer than the gap; ``not_after`` flags the event occurring strictly after another event's
    first occurrence in the same group; ``not_before`` flags it occurring strictly *before* another
    event's first occurrence (or with that event absent entirely); ``max_count`` flags occurrences
    beyond the cap.

    ``not_after`` and ``not_before`` are not mirror images: ``not_after`` only constrains groups
    where the other event exists, while ``not_before`` treats a missing anchor as a violation (a
    divorce is illegal both before the first marriage and with no marriage at all).
    """
    _check_keys(df, keys)
    parts = [_check_one(df, keys, rule) for rule in rules]
    parts = [p for p in parts if len(p)]
    if not parts:
        return pd.DataFrame(columns=[*keys, *_VIOLATION_COLS])
    return pd.concat(parts, ignore_index=True).sort_values([*keys, "age"]).reset_index(drop=True)


def _check_one(df: pd.DataFrame, keys: list[str], rule: Rule) -> pd.DataFrame:
    """Rows of ``df`` violating a single rule (found by original index), as a violations frame."""
    ev = df[df["event"] == rule.event]
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
        first_after = df[df["event"] == rule.not_after].groupby(keys, observed=True)["age"].min()
        # carry the original df index through the join so we flag the right rows
        merged = ev.reset_index(names="_idx").merge(
            first_after.rename("_after").reset_index(), on=keys, how="inner"
        )
        hits.append(merged.loc[merged["age"] > merged["_after"], "_idx"].to_numpy())
    if rule.not_before is not None and len(ev):
        first_before = df[df["event"] == rule.not_before].groupby(keys, observed=True)["age"].min()
        # left join: an absent anchor leaves _before NaN, which is itself a violation
        merged = ev.reset_index(names="_idx").merge(
            first_before.rename("_before").reset_index(), on=keys, how="left"
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
    """Per-rule violation rates, both per sequence-group and per event, by seed (and window).

    Returns ``[*by, rule, severity, n_violations, n_groups, n_events, rate_per_group,
    rate_per_event]`` — the "rate of illegal moves" headline. Every rule that fired at least once is
    reported for *every* ``by`` cell (rate 0 where it did not fire), so a clean seed is visible.
    """
    _check_keys(df, keys)
    by = [c for c in by if c in df.columns]
    window = [c for c in ("age_start", "age_stop") if c in df.columns and c not in by]
    by = by + window

    n_groups = _group_size(df[keys].drop_duplicates(), by)  # cells -> n distinct sequence groups

    rows = []
    for rule in violations["rule"].unique():
        rv = violations[violations["rule"] == rule]
        severity, event = rv["severity"].iloc[0], rv["event"].iloc[0]
        n_viol = _group_size(rv, by).reindex(n_groups.index, fill_value=0)
        n_events = _group_size(df[df["event"] == event], by).reindex(n_groups.index, fill_value=0)
        cell = pd.DataFrame(
            {
                "rule": rule,
                "severity": severity,
                "n_violations": n_viol.to_numpy(),
                "n_groups": n_groups.to_numpy(),
                "n_events": n_events.to_numpy(),
            }
        )
        for i, col in enumerate(by):
            cell.insert(i, col, [k[i] if isinstance(k, tuple) else k for k in n_groups.index])
        cell["rate_per_group"] = cell["n_violations"] / cell["n_groups"].replace(0, np.nan)
        cell["rate_per_event"] = cell["n_violations"] / cell["n_events"].replace(0, np.nan)
        rows.append(cell)

    cols = [
        *by,
        "rule",
        "severity",
        "n_violations",
        "n_groups",
        "n_events",
        "rate_per_group",
        "rate_per_event",
    ]
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


def _check_keys(df: pd.DataFrame, keys: list[str]) -> None:
    missing = [k for k in keys if k not in df.columns]
    if missing:
        raise ValueError(f"keys {missing} not in frame columns {list(df.columns)}")
