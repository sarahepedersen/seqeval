"""Resolver target types — the frozen, day-valued, raw-token-valued "question specs".

Reproduced from ``00a_dataclass_reference.md``. Data in this system flows through three shapes.
**Carriers** hold validated data (``Bundle``, ``AgeBins``). **Question specs** (this module) are
frozen, day-valued, raw-token-valued objects that state a question precisely — they are what
``config.resolve_*`` produces from the year-valued, alias-valued YAML, and what ``core/`` evaluator
functions consume. **Test scaffolding** (``HazardSpec``) exists only under ``tests/``. No dataclass
contains logic; all evaluation lives in functions that take ``(DataFrame, keys, spec)``.

The specs, one by one
---------------------

``TTESpec`` — a timing quantity. States: *the time at which the ``occurrence``-th ``target`` event
happens, measured from ``origin`` (person's birth when ``origin is None``).* This is the registry
primitive — sequence-intrinsic, context-free, valid for observed and generated data alike.
``origin`` nests one level at most (enforced at config parse); when set, the quantity is implicitly
conditioned on the origin occurring. Used directly (no frame) wherever a duration is the object of
study: Kaplan-Meier curves, ``km:*`` aggregate backtest targets, the Lexis outcome.

``Frame`` — a window that makes a question binary. A timing quantity answers "when"; attaching a
``Frame`` converts it to "does it happen inside this window". Frames are the *only* place evaluation
context enters: ``within`` means "within ``value`` of the jump-off (t2)", which is why frames live
in arm config, never in the registry. ``by_age`` is absolute; ``within_origin`` is relative to a
``TTESpec``'s own origin event and is therefore only legal on framed references whose registry
outcome declares an ``origin``.

``FramedOutcome`` vs ``CountQuery`` — the distinction that matters. Both resolve a
``probability_outcomes:`` entry and both evaluate to the same output shape, so everything downstream
is agnostic to which produced the table. They are separate classes because they ask genuinely
different scientific questions that are easy to conflate. A **FramedOutcome** asks about a specific
ordinal occurrence in the whole life course ("does the person's 2nd birth happen by age 35?"); the
ordinal is absolute, counted from birth, including events shown in the prompt. A **CountQuery** asks
how many events happen after the jump-off ("do >= 1 births occur in (t2, t2+5]?"). Only
``evaluate_framed`` performs the settled-at-jump-off check; ``evaluate_count`` never needs it
because it counts strictly after t2 by construction.

``Condition`` — a population filter, the mirror image of ``CountQuery``. States: *keep only
sequence-groups where the count of ``event`` occurrences at ages <= anchor lies in
[min_count, max_count]*, where the anchor is ``before_age`` if set, else the caller's jump-off.
Parity conditioning is the fertility instance (``event=birth, min_count=1, max_count=1``); nothing
in ``core/`` knows the word parity.

``Rule`` — an illegal-move pattern. States one impossible or implausible pattern to flag in
sequences (all age/spacing values in days). Purely declarative — the rules engine (05) interprets
whichever fields are set; adding a rule never means adding code.

``ReplicateSpec`` — how seed-stochasticity becomes probability (plan 02b). Resolved from the
top-level ``replicates:`` config block; states the *policy* for turning per-run replicate counts
(k of n) into probability estimates. A policy object rather than loose kwargs so that 04 and 05
provably apply identical estimation settings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

__all__ = [
    "TTESpec",
    "Frame",
    "FramedOutcome",
    "CountQuery",
    "Condition",
    "Rule",
    "ReplicateSpec",
    "FertilityGrid",
]


@dataclass(frozen=True)
class TTESpec:
    """A timing quantity: when the ``occurrence``-th ``target`` event happens, from ``origin``.

    Parameters
    ----------
    target : Any
        Raw event token (as emitted by the model, e.g. int ``42`` or a string).
    occurrence : int, default 1
        Which ordinal occurrence of ``target`` (1 = first).
    origin : TTESpec or None, default None
        The event the duration is measured from; ``None`` means the person's birth (age 0).
        Nests at most one level (enforced at config parse).
    """

    target: Any
    occurrence: int = 1
    origin: TTESpec | None = None


@dataclass(frozen=True)
class Frame:
    """A time window that turns a timing quantity into a yes/no question.

    Parameters
    ----------
    kind : {"by_age", "within", "within_origin"}
        ``by_age`` is an absolute age; ``within`` is a duration from the jump-off (t2);
        ``within_origin`` is a duration from the outcome's own origin event.
    value : int
        The window bound, in **days** (an absolute age for ``by_age``, a duration otherwise).
    """

    kind: Literal["by_age", "within", "within_origin"]
    value: int  # days


@dataclass(frozen=True)
class FramedOutcome:
    """Does a specific ordinal occurrence land inside a frame? (evaluated by 02.)

    Parameters
    ----------
    name : str
        Stable identifier used in output tables.
    tte : TTESpec
        The timing quantity whose occurrence is being framed.
    frame : Frame
        The window that makes it binary.
    given : str or None, default None
        Name of a :class:`Condition` restricting the evaluated population.
    """

    name: str
    tte: TTESpec
    frame: Frame
    given: str | None = None


@dataclass(frozen=True)
class CountQuery:
    """Do >= ``min_events`` post-jump-off occurrences land inside a frame? (evaluated by 02.)

    Parameters
    ----------
    name : str
        Stable identifier; auto-named when unnamed (e.g. ``"birth_ge1_within_5y"``).
    event : Any
        Raw event token to count.
    min_events : int
        Threshold; the query is true iff at least this many events fall in ``frame``.
    frame : Frame
        The post-jump-off window (only ``by_age`` or ``within`` are legal here).
    given : str or None, default None
        Name of a :class:`Condition` restricting the evaluated population.
    """

    name: str
    event: Any
    min_events: int
    frame: Frame
    given: str | None = None


@dataclass(frozen=True)
class Condition:
    """A count predicate on the observed prefix, used as a population filter.

    Keep only sequence-groups where the count of ``event`` occurrences at ages <= anchor lies in
    ``[min_count, max_count]``; the anchor is ``before_age`` if set, else the caller's jump-off.

    Parameters
    ----------
    name : str
        Stable identifier, referenced by ``given:`` on framed outcomes / count queries.
    event : Any
        Raw event token to count.
    min_count : int, default 0
        Lower bound (inclusive).
    max_count : int or None, default None
        Upper bound (inclusive); ``None`` means no upper bound.
    before_age : int or None, default None
        Anchor age in **days**; ``None`` anchors at the jump-off (t2).
    """

    name: str
    event: Any
    min_count: int = 0
    max_count: int | None = None
    before_age: int | None = None  # days; None -> anchor at jump-off


@dataclass(frozen=True)
class Rule:
    """An impossible or implausible pattern to flag in sequences (interpreted by 05).

    Purely declarative: the rules engine acts on whichever fields are set. All age/spacing values
    are in **days**.

    Parameters
    ----------
    name : str
        Stable identifier for the rule.
    event : Any
        Raw event token the rule applies to.
    occurrence : int or None, default None
        Which ordinal occurrence of ``event`` the rule is about (1 = first). ``None`` — what an
        ``event:``-keyed config entry produces — means every occurrence. Naming an outcome in the
        config pins this, so ``second_birth`` constrains the second birth and leaves the first
        alone.
    min_age, max_age : int or None
        Flag occurrences younger than ``min_age`` / older than ``max_age`` (days).
    min_spacing : int or None
        Flag consecutive occurrences closer together than this (days). About the whole stream, so it
        is only meaningful with ``occurrence`` unset.
    not_after : Any or None
        Flag the subject occurring after the anchor token's ``not_after_occurrence``-th occurrence.
    not_before : Any or None
        Flag the subject occurring before the anchor token's ``not_before_occurrence``-th occurrence
        — including when that occurrence never happens at all (a divorce with no marriage anywhere
        is a violation).
    not_after_occurrence, not_before_occurrence : int, default 1
        Which ordinal occurrence of the anchor token to measure against. An event alias in the
        config leaves these at 1 (the first occurrence); an outcome name sets them.
    max_count : int or None
        Flag sequences with more than this many occurrences of ``event``. About the whole stream, so
        it is only meaningful with ``occurrence`` unset.
    severity : {"illegal", "warn"}, default "illegal"
        Whether a violation is a hard illegal move or a soft implausibility warning.
    """

    name: str
    event: Any
    occurrence: int | None = None
    min_age: int | None = None
    max_age: int | None = None
    min_spacing: int | None = None
    not_after: Any | None = None
    not_before: Any | None = None
    not_after_occurrence: int = 1
    not_before_occurrence: int = 1
    max_count: int | None = None
    severity: Literal["illegal", "warn"] = "illegal"


@dataclass(frozen=True)
class ReplicateSpec:
    """Policy for turning per-run replicate counts (k of n) into probabilities (consumed by 02b).

    Parameters
    ----------
    interval : {"jeffreys", "wilson"}, default "jeffreys"
        Interval method for the per-run probability. The point estimate is always the unsmoothed
        MLE ``k/n``.
    level : float, default 0.95
        Confidence level for intervals.
    min_replicates : int, default 5
        Warn when a run has fewer replicates than this (probability grid coarser than 1/n).
    """

    interval: Literal["jeffreys", "wilson"] = "jeffreys"
    level: float = 0.95
    min_replicates: int = 5


@dataclass(frozen=True)
class FertilityGrid:
    """Shared cell geometry for the fertility aggregates, so every arm bins them alike.

    A backtest PPR or ASFR is meant to be read against the descriptive one it sits near in the
    report; on a different parity ceiling or a different age-bin width the two are not comparable.
    Resolved once from the descriptives fertility block (``config.resolve_fertility_grid``) and
    handed to whichever arm needs it, rather than each arm carrying its own constants.

    Parameters
    ----------
    max_parity : int, default 6
        Highest parity transition a PPR table reports (``max_parity-1 -> max_parity``).
    age_bin_width : float, default 1.0
        Width in years of an ASFR/exposure age bin.
    """

    max_parity: int = 6
    age_bin_width: float = 1.0
