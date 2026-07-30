"""The full typed config schema (00 section 5) plus the year -> day resolution boundary.

The config *is* the experiment specification. Every model uses ``extra="forbid"`` so unknown or
misspelled keys are hard errors, and cross-references (outcome names, condition names, covariates,
event aliases) are validated at parse time — before any data loads — with messages that name the
exact YAML path and enumerate the valid alternatives.

Config models store **years** and **aliases** exactly as written. The ``resolve_*`` functions at the
bottom of this module are the boundary (00 section 5.2): they produce day-valued, raw-token-valued
spec objects (:mod:`seqeval.core.specs`) that the rest of the system consumes. Arms never touch
year-valued config numbers or aliases directly.

Deviation from 01's signatures (noted per 00 section 7): the ``resolve_*`` helpers take the full
:class:`Config` rather than an arm sub-config, because resolving an event *alias* to its raw token
requires the top-level ``events:`` map, which only the full config carries. Each resolver reads only
the arm block named in 01 (backtesting for conditions/probability outcomes, forecasting for rules).
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Literal

import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator, model_validator

from seqeval.core.specs import (
    Condition,
    CountQuery,
    FertilityGrid,
    Frame,
    FramedOutcome,
    ReplicateSpec,
    Rule,
    TTESpec,
)
from seqeval.metrics._disclosure import MIN_CELL
from seqeval.units import years_to_days

logger = logging.getLogger("seqeval")

# fertiity-specific --> require 'birth' token in sequence
FERTILITY_TARGETS = {"ccf", "asfr_cohort", "ppr"}

# require `persons` table (exposure denominators)
FERTILITY_TARGETS_NEEDING_PERSONS = FERTILITY_TARGETS - {"ppr"}


class _Strict(BaseModel):
    """Base for every config model: forbid unknown keys (00 section 5 rule 9)."""

    model_config = ConfigDict(extra="forbid")


# =================================================================================================
# leaf config models
# =================================================================================================
class ModelConfig(_Strict):
    """``model:`` block. ``name`` is stamped as a ``model`` column into every result table."""

    name: str


class DataConfig(_Strict):
    """``data:`` block. Paths are resolved relative to the YAML's directory by ``load_config``."""

    observed: str
    generated: str | None = None
    persons: str | None = None
    event_definitions: str | None = None
    age_unit: Literal["days", "years"] = "days"


class EventConfig(BaseModel):
    """The ``events:`` alias -> raw-token map. Generic: no alias is required by the data layer.

    Modeled as a small mapping wrapper (rather than a bare dict) so it can be threaded through the
    :class:`~seqeval.io.loaders.Bundle` as one typed object.
    """

    model_config = ConfigDict(extra="allow")

    def __init__(self, **data: int | str) -> None:
        super().__init__(**data)

    @property
    def mapping(self) -> dict[str, int | str]:
        """The alias -> token dict."""
        return dict(self.__pydantic_extra__ or {})

    def __getitem__(self, alias: str) -> int | str:
        return self.mapping[alias]

    def __contains__(self, alias: object) -> bool:
        return alias in self.mapping

    def __iter__(self):  # type: ignore[override]
        return iter(self.mapping)

    def items(self):
        """Iterate ``(alias, token)`` pairs."""
        return self.mapping.items()

    def keys(self):
        """The declared aliases."""
        return self.mapping.keys()


#: Default birth-cohort band width (years) when no ``persons.cohort_width`` is configured.
DEFAULT_COHORT_WIDTH = 5


class PersonsConfig(_Strict):
    """``persons:`` block — covariate allowlist and the population's cohort definition."""

    covariates: list[str] = []
    # Width (years) of birth-cohort bands; shared by every arm that groups/stratifies by `cohort`
    # (descriptives CCF/ASFR/KM, backtesting stratification, forecasting Lexis subgroups).
    cohort_width: int = DEFAULT_COHORT_WIDTH


class ReplicatesConfig(_Strict):
    """``replicates:`` block — how seed-stochasticity becomes probability (00 section 3b)."""

    interval: Literal["jeffreys", "wilson"] = "jeffreys"
    level: float = 0.95
    min_replicates: int = 5


class TimingOutcomeConfig(_Strict):
    """A ``outcomes:`` registry entry — a timing quantity. No frame keys exist here (rule 5)."""

    event: str
    n: int = 1
    origin: str | None = None


class ConditionConfig(_Strict):
    """A backtesting ``conditions:`` entry — a generic count predicate on the observed prefix."""

    name: str
    event: str
    min_count: int | None = None
    max_count: int | None = None
    before_age: float | None = None  # years

    @model_validator(mode="after")
    def _at_least_one_bound(self) -> ConditionConfig:
        if self.min_count is None and self.max_count is None:
            raise ValueError(
                f"conditions[{self.name!r}]: at least one of min_count / max_count is required"
            )
        return self


class ProbabilityOutcomeConfig(_Strict):
    """A ``probability_outcomes:`` entry — either a framed reference or a count query.

    Exactly one *form* is legal:

    - **framed reference**: ``outcome`` (a registry name) + exactly one frame key
      (``by_age`` | ``within`` | ``within_origin``), optional ``given``.
    - **count query**: ``event`` + ``min_events`` + exactly one frame key (``by_age`` | ``within``;
      ``within_origin`` is illegal here), optional ``given``.

    Form and frame-arity are validated here; cross-references (registry name, condition name,
    ``within_origin`` requires an ``origin``) are validated on the parent :class:`Config`.
    """

    outcome: str | None = None
    event: str | None = None
    min_events: int | None = None
    by_age: float | None = None
    within: float | None = None
    within_origin: float | None = None
    given: str | None = None

    @property
    def is_framed(self) -> bool:
        return self.outcome is not None

    @property
    def frame_kind(self) -> Literal["by_age", "within", "within_origin"]:
        if self.by_age is not None:
            return "by_age"
        if self.within is not None:
            return "within"
        return "within_origin"

    @property
    def frame_value_years(self) -> float:
        return {"by_age": self.by_age, "within": self.within, "within_origin": self.within_origin}[
            self.frame_kind
        ]

    @model_validator(mode="after")
    def _validate_form(self) -> ProbabilityOutcomeConfig:
        has_framed = self.outcome is not None
        has_count = self.event is not None or self.min_events is not None
        frames_set = [
            k for k in ("by_age", "within", "within_origin") if getattr(self, k) is not None
        ]

        if has_framed and has_count:
            raise ValueError(
                "probability_outcomes entry: cannot mix a framed reference ('outcome') with a "
                "count query ('event'/'min_events') — use exactly one form"
            )
        if not has_framed and not has_count:
            raise ValueError(
                "probability_outcomes entry: must be a framed reference ('outcome: <name>') or a "
                "count query ('event: <alias>' + 'min_events: <int>')"
            )
        if len(frames_set) != 1:
            raise ValueError(
                "probability_outcomes entry: exactly one frame key is required, got "
                f"{frames_set or 'none'} (choose one of by_age / within / within_origin)"
            )
        if has_count:
            if self.event is None or self.min_events is None:
                raise ValueError(
                    "probability_outcomes count query: both 'event' and 'min_events' are required"
                )
            if self.frame_kind == "within_origin":
                raise ValueError(
                    "probability_outcomes count query: 'within_origin' is illegal on a count "
                    "query (no origin exists); use 'by_age' or 'within'"
                )
        return self


class WindowConfig(_Strict):
    """An explicit ``windows:`` entry (years) that subsets the windows present in the data."""

    age_start: float
    age_stop: float


class PprConfig(_Strict):
    """``fertility.ppr`` — parity-progression-ratio settings."""

    max_parity: int


class FertilityConfig(_Strict):
    """``descriptives.fertility`` block."""

    ccf: bool = False
    asfr: list[Literal["cohort"]] = []
    ppr: PprConfig | None = None
    age_bin_width: float = 1.0  # years; ASFR/exposure age-bin width


class DescriptivesConfig(_Strict):
    """``arms.descriptives`` block (past/observed).

    ``max_cohort_year`` drops people born after that year from *every* descriptive metric, not just
    from the cohort-indexed ones. The youngest cohorts are observed for only a few years, so their
    rates rest on a handful of person-years and read as jagged noise; excluding the people rather
    than trimming the plots keeps period and cohort metrics describing one population.
    """

    kaplan_meier: list[str] = []
    fertility: FertilityConfig | None = None
    stratify_by: list[str] = []
    max_cohort_year: int | None = None


class BacktestingConfig(_Strict):
    """``arms.backtesting`` block (past/generated)."""

    windows: Literal["all"] | list[WindowConfig] = "all"
    conditions: list[ConditionConfig] = []
    probability_outcomes: list[ProbabilityOutcomeConfig] = []
    aggregate_targets: list[str] = []
    min_seeds: int = 5
    # How the reliability curve / p_hat histogram group predicted probabilities: "quantile" aims at
    # equal-count bins (each point resting on roughly the same number of persons); "uniform" makes
    # fixed-width [0,1] bins. Same choice feeds the reported ECE so graph and score agree.
    calibration_binning: Literal["uniform", "quantile"] = "quantile"
    # How many bins to ask for. `p_hat = k/n` lives on a grid of `n + 1` points, so quantile binning
    # can realize at most one bin per distinct p_hat: asking for more than the replicate count buys
    # nothing. Roughly `n_seeds / 5` is a sane ceiling — 10 bins wants ~50 seeds behind it.
    calibration_bins: int = Field(default=10, ge=2)


class RuleConfig(_Strict):
    """An ``illegal_moves:`` entry (years) — a declarative illegal/implausible pattern.

    The subject is either an ``event`` alias — every occurrence of that token — or an ``outcome``
    name from the top-level registry, which pins one ordinal occurrence. Exactly one of the two.
    ``not_after``/``not_before`` take either kind: an alias anchors on the token's *first*
    occurrence, an outcome name on the occurrence that outcome names. So::

        - {event: birth, max_age: 45}                          # no birth after 45, ever
        - {outcome: first_divorce, not_before: first_marriage}  # and not with no marriage at all
        - {outcome: second_birth, not_before: first_marriage}   # the first birth is left alone

    An outcome's ``origin`` is ignored here: a rule is about the absolute ordering of occurrences,
    not about a duration measured from something else. ``min_spacing`` and ``max_count`` describe
    the whole stream, so they need ``event``, not ``outcome``.
    """

    event: str | None = None
    outcome: str | None = None
    name: str | None = None
    min_age: float | None = None
    max_age: float | None = None
    min_spacing: float | None = None
    not_after: str | None = None
    not_before: str | None = None
    max_count: int | None = None
    severity: Literal["illegal", "warn"] = "illegal"

    @model_validator(mode="after")
    def _exactly_one_subject(self) -> RuleConfig:
        if (self.event is None) == (self.outcome is None):
            raise ValueError("set exactly one of `event` or `outcome`")
        if self.outcome is not None:
            for field in ("min_spacing", "max_count"):
                if getattr(self, field) is not None:
                    raise ValueError(
                        f"`{field}` counts across every occurrence, so it needs `event`, "
                        f"not `outcome`"
                    )
        return self


class LexisConfig(_Strict):
    """``forecasting.lexis`` block."""

    outcome: str
    ages: list[float]
    years: list[int]
    subgroup_by: list[str] = []


class ReplicateVarianceConfig(_Strict):
    """One ``forecasting.replicate_variance`` block.

    ``event`` is the event alias the *within-seed spread* counts — how much one person's replicates
    disagree about how many times it happens. It defaults to the target event of the `lexis` outcome
    (births, in a fertility run), and set it to any other alias to ask the same question of that
    event. ``aggregate`` is unaffected: the CCF roll-up is about births by definition, so a
    non-birth block should leave it empty.

    ``name`` distinguishes this block's outputs on disk — every stem it writes ends in
    ``_<name>``. Left unset it falls back to ``event``, then to the `lexis` outcome, so a
    single-block config needs nothing. Two blocks must resolve to different names.
    """

    individual: bool = False
    aggregate: list[str] = []
    subgroup_by: list[str] = []
    event: str | None = None
    name: str | None = None


class ForecastingConfig(_Strict):
    """``arms.forecasting`` block (future/generated)."""

    windows: Literal["all"] | list[WindowConfig] = "all"
    lexis: LexisConfig | None = None
    illegal_moves: list[RuleConfig] = []
    replicate_variance: list[ReplicateVarianceConfig] = []
    sequence_descriptives: bool = False

    @field_validator("replicate_variance", mode="before")
    @classmethod
    def _accept_a_single_block(cls, v):
        """Wrap a lone mapping into a one-element list; ``None`` means the feature is off."""
        if v is None:
            return []
        return [v] if isinstance(v, dict | ReplicateVarianceConfig) else v

    @model_validator(mode="after")
    def _name_the_replicate_variance_blocks(self) -> ForecastingConfig:
        """Fill in each block's output name and require the names to be distinct.

        Resolved here rather than in the arm because uniqueness is a config error, and the fallback
        chain reaches the sibling ``lexis`` block — both of which only the parent can see.
        """
        for block in self.replicate_variance:
            if block.name is None:
                block.name = block.event or (self.lexis.outcome if self.lexis else None)
            if block.name is None:
                raise ValueError(
                    "arms.forecasting.replicate_variance: cannot name this block's outputs — "
                    "set `name`, or `event`, or configure a `lexis` block to inherit from"
                )
        names = [b.name for b in self.replicate_variance]
        duplicates = {n for n in names if names.count(n) > 1}
        if duplicates:
            raise ValueError(
                "arms.forecasting.replicate_variance: blocks would overwrite each other's output; "
                f"set a distinct `name` for {sorted(duplicates)}"
            )
        return self


class ArmsConfig(_Strict):
    """``arms:`` block. Presence of a field enables that arm (00 section 5 rule 1)."""

    descriptives: DescriptivesConfig | None = None
    backtesting: BacktestingConfig | None = None
    forecasting: ForecastingConfig | None = None


class OutputConfig(_Strict):
    """``output:`` block.

    ``individual_level`` defaults to false: per-person tables and figures name identifiable people,
    so they are opt-in. Setting it true writes them (``probabilities``, ``violations``,
    ``replicate_variance_individual``, ``replicate_occurrence``) alongside the aggregates.

    ``min_cell`` is the largest cell size still withheld: a cell resting on ``1..min_cell`` people
    *or events* is suppressed, a true zero is published. It guards counts of both kinds, plus the
    columns that invert to a count — raw variances and count-derived rates — per
    :data:`seqeval.metrics._disclosure.POLICIES`. ``min_cell: 0`` disables suppression entirely.
    """

    dir: str = "results/"
    figure_format: str = "png"
    individual_level: bool = False
    min_cell: int = Field(default=MIN_CELL, ge=0)


# =================================================================================================
# top-level config + cross-reference validation
# =================================================================================================
class Config(_Strict):
    """The whole experiment specification, parsed and cross-validated."""

    model: ModelConfig
    data: DataConfig
    events: EventConfig
    persons: PersonsConfig | None = None
    replicates: ReplicatesConfig = ReplicatesConfig()
    outcomes: dict[str, TimingOutcomeConfig] = {}
    arms: ArmsConfig = ArmsConfig()
    output: OutputConfig = OutputConfig()

    _base_dir: Path = PrivateAttr(default_factory=Path.cwd)

    # --- path resolution (relative to the YAML's directory) -------------------------------------
    def _abs(self, rel: str | None) -> Path | None:
        return (self._base_dir / rel) if rel is not None else None

    @property
    def observed_path(self) -> Path:
        p = self._abs(self.data.observed)
        assert p is not None  # observed is required
        return p

    @property
    def generated_path(self) -> Path | None:
        return self._abs(self.data.generated)

    @property
    def persons_path(self) -> Path | None:
        return self._abs(self.data.persons)

    @property
    def event_definitions_path(self) -> Path | None:
        return self._abs(self.data.event_definitions)

    @property
    def covariates(self) -> list[str]:
        return list(self.persons.covariates) if self.persons else []

    @property
    def cohort_width(self) -> int:
        """Shared birth-cohort band width (years); every arm reads this one value."""
        return self.persons.cohort_width if self.persons is not None else DEFAULT_COHORT_WIDTH

    # --- cross-reference validation -------------------------------------------------------------
    @model_validator(mode="after")
    def _cross_references(self) -> Config:
        aliases = set(self.events.keys())
        outcome_names = list(self.outcomes.keys())

        def _check_alias(alias: str, path: str) -> None:
            if alias not in aliases:
                raise ValueError(
                    f"{path}: unknown event alias {alias!r}; declared aliases are: "
                    f"{', '.join(sorted(aliases)) or '(none)'}"
                )

        # outcomes registry: event alias exists; origin references another declared outcome with
        # no origin of its own (reject depth > 1).
        for name, oc in self.outcomes.items():
            _check_alias(oc.event, f"outcomes.{name}.event")
            if oc.origin is not None:
                if oc.origin not in self.outcomes:
                    raise ValueError(
                        f"outcomes.{name}.origin: unknown outcome {oc.origin!r}; declared "
                        f"outcomes are: {', '.join(outcome_names) or '(none)'}"
                    )
                if self.outcomes[oc.origin].origin is not None:
                    raise ValueError(
                        f"outcomes.{name}.origin: chained origins are not allowed — {oc.origin!r} "
                        "itself declares an origin (nesting depth is limited to 1)"
                    )

        self._validate_backtesting(aliases, outcome_names, _check_alias)
        self._validate_forecasting(outcome_names, _check_alias)
        return self

    def _validate_backtesting(self, aliases, outcome_names, _check_alias) -> None:
        bt = self.arms.backtesting
        if bt is None:
            return

        # conditions: unique names, event alias exists.
        condition_names: list[str] = []
        for i, cond in enumerate(bt.conditions):
            if cond.name in condition_names:
                raise ValueError(
                    f"arms.backtesting.conditions[{i}].name: duplicate condition name "
                    f"{cond.name!r}; condition names must be unique"
                )
            condition_names.append(cond.name)
            _check_alias(cond.event, f"arms.backtesting.conditions[{i}].event")

        def _check_given(given: str | None, path: str) -> None:
            if given is not None and given not in condition_names:
                raise ValueError(
                    f"{path}: unknown condition {given!r}; declared conditions are: "
                    f"{', '.join(condition_names) or '(none)'}"
                )

        # probability outcomes: registry refs, within_origin needs origin, given refs, aliases.
        for i, po in enumerate(bt.probability_outcomes):
            path = f"arms.backtesting.probability_outcomes[{i}]"
            if po.is_framed:
                if po.outcome not in self.outcomes:
                    raise ValueError(
                        f"{path}.outcome: unknown outcome {po.outcome!r}; declared outcomes are: "
                        f"{', '.join(outcome_names) or '(none)'}"
                    )
                if po.frame_kind == "within_origin" and self.outcomes[po.outcome].origin is None:
                    raise ValueError(
                        f"{path}.within_origin: outcome {po.outcome!r} declares no 'origin', so "
                        "'within_origin' has no anchor — use by_age or within, or give the "
                        "outcome an origin"
                    )
            else:
                _check_alias(po.event, f"{path}.event")
            _check_given(po.given, f"{path}.given")

        # aggregate targets.
        for i, tgt in enumerate(bt.aggregate_targets):
            self._check_aggregate_target(
                tgt, outcome_names, f"arms.backtesting.aggregate_targets[{i}]"
            )

    def _validate_forecasting(self, outcome_names, _check_alias) -> None:
        fc = self.arms.forecasting
        if fc is None:
            return

        def _check_ref(ref: str, where: str, *, allow_outcome: bool = True) -> None:
            """A rule reference is an event alias or (where allowed) an outcome-registry name."""
            if allow_outcome and ref in self.outcomes:
                return
            if ref in self.events:
                return
            known = ", ".join(sorted(self.events.keys())) or "(none)"
            outcomes = ", ".join(outcome_names) or "(none)"
            raise ValueError(
                f"{where}: unknown reference {ref!r}; declared event aliases are: {known}"
                + (f"; declared outcomes are: {outcomes}" if allow_outcome else "")
            )

        for i, rule in enumerate(fc.illegal_moves):
            at = f"arms.forecasting.illegal_moves[{i}]"
            if rule.outcome is not None:
                if rule.outcome not in self.outcomes:
                    raise ValueError(
                        f"{at}.outcome: unknown outcome {rule.outcome!r}; declared outcomes "
                        f"are: {', '.join(outcome_names) or '(none)'}"
                    )
            else:
                _check_alias(rule.event, f"{at}.event")
            if rule.not_after is not None:
                _check_ref(rule.not_after, f"{at}.not_after")
            if rule.not_before is not None:
                _check_ref(rule.not_before, f"{at}.not_before")
        if fc.lexis is not None:
            if fc.lexis.outcome not in self.outcomes:
                raise ValueError(
                    f"arms.forecasting.lexis.outcome: unknown outcome {fc.lexis.outcome!r}; "
                    f"declared outcomes are: {', '.join(outcome_names) or '(none)'}"
                )
            self._check_stratifiers(fc.lexis.subgroup_by, "arms.forecasting.lexis.subgroup_by")
        for i, rv in enumerate(fc.replicate_variance):
            at = f"arms.forecasting.replicate_variance[{i}]"
            if rv.event is not None:
                _check_alias(rv.event, f"{at}.event")
            for j, tgt in enumerate(rv.aggregate):
                self._check_aggregate_target(tgt, outcome_names, f"{at}.aggregate[{j}]")
            self._check_stratifiers(rv.subgroup_by, f"{at}.subgroup_by")
        # descriptives stratifiers (validated here too, after covariates are known).
        if self.arms.descriptives is not None:
            self._check_stratifiers(
                self.arms.descriptives.stratify_by, "arms.descriptives.stratify_by"
            )

    def _check_aggregate_target(self, tgt: str, outcome_names, path: str) -> None:
        if tgt.startswith("km:"):
            name = tgt[len("km:") :]
            if name not in self.outcomes:
                raise ValueError(
                    f"{path}: km target references unknown outcome {name!r}; declared outcomes "
                    f"are: {', '.join(outcome_names) or '(none)'}"
                )
        elif tgt not in FERTILITY_TARGETS:
            raise ValueError(
                f"{path}: unknown aggregate target {tgt!r}; valid targets are "
                f"{', '.join(sorted(FERTILITY_TARGETS))} or km:<outcome>"
            )

    def _check_stratifiers(self, cols: list[str], path: str) -> None:
        # Allowlist = {cohort} u persons.covariates (00 section 5 rule 8; bare 'sex' not allowed).
        allowed = {"cohort", *self.covariates}
        for col in cols:
            if col not in allowed:
                raise ValueError(
                    f"{path}: {col!r} is not an allowed stratifier; allowed are "
                    f"{', '.join(sorted(allowed))} (declare it under persons.covariates to use it)"
                )

    # --- hashing (manifest, 06) -----------------------------------------------------------------
    def hash(self) -> str:
        """SHA-256 of a canonical JSON dump of the config (stable under mapping-key reordering)."""
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# =================================================================================================
# loading
# =================================================================================================
def load_config(path: str | Path) -> Config:
    """Parse and cross-validate a YAML config; resolve data paths relative to its directory."""
    path = Path(path)
    with path.open("r") as fh:
        raw = yaml.safe_load(fh) or {}
    cfg = Config.model_validate(raw)
    cfg._base_dir = path.resolve().parent
    return cfg


# =================================================================================================
# year -> day / alias -> token resolvers (00 section 5.2)
# =================================================================================================
def resolve_outcomes(cfg: Config) -> dict[str, TTESpec]:
    """Resolve the timing registry to day-valued, raw-token :class:`TTESpec` objects."""
    events = cfg.events

    def build(name: str) -> TTESpec:
        oc = cfg.outcomes[name]
        origin = build(oc.origin) if oc.origin is not None else None
        return TTESpec(target=events[oc.event], occurrence=oc.n, origin=origin)

    return {name: build(name) for name in cfg.outcomes}


def resolve_conditions(cfg: Config) -> dict[str, Condition]:
    """Resolve ``arms.backtesting.conditions`` to raw-token :class:`Condition` objects."""
    bt = cfg.arms.backtesting
    if bt is None:
        return {}
    out: dict[str, Condition] = {}
    for cc in bt.conditions:
        out[cc.name] = Condition(
            name=cc.name,
            event=cfg.events[cc.event],
            min_count=cc.min_count if cc.min_count is not None else 0,
            max_count=cc.max_count,
            before_age=years_to_days(cc.before_age) if cc.before_age is not None else None,
        )
    return out


def _frame(kind: str, value_years: float) -> Frame:
    return Frame(kind=kind, value=years_to_days(value_years))  # type: ignore[arg-type]


def resolve_probability_outcomes(
    cfg: Config, outcomes: dict[str, TTESpec]
) -> list[FramedOutcome | CountQuery]:
    """Resolve ``probability_outcomes`` to :class:`FramedOutcome` / :class:`CountQuery` objects.

    Unnamed entries get a readable, unique auto-name that encodes the frame and any ``given``.
    """
    bt = cfg.arms.backtesting
    if bt is None:
        return []
    resolved: list[FramedOutcome | CountQuery] = []
    for po in bt.probability_outcomes:
        frame = _frame(po.frame_kind, po.frame_value_years)
        value_yr = _fmt_years(po.frame_value_years)
        given_suffix = f"_given_{po.given}" if po.given else ""
        if po.is_framed:
            name = f"{po.outcome}_{po.frame_kind}_{value_yr}y{given_suffix}"
            resolved.append(
                FramedOutcome(name=name, tte=outcomes[po.outcome], frame=frame, given=po.given)
            )
        else:
            name = f"{po.event}_ge{po.min_events}_{po.frame_kind}_{value_yr}y{given_suffix}"
            resolved.append(
                CountQuery(
                    name=name,
                    event=cfg.events[po.event],
                    min_events=po.min_events,
                    frame=frame,
                    given=po.given,
                )
            )
    return resolved


def _fmt_years(y: float) -> str:
    """Render a year value for auto-names: ``5.0 -> '5'``, ``2.5 -> '2p5'``."""
    return str(int(y)) if float(y).is_integer() else str(y).replace(".", "p")


def resolve_rules(cfg: Config) -> list[Rule]:
    """Resolve ``arms.forecasting.illegal_moves`` to day-valued, raw-token :class:`Rule` objects.
    """
    fc = cfg.arms.forecasting
    if fc is None:
        return []

    def token_and_occurrence(ref: str) -> tuple[object, int | None]:
        """``(token, occurrence)`` for an outcome name; ``(token, None)`` for an event alias."""
        if ref in cfg.outcomes:
            oc = cfg.outcomes[ref]
            return cfg.events[oc.event], oc.n
        return cfg.events[ref], None

    rules: list[Rule] = []
    for i, rc in enumerate(fc.illegal_moves):
        subject_ref = rc.outcome if rc.outcome is not None else rc.event
        event, occurrence = token_and_occurrence(subject_ref)
        after = token_and_occurrence(rc.not_after) if rc.not_after is not None else None
        before = token_and_occurrence(rc.not_before) if rc.not_before is not None else None
        rules.append(
            Rule(
                name=rc.name or f"{subject_ref}_rule_{i}",
                event=event,
                occurrence=occurrence,
                min_age=years_to_days(rc.min_age) if rc.min_age is not None else None,
                max_age=years_to_days(rc.max_age) if rc.max_age is not None else None,
                min_spacing=years_to_days(rc.min_spacing) if rc.min_spacing is not None else None,
                not_after=after[0] if after is not None else None,
                not_before=before[0] if before is not None else None,
                # An alias anchor has no ordinal of its own, so it means the first occurrence.
                not_after_occurrence=(after[1] or 1) if after is not None else 1,
                not_before_occurrence=(before[1] or 1) if before is not None else 1,
                max_count=rc.max_count,
                severity=rc.severity,
            )
        )
    return rules


def resolve_replicates(cfg: Config) -> ReplicateSpec:
    """Pure passthrough of the ``replicates:`` block to a :class:`ReplicateSpec` (with defaults)."""
    rc = cfg.replicates
    return ReplicateSpec(
        interval=rc.interval,
        level=rc.level,
        min_replicates=rc.min_replicates,
    )


def resolve_fertility_grid(cfg: Config) -> FertilityGrid:
    """Cell geometry for the fertility aggregates, read off the descriptives fertility block.

    Backtesting bins the same quantities descriptives does, and the two end up beside each other in
    the report, so they share one grid rather than each carrying its own constants. Anything the
    descriptives block does not set (or the whole block being absent) falls back to
    :class:`~seqeval.core.specs.FertilityGrid`'s defaults — there is no separate backtesting key to
    set, by design.
    """
    fert = cfg.arms.descriptives.fertility if cfg.arms.descriptives else None
    if fert is None:
        return FertilityGrid()
    defaults = FertilityGrid()
    return FertilityGrid(
        max_parity=fert.ppr.max_parity if fert.ppr else defaults.max_parity,
        age_bin_width=fert.age_bin_width,
    )


def resolve_windows(
    spec: Literal["all"] | list[WindowConfig], available: pd.DataFrame
) -> list[tuple[int, int]]:
    """Resolve a ``windows:`` spec against the windows actually present in the data.

    Parameters
    ----------
    spec : "all" or list of WindowConfig
        ``"all"`` consumes every available window; an explicit (year-valued) list *subsets*.
    available : pandas.DataFrame
        Output of ``Bundle.available_windows()`` — day-valued ``age_start``/``age_stop`` columns.

    Returns
    -------
    list of (int, int)
        Day-valued ``(age_start, age_stop)`` pairs, restricted to what exists. Requested windows
        absent from the data are dropped with a logged warning.
    """
    avail = {(int(r.age_start), int(r.age_stop)) for r in available.itertuples(index=False)}
    if spec == "all":
        return sorted(avail)
    requested = [(years_to_days(w.age_start), years_to_days(w.age_stop)) for w in spec]
    present = [w for w in requested if w in avail]
    missing = [w for w in requested if w not in avail]
    if missing:
        logger.warning(
            "resolve_windows: %d requested window(s) absent from the generated data and skipped: "
            "%s (in days)",
            len(missing),
            missing,
        )
    return sorted(set(present))
