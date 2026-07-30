"""Small-cell suppression for published aggregates.

Every table that is safe to publish is a table of binned counts, and a binned count is only safe
while no cell is thin enough to be about one identifiable person. This module holds that rule, once,
so the ridge (:func:`seqeval.metrics.ml.timing_error_distribution`) and the parity distribution
(:func:`seqeval.metrics.fertility.parity_distribution`) suppress on identical terms — a reader who
learns the convention in one figure reads the other.

Two layers:

- :func:`suppress_small_cells` is the rule itself, called by the metric that builds a distribution.
- :data:`POLICIES` declares, per published table, which columns are counts and which columns invert
  to a count. :func:`apply_policy` runs the rule from that declaration and
  :func:`assert_publishable` re-checks it at the single write
  (:meth:`seqeval.arms._common.OutputWriter.frame`), so a table cannot reach disk unguarded.

The count applies to both people and events. A cell resting on four hundred women but holding one
birth is a cell about one birth, and is suppressed.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import pandas as pd

#: Cells resting on this many people or events, or fewer, are withheld.
MIN_CELL = 3

__all__ = [
    "MIN_CELL",
    "DisclosureError",
    "Policy",
    "POLICIES",
    "apply_policy",
    "assert_publishable",
    "policy_for",
    "suppress_small_cells",
]


class DisclosureError(AssertionError):
    """A frame reached the writer still holding a cell below the publication threshold."""


def _as_cols(cols: str | Sequence[str]) -> tuple[str, ...]:
    return (cols,) if isinstance(cols, str) else tuple(cols)


def _mask(series: pd.Series, hidden) -> pd.Series:
    """NA out ``hidden`` rows, widening integers to nullable ``Int64`` so the NA survives."""
    if pd.api.types.is_integer_dtype(series.dtype) and not pd.api.types.is_extension_array_dtype(
        series.dtype
    ):
        series = series.astype("Int64")
    return series.mask(hidden)


def suppress_small_cells(
    df: pd.DataFrame,
    *,
    count_cols: str | Sequence[str],
    by: Sequence[str] = (),
    min_cell: int = MIN_CELL,
    also_null: tuple[str, ...] = (),
    complement: bool = True,
) -> pd.DataFrame:
    """Withhold cells resting on too few people or events; add a ``suppressed`` flag.

    A suppressed cell keeps its row and its bin edges — only the columns in ``count_cols`` and
    ``also_null`` become NA, so the shape of the table (and of the figure drawn from it) survives.
    Four rules:

    - ``0 < count <= min_cell`` is suppressed, for *any* column in ``count_cols``. A row is judged
      on its thinnest count, so an event count guards a cell that a person count would pass.
    - ``count == 0`` is published.
    - Within each ``by`` group, a *lone* suppressed cell forces a second: the smallest non-zero cell
      still standing is suppressed too. One withheld cell is recoverable by subtracting the
      published cells from the group total, so suppressing it alone withholds nothing. The
      complement is chosen on the first column in ``count_cols``, and is only meaningful where that
      group total is itself published — pass ``complement=False`` where the rows do not partition
      anything a reader can see. Empty ``by`` treats the whole frame as one group.
    """
    out = df.copy()
    counts = [c for c in _as_cols(count_cols) if c in out.columns]

    prior = (
        out["suppressed"].fillna(False).astype(bool)
        if "suppressed" in out.columns
        else pd.Series(False, index=out.index)
    )
    hidden = prior.copy()
    for col in counts:
        values = pd.to_numeric(out[col], errors="coerce")
        thin = ((values > 0) & (values <= min_cell)).fillna(False).astype(bool)
        hidden |= thin
    out["suppressed"] = hidden

    if counts and complement:
        primary = counts[0]
        group_keys = [c for c in by if c in out.columns]
        groups = (
            out.groupby(group_keys, observed=True, dropna=False).indices.values()
            if group_keys
            else [out.index.to_numpy()]
        )
        for idx in groups:
            grp = out.loc[idx]
            if int(grp["suppressed"].sum()) != 1:
                continue
            eligible = grp[~grp["suppressed"] & (pd.to_numeric(grp[primary], errors="coerce") > 0)]
            if len(eligible):
                out.loc[eligible[primary].idxmin(), "suppressed"] = True

    hidden = out["suppressed"].to_numpy()
    for col in [*counts, *(c for c in also_null if c in out.columns)]:
        out[col] = _mask(out[col], hidden)
    return out


# --------------------------------------------------------------------------------------------
# Per-table policy
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Policy:
    """What a published table's counts are, and what else falls with them.

    ``trip`` columns are those that hold the event/person counts being checked. ``also_null``
    columns are withheld alongside them because a count is recoverable from them.
    The point estimate and its confidence
    interval stay on a suppressed row, so a curve keeps its shape.
    """

    trip: tuple[str, ...]
    also_null: tuple[str, ...] = ()
    by: tuple[str, ...] = field(default=())
    complement: bool = False


#: Columns carrying the sampling variance a pooled estimate is built from. Each is a monotone
#: function of the cell's count, so they fall with it wherever they appear.
_POOLING_VAR = ("mean_var", "between_var", "pooled_var", "se")

#: A pooled cell is judged on the real people behind it, never on ``n_units`` — seeds multiply
#: trajectories, not persons, and cannot manufacture privacy.
_POOLED_EXTRA = ("n_units", *_POOLING_VAR)

#: The log-log interval is built from the same Greenwood sum as the variance, so publishing it is
#: publishing the variance: ``se = ln(log(ci_lo)/log(S))/z`` returns ``cum_v``, and its increment
#: together with the survival ratio ``S_i/S_(i-1) = 1 - d/n`` solves exactly for ``d`` and ``n``.
#: A suppressed KM row therefore keeps its time and its survival, and nothing else.
_KM_INVERTS = ("greenwood_var", "ci_lo", "ci_hi")

_KM = Policy(trip=("n_events", "n_at_risk", "n_persons"), also_null=_KM_INVERTS)
_PPR = Policy(trip=("n_at_risk", "n_progressed", "n_persons"), also_null=("ppr_var",))
_ASFR = Policy(trip=("births", "n_persons"), also_null=("person_years", "asfr_var"))
_LEXIS = Policy(trip=("n_events", "n_persons"), also_null=("person_years", "rate_var"))

#: Published table stem -> policy. Stems are the names passed to
#: :meth:`seqeval.arms._common.OutputWriter.frame`; :func:`policy_for` also matches the
#: parameterised stems (``km_<outcome>``, ``within_seed_variance_distribution_by_<col>``).
POLICIES: dict[str, Policy] = {
    # -- survival ---------------------------------------------------------------------------
    "km_by_seed": _KM,
    "km_pooled": Policy(
        trip=("n_events", "n_at_risk", "n_source_persons"),
        also_null=(*_KM_INVERTS, *_POOLED_EXTRA),
    ),
    # -- fertility --------------------------------------------------------------------------
    "ppr": _PPR,
    "ppr_by_seed": _PPR,
    "ppr_pooled": Policy(
        trip=("n_at_risk", "n_progressed", "n_source_persons"),
        also_null=("ppr_var", *_POOLED_EXTRA),
    ),
    "ccf": Policy(
        trip=("n_persons",),
        also_null=("within_var", "between_var", "total_var"),
    ),
    # The generated CCF-by-cohort curve the backtesting overlays draw: same decomposition, same
    # inversion (`total_var = var_i(mu_i)/n`), so the same policy. `ccf` and `complete` survive.
    "ccf_variance": Policy(
        trip=("n_persons",),
        also_null=("within_var", "between_var", "total_var"),
    ),
    "asfr_cohort": _ASFR,
    "asfr_by_seed": _ASFR,
    "asfr_pooled": Policy(
        trip=("births", "n_source_persons"),
        also_null=("person_years", "asfr_var", *_POOLED_EXTRA),
    ),
    # `n_women_total` and `n_replicates_total` are the cohort totals these cells partition, so a
    # lone withheld parity is recoverable by subtraction.
    # `n_women_total` is the cohort's own head count and is inspected on its own terms: a cohort of
    # three women publishes nothing, not even how many there were. `n_replicates_total` is that
    # number times the seed count, so it falls with it.
    "parity_distribution": Policy(
        trip=("n_persons", "n_replicates", "n_women_total"),
        also_null=("n_women_equiv", "share", "n_replicates_total"),
        by=("cohort",),
        complement=True,
    ),
    # -- forecasting ------------------------------------------------------------------------
    "lexis_cohort_observed": _LEXIS,
    "lexis_cohort_forecast": _LEXIS,
    "lexis_cohort_pooled": Policy(
        trip=("n_events", "n_source_persons"),
        also_null=("person_years", "rate_var", *_POOLED_EXTRA),
    ),
    # Observed rows carry `n_persons` and forecast rows `n_source_persons`; the other side is NA
    # and an NA never trips, so the union of both policies reads each row on its own terms.
    "lexis_cohort_combined": Policy(
        trip=("n_events", "n_persons", "n_source_persons"),
        also_null=("person_years", "rate_var", *_POOLED_EXTRA),
    ),
    # -- sequence descriptives ---------------------------------------------------------------
    # Age bins partition the token's own `n_events_total`, which this table publishes, so a lone
    # withheld bin is recoverable by subtraction.
    "event_age_distribution": Policy(
        trip=("n_events", "n_source_persons", "n_events_total"),
        also_null=("n_units", "person_years", "rate", "share"),
        # `cohort` is in the key: the all-cohorts row and a cohort's rows are different
        # populations, and the complement rule must not group them together.
        by=("source", "age_start", "age_stop", "cohort", "alias"),
        complement=True,
    ),
    # Declared tokens partition nothing published here, so no complement.
    "token_frequency": Policy(
        trip=("n_events", "n_units_with_any", "n_persons_with_any", "n_source_persons"),
        also_null=("n_units", "share_with_any", "mean_person_share"),
    ),
    "violation_rates": Policy(
        trip=("n_violations", "n_events", "n_persons"),
        also_null=("rate_per_event",),
    ),
    "replicate_variance_aggregate": Policy(
        trip=("n_persons",),
        also_null=("within_var", "between_var", "total_var", "se_total"),
    ),
    # -- backtesting ------------------------------------------------------------------------
    # `n_evaluable` is this table's head count — it carries no `n_persons` alias, because the
    # accounting identity `n_uncovered = n_condition - n_evaluable - n_settled` is written in these
    # names. `n_seed_min/median/max` are replicate depths, not head counts, and stay.
    "coverage": Policy(
        trip=(
            "n_evaluable",
            "n_condition",
            "n_excluded_condition",
            "n_settled",
            "n_uncovered",
        )
    ),
    "scores": Policy(trip=("n_persons",)),
    # `p_mean` is the model's own prediction and survives; `y_rate` is observed events over `n`.
    "calibration": Policy(trip=("n", "n_persons"), also_null=("y_rate",)),
    "p_hat_distribution": Policy(trip=("n_persons", "n_total"), complement=True),
    "aggregate_error": Policy(trip=("n_persons",)),
    # `n_trajectories`/`n_excluded` are group constants repeated on every row, so a thin group nulls
    # itself out across the whole group. The lone-cell rule already ran upstream with the full
    # grouping (`ml.timing_error_distribution`); re-running it here on a partial key would suppress
    # across windows, so this pass only backstops the columns that pass added.
    "timing_error": Policy(trip=("n_persons", "n_pred_bin", "n_trajectories", "n_excluded")),
    # -- dispersion -------------------------------------------------------------------------
    "within_seed_variance_distribution": Policy(trip=("n_persons", "n_group")),
    "within_seed_quantile_summary": Policy(
        trip=("n_persons",),
        also_null=("mean_k", "mean_q0", "mean_q25", "mean_q50", "mean_q75", "mean_q100"),
    ),
}

POLICIES["timing_error_by_seed"] = POLICIES["timing_error"]

#: Stems that are parameterised at write time, longest first so the exact entries above win.
#:
#: This is why a `replicate_variance` block's name is appended rather than infixed: the
#: parameterised part of a stem has to sit at the *end* for a prefix to still find its policy.
_PREFIXES: tuple[tuple[str, str], ...] = (
    ("within_seed_variance_distribution", "within_seed_variance_distribution"),
    ("within_seed_quantile_summary", "within_seed_quantile_summary"),
    ("replicate_variance_aggregate", "replicate_variance_aggregate"),
    ("km_", "km_by_seed"),
)


def policy_for(name: str) -> Policy | None:
    """The policy governing a published table stem, or ``None`` if the table has no counts.

    Exact stems win; the rest resolve by prefix, which is how the parameterised names
    (``km_first_birth``, ``within_seed_variance_distribution_by_cohort``) reach their policy.
    """
    if name in POLICIES:
        return POLICIES[name]
    for prefix, key in _PREFIXES:
        if name.startswith(prefix):
            return POLICIES[key]
    return None


def apply_policy(name: str, df: pd.DataFrame, *, min_cell: int = MIN_CELL) -> pd.DataFrame:
    """Suppress ``df`` under the policy registered for ``name``; a no-op where none is registered.

    Idempotent: a re-run sees NA where counts were withheld, an NA never trips, and the existing
    ``suppressed`` flag is carried forward. Arms call this before drawing a figure — the figures are
    built from the frame, not from the parquet — and the writer calls it again before the write.
    """
    policy = policy_for(name)
    if policy is None or df.empty:
        return df
    if not any(c in df.columns for c in policy.trip):
        return df
    if min_cell <= 0:
        return df
    return suppress_small_cells(
        df,
        count_cols=policy.trip,
        by=policy.by,
        min_cell=min_cell,
        also_null=policy.also_null,
        complement=policy.complement,
    )


def assert_publishable(name: str, df: pd.DataFrame, *, min_cell: int = MIN_CELL) -> None:
    """Raise :class:`DisclosureError` if any registered count in ``df`` is still ``1..min_cell``.

    The backstop, not the mechanism: reaching it means a frame was built or reshaped after the
    policy ran. Cheap enough to run on every write.
    """
    policy = policy_for(name)
    if policy is None or df.empty or min_cell <= 0:
        return
    for col in policy.trip:
        if col not in df.columns:
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        thin = (values > 0) & (values <= min_cell)
        if bool(thin.any()):
            raise DisclosureError(
                f"{name}.{col} publishes {int(thin.sum())} cell(s) of size <= {min_cell}; "
                "the table was reshaped after suppression ran"
            )
