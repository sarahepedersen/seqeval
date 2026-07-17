"""Pandera schemas for the four input artifacts, plus the canonical key constants.

These schemas validate the **post-load canonical form** (00 architecture, section 4.1): ages are
already normalized to integer days, ``event`` is a category, ids/seeds have their target integer
dtypes. Unit normalization and dtype casting happen in :mod:`seqeval.io.loaders`; validation here
is the gate that guarantees every downstream ``(df, keys)`` function can trust the shape.

All violations are surfaced as :class:`SchemaError` — a single exception type carrying the
artifact name and an actionable fix hint, so loaders never leak raw pandera errors.
"""

from __future__ import annotations

import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaError as PanderaSchemaError
from pandera.errors import SchemaErrors as PanderaSchemaErrors

from seqeval.units import years_to_days

__all__ = [
    "OBS_KEYS",
    "GEN_KEYS",
    "RUN_KEYS",
    "SchemaError",
    "OBSERVED_SCHEMA",
    "GENERATED_SCHEMA",
    "PERSONS_SCHEMA",
    "EVENT_DEFINITIONS_SCHEMA",
    "validate",
]

# --- canonical key columns (00 section 4.3) -----------------------------------------------------
#: Group keys for observed sequences: one real sequence per person.
OBS_KEYS = ["person_id"]
#: Group keys for generated sequences: many runs per person, keyed by seed and window.
GEN_KEYS = ["person_id", "seed", "age_start", "age_stop"]
#: A *run* is a (person, window) pair — GEN_KEYS minus the replicate seed. The replicate engine
#: (plan 02b) groups by this to pool a run's seeds into an empirical probability.
RUN_KEYS = ["person_id", "age_start", "age_stop"]

#: Oldest biologically plausible age, in days — the upper bound on every ``age`` column.
_MAX_AGE_DAYS = years_to_days(110)


class SchemaError(ValueError):
    """Raised when an artifact fails validation.

    Wraps the underlying pandera failure with the artifact name and a fix hint so the message is
    actionable (00 section 6: "loaders raise ``SchemaError`` with actionable messages").
    """


# --- schemas ------------------------------------------------------------------------------------
# ``person_id`` is intentionally dtype-free: the data model allows int64 *or* string (00 section
# 4.1). We only require presence and non-nullness; the join in ``load_all`` catches type drift.
_PERSON_ID = pa.Column(nullable=False, coerce=False, required=True)


OBSERVED_SCHEMA = pa.DataFrameSchema(
    {
        "person_id": _PERSON_ID,
        "age": pa.Column(
            "int32",
            checks=[pa.Check.in_range(0, _MAX_AGE_DAYS)],
            nullable=False,
            coerce=True,
        ),
        "event": pa.Column("category", nullable=False, coerce=True),
    },
    strict=False,
    coerce=True,
    name="observed",
)


GENERATED_SCHEMA = pa.DataFrameSchema(
    {
        "person_id": _PERSON_ID,
        "seed": pa.Column("int32", nullable=False, coerce=True),
        "age_start": pa.Column(
            "int32",
            checks=[pa.Check.ge(0)],
            nullable=False,
            coerce=True,
        ),
        "age_stop": pa.Column("int32", nullable=False, coerce=True),
        "age": pa.Column(
            "int32",
            checks=[pa.Check.in_range(0, _MAX_AGE_DAYS)],
            nullable=False,
            coerce=True,
        ),
        "event": pa.Column("category", nullable=False, coerce=True),
    },
    checks=[
        # Observation window is well-ordered: t1 <= t2.
        pa.Check(
            lambda df: df["age_start"] <= df["age_stop"],
            error="age_start must be <= age_stop (start of window after its end)",
        ),
        # Generated rows are strictly in the future of the jump-off (00 section 4.1: "generated
        # rows have age > age_stop"). This is the defining invariant of a generated artifact.
        pa.Check(
            lambda df: df["age"] > df["age_stop"],
            error="generated rows must have age > age_stop (row at or before the jump-off)",
        ),
    ],
    strict=False,
    coerce=True,
    name="generated",
)


PERSONS_SCHEMA = pa.DataFrameSchema(
    {
        "person_id": pa.Column(nullable=False, unique=True, coerce=False, required=True),
        "birth_year": pa.Column(
            "int16",
            checks=[pa.Check.in_range(1850, 2100)],
            nullable=False,
            coerce=True,
        ),
        "sex": pa.Column("category", nullable=True, coerce=True, required=False),
    },
    # Declared covariates are ordinary extra columns; do not reject them.
    strict=False,
    coerce=True,
    name="persons",
)


EVENT_DEFINITIONS_SCHEMA = pa.DataFrameSchema(
    {
        "model_representation": pa.Column(nullable=False, unique=True, coerce=False),
        "event_definition": pa.Column(str, nullable=False, coerce=True),
    },
    # Exactly the two cosmetic columns (00 section 4.1) — reject anything else.
    strict=True,
    coerce=True,
    name="event_definitions",
)


# --- validation entry point ---------------------------------------------------------------------
_FIX_HINTS = {
    "observed": "expected columns person_id, age (int32 days, 0..110y), event (category).",
    "generated": (
        "expected person_id, seed (int32), age_start/age_stop/age (int32 days) with "
        "0 <= age_start <= age_stop < age."
    ),
    "persons": "expected unique person_id, birth_year (int16 in 1850..2100), optional sex.",
    "event_definitions": "expected exactly columns (model_representation, event_definition).",
}


def validate(df: pd.DataFrame, schema: pa.DataFrameSchema, artifact: str) -> pd.DataFrame:
    """Validate ``df`` against ``schema``, re-raising failures as :class:`SchemaError`.

    Parameters
    ----------
    df : pandas.DataFrame
        The already-normalized (day-valued) candidate frame.
    schema : pandera.pandas.DataFrameSchema
        One of the module-level schemas.
    artifact : str
        Human name of the artifact, used in the error message (e.g. ``"generated"``).

    Returns
    -------
    pandas.DataFrame
        The coerced, validated frame.

    Raises
    ------
    SchemaError
        If the frame violates the schema. The message names the artifact, the failing
        column/check, and a fix hint.
    """
    try:
        return schema.validate(df, lazy=True)
    except (PanderaSchemaError, PanderaSchemaErrors) as exc:
        hint = _FIX_HINTS.get(artifact, "")
        raise SchemaError(f"{artifact}: schema validation failed. {hint}\n{exc}") from exc
