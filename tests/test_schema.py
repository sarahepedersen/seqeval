"""Schema: valid frames pass; each violation class raises SchemaError (01 section 9)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from seqeval.io.schema import (
    EVENT_DEFINITIONS_SCHEMA,
    GENERATED_SCHEMA,
    OBSERVED_SCHEMA,
    PERSONS_SCHEMA,
    SchemaError,
    validate,
)
from seqeval.units import years_to_days
from tests.fixtures import tiny


def _observed_ok():
    return tiny.observed_fixture()


def _generated_ok():
    df = pd.DataFrame(
        {
            "person_id": np.array([1, 1], dtype=np.int64),
            "seed": np.array([0, 1], dtype=np.int32),
            "age_start": np.array([0, 0], dtype=np.int32),
            "age_stop": np.array([years_to_days(25)] * 2, dtype=np.int32),
            "age": np.array([years_to_days(27)] * 2, dtype=np.int32),
            "event": pd.Categorical(["birth", "birth"]),
        }
    )
    return df


def test_valid_frames_pass():
    validate(_observed_ok(), OBSERVED_SCHEMA, "observed")
    validate(_generated_ok(), GENERATED_SCHEMA, "generated")
    validate(tiny.persons_fixture(), PERSONS_SCHEMA, "persons")
    validate(
        pd.DataFrame({"model_representation": ["birth"], "event_definition": ["live birth"]}),
        EVENT_DEFINITIONS_SCHEMA,
        "event_definitions",
    )


def test_age_out_of_range_rejected():
    df = _observed_ok()
    df.loc[0, "age"] = years_to_days(200)
    with pytest.raises(SchemaError, match="observed"):
        validate(df, OBSERVED_SCHEMA, "observed")


def test_generated_age_not_after_jumpoff_rejected():
    df = _generated_ok()
    df.loc[0, "age"] = df.loc[0, "age_stop"]  # age == age_stop violates age > age_stop
    with pytest.raises(SchemaError, match="age > age_stop"):
        validate(df, GENERATED_SCHEMA, "generated")


def test_generated_start_after_stop_rejected():
    df = _generated_ok()
    df["age_start"] = df["age_stop"] + 1
    with pytest.raises(SchemaError, match="age_start"):
        validate(df, GENERATED_SCHEMA, "generated")


def test_persons_duplicate_id_rejected():
    df = tiny.persons_fixture()
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    with pytest.raises(SchemaError, match="persons"):
        validate(df, PERSONS_SCHEMA, "persons")


def test_persons_birth_year_out_of_range_rejected():
    df = tiny.persons_fixture()
    df.loc[0, "birth_year"] = 1000
    with pytest.raises(SchemaError, match="persons"):
        validate(df, PERSONS_SCHEMA, "persons")


def test_event_definitions_extra_column_rejected():
    df = pd.DataFrame(
        {
            "model_representation": ["birth"],
            "event_definition": ["live birth"],
            "extra": ["x"],
        }
    )
    with pytest.raises(SchemaError, match="event_definitions"):
        validate(df, EVENT_DEFINITIONS_SCHEMA, "event_definitions")


def test_event_definitions_duplicate_token_rejected():
    df = pd.DataFrame({"model_representation": ["birth", "birth"], "event_definition": ["a", "b"]})
    with pytest.raises(SchemaError, match="event_definitions"):
        validate(df, EVENT_DEFINITIONS_SCHEMA, "event_definitions")
