"""Descriptives arm: writes expected files; degrades gracefully when persons is missing."""

from __future__ import annotations

import logging

import numpy as np

from seqeval.arms import descriptives as D
from seqeval.arms._common import OutputWriter
from seqeval.config import (
    DescriptivesConfig,
    EventConfig,
    FertilityConfig,
    LifeTableConfig,
    PprConfig,
)
from seqeval.core.specs import TTESpec
from seqeval.io.loaders import Bundle
from tests import synthetic as S

OUTCOMES = {
    "first_birth": TTESpec("birth", 1),
    "second_birth": TTESpec("birth", 2, origin=TTESpec("birth", 1)),
}


def _bundle(with_persons=True):
    rng = np.random.default_rng(0)
    obs, pers = S.simulate_cohort(
        1200, (1965, 1975), S.default_hazards(), None, rng, no_event_fraction=1.0
    )
    return Bundle(
        observed=obs,
        generated=None,
        persons=pers if with_persons else None,
        event_defs=None,
        events=EventConfig(birth="birth"),
    )


def _full_cfg():
    return DescriptivesConfig(
        kaplan_meier=["first_birth", "second_birth"],
        fertility=FertilityConfig(ccf=True, asfr=["period", "cohort"], ppr=PprConfig(max_parity=4)),
        life_table=LifeTableConfig(max_parity=4),
        stratify_by=["cohort"],
    )


def test_arm_writes_all_files(tmp_path):
    out = OutputWriter(base_dir=tmp_path, arm="descriptives", model="demo")
    D.run(_bundle(), _full_cfg(), out, outcomes=OUTCOMES)

    names = {p.name for p in out.written}
    expected = {
        "km_first_birth.parquet",
        "km_second_birth.parquet",
        "ccf.parquet",
        "asfr_period.parquet",
        "asfr_cohort.parquet",
        "tfr.parquet",
        "ppr.parquet",
        "life_table.parquet",
    }
    assert expected <= names
    for path in out.written:
        assert path.exists() and path.stat().st_size > 0


def test_result_frames_stamped_with_model(tmp_path):
    import pandas as pd

    out = OutputWriter(base_dir=tmp_path, arm="descriptives", model="my_model")
    D.run(_bundle(), _full_cfg(), out, outcomes=OUTCOMES)
    ccf = pd.read_parquet(out.dir / "ccf.parquet")
    assert (ccf["model"] == "my_model").all()


def test_km_xlabel_reflects_origin():
    from seqeval.arms.descriptives import _km_xlabel

    assert _km_xlabel(OUTCOMES["first_birth"], OUTCOMES) == "age (years)"
    # a duration measured from an origin event is "years since <origin>", not age
    assert _km_xlabel(OUTCOMES["second_birth"], OUTCOMES) == "years since first birth"


def test_cohort_width_config_produces_five_year_bands(tmp_path):
    import pandas as pd

    cfg = _full_cfg()
    cfg.cohort_width = 5
    out = OutputWriter(base_dir=tmp_path, arm="descriptives", model="demo")
    D.run(_bundle(), cfg, out, outcomes=OUTCOMES)
    cohorts = set(pd.read_parquet(out.dir / "ccf.parquet")["cohort"])
    assert all(c % 5 == 0 for c in cohorts)


def test_missing_persons_skips_cohort_metrics(tmp_path, caplog):
    out = OutputWriter(base_dir=tmp_path, arm="descriptives", model="demo")
    with caplog.at_level(logging.WARNING, logger="seqeval"):
        D.run(_bundle(with_persons=False), _full_cfg(), out, outcomes=OUTCOMES)

    names = {p.name for p in out.written}
    # Age-only metrics still run...
    assert {"km_first_birth.parquet", "ppr.parquet", "life_table.parquet"} <= names
    # ...cohort/period metrics are skipped.
    assert "ccf.parquet" not in names
    assert "asfr_period.parquet" not in names
    assert any("skipping CCF/ASFR" in r.message for r in caplog.records)
