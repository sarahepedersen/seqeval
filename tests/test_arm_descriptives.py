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
        "ppr.parquet",
    }
    assert expected <= names
    assert {"asfr_cohort.png", "km_first_birth.png", "ccf_uncertainty.png"} <= names
    # The plain CCF-by-cohort and PPR curves are dropped: ccf_uncertainty.png carries the same
    # estimate, and both tables are still written.
    assert not {"ccf.png", "ppr.png"} & names
    # Period fertility is written but never drawn: a calendar-year cell is part observed and part
    # forecast, so neither the year x age surface nor its TFR summary is reported.
    assert not {"asfr_period.png", "tfr.png", "tfr.parquet"} & names
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


def test_cohort_width_param_produces_five_year_bands(tmp_path):
    import pandas as pd

    out = OutputWriter(base_dir=tmp_path, arm="descriptives", model="demo")
    D.run(_bundle(), _full_cfg(), out, outcomes=OUTCOMES, cohort_width=5)
    cohorts = set(pd.read_parquet(out.dir / "ccf.parquet")["cohort"])
    assert all(c % 5 == 0 for c in cohorts)


def test_missing_persons_skips_cohort_metrics(tmp_path, caplog):
    out = OutputWriter(base_dir=tmp_path, arm="descriptives", model="demo")
    with caplog.at_level(logging.WARNING, logger="seqeval"):
        D.run(_bundle(with_persons=False), _full_cfg(), out, outcomes=OUTCOMES)

    names = {p.name for p in out.written}
    # Age-only metrics still run...
    assert {"km_first_birth.parquet", "ppr.parquet"} <= names
    # ...cohort/period metrics are skipped.
    assert "ccf.parquet" not in names
    assert "asfr_period.parquet" not in names
    assert any("skipping CCF/ASFR" in r.message for r in caplog.records)


def _run_with(tmp_path, cfg, bundle=None):
    out = OutputWriter(base_dir=tmp_path, arm="descriptives", model="m")
    D.run(bundle or _bundle(), cfg, out, outcomes=OUTCOMES, cohort_width=5)
    return out


def test_max_cohort_year_excludes_later_births_everywhere(tmp_path):
    """The people are dropped, not just their rows: period metrics shrink with the cohort ones."""
    import pandas as pd

    cfg = _full_cfg()
    full = _run_with(tmp_path / "full", cfg)
    cfg_cut = _full_cfg()
    cfg_cut.max_cohort_year = 1970
    cut = _run_with(tmp_path / "cut", cfg_cut)

    ccf_full = pd.read_parquet(full.dir / "ccf.parquet")
    ccf_cut = pd.read_parquet(cut.dir / "ccf.parquet")
    assert ccf_cut["cohort"].max() <= 1970
    assert ccf_full["cohort"].max() > 1970
    assert ccf_cut["n_women"].sum() < ccf_full["n_women"].sum()

    # the period surface is computed on the same restricted population, not the full one
    asfr_full = pd.read_parquet(full.dir / "asfr_period.parquet")
    asfr_cut = pd.read_parquet(cut.dir / "asfr_period.parquet")
    assert asfr_cut["person_years"].sum() < asfr_full["person_years"].sum()


def test_max_cohort_year_leaves_an_untouched_population_alone(tmp_path):
    """A cutoff above everyone's birth year changes nothing."""
    import pandas as pd

    cfg = _full_cfg()
    cfg.max_cohort_year = 2100
    cut = pd.read_parquet(_run_with(tmp_path / "cut", cfg).dir / "ccf.parquet")
    base = pd.read_parquet(_run_with(tmp_path / "base", _full_cfg()).dir / "ccf.parquet")
    pd.testing.assert_frame_equal(cut, base)


def test_max_cohort_year_without_persons_is_reported_not_silently_ignored(tmp_path, caplog):
    cfg = DescriptivesConfig(kaplan_meier=["first_birth"], max_cohort_year=1970)
    with caplog.at_level(logging.WARNING, logger="seqeval"):
        _run_with(tmp_path, cfg, bundle=_bundle(with_persons=False))
    assert any("max_cohort_year" in r.message for r in caplog.records)


def test_ccf_gains_a_parity_distribution_and_uncertainty_figure(tmp_path):
    """The observed CCF gets the same two-panel contrast the backtest draws."""
    import pandas as pd

    out = _run_with(tmp_path, _full_cfg())
    names = {p.name for p in out.written}
    assert {"ccf_uncertainty.png", "parity_distribution.parquet"} <= names

    par = pd.read_parquet(out.dir / "parity_distribution.parquet")
    assert "person_id" not in par.columns
    shares = par[~par["suppressed"]].groupby("cohort")["share"].sum()
    assert (shares <= 1.0 + 1e-9).all()

    # the observed CCF table now carries the variance behind its own interval
    ccf = pd.read_parquet(out.dir / "ccf.parquet")
    assert {"within_var", "between_var", "total_var"} <= set(ccf.columns)
    # nothing is replicated in observed data, so there is no inference noise to report
    assert (ccf["within_var"] == 0).all()
    assert np.allclose(ccf["between_var"], ccf["total_var"])


def test_observed_parity_distribution_mean_is_the_observed_ccf(tmp_path):
    """The two panels describe one quantity: the bars average to the marker."""
    import pandas as pd

    from seqeval.core import outcomes as O
    from seqeval.metrics import fertility as FE

    bundle = _bundle()
    births = O.births(bundle.observed, ["person_id"], birth_event="birth")
    spans = O.observation_spans(bundle.observed, ["person_id"])
    par = FE.parity_distribution(
        births, spans, bundle.persons, max_parity=20, cohort_width=5, min_cell=0
    )
    ccf = FE.ccf(births, spans, bundle.persons, by_cohort=True, cohort_width=5)
    mean = par.assign(w=par["parity"] * par["share"]).groupby("cohort")["w"].sum()
    pd.testing.assert_series_equal(
        mean, ccf.set_index("cohort")["ccf"], check_names=False, rtol=1e-12
    )
