"""Fertility metrics: exact fixtures, synthetic convergence, censoring, key-agnosticism."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from seqeval.core import outcomes as O
from seqeval.core.slicing import AgeBins
from seqeval.metrics import fertility as FE
from seqeval.units import years_to_days as yd
from tests import synthetic as S
from tests.fixtures import tiny

OBS_KEYS = ["person_id"]
GEN_KEYS = ["person_id", "seed", "age_start", "age_stop"]


def _tables(obs, keys=OBS_KEYS):
    return O.births(obs, keys, birth_event="birth"), O.observation_spans(obs, keys)


def test_ccf_exact_on_fixture():
    obs, pers = tiny.observed_fixture(), tiny.persons_fixture()
    births, spans = _tables(obs)
    ccf = FE.ccf(births, spans, pers, by_cohort=False)
    assert ccf.iloc[0]["ccf"] == pytest.approx(tiny.EXPECTED_CCF)


def test_ppr_exact_on_fixture():
    obs = tiny.observed_fixture()
    births, spans = _tables(obs)
    ppr = FE.ppr(births, spans, max_parity=4).set_index("parity_from")["ppr"]
    for (k, _), expected in tiny.EXPECTED_PPR.items():
        assert ppr.loc[k] == pytest.approx(expected)


def test_ccf_converges_to_expected():
    rng = np.random.default_rng(0)
    h = S.default_hazards()
    obs, pers = S.simulate_cohort(5000, (1965, 1970), h, None, rng, no_event_fraction=1.0)
    births, spans = _tables(obs)
    ccf = FE.ccf(births, spans, pers, by_cohort=False)
    assert ccf.iloc[0]["ccf"] == pytest.approx(S.expected_ccf(h), abs=0.1)


def test_cohort_asfr_sums_to_ccf():
    rng = np.random.default_rng(1)
    h = S.default_hazards()
    obs, pers = S.simulate_cohort(5000, (1965, 1970), h, None, rng, no_event_fraction=1.0)
    births, spans = _tables(obs)
    bins = AgeBins.from_years(15, 50, 1)
    af = FE.asfr(births, spans, pers, mode="cohort", bins=bins)
    ccf = FE.ccf(births, spans, pers, by_cohort=True).set_index("cohort")["ccf"]
    asfr_sum = af.groupby("cohort")["asfr"].sum()
    for cohort, s in asfr_sum.items():
        assert s == pytest.approx(ccf.loc[cohort], abs=0.15)


def test_censoring_sets_incomplete_and_shrinks_ppr():
    rng = np.random.default_rng(2)
    h = S.default_hazards()
    full_obs, pers = S.simulate_cohort(3000, (1965, 1970), h, None, rng, no_event_fraction=1.0)
    cens_obs, _ = S.simulate_cohort(3000, (1965, 1970), h, 30.0, rng)

    fb, fs = _tables(full_obs)
    cb, cs = _tables(cens_obs)

    ccf_cens = FE.ccf(cb, cs, pers, by_cohort=True)
    assert not ccf_cens["complete"].any()  # censored at 30 -> no cohort reaches 50

    # Progression to higher parity is throttled by early censoring.
    ppr_full = FE.ppr(fb, fs, max_parity=4).set_index("parity_from")["n_at_risk"]
    ppr_cens = FE.ppr(cb, cs, max_parity=4).set_index("parity_from")["n_at_risk"]
    assert ppr_cens.loc[2] < ppr_full.loc[2]


def test_tfr_is_sum_of_period_asfr():
    rng = np.random.default_rng(3)
    obs, pers = S.simulate_cohort(2000, (1960, 1970), S.default_hazards(), None, rng)
    births, spans = _tables(obs)
    bins = AgeBins.from_years(15, 50, 1)
    ap = FE.asfr(births, spans, pers, mode="period", bins=bins)
    tfr = FE.tfr(ap).set_index("year")["tfr"]
    manual = ap.groupby("year")["asfr"].sum()
    assert np.allclose(tfr.to_numpy(), manual.reindex(tfr.index).to_numpy())


def test_fertility_key_agnostic_on_generated():
    rng = np.random.default_rng(4)
    h = S.default_hazards()
    obs, pers = S.simulate_cohort(400, (1965, 1970), h, None, rng)
    gen = S.simulate_generated(obs, pers, h, [(0.0, 25.0), (0.0, 30.0)], 3, rng)
    gb, gs = _tables(gen, GEN_KEYS)
    extra = ("seed", "age_start", "age_stop")

    ccf = FE.ccf(gb, gs, pers, by_cohort=False, extra_by=extra)
    assert ccf.groupby(list(extra)).ngroups == 6  # 3 seeds x 2 windows
    ppr = FE.ppr(gb, gs, max_parity=3, extra_by=extra)
    assert ppr.groupby(list(extra)).ngroups == 6
    bins = AgeBins.from_years(15, 50, 1)
    af = FE.asfr(gb, gs, pers, mode="cohort", bins=bins, extra_by=extra)
    assert {"seed", "age_start", "age_stop", "cohort", "age_bin"} <= set(af.columns)


def test_cohort_width_groups_into_bands():
    rng = np.random.default_rng(6)
    obs, pers = S.simulate_cohort(2000, (1960, 1974), S.default_hazards(), None, rng)
    births, spans = _tables(obs)
    ccf1 = FE.ccf(births, spans, pers, by_cohort=True, cohort_width=1)
    ccf5 = FE.ccf(births, spans, pers, by_cohort=True, cohort_width=5)
    assert set(ccf5["cohort"]) == {1960, 1965, 1970}  # 5-year bands anchored at the min birth year
    assert len(ccf5) < len(ccf1)  # coarser bands -> fewer groups


def test_ppr_min_exposure_excludes_recent_reachers():
    obs = tiny.observed_fixture()
    births, spans = _tables(obs)
    # Require 5y of exposure after the first birth: p1 (25, ends 25) and p4 (30, ends 30) are
    # excluded (0y after); p2 (24->29), p3 (22->31), p5 (27->33) survive -> 3 at risk.
    ppr = FE.ppr(births, spans, max_parity=2, min_exposure_after_k=yd(5)).set_index("parity_from")
    assert ppr.loc[1, "n_at_risk"] == 3


def test_ccf_variance_splits_seed_noise_from_between_woman_spread():
    """The components add to the total, and each cohort's mean matches :func:`ccf` over seeds."""
    rng = np.random.default_rng(7)
    h = S.default_hazards()
    obs, pers = S.simulate_cohort(400, (1960, 1969), h, None, rng, no_event_fraction=1.0)
    gen = S.simulate_generated(obs, pers, h, [(0.0, 25.0)], 8, rng)
    births, spans = _tables(gen, GEN_KEYS)

    var = FE.ccf_variance(births, spans, pers, cohort_width=5)
    np.testing.assert_allclose(var["within_var"] + var["between_var"], var["total_var"])
    assert (var["within_var"] > 0).all() and (var["between_var"] >= 0).all()
    # the reported centre is the across-seed CCF, so the band it feeds sits on the plotted curve
    per_seed = FE.ccf(births, spans, pers, by_cohort=True, extra_by=("seed",), cohort_width=5)
    np.testing.assert_allclose(
        var.set_index("cohort")["ccf"],
        per_seed.groupby("cohort", observed=True)["ccf"].mean(),
        rtol=1e-12,
    )


def test_ccf_variance_counts_childless_women_in_the_denominator():
    """Spans define the population, so a childless woman lowers the mean and widens the spread."""
    obs, pers = tiny.observed_fixture(), tiny.persons_fixture()
    births, spans = _tables(obs)
    var = FE.ccf_variance(
        births.assign(seed=0), spans.assign(seed=0), pers, cohort_width=100
    ).iloc[0]
    assert var["n_women"] == spans["person_id"].nunique()
    assert var["ccf"] == pytest.approx(tiny.EXPECTED_CCF)
    # one seed: nothing varies across replicates, so the whole spread is between women
    assert var["within_var"] == pytest.approx(0.0)
    assert var["between_var"] == pytest.approx(var["total_var"])


def test_parity_distribution_shares_sum_to_one_per_cohort():
    rng = np.random.default_rng(8)
    obs, pers = S.simulate_cohort(400, (1960, 1969), S.default_hazards(), None, rng,
                                  no_event_fraction=1.0)
    births, spans = _tables(obs)
    par = FE.parity_distribution(births.assign(seed=0), spans.assign(seed=0), pers, cohort_width=5)
    totals = par.groupby("cohort")["share"].sum()
    np.testing.assert_allclose(totals, 1.0, atol=1e-12)


def test_parity_distribution_mean_reproduces_the_ccf():
    """The outcome layer and the inference layer describe one quantity, not two.

    Stated on the unsuppressed table: withholding a thin cell removes its contribution, so the
    identity is a property of the distribution, not of what is publishable from it.
    """
    rng = np.random.default_rng(9)
    h = S.default_hazards()
    obs, pers = S.simulate_cohort(400, (1960, 1969), h, None, rng, no_event_fraction=1.0)
    gen = S.simulate_generated(obs, pers, h, [(0.0, 25.0)], 6, rng)
    births, spans = _tables(gen, GEN_KEYS)

    # max_parity above the largest count present, so nothing is top-coded away
    par = FE.parity_distribution(births, spans, pers, max_parity=20, cohort_width=5, min_cell=0)
    var = FE.ccf_variance(births, spans, pers, cohort_width=5).set_index("cohort")["ccf"]
    mean = par.assign(w=par["parity"] * par["share"]).groupby("cohort")["w"].sum()
    np.testing.assert_allclose(mean, var.reindex(mean.index), rtol=1e-12)


def test_suppression_costs_the_published_table_some_of_its_mass():
    """A withheld cell is genuinely gone: the published shares no longer sum to one."""
    rng = np.random.default_rng(10)
    h = S.default_hazards()
    obs, pers = S.simulate_cohort(400, (1960, 1969), h, None, rng, no_event_fraction=1.0)
    gen = S.simulate_generated(obs, pers, h, [(0.0, 25.0)], 6, rng)
    births, spans = _tables(gen, GEN_KEYS)

    # a long parity tail gives thin cells at the top end
    par = FE.parity_distribution(births, spans, pers, max_parity=20, cohort_width=5)
    assert par["suppressed"].any()
    assert (par.groupby("cohort")["share"].sum() < 1.0).all()


def test_parity_distribution_top_codes_the_final_category():
    obs, pers = tiny.observed_fixture(), tiny.persons_fixture()
    births, spans = _tables(obs)
    kw = dict(cohort_width=100, min_cell=0)
    full = FE.parity_distribution(births.assign(seed=0), spans.assign(seed=0), pers,
                                  max_parity=20, **kw)
    capped = FE.parity_distribution(births.assign(seed=0), spans.assign(seed=0), pers,
                                    max_parity=2, **kw)
    at_cap = capped[capped["parity"] == 2]["share"].sum()
    assert at_cap == pytest.approx(full[full["parity"] >= 2]["share"].sum())
    assert capped["parity"].max() == 2  # nothing above the cap survives as its own category


def test_parity_distribution_weights_each_woman_equally_across_seeds():
    """A woman with more replicates must not count for more of the population."""
    obs, pers = tiny.observed_fixture(), tiny.persons_fixture()
    births, spans = _tables(obs)
    ragged_b = pd.concat([births.assign(seed=s) for s in (0, 1)])
    ragged_s = pd.concat([spans.assign(seed=s) for s in (0, 1)])
    # person 1 gets a third replicate; nobody else does
    extra = spans[spans["person_id"] == 1].assign(seed=2)
    ragged_s = pd.concat([ragged_s, extra])
    ragged_b = pd.concat([ragged_b, births[births["person_id"] == 1].assign(seed=2)])

    par = FE.parity_distribution(ragged_b, ragged_s, pers, max_parity=20, cohort_width=100,
                                 min_cell=0)
    assert par["n_women_equiv"].sum() == pytest.approx(spans["person_id"].nunique())


def test_parity_distribution_counts_childless_women_in_the_denominator():
    obs, pers = tiny.observed_fixture(), tiny.persons_fixture()
    births, spans = _tables(obs)
    par = FE.parity_distribution(births.assign(seed=0), spans.assign(seed=0), pers,
                                 cohort_width=100, min_cell=0)
    assert par["n_women_total"].iloc[0] == spans["person_id"].nunique()
    # the mean over the distribution is the CCF, which only holds if the childless are counted
    mean = (par["parity"] * par["share"]).sum()
    assert mean == pytest.approx(tiny.EXPECTED_CCF)


def test_parity_distribution_withholds_thin_cells():
    obs, pers = tiny.observed_fixture(), tiny.persons_fixture()
    births, spans = _tables(obs)
    par = FE.parity_distribution(births.assign(seed=0), spans.assign(seed=0), pers,
                                 cohort_width=100)
    assert par["suppressed"].any()  # a 5-woman fixture cannot publish a per-parity breakdown
    assert par.loc[par["suppressed"], "share"].isna().all()
