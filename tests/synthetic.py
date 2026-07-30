"""Synthetic cohort generator built on a known piecewise-constant hazard model.

Later modules test metrics against *converged truth* and test calibration against a "perfect
model", so the ground truth must be knowable. Hazards are specified in **years** (human-readable);
every emitted frame is canonical (ages in ``int32`` days, ``event`` category), so the synthetic
data flows straight into the loaders' schemas.

Two event tokens are used: :data:`BIRTH_TOKEN` for a birth and :data:`NO_EVENT_TOKEN` for a
trailing time-marker row. The marker is an ordinary row (no padding concept, 00 section 4.2); it
exists so the last-age span convention is exercised by realistic input, and so that every person
and every generated run appears in the data even with zero births.

Histories run to the top of the fertile range by default, so every sequence is complete and a
generated future is always a backtest of history the observed file already holds. Passing
``observation_year`` to :func:`simulate_cohort` instead censors each person at
``observation_year - birth_year``: young cohorts end mid-life course (**unfinished** sequences),
and with ``simulate_generated(..., require_observed_prefix=True)`` their trajectories past the
jump-off are a real forecast, with no observed truth to score them against.

Deviations from 01's signatures (noted per 00 section 7): ``simulate_cohort`` returns
``(observed, persons)`` — the two frames named in its docstring, not three; ``simulate_generated``
returns the single generated frame it produces. These match what downstream tests actually consume.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from seqeval.units import DAYS_PER_YEAR

#: Raw event token for a birth in synthetic data.
BIRTH_TOKEN = "birth"
#: Raw event token for a trailing "no event" time-marker row.
NO_EVENT_TOKEN = "no_event"

_CACHE_DIR = Path(__file__).parent / ".cache"


@dataclass
class HazardSpec:
    """A piecewise-constant birth-hazard model, the known truth behind synthetic data.

    Parameters
    ----------
    rates : dict[tuple[float, float, int], float]
        Birth hazard per (``lo_age_yr``, ``hi_age_yr``, ``parity``) band, in events per year.
        ``parity`` is the parity *before* the birth (0 = hazard of the first birth).
    max_parity : int, default 6
        Maximum number of births any person can have.
    fertile_ages : tuple[float, float], default (15.0, 50.0)
        Ages (years) outside which the hazard is zero and histories are not simulated.
    """

    rates: dict[tuple[float, float, int], float]
    max_parity: int = 6
    fertile_ages: tuple[float, float] = (15.0, 50.0)


def default_hazards() -> HazardSpec:
    """A roughly Denmark-like schedule: peak in the late 20s/early 30s, strong 1->2 progression."""
    bands = [(15, 20), (20, 25), (25, 30), (30, 35), (35, 40), (40, 45), (45, 50)]
    by_parity = {
        0: [0.02, 0.10, 0.18, 0.15, 0.06, 0.010, 0.0020],
        1: [0.01, 0.10, 0.22, 0.18, 0.07, 0.010, 0.0020],
        2: [0.00, 0.04, 0.10, 0.10, 0.05, 0.008, 0.0010],
        3: [0.00, 0.02, 0.05, 0.05, 0.03, 0.005, 0.0005],
        4: [0.00, 0.01, 0.02, 0.02, 0.01, 0.002, 0.0002],
        5: [0.00, 0.005, 0.01, 0.01, 0.005, 0.001, 0.0001],
    }
    rates: dict[tuple[float, float, int], float] = {}
    for parity, row in by_parity.items():
        for (lo, hi), rate in zip(bands, row, strict=True):
            rates[(float(lo), float(hi), parity)] = rate
    return HazardSpec(rates=rates, max_parity=6, fertile_ages=(15.0, 50.0))


def perturb(hazards: HazardSpec, factor: float) -> HazardSpec:
    """Return a copy of ``hazards`` with every rate scaled by ``factor`` (a mis-calibrated model).

    Scaling all hazards up/down shifts the whole schedule, making a deliberately mis-calibrated
    reference model for calibration tests.
    """
    return HazardSpec(
        rates={k: v * factor for k, v in hazards.rates.items()},
        max_parity=hazards.max_parity,
        fertile_ages=hazards.fertile_ages,
    )


# =================================================================================================
# hazard sampling (vectorized, piecewise-exponential)
# =================================================================================================
def _bands_for_parity(hazards: HazardSpec, parity: int) -> list[tuple[float, float, float]]:
    """Sorted ``(lo, hi, rate)`` bands for a parity, clipped to the fertile age range."""
    lo_f, hi_f = hazards.fertile_ages
    bands = [
        (max(lo, lo_f), min(hi, hi_f), rate)
        for (lo, hi, p), rate in hazards.rates.items()
        if p == parity
    ]
    return sorted((lo, hi, rate) for lo, hi, rate in bands if hi > lo)


def _sample_next_birth(
    age_yr: np.ndarray,
    parity: int,
    hazards: HazardSpec,
    horizon: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample the next birth age (years) for each row, ``inf`` if none occurs before ``horizon``.

    Inverse-transform sampling on a piecewise-exponential: draw a target cumulative hazard
    ``E ~ Exp(1)`` and walk the age bands, consuming ``rate * span`` from ``E`` in each until it is
    exhausted (a birth) or the horizon is reached (no birth). Vectorized across rows; the only loop
    is over the handful of age bands. ``horizon`` is per row, so rows censored at different ages
    (e.g. by a common observation year) are simulated in the same pass.
    """
    n = len(age_yr)
    remaining = rng.exponential(1.0, size=n)
    age = age_yr.astype(float).copy()
    result = np.full(n, np.inf)
    done = np.zeros(n, dtype=bool)

    for lo, hi, rate in _bands_for_parity(hazards, parity):
        hi_eff = np.minimum(hi, horizon)
        start = np.maximum(age, lo)
        span = hi_eff - start
        in_band = (~done) & (span > 0)
        if not in_band.any():
            continue
        if rate > 0:
            capacity = rate * span
            fires = in_band & (remaining <= capacity)
            result[fires] = start[fires] + remaining[fires] / rate
            done[fires] = True
            consume = in_band & ~fires
            remaining[consume] -= capacity[consume]
        # Advance anyone still searching up to the top of this band.
        adv = (~done) & (age < hi_eff)
        age[adv] = hi_eff[adv]

    return result


def _simulate_births(
    current_age_yr: np.ndarray,
    start_parity: np.ndarray,
    hazards: HazardSpec,
    horizon: float | np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate all births for a set of rows from their current age/parity to ``horizon``.

    ``horizon`` is a scalar (everyone followed to the same age) or one value per row (heterogeneous
    censoring). Returns ``(row_index, birth_age_yr)`` for every birth. Vectorized: the outer loop is
    over parity levels (bounded by ``max_parity``); at level ``p`` only rows currently at parity
    ``p`` and still fertile are sampled, so heterogeneous starting parities are handled in one pass.
    """
    current = current_age_yr.astype(float).copy()
    horizon = np.broadcast_to(np.asarray(horizon, dtype=float), current.shape)
    parity = start_parity.astype(int).copy()
    alive = np.ones(len(current), dtype=bool)
    out_idx: list[np.ndarray] = []
    out_age: list[np.ndarray] = []

    for p in range(hazards.max_parity):
        active = alive & (parity == p)
        if not active.any():
            continue
        idx = np.nonzero(active)[0]
        nxt = _sample_next_birth(current[idx], p, hazards, horizon[idx], rng)
        got = np.isfinite(nxt)
        fired = idx[got]
        out_idx.append(fired)
        out_age.append(nxt[got])
        current[fired] = nxt[got]
        parity[fired] = p + 1
        alive[idx[~got]] = False  # no birth at this parity -> no further births

    if out_idx:
        return np.concatenate(out_idx), np.concatenate(out_age)
    return np.array([], dtype=int), np.array([], dtype=float)


def _to_days(age_yr: np.ndarray) -> np.ndarray:
    """Convert year-valued ages to canonical ``int32`` days (round-to-nearest, matches units)."""
    return np.rint(np.asarray(age_yr, dtype=float) * DAYS_PER_YEAR).astype(np.int32)


# =================================================================================================
# public generators
# =================================================================================================
def simulate_cohort(
    n: int,
    birth_years: tuple[int, int],
    hazards: HazardSpec,
    censor_age_yr: float | None,
    rng: np.random.Generator,
    no_event_fraction: float = 0.3,
    observation_year: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate ``n`` observed fertility histories and their persons frame.

    Parameters
    ----------
    n : int
        Number of persons.
    birth_years : tuple[int, int]
        Inclusive range to draw each person's ``birth_year`` from.
    hazards : HazardSpec
        The ground-truth model.
    censor_age_yr : float or None
        Right-censor everyone at this age; ``None`` runs full histories to the top of the fertile
        range.
    rng : numpy.random.Generator
        Source of randomness (determinism contract, 00 section 6).
    no_event_fraction : float, default 0.3
        Fraction of *birth-having* persons that also get a trailing no-event marker row (persons
        with zero births always get one, so every person appears).
    observation_year : int or None, default None
        Calendar year the data were collected. When given, each person is additionally censored at
        age ``observation_year - birth_year``, so young cohorts end mid-life course: their
        sequences are **unfinished** and the fertile years past the censoring age are a genuine
        future for the forecasting arm to predict. Older cohorts are unaffected (their censoring
        age is above the fertile range). Unfinished persons always get a trailing marker row, so
        the observation span records where the data stop rather than at the last birth.

    Returns
    -------
    (observed, persons) : tuple of pandas.DataFrame
        Both schema-conformant (ages in int32 days).
    """
    full_horizon = censor_age_yr if censor_age_yr is not None else hazards.fertile_ages[1]
    person_id = np.arange(n, dtype=np.int64)

    def draw_birth_years() -> np.ndarray:
        return rng.integers(birth_years[0], birth_years[1] + 1, size=n).astype(np.int16)

    # Birth years are drawn before the histories only when they set the censoring age; without
    # ``observation_year`` they are drawn after, so the RNG stream (and every fixed-seed
    # expectation built on it) is unchanged by this option existing.
    birth_year = draw_birth_years() if observation_year is not None else None
    if birth_year is None:
        horizon = np.full(n, float(full_horizon))
    else:
        horizon = np.clip(observation_year - birth_year.astype(float), 0.0, full_horizon)
    unfinished = horizon < full_horizon

    idx, birth_age_yr = _simulate_births(np.zeros(n), np.zeros(n, dtype=int), hazards, horizon, rng)
    birth_pid = person_id[idx]
    birth_days = _to_days(birth_age_yr)

    # Trailing no-event markers: always for zero-birth and unfinished persons (the latter need the
    # censoring age on the record), a random fraction of the rest.
    n_births = np.bincount(idx, minlength=n)
    last_birth_day = np.zeros(n, dtype=np.int64)
    if len(idx):
        np.maximum.at(last_birth_day, idx, birth_days.astype(np.int64))
    end_days = np.rint(horizon * DAYS_PER_YEAR).astype(np.int64)
    has_birth = n_births > 0
    chosen = rng.random(n) < no_event_fraction
    marker = ((~has_birth) | unfinished | chosen) & ((end_days > last_birth_day) | ~has_birth)

    marker_pid = person_id[marker]
    marker_days = end_days[marker].astype(np.int32)

    observed = pd.DataFrame(
        {
            "person_id": np.concatenate([birth_pid, marker_pid]),
            "age": np.concatenate([birth_days, marker_days]),
            "event": np.concatenate(
                [np.full(birth_pid.shape, BIRTH_TOKEN), np.full(marker_pid.shape, NO_EVENT_TOKEN)]
            ),
        }
    )
    observed["age"] = observed["age"].astype(np.int32)
    observed["event"] = observed["event"].astype("category")
    observed = observed.sort_values(["person_id", "age"]).reset_index(drop=True)

    persons = pd.DataFrame(
        {
            "person_id": person_id,
            "birth_year": birth_year if birth_year is not None else draw_birth_years(),
            "sex": pd.Categorical(np.full(n, "F")),
            "education": pd.Categorical(rng.choice(["low", "high"], size=n)),
            "region": pd.Categorical(rng.choice(["A", "B"], size=n)),
            # Age (days) each person is followed to. Not part of the persons schema — an extra
            # column seqeval ignores — but it is the exact follow-up, which the last observed row
            # only bounds from below (a person with no trailing marker ends at their last birth).
            "observed_through": end_days.astype(np.int32),
        }
    )
    return observed, persons


def simulate_generated(
    observed: pd.DataFrame,
    persons: pd.DataFrame,
    hazards: HazardSpec,
    windows_yr: list[tuple[float, float]],
    n_seeds: int,
    rng: np.random.Generator,
    require_observed_prefix: bool = False,
) -> pd.DataFrame:
    """Simulate a "perfect model": futures drawn from the same hazards, conditioned on true parity.

    For each (person, window, seed) the person's true parity at ``age_stop`` is computed from the
    observed births, and the future is simulated from ``age_stop`` to the top of the fertile range
    using the same ground-truth hazards. Across many seeds the empirical birth probabilities must
    converge to the true probabilities — the gold standard for calibration tests. A trailing
    no-event marker is emitted for every run so runs with no future births still appear.

    Parameters
    ----------
    require_observed_prefix : bool, default False
        When True, a person is only given runs at jump-offs their observed sequence actually
        reaches (last observed age >= ``age_stop``). Pair this with ``simulate_cohort``'s
        ``observation_year``: a person censored at age 30 is then forecast only from jump-offs at
        or before 30, and everything after is a real forecast rather than a backtest of history the
        file already contains. Leave False (the default) for fully observed cohorts, where every
        person is eligible at every window.

    Returns
    -------
    generated : pandas.DataFrame
        Schema-conformant (ages in int32 days; ``age > age_stop`` for every row).
    """
    horizon = hazards.fertile_ages[1]
    all_person_id = persons["person_id"].to_numpy()

    births = observed.loc[observed["event"] == BIRTH_TOKEN, ["person_id", "age"]]
    # Follow-up per person: the exact censoring age when ``simulate_cohort`` recorded it, else the
    # last observed row (a lower bound on how far the person was followed).
    if "observed_through" in persons.columns:
        obs_end = persons["observed_through"].to_numpy()
    else:
        obs_end = (
            observed.groupby("person_id")["age"]
            .max()
            .reindex(all_person_id, fill_value=-1)
            .to_numpy()
        )

    frames: list[pd.DataFrame] = []
    for start_yr, stop_yr in windows_yr:
        start_day = int(round(start_yr * DAYS_PER_YEAR))
        stop_day = int(round(stop_yr * DAYS_PER_YEAR))
        if stop_day >= int(round(horizon * DAYS_PER_YEAR)):
            continue  # no future to simulate past the fertile range

        person_id = all_person_id[obs_end >= stop_day] if require_observed_prefix else all_person_id
        n = len(person_id)
        if n == 0:
            continue  # nobody is observed this far — the window has no runs

        # True parity at the jump-off = births with age <= stop_day.
        parity_at_stop = (
            births[births["age"] <= stop_day]
            .groupby("person_id")
            .size()
            .reindex(person_id, fill_value=0)
            .to_numpy()
        )

        # Tile persons across seeds.
        pid_run = np.repeat(person_id, n_seeds)
        seed_run = np.tile(np.arange(n_seeds, dtype=np.int32), n)
        parity_run = np.repeat(parity_at_stop, n_seeds).astype(int)
        current_run = np.full(pid_run.shape, stop_yr)

        idx, birth_age_yr = _simulate_births(current_run, parity_run, hazards, horizon, rng)
        birth_days = _to_days(birth_age_yr)
        # A future is strictly after the jump-off: drop births that round back onto age_stop.
        keep = birth_days > stop_day
        idx, birth_days = idx[keep], birth_days[keep]

        end_day = int(round(horizon * DAYS_PER_YEAR))
        n_runs = len(pid_run)
        # Births for this window.
        gen = pd.DataFrame(
            {
                "person_id": pid_run[idx],
                "seed": seed_run[idx],
                "age_start": np.int32(start_day),
                "age_stop": np.int32(stop_day),
                "age": birth_days,
                "event": np.full(len(idx), BIRTH_TOKEN),
            }
        )
        # One trailing no-event marker per run (guarantees every run is represented).
        markers = pd.DataFrame(
            {
                "person_id": pid_run,
                "seed": seed_run,
                "age_start": np.int32(start_day),
                "age_stop": np.int32(stop_day),
                "age": np.full(n_runs, end_day, dtype=np.int32),
                "event": np.full(n_runs, NO_EVENT_TOKEN),
            }
        )
        frames.append(pd.concat([gen, markers], ignore_index=True))

    generated = pd.concat(frames, ignore_index=True)
    generated["person_id"] = generated["person_id"].astype(np.int64)
    for col in ("seed", "age_start", "age_stop", "age"):
        generated[col] = generated[col].astype(np.int32)
    generated["event"] = generated["event"].astype("category")
    return generated.sort_values(["person_id", "age_start", "age_stop", "seed", "age"]).reset_index(
        drop=True
    )


# =================================================================================================
# analytic reference values
# =================================================================================================
def _hazard_key(hazards: HazardSpec, censor_age_yr: float | None) -> str:
    payload = {
        "rates": sorted((list(k), v) for k, v in hazards.rates.items()),
        "max_parity": hazards.max_parity,
        "fertile_ages": list(hazards.fertile_ages),
        "censor": censor_age_yr,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def expected_ccf(
    hazards: HazardSpec, censor_age_yr: float | None = None, n: int = 200_000
) -> float:
    """Completed cohort fertility (mean births per person) under ``hazards``, by large-n MC.

    Uses a fixed RNG seed and an on-disk cache (``tests/.cache``) so the expensive simulation runs
    once. This is the converged truth that ``simulate_cohort``'s empirical CCF must approach.
    """
    _CACHE_DIR.mkdir(exist_ok=True)
    cache = _CACHE_DIR / f"ccf_{_hazard_key(hazards, censor_age_yr)}.json"
    if cache.exists():
        return json.loads(cache.read_text())["ccf"]

    rng = np.random.default_rng(20240517)
    horizon = censor_age_yr if censor_age_yr is not None else hazards.fertile_ages[1]
    idx, _ = _simulate_births(np.zeros(n), np.zeros(n, dtype=int), hazards, horizon, rng)
    ccf = float(len(idx) / n)
    cache.write_text(json.dumps({"ccf": ccf}))
    return ccf
