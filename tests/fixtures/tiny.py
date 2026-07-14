"""Hand-built micro-fixtures: 6 women whose every downstream metric is computable on paper.

Ages are exact year multiples (``A(y) = years_to_days(y)``) so derivations stay readable. Expected
metric values are stored as constants next to the data with their derivations. Later modules import
these to anchor their exact-value tests.

The cohort (births per person in parentheses):

======  ==================  ====================  ===============================================
person  births at (years)   last observed age     note
======  ==================  ====================  ===============================================
p0      (none)              28 (no-event marker)  right-censored childless at 28
p1      25                  25                    parity 1, censored at the birth
p2      24, 29              29                    parity 2
p3      22, 26, 31          31                    parity 3
p4      30                  30                    parity 1
p5      27, 33              33                    parity 2
======  ==================  ====================  ===============================================
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from seqeval.units import years_to_days

#: Event tokens (shared with :mod:`tests.synthetic` so fixtures and synthetic data interoperate).
BIRTH_TOKEN = "birth"
NO_EVENT_TOKEN = "no_event"

#: Alias map a config would carry for this fixture.
EVENTS = {"birth": BIRTH_TOKEN}


def A(years: float) -> int:
    """Age in canonical days for a whole-year age (readability helper)."""
    return years_to_days(years)


# (person_id, event, age_years) — the raw hand-built rows.
_ROWS = [
    (0, NO_EVENT_TOKEN, 28),  # childless, censored at 28
    (1, BIRTH_TOKEN, 25),
    (2, BIRTH_TOKEN, 24),
    (2, BIRTH_TOKEN, 29),
    (3, BIRTH_TOKEN, 22),
    (3, BIRTH_TOKEN, 26),
    (3, BIRTH_TOKEN, 31),
    (4, BIRTH_TOKEN, 30),
    (5, BIRTH_TOKEN, 27),
    (5, BIRTH_TOKEN, 33),
]

# birth_year chosen so cohorts are distinct but simple; sex all female.
_BIRTH_YEARS = {0: 1970, 1: 1970, 2: 1975, 3: 1975, 4: 1980, 5: 1980}


def observed_fixture() -> pd.DataFrame:
    """The 6-person observed frame (schema-conformant, ages in int32 days)."""
    df = pd.DataFrame(
        {
            "person_id": np.array([pid for pid, _, _ in _ROWS], dtype=np.int64),
            "age": np.array([A(yr) for _, _, yr in _ROWS], dtype=np.int32),
            "event": [ev for _, ev, _ in _ROWS],
        }
    )
    df["event"] = df["event"].astype("category")
    return df.sort_values(["person_id", "age"]).reset_index(drop=True)


def persons_fixture() -> pd.DataFrame:
    """The companion persons frame (schema-conformant)."""
    pids = sorted(_BIRTH_YEARS)
    return pd.DataFrame(
        {
            "person_id": np.array(pids, dtype=np.int64),
            "birth_year": np.array([_BIRTH_YEARS[p] for p in pids], dtype=np.int16),
            "sex": pd.Categorical(["F"] * len(pids)),
        }
    )


# =================================================================================================
# expected values (derivations in comments)
# =================================================================================================
#: Completed cohort fertility = mean births per person = (0+1+2+3+1+2)/6 = 9/6.
EXPECTED_CCF = 1.5

#: Parity progression ratios (histories treated as complete).
#:   counts: >=1 birth = 5, >=2 = 3, >=3 = 1, >=4 = 0, of 6 women.
#:   PPR(k->k+1) = n(>=k+1) / n(>=k).
EXPECTED_PPR = {
    (0, 1): 5 / 6,  # 5 of 6 women have a first birth
    (1, 2): 3 / 5,  # 3 of the 5 mothers have a second birth
    (2, 3): 1 / 3,  # 1 of the 3 parity-2+ women has a third
    (3, 4): 0 / 1,  # 0 of the 1 parity-3+ women has a fourth
}

#: Kaplan-Meier survival for time-to-first-birth, as (age_years, S(t)) *just after* each event.
#:   Event/censor times: 22, 24, 25, 27, 28(censor p0), 30. n starts at 6.
#:   S: 22 -> 5/6; 24 -> 5/6*4/5 = 2/3; 25 -> *3/4 = 1/2; 27 -> *2/3 = 1/3;
#:      28 is a censor (p0 leaves; at-risk falls to {p4} only); 30 -> the last woman (p4) has her
#:      first birth, so n_at_risk=1, d=1 and S drops to 0.
EXPECTED_KM_FIRST_BIRTH = [
    (22, 5 / 6),
    (24, 2 / 3),
    (25, 1 / 2),
    (27, 1 / 3),
    (30, 0.0),
]

#: Total person-days observed within the age band [25, 30) years, summed over the 6 women.
#:   Band = [A(25), A(30)) = [9131, 10958) days (width 1827).
#:   Overlap of each woman's [0, last_age] span with the band:
#:     p0 (last 28 = 10227): 10227-9131 = 1096
#:     p1 (last 25 =  9131):                    0
#:     p2 (last 29 = 10592): 10592-9131 = 1461
#:     p3 (last 31, full band):             1827
#:     p4 (last 30 = 10958, full band):     1827
#:     p5 (last 33, full band):             1827
#:   total = 1096 + 0 + 1461 + 1827*3 = 8038
EXPECTED_PERSON_DAYS_25_30 = 8038
