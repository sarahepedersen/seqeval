"""Extract a persons file (person_id, birth_year) from sequences that encode birth year.

Dataset-specific utility, deliberately **outside** the ``seqeval`` package: no CLI subcommand, no
README mention (00 section 2 / 01 section 6). The framework's contract is simply "provide a persons
file"; this script is one way to satisfy it for a family of models whose sequences carry a calendar
token (e.g. ``YEAR_1987`` at age 0). Other datasets will produce their persons file some other way.

Run as a script:

    python examples/make_persons.py observed.parquet persons.parquet --pattern 'YEAR_(\\d{4})'
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger("seqeval.examples")


def persons_from_sequences(
    observed: pd.DataFrame,
    *,
    pattern: str = r"YEAR_(\d{4})",
    token_map: dict | None = None,
    allow_missing: bool = False,
) -> pd.DataFrame:
    """Build a persons-schema frame by extracting a birth year from each person's sequence.

    Parameters
    ----------
    observed : pandas.DataFrame
        Canonical observed frame (needs ``person_id`` and ``event``).
    pattern : str, default ``r"YEAR_(\\d{4})"``
        Regex with one capture group yielding a 4-digit year, matched against the string form of
        ``event``. Ignored when ``token_map`` is given.
    token_map : dict or None, default None
        Optional explicit ``{raw_token: birth_year}`` map, used instead of the regex when the
        birth-year token is not self-describing.
    allow_missing : bool, default False
        If a person has no birth-year token: raise (listing offending ids) when False; drop them
        with a logged count when True.

    Returns
    -------
    pandas.DataFrame
        Columns ``person_id`` and ``birth_year`` (int16), one row per retained person.
    """
    df = observed[["person_id", "event"]].copy()
    event_str = df["event"].astype(str)

    if token_map is not None:
        tm = {str(k): int(v) for k, v in token_map.items()}
        df["birth_year"] = event_str.map(tm)
    else:
        df["birth_year"] = pd.to_numeric(event_str.str.extract(pattern, expand=False))

    found = df.dropna(subset=["birth_year"]).groupby("person_id", sort=True)["birth_year"].first()

    all_ids = pd.Index(observed["person_id"].unique(), name="person_id")
    missing = all_ids.difference(found.index)
    if len(missing):
        if not allow_missing:
            shown = list(missing[:20])
            more = "" if len(missing) <= 20 else f" (and {len(missing) - 20} more)"
            raise ValueError(
                f"{len(missing)} person(s) have no birth-year token matching {pattern!r}: "
                f"{shown}{more}; pass allow_missing=True to drop them"
            )
        logger.warning(
            "persons_from_sequences: dropping %d person(s) with no birth-year token", len(missing)
        )

    out = found.reset_index()
    out["birth_year"] = out["birth_year"].astype(np.int16)
    return out.sort_values("person_id").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("observed", help="path to observed parquet")
    parser.add_argument("out", help="path to write persons parquet")
    parser.add_argument("--pattern", default=r"YEAR_(\d{4})", help="regex with one year group")
    parser.add_argument("--allow-missing", action="store_true", help="drop persons lacking a token")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    observed = pd.read_parquet(args.observed, engine="pyarrow")
    persons = persons_from_sequences(
        observed, pattern=args.pattern, allow_missing=args.allow_missing
    )
    persons.to_parquet(args.out, engine="pyarrow", index=False)
    logger.info("wrote %d persons to %s", len(persons), args.out)


if __name__ == "__main__":
    main()
