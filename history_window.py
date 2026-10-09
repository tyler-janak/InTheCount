"""
history_window.py
=================
The single place that decides WHICH historical data a prediction may see.

Rule (player modelling)
-----------------------
A prediction for a game in season S may only use:

    * the previous two seasons, S-2 and S-1   (complete seasons), and
    * season S strictly BEFORE the game being predicted.

Nothing from S-3 or earlier, nothing from the target game itself, nothing
after the prediction cutoff.  Examples:

    2026 prediction -> 2024 + 2025 + 2026-to-date
    2027 prediction -> 2025 + 2026 + 2027-to-date

How it is enforced
------------------
The window is applied to the RAW pitch data **before** any feature is
computed (`window_pitches`).  Feature tables are then built once per
prediction season on that season's window, and only rows that belong to
the prediction season are kept (`hitterspitchers_data.build_windowed_tables`).
Because rolling windows, expanding means, empirical-Bayes true-talent
accumulators, handedness splits, team context and matchup aggregates are all
computed on the already-restricted window, information older than S-2 cannot
reach a season-S row indirectly through any aggregate.

The "strictly before the game" half of the rule is enforced inside the
feature builders (every trailing feature is `shift(1)`-ed within the player
/ team before it is aggregated) and is verified by `tests/test_leakage.py`.

Only MLB competitive games enter the modelling layer: regular season (R) and
postseason (F/D/L/W).  Spring training (S) and exhibitions (E) are excluded -
they are not MLB-level competition, starters throw 1-3 innings by design,
and lineups are mostly minor leaguers, which corrupted early-season
workload / rate features in the previous pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

N_PRIOR_SEASONS = 2
COMPETITIVE_GAME_TYPES = {"R", "F", "D", "L", "W"}
REGULAR_SEASON = {"R"}


def game_season(df: pd.DataFrame, date_col: str = "game_date") -> pd.Series:
    if "game_year" in df.columns:
        y = pd.to_numeric(df["game_year"], errors="coerce")
        if y.notna().all():
            return y.astype(int)
    return pd.to_datetime(df[date_col], errors="coerce").dt.year.astype(int)


def eligible_seasons(prediction_season: int, n_prior: int = N_PRIOR_SEASONS) -> list[int]:
    """Seasons whose data a season-`prediction_season` prediction may touch."""
    return list(range(int(prediction_season) - n_prior, int(prediction_season) + 1))


def competitive_only(pitches: pd.DataFrame) -> pd.DataFrame:
    if "game_type" not in pitches.columns:
        return pitches
    gt = pitches["game_type"].astype("object").fillna("")
    return pitches[gt.isin(COMPETITIVE_GAME_TYPES)]


def window_pitches(pitches: pd.DataFrame, prediction_season: int,
                   n_prior: int = N_PRIOR_SEASONS) -> pd.DataFrame:
    """Restrict raw pitches to the eligible window for `prediction_season`.

    Applied BEFORE feature engineering.  Data later than the prediction
    season is dropped too (a 2025 window must not see 2026)."""
    seasons = set(eligible_seasons(prediction_season, n_prior))
    s = game_season(pitches)
    out = competitive_only(pitches[s.isin(seasons)])
    return out.copy()


def assert_window(pitches: pd.DataFrame, prediction_season: int,
                  n_prior: int = N_PRIOR_SEASONS) -> None:
    """Fail loudly if a frame contains data outside the eligible window."""
    if pitches.empty:
        return
    s = set(game_season(pitches).unique().tolist())
    allowed = set(eligible_seasons(prediction_season, n_prior))
    bad = sorted(s - allowed)
    if bad:
        raise AssertionError(
            f"history window violation for {prediction_season}: seasons {bad} present "
            f"(allowed {sorted(allowed)})")
    if "game_type" in pitches.columns:
        gt = set(pitches["game_type"].astype("object").dropna().unique().tolist())
        extra = sorted(gt - COMPETITIVE_GAME_TYPES)
        if extra:
            raise AssertionError(f"non-competitive game types in window: {extra}")


def describe_window(prediction_season: int, available_seasons: list[int],
                    n_prior: int = N_PRIOR_SEASONS) -> dict:
    want = eligible_seasons(prediction_season, n_prior)
    have = [s for s in want if s in set(available_seasons)]
    return {
        "prediction_season": int(prediction_season),
        "eligible_seasons": want,
        "seasons_available": have,
        "prior_seasons_missing": [s for s in want[:-1] if s not in have],
    }
