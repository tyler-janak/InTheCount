"""
leakage_guard.py
================
Static guard against same-game (post-game) information entering a model.

Every per-game feature table built by `hitterspitchers_data.py` contains BOTH
the game's realised outcome columns (K, K_rate, IP, hitter_k_rate_vs_hand_R,
team_k_rate_vs_hand, ...) - which are the targets / raw material - and the
trailing, pre-game versions of them (`*_last10`, `*_std`, ...).  The original
leakage in this project came from the raw versions being listed as features
(see the "NO-WINDOW ... removed" notes in hitterspitchers_train.py and the
opp_sp_* columns fixed in this audit).

`assert_pregame_features()` is called by every trainer before fitting.  A
feature is accepted only if

  * its name carries a trailing-window marker (`_lastN`, `_std`, ...), or
  * it is on the explicit PREGAME_WHITELIST below, each entry justified.

Anything else raises `LeakageError`, so a same-game column can't be added to a
feature list by accident.  The empirical counterpart (perturbing a game's
outcome and checking that game's features do not move) lives in
tests/test_leakage.py.
"""

from __future__ import annotations

import re

LAGGED_PATTERNS = [
    r"_last\d+(_[RL])?$",          # trailing N-game windows (shift(1) then rolling)
    r"_last\d+_[RL]$",
    r"_std(_[RL])?$",              # season-to-date over PRIOR games
    r"^max_\w+_last\d+$",
    r"^starter_pct_last\d+$",
]

# Explicit pre-game features whose names do not follow the window convention.
PREGAME_WHITELIST = {
    # true-talent: cumulative shrunk rates over prior games (enrich_truetalent.py)
    "p_tt_k", "p_tt_bb", "p_tt_h", "p_tt_hr",
    "h_tt_k", "h_tt_bb", "h_tt_h", "h_tt_hr",
    "lineup_tt_k", "lineup_tt_bb", "lineup_tt_h", "lineup_tt_hr",
    "matchup_k", "matchup_bb", "matchup_h", "matchup_hr",
    # schedule facts known before first pitch
    "days_rest", "days_since_game",
    # static venue context (data/park_factors.csv)
    "park_factor",
    # opposing starter's PREVIOUS start (shifted in enrich_hitter_with_opp_starter)
    "opp_sp_k_rate", "opp_sp_bb_rate", "opp_sp_hr_rate", "opp_sp_h_rate", "opp_sp_ip",
    # lineup aggregates built from batters' trailing _std/_last10 values
    "lineup_k_rate", "lineup_bb_rate", "lineup_h_rate", "lineup_hr_rate",
    "lineup_avg_ev", "lineup_hard_hit_pct",
    # batting-order slot the hitter STARTS in: posted before first pitch (served
    # from the posted lineup); substitutes are blank, never 10 (see build_hitter_games)
    "lineup_spot",
    # game model (build_game_features.py): all pre-game by construction
    "home_field", "same_starter_hand", "diff_starter_hand",
}

# Columns that are KNOWN to hold the current game's outcome. Never features.
SAME_GAME_PATTERNS = [
    r"^(K|BB|HR|H|R|IP|PA|TB|2B|3B|SB|CS|RBI|BF|outs|pitches)$",
    r"^(K|BB|HR|H)_rate$", r"^(h|hr|bb|k)_rate$",
    r"_rate_vs_hand(_[RL])?$",         # un-windowed platoon / team context
    r"^team_(allowed_)?\w+_rate_vs_hand$",
    r"^pitch_pct_",
    r"^(avg|max)_(EV|LA|velocity|spin|direction)$",
    r"_per_(9|PA)$",
    r"^(home|away)_(runs|score|win)$", r"^home_win$",
]


class LeakageError(AssertionError):
    pass


def is_same_game(name: str) -> bool:
    return any(re.search(p, name) for p in SAME_GAME_PATTERNS)


def is_pregame(name: str) -> bool:
    if name in PREGAME_WHITELIST:
        return True
    if any(re.search(p, name) for p in LAGGED_PATTERNS):
        return True
    if name.startswith("diff_"):                    # game model home-minus-away
        return is_pregame_game_feature(name[5:])
    return False


GAME_FEATURE_SUFFIXES = ("_10g", "_14g", "_10", "_season", "_winpct", "_run_diff")


def is_pregame_game_feature(base: str) -> bool:
    # build_game_features.py computes every one of these from games strictly
    # before the game (season_* = season-to-date of PRIOR games)
    return base.endswith(GAME_FEATURE_SUFFIXES) or base.startswith("season_") or base in ("starter_hand",)


def assert_pregame_features(features, context: str = "") -> None:
    bad = [f for f in features if not is_pregame(f) or (f not in PREGAME_WHITELIST and is_same_game(f))]
    if bad:
        raise LeakageError(
            f"{context}: {len(bad)} feature(s) are not provably pre-game: {bad[:20]}. "
            "Use a trailing (_lastN/_std) version or add a justified entry to "
            "leakage_guard.PREGAME_WHITELIST.")
