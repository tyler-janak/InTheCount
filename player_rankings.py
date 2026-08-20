"""
player_rankings.py
===================
Season player power rankings for the Power Rankings tab.

HITTERS are ranked by a projected rest-of-season WAR built from three
separate sub-models, weighted together the way real WAR is:

    Bat   — batting runs above average, from a linear-weights model
            (H/2B/3B/HR/BB/outs) applied to this pipeline's own projected
            rest-of-season counting stats. "Above average" is relative to
            the league-average rate computed from this same hitter pool
            (self-consistent — no external league constant needed).
    Def   — fielding runs, from an in-house model that regresses exit
            velocity + launch angle + which fielding zone a ball was hit
            toward against this pipeline's own linear weights (what
            actually happened, in runs), then for every batted ball
            attributes (expected − actual) to whichever player was
            standing at that position on that play. A fielder's Def is
            the sum of that over their CURRENT season's worth of
            chances — outperform the model for your zone, gain runs;
            underperform, lose them. The underlying regression model is
            trained ONCE PER SEASON (not every run) on pooled batted-ball
            data from this season plus last season for a bigger, more
            stable sample, then cached to models/ and reused all season
            — see build_defense_model() and DEFENSE_MODEL_CACHE.
    BsR   — baserunning runs, from stolen bases and caught-stealing.
            The counts (SB/CS) still have to come from the MLB Stats
            API — that's just the official box score, nothing to model,
            and those are re-fetched every run since they change every
            day a player plays. What used to be fixed textbook weights
            (0.20 / -0.40) are now derived from a real run-expectancy
            matrix built from this season plus last season's own
            play-by-play base/out states — see
            build_run_expectancy_and_baserunning_weights(). Like the
            defense model, this derivation is cached and only runs once
            per season, not on every pipeline tick. SB/CS counts are
            only fetched for the top hitters by projected Bat runs
            (bounded MLB Stats API calls, same as the age lookup below);
            everyone else defaults to BsR = 0.

    WAR = (Bat + Def + BsR + Replacement) / RUNS_PER_WIN
    Replacement = 20 runs / 600 PA (standard replacement-level constant).

This is a real WAR *structure* — three components combined into wins
above replacement. Bat is computed fresh every run (it's cheap — just
this run's own projected stats). Def and the SB/CS run values are
trained/derived from this pipeline's own data too, but only once per
season (cached under models/, like the hitter/pitcher projection
models), not re-trained on every cron tick — see "Model caching" below.
Only the raw SB/CS event counts and player ages come from an outside
feed (MLB's official Stats API — factual box-score data, not someone
else's model). Def and BsR are best-effort: if the local pitch data or
the Stats API is unavailable on a given run, those components degrade
to 0 for the affected players rather than failing the whole build (see
`data_availability` in the output bundle).

Model caching: the defense regression model and the baserunning run-
values are expensive-ish to (re)build (they load the season's full
Statcast pitch file) but change slowly — real defensive/baserunning
value doesn't meaningfully shift from one cron tick to the next. So
both are trained once and cached to disk (models/defense_run_value_
model.pkl, models/baserunning_re_weights.json) the first time this
runs each season, then just loaded on every subsequent run — the same
pattern the hitter/pitcher stat models already use for their own
tuned hyperparameters. Pass retrain_models=True to build_rankings()
(or --retrain-defense on the CLI, or set BULLPEN_RETRAIN_DEFENSE=force)
to force a fresh retrain, e.g. later in the season once meaningfully
more data has accumulated.

PITCHERS get a real WAR too, RA9-based (runs-allowed per 9 innings)
rather than FIP-based:

    Pitching runs = (league_RA9 − pitcher_RA9) * (IP / 9)
    Replacement    = league_RA9 * (PITCHER_REPLACEMENT_RA9_MULTIPLIER − 1) * (IP / 9)
    WAR = (Pitching runs + Replacement) / RUNS_PER_WIN

league_RA9 is the IP-weighted average across this pitcher pool, computed
fresh from this pipeline's own data every run (self-consistent, same
pattern as the hitters' league-average Bat rate — no external league
constant). PITCHER_REPLACEMENT_RA9_MULTIPLIER is NOT derived from this
pipeline's data — it's a standard, published sabermetric convention
(replacement-level pitching ≈ .380 win%, which via the Pythagorean
win% relationship works out to allowing runs at roughly 1.28x the
league rate). FIP-based WAR (the more standard approach) would need
home-runs-allowed and hit-by-pitch counts, which the accuracy log
doesn't track yet — RA9-based is the best available with today's data.
"ER" here is actually total runs allowed (the accuracy log doesn't
separate earned from unearned), a pre-existing simplification.

    Old production-points formula (still computed, kept for reference):
    Pitcher: 3*IP + K - ER - H_allowed - BB

Every stat and score below is computed BOTH ways and shipped side by side
so the front end can offer it as a toggle:

    ros_*   — rest-of-season only: per-game rate x games remaining.
    full_*  — full season: real accumulated season-to-date totals (actual
              box scores) PLUS the same rest-of-season projection. The
              "already happened" portion is ground truth, not rate*gp.

Per-game rates are taken from actual box-score results when a player has
played, projected otherwise, then scaled by each player's estimated games
remaining in the season.

Outputs: outputs/player_rankings.json
Player ages cached at data/player_ages.json (MLB Stats API, fetched lazily).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
ACC_CSV = HERE / "2026_player_accuracy.csv"
OUT_JSON = HERE / "outputs" / "player_rankings.json"
AGE_CACHE = HERE / "data" / "player_ages.json"
MODELS_DIR = HERE / "models"
# Cached artifacts for the in-house defense/baserunning models — trained
# once per season (see the "Model caching" note above), not every run.
DEFENSE_MODEL_CACHE = MODELS_DIR / "defense_run_value_model.pkl"
BASERUNNING_WEIGHTS_CACHE = MODELS_DIR / "baserunning_re_weights.json"
ET = ZoneInfo("America/New_York")

def _pitch_data_csv(season_year: int) -> Path:
    return HERE / f"pitch_data_{season_year}.csv"


def _resolve_retrain_defense(explicit: bool) -> bool:
    """Whether to force-retrain the cached defense/baserunning models this
    run instead of reusing models/ on disk. True if the caller explicitly
    asked (--retrain-defense / build_rankings(retrain_models=True)), or if
    BULLPEN_RETRAIN_DEFENSE=force is set in the environment (same override
    pattern daily_update.py already uses for BULLPEN_RETRAIN, but distinct
    — these models retrain once per SEASON by default, not once per day)."""
    if explicit:
        return True
    return os.environ.get("BULLPEN_RETRAIN_DEFENSE", "auto").lower() == "force"

DEFAULT_SEASON_GAMES = 162
AGE_CUTOFF = 25
# Only fetch/attach ages for the top N by WAR / power score — deep-bench
# players are noise for a "power rankings" spotlight feature and this
# keeps MLB Stats API calls bounded.
TOP_N_HITTERS_FOR_AGE = 400
TOP_N_PITCHERS_FOR_AGE = 300
# Baserunning (SB/CS) is a per-player MLB Stats API lookup, so it's bounded
# the same way — top N hitters by projected Bat runs get real SB/CS,
# everyone else defaults to BsR = 0.
TOP_N_HITTERS_FOR_BASERUNNING = 400

PITCHER_WEIGHTS = {"IP": 3.0, "K": 1.0, "ER": -1.0, "H": -0.5, "BB": -0.5}
# Old production-points formula — still computed and shipped (ros/full_
# power_score) for reference, but WAR (below) is now the primary pitcher
# ranking metric.

# Replacement-level pitching, expressed as a multiplier on league-average
# RA9 (runs allowed per 9 IP): a replacement-level pitcher is modeled as
# allowing runs at PITCHER_REPLACEMENT_RA9_MULTIPLIER x the league rate.
# This is a standard published sabermetric convention (replacement level
# ≈ .380 win%, which via the Pythagorean win%-expectation relationship
# — win% = RS^2/(RS^2+RA^2) — works out to RA/RS ≈ 1.28 when RS is held
# at the league rate), NOT something derived from this pipeline's own
# data, unlike league_RA9 itself (which IS computed fresh from this
# pitcher pool every run).
PITCHER_REPLACEMENT_RA9_MULTIPLIER = 1.28

# ---------------------------------------------------------------------------
# WAR model constants.
# BATTING_LINEAR_WEIGHTS: standard public sabermetric linear weights, used
# both for the Bat component and as the training target for the in-house
# defense model below (see build_defense_model). REPLACEMENT_RUNS_PER_600PA
# and RUNS_PER_WIN are standard replacement-level / win-conversion
# constants. SB_RUN / CS_RUN are only a FALLBACK — the real per-run values
# are derived from this pipeline's own play-by-play data (pooled across
# this season + last season) in build_run_expectancy_and_baserunning_
# weights(), cached, and reused all season; these fixed numbers only kick
# in if that derivation has never been able to run (pitch data missing /
# too thin, and no cache exists yet either).
# ---------------------------------------------------------------------------
BATTING_LINEAR_WEIGHTS = {
    "BB": 0.69, "1B": 0.89, "2B": 1.27, "3B": 1.62, "HR": 2.10, "OUT": -0.25,
}
FALLBACK_SB_RUN, FALLBACK_CS_RUN = 0.20, -0.40
REPLACEMENT_RUNS_PER_600PA = 20.0
RUNS_PER_WIN = 10.0

POSITION_NAMES = {1: "P", 2: "C", 3: "1B", 4: "2B", 5: "3B",
                   6: "SS", 7: "LF", 8: "CF", 9: "RF"}
# Batted-ball outcome -> run value, using the SAME linear weights as Bat
# runs above, so the defense model's "actual value of this play" is
# denominated in the same units as everything else in this file. Rare /
# ambiguous outcomes (fielder's choice where the batter is safe, catcher
# interference, etc.) are left out of the training set entirely rather
# than guessed at.
_HIT_EVENT_WEIGHT_KEY = {"single": "1B", "double": "2B", "triple": "3B", "home_run": "HR"}
_OUT_EVENTS = {"field_out", "force_out", "grounded_into_double_play", "double_play",
               "fielders_choice_out", "sac_fly", "sac_bunt", "sac_fly_double_play"}


# ---------------------------------------------------------------------------
# Per-game rollup
# ---------------------------------------------------------------------------
def _hitter_per_game(group: pd.DataFrame) -> dict:
    played = group[group["actual_pa"].notna()]
    games_played = int(len(played))
    if games_played >= 1:
        avg = {
            "H":   float(played["actual_hits"].mean()),
            "HR":  float(played["actual_hr"].mean()),
            "BB":  float(played["actual_walks"].mean()),
            "K":   float(played["actual_strikeouts"].mean()),
            "R":   float(played["actual_runs"].mean()),
            "RBI": float(played["actual_rbi"].mean()),
            "2B":  float(played.get("actual_doubles", pd.Series(dtype=float)).fillna(0).mean()) if "actual_doubles" in played else 0.0,
            "3B":  float(played.get("actual_triples", pd.Series(dtype=float)).fillna(0).mean()) if "actual_triples" in played else 0.0,
            "PA":  float(played["actual_pa"].mean()),
        }
    else:
        last = group.sort_values("game_date").iloc[-1]
        avg = {
            "H":   float(last.get("proj_hits") or 0),
            "HR":  float(last.get("proj_hr") or 0),
            "BB":  float(last.get("proj_walks") or 0),
            "K":   float(last.get("proj_strikeouts") or 0),
            "R":   float(last.get("proj_runs") or 0),
            "RBI": float(last.get("proj_rbi") or 0),
            "2B":  0.0,
            "3B":  0.0,
            "PA":  float(last.get("proj_pa") or 0),
        }
    avg["games_played"] = games_played
    return avg


def _hitter_totals(group: pd.DataFrame) -> dict:
    """Real season-to-date accumulated counting stats (sums, not
    averages) from actual box scores. All zero if the player hasn't
    played yet. Used as the "already happened" half of a full-season
    projection — the rest-of-season rate estimate covers the other half."""
    played = group[group["actual_pa"].notna()]
    if played.empty:
        return {"H": 0.0, "HR": 0.0, "BB": 0.0, "K": 0.0, "R": 0.0,
                "RBI": 0.0, "2B": 0.0, "3B": 0.0, "PA": 0.0}
    return {
        "H":   float(played["actual_hits"].sum()),
        "HR":  float(played["actual_hr"].sum()),
        "BB":  float(played["actual_walks"].sum()),
        "K":   float(played["actual_strikeouts"].sum()),
        "R":   float(played["actual_runs"].sum()),
        "RBI": float(played["actual_rbi"].sum()),
        "2B":  float(played.get("actual_doubles", pd.Series(dtype=float)).fillna(0).sum()) if "actual_doubles" in played else 0.0,
        "3B":  float(played.get("actual_triples", pd.Series(dtype=float)).fillna(0).sum()) if "actual_triples" in played else 0.0,
        "PA":  float(played["actual_pa"].sum()),
    }


def _pitcher_totals(group: pd.DataFrame) -> dict:
    """Real season-to-date accumulated pitching stats (sums). All zero
    if the player hasn't pitched yet."""
    played = group[group["actual_ip"].notna()]
    if played.empty:
        return {"IP": 0.0, "K": 0.0, "BB": 0.0, "H": 0.0, "ER": 0.0}
    return {
        "IP": float(played["actual_ip"].sum()),
        "K":  float(played["actual_strikeouts"].sum()),
        "BB": float(played["actual_walks"].sum()),
        "H":  float(played["actual_hits_allowed"].sum()),
        "ER": float(played["actual_runs_allowed"].sum()),
    }


def _pitcher_per_game(group: pd.DataFrame) -> dict:
    played = group[group["actual_ip"].notna()]
    games_played = int(len(played))
    if games_played >= 1:
        avg = {
            "IP": float(played["actual_ip"].mean()),
            "K":  float(played["actual_strikeouts"].mean()),
            "BB": float(played["actual_walks"].mean()),
            "H":  float(played["actual_hits_allowed"].mean()),
            "ER": float(played["actual_runs_allowed"].mean()),
        }
    else:
        last = group.sort_values("game_date").iloc[-1]
        avg = {
            "IP": float(last.get("proj_ip") or 0),
            "K":  float(last.get("proj_strikeouts") or 0),
            "BB": float(last.get("proj_walks") or 0),
            "H":  float(last.get("proj_hits_allowed") or 0),
            "ER": float(last.get("proj_runs_allowed") or 0),
        }
    avg["games_played"] = games_played
    return avg


# ---------------------------------------------------------------------------
# Remaining games estimate
# ---------------------------------------------------------------------------
def _team_games_played(df: pd.DataFrame) -> dict[str, int]:
    out: dict[str, int] = {}
    if "game_pk" in df.columns:
        for team, sub in df.dropna(subset=["team"]).groupby("team"):
            out[team] = int(sub["game_pk"].nunique())
    return out


def _remaining_games(games_played: int, team_games: int, season_games: int) -> float:
    if team_games <= 0: return 0.0
    participation = max(0.05, min(1.0, games_played / team_games))
    remaining_team = max(0, season_games - team_games)
    return remaining_team * participation


# ---------------------------------------------------------------------------
# Batting runs (offense component of WAR)
# ---------------------------------------------------------------------------
def _batting_runs_per_pa(rate: dict) -> float:
    """Raw linear-weights runs per PA for a per-game rate dict with
    H/2B/3B/HR/BB/PA keys (season-to-date or projected rate)."""
    pa = rate.get("PA") or 0.0
    if pa <= 0:
        return 0.0
    h, doubles, triples, hr = rate.get("H", 0.0), rate.get("2B", 0.0), rate.get("3B", 0.0), rate.get("HR", 0.0)
    bb = rate.get("BB", 0.0)
    singles = max(0.0, h - doubles - triples - hr)
    outs = max(0.0, pa - h - bb)
    w = BATTING_LINEAR_WEIGHTS
    runs = (w["BB"] * bb + w["1B"] * singles + w["2B"] * doubles +
            w["3B"] * triples + w["HR"] * hr + w["OUT"] * outs)
    return runs / pa


def _league_avg_batting_runs_per_pa(per_game_by_player: list[dict]) -> float:
    """PA-weighted league-average batting runs/PA across the current
    hitter pool — makes "above average" self-consistent without needing
    an external league constant."""
    total_runs, total_pa = 0.0, 0.0
    for rate in per_game_by_player:
        pa = rate.get("PA") or 0.0
        if pa <= 0:
            continue
        total_runs += _batting_runs_per_pa(rate) * pa
        total_pa += pa
    return (total_runs / total_pa) if total_pa > 0 else 0.0


# ---------------------------------------------------------------------------
# Defense — in-house model, trained once per season (cached) on pooled
# multi-year Statcast batted-ball data, then scored against just the
# current season's chances. Best-effort, never raises.
# ---------------------------------------------------------------------------
def _load_pitch_data(season_year: int, columns: list[str]) -> pd.DataFrame | None:
    path = _pitch_data_csv(season_year)
    if not path.exists():
        print(f"⚠️  {path.name} not found, skipping model(s) that need it.")
        return None
    try:
        return pd.read_csv(path, usecols=columns, low_memory=False)
    except Exception as e:
        print(f"⚠️  Failed to load {path.name}: {e}")
        return None


def _load_multi_year_pitch_data(years: list[int], columns: list[str]) -> pd.DataFrame | None:
    """Pools pitch_data_<year>.csv across several years (e.g. this season
    plus last season) into one DataFrame for training — a bigger, more
    stable sample than this season alone, especially early in the year.
    Years whose file doesn't exist are silently skipped (best-effort);
    returns None only if none of the years have data available."""
    frames = []
    for yr in years:
        d = _load_pitch_data(yr, columns)
        if d is not None:
            frames.append(d)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def _outcome_run_value(events) -> float | None:
    """Maps a Statcast `events` value to this file's own linear-weights
    run value (same BATTING_LINEAR_WEIGHTS used for Bat runs), or None
    for outcomes we don't want in the defense model's training set
    (still mid-PA, ambiguous, or too rare to trust)."""
    if events in _HIT_EVENT_WEIGHT_KEY:
        return BATTING_LINEAR_WEIGHTS[_HIT_EVENT_WEIGHT_KEY[events]]
    if events == "field_error":
        # Batter reaches, most commonly at 1st — treated as roughly
        # single-equivalent. An approximation; the alternative (excluding
        # errors from training entirely) would bias the model toward
        # thinking every hard-hit ball in that zone was fielded cleanly.
        return BATTING_LINEAR_WEIGHTS["1B"]
    if events in _OUT_EVENTS:
        return BATTING_LINEAR_WEIGHTS["OUT"]
    return None


_DEFENSE_FEATURE_COLS = ["launch_speed", "launch_angle", "hit_location"]
_DEFENSE_RAW_COLS = ["events", "bb_type", "hit_location", "launch_speed", "launch_angle",
                     "pitcher", "fielder_2", "fielder_3", "fielder_4", "fielder_5",
                     "fielder_6", "fielder_7", "fielder_8", "fielder_9"]


def _prep_batted_ball_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Filters raw Statcast pitch rows down to balls in play with a usable
    outcome value + the three model features, ready for either training
    or scoring."""
    bip = df[df["bb_type"].notna()].copy()
    bip["outcome_value"] = bip["events"].apply(_outcome_run_value)
    return bip.dropna(subset=["outcome_value", *_DEFENSE_FEATURE_COLS])


def _train_defense_model(season_year: int, min_training_rows: int, train_years: list[int]):
    """Fits (and caches to DEFENSE_MODEL_CACHE) the expected-run-value
    regression on pooled batted-ball data across `train_years`. Returns
    the fitted model, or None if there isn't enough data / scikit-learn
    is unavailable."""
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
    except Exception as e:
        print(f"⚠️  scikit-learn unavailable, skipping defense model: {e}")
        return None

    raw = _load_multi_year_pitch_data(train_years, _DEFENSE_RAW_COLS)
    if raw is None:
        return None
    train = _prep_batted_ball_rows(raw)
    if len(train) < min_training_rows:
        print(f"⚠️  Only {len(train)} usable batted-ball rows across {train_years}, "
              f"skipping defense model (need >= {min_training_rows}).")
        return None

    model = HistGradientBoostingRegressor(max_depth=5, random_state=0)
    model.fit(train[_DEFENSE_FEATURE_COLS].values, train["outcome_value"].values)

    try:
        import joblib
        MODELS_DIR.mkdir(exist_ok=True)
        joblib.dump({
            "model": model,
            "meta": {
                "train_years": train_years,
                "training_rows": len(train),
                "trained_at": datetime.now(ET).isoformat(),
            },
        }, DEFENSE_MODEL_CACHE)
        print(f"   Trained + cached defense model -> {DEFENSE_MODEL_CACHE.name} "
              f"(years={train_years}, rows={len(train):,}). Reused every run until "
              f"--retrain-defense / BULLPEN_RETRAIN_DEFENSE=force.")
    except Exception as e:
        print(f"⚠️  Trained defense model but failed to cache it "
              f"(will retrain every run until this succeeds): {e}")

    return model


def build_defense_model(season_year: int, min_training_rows: int = 500,
                        retrain: bool = False,
                        train_years: list[int] | None = None) -> tuple[dict[int, float], dict[int, str]]:
    """Scores THIS season's own batted balls (launch speed + launch angle
    + which fielding zone it was hit toward) against an in-house
    "expected run value of this contact" model (using this file's own
    BATTING_LINEAR_WEIGHTS as the target), and credits the residual
    (expected − actual) to whichever player was standing at that
    position on that specific play (via the fielder_<N> / pitcher
    columns Statcast already tags each pitch with). A fielder's season
    Def is the sum of their residuals — save more hits than the model
    expects for that zone/contact quality, gain runs; allow more, lose
    them.

    The regression model itself is trained ONCE PER SEASON (not every
    call) on pooled batted-ball data from `train_years` (default: this
    season plus last season, for a bigger/more stable sample) and cached
    to DEFENSE_MODEL_CACHE; subsequent calls just load the cached model
    and re-score it against the current season's own chances, so a
    player's Def total always reflects only THIS season's fielding —
    last season's data improves the model's accuracy, it doesn't leak
    into anyone's credited runs. Pass retrain=True (or set
    BULLPEN_RETRAIN_DEFENSE=force) to force a fresh fit.

    This intentionally conditions on hit_location (the fielding zone),
    not just contact quality — so a shortstop is implicitly compared to
    the league's shortstops, not to first basemen, without needing a
    separately hardcoded positional adjustment.

    Known simplifications: runs scored mid-plate-appearance (a balk or
    wild pitch before the ball is even put in play) aren't attributed to
    this specific play; errors are valued as single-equivalent regardless
    of how far the batter actually advanced; catcher/pitcher "defense"
    here only covers their rare balls-in-play chances (bunts, comebacks),
    not framing/blocking/pickoffs, so it's a thin signal for those two
    spots specifically.

    Returns ({mlb_id: season Def runs}, {mlb_id: primary position name}).
    Best-effort: returns ({}, {}) if this season's pitch data isn't
    available, or if no model (cached or freshly trained) can be
    produced.
    """
    df = _load_pitch_data(season_year, _DEFENSE_RAW_COLS)
    if df is None:
        return {}, {}

    retrain = _resolve_retrain_defense(retrain)
    model = None
    if not retrain and DEFENSE_MODEL_CACHE.exists():
        try:
            import joblib
            cache = joblib.load(DEFENSE_MODEL_CACHE)
            model = cache["model"]
            meta = cache.get("meta", {})
            print(f"   Using cached defense model (years={meta.get('train_years')}, "
                  f"rows={meta.get('training_rows')}, trained {meta.get('trained_at')})")
        except Exception as e:
            print(f"⚠️  Failed to load cached defense model, retraining: {e}")
            model = None

    if model is None:
        model = _train_defense_model(
            season_year, min_training_rows,
            train_years or [season_year - 1, season_year],
        )
    if model is None:
        return {}, {}

    scored = _prep_batted_ball_rows(df)
    if scored.empty:
        return {}, {}
    scored["expected_value"] = model.predict(scored[_DEFENSE_FEATURE_COLS].values)
    scored["fielder_run_value"] = scored["expected_value"] - scored["outcome_value"]

    def _fielder_id(row) -> float:
        pos_num = int(row["hit_location"])
        if pos_num == 1:
            return row.get("pitcher")
        return row.get(f"fielder_{pos_num}")

    scored["fielder_id"] = scored.apply(_fielder_id, axis=1)
    scored = scored.dropna(subset=["fielder_id"])
    scored["fielder_id"] = scored["fielder_id"].astype(int)
    scored["position_num"] = scored["hit_location"].astype(int)

    defense_runs = scored.groupby("fielder_id")["fielder_run_value"].sum().to_dict()
    # Primary position = whichever zone a player recorded the most
    # chances at this season (an approximation for multi-position players).
    chance_counts = scored.groupby(["fielder_id", "position_num"]).size()
    primary_position: dict[int, str] = {}
    for fid, sub in chance_counts.groupby(level=0):
        best_pos = sub.loc[fid].idxmax()
        primary_position[int(fid)] = POSITION_NAMES.get(int(best_pos), "?")

    return {int(k): float(v) for k, v in defense_runs.items()}, primary_position


# ---------------------------------------------------------------------------
# Baserunning: run-expectancy matrix (in-house) + SB/CS counts (MLB Stats
# API — official box score data, not a model to build).
# ---------------------------------------------------------------------------
def build_run_expectancy_and_baserunning_weights(
    season_year: int, min_state_sample: int = 30,
    retrain: bool = False, train_years: list[int] | None = None,
) -> tuple[float, float, bool]:
    """Builds a base/out-state run-expectancy matrix (the classic 8
    base-states x 3 out-counts grid: for each state, the average number
    of runs that score in the rest of that half-inning) from pooled
    play-by-play Statcast data across `train_years` (default: this
    season plus last season, for a bigger/more stable sample), then
    derives the stolen-base and caught-stealing run values from it
    directly — the same "marginal change in run expectancy" derivation
    that produced the published SB=0.20 / CS=-0.40 constants in the
    first place, just run on this pipeline's own data instead of reusing
    a decades-old number:

        SB value = RE(runner on 2nd, outs) − RE(runner on 1st, outs)
        CS value = RE(bases empty, outs+1) − RE(runner on 1st, outs)

    averaged across the out counts where both sides of the comparison
    have enough samples to trust.

    Like build_defense_model(), this is trained ONCE PER SEASON, not
    every call — the result is cached to BASERUNNING_WEIGHTS_CACHE and
    just reloaded on subsequent runs. Pass retrain=True (or set
    BULLPEN_RETRAIN_DEFENSE=force) to force a fresh derivation.

    Known simplification: a run that scores mid-plate-appearance (e.g. a
    wild pitch before the ball is put in play) is attributed to whichever
    plate appearance's own before/after score it shows up on, which can
    occasionally undercount runs charged to an earlier state — a minor
    approximation, not a structural one.

    Returns (sb_run_value, cs_run_value, built_from_data: bool). Falls
    back to the fixed FALLBACK_SB_RUN / FALLBACK_CS_RUN constants (with
    built_from_data=False) if no cache exists yet and pitch data is
    missing or the pooled sample is too thin to trust (e.g. very early
    in the season, before last season's data alone is even cached).
    """
    retrain = _resolve_retrain_defense(retrain)
    if not retrain and BASERUNNING_WEIGHTS_CACHE.exists():
        try:
            cached = json.loads(BASERUNNING_WEIGHTS_CACHE.read_text())
            print(f"   Using cached baserunning weights (years={cached.get('train_years')}, "
                  f"trained {cached.get('trained_at')}): "
                  f"SB={cached['sb_run_value']:.3f} CS={cached['cs_run_value']:.3f}")
            return cached["sb_run_value"], cached["cs_run_value"], cached["built_from_data"]
        except Exception as e:
            print(f"⚠️  Failed to load cached baserunning weights, recomputing: {e}")

    years = train_years or [season_year - 1, season_year]
    cols = ["game_pk", "inning", "inning_topbot", "at_bat_number", "events",
            "on_1b", "on_2b", "on_3b", "outs_when_up", "bat_score", "post_bat_score"]
    df = _load_multi_year_pitch_data(years, cols)
    if df is None:
        return FALLBACK_SB_RUN, FALLBACK_CS_RUN, False

    pa = df[df["events"].notna()].copy()
    if pa.empty:
        return FALLBACK_SB_RUN, FALLBACK_CS_RUN, False

    pa = pa.sort_values(["game_pk", "inning", "inning_topbot", "at_bat_number"])
    pa["runs_on_play"] = (pa["post_bat_score"] - pa["bat_score"]).clip(lower=0)
    pa["base_state"] = (pa["on_1b"].notna().astype(int).astype(str) +
                         pa["on_2b"].notna().astype(int).astype(str) +
                         pa["on_3b"].notna().astype(int).astype(str))
    pa["outs_when_up"] = pa["outs_when_up"].clip(0, 2)
    # Runs scored from this PA through the end of the half-inning: a
    # reverse cumulative sum of per-PA runs within each half-inning.
    pa["runs_remaining"] = (
        pa.groupby(["game_pk", "inning", "inning_topbot"])["runs_on_play"]
          .transform(lambda s: s[::-1].cumsum()[::-1])
    )

    grouped = pa.groupby(["base_state", "outs_when_up"])["runs_remaining"]
    re_matrix, re_counts = grouped.mean(), grouped.size()

    def re(base_state: str, outs: int) -> float | None:
        key = (base_state, outs)
        if key not in re_matrix.index or re_counts.get(key, 0) < min_state_sample:
            return None
        return float(re_matrix.loc[key])

    sb_deltas, cs_deltas = [], []
    for outs in (0, 1, 2):
        re_1st, re_2nd = re("100", outs), re("010", outs)
        if re_1st is not None and re_2nd is not None:
            sb_deltas.append(re_2nd - re_1st)
        re_after_cs = 0.0 if outs == 2 else re("000", outs + 1)
        if re_1st is not None and re_after_cs is not None:
            cs_deltas.append(re_after_cs - re_1st)

    if not sb_deltas or not cs_deltas:
        return FALLBACK_SB_RUN, FALLBACK_CS_RUN, False

    sb_run_value = sum(sb_deltas) / len(sb_deltas)
    cs_run_value = sum(cs_deltas) / len(cs_deltas)
    try:
        MODELS_DIR.mkdir(exist_ok=True)
        BASERUNNING_WEIGHTS_CACHE.write_text(json.dumps({
            "sb_run_value": sb_run_value, "cs_run_value": cs_run_value,
            "built_from_data": True, "train_years": years,
            "trained_at": datetime.now(ET).isoformat(),
        }, indent=2))
        print(f"   Derived + cached baserunning weights -> {BASERUNNING_WEIGHTS_CACHE.name} "
              f"(years={years}): SB={sb_run_value:.3f} CS={cs_run_value:.3f}. Reused every "
              f"run until --retrain-defense / BULLPEN_RETRAIN_DEFENSE=force.")
    except Exception as e:
        print(f"⚠️  Derived baserunning weights but failed to cache them "
              f"(will recompute every run until this succeeds): {e}")

    return sb_run_value, cs_run_value, True


def fetch_baserunning(ids: set[int], season: int, throttle: float = 0.05) -> dict[int, dict]:
    """{mlb_id: {"sb": int, "cs": int}} via the MLB Stats API season
    hitting stats endpoint. No persistent cache — unlike age, SB/CS
    change every day the player plays, so this is re-fetched fresh on
    every run for the bounded top-N pool. Best-effort per player."""
    out: dict[int, dict] = {}
    for mid in ids:
        url = f"https://statsapi.mlb.com/api/v1/people/{int(mid)}/stats?stats=season&group=hitting&season={season}"
        try:
            r = requests.get(url, timeout=8)
            r.raise_for_status()
            data = r.json()
            splits = (data.get("stats") or [{}])[0].get("splits") or []
            if not splits:
                continue
            stat = splits[0].get("stat") or {}
            out[int(mid)] = {
                "sb": int(stat.get("stolenBases") or 0),
                "cs": int(stat.get("caughtStealing") or 0),
            }
        except Exception:
            continue
        time.sleep(throttle)
    return out


# ---------------------------------------------------------------------------
# Pitcher points (old formula, kept for reference) + RA9-based pitcher WAR.
# ---------------------------------------------------------------------------
def _pitcher_points(ros: dict) -> float:
    return sum(ros.get(k, 0.0) * w for k, w in PITCHER_WEIGHTS.items())


def _ra9(rate: dict) -> float | None:
    """Runs allowed per 9 IP for a per-game rate/totals dict with IP/ER
    keys. None if IP is 0 (hasn't pitched / no innings projected)."""
    ip = rate.get("IP") or 0.0
    if ip <= 0:
        return None
    return rate.get("ER", 0.0) / ip * 9.0


def _league_avg_ra9(per_game_by_pitcher: list[dict]) -> float:
    """IP-weighted league-average RA9 across the current pitcher pool —
    self-consistent, no external league constant, same pattern as
    _league_avg_batting_runs_per_pa for hitters."""
    total_er, total_ip = 0.0, 0.0
    for rate in per_game_by_pitcher:
        ip = rate.get("IP") or 0.0
        if ip <= 0:
            continue
        total_er += rate.get("ER", 0.0)
        total_ip += ip
    return (total_er / total_ip * 9.0) if total_ip > 0 else 0.0


def _pitching_runs_above_avg(rate: dict, league_ra9: float) -> float:
    """Runs above average: how many fewer (or more) runs this pitcher
    allowed than a league-average pitcher would have in the same IP."""
    ip = rate.get("IP") or 0.0
    ra9 = _ra9(rate)
    if ra9 is None:
        return 0.0
    return (league_ra9 - ra9) * (ip / 9.0)


def _pitching_replacement_runs(rate: dict, league_ra9: float) -> float:
    """The extra runs-above-average a REPLACEMENT-level pitcher (not an
    average one) would have allowed in the same IP — the piece that
    turns runs-above-average into runs-above-replacement when added to
    _pitching_runs_above_avg. See PITCHER_REPLACEMENT_RA9_MULTIPLIER."""
    ip = rate.get("IP") or 0.0
    if ip <= 0:
        return 0.0
    return league_ra9 * (PITCHER_REPLACEMENT_RA9_MULTIPLIER - 1.0) * (ip / 9.0)


# ---------------------------------------------------------------------------
# Player ages (for the 25-and-under cut)
# ---------------------------------------------------------------------------
def _load_age_cache() -> dict:
    if not AGE_CACHE.exists():
        return {}
    try:
        return json.loads(AGE_CACHE.read_text())
    except Exception:
        return {}


def _save_age_cache(cache: dict) -> None:
    AGE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    AGE_CACHE.write_text(json.dumps(cache, indent=2, sort_keys=True))


def _fetch_age_one(mlb_id: int) -> dict | None:
    """One MLB Stats API people lookup. Returns {age, birth_date, name}."""
    url = f"https://statsapi.mlb.com/api/v1/people/{mlb_id}"
    try:
        r = requests.get(url, timeout=8)
        r.raise_for_status()
        data = r.json()
    except Exception:
        return None
    people = data.get("people") or []
    if not people:
        return None
    p = people[0]
    bdate = p.get("birthDate") or ""
    name = p.get("fullName") or ""
    if not bdate:
        return None
    try:
        bd = datetime.strptime(bdate, "%Y-%m-%d")
        today = datetime.now()
        years = today.year - bd.year - ((today.month, today.day) < (bd.month, bd.day))
        return {"age": years, "birth_date": bdate, "name": name}
    except Exception:
        return None


def get_player_ages(ids: set[int], throttle: float = 0.05) -> dict[int, dict]:
    """Return {mlb_id: {age, birth_date, name}} for `ids`. Uses
    data/player_ages.json as a persistent cache and only hits the MLB
    Stats API for IDs not already cached. Network failures degrade
    gracefully — missing IDs just don't get an age in the result.
    """
    cache = _load_age_cache()
    needs_save = False
    now_iso = datetime.now(ET).strftime("%Y-%m-%d")
    out: dict[int, dict] = {}

    for mid in ids:
        key = str(mid)
        entry = cache.get(key)
        if entry and entry.get("age") is not None:
            out[int(mid)] = entry
            continue
        info = _fetch_age_one(int(mid))
        if info:
            cache[key] = info
            out[int(mid)] = info
            needs_save = True
            time.sleep(throttle)
    if needs_save:
        cache["__updated__"] = now_iso
        try:
            _save_age_cache(cache)
        except Exception:
            pass
    return out


def _attach_ages(rows: list[dict], top_n: int) -> None:
    """Mutates `rows` in place, adding an "age" key (int or None) to the
    top `top_n` entries by rank (rows are already sorted)."""
    ids = {r["mlb_id"] for r in rows[:top_n] if r.get("mlb_id") is not None}
    ages = get_player_ages(ids)
    for r in rows:
        info = ages.get(r["mlb_id"]) if r.get("mlb_id") is not None else None
        r["age"] = info.get("age") if info else None


# ---------------------------------------------------------------------------
# Main bundle
# ---------------------------------------------------------------------------
def build_rankings(season_games: int = DEFAULT_SEASON_GAMES,
                   as_of: str | None = None,
                   fetch_ages: bool = True,
                   fetch_defense: bool = True,
                   fetch_baserunning_data: bool = True,
                   retrain_models: bool = False) -> dict:
    if not ACC_CSV.exists():
        raise FileNotFoundError(f"Accuracy log not found at {ACC_CSV}")

    df = pd.read_csv(ACC_CSV, low_memory=False)
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    if as_of:
        df = df[df["game_date"] <= pd.to_datetime(as_of)]

    team_games = _team_games_played(df)
    season_year = int(df["game_date"].dropna().dt.year.max()) if df["game_date"].notna().any() \
        else datetime.now(ET).year

    # ------- HITTERS: per-game rates + totals + Bat runs -------
    hitters_df = df[df["player_type"].astype(str).str.lower() == "hitter"].copy()
    prelim = []  # per-player working dict, discarded after hitter_rows is built
    for (mid, name), grp in hitters_df.groupby(["mlb_id", "player_name"]):
        team = str(grp["team"].dropna().iloc[-1]) if grp["team"].notna().any() else ""
        per_game = _hitter_per_game(grp)
        gp = per_game.pop("games_played")
        totals = _hitter_totals(grp)
        team_gp = team_games.get(team, max(gp, 1))
        gr = _remaining_games(gp, team_gp, season_games)
        ros = {k: per_game[k] * gr for k in per_game}
        full = {k: totals[k] + ros[k] for k in per_game}
        prelim.append({
            "mlb_id": int(mid) if not pd.isna(mid) else None,
            "name": name, "team": team, "team_gp": team_gp,
            "per_game": per_game, "totals": totals,
            "gp": gp, "gr": gr, "ros": ros, "full": full,
        })

    league_bat_rate = _league_avg_batting_runs_per_pa([p["per_game"] for p in prelim])

    for p in prelim:
        player_bat_rate = _batting_runs_per_pa(p["per_game"])
        p["ros_bat_runs"] = (player_bat_rate - league_bat_rate) * p["ros"]["PA"]
        p["full_bat_runs"] = (player_bat_rate - league_bat_rate) * p["full"]["PA"]

    # Bound the per-player baserunning lookup to the top N by Bat runs —
    # this is what "power" hitters look like before defense/baserunning
    # are folded in, and keeps the MLB Stats API call volume sane.
    prelim.sort(key=lambda p: p["ros_bat_runs"], reverse=True)
    baserunning_data = {}
    sb_run_value, cs_run_value, re_matrix_built = FALLBACK_SB_RUN, FALLBACK_CS_RUN, False
    defense_runs_by_player, position_by_player = {}, {}
    if fetch_baserunning_data:
        top_ids = {p["mlb_id"] for p in prelim[:TOP_N_HITTERS_FOR_BASERUNNING] if p["mlb_id"] is not None}
        baserunning_data = fetch_baserunning(top_ids, season_year)
        sb_run_value, cs_run_value, re_matrix_built = build_run_expectancy_and_baserunning_weights(
            season_year, retrain=retrain_models)
    if fetch_defense:
        defense_runs_by_player, position_by_player = build_defense_model(
            season_year, retrain=retrain_models)

    hitter_rows = []
    for p in prelim:
        mid, gp, gr, team_gp = p["mlb_id"], p["gp"], p["gr"], p["team_gp"]
        ros, full = p["ros"], p["full"]

        br = baserunning_data.get(mid)
        if br and gp > 0:
            sb_rate, cs_rate = br["sb"] / gp, br["cs"] / gp
            ros_bsr_runs = (sb_run_value * sb_rate + cs_run_value * cs_rate) * gr
            full_bsr_runs_to_date = sb_run_value * br["sb"] + cs_run_value * br["cs"]
            full_bsr_runs = full_bsr_runs_to_date + ros_bsr_runs
        else:
            ros_bsr_runs = 0.0
            full_bsr_runs = 0.0

        season_def_runs = defense_runs_by_player.get(mid, 0.0)
        ros_def_runs = (season_def_runs / gp * gr) if gp > 0 else 0.0
        full_def_runs = season_def_runs + ros_def_runs

        ros_replacement_runs = REPLACEMENT_RUNS_PER_600PA * (ros["PA"] / 600.0)
        full_replacement_runs = REPLACEMENT_RUNS_PER_600PA * (full["PA"] / 600.0)
        ros_war = (p["ros_bat_runs"] + ros_def_runs + ros_bsr_runs + ros_replacement_runs) / RUNS_PER_WIN
        full_war = (p["full_bat_runs"] + full_def_runs + full_bsr_runs + full_replacement_runs) / RUNS_PER_WIN

        hitter_rows.append({
            "mlb_id": mid, "name": p["name"], "team": p["team"],
            "position": position_by_player.get(mid),
            "games_played": gp,
            "games_remaining": round(gr, 1),
            "ros_h":  round(ros["H"], 1),   "full_h":  round(full["H"], 1),
            "ros_2b": round(ros["2B"], 1),  "full_2b": round(full["2B"], 1),
            "ros_3b": round(ros["3B"], 1),  "full_3b": round(full["3B"], 1),
            "ros_hr": round(ros["HR"], 1),  "full_hr": round(full["HR"], 1),
            "ros_r":  round(ros["R"], 1),   "full_r":  round(full["R"], 1),
            "ros_rbi": round(ros["RBI"], 1), "full_rbi": round(full["RBI"], 1),
            "ros_bb": round(ros["BB"], 1),  "full_bb": round(full["BB"], 1),
            "ros_k":  round(ros["K"], 1),   "full_k":  round(full["K"], 1),
            "ros_pa": round(ros["PA"], 1),  "full_pa": round(full["PA"], 1),
            "ros_bat_runs": round(p["ros_bat_runs"], 1), "full_bat_runs": round(p["full_bat_runs"], 1),
            "ros_def_runs": round(ros_def_runs, 1),      "full_def_runs": round(full_def_runs, 1),
            "ros_bsr_runs": round(ros_bsr_runs, 1),      "full_bsr_runs": round(full_bsr_runs, 1),
            "ros_replacement_runs": round(ros_replacement_runs, 1),
            "full_replacement_runs": round(full_replacement_runs, 1),
            "ros_war": round(ros_war, 1), "full_war": round(full_war, 1),
            "has_baserunning_data": mid in baserunning_data,
            "has_defense_data": mid in defense_runs_by_player,
        })
    hitter_rows.sort(key=lambda r: r["ros_war"], reverse=True)
    for i, r in enumerate(hitter_rows, start=1):
        r["rank"] = i

    # ------- PITCHERS: RA9-based WAR (Pitching runs + Replacement) / RUNS_PER_WIN,
    #         for both rest-of-season and full-season. Old production-points
    #         formula still computed alongside (ros/full_power_score). -------
    pitchers_df = df[df["player_type"].astype(str).str.lower() == "pitcher"].copy()
    prelim_p = []  # per-pitcher working dict, discarded after pitcher_rows is built
    for (mid, name), grp in pitchers_df.groupby(["mlb_id", "player_name"]):
        team = str(grp["team"].dropna().iloc[-1]) if grp["team"].notna().any() else ""
        per_game = _pitcher_per_game(grp)
        gp = per_game.pop("games_played")
        totals = _pitcher_totals(grp)
        team_gp = team_games.get(team, max(gp, 1))
        gr = _remaining_games(gp, team_gp, season_games)
        ros = {k: per_game[k] * gr for k in per_game}
        full = {k: totals[k] + ros[k] for k in per_game}
        prelim_p.append({
            "mlb_id": int(mid) if not pd.isna(mid) else None,
            "name": name, "team": team,
            "per_game": per_game, "gp": gp, "gr": gr, "ros": ros, "full": full,
        })

    league_ra9 = _league_avg_ra9([p["per_game"] for p in prelim_p])

    pitcher_rows = []
    for p in prelim_p:
        ros, full = p["ros"], p["full"]
        ros_pitching_runs = _pitching_runs_above_avg(ros, league_ra9)
        full_pitching_runs = _pitching_runs_above_avg(full, league_ra9)
        ros_replacement_runs = _pitching_replacement_runs(ros, league_ra9)
        full_replacement_runs = _pitching_replacement_runs(full, league_ra9)
        ros_war = (ros_pitching_runs + ros_replacement_runs) / RUNS_PER_WIN
        full_war = (full_pitching_runs + full_replacement_runs) / RUNS_PER_WIN
        pitcher_rows.append({
            "mlb_id": p["mlb_id"],
            "name": p["name"], "team": p["team"],
            "starts_made": p["gp"],
            "starts_remaining": round(p["gr"], 1),
            "ros_ip": round(ros["IP"], 1),   "full_ip": round(full["IP"], 1),
            "ros_k":  round(ros["K"], 1),    "full_k":  round(full["K"], 1),
            "ros_bb": round(ros["BB"], 1),   "full_bb": round(full["BB"], 1),
            "ros_h":  round(ros["H"], 1),    "full_h":  round(full["H"], 1),
            "ros_er": round(ros["ER"], 1),   "full_er": round(full["ER"], 1),
            "ros_ra9":  round(r, 2) if (r := _ra9(ros)) is not None else None,
            "full_ra9": round(r, 2) if (r := _ra9(full)) is not None else None,
            "ros_pitching_runs": round(ros_pitching_runs, 1),
            "full_pitching_runs": round(full_pitching_runs, 1),
            "ros_replacement_runs": round(ros_replacement_runs, 1),
            "full_replacement_runs": round(full_replacement_runs, 1),
            "ros_war": round(ros_war, 1), "full_war": round(full_war, 1),
            "ros_power_score": round(_pitcher_points(ros), 1),
            "full_power_score": round(_pitcher_points(full), 1),
        })
    pitcher_rows.sort(key=lambda r: r["ros_war"], reverse=True)
    for i, r in enumerate(pitcher_rows, start=1):
        r["rank"] = i

    # ------- AGES (for the 25-and-under view) -------
    if fetch_ages:
        _attach_ages(hitter_rows, TOP_N_HITTERS_FOR_AGE)
        _attach_ages(pitcher_rows, TOP_N_PITCHERS_FOR_AGE)
    else:
        for r in hitter_rows + pitcher_rows:
            r["age"] = None

    # ------- BUNDLE -------
    return {
        "as_of": (as_of or datetime.now(ET).strftime("%Y-%m-%d")),
        "season_games": season_games,
        "season_year": season_year,
        "age_cutoff": AGE_CUTOFF,
        "data_availability": {
            "defense": bool(defense_runs_by_player),
            "baserunning": bool(baserunning_data),
            "baserunning_weights_from_own_data": re_matrix_built,
        },
        "scoring": {
            "type": "war_v3_inhouse",
            "hitter": {
                "model": "Bat + Def + BsR + Replacement, / 10 runs per win",
                "batting_linear_weights": BATTING_LINEAR_WEIGHTS,
                "baserunning_weights_used": {"SB": round(sb_run_value, 3), "CS": round(cs_run_value, 3)},
                "baserunning_weights_from_own_run_expectancy_matrix": re_matrix_built,
                "replacement_runs_per_600pa": REPLACEMENT_RUNS_PER_600PA,
                "runs_per_win": RUNS_PER_WIN,
                "note": "Rest-of-season / full-season WAR projection. Bat "
                        "runs are self-consistent (above this hitter "
                        "pool's own PA-weighted average). Def comes from "
                        "an in-house model (exit velo + launch angle + "
                        "fielding zone -> expected run value, vs. what "
                        "actually happened) — no external leaderboard. "
                        "The model is trained ONCE PER SEASON on pooled "
                        "batted-ball data from this season + last season "
                        "and cached (models/defense_run_value_model.pkl), "
                        "then re-scored against just this season's own "
                        "chances every run, not retrained every run — "
                        "pass --retrain-defense to force a refresh. SB/CS "
                        "counts still come from the MLB Stats API "
                        "(official box score, refetched every run, "
                        f"limited to the top {TOP_N_HITTERS_FOR_BASERUNNING} "
                        "hitters by projected Bat runs to bound API calls) "
                        "but their run values are derived from a "
                        "run-expectancy matrix built from pooled "
                        "this-season + last-season play-by-play data, "
                        "also cached the same way "
                        "(models/baserunning_re_weights.json) — see "
                        "baserunning_weights_used above for what was "
                        "actually applied this run. Both Def and BsR "
                        "degrade gracefully (Def to 0, BsR weights to the "
                        "public fallback) if no cache exists yet and "
                        "local pitch data is missing or too thin; see "
                        "data_availability above.",
            },
            "pitcher": {
                "model": "(Pitching runs + Replacement) / 10 runs per win, RA9-based",
                "league_ra9_used": round(league_ra9, 3),
                "replacement_ra9_multiplier": PITCHER_REPLACEMENT_RA9_MULTIPLIER,
                "runs_per_win": RUNS_PER_WIN,
                "old_production_points_weights": PITCHER_WEIGHTS,
                "note": "Pitching runs above average = (league RA9 − "
                        "pitcher RA9) x (IP/9); league RA9 is computed "
                        "fresh from this pitcher pool every run "
                        "(self-consistent, no external constant). "
                        "Replacement level uses a standard published "
                        "sabermetric convention (~.380 win% ≈ allowing "
                        "runs at 1.28x league rate via the Pythagorean "
                        "win%-expectation relationship) rather than "
                        "something derived from this pipeline's data — "
                        "see replacement_ra9_multiplier above. This is "
                        "RA9-based, not FIP-based: the accuracy log "
                        "doesn't track home-runs-allowed or hit-by-pitch "
                        "for pitchers yet, which a proper FIP would "
                        "need, and 'ER' here is actually total runs "
                        "allowed (earned vs. unearned isn't split out). "
                        "The old production-points formula "
                        "(old_production_points_weights) is still "
                        "computed and shipped as ros/full_power_score "
                        "for reference, but WAR is now the primary "
                        "pitcher ranking metric.",
            },
        },
        "team_games_played": team_games,
        "hitters": hitter_rows,
        "pitchers": pitcher_rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--season-games", type=int, default=DEFAULT_SEASON_GAMES)
    ap.add_argument("--as-of", default=None)
    ap.add_argument("--out", default=str(OUT_JSON))
    ap.add_argument("--no-ages", action="store_true",
                    help="Skip MLB Stats API age fetch (25-and-under view will be empty).")
    ap.add_argument("--no-defense", action="store_true",
                    help="Skip training the in-house defense model (Def will be 0 for everyone).")
    ap.add_argument("--no-baserunning", action="store_true",
                    help="Skip the per-player SB/CS fetch + run-expectancy matrix (BsR will be 0 for everyone).")
    ap.add_argument("--retrain-defense", action="store_true",
                    help="Force retraining the in-house defense model and re-deriving the "
                         "baserunning run-expectancy weights instead of reusing the cached "
                         "models/defense_run_value_model.pkl / models/baserunning_re_weights.json "
                         "(these normally train once per season, not every run).")
    args = ap.parse_args()

    print(f"Building player WAR rankings (season_games={args.season_games}, "
          f"as_of={args.as_of or 'today'}, fetch_ages={not args.no_ages}, "
          f"fetch_defense={not args.no_defense}, "
          f"fetch_baserunning={not args.no_baserunning}, "
          f"retrain_defense={args.retrain_defense})...")
    bundle = build_rankings(season_games=args.season_games,
                             as_of=args.as_of,
                             fetch_ages=not args.no_ages,
                             fetch_defense=not args.no_defense,
                             fetch_baserunning_data=not args.no_baserunning,
                             retrain_models=args.retrain_defense)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(bundle, f, indent=2)

    print(f"Wrote bundle -> {out_path}")
    print(f"  {len(bundle['hitters'])} hitters / {len(bundle['pitchers'])} pitchers")
    print(f"  defense data available: {bundle['data_availability']['defense']}, "
          f"baserunning data available: {bundle['data_availability']['baserunning']}")
    n_young_h = sum(1 for r in bundle["hitters"] if r.get("age") and r["age"] <= AGE_CUTOFF)
    n_young_p = sum(1 for r in bundle["pitchers"] if r.get("age") and r["age"] <= AGE_CUTOFF)
    print(f"  {n_young_h} hitters / {n_young_p} pitchers age {AGE_CUTOFF} or under")

    print(f"\nTop 5 hitters by projected ROS WAR (full-season WAR alongside):")
    for r in bundle["hitters"][:5]:
        print(f"  {r['rank']:>2}. {r['name']:<25} {r['team']:<4}  "
              f"age={r.get('age','—')}  ROS_WAR={r['ros_war']:>5.1f}  FULL_WAR={r['full_war']:>5.1f}  "
              f"(bat={r['ros_bat_runs']:>5.1f} def={r['ros_def_runs']:>5.1f} bsr={r['ros_bsr_runs']:>4.1f})")
    print(f"\nTop 5 pitchers by projected ROS WAR (full-season WAR alongside):")
    for r in bundle["pitchers"][:5]:
        print(f"  {r['rank']:>2}. {r['name']:<25} {r['team']:<4}  "
              f"age={r.get('age','—')}  ROS_WAR={r['ros_war']:>5.1f}  FULL_WAR={r['full_war']:>5.1f}  "
              f"(RA9={r['ros_ra9']})")


if __name__ == "__main__":
    main()
