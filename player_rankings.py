"""
player_rankings.py
===================
Season player power rankings for the Power Rankings tab.

HITTERS are ranked by a projected rest-of-season WAR built from three
separate sub-models. In real WAR these would be denominated in the same
currency (runs above average/replacement) and just added — no component
gets a "weight" multiplier, because a run is a run regardless of whether
it came from a double or a diving catch. THIS VERSION DELIBERATELY BREAKS
FROM THAT, per explicit direction: Def is weighted DEF_RUNS_WEIGHT (1.15x)
relative to Bat. There is NO positional adjustment — removed per explicit
direction (an earlier version applied one; see git history if you ever
want it back). That makes this a house-rules scoring system, not textbook
fWAR/bWAR — worth knowing if you're ever comparing these numbers to
FanGraphs or Baseball-Reference. In particular: without a positional
adjustment, a bat-only DH-type player's WAR is NOT penalized for
providing no defensive value most nights the way real WAR would penalize
it — Def itself (see below) still reflects only their thin, occasional
fielding chances.

    Bat   — real wOBA -> wRAA, the same two-step FanGraphs uses: wOBA
            (H/2B/3B/HR/BB weighted by WOBA_WEIGHTS, FanGraphs' own
            published Guts! constants) is a rate stat, not runs — it's
            converted to runs above average via wRAA = ((player wOBA −
            league wOBA) / WOBA_SCALE) * PA. League wOBA is the PA-
            weighted average across this same hitter pool (self-
            consistent — no external lgwOBA constant needed); WOBA_SCALE
            itself is a published external constant (see the comment
            above WOBA_WEIGHTS). HBP isn't tracked by this pipeline's
            box-score log so it's dropped from the wOBA numerator, and PA
            stands in for the technically-correct AB+BB-IBB+SF+HBP
            denominator — both standard, sub-1%-impact simplifications.
    Def   — fielding runs, from an in-house model that regresses exit
            velocity + launch angle + which fielding zone a ball was hit
            toward against this pipeline's own absolute run-value table
            (DEFENSE_RUN_VALUE_WEIGHTS — a separate table from Bat's wOBA
            weights; see the comment above WOBA_WEIGHTS for why they
            can't be shared), then for every batted ball attributes
            (expected − actual) to whichever player was standing at that
            position on that play. A fielder's Def is the empirical-Bayes
            -SHRUNK sum of that over their CURRENT season's worth of
            chances (see _shrink_defense_runs) — outperform the model for
            your zone, gain runs; underperform, lose them — but pulled
            toward 0 based on chance count, since a 3-feature model
            leaves a lot of per-play variance unexplained, and an UNshrunk
            sum over a partial season was mostly that noise, not real
            skill (confirmed: a zero-true-skill-difference simulation
            using this same per-play noise level produced top-of-pool
            values statistically indistinguishable from what the unshrunk
            version was crediting real fielders — see build_defense_model
            for the verification). This is an Outs-Above-Average-style
            proxy, not literal UZR/DRS — those are proprietary vendor
            metrics (BIS zone ratings for DRS, MLBAM positioning data for
            UZR) built on manually-charted data this pipeline has no
            access to; nothing public can reproduce them exactly. The
            underlying regression model
            is trained ONCE PER SEASON (not every run) on pooled batted-
            ball data from this season plus last season for a bigger,
            more stable sample, then cached to models/ and reused all
            season — see build_defense_model() and DEFENSE_MODEL_CACHE.
    BsR   — baserunning runs. Currently covers stolen-base value only
            (the wSB-equivalent piece of FanGraphs' BsR) — extra-bases-
            taken (UBR) and double-play avoidance (wGDP) are NOT
            implemented. wGDP needs a batter ID on every pitch to know
            who to charge/credit, and this pipeline's pitch_data_
            <year>.csv doesn't currently carry one (it has pitcher +
            fielder IDs and on-base runner IDs, but no batter column) —
            adding it upstream in the Statcast pull would make wGDP
            straightforward with the same run-expectancy-matrix machinery
            already built for SB/CS. UBR is harder even with a batter ID:
            it needs hit-location/difficulty context to judge what an
            "average" runner would have done on a given ball, which
            isn't reliable to infer from base-occupancy alone (FanGraphs
            itself doesn't publish play-by-play UBR data for this
            reason). The SB/CS counts still have to come from the MLB
            Stats API — that's just the official box score, nothing to
            model, and those are re-fetched every run since they change
            every day a player plays. What used to be fixed textbook
            weights (0.20 / -0.40) are now derived from a real run-
            expectancy matrix built from this season plus last season's
            own play-by-play base/out states — see
            build_run_expectancy_and_baserunning_weights(). Like the
            defense model, this derivation is cached and only runs once
            per season, not on every pipeline tick. SB/CS counts are
            only fetched for the top hitters by projected Bat runs
            (bounded MLB Stats API calls, same as the age lookup below);
            everyone else defaults to BsR = 0.
    WAR = (Bat + Def*DEF_RUNS_WEIGHT + BsR + Replacement) / RUNS_PER_WIN
    Replacement = 20 runs / 600 PA (standard replacement-level constant).

This is a WAR *structure* — three components combined into wins above
replacement — customized in one way per explicit direction: Def gets a
1.15x weight relative to Bat. Bat is computed fresh every run (it's
cheap — just this run's own projected stats). Def and the SB/CS run
values are trained/derived from this pipeline's own data too, but only
once per season (cached under models/, like the hitter/pitcher
projection models), not re-trained on every cron tick — see "Model
caching" below. Only the raw SB/CS event counts and player ages come
from an outside feed (MLB's official Stats API — factual box-score
data, not someone else's model). Def and BsR are both best-effort: if
the local pitch data or the Stats API is unavailable on a given run,
those components degrade to 0 for the affected players rather than
failing the whole build (see `data_availability` in the output bundle).

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

# Positional adjustment was removed per explicit direction (an earlier
# version applied a runs/162-games-by-position table on top of Def — see
# git history if you ever want it back). WAR below has no positional
# term at all: a bat-only DH-type player's WAR is NOT penalized for
# playing no real defense the way textbook WAR would penalize it.

# Def is weighted slightly higher than Bat, per explicit request — a
# deliberate departure from real WAR, where every component is worth
# exactly 1 run per run with no multiplier (that's what makes it WAR
# instead of a house-rules score in the first place). Applied directly
# to a player's measured Def runs before they're combined into WAR.
DEF_RUNS_WEIGHT = 1.15

# ---------------------------------------------------------------------------
# WAR model constants.
#
# Two SEPARATE weight tables, for two purposes that must not share one set
# of numbers (an earlier version of this file conflated them, which is what
# was inflating Bat runs — see below):
#
# WOBA_WEIGHTS / WOBA_SCALE: the real wOBA -> wRAA pipeline (Offense),
# published each year by FanGraphs' Guts! page (fangraphs.com/guts.aspx).
# Values below are the 2025 constants (checked via web search when this was
# implemented — re-check the Guts page each new season and update if they've
# moved). wOBA itself is just a rate stat: wOBA = (wBB*BB + w1B*1B + w2B*2B +
# w3B*3B + wHR*HR) / PA (HBP is dropped from the numerator — not tracked by
# this pipeline's box-score log — and PA is used in place of the technically
# -correct AB+BB-IBB+SF+HBP denominator for the same reason; both are
# standard, sub-1%-impact simplifications when HBP/SF/IBB aren't available).
# wOBA on its own is NOT in run units — converting it to actual runs above
# average (wRAA) requires dividing by WOBA_SCALE:
#     wRAA = ((player_wOBA - league_wOBA) / WOBA_SCALE) * PA
# league_wOBA is computed self-consistently from this pipeline's own hitter
# pool (PA-weighted average), same as the old league_bat_rate approach — no
# external lgwOBA constant needed. WOBA_SCALE itself, unlike lgwOBA, can't
# be self-derived without modeling the league's full run environment, so
# it's a published external constant, same footing as RUNS_PER_WIN below.
#
# DEFENSE_RUN_VALUE_WEIGHTS: true ABSOLUTE run values (classic Palmer-style
# Linear Weights — "runs above an out"), used ONLY as the training target
# for the in-house defense model (see build_defense_model / _outcome_run_
# value) — it needs every outcome, including an out, priced in the same
# real-runs units with no separate scale step, which is exactly what wOBA's
# coefficients are NOT designed for (they only mean something once divided
# by WOBA_SCALE, and even then have no "OUT" term at all — outs are implicit
# in wOBA, not priced directly). Mixing the two tables up — using wOBA's
# coefficients as if they were absolute run values, paired with a real Out
# value from this table — is exactly what was inflating Bat runs (and
# therefore WAR) for high-power/high-BB hitters before this fix: their hit
# events were valued ~1.5-2x too high relative to the Out penalty.
#
# REPLACEMENT_RUNS_PER_600PA and RUNS_PER_WIN are standard replacement-
# level / win-conversion constants (see the replacement-level derivation
# note further down for where the 20.0 figure comes from). SB_RUN / CS_RUN
# are only a FALLBACK — the real per-run values are derived from this
# pipeline's own play-by-play data (pooled across this season + last
# season) in build_run_expectancy_and_baserunning_weights(), cached, and
# reused all season; these fixed numbers only kick in if that derivation
# has never been able to run (pitch data missing / too thin, and no cache
# exists yet either).
# ---------------------------------------------------------------------------
WOBA_WEIGHTS = {
    "BB": 0.691, "1B": 0.882, "2B": 1.252, "3B": 1.584, "HR": 2.037,
}
WOBA_SCALE = 1.232
DEFENSE_RUN_VALUE_WEIGHTS = {
    "BB": 0.33, "1B": 0.47, "2B": 0.78, "3B": 1.09, "HR": 1.40, "OUT": -0.25,
}
FALLBACK_SB_RUN, FALLBACK_CS_RUN = 0.20, -0.40
# Replacement level ≈ 20 runs below average per 600 PA — the standard,
# widely-cited rounding of FanGraphs' actual derivation: a replacement-level
# team wins ~.294 (about 47-48 games/162), which nets to roughly 1,000 wins
# above replacement leaguewide/season, ~57% (570) of which goes to position
# players; spread across a league-PA-weighted share of that, a full-time
# 600-PA regular's slice comes out close to 18-20 runs depending on the
# season's exact run environment and total league PA — see
# https://library.fangraphs.com/misc/war/replacement-level/. Applied
# uniformly regardless of position — this pipeline has no positional-
# adjustment term at all (removed per explicit direction), unlike real
# WAR, which would normally handle position separately from replacement.
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
# Batting runs (offense component of WAR) — real wOBA -> wRAA, the same
# two-step FanGraphs uses (see the WOBA_WEIGHTS / WOBA_SCALE comment above).
# ---------------------------------------------------------------------------
def _woba(rate: dict) -> float:
    """wOBA for a per-game rate dict with H/2B/3B/HR/BB/PA keys (season-
    to-date or projected rate — a rate stat, so per-game vs per-season
    inputs give the same ratio). HBP is dropped from the numerator (not
    tracked by this pipeline's box-score log) and PA stands in for the
    technically-correct AB+BB-IBB+SF+HBP denominator (IBB/SF aren't
    tracked either) — both are standard, sub-1%-impact simplifications."""
    pa = rate.get("PA") or 0.0
    if pa <= 0:
        return 0.0
    h, doubles, triples, hr = rate.get("H", 0.0), rate.get("2B", 0.0), rate.get("3B", 0.0), rate.get("HR", 0.0)
    bb = rate.get("BB", 0.0)
    singles = max(0.0, h - doubles - triples - hr)
    w = WOBA_WEIGHTS
    numerator = (w["BB"] * bb + w["1B"] * singles + w["2B"] * doubles +
                 w["3B"] * triples + w["HR"] * hr)
    return numerator / pa


def _league_avg_woba(per_game_by_player: list[dict]) -> float:
    """PA-weighted league-average wOBA across the current hitter pool —
    makes "above average" self-consistent without needing an external
    lgwOBA constant (WOBA_SCALE still has to come from FanGraphs — see
    above — but the league baseline itself doesn't)."""
    total_num, total_pa = 0.0, 0.0
    for rate in per_game_by_player:
        pa = rate.get("PA") or 0.0
        if pa <= 0:
            continue
        total_num += _woba(rate) * pa
        total_pa += pa
    return (total_num / total_pa) if total_pa > 0 else 0.0


def _wraa_per_pa(rate: dict, league_woba: float) -> float:
    """wRAA per PA: (player wOBA − league wOBA) / WOBA_SCALE. The
    standard FanGraphs conversion from wOBA points (not run units) into
    actual runs above average."""
    return (_woba(rate) - league_woba) / WOBA_SCALE


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


def _shrink_defense_runs(scored: pd.DataFrame) -> dict[int, float]:
    """Empirical-Bayes (random-effects ANOVA) shrinkage of each fielder's
    raw summed per-play residual toward the pool's grand mean (≈0).

    Why this is necessary: the defense model has only 3 coarse features
    (exit velo, launch angle, hit_location) to predict a play's expected
    value — nowhere near enough to explain everything about whether a
    given ball becomes a hit or an out, so a large share of every single
    play's residual is irreducible noise, not fielder skill. Verified
    directly against this pipeline's own data: simulating a "zero true
    skill difference" null with the SAME per-play noise level and chance
    counts as the real scored data regularly produces top-of-pool summed
    values (~9-11 runs from 200 trials) statistically indistinguishable
    from what the unshrunk model was crediting real fielders — i.e. the
    raw sums were mostly noise dressed up as "runs saved." Real UZR/DRS
    are known to be this noisy over a single season too, which is why
    they get heavily regressed / why analysts caution against reading
    too much into one year of either.

    This uses the classic one-way random-effects ANOVA method-of-moments
    variance-component estimator to split "true between-fielder variance"
    from "within-fielder (noise) variance" directly from this season's
    own play-level residuals — self-consistent, no external constant,
    same "derive it from our own data" approach as the SB/CS run values.
    A fielder with more chances (more evidence) gets less shrinkage; one
    with few chances gets pulled hard toward 0, same as everyone else,
    rather than keeping whatever multiple of pure sampling luck they
    happened to run into.
    """
    g = scored.groupby("fielder_id")["fielder_run_value"]
    n = g.count()
    k = len(n)
    total_n = int(n.sum())
    if k < 2 or total_n <= k:
        return g.sum().to_dict()

    grand_mean = float(scored["fielder_run_value"].mean())
    group_mean = g.mean()
    group_var = g.var(ddof=1).fillna(0.0)  # 0 for any singleton-chance fielder

    ssb = float((n * (group_mean - grand_mean) ** 2).sum())
    ssw = float(((n - 1) * group_var).sum())
    df_b, df_w = k - 1, total_n - k

    msb = ssb / df_b if df_b > 0 else 0.0
    msw = ssw / df_w if df_w > 0 else float(scored["fielder_run_value"].var(ddof=1))
    # n0: the standard unequal-group-size correction factor for the ANOVA
    # method of moments (reduces to the common group size when n is equal).
    n0 = (total_n - float((n ** 2).sum()) / total_n) / df_b if df_b > 0 else float(n.mean())

    # True between-fielder variance — clipped at 0 (can't be negative;
    # a negative method-of-moments estimate just means the data can't
    # statistically distinguish any real skill differences at all this
    # season, in which case everyone shrinks fully to the grand mean).
    tau2 = max(0.0, (msb - msw) / n0) if n0 > 0 else 0.0

    if tau2 <= 0:
        return {int(fid): float(grand_mean * cnt) for fid, cnt in n.items()}

    reliability = tau2 / (tau2 + msw / n)
    shrunk_mean = grand_mean + reliability * (group_mean - grand_mean)
    shrunk_sum = shrunk_mean * n
    return shrunk_sum.to_dict()


def _outcome_run_value(events) -> float | None:
    """Maps a Statcast `events` value to this file's own absolute-run-
    value table (DEFENSE_RUN_VALUE_WEIGHTS — NOT the wOBA weights used
    for Bat runs; see the comment above WOBA_WEIGHTS/DEFENSE_RUN_VALUE_
    WEIGHTS for why those two tables have to stay separate), or None for
    outcomes we don't want in the defense model's training set (still
    mid-PA, ambiguous, or too rare to trust)."""
    if events in _HIT_EVENT_WEIGHT_KEY:
        return DEFENSE_RUN_VALUE_WEIGHTS[_HIT_EVENT_WEIGHT_KEY[events]]
    if events == "field_error":
        # Batter reaches, most commonly at 1st — treated as roughly
        # single-equivalent. An approximation; the alternative (excluding
        # errors from training entirely) would bias the model toward
        # thinking every hard-hit ball in that zone was fielded cleanly.
        return DEFENSE_RUN_VALUE_WEIGHTS["1B"]
    if events in _OUT_EVENTS:
        return DEFENSE_RUN_VALUE_WEIGHTS["OUT"]
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
    DEFENSE_RUN_VALUE_WEIGHTS as the target), and credits the residual
    (expected − actual) to whichever player was standing at that
    position on that specific play (via the fielder_<N> / pitcher
    columns Statcast already tags each pitch with). A fielder's season
    Def is the empirical-Bayes-SHRUNK sum of their residuals (see
    _shrink_defense_runs) — save more hits than the model expects for
    that zone/contact quality, gain runs; allow more, lose them — but
    pulled toward 0 in proportion to how few chances back it up, since a
    3-feature model has a lot of per-play noise it can't explain and an
    unshrunk sum over a partial season is mostly that noise, not skill
    (verified: a zero-true-skill simulation using this same per-play
    noise level and chance counts produced top-of-pool sums statistically
    indistinguishable from what the unshrunk version credited real
    players).

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

    defense_runs = _shrink_defense_runs(scored)
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
# Official games-played-by-position from the MLB Stats API, bounded to the
# same top-N pool as baserunning. NOTE: this no longer feeds a positional-
# adjustment WAR component (removed per explicit direction) — it's kept
# only to label each hitter with their real primary position (more
# reliable than build_defense_model's Statcast-chance-based guess, which
# undercounts a good defender who simply doesn't get many balls hit their
# way).
# ---------------------------------------------------------------------------
def fetch_position_games(ids: set[int], season: int, throttle: float = 0.05) -> dict[int, dict[str, float]]:
    """{mlb_id: {position_abbrev: games_played}} via the MLB Stats API
    season fielding-stats-by-position endpoint — OFFICIAL games-played
    counts (one split per position a player appeared at), not inferred
    from Statcast batted-ball chances the way build_defense_model's
    primary_position guess is (which undercounts a good defender who
    simply doesn't get many balls hit their way). This endpoint DOES
    include an explicit "DH" split (confirmed against real data — 0
    fielding chances, but a real gamesPlayed count), so use it directly
    downstream rather than re-deriving DH games from (total games played)
    − (sum of fielded games), which would double-count the DH split's
    own games into "fielded" and wipe out the real number. No persistent
    cache, same reasoning as fetch_baserunning: this changes every day a
    player plays. Best-effort per player."""
    out: dict[int, dict[str, float]] = {}
    for mid in ids:
        url = (f"https://statsapi.mlb.com/api/v1/people/{int(mid)}/stats"
               f"?stats=season&group=fielding&season={season}")
        try:
            r = requests.get(url, timeout=8)
            r.raise_for_status()
            data = r.json()
            splits = (data.get("stats") or [{}])[0].get("splits") or []
            by_pos: dict[str, float] = {}
            for s in splits:
                pos = ((s.get("position") or {}).get("abbreviation") or "").upper()
                games = float((s.get("stat") or {}).get("gamesPlayed") or 0.0)
                if pos and games:
                    by_pos[pos] = by_pos.get(pos, 0.0) + games
            if by_pos:
                out[int(mid)] = by_pos
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
    _league_avg_woba for hitters."""
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
                   fetch_official_positions: bool = True,
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

    league_woba = _league_avg_woba([p["per_game"] for p in prelim])

    for p in prelim:
        wraa_rate = _wraa_per_pa(p["per_game"], league_woba)
        p["ros_bat_runs"] = wraa_rate * p["ros"]["PA"]
        p["full_bat_runs"] = wraa_rate * p["full"]["PA"]

    # Bound the per-player baserunning / positional-adjustment lookups to
    # the top N by Bat runs — this is what "power" hitters look like
    # before defense/baserunning/position are folded in, and keeps the
    # MLB Stats API call volume sane.
    prelim.sort(key=lambda p: p["ros_bat_runs"], reverse=True)
    top_ids = {p["mlb_id"] for p in prelim[:TOP_N_HITTERS_FOR_BASERUNNING] if p["mlb_id"] is not None}

    baserunning_data = {}
    sb_run_value, cs_run_value, re_matrix_built = FALLBACK_SB_RUN, FALLBACK_CS_RUN, False
    if fetch_baserunning_data:
        baserunning_data = fetch_baserunning(top_ids, season_year)
        sb_run_value, cs_run_value, re_matrix_built = build_run_expectancy_and_baserunning_weights(
            season_year, retrain=retrain_models)

    defense_runs_by_player, position_by_player = {}, {}
    if fetch_defense:
        defense_runs_by_player, position_by_player = build_defense_model(
            season_year, retrain=retrain_models)

    position_games_data = {}
    if fetch_official_positions:
        position_games_data = fetch_position_games(top_ids, season_year)

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

        # DEF_RUNS_WEIGHT applied here (not to the cached raw model output)
        # so the weight is easy to find/adjust and every downstream use of
        # Def runs (WAR, the displayed Def column) reflects it consistently.
        season_def_runs = defense_runs_by_player.get(mid, 0.0) * DEF_RUNS_WEIGHT
        ros_def_runs = (season_def_runs / gp * gr) if gp > 0 else 0.0
        full_def_runs = season_def_runs + ros_def_runs

        # No positional-adjustment WAR component (removed per explicit
        # direction — WAR here is Bat + Def*DEF_RUNS_WEIGHT + BsR +
        # Replacement, nothing else). position_games_data is still fetched
        # (see fetch_official_positions) purely to label each hitter with
        # their real primary position below.
        ros_replacement_runs = REPLACEMENT_RUNS_PER_600PA * (ros["PA"] / 600.0)
        full_replacement_runs = REPLACEMENT_RUNS_PER_600PA * (full["PA"] / 600.0)
        ros_war = (p["ros_bat_runs"] + ros_def_runs + ros_bsr_runs
                  + ros_replacement_runs) / RUNS_PER_WIN
        full_war = (p["full_bat_runs"] + full_def_runs + full_bsr_runs
                   + full_replacement_runs) / RUNS_PER_WIN

        # Prefer the MLB Stats API's official primary position (most games
        # actually played there) over the defense model's Statcast-chance
        # inference — official is more reliable and this pool already has
        # it (fetched above for exactly this purpose).
        raw_positions = position_games_data.get(mid, {})
        if raw_positions:
            official_position = max(raw_positions.items(), key=lambda kv: kv[1])[0]
        else:
            official_position = None

        hitter_rows.append({
            "mlb_id": mid, "name": p["name"], "team": p["team"],
            "position": official_position or position_by_player.get(mid),
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
            "official_position_labels": bool(position_games_data),
        },
        "scoring": {
            "type": "war_v6_custom",
            "hitter": {
                "model": "Bat + Def*1.15 + BsR + Replacement, / 10 runs per win",
                "customized": True,
                "def_runs_weight": DEF_RUNS_WEIGHT,
                "woba_weights": WOBA_WEIGHTS,
                "woba_scale": WOBA_SCALE,
                "defense_run_value_weights": DEFENSE_RUN_VALUE_WEIGHTS,
                "baserunning_weights_used": {"SB": round(sb_run_value, 3), "CS": round(cs_run_value, 3)},
                "baserunning_weights_from_own_run_expectancy_matrix": re_matrix_built,
                "replacement_runs_per_600pa": REPLACEMENT_RUNS_PER_600PA,
                "runs_per_win": RUNS_PER_WIN,
                "note": "Rest-of-season / full-season WAR-style projection "
                        "— CUSTOMIZED, not textbook WAR: Def is weighted "
                        f"{DEF_RUNS_WEIGHT}x relative to Bat (def_runs_"
                        "weight above), and there is NO positional "
                        "adjustment (removed per explicit direction — an "
                        "earlier version had one). That means a bat-only "
                        "DH-type player's WAR is not penalized for playing "
                        "no real defense the way textbook WAR would "
                        "penalize it; official_position_labels above still "
                        "gets fetched, but purely to label each hitter "
                        "with their real primary position, not to adjust "
                        "their score. Bat runs are real wOBA -> wRAA "
                        "(woba_weights / woba_scale "
                        "above are FanGraphs' published 2025 Guts! "
                        "constants — re-check fangraphs.com/guts.aspx "
                        "each season), compared against this hitter "
                        "pool's own PA-weighted average wOBA (self-"
                        "consistent, no external league-average needed). "
                        "BsR currently covers stolen-base value only "
                        "(wSB-equivalent) — extra-bases-taken (UBR) and "
                        "double-play avoidance (wGDP) are NOT included: "
                        "wGDP needs a batter ID on every pitch, which "
                        "this pipeline's pitch_data_<year>.csv does not "
                        "currently capture, and UBR needs hit-location/"
                        "difficulty context beyond what's reliable to "
                        "infer from base-occupancy alone. Def comes from "
                        "an in-house model (exit velo + launch angle + "
                        "fielding zone -> expected run value, vs. what "
                        "actually happened) — an Outs-Above-Average-style "
                        "proxy, not literal UZR/DRS (those are proprietary "
                        "vendor metrics built on manually-charted zone/"
                        "difficulty data this pipeline has no access to). "
                        "That raw per-play residual is then shrunk toward "
                        "0 with empirical-Bayes/random-effects regression "
                        "based on each fielder's chance count this season "
                        "(see _shrink_defense_runs) — without it, a "
                        "3-feature model's per-play noise summed over a "
                        "partial season regularly produced 15-20+ run "
                        "'Def' totals that were mostly noise, not skill "
                        "(verified against a zero-true-skill simulation "
                        "using this pipeline's own observed noise level). "
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
                        "actually applied this run. Def and BsR both "
                        "degrade gracefully to 0 (BsR weights to the "
                        "public fallback) if no cache exists yet and "
                        "local pitch data / API calls are unavailable; "
                        "see data_availability above.",
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
    ap.add_argument("--no-official-positions", action="store_true",
                    help="Skip the per-player games-by-position fetch (position labels fall "
                         "back to the defense model's Statcast-chance-based guess). Purely "
                         "cosmetic — there's no positional-adjustment WAR component to affect.")
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
          f"fetch_official_positions={not args.no_official_positions}, "
          f"retrain_defense={args.retrain_defense})...")
    bundle = build_rankings(season_games=args.season_games,
                             as_of=args.as_of,
                             fetch_ages=not args.no_ages,
                             fetch_defense=not args.no_defense,
                             fetch_baserunning_data=not args.no_baserunning,
                             fetch_official_positions=not args.no_official_positions,
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
        print(f"  {r['rank']:>2}. {r['name']:<25} {r['team']:<4} {r.get('position') or '—':<3} "
              f"age={r.get('age','—')}  ROS_WAR={r['ros_war']:>5.1f}  FULL_WAR={r['full_war']:>5.1f}  "
              f"(bat={r['ros_bat_runs']:>5.1f} def={r['ros_def_runs']:>5.1f} "
              f"bsr={r['ros_bsr_runs']:>4.1f})")
    print(f"\nTop 5 pitchers by projected ROS WAR (full-season WAR alongside):")
    for r in bundle["pitchers"][:5]:
        print(f"  {r['rank']:>2}. {r['name']:<25} {r['team']:<4}  "
              f"age={r.get('age','—')}  ROS_WAR={r['ros_war']:>5.1f}  FULL_WAR={r['full_war']:>5.1f}  "
              f"(RA9={r['ros_ra9']})")


if __name__ == "__main__":
    main()
