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
    Def   — fielding runs, from Statcast Outs Above Average (Baseball
            Savant) converted to runs at ~0.8 runs/OAA (a standard public
            approximation; OAA itself is a "plays," not "runs," scale),
            plus a standard positional adjustment (prorated by playing
            time) using the position Statcast credits the player at.
    BsR   — baserunning runs, from stolen bases and caught-stealing
            (SB * 0.20 - CS * 0.40, standard linear weight run values).
            Only computed for the top hitters by projected Bat runs —
            this is a per-player MLB Stats API lookup, bounded the same
            way the age lookup below is bounded, to keep the daily run
            fast. Everyone else defaults to BsR = 0.

    WAR = (Bat + Def + BsR + Replacement) / RUNS_PER_WIN
    Replacement = 20 runs / 600 PA (standard replacement-level constant).

This is a real WAR *structure* — three components combined into wins
above replacement — built on this pipeline's own projections plus two
public Statcast/MLB Stats API feeds, not a copy of any third party's
computed WAR. Def and BsR are best-effort: if Statcast or the Stats API
is unreachable on a given run, those components degrade to 0 for the
affected players rather than failing the whole build (see
`data_availability` in the output bundle).

PITCHERS keep the prior simple production-points formula — pitching WAR
normally runs through FIP/RA9, which needs earned-run and home-run-allowed
components this pipeline doesn't track yet. Not part of this rework.

    Pitcher: 3*IP + K - ER - H_allowed - BB

Per-game rates are taken from actual box-score results when a player has
played, projected otherwise, then scaled by each player's estimated games
remaining in the season.

Outputs: outputs/player_rankings.json
Player ages cached at data/player_ages.json (MLB Stats API, fetched lazily).
"""

from __future__ import annotations

import argparse
import json
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
ET = ZoneInfo("America/New_York")

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

# ---------------------------------------------------------------------------
# WAR model constants (standard public sabermetric linear weights /
# approximations — not scraped or copied from any third party's
# proprietary WAR calculation).
# ---------------------------------------------------------------------------
BATTING_LINEAR_WEIGHTS = {
    "BB": 0.69, "1B": 0.89, "2B": 1.27, "3B": 1.62, "HR": 2.10, "OUT": -0.25,
}
SB_RUN, CS_RUN = 0.20, -0.40
OAA_RUNS_PER_PLAY = 0.80
# Standard positional adjustments, runs per 162 team-games, prorated by
# each player's own participation share.
POSITIONAL_ADJUSTMENT_162 = {
    "C": 12.5, "SS": 7.5, "2B": 2.5, "3B": 2.5, "CF": 2.5,
    "LF": -7.5, "RF": -7.5, "1B": -12.5, "DH": -17.5,
}
# No "C" here on purpose — Baseball Savant's OAA leaderboard explicitly
# does not cover catchers (catcher defense runs through framing/blocking
# metrics Statcast doesn't expose via this endpoint), so requesting it
# always fails. See _defense_runs() for how catchers (and true DH's) are
# handled without an OAA record.
FIELDING_POSITIONS = ["1B", "2B", "3B", "SS", "LF", "CF", "RF"]
REPLACEMENT_RUNS_PER_600PA = 20.0
RUNS_PER_WIN = 10.0


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
# Defense (Statcast Outs Above Average) — best-effort, never raises
# ---------------------------------------------------------------------------
def fetch_oaa_by_player(year: int) -> dict[int, dict]:
    """{mlb_id: {"oaa": float, "position": str}} from Baseball Savant's
    Outs Above Average leaderboard, one call per fielding position.
    Best-effort: returns {} (or a partial dict) on any failure rather
    than raising, so a Statcast outage never blocks the rest of the
    pipeline. A player credited at more than one position is tagged with
    whichever position has the larger-magnitude OAA — an approximation,
    not a true "primary position" lookup.
    """
    out: dict[int, dict] = {}
    try:
        import pybaseball as pb
    except Exception as e:
        print(f"⚠️  pybaseball unavailable, skipping defense component: {e}")
        return out

    for pos in FIELDING_POSITIONS:
        try:
            df = pb.statcast_outs_above_average(year, pos, min_att=1)
        except Exception as e:
            print(f"⚠️  OAA fetch failed for pos={pos}: {e}")
            continue
        if df is None or df.empty:
            continue
        id_col = next((c for c in df.columns if c.lower() in ("player_id", "mlbid", "mlb_id")), None)
        oaa_col = next((c for c in df.columns if "outs_above_average" in c.lower()), None)
        if not id_col or not oaa_col:
            continue
        for _, row in df.iterrows():
            try:
                pid = int(row[id_col])
                oaa_val = float(row[oaa_col])
            except (TypeError, ValueError):
                continue
            prev = out.get(pid)
            if prev is None or abs(oaa_val) > abs(prev["oaa"]):
                out[pid] = {"oaa": oaa_val, "position": pos}
    return out


def _defense_runs(oaa_info: dict | None, season_gp: float, team_gp: int) -> float:
    """Season-to-date fielding + positional-adjustment runs, prorated by
    the player's own participation share of the team's games so far.

    Players with no OAA record default to Def = 0 (no adjustment either
    way), NOT a DH penalty. That group is a mix of true DH's (who should
    get the -17.5 DH penalty) and catchers (who should get the +12.5 C
    bonus) — Statcast's OAA leaderboard doesn't cover catchers at all, so
    there's no reliable way to tell the two apart from this feed alone.
    Defaulting to 0 avoids the worse error of mislabeling every catcher
    on the site as a DH and tanking their WAR; the tradeoff is that real
    DH's are under-penalized until a position feed is added."""
    if team_gp <= 0 or oaa_info is None:
        return 0.0
    fielding_runs = oaa_info["oaa"] * OAA_RUNS_PER_PLAY
    pos_adj = POSITIONAL_ADJUSTMENT_162.get(oaa_info["position"], 0.0) * (season_gp / 162.0)
    return fielding_runs + pos_adj


# ---------------------------------------------------------------------------
# Baserunning (SB/CS) — best-effort, bounded to the top N hitters
# ---------------------------------------------------------------------------
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
# Pitcher points (unchanged)
# ---------------------------------------------------------------------------
def _pitcher_points(ros: dict) -> float:
    return sum(ros.get(k, 0.0) * w for k, w in PITCHER_WEIGHTS.items())


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
                   fetch_baserunning_data: bool = True) -> dict:
    if not ACC_CSV.exists():
        raise FileNotFoundError(f"Accuracy log not found at {ACC_CSV}")

    df = pd.read_csv(ACC_CSV, low_memory=False)
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    if as_of:
        df = df[df["game_date"] <= pd.to_datetime(as_of)]

    team_games = _team_games_played(df)
    season_year = int(df["game_date"].dropna().dt.year.max()) if df["game_date"].notna().any() \
        else datetime.now(ET).year

    # ------- HITTERS: per-game rates + Bat runs -------
    hitters_df = df[df["player_type"].astype(str).str.lower() == "hitter"].copy()
    prelim = []  # (mlb_id, name, team, per_game, gp, gr, ros)
    for (mid, name), grp in hitters_df.groupby(["mlb_id", "player_name"]):
        team = str(grp["team"].dropna().iloc[-1]) if grp["team"].notna().any() else ""
        per_game = _hitter_per_game(grp)
        gp = per_game.pop("games_played")
        team_gp = team_games.get(team, max(gp, 1))
        gr = _remaining_games(gp, team_gp, season_games)
        ros = {k: per_game[k] * gr for k in per_game}
        prelim.append({
            "mlb_id": int(mid) if not pd.isna(mid) else None,
            "name": name, "team": team, "team_gp": team_gp,
            "per_game": per_game, "gp": gp, "gr": gr, "ros": ros,
        })

    league_bat_rate = _league_avg_batting_runs_per_pa([p["per_game"] for p in prelim])

    for p in prelim:
        ros_pa = p["ros"]["PA"]
        player_bat_rate = _batting_runs_per_pa(p["per_game"])
        p["bat_runs"] = (player_bat_rate - league_bat_rate) * ros_pa

    # Bound the per-player baserunning lookup to the top N by Bat runs —
    # this is what "power" hitters look like before defense/baserunning
    # are folded in, and keeps the MLB Stats API call volume sane.
    prelim.sort(key=lambda p: p["bat_runs"], reverse=True)
    baserunning_data = {}
    defense_data = {}
    if fetch_baserunning_data:
        top_ids = {p["mlb_id"] for p in prelim[:TOP_N_HITTERS_FOR_BASERUNNING] if p["mlb_id"] is not None}
        baserunning_data = fetch_baserunning(top_ids, season_year)
    if fetch_defense:
        defense_data = fetch_oaa_by_player(season_year)

    hitter_rows = []
    for p in prelim:
        mid, gp, gr, team_gp = p["mlb_id"], p["gp"], p["gr"], p["team_gp"]
        ros = p["ros"]

        br = baserunning_data.get(mid)
        if br and gp > 0:
            sb_rate, cs_rate = br["sb"] / gp, br["cs"] / gp
            bsr_runs = (SB_RUN * sb_rate + CS_RUN * cs_rate) * gr
        else:
            bsr_runs = 0.0

        def_info = defense_data.get(mid)
        season_def_runs = _defense_runs(def_info, gp, team_gp)
        def_runs = (season_def_runs / gp * gr) if gp > 0 else 0.0

        replacement_runs = REPLACEMENT_RUNS_PER_600PA * (ros["PA"] / 600.0)
        war = (p["bat_runs"] + def_runs + bsr_runs + replacement_runs) / RUNS_PER_WIN

        hitter_rows.append({
            "mlb_id": mid, "name": p["name"], "team": p["team"],
            "position": (def_info or {}).get("position"),
            "games_played": gp,
            "games_remaining": round(gr, 1),
            "ros_h":  round(ros["H"], 1),
            "ros_2b": round(ros["2B"], 1),
            "ros_3b": round(ros["3B"], 1),
            "ros_hr": round(ros["HR"], 1),
            "ros_r":  round(ros["R"], 1),
            "ros_rbi": round(ros["RBI"], 1),
            "ros_bb": round(ros["BB"], 1),
            "ros_k":  round(ros["K"], 1),
            "ros_pa": round(ros["PA"], 1),
            "bat_runs": round(p["bat_runs"], 1),
            "def_runs": round(def_runs, 1),
            "bsr_runs": round(bsr_runs, 1),
            "replacement_runs": round(replacement_runs, 1),
            "war": round(war, 1),
            "has_baserunning_data": mid in baserunning_data,
            "has_defense_data": def_info is not None,
        })
    hitter_rows.sort(key=lambda r: r["war"], reverse=True)
    for i, r in enumerate(hitter_rows, start=1):
        r["rank"] = i

    # ------- PITCHERS (unchanged formula) -------
    pitchers_df = df[df["player_type"].astype(str).str.lower() == "pitcher"].copy()
    pitcher_rows = []
    for (mid, name), grp in pitchers_df.groupby(["mlb_id", "player_name"]):
        team = str(grp["team"].dropna().iloc[-1]) if grp["team"].notna().any() else ""
        per_game = _pitcher_per_game(grp)
        gp = per_game.pop("games_played")
        team_gp = team_games.get(team, max(gp, 1))
        gr = _remaining_games(gp, team_gp, season_games)
        ros = {k: per_game[k] * gr for k in per_game}
        power_score = round(_pitcher_points(ros), 1)
        pitcher_rows.append({
            "mlb_id": int(mid) if not pd.isna(mid) else None,
            "name": name, "team": team,
            "starts_made": gp,
            "starts_remaining": round(gr, 1),
            "ros_ip": round(ros["IP"], 1),
            "ros_k":  round(ros["K"], 1),
            "ros_bb": round(ros["BB"], 1),
            "ros_h":  round(ros["H"], 1),
            "ros_er": round(ros["ER"], 1),
            "power_score": power_score,
        })
    pitcher_rows.sort(key=lambda r: r["power_score"], reverse=True)
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
            "defense": bool(defense_data),
            "baserunning": bool(baserunning_data),
        },
        "scoring": {
            "type": "war_v1",
            "hitter": {
                "model": "Bat + Def + BsR + Replacement, / 10 runs per win",
                "batting_linear_weights": BATTING_LINEAR_WEIGHTS,
                "baserunning_weights": {"SB": SB_RUN, "CS": CS_RUN},
                "oaa_runs_per_play": OAA_RUNS_PER_PLAY,
                "positional_adjustment_per_162": POSITIONAL_ADJUSTMENT_162,
                "replacement_runs_per_600pa": REPLACEMENT_RUNS_PER_600PA,
                "runs_per_win": RUNS_PER_WIN,
                "note": "Rest-of-season WAR projection. Bat runs are "
                        "self-consistent (above this hitter pool's own "
                        "PA-weighted average, from projected rest-of-season "
                        "counting stats). Def comes from Statcast Outs "
                        "Above Average; BsR is limited to the top "
                        f"{TOP_N_HITTERS_FOR_BASERUNNING} hitters by "
                        "projected Bat runs (bounded MLB Stats API calls) "
                        "— everyone else defaults to BsR = 0. Both degrade "
                        "gracefully to 0 if the upstream feed is down for "
                        "a given run; see data_availability above. Def "
                        "also defaults to 0 (no adjustment) for anyone "
                        "Statcast's OAA leaderboard doesn't cover — most "
                        "notably catchers, which that leaderboard excludes "
                        "entirely, so this is not a full defensive model "
                        "for the position.",
            },
            "pitcher": PITCHER_WEIGHTS,
            "pitcher_note": "Pitching WAR needs FIP/RA9 inputs (earned "
                             "runs, HR allowed) this pipeline doesn't "
                             "track yet, so pitchers keep the prior "
                             "simple production-points formula.",
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
                    help="Skip the Statcast OAA fetch (Def will be positional-adjustment only).")
    ap.add_argument("--no-baserunning", action="store_true",
                    help="Skip the per-player SB/CS fetch (BsR will be 0 for everyone).")
    args = ap.parse_args()

    print(f"Building player WAR rankings (season_games={args.season_games}, "
          f"as_of={args.as_of or 'today'}, fetch_ages={not args.no_ages}, "
          f"fetch_defense={not args.no_defense}, "
          f"fetch_baserunning={not args.no_baserunning})...")
    bundle = build_rankings(season_games=args.season_games,
                             as_of=args.as_of,
                             fetch_ages=not args.no_ages,
                             fetch_defense=not args.no_defense,
                             fetch_baserunning_data=not args.no_baserunning)

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

    print(f"\nTop 5 hitters by projected ROS WAR:")
    for r in bundle["hitters"][:5]:
        print(f"  {r['rank']:>2}. {r['name']:<25} {r['team']:<4}  "
              f"age={r.get('age','—')}  WAR={r['war']:>5.1f}  "
              f"(bat={r['bat_runs']:>5.1f} def={r['def_runs']:>5.1f} bsr={r['bsr_runs']:>4.1f})")
    print(f"\nTop 5 pitchers:")
    for r in bundle["pitchers"][:5]:
        print(f"  {r['rank']:>2}. {r['name']:<25} {r['team']:<4}  "
              f"age={r.get('age','—')}  power={r['power_score']:>5.1f}")


if __name__ == "__main__":
    main()
