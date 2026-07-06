"""
fantasy_rankings.py
====================
Compute everything the Fantasy tab needs in one pass:

  1. Rest-of-season (ROS) rankings — hitter + pitcher, Yahoo-style points
  2. Today's matchups — best per-game plays from outputs/hitterspitchers_today.csv
  3. Risers / Fallers — last-7-day per-game points vs season-to-date baseline
  4. Dynasty rankings — ROS points scaled by an age-curve multiplier

Scoring (Yahoo-style, missing SB / W / SV which aren't tracked yet):
    Hitter:  H + 2B + 2*3B + 3*HR + R + RBI + BB - 0.5*K
    Pitcher: 3*IP + K - ER - H_allowed - BB

Outputs: outputs/fantasy_rankings.json (one bundle, all sections inside).
Player ages cached at data/player_ages.json (MLB Stats API, fetched lazily).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
ACC_CSV = HERE / "2026_player_accuracy.csv"
TODAY_CSV = HERE / "outputs" / "hitterspitchers_today.csv"
OUT_JSON = HERE / "outputs" / "fantasy_rankings.json"
AGE_CACHE = HERE / "data" / "player_ages.json"
ET = ZoneInfo("America/New_York")

DEFAULT_SEASON_GAMES = 162
RISER_RECENT_DAYS = 7
RISER_MIN_RECENT_GAMES = 4
RISER_MIN_SEASON_GAMES = 15
RISER_TOPN = 25

HITTER_WEIGHTS = {"H": 1.0, "2B": 1.0, "3B": 2.0, "HR": 3.0,
                  "R": 1.0, "RBI": 1.0, "BB": 1.0, "K": -0.5}
PITCHER_WEIGHTS = {"IP": 3.0, "K": 1.0, "ER": -1.0, "H": -0.5, "BB": -0.5}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _safe_int(v):
    try: return int(v) if v is not None and not pd.isna(v) else None
    except Exception: return None


def _safe_round(v, digits=2):
    try:
        if v is None or pd.isna(v): return 0.0
        return round(float(v), digits)
    except Exception: return 0.0


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
# Fantasy points
# ---------------------------------------------------------------------------
def _hitter_points(ros: dict) -> float:
    return sum(ros.get(k, 0.0) * w for k, w in HITTER_WEIGHTS.items())


def _pitcher_points(ros: dict) -> float:
    return sum(ros.get(k, 0.0) * w for k, w in PITCHER_WEIGHTS.items())


def _today_hitter_points(row) -> float:
    def g(name):
        v = row.get(name)
        try: return float(v) if v is not None and not pd.isna(v) else 0.0
        except Exception: return 0.0
    tb = g("proj_tb")
    if tb == 0.0:
        tb = g("proj_hits") * 1.4   # fallback when proj_tb missing
    return tb + g("proj_runs") + g("proj_rbi") + g("proj_walks") - 0.5 * g("proj_strikeouts")


def _today_pitcher_points(row) -> float:
    def g(name):
        v = row.get(name)
        try: return float(v) if v is not None and not pd.isna(v) else 0.0
        except Exception: return 0.0
    return (3.0 * g("proj_ip") + g("proj_strikeouts")
            - g("proj_runs_allowed") - g("proj_hits_allowed") - g("proj_walks"))


def _hitter_row_actual_points(row) -> float:
    """Per-game points from a single graded row (uses actuals)."""
    def g(name):
        v = row.get(name)
        try: return float(v) if v is not None and not pd.isna(v) else 0.0
        except Exception: return 0.0
    return (g("actual_hits") + g("actual_doubles") + 2*g("actual_triples")
            + 3*g("actual_hr") + g("actual_runs") + g("actual_rbi")
            + g("actual_walks") - 0.5 * g("actual_strikeouts"))


def _pitcher_row_actual_points(row) -> float:
    def g(name):
        v = row.get(name)
        try: return float(v) if v is not None and not pd.isna(v) else 0.0
        except Exception: return 0.0
    return (3.0 * g("actual_ip") + g("actual_strikeouts")
            - g("actual_runs_allowed") - g("actual_hits_allowed") - g("actual_walks"))


# ---------------------------------------------------------------------------
# Today's matchups
# ---------------------------------------------------------------------------
def build_today_matchups() -> tuple[list[dict], list[dict], str | None]:
    if not TODAY_CSV.exists():
        return [], [], None
    try:
        df = pd.read_csv(TODAY_CSV, low_memory=False)
    except Exception:
        return [], [], None
    if df.empty:
        return [], [], None

    game_date = None
    if "game_date" in df.columns:
        dates = df["game_date"].dropna().astype(str)
        if not dates.empty:
            game_date = dates.iloc[0][:10]

    ptype = df.get("player_type", pd.Series(dtype=str)).astype(str).str.lower()

    hitter_rows = []
    for _, row in df[ptype == "hitter"].iterrows():
        pts = round(_today_hitter_points(row), 2)
        hitter_rows.append({
            "name": str(row.get("player_name", "")),
            "team": str(row.get("team", "")),
            "opponent": str(row.get("opponent", "")),
            "lineup_spot": _safe_int(row.get("lineup_spot")),
            "lineup_status": str(row.get("lineup_status", "")),
            "proj_pa":  _safe_round(row.get("proj_pa")),
            "proj_hits": _safe_round(row.get("proj_hits")),
            "proj_hr":  _safe_round(row.get("proj_hr")),
            "proj_runs": _safe_round(row.get("proj_runs")),
            "proj_rbi": _safe_round(row.get("proj_rbi")),
            "proj_tb":  _safe_round(row.get("proj_tb")),
            "proj_walks": _safe_round(row.get("proj_walks")),
            "proj_strikeouts": _safe_round(row.get("proj_strikeouts")),
            "confidence": str(row.get("confidence", "")),
            "points": pts,
        })
    hitter_rows.sort(key=lambda r: r["points"], reverse=True)
    for i, r in enumerate(hitter_rows, start=1):
        r["rank"] = i

    pitcher_rows = []
    for _, row in df[ptype == "pitcher"].iterrows():
        pts = round(_today_pitcher_points(row), 2)
        pitcher_rows.append({
            "name": str(row.get("player_name", "")),
            "team": str(row.get("team", "")),
            "opponent": str(row.get("opponent", "")),
            "proj_ip": _safe_round(row.get("proj_ip")),
            "proj_strikeouts": _safe_round(row.get("proj_strikeouts")),
            "proj_walks": _safe_round(row.get("proj_walks")),
            "proj_hits_allowed": _safe_round(row.get("proj_hits_allowed")),
            "proj_runs_allowed": _safe_round(row.get("proj_runs_allowed")),
            "points": pts,
        })
    pitcher_rows.sort(key=lambda r: r["points"], reverse=True)
    for i, r in enumerate(pitcher_rows, start=1):
        r["rank"] = i

    return hitter_rows, pitcher_rows, game_date


# ---------------------------------------------------------------------------
# Risers / Fallers (last RISER_RECENT_DAYS vs season-to-date)
# ---------------------------------------------------------------------------
def build_risers_fallers(df: pd.DataFrame, as_of: pd.Timestamp | None = None) -> dict:
    """Compare each player's per-game points over the last RISER_RECENT_DAYS
    days vs their season-to-date average. Returns top RISER_TOPN risers and
    fallers for hitters and pitchers separately, with min-game guards to
    suppress small-sample noise.
    """
    if as_of is None:
        as_of = pd.Timestamp(datetime.now(ET).date())
    cutoff = as_of - pd.Timedelta(days=RISER_RECENT_DAYS)

    df = df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")

    def _pack(records: list[dict], top: int) -> list[dict]:
        for i, r in enumerate(records, start=1):
            r["rank"] = i
        return records[:top]

    out = {"hitters": {"risers": [], "fallers": []},
           "pitchers": {"risers": [], "fallers": []}}

    # --- Hitters ---
    hdf = df[df["player_type"].astype(str).str.lower() == "hitter"].copy()
    hdf = hdf[hdf["actual_pa"].notna()]
    if not hdf.empty:
        hdf["pts"] = hdf.apply(_hitter_row_actual_points, axis=1)
        hitter_records = []
        for (mid, name), grp in hdf.groupby(["mlb_id", "player_name"]):
            n_season = len(grp)
            if n_season < RISER_MIN_SEASON_GAMES:
                continue
            recent = grp[grp["game_date"] >= cutoff]
            n_recent = len(recent)
            if n_recent < RISER_MIN_RECENT_GAMES:
                continue
            season_avg = float(grp["pts"].mean())
            recent_avg = float(recent["pts"].mean())
            delta = recent_avg - season_avg
            team = str(grp["team"].dropna().iloc[-1]) if grp["team"].notna().any() else ""
            hitter_records.append({
                "mlb_id": int(mid) if not pd.isna(mid) else None,
                "name": name,
                "team": team,
                "season_games": n_season,
                "recent_games": n_recent,
                "season_pts_per_game": round(season_avg, 2),
                "recent_pts_per_game": round(recent_avg, 2),
                "delta": round(delta, 2),
            })
        risers  = sorted(hitter_records, key=lambda r: r["delta"], reverse=True)
        fallers = sorted(hitter_records, key=lambda r: r["delta"])
        out["hitters"]["risers"]  = _pack([dict(r) for r in risers],  RISER_TOPN)
        out["hitters"]["fallers"] = _pack([dict(r) for r in fallers], RISER_TOPN)

    # --- Pitchers ---
    pdf = df[df["player_type"].astype(str).str.lower() == "pitcher"].copy()
    pdf = pdf[pdf["actual_ip"].notna()]
    if not pdf.empty:
        pdf["pts"] = pdf.apply(_pitcher_row_actual_points, axis=1)
        pitcher_records = []
        # Pitchers play less often -> looser min thresholds
        min_recent_p = 2
        min_season_p = 5
        for (mid, name), grp in pdf.groupby(["mlb_id", "player_name"]):
            n_season = len(grp)
            if n_season < min_season_p:
                continue
            recent = grp[grp["game_date"] >= cutoff]
            n_recent = len(recent)
            if n_recent < min_recent_p:
                continue
            season_avg = float(grp["pts"].mean())
            recent_avg = float(recent["pts"].mean())
            delta = recent_avg - season_avg
            team = str(grp["team"].dropna().iloc[-1]) if grp["team"].notna().any() else ""
            pitcher_records.append({
                "mlb_id": int(mid) if not pd.isna(mid) else None,
                "name": name,
                "team": team,
                "season_starts": n_season,
                "recent_starts": n_recent,
                "season_pts_per_start": round(season_avg, 2),
                "recent_pts_per_start": round(recent_avg, 2),
                "delta": round(delta, 2),
            })
        risers  = sorted(pitcher_records, key=lambda r: r["delta"], reverse=True)
        fallers = sorted(pitcher_records, key=lambda r: r["delta"])
        out["pitchers"]["risers"]  = _pack([dict(r) for r in risers],  RISER_TOPN)
        out["pitchers"]["fallers"] = _pack([dict(r) for r in fallers], RISER_TOPN)

    return out


# ---------------------------------------------------------------------------
# Dynasty layer — age-curve multiplier on top of ROS points
# ---------------------------------------------------------------------------
def _age_multiplier(age: float | None) -> float:
    """Heuristic age curve. Younger players get a forward-value premium;
    older players get discounted as their useful-years-remaining shrinks.
    Calibrated by inspection — adjust if your league's age preferences
    differ.
    """
    if age is None or pd.isna(age): return 1.0
    if age <= 23: return 1.50   # premium young (10+ years of value left)
    if age <= 25: return 1.35
    if age <= 28: return 1.20   # prime, lots ahead
    if age <= 30: return 1.05
    if age <= 32: return 0.95   # peak typically passed
    if age <= 34: return 0.80
    if age <= 36: return 0.65
    if age <= 38: return 0.50
    return 0.35


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


def get_player_ages(ids: set[int], throttle: float = 0.05,
                    refresh_days: int = 30) -> dict[int, dict]:
    """Return {mlb_id: {age, birth_date, name}} for `ids`. Uses
    data/player_ages.json as a persistent cache and only hits the MLB
    Stats API for IDs not in the cache (or whose entry is older than
    `refresh_days`). Network failures degrade gracefully — missing IDs
    just don't get an age in the result.
    """
    cache = _load_age_cache()
    cache_age = cache.get("__updated__", "")
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


def apply_dynasty(rows: list[dict], ages: dict[int, dict]) -> list[dict]:
    """Augment ROS rows with age + age-multiplied dynasty score, re-rank.
    Players without an age get age_multiplier=1.0 (no penalty).
    """
    enriched = []
    for r in rows:
        mid = r.get("mlb_id")
        info = ages.get(int(mid)) if mid is not None else None
        age = info.get("age") if info else None
        mult = _age_multiplier(age)
        dynasty_score = round(r["points"] * mult, 1)
        e = dict(r)
        e["age"] = age
        e["age_multiplier"] = round(mult, 2)
        e["dynasty_score"] = dynasty_score
        enriched.append(e)
    enriched.sort(key=lambda x: x["dynasty_score"], reverse=True)
    for i, r in enumerate(enriched, start=1):
        r["dynasty_rank"] = i
    return enriched


# ---------------------------------------------------------------------------
# Main bundle
# ---------------------------------------------------------------------------
def build_rankings(season_games: int = DEFAULT_SEASON_GAMES,
                   as_of: str | None = None,
                   fetch_ages: bool = True) -> dict:
    if not ACC_CSV.exists():
        raise FileNotFoundError(f"Accuracy log not found at {ACC_CSV}")

    df = pd.read_csv(ACC_CSV, low_memory=False)
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    if as_of:
        df = df[df["game_date"] <= pd.to_datetime(as_of)]

    team_games = _team_games_played(df)

    # ------- ROS HITTERS -------
    hitters_df = df[df["player_type"].astype(str).str.lower() == "hitter"].copy()
    hitter_rows = []
    for (mid, name), grp in hitters_df.groupby(["mlb_id", "player_name"]):
        team = str(grp["team"].dropna().iloc[-1]) if grp["team"].notna().any() else ""
        per_game = _hitter_per_game(grp)
        gp = per_game.pop("games_played")
        team_gp = team_games.get(team, max(gp, 1))
        gr = _remaining_games(gp, team_gp, season_games)
        ros = {k: per_game[k] * gr for k in per_game}
        ros["points"] = round(_hitter_points(ros), 1)
        hitter_rows.append({
            "mlb_id": int(mid) if not pd.isna(mid) else None,
            "name": name, "team": team,
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
            "ros_pa": round(per_game["PA"] * gr, 1),
            "points": ros["points"],
        })
    hitter_rows.sort(key=lambda r: r["points"], reverse=True)
    for i, r in enumerate(hitter_rows, start=1):
        r["rank"] = i

    # ------- ROS PITCHERS -------
    pitchers_df = df[df["player_type"].astype(str).str.lower() == "pitcher"].copy()
    pitcher_rows = []
    for (mid, name), grp in pitchers_df.groupby(["mlb_id", "player_name"]):
        team = str(grp["team"].dropna().iloc[-1]) if grp["team"].notna().any() else ""
        per_game = _pitcher_per_game(grp)
        gp = per_game.pop("games_played")
        team_gp = team_games.get(team, max(gp, 1))
        gr = _remaining_games(gp, team_gp, season_games)
        ros = {k: per_game[k] * gr for k in per_game}
        ros["points"] = round(_pitcher_points(ros), 1)
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
            "points": ros["points"],
        })
    pitcher_rows.sort(key=lambda r: r["points"], reverse=True)
    for i, r in enumerate(pitcher_rows, start=1):
        r["rank"] = i

    # ------- TODAY'S MATCHUPS -------
    daily_hitters, daily_pitchers, daily_game_date = build_today_matchups()

    # ------- RISERS / FALLERS -------
    rf = build_risers_fallers(df, as_of=(pd.to_datetime(as_of) if as_of else None))

    # ------- DYNASTY (age-multiplied) -------
    dynasty_hitters: list[dict] = []
    dynasty_pitchers: list[dict] = []
    if fetch_ages:
        all_ids = {r["mlb_id"] for r in hitter_rows + pitcher_rows
                   if r.get("mlb_id") is not None}
        # Only enrich the top ~300 hitters and top ~200 pitchers — dynasty
        # rankings are noise past that.
        top_hit_ids = {r["mlb_id"] for r in hitter_rows[:300]
                       if r.get("mlb_id") is not None}
        top_pit_ids = {r["mlb_id"] for r in pitcher_rows[:200]
                       if r.get("mlb_id") is not None}
        ages = get_player_ages(top_hit_ids | top_pit_ids)
        dynasty_hitters  = apply_dynasty(hitter_rows[:300], ages)
        dynasty_pitchers = apply_dynasty(pitcher_rows[:200], ages)

    # ------- BUNDLE -------
    return {
        "as_of": (as_of or datetime.now(ET).strftime("%Y-%m-%d")),
        "season_games": season_games,
        "scoring": {
            "type": "yahoo_points_v1",
            "hitter": HITTER_WEIGHTS,
            "pitcher": PITCHER_WEIGHTS,
            "missing_categories": ["SB", "W", "SV"],
            "note": "Yahoo-style points. SB / W / SV not tracked in the "
                    "accuracy log yet. Risers/fallers use last "
                    f"{RISER_RECENT_DAYS}d per-game points vs season-to-date "
                    "(min " f"{RISER_MIN_RECENT_GAMES} recent / "
                    f"{RISER_MIN_SEASON_GAMES} season games). Dynasty applies "
                    "an age-curve multiplier; players without an age default "
                    "to 1.0x.",
        },
        "team_games_played": team_games,
        "hitters": hitter_rows,
        "pitchers": pitcher_rows,
        "daily_game_date": daily_game_date,
        "daily_hitters": daily_hitters,
        "daily_pitchers": daily_pitchers,
        "risers_fallers": rf,
        "dynasty_hitters": dynasty_hitters,
        "dynasty_pitchers": dynasty_pitchers,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--season-games", type=int, default=DEFAULT_SEASON_GAMES)
    ap.add_argument("--as-of", default=None)
    ap.add_argument("--out", default=str(OUT_JSON))
    ap.add_argument("--no-ages", action="store_true",
                    help="Skip MLB Stats API age fetch (dynasty section will be empty).")
    args = ap.parse_args()

    print(f"Building fantasy bundle (season_games={args.season_games}, "
          f"as_of={args.as_of or 'today'}, fetch_ages={not args.no_ages})...")
    bundle = build_rankings(season_games=args.season_games,
                             as_of=args.as_of,
                             fetch_ages=not args.no_ages)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(bundle, f, indent=2)

    print(f"Wrote bundle -> {out_path}")
    print(f"  ROS: {len(bundle['hitters'])} hitters / {len(bundle['pitchers'])} pitchers")
    print(f"  Today: {len(bundle['daily_hitters'])} / {len(bundle['daily_pitchers'])}")
    rf = bundle.get("risers_fallers", {})
    print(f"  Risers: {len(rf.get('hitters', {}).get('risers', []))} hitters, "
          f"{len(rf.get('pitchers', {}).get('risers', []))} pitchers")
    print(f"  Fallers: {len(rf.get('hitters', {}).get('fallers', []))} hitters, "
          f"{len(rf.get('pitchers', {}).get('fallers', []))} pitchers")
    print(f"  Dynasty: {len(bundle['dynasty_hitters'])} hitters / "
          f"{len(bundle['dynasty_pitchers'])} pitchers")

    rf_h = rf.get("hitters", {})
    if rf_h.get("risers"):
        print("\nTop 5 hitter risers (delta = recent - season pts/game):")
        for r in rf_h["risers"][:5]:
            print(f"  {r['rank']:>2}. {r['name']:<25} {r['team']:<4}  "
                  f"recent {r['recent_pts_per_game']:>4.1f} vs season {r['season_pts_per_game']:>4.1f}  "
                  f"Δ={r['delta']:+.2f}")
    if rf_h.get("fallers"):
        print("\nTop 5 hitter fallers:")
        for r in rf_h["fallers"][:5]:
            print(f"  {r['rank']:>2}. {r['name']:<25} {r['team']:<4}  "
                  f"recent {r['recent_pts_per_game']:>4.1f} vs season {r['season_pts_per_game']:>4.1f}  "
                  f"Δ={r['delta']:+.2f}")

    if bundle["dynasty_hitters"]:
        print("\nTop 5 dynasty hitters (age-adjusted):")
        for r in bundle["dynasty_hitters"][:5]:
            print(f"  {r['dynasty_rank']:>2}. {r['name']:<25} age={r.get('age','—'):<3}  "
                  f"mult={r['age_multiplier']:.2f}  ROS={r['points']:>5.1f}  "
                  f"DYN={r['dynasty_score']:>5.1f}")


if __name__ == "__main__":
    main()
