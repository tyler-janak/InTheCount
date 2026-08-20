"""
player_rankings.py
===================
Season player power rankings for the Power Rankings tab: a rest-of-season
production estimate for every hitter and pitcher with graded games this
season, ranked highest to lowest. Each row also carries the player's age
(when available) so the front-end can filter down to a 25-and-under view.

Scoring (missing SB / W / SV, which aren't tracked in the accuracy log yet):
    Hitter:  H + 2B + 2*3B + 3*HR + R + RBI + BB - 0.5*K
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
# Only fetch/attach ages for the top N by power score — deep-bench players
# are noise for a "power rankings" spotlight feature and this keeps MLB
# Stats API calls bounded.
TOP_N_HITTERS_FOR_AGE = 400
TOP_N_PITCHERS_FOR_AGE = 300

HITTER_WEIGHTS = {"H": 1.0, "2B": 1.0, "3B": 2.0, "HR": 3.0,
                  "R": 1.0, "RBI": 1.0, "BB": 1.0, "K": -0.5}
PITCHER_WEIGHTS = {"IP": 3.0, "K": 1.0, "ER": -1.0, "H": -0.5, "BB": -0.5}


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
# Power score
# ---------------------------------------------------------------------------
def _hitter_points(ros: dict) -> float:
    return sum(ros.get(k, 0.0) * w for k, w in HITTER_WEIGHTS.items())


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
    top `top_n` entries by power score (rows are already sorted)."""
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
                   fetch_ages: bool = True) -> dict:
    if not ACC_CSV.exists():
        raise FileNotFoundError(f"Accuracy log not found at {ACC_CSV}")

    df = pd.read_csv(ACC_CSV, low_memory=False)
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    if as_of:
        df = df[df["game_date"] <= pd.to_datetime(as_of)]

    team_games = _team_games_played(df)

    # ------- HITTERS -------
    hitters_df = df[df["player_type"].astype(str).str.lower() == "hitter"].copy()
    hitter_rows = []
    for (mid, name), grp in hitters_df.groupby(["mlb_id", "player_name"]):
        team = str(grp["team"].dropna().iloc[-1]) if grp["team"].notna().any() else ""
        per_game = _hitter_per_game(grp)
        gp = per_game.pop("games_played")
        team_gp = team_games.get(team, max(gp, 1))
        gr = _remaining_games(gp, team_gp, season_games)
        ros = {k: per_game[k] * gr for k in per_game}
        power_score = round(_hitter_points(ros), 1)
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
            "power_score": power_score,
        })
    hitter_rows.sort(key=lambda r: r["power_score"], reverse=True)
    for i, r in enumerate(hitter_rows, start=1):
        r["rank"] = i

    # ------- PITCHERS -------
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
        "age_cutoff": AGE_CUTOFF,
        "scoring": {
            "type": "power_score_v1",
            "hitter": HITTER_WEIGHTS,
            "pitcher": PITCHER_WEIGHTS,
            "missing_categories": ["SB", "W", "SV"],
            "note": "Rest-of-season production estimate. SB / W / SV not "
                    "tracked in the accuracy log yet. Age is fetched from "
                    "the MLB Stats API for the top "
                    f"{TOP_N_HITTERS_FOR_AGE} hitters / {TOP_N_PITCHERS_FOR_AGE} "
                    "pitchers by power score; players without an age are "
                    "excluded from the 25-and-under view.",
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
    args = ap.parse_args()

    print(f"Building player power rankings (season_games={args.season_games}, "
          f"as_of={args.as_of or 'today'}, fetch_ages={not args.no_ages})...")
    bundle = build_rankings(season_games=args.season_games,
                             as_of=args.as_of,
                             fetch_ages=not args.no_ages)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(bundle, f, indent=2)

    print(f"Wrote bundle -> {out_path}")
    print(f"  {len(bundle['hitters'])} hitters / {len(bundle['pitchers'])} pitchers")
    n_young_h = sum(1 for r in bundle["hitters"] if r.get("age") and r["age"] <= AGE_CUTOFF)
    n_young_p = sum(1 for r in bundle["pitchers"] if r.get("age") and r["age"] <= AGE_CUTOFF)
    print(f"  {n_young_h} hitters / {n_young_p} pitchers age {AGE_CUTOFF} or under")

    print(f"\nTop 5 hitters:")
    for r in bundle["hitters"][:5]:
        print(f"  {r['rank']:>2}. {r['name']:<25} {r['team']:<4}  "
              f"age={r.get('age','—')}  power={r['power_score']:>5.1f}")
    print(f"\nTop 5 pitchers:")
    for r in bundle["pitchers"][:5]:
        print(f"  {r['rank']:>2}. {r['name']:<25} {r['team']:<4}  "
              f"age={r.get('age','—')}  power={r['power_score']:>5.1f}")


if __name__ == "__main__":
    main()
