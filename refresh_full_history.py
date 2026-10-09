"""
refresh_full_history.py
=======================
Multi-season Statcast refresh. Pulls each season in the modelling window
(by default the current season and the two before it - 2024, 2025, 2026)
with `pybaseball.statcast()` into its own cache file, then rebuilds the
per-game feature tables (build_features.py).

    pitch_data_2024.csv   (pulled once, then cached)
    pitch_data_2025.csv   (pulled once, then cached)
    pitch_data_2026.csv   (incremental, updated every cron tick)

Same approach the project already used for 2025 + 2026, with the season made
a parameter so 2024 (or 2027 next year) needs no new code, plus two fixes:

* Overlap: the current season re-pulls the last 3 cached days. Restarting at
  "latest cached date + 1" skipped night games whenever the cache was
  refreshed mid-day (it already held that day's afternoon games).
* Timeout: each weekly request has a hard timeout and a sequential retry,
  so a stalled Baseball Savant request can't hang the run.

Usage
-----
    python refresh_full_history.py                        # 2024 2025 2026 + features
    python refresh_full_history.py --seasons 2024         # just collect 2024
    python refresh_full_history.py --rebuild-history      # force re-pull of past seasons
    python refresh_full_history.py --skip-2026            # don't touch the current season
    python refresh_full_history.py --skip-features
    python refresh_full_history.py --validate             # validation report only
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

HERE = Path(__file__).resolve().parent
ET = ZoneInfo("America/New_York")
OVERLAP_DAYS = 3
CHUNK_DAYS = 7
CHUNK_TIMEOUT_S = 15 * 60
PITCH_KEY = ["game_pk", "at_bat_number", "pitch_number"]
VALIDATION_JSON = HERE / "outputs" / "model_evaluation" / "data_validation" / "statcast_validation.json"


# ---------------------------------------------------------------------------
# Cache helpers (also used by build_features / the game runner / tests)
# ---------------------------------------------------------------------------
def current_season() -> int:
    return datetime.now(ET).year


def season_path(season: int) -> Path:
    return HERE / f"pitch_data_{season}.csv"


def cached_seasons() -> list[int]:
    out = []
    for p in HERE.glob("pitch_data_*.csv"):
        tail = p.stem.split("_")[-1]
        if tail.isdigit():
            out.append(int(tail))
    return sorted(out)


def read_season(season: int, columns: list[str] | None = None) -> pd.DataFrame:
    p = season_path(season)
    if not p.exists():
        return pd.DataFrame()
    want = None if columns is None else {c.lower() for c in columns}
    df = pd.read_csv(p, usecols=(lambda c: c.strip().lower() in want) if want else None, low_memory=False)
    df.columns = df.columns.str.strip().str.lower()
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    return df


def load_seasons(seasons: list[int], columns: list[str] | None = None) -> pd.DataFrame:
    parts = [read_season(s, columns) for s in seasons]
    parts = [p for p in parts if not p.empty]
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True, sort=False)
    keys = [c for c in PITCH_KEY if c in out.columns]
    return out.drop_duplicates(subset=keys, keep="last") if len(keys) == 3 else out


def _latest_cached_date(season: int) -> date | None:
    p = season_path(season)
    if not p.exists():
        return None
    d = pd.to_datetime(pd.read_csv(p, usecols=["game_date"])["game_date"], errors="coerce").max()
    return None if pd.isna(d) else d.date()


# ---------------------------------------------------------------------------
# pybaseball pull
# ---------------------------------------------------------------------------
def _try_import_statcast():
    try:
        from pybaseball import statcast
        return statcast
    except ImportError as e:
        print(f"WARNING:  pybaseball not available: {e}")
        return None


def _with_timeout(fn, timeout, **kwargs):
    box = {}

    def run():
        try:
            box["v"] = fn(**kwargs)
        except Exception as e:          # noqa: BLE001
            box["e"] = e
    t = threading.Thread(target=run, daemon=True)   # daemon: can't keep the process alive
    t.start()
    t.join(timeout)
    if t.is_alive():
        raise TimeoutError(f"no response within {timeout}s")
    if "e" in box:
        raise box["e"]
    return box.get("v")


def _fetch_range(start: date, end: date, statcast_fn) -> pd.DataFrame:
    """Pull [start, end] week by week. Attempt 1 uses pybaseball's parallel
    requests; retries are sequential. Raises if a week can't be pulled."""
    frames, cur = [], start
    while cur <= end:
        stop = min(cur + timedelta(days=CHUNK_DAYS - 1), end)
        for attempt in (1, 2, 3):
            try:
                print(f"  -> Statcast {cur} -> {stop} (attempt {attempt})", flush=True)
                df = _with_timeout(statcast_fn, CHUNK_TIMEOUT_S, start_dt=str(cur), end_dt=str(stop),
                                   parallel=(attempt == 1))
                break
            except Exception as e:      # noqa: BLE001
                print(f"     WARNING: {type(e).__name__}: {e}", flush=True)
                time.sleep(5 * attempt)
        else:
            raise RuntimeError(f"Statcast pull failed for {cur} -> {stop}")
        if df is not None and not df.empty:
            frames.append(df)
            print(f"     {len(df):,} pitches", flush=True)
        cur = stop + timedelta(days=1)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def _save(df: pd.DataFrame, path: Path) -> None:
    keys = [c for c in PITCH_KEY if c in df.columns]
    if len(keys) == 3:
        before = len(df)
        df = df.drop_duplicates(subset=keys, keep="last")
        if before != len(df):
            print(f"   deduped {before - len(df):,} duplicate pitches")
    df.to_csv(path, index=False)
    print(f"   wrote {path.name} ({len(df):,} rows)")


def refresh_season(season: int, rebuild: bool = False, end: str | None = None) -> Path | None:
    """Make sure pitch_data_<season>.csv covers the season (through `end`)."""
    path = season_path(season)
    season_start, season_end = date(season, 3, 1), date(season, 11, 30)
    today = datetime.now(ET).date()
    end_d = min(datetime.strptime(end, "%Y-%m-%d").date() if end else season_end, today)
    statcast = _try_import_statcast()
    print(f"\n-- {season} ({path.name}, exists={path.exists()}) --")

    if path.exists() and not rebuild:
        latest = _latest_cached_date(season)
        first = pd.to_datetime(pd.read_csv(path, usecols=["game_date"])["game_date"]).min().date()
        complete = season_end < today and latest and latest >= date(season, 9, 25) and first <= date(season, 3, 31)
        if complete or statcast is None:
            print(f"   cached {first} -> {latest}; {'season complete' if complete else 'no pybaseball'} - no pull")
            return path
        frames = [pd.read_csv(path, low_memory=False)]
        if first > date(season, 3, 31):                       # cache started late: fill the front gap
            frames.append(_fetch_range(season_start, first - timedelta(days=1), statcast))
        pull_start = max(season_start, latest - timedelta(days=OVERLAP_DAYS))
        if pull_start <= end_d:
            frames.append(_fetch_range(pull_start, end_d, statcast))
        _save(pd.concat(frames, ignore_index=True, sort=False), path)
        return path

    if statcast is None:
        return None
    df = _fetch_range(season_start, end_d, statcast)
    if df.empty:
        print("   (no rows returned)")
        return path if path.exists() else None
    _save(df, path)
    return path


# ---------------------------------------------------------------------------
# Validation (per season, machine-readable)
# ---------------------------------------------------------------------------
REQUIRED = ["game_pk", "game_date", "game_type", "game_year", "pitcher", "batter", "events",
            "description", "stand", "p_throws", "home_team", "away_team", "inning_topbot",
            "at_bat_number", "pitch_number", "bat_score", "post_bat_score",
            "post_home_score", "post_away_score"]


def validate_season(season: int) -> dict:
    df = read_season(season)
    r = {"season": season, "critical_failures": [], "warnings": []}
    if df.empty:
        r["critical_failures"].append("no data")
        return r
    fail = r["critical_failures"].append
    reg = df[df["game_type"].astype(str) == "R"]
    r.update(records=len(df), min_game_date=str(df["game_date"].min().date()),
             max_game_date=str(df["game_date"].max().date()),
             unique_games=int(df["game_pk"].nunique()), unique_players=int(pd.concat([df["pitcher"], df["batter"]]).nunique()),
             records_by_game_type={str(k): int(v) for k, v in df["game_type"].value_counts().items()},
             regular_season={"records": int(len(reg)), "games": int(reg["game_pk"].nunique()),
                             "first_date": str(reg["game_date"].min().date()) if len(reg) else None,
                             "last_date": str(reg["game_date"].max().date()) if len(reg) else None,
                             "pitchers": int(reg["pitcher"].nunique()), "batters": int(reg["batter"].nunique())})
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        fail(f"missing columns {missing}")
    r["missing_rate"] = {c: round(float(df[c].isna().mean()), 4) for c in REQUIRED if c in df.columns}
    for c in ("game_pk", "pitcher", "batter", "at_bat_number", "pitch_number", "home_team", "away_team"):
        if c in df.columns and df[c].isna().any():
            fail(f"{c} has missing values")
    r["duplicate_pitch_keys"] = int(df.duplicated(PITCH_KEY).sum())
    if r["duplicate_pitch_keys"]:
        fail(f"{r['duplicate_pitch_keys']} duplicate pitches")
    g = df.groupby("game_pk").agg(d=("game_date", "nunique"), h=("home_team", "nunique"), a=("away_team", "nunique"))
    if ((g > 1).any(axis=1)).any():
        fail("game_pk with more than one date/home/away")
    multi_bat = int((df.groupby(["game_pk", "at_bat_number"])["batter"].nunique() > 1).sum())
    if multi_bat > 50:
        fail(f"{multi_bat} plate appearances with more than one batter")
    if "game_year" in df.columns and set(df["game_year"].dropna().astype(int)) - {season}:
        fail("game_year outside season")
    teams = set(pd.concat([reg["home_team"], reg["away_team"]]).dropna())
    if season < current_season() and len(teams) != 30:
        fail(f"{len(teams)} regular-season teams")
    if season < current_season() and not 2400 <= r["regular_season"]["games"] <= 2435:
        fail(f"{r['regular_season']['games']} regular-season games (expected ~2430)")
    if r["regular_season"]["first_date"] and r["regular_season"]["first_date"] > f"{season}-04-05":
        fail(f"regular season starts {r['regular_season']['first_date']} (early weeks missing)")
    days = pd.Series(sorted(reg["game_date"].unique()))
    gaps = days[days.diff().dt.days > 4]
    if len(gaps) > 2:
        r["warnings"].append(f"gaps >4 days before {[str(d.date()) for d in gaps]}")
    r["passed"] = not r["critical_failures"]
    return r


def validate(seasons: list[int]) -> dict:
    rep = {"generated": datetime.now(ET).isoformat(timespec="seconds"),
           "source": "pybaseball.statcast", "seasons": {str(s): validate_season(s) for s in seasons}}
    rep["all_passed"] = all(v.get("passed") for v in rep["seasons"].values())
    VALIDATION_JSON.parent.mkdir(parents=True, exist_ok=True)
    VALIDATION_JSON.write_text(json.dumps(rep, indent=2, default=str))
    for s, v in rep["seasons"].items():
        print(f"  {s}: {'PASS' if v.get('passed') else 'FAIL'} {v.get('records', 0):,} pitches "
              f"{v.get('min_game_date')} -> {v.get('max_game_date')} {v['critical_failures'] or ''}")
    return rep


# ---------------------------------------------------------------------------
def refresh(seasons=None, rebuild_history=False, skip_current=False, skip_features=False,
            end_current: str | None = None) -> bool:
    cur = current_season()
    seasons = seasons or [cur - 2, cur - 1, cur]
    print("\n========== Multi-season Statcast refresh ==========")
    for s in seasons:
        if s == cur and skip_current:
            print(f"\n-- {s}: skipped")
            continue
        try:
            refresh_season(s, rebuild=(rebuild_history and s != cur), end=end_current if s == cur else None)
        except Exception as e:          # one season failing never blocks the others
            print(f"WARNING:  {s} refresh failed: {e}")
    have = [s for s in seasons if season_path(s).exists()]
    if have:
        validate(have)
    if skip_features or not have:
        return bool(have)
    import build_features as bf
    bf.build(seasons=[s for s in seasons if s in cached_seasons()])
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seasons", nargs="+", type=int, default=None)
    ap.add_argument("--rebuild-history", "--rebuild-2025", dest="rebuild_history", action="store_true")
    ap.add_argument("--skip-2026", "--skip-current", dest="skip_current", action="store_true")
    ap.add_argument("--end-2026", default=None)
    ap.add_argument("--skip-features", action="store_true")
    ap.add_argument("--validate", action="store_true")
    a = ap.parse_args()
    if a.validate:
        rep = validate(a.seasons or cached_seasons())
        sys.exit(0 if rep["all_passed"] else 2)
    ok = refresh(a.seasons, a.rebuild_history, a.skip_current, a.skip_features, a.end_2026)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
