"""
nrfi_today.py
=============
Generate today's per-game NRFI probability predictions.

Steps
-----
1. Fetch today's MLB schedule (starters via MLB Stats API)
2. Recompute each starter's / team's rolling features FRESH through their
   most recent completed game (the stored pre-game snapshots in
   nrfi_game_data.csv exclude their own game, so reusing the last row
   directly is one start stale). Falls back to the stored snapshot for
   anything that can't be recomputed.
3. Load nrfi_model.pkl and predict NRFI probability for each game
4. Optionally pull FanDuel NRFI odds from the props long CSV
5. Save outputs/nrfi_today.csv (+ outputs/nrfi_status.json for the site)

Usage
-----
    python nrfi_today.py
    python nrfi_today.py --date 2026-05-03
"""

import argparse
import json
import pickle
import re
import unicodedata
import warnings
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")

MODEL_DIR = Path("models")
DATA_DIR  = Path("data")
OUT_DIR   = Path("outputs")

ROLLING_WINDOWS      = [5, 10]
ROLLING_WINDOWS_FULL = [5, 10]
ROLLING_STD_WINDOW   = 10   # window used for the *_std features

# Rolling columns to pull from pitcher_game_data.csv for full-game features
PITCHER_GAME_STAT_COLS = [
    "K_rate_last5",  "K_rate_last10",  "K_rate_std",
    "BB_rate_last5", "BB_rate_last10", "BB_rate_std",
    "H_rate_last5",  "H_rate_last10",  "H_rate_std",
    "HR_rate_last5", "HR_rate_last10", "HR_rate_std",
    "IP_last5",      "IP_last10",      "IP_std",
    "days_rest",
]

# Raw per-game base columns in pitcher_game_data.csv (if present, we
# recompute the rolling stats fresh through the most recent start instead
# of reusing the stored — one-game-stale — snapshot).
PITCHER_GAME_BASE_COLS = ["K_rate", "BB_rate", "H_rate", "HR_rate", "IP"]

# Substrings that mark identifier / label columns we must never treat as
# numeric features when recomputing rolling stats.
NON_FEATURE_TOKENS = ("_id", "_name", "gamePk", "game_pk", "team")

TEAM_MAP = {
    "Arizona Diamondbacks": "AZ",  "Atlanta Braves": "ATL",
    "Baltimore Orioles": "BAL",    "Boston Red Sox": "BOS",
    "Chicago Cubs": "CHC",         "Chicago White Sox": "CWS",
    "Cincinnati Reds": "CIN",      "Cleveland Guardians": "CLE",
    "Colorado Rockies": "COL",     "Detroit Tigers": "DET",
    "Houston Astros": "HOU",       "Kansas City Royals": "KC",
    "Los Angeles Angels": "LAA",   "Los Angeles Dodgers": "LAD",
    "Miami Marlins": "MIA",        "Milwaukee Brewers": "MIL",
    "Minnesota Twins": "MIN",      "New York Mets": "NYM",
    "New York Yankees": "NYY",     "Athletics": "ATH",
    "Philadelphia Phillies": "PHI","Pittsburgh Pirates": "PIT",
    "San Diego Padres": "SD",      "San Francisco Giants": "SF",
    "Seattle Mariners": "SEA",     "St. Louis Cardinals": "STL",
    "Tampa Bay Rays": "TB",        "Texas Rangers": "TEX",
    "Toronto Blue Jays": "TOR",    "Washington Nationals": "WSH",
    "Oakland Athletics": "ATH",
}


def team_to_abbr(val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return val
    s = str(val).strip()
    return TEAM_MAP.get(s, s)


def normalize_name(name) -> str:
    if not name or (isinstance(name, float) and np.isnan(name)):
        return ""
    s = str(name).strip().lower()
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", " ", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


# ── schedule ──────────────────────────────────────────────────────────────────

def fetch_schedule(target_date: str) -> list[dict]:
    url = (f"https://statsapi.mlb.com/api/v1/schedule"
           f"?sportId=1&date={target_date}&hydrate=probablePitcher,team")
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    data = r.json()

    games = []
    for d in data.get("dates", []):
        for g in d.get("games", []):
            teams = g.get("teams", {})
            away  = teams.get("away", {})
            home  = teams.get("home", {})
            away_t = away.get("team", {}) or {}
            home_t = home.get("team", {}) or {}
            away_p = away.get("probablePitcher", {}) or {}
            home_p = home.get("probablePitcher", {}) or {}

            games.append({
                "gamePk":           g.get("gamePk"),
                "game_date":        target_date,
                "away_team":        team_to_abbr(away_t.get("abbreviation")),
                "home_team":        team_to_abbr(home_t.get("abbreviation")),
                "away_full":        away_t.get("name"),
                "home_full":        home_t.get("name"),
                "away_sp_name":     away_p.get("fullName"),
                "away_sp_id":       away_p.get("id"),
                "home_sp_name":     home_p.get("fullName"),
                "home_sp_id":       home_p.get("id"),
            })
    return games


# ── fresh rolling recomputation ───────────────────────────────────────────────
#
# WHY: nrfi_data.py builds *pre-game* (leakage-free) rolling features, so the
# row stored for a pitcher's most recent start contains stats through his
# SECOND-to-last start. Reusing that row for today's prediction is therefore
# one game stale. These helpers rebuild the rolling windows fresh from the raw
# per-game values — including the most recent completed game — and the stored
# snapshot is only used as a fallback for anything we can't recompute.

def _rolling_from_games(games: pd.DataFrame, date_col: str) -> dict:
    """Compute last5/last10 means and a 10-game std for every raw numeric
    column in `games` (rows must already be sorted by date ascending).
    Columns that are themselves rolling stats or identifiers are skipped."""
    out = {}
    for col in games.columns:
        if col == date_col:
            continue
        if col.endswith(("_last5", "_last10", "_std")):
            continue
        if any(tok in col for tok in NON_FEATURE_TOKENS):
            continue
        vals = pd.to_numeric(games[col], errors="coerce").dropna()
        if vals.empty:
            continue
        for w in ROLLING_WINDOWS:
            out[f"{col}_last{w}"] = float(vals.tail(w).mean())
        std = vals.tail(ROLLING_STD_WINDOW).std()
        if pd.notna(std):
            out[f"{col}_std"] = float(std)
    return out


def _collect_sp_games(nrfi_df: pd.DataFrame,
                      target_date: pd.Timestamp,
                      sp_id=None, sp_name: str = "") -> tuple[pd.DataFrame, str]:
    """Gather every historical row for one starter from BOTH the away and
    home sides of nrfi_game_data.csv, with side prefixes stripped to a
    common 'sp_' prefix. Returns (games sorted by date, date_col)."""
    if nrfi_df.empty:
        return pd.DataFrame(), ""
    date_col = next((c for c in ["game_date", "date"] if c in nrfi_df.columns), None)
    if date_col is None:
        return pd.DataFrame(), ""

    tmp = nrfi_df[pd.to_datetime(nrfi_df[date_col], errors="coerce") < target_date]
    if tmp.empty:
        return pd.DataFrame(), date_col

    norm = normalize_name(sp_name) if sp_name else ""
    frames = []
    for side in ["away", "home"]:
        rows = pd.DataFrame()
        id_col = f"{side}_sp_id"
        if id_col in tmp.columns and sp_id is not None:
            rows = tmp[pd.to_numeric(tmp[id_col], errors="coerce") == float(sp_id)]
        if rows.empty and norm:
            for name_col in [f"{side}_sp_name", f"{side}_sp_pitcher_name"]:
                if name_col in tmp.columns:
                    mask = tmp[name_col].astype(str).apply(normalize_name) == norm
                    if mask.any():
                        rows = tmp[mask]
                        break
        if rows.empty:
            continue
        side_cols = {c: c.replace(f"{side}_sp_", "sp_", 1)
                     for c in rows.columns if c.startswith(f"{side}_sp_")}
        sub = rows[[date_col] + list(side_cols)].rename(columns=side_cols)
        frames.append(sub)

    if not frames:
        return pd.DataFrame(), date_col

    games = pd.concat(frames, ignore_index=True)
    games[date_col] = pd.to_datetime(games[date_col], errors="coerce")
    return games.sort_values(date_col), date_col


def recompute_sp_rolling(nrfi_df: pd.DataFrame,
                         target_date: pd.Timestamp,
                         sp_id=None, sp_name: str = "") -> dict:
    """Fresh 1st-inning rolling stats through the pitcher's MOST RECENT
    completed start. Keys come back prefixed 'sp_' (no side), e.g.
    'sp_K_rate_last5'. Empty dict if the raw base columns aren't in the
    table (in which case the caller falls back to the stored snapshot)."""
    games, date_col = _collect_sp_games(nrfi_df, target_date, sp_id, sp_name)
    if games.empty:
        return {}
    return _rolling_from_games(games, date_col)


def recompute_team_bat_rolling(nrfi_df: pd.DataFrame,
                               target_date: pd.Timestamp,
                               team: str) -> dict:
    """Fresh rolling team-batting stats through the team's most recent
    completed game, pooled across away/home rows. Keys prefixed 'bat_'."""
    if nrfi_df.empty or not team:
        return {}
    date_col = next((c for c in ["game_date", "date"] if c in nrfi_df.columns), None)
    if date_col is None:
        return {}

    tmp = nrfi_df[pd.to_datetime(nrfi_df[date_col], errors="coerce") < target_date]
    if tmp.empty:
        return {}

    frames = []
    for side in ["away", "home"]:
        team_col = f"{side}_team"
        if team_col not in tmp.columns:
            continue
        rows = tmp[tmp[team_col] == team]
        if rows.empty:
            continue
        side_cols = {c: c.replace(f"{side}_bat_", "bat_", 1)
                     for c in rows.columns if c.startswith(f"{side}_bat_")}
        if not side_cols:
            continue
        sub = rows[[date_col] + list(side_cols)].rename(columns=side_cols)
        frames.append(sub)

    if not frames:
        return {}

    games = pd.concat(frames, ignore_index=True)
    games[date_col] = pd.to_datetime(games[date_col], errors="coerce")
    games = games.sort_values(date_col)
    return _rolling_from_games(games, date_col)


def recompute_pitcher_fg_rolling(pitcher_game_df: pd.DataFrame,
                                 target_date: pd.Timestamp,
                                 sp_id=None, sp_name: str = "",
                                 side_prefix: str = "away_sp_fg_") -> dict:
    """Fresh full-game rolling stats (K/BB/H/HR rates, IP) through the
    pitcher's most recent start, plus days_rest computed against
    target_date. Requires the raw base columns to exist in
    pitcher_game_data.csv; otherwise returns {} and the caller falls back
    to the stored snapshot."""
    if pitcher_game_df.empty:
        return {}
    if not all(c in pitcher_game_df.columns for c in PITCHER_GAME_BASE_COLS):
        return {}

    tmp = pitcher_game_df[pitcher_game_df["game_date"] < target_date]
    if tmp.empty:
        return {}

    rows = pd.DataFrame()
    if sp_id is not None and "pitcher" in tmp.columns:
        rows = tmp[pd.to_numeric(tmp["pitcher"], errors="coerce") == float(sp_id)]
    if rows.empty and sp_name:
        norm = normalize_name(sp_name)
        for name_col in ["player_name", "pitcher_name"]:
            if name_col in tmp.columns:
                mask = tmp[name_col].fillna("").astype(str).apply(normalize_name) == norm
                if mask.any():
                    rows = tmp[mask]
                    break
    if rows.empty:
        return {}

    rows = rows.sort_values("game_date")
    out = {}
    for col in PITCHER_GAME_BASE_COLS:
        vals = pd.to_numeric(rows[col], errors="coerce").dropna()
        if vals.empty:
            continue
        for w in ROLLING_WINDOWS_FULL:
            out[f"{side_prefix}{col}_last{w}"] = float(vals.tail(w).mean())
        std = vals.tail(ROLLING_STD_WINDOW).std()
        if pd.notna(std):
            out[f"{side_prefix}{col}_std"] = float(std)

    last_game = rows["game_date"].max()
    if pd.notna(last_game):
        out[f"{side_prefix}days_rest"] = float((target_date - last_game).days)

    return out


# ── stored-snapshot fallbacks (one start stale, used only to fill gaps) ───────

def load_pitcher_game_df() -> pd.DataFrame:
    """Load pitcher_game_data.csv for full-game stats (primary NRFI signal).
    Pulls both the raw base columns (for fresh recomputation) and the stored
    rolling columns (fallback)."""
    path = DATA_DIR / "pitcher_game_data.csv"
    if not path.exists():
        return pd.DataFrame()
    cols_needed = (["pitcher", "game_date", "player_name", "pitcher_name", "team"]
                   + PITCHER_GAME_STAT_COLS + PITCHER_GAME_BASE_COLS)
    try:
        header = pd.read_csv(path, nrows=0)
        usecols = [c for c in header.columns if c in cols_needed]
        df = pd.read_csv(path, usecols=usecols, low_memory=False)
        df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
        return df
    except Exception as e:
        print(f"  [warn] Could not load pitcher_game_data.csv: {e}")
        return pd.DataFrame()


def get_pitcher_fg_features(pitcher_game_df: pd.DataFrame,
                             target_date: pd.Timestamp,
                             sp_id=None, sp_name: str = "",
                             side_prefix: str = "away_sp_fg_") -> dict:
    """FALLBACK: stored full-game rolling stats from the pitcher's most
    recent row (pre-game snapshot → one start stale). Only used for
    features recompute_pitcher_fg_rolling couldn't produce."""
    if pitcher_game_df.empty:
        return {}

    tmp = pitcher_game_df[pitcher_game_df["game_date"] < target_date].copy()
    if tmp.empty:
        return {}

    row = pd.Series(dtype=object)

    if sp_id is not None and "pitcher" in tmp.columns:
        m = tmp[pd.to_numeric(tmp["pitcher"], errors="coerce") == float(sp_id)]
        if not m.empty:
            row = m.sort_values("game_date").iloc[-1]

    if row.empty and sp_name:
        norm = normalize_name(sp_name)
        for name_col in ["player_name", "pitcher_name"]:
            if name_col in tmp.columns:
                mask = tmp[name_col].fillna("").astype(str).apply(normalize_name) == norm
                if mask.any():
                    row = tmp[mask].sort_values("game_date").iloc[-1]
                    break

    if row.empty:
        return {}

    return {
        f"{side_prefix}{col}": row[col]
        for col in PITCHER_GAME_STAT_COLS
        if col in row.index and pd.notna(row[col])
    }


def get_sp_features(nrfi_df: pd.DataFrame,
                    target_date: pd.Timestamp,
                    sp_id=None, sp_name: str = "") -> pd.Series:
    """FALLBACK: most recent stored pre-game 1st-inning feature row for a
    starter (one start stale). Tries pitcher ID first, then name."""
    if nrfi_df.empty:
        return pd.Series(dtype=object)

    date_col = next((c for c in ["game_date", "date"] if c in nrfi_df.columns), None)
    if date_col is None:
        return pd.Series(dtype=object)

    tmp = nrfi_df.copy()
    tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
    tmp = tmp[tmp[date_col] < target_date]
    if tmp.empty:
        return pd.Series(dtype=object)

    for id_col in ["away_sp_id", "home_sp_id"]:
        if id_col in tmp.columns and sp_id is not None:
            m = tmp[pd.to_numeric(tmp[id_col], errors="coerce") == float(sp_id)]
            if not m.empty:
                return m.sort_values(date_col).iloc[-1]

    if sp_name:
        norm = normalize_name(sp_name)
        for name_col in ["away_sp_name", "home_sp_name",
                         "away_sp_pitcher_name", "home_sp_pitcher_name"]:
            if name_col in tmp.columns:
                mask = tmp[name_col].astype(str).apply(normalize_name) == norm
                if mask.any():
                    return tmp[mask].sort_values(date_col).iloc[-1]

    return pd.Series(dtype=object)


def get_team_bat_features(nrfi_df: pd.DataFrame,
                          target_date: pd.Timestamp,
                          team: str,
                          side: str) -> pd.Series:
    """FALLBACK: most recent stored pre-game 1st-inning batting features
    for a team (one game stale). side = 'away' or 'home'."""
    if nrfi_df.empty:
        return pd.Series(dtype=object)

    date_col = next((c for c in ["game_date", "date"] if c in nrfi_df.columns), None)
    if date_col is None:
        return pd.Series(dtype=object)

    team_col = f"{side}_team"
    if team_col not in nrfi_df.columns:
        return pd.Series(dtype=object)

    tmp = nrfi_df.copy()
    tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
    tmp = tmp[(tmp[date_col] < target_date) & (tmp[team_col] == team)]
    if tmp.empty:
        return pd.Series(dtype=object)

    return tmp.sort_values(date_col).iloc[-1]


# ── feature assembly ──────────────────────────────────────────────────────────

def build_game_feature_row(
    game: dict,
    nrfi_df: pd.DataFrame,
    pitcher_game_df: pd.DataFrame,
    target_date: pd.Timestamp,
    model_features: list[str],
) -> pd.Series:
    """
    Assemble one feature row for a single game.

    Priority for every feature:
      1. FRESH recomputation through the most recent completed game
         (recompute_* helpers) — this is the fix for the one-start-stale
         snapshots.
      2. Stored pre-game snapshot from the CSVs (fallback only, fills
         whatever the fresh pass couldn't produce).
    """
    row = pd.Series(dtype=float)

    # ── Full-game pitcher stats (primary signal) ──────────────────────────────
    # Fresh first…
    for side, prefix in [("away", "away_sp_fg_"), ("home", "home_sp_fg_")]:
        fresh = recompute_pitcher_fg_rolling(
            pitcher_game_df, target_date,
            sp_id=game.get(f"{side}_sp_id"),
            sp_name=game.get(f"{side}_sp_name", ""),
            side_prefix=prefix,
        )
        for k, v in fresh.items():
            row[k] = v

    # …stored snapshot fills only what's missing.
    for side, prefix in [("away", "away_sp_fg_"), ("home", "home_sp_fg_")]:
        stored = get_pitcher_fg_features(
            pitcher_game_df, target_date,
            sp_id=game.get(f"{side}_sp_id"),
            sp_name=game.get(f"{side}_sp_name", ""),
            side_prefix=prefix,
        )
        for k, v in stored.items():
            if k not in row.index:
                row[k] = v

    # ── 1st-inning pitcher stats ──────────────────────────────────────────────
    # Fresh recompute (keys come back 'sp_*'; add the side prefix here).
    for side in ["away", "home"]:
        fresh = recompute_sp_rolling(
            nrfi_df, target_date,
            sp_id=game.get(f"{side}_sp_id"),
            sp_name=game.get(f"{side}_sp_name", ""),
        )
        for k, v in fresh.items():
            row[f"{side}_{k}"] = v

    # Stored snapshot fallback — only fills columns still missing.
    away_feat = get_sp_features(nrfi_df, target_date,
                                sp_id=game.get("away_sp_id"),
                                sp_name=game.get("away_sp_name", ""))
    for col in [c for c in (away_feat.index if not away_feat.empty else [])
                if "away_sp_" in c and "fg_" not in c and (
                    c.endswith("_std") or any(c.endswith(f"_last{w}") for w in ROLLING_WINDOWS)
                )]:
        if col not in row.index:
            row[col] = away_feat[col]

    home_feat = get_sp_features(nrfi_df, target_date,
                                sp_id=game.get("home_sp_id"),
                                sp_name=game.get("home_sp_name", ""))
    for col in [c for c in (home_feat.index if not home_feat.empty else [])
                if "home_sp_" in c and "fg_" not in c and (
                    c.endswith("_std") or any(c.endswith(f"_last{w}") for w in ROLLING_WINDOWS)
                )]:
        if col not in row.index:
            row[col] = home_feat[col]

    # ── Team batting features ─────────────────────────────────────────────────
    # Fresh recompute (keys come back 'bat_*'; add the side prefix here).
    for side in ["away", "home"]:
        fresh = recompute_team_bat_rolling(nrfi_df, target_date, game[f"{side}_team"])
        for k, v in fresh.items():
            row[f"{side}_{k}"] = v

    # Stored snapshot fallback — only fills columns still missing.
    away_bat = get_team_bat_features(nrfi_df, target_date, game["away_team"], "away")
    for col in [c for c in (away_bat.index if not away_bat.empty else [])
                if "away_bat_" in c]:
        if col not in row.index:
            row[col] = away_bat[col]

    home_bat = get_team_bat_features(nrfi_df, target_date, game["home_team"], "home")
    for col in [c for c in (home_bat.index if not home_bat.empty else [])
                if "home_bat_" in c]:
        if col not in row.index:
            row[col] = home_bat[col]

    # Park factor (venue-level, doesn't go stale the same way)
    if "park_factor" not in row.index and not home_feat.empty \
            and "park_factor" in home_feat.index:
        row["park_factor"] = home_feat["park_factor"]

    # Ensure all model features exist
    for f in model_features:
        if f not in row.index:
            row[f] = np.nan

    return row


# ── load FanDuel NRFI odds ────────────────────────────────────────────────────

def load_fd_nrfi_odds() -> pd.DataFrame:
    """
    Look for NRFI lines in the FanDuel props long CSV.
    FanDuel sometimes lists NRFI as a team-level market.
    Returns empty DataFrame if not available.
    """
    path = OUT_DIR / "fanduel_props_today_long.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, low_memory=False)
    if "market" not in df.columns:
        return pd.DataFrame()
    nrfi_rows = df[df["market"].str.lower().str.contains("nrfi|first.inning", na=False)]
    return nrfi_rows


# ── status sidecar ────────────────────────────────────────────────────────────

OUTPUT_COLS = [
    "game_date", "gamePk", "away_team", "home_team", "team_a", "team_b",
    "away_full", "home_full", "away_sp", "home_sp",
    "nrfi_prob", "yrfi_prob", "pick", "lean", "threshold",
]


def _write_status(target_date: str, status: str, reason: str = "", games: int = 0):
    """Write outputs/nrfi_status.json so the site can tell an off-day /
    pipeline problem apart from a stale file."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "nrfi_status.json", "w") as f:
        json.dump({
            "date":   target_date,
            "status": status,          # "ok" | "empty"
            "reason": reason,
            "games":  games,
        }, f, indent=2)


def _write_dated_stub(target_date: str, reason: str) -> pd.DataFrame:
    """Overwrite outputs/nrfi_today.csv with an EMPTY (header-only) CSV so
    the site shows "no predictions today" instead of a phantom
    "No Team vs No Team" card. A status JSON carries the date + reason.
    Still called on every early bail so the tab never looks frozen on
    yesterday's file."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=OUTPUT_COLS).to_csv(OUT_DIR / "nrfi_today.csv", index=False)
    _write_status(target_date, "empty", reason=reason)
    print(f"  ⚠️  [nrfi_today] wrote empty stub for {target_date} — {reason}")
    return pd.DataFrame()


# ── main runner ───────────────────────────────────────────────────────────────

def run_nrfi(target_date: str | None = None) -> pd.DataFrame:
    target_date = target_date or str(date.today())
    target_ts   = pd.Timestamp(target_date)

    print(f"\nGenerating NRFI predictions for: {target_date}")

    # Load model
    model_path = MODEL_DIR / "nrfi_model.pkl"
    if not model_path.exists():
        return _write_dated_stub(
            target_date, "nrfi_model.pkl not found (run nrfi_train.py)"
        )

    with open(model_path, "rb") as f:
        model_obj = pickle.load(f)

    pipeline  = model_obj["pipeline"]
    features  = model_obj["features"]
    threshold = model_obj.get("threshold", 0.50)
    print(f"  Model loaded | {len(features)} features | threshold={threshold:.2f}")

    # Load historical NRFI data
    nrfi_path = DATA_DIR / "nrfi_game_data.csv"
    if not nrfi_path.exists():
        return _write_dated_stub(
            target_date, "data/nrfi_game_data.csv not found (run nrfi_data.py)"
        )
    nrfi_df = pd.read_csv(nrfi_path, low_memory=False)
    if "game_date" in nrfi_df.columns:
        nrfi_df["game_date"] = pd.to_datetime(nrfi_df["game_date"], errors="coerce")
        # Loud freshness check: if the feature table is frozen days behind,
        # fresh recomputation can't help — the rebuild upstream is broken.
        last_dt = nrfi_df["game_date"].max()
        if pd.notna(last_dt):
            staleness = (target_ts - last_dt).days
            print(f"  nrfi_game_data.csv last game_date: {last_dt.date()} "
                  f"({staleness} day(s) before target)")
            if staleness > 2:
                print("  ⚠️  feature table looks STALE — check the nrfi_data.py "
                      "rebuild step in daily_update.py (step 4a)")

    # Fetch today's schedule
    schedule = fetch_schedule(target_date)
    print(f"  Games found: {len(schedule)}")
    if not schedule:
        return _write_dated_stub(
            target_date, "MLB Stats API returned no games for this date"
        )

    fd_odds = load_fd_nrfi_odds()

    # Load full-game pitcher stats (primary NRFI signal)
    pitcher_game_df = load_pitcher_game_df()
    if pitcher_game_df.empty:
        print("  [warn] pitcher_game_data.csv not found — full-game features unavailable.")
        print("         Run: python hitterspitchers_data.py --input <pitch_csv>")
    else:
        print(f"  Pitcher game data loaded: {len(pitcher_game_df):,} rows")
        has_base = all(c in pitcher_game_df.columns for c in PITCHER_GAME_BASE_COLS)
        if has_base:
            print("  Fresh full-game rolling recompute: ENABLED (raw base cols found)")
        else:
            print("  [warn] raw base cols missing from pitcher_game_data.csv — "
                  "falling back to stored (one-start-stale) snapshots")

    rows = []
    for game in schedule:
        if not game.get("away_team") or not game.get("home_team"):
            continue

        feat_row = build_game_feature_row(game, nrfi_df, pitcher_game_df, target_ts, features)

        X = pd.DataFrame([feat_row[features].to_dict()])
        try:
            nrfi_prob = float(pipeline.predict_proba(X)[0, 1])
        except Exception as e:
            print(f"  [{game['away_team']} @ {game['home_team']}] prediction failed: {e}")
            nrfi_prob = np.nan

        yrfi_prob = 1.0 - nrfi_prob if pd.notna(nrfi_prob) else np.nan
        nrfi_pick = "NRFI" if (pd.notna(nrfi_prob) and nrfi_prob >= threshold) else "YRFI"

        out_row = {
            "game_date":     target_date,
            "gamePk":        game.get("gamePk"),
            "away_team":     game["away_team"],
            "home_team":     game["home_team"],
            "team_a":        game["away_team"],   # alias for nrfi_grade.py
            "team_b":        game["home_team"],   # alias for nrfi_grade.py
            "away_full":     game.get("away_full"),
            "home_full":     game.get("home_full"),
            "away_sp":       game.get("away_sp_name") or "TBD",
            "home_sp":       game.get("home_sp_name") or "TBD",
            "nrfi_prob":     round(nrfi_prob, 4) if pd.notna(nrfi_prob) else None,
            "yrfi_prob":     round(yrfi_prob, 4) if pd.notna(yrfi_prob) else None,
            "pick":          nrfi_pick,
            "lean":          "YES" if nrfi_pick == "NRFI" else "NO",  # for nrfi_grade.py
            "threshold":     threshold,
        }
        rows.append(out_row)

    if not rows:
        return _write_dated_stub(target_date, "no games matched (empty schedule after filtering)")

    results = pd.DataFrame(rows).sort_values("nrfi_prob", ascending=False, na_position="last")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "nrfi_today.csv"
    results.to_csv(out_path, index=False)
    _write_status(target_date, "ok", games=len(results))
    print(f"\nSaved: {out_path}  ({len(results)} games)")

    print("\n── NRFI Predictions ─────────────────────────────────────────────")
    print(results[["away_team", "home_team", "away_sp", "home_sp",
                   "nrfi_prob", "pick"]].to_string(index=False))

    return results


def main():
    parser = argparse.ArgumentParser(description="Generate today's NRFI predictions")
    parser.add_argument("--date", default=None, help="Target date YYYY-MM-DD (default=today)")
    args = parser.parse_args()
    run_nrfi(args.date)


if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
