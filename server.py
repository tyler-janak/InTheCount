"""
TheBullpenBet — FastAPI Backend
================================
Serves the static index.html and the data endpoints that read from the
daily-refreshed CSVs produced by the projection pipeline. This is a free,
public research site — no accounts, no paywall, no odds.

ENDPOINTS:
  GET  /                       → index.html
  GET  /api/games               today's games + projected winners
  GET  /api/pitchers            today's pitcher projections
  GET  /api/hitters              today's hitter projections
  GET  /api/accuracy            game-projection season accuracy + last 30
  GET  /api/player-accuracy     hitter / pitcher projection accuracy summary
  GET  /api/calibration         current bias-correction block applied to projections
  GET  /api/rankings            season player power rankings (all + 25-and-under)
  GET  /api/player/{mlb_id}     recent games + last-5/10/season splits by hand
  GET  /health                  liveness probe

LOCAL RUN:
  pip install -r requirements-server.txt
  uvicorn server:app --reload --port 8000
  → open http://localhost:8000

DEPLOYS:
  On Render the start command is:
    uvicorn server:app --host 0.0.0.0 --port $PORT
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

# ───── PATHS ─────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
DATA_DIR = BASE_DIR / "data"
PREDS_PATH = OUTPUT_DIR / "today_predictions.csv"
PROJ_PATH = OUTPUT_DIR / "hitterspitchers_today.csv"
PICKS_PATH = BASE_DIR / "2026_picks_accuracy.csv"
PLAYER_ACC_PATH = BASE_DIR / "2026_player_accuracy.csv"
HITTER_GAMES_PATH = DATA_DIR / "hitter_game_data.csv"
PITCHER_GAMES_PATH = DATA_DIR / "pitcher_game_data.csv"
RANKINGS_PATH = OUTPUT_DIR / "player_rankings.json"
INDEX_PATH = BASE_DIR / "index.html"
FAVICON_PATH = BASE_DIR / "favicon.svg"

# ───── FASTAPI APP ──────────────────────────────────────────
app = FastAPI(title="TheBullpenBet API", version="2.0")

# CORS open — public read-only API, no accounts, nothing to protect.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ───── 5-MINUTE IN-MEMORY CACHE ─────────────────────────────
_cache: dict[str, tuple[float, Any]] = {}
CACHE_TTL = 300  # seconds


def _cached(key: str, builder: Callable[[], Any]) -> Any:
    now = datetime.now().timestamp()
    hit = _cache.get(key)
    if hit and (now - hit[0] < CACHE_TTL):
        return hit[1]
    value = builder()
    _cache[key] = (now, value)
    return value


# ───── HELPERS ──────────────────────────────────────────────
def _safe(v, default=None):
    """Return default for NaN/None/inf, else the value."""
    if v is None:
        return default
    try:
        if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            return default
        if pd.isna(v):
            return default
    except (TypeError, ValueError):
        pass
    return v


def _f(v, default=0.0) -> float:
    try:
        f = float(_safe(v, default))
        if np.isnan(f) or np.isinf(f):
            return default
        return f
    except (TypeError, ValueError):
        return default


def _format_time_et(iso_str) -> str:
    if iso_str is None or pd.isna(iso_str) or iso_str == "":
        return ""
    try:
        dt = pd.to_datetime(iso_str, utc=True)
        et = dt.tz_convert(ZoneInfo("America/New_York"))
        # %-I removes leading zero on Linux; %#I on Windows. Strip leading 0 manually for portability.
        s = et.strftime("%I:%M %p ET")
        if s.startswith("0"):
            s = s[1:]
        return s
    except Exception:
        return ""


def _conf_label(c) -> str:
    s = str(c or "").strip().lower()
    if s.startswith("h"):
        return "HIGH"
    if s.startswith("m"):
        return "MED"
    return "LOW"


# ───── PAYLOAD BUILDERS ─────────────────────────────────────
def games_payload() -> list[dict]:
    if not PREDS_PATH.exists():
        return []
    df = pd.read_csv(PREDS_PATH, low_memory=False)
    rows: list[dict] = []
    for _, r in df.iterrows():
        home = _safe(r.get("home_team"), "")
        away = _safe(r.get("away_team"), "")
        if not home or not away:
            continue
        home_prob = _f(r.get("home_win_prob"), 0.5)
        away_prob = _f(r.get("away_win_prob"), 1 - home_prob)
        pick = _safe(r.get("predicted_winner"), home) or home
        rows.append({
            "away": away,
            "home": home,
            "time": _format_time_et(_safe(r.get("commence_time"))),
            "awayStarter": _safe(r.get("away_starter"), "TBD") or "TBD",
            "homeStarter": _safe(r.get("home_starter"), "TBD") or "TBD",
            "homeProb": home_prob,
            "awayProb": away_prob,
            "pick": pick,
            "pickProb": home_prob if pick == home else away_prob,
        })
    return rows


# Standard MLB scouting grades. 25 and 75 are intentionally excluded — the
# convention is to go straight from 20→30 and 70→80, with half-grades only
# filling the middle of the scale.
SCOUTING_GRADES = [20, 30, 35, 40, 45, 50, 55, 60, 65, 70, 80]


def _grade(v, default=50) -> int:
    """Coerce a confidence value to the nearest valid MLB scouting grade."""
    try:
        n = int(round(float(_safe(v, default))))
    except (TypeError, ValueError):
        n = default
    n = max(20, min(80, n))
    # Snap to the nearest valid scouting grade (ties resolve to the lower one).
    return min(SCOUTING_GRADES, key=lambda g: abs(g - n))


def pitchers_payload() -> list[dict]:
    if not PROJ_PATH.exists():
        return []
    df = pd.read_csv(PROJ_PATH, low_memory=False)
    df = df[df["player_type"].astype(str).str.lower() == "pitcher"]
    rows: list[dict] = []
    for _, r in df.iterrows():
        rows.append({
            "player_name": _safe(r.get("player_name"), "") or "",
            "mlb_id": int(_f(r.get("mlb_id"), 0)) or None,
            "team": _safe(r.get("team"), "") or "",
            "opponent": _safe(r.get("opponent"), "") or "",
            "proj_ip": _f(r.get("proj_ip")),
            "proj_strikeouts": _f(r.get("proj_strikeouts")),
            "proj_walks": _f(r.get("proj_walks")),
            "proj_hits_allowed": _f(r.get("proj_hits_allowed")),
            "proj_runs_allowed": _f(r.get("proj_runs_allowed")),
            "confidence": _conf_label(_safe(r.get("confidence"))),
            "confidence_score": _grade(r.get("confidence_score")),
        })
    return rows


def hitters_payload() -> list[dict]:
    if not PROJ_PATH.exists():
        return []
    df = pd.read_csv(PROJ_PATH, low_memory=False)
    df = df[df["player_type"].astype(str).str.lower() == "hitter"]
    rows: list[dict] = []
    for _, r in df.iterrows():
        ls_raw = _safe(r.get("lineup_spot"))
        try:
            lineup_spot = int(ls_raw) if ls_raw is not None else None
        except (TypeError, ValueError):
            lineup_spot = None
        rows.append({
            "player_name": _safe(r.get("player_name"), "") or "",
            "mlb_id": int(_f(r.get("mlb_id"), 0)) or None,
            "team": _safe(r.get("team"), "") or "",
            "opponent": _safe(r.get("opponent"), "") or "",
            "pos": _safe(r.get("pos"), "") or "",
            "lineup_spot": lineup_spot,
            "proj_pa": _f(r.get("proj_pa")),
            "proj_hits": _f(r.get("proj_hits")),
            "proj_hr": _f(r.get("proj_hr")),
            "proj_strikeouts": _f(r.get("proj_strikeouts")),
            "proj_walks": _f(r.get("proj_walks")),
            "proj_runs": _f(r.get("proj_runs")),
            "proj_rbi": _f(r.get("proj_rbi")),
            "confidence": _conf_label(_safe(r.get("confidence"))),
            "confidence_score": _grade(r.get("confidence_score")),
        })
    return rows


def accuracy_payload() -> dict:
    empty = {
        "overall_acc": 0.0,
        "correct_games": 0,
        "total_games": 0,
        "cumulative_accuracy": [],
        "labels": [],
        "picks": [],
    }
    if not PICKS_PATH.exists():
        return empty

    df = pd.read_csv(PICKS_PATH, low_memory=False)
    if "correct" not in df.columns:
        return empty

    df = df.copy()
    df["correct"] = pd.to_numeric(df["correct"], errors="coerce")
    if "game_date" in df.columns:
        df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")

    graded = df[df["correct"].notna()].copy()
    if graded.empty:
        return empty

    sort_cols = [c for c in ["game_date", "game_pk"] if c in graded.columns]
    if sort_cols:
        graded = graded.sort_values(sort_cols)

    total = len(graded)
    correct = int(graded["correct"].sum())
    overall = (correct / total) * 100.0 if total else 0.0

    cum_correct = 0
    cumulative: list[float] = []
    labels: list[int] = []
    for i, (_, row) in enumerate(graded.iterrows(), start=1):
        if row["correct"] == 1:
            cum_correct += 1
        cumulative.append((cum_correct / i) * 100.0)
        labels.append(i)

    # Take last 30 chronologically, then reverse so newest is first.
    recent = graded.tail(30).iloc[::-1]
    picks: list[dict] = []
    for _, r in recent.iterrows():
        gd = r.get("game_date")
        date_str = ""
        if isinstance(gd, pd.Timestamp) and pd.notna(gd):
            date_str = gd.strftime("%m-%d")
        is_correct = r["correct"] == 1
        picks.append({
            "date": date_str,
            "away": _safe(r.get("away_team"), "") or "",
            "home": _safe(r.get("home_team"), "") or "",
            "pick": _safe(r.get("predicted_winner"), "") or "",
            "result": "Win" if is_correct else "Loss",
            "correct": bool(is_correct),
        })

    return {
        "overall_acc": overall,
        "correct_games": correct,
        "total_games": total,
        "cumulative_accuracy": cumulative,
        "labels": labels,
        "picks": picks,
    }


def _player_metric_summary(df: pd.DataFrame, metrics: list[tuple[str, str, float]]) -> list[dict]:
    """
    Build per-metric accuracy summaries.

    metrics: list of (proj_col, actual_col, within_tol)
        within_tol — projection counts as "close" if |proj - actual| <= within_tol
    """
    out: list[dict] = []
    for proj_col, actual_col, tol in metrics:
        if proj_col not in df.columns or actual_col not in df.columns:
            continue
        sub = df[[proj_col, actual_col]].copy()
        sub[proj_col] = pd.to_numeric(sub[proj_col], errors="coerce")
        sub[actual_col] = pd.to_numeric(sub[actual_col], errors="coerce")
        sub = sub.dropna()
        n = len(sub)
        if n == 0:
            out.append({
                "metric": proj_col.replace("proj_", ""),
                "n": 0,
                "mae": None, "rmse": None, "bias": None,
                "within_tol": None, "tol": tol,
                "avg_proj": None, "avg_actual": None,
            })
            continue
        diff = sub[proj_col] - sub[actual_col]
        mae = float(np.mean(np.abs(diff)))
        rmse = float(np.sqrt(np.mean(diff ** 2)))
        bias = float(np.mean(diff))
        within = float((np.abs(diff) <= tol).mean() * 100.0)
        out.append({
            "metric": proj_col.replace("proj_", ""),
            "n": int(n),
            "mae": round(mae, 3),
            "rmse": round(rmse, 3),
            "bias": round(bias, 3),
            "within_tol": round(within, 1),
            "tol": tol,
            "avg_proj": round(float(sub[proj_col].mean()), 3),
            "avg_actual": round(float(sub[actual_col].mean()), 3),
        })
    return out


def _recent_player_rows(df: pd.DataFrame, kind: str, limit: int = 20) -> list[dict]:
    """Return the most recent graded rows for a player_type."""
    if df.empty:
        return []
    sub = df[df["player_type"].astype(str).str.lower() == kind].copy()
    sub = sub[sub["played"] == True]  # noqa: E712 — pandas boolean
    if sub.empty:
        return []
    if "game_date" in sub.columns:
        sub["_d"] = pd.to_datetime(sub["game_date"], errors="coerce")
        sub = sub.sort_values("_d", ascending=False)
    sub = sub.head(limit)

    rows: list[dict] = []
    for _, r in sub.iterrows():
        gd = r.get("game_date")
        date_str = str(gd)[:10] if gd else ""
        if kind == "pitcher":
            rows.append({
                "date": date_str,
                "player": _safe(r.get("player_name"), "") or "",
                "team": _safe(r.get("team"), "") or "",
                "opp": _safe(r.get("opponent"), "") or "",
                "proj_k": _f(r.get("proj_strikeouts")),
                "act_k": _f(r.get("actual_strikeouts")),
                "proj_ip": _f(r.get("proj_ip")),
                "act_ip": _f(r.get("actual_ip")),
                "proj_h": _f(r.get("proj_hits_allowed")),
                "act_h": _f(r.get("actual_hits_allowed")),
                "proj_r": _f(r.get("proj_runs_allowed")),
                "act_r": _f(r.get("actual_runs_allowed")),
            })
        else:  # hitter
            rows.append({
                "date": date_str,
                "player": _safe(r.get("player_name"), "") or "",
                "team": _safe(r.get("team"), "") or "",
                "opp": _safe(r.get("opponent"), "") or "",
                "proj_pa": _f(r.get("proj_pa")),
                "act_pa": _f(r.get("actual_pa")),
                "proj_h": _f(r.get("proj_hits")),
                "act_h": _f(r.get("actual_hits")),
                "proj_hr": _f(r.get("proj_hr")),
                "act_hr": _f(r.get("actual_hr")),
                "proj_k": _f(r.get("proj_strikeouts")),
                "act_k": _f(r.get("actual_strikeouts")),
            })
    return rows


PITCHER_METRICS = [
    # (proj_col, actual_col, within_tol)
    ("proj_strikeouts", "actual_strikeouts", 1.5),
    ("proj_walks",      "actual_walks",      1.0),
    ("proj_ip",         "actual_ip",         1.0),
    ("proj_hits_allowed", "actual_hits_allowed", 1.5),
    ("proj_runs_allowed", "actual_runs_allowed", 1.5),
]

HITTER_METRICS = [
    ("proj_pa",         "actual_pa",         1.0),
    ("proj_hits",       "actual_hits",       1.0),
    ("proj_hr",         "actual_hr",         0.5),
    ("proj_strikeouts", "actual_strikeouts", 1.0),
    ("proj_walks",      "actual_walks",      1.0),
]


def _apply_calibration_to_projections(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    """
    Shift each calibratable projection column by -bias if calibration says
    it's applied. Returns a copy.

    The accuracy log stores RAW projections; calibration is applied here at
    display time so:
      • bias measurement (in calibrate_projections.py) always sees raw-vs-actual
        and stays accurate, no oscillation
      • MAE/within numbers shown to users reflect the corrected projections
        they actually see on the live site
    """
    if df is None or df.empty:
        return df
    try:
        import json as _json
        cal_path = BASE_DIR / "calibration.json"
        if not cal_path.exists():
            return df
        cal = _json.loads(cal_path.read_text())
    except Exception:
        return df
    block = (cal.get(kind) or {})
    if not block:
        return df

    out = df.copy()
    for col, info in block.items():
        if not info.get("applied", False):
            continue
        if col not in out.columns:
            continue
        bias = float(info.get("bias", 0.0))
        if not bias:
            continue
        out[col] = pd.to_numeric(out[col], errors="coerce") - bias
        out[col] = out[col].clip(lower=0)
    return out


def player_accuracy_payload() -> dict:
    """
    Aggregate player-projection accuracy for the Player Predictions sub-tab.

    The graded log stores RAW projections (so future calibration measurement
    stays stable — see calibrate_projections.py). We apply calibration here
    just before computing metrics so the MAE / within-tolerance numbers
    reflect what users actually see on the live site (which IS calibrated).
    """
    empty = {
        "available": False,
        "total_graded": 0,
        "pitchers": {"n": 0, "metrics": [], "recent": []},
        "hitters":  {"n": 0, "metrics": [], "recent": []},
    }
    if not PLAYER_ACC_PATH.exists():
        return empty
    try:
        df = pd.read_csv(PLAYER_ACC_PATH, low_memory=False)
    except Exception:
        return empty
    if df.empty or "played" not in df.columns:
        return empty

    df["played"] = df["played"].astype(str).str.lower().isin({"true", "1", "1.0"})
    graded = df[df["played"] == True].copy()  # noqa: E712
    if graded.empty:
        return empty

    pitchers_df = graded[graded["player_type"].astype(str).str.lower() == "pitcher"].copy()
    hitters_df  = graded[graded["player_type"].astype(str).str.lower() == "hitter"].copy()

    # Hitters: only count appearances where the hitter actually batted.
    if not hitters_df.empty and "actual_pa" in hitters_df.columns:
        hitters_df = hitters_df[pd.to_numeric(hitters_df["actual_pa"], errors="coerce").fillna(0) > 0]

    # Apply calibration to projections AT DISPLAY TIME — the underlying log
    # stays raw so future calibration measurement remains stable.
    pitchers_calibrated = _apply_calibration_to_projections(pitchers_df, "pitcher")
    hitters_calibrated  = _apply_calibration_to_projections(hitters_df,  "hitter")

    # Recent rows shown in the table also use calibrated values so users see
    # the same numbers everywhere on the site.
    recent_pool_pitcher = _apply_calibration_to_projections(graded, "pitcher")
    recent_pool_hitter  = _apply_calibration_to_projections(graded, "hitter")

    return {
        "available": True,
        "total_graded": int(len(graded)),
        "pitchers": {
            "n": int(len(pitchers_df)),
            "metrics": _player_metric_summary(pitchers_calibrated, PITCHER_METRICS),
            "recent": _recent_player_rows(recent_pool_pitcher, "pitcher", limit=25),
        },
        "hitters": {
            "n": int(len(hitters_df)),
            "metrics": _player_metric_summary(hitters_calibrated, HITTER_METRICS),
            "recent": _recent_player_rows(recent_pool_hitter, "hitter", limit=25),
        },
    }


# ─────────────────────────────────────────────────────────────
#  PLAYER DETAIL — used by the modal that opens when a user
#  clicks a player name. Returns recent-game-by-game stats plus
#  season / last10 / last5 aggregates split by opposing handedness.
# ─────────────────────────────────────────────────────────────
def _hitter_games_df() -> pd.DataFrame:
    """Loaded once and cached by _cached(). Slim columns to keep memory tight."""
    if not HITTER_GAMES_PATH.exists():
        return pd.DataFrame()
    cols = [
        "batter", "game_pk", "game_date", "team", "batter_hand",
        "pitcher_team", "opp_pitcher_name", "opp_pitcher_hand",
        "PA", "H", "HR", "BB", "K",
        "avg_EV", "max_EV", "avg_LA",
    ]
    df = pd.read_csv(HITTER_GAMES_PATH, usecols=lambda c: c in cols, low_memory=False)
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    return df


def _pitcher_games_df() -> pd.DataFrame:
    if not PITCHER_GAMES_PATH.exists():
        return pd.DataFrame()
    cols = [
        "pitcher", "game_pk", "game_date", "team",
        "pitcher_name", "pitcher_hand", "opponent_team", "is_actual_starter",
        "BF", "K", "BB", "HR", "H", "outs", "pitches",
        "avg_velocity", "avg_spin",
        # Pre-computed rolling vs-hand rates from hitterspitchers_data.py
        "pitcher_k_rate_vs_hand_L", "pitcher_k_rate_vs_hand_R",
        "pitcher_bb_rate_vs_hand_L", "pitcher_bb_rate_vs_hand_R",
        "pitcher_h_rate_vs_hand_L", "pitcher_h_rate_vs_hand_R",
        "pitcher_hr_rate_vs_hand_L", "pitcher_hr_rate_vs_hand_R",
        "pitcher_k_rate_vs_hand_last10_L", "pitcher_k_rate_vs_hand_last10_R",
        "pitcher_bb_rate_vs_hand_last10_L", "pitcher_bb_rate_vs_hand_last10_R",
        "pitcher_h_rate_vs_hand_last10_L", "pitcher_h_rate_vs_hand_last10_R",
        "pitcher_hr_rate_vs_hand_last10_L", "pitcher_hr_rate_vs_hand_last10_R",
    ]
    df = pd.read_csv(PITCHER_GAMES_PATH, usecols=lambda c: c in cols, low_memory=False)
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    return df


def _hitter_aggregate(rows: pd.DataFrame) -> dict:
    """Compute slash line + counting totals for a slice of hitter game rows."""
    if rows.empty:
        return {"games": 0, "pa": 0, "h": 0, "hr": 0, "bb": 0, "k": 0,
                "avg": None, "obp": None, "slg": None, "ops": None}
    pa = int(pd.to_numeric(rows["PA"], errors="coerce").fillna(0).sum())
    h  = int(pd.to_numeric(rows["H"],  errors="coerce").fillna(0).sum())
    hr = int(pd.to_numeric(rows["HR"], errors="coerce").fillna(0).sum())
    bb = int(pd.to_numeric(rows["BB"], errors="coerce").fillna(0).sum())
    k  = int(pd.to_numeric(rows["K"],  errors="coerce").fillna(0).sum())
    # Approx AB (we don't track HBP/SF separately — close enough for a slash line):
    ab = max(pa - bb, 0)
    # SLG approximation: only HR is a known XBH; the rest of H is treated as 1B.
    # This understates SLG slightly (no 2B/3B distinction) — note in the UI.
    tb = (h - hr) + (4 * hr)  # singles=1, hr=4. Doubles/triples folded into H.
    avg = (h / ab) if ab else None
    obp = ((h + bb) / pa) if pa else None
    slg = (tb / ab) if ab else None
    ops = (obp + slg) if (obp is not None and slg is not None) else None
    return {
        "games": int(len(rows)),
        "pa": pa, "h": h, "hr": hr, "bb": bb, "k": k,
        "avg": round(avg, 3) if avg is not None else None,
        "obp": round(obp, 3) if obp is not None else None,
        "slg": round(slg, 3) if slg is not None else None,
        "ops": round(ops, 3) if ops is not None else None,
    }


def _pitcher_aggregate(rows: pd.DataFrame) -> dict:
    """Compute rate stats over a slice of pitcher game rows."""
    if rows.empty:
        return {"games": 0, "ip": 0.0, "bf": 0, "k": 0, "bb": 0, "h": 0, "hr": 0,
                "k_rate": None, "bb_rate": None, "h_rate": None, "hr_rate": None,
                "whip": None, "k_per_9": None}
    bf  = int(pd.to_numeric(rows["BF"], errors="coerce").fillna(0).sum())
    k   = int(pd.to_numeric(rows["K"],  errors="coerce").fillna(0).sum())
    bb  = int(pd.to_numeric(rows["BB"], errors="coerce").fillna(0).sum())
    h   = int(pd.to_numeric(rows["H"],  errors="coerce").fillna(0).sum())
    hr  = int(pd.to_numeric(rows["HR"], errors="coerce").fillna(0).sum())
    outs = pd.to_numeric(rows["outs"], errors="coerce").fillna(0).sum()
    ip   = float(outs) / 3.0 if outs else 0.0
    return {
        "games": int(len(rows)),
        "ip": round(ip, 1),
        "bf": bf, "k": k, "bb": bb, "h": h, "hr": hr,
        "k_rate":  round(k / bf, 3)  if bf else None,
        "bb_rate": round(bb / bf, 3) if bf else None,
        "h_rate":  round(h / bf, 3)  if bf else None,
        "hr_rate": round(hr / bf, 3) if bf else None,
        "whip":    round((bb + h) / ip, 2) if ip else None,
        "k_per_9": round(k / ip * 9, 2) if ip else None,
    }


def _hitter_game_row(r) -> dict:
    """Compact row representation for the recent-games chart."""
    pa = int(pd.to_numeric(r.get("PA"), errors="coerce") or 0)
    h  = int(pd.to_numeric(r.get("H"),  errors="coerce") or 0)
    bb = int(pd.to_numeric(r.get("BB"), errors="coerce") or 0)
    hr = int(pd.to_numeric(r.get("HR"), errors="coerce") or 0)
    kk = int(pd.to_numeric(r.get("K"),  errors="coerce") or 0)
    ab = max(pa - bb, 0)
    avg = round(h / ab, 3) if ab else None
    return {
        "date": r.get("game_date").strftime("%m-%d") if pd.notna(r.get("game_date")) else "",
        "opp": _safe(r.get("pitcher_team"), "") or "",
        "opp_starter": _safe(r.get("opp_pitcher_name"), "") or "",
        "opp_hand": _safe(r.get("opp_pitcher_hand"), "") or "",
        "PA": pa, "H": h, "HR": hr, "BB": bb, "K": kk,
        "AVG": avg,
    }


def _pitcher_game_row(r) -> dict:
    bf  = int(pd.to_numeric(r.get("BF"), errors="coerce") or 0)
    k   = int(pd.to_numeric(r.get("K"),  errors="coerce") or 0)
    bb  = int(pd.to_numeric(r.get("BB"), errors="coerce") or 0)
    h   = int(pd.to_numeric(r.get("H"),  errors="coerce") or 0)
    hr  = int(pd.to_numeric(r.get("HR"), errors="coerce") or 0)
    outs = float(pd.to_numeric(r.get("outs"), errors="coerce") or 0)
    return {
        "date": r.get("game_date").strftime("%m-%d") if pd.notna(r.get("game_date")) else "",
        "opp": _safe(r.get("opponent_team"), "") or "",
        "started": bool(_safe(r.get("is_actual_starter"), False)),
        "BF": bf, "K": k, "BB": bb, "H": h, "HR": hr,
        "IP": round(outs / 3.0, 1) if outs else 0.0,
    }


def _build_hitter_detail(mlb_id: int) -> dict | None:
    """Build the hitter-side payload for `mlb_id`, or None if the player has
    no rows in the hitter game-data table."""
    hdf = _cached("__hitter_games_df", _hitter_games_df)
    if hdf.empty:
        return None
    sub = hdf[pd.to_numeric(hdf["batter"], errors="coerce") == mlb_id]
    if sub.empty:
        return None
    sub = sub.sort_values("game_date", ascending=False)
    recent10 = sub.head(10)
    recent5  = sub.head(5)

    def _split(rows: pd.DataFrame, hand: str | None) -> dict:
        if hand is None:
            return _hitter_aggregate(rows)
        return _hitter_aggregate(rows[rows["opp_pitcher_hand"].astype(str).str.upper() == hand])

    return {
        "player_type": "hitter",
        "mlb_id": int(mlb_id),
        "player_name": "",
        "team": _safe(sub.iloc[0].get("team"), "") or "",
        "hand": _safe(sub.iloc[0].get("batter_hand"), "") or "",
        "games_count": int(len(sub)),
        "recent_games": [_hitter_game_row(r) for _, r in recent10.iterrows()],
        "splits": {
            "season": {"all": _split(sub, None),
                       "vs_R": _split(sub, "R"),
                       "vs_L": _split(sub, "L")},
            "last10": {"all": _split(recent10, None),
                       "vs_R": _split(recent10, "R"),
                       "vs_L": _split(recent10, "L")},
            "last5":  {"all": _split(recent5, None),
                       "vs_R": _split(recent5, "R"),
                       "vs_L": _split(recent5, "L")},
        },
    }


def _build_pitcher_detail(mlb_id: int) -> dict | None:
    """Build the pitcher-side payload for `mlb_id`, or None if the player has
    no rows in the pitcher game-data table."""
    pdf = _cached("__pitcher_games_df", _pitcher_games_df)
    if pdf.empty:
        return None
    sub = pdf[pd.to_numeric(pdf["pitcher"], errors="coerce") == mlb_id]
    if sub.empty:
        return None
    sub = sub.sort_values("game_date", ascending=False)
    # Only actual starts in the recent-games view; relief outings would
    # dominate the per-game bar charts.
    starts = sub[sub["is_actual_starter"].astype(str).str.lower().isin({"true", "1", "1.0"})] \
        if "is_actual_starter" in sub.columns else sub
    if starts.empty:
        starts = sub
    recent10 = starts.head(10)
    recent5  = starts.head(5)

    latest = sub.iloc[0]
    def _vs_hand_block(suffix: str) -> dict:
        base = "pitcher_{stat}_rate_vs_hand{sfx}_{hand}"
        rates: dict = {}
        for hand in ("L", "R"):
            h = {}
            for stat, key in [("k", "k_rate"), ("bb", "bb_rate"),
                              ("h", "h_rate"), ("hr", "hr_rate")]:
                col = base.format(stat=stat, sfx=suffix, hand=hand)
                v = latest.get(col)
                h[key] = round(float(v), 3) if pd.notna(v) else None
            rates[f"vs_{hand}"] = h
        return rates

    return {
        "player_type":   "pitcher",
        "mlb_id":        int(mlb_id),
        "player_name":   _safe(latest.get("pitcher_name"), "") or "",
        "team":          _safe(latest.get("team"), "") or "",
        "hand":          _safe(latest.get("pitcher_hand"), "") or "",
        "games_count":   int(len(sub)),
        "starts_count":  int(len(starts)),
        "recent_games":  [_pitcher_game_row(r) for _, r in recent10.iterrows()],
        "splits": {
            "season": {"all": _pitcher_aggregate(starts), **_vs_hand_block("")},
            "last10": {"all": _pitcher_aggregate(recent10), **_vs_hand_block("_last10")},
            "last5":  {"all": _pitcher_aggregate(recent5)},
        },
    }


def _player_detail_payload(mlb_id: int) -> dict | None:
    """Build the modal payload. Detects two-way players (Shohei Ohtani et al.)
    by checking BOTH game-data tables and returns a combined response with the
    full hitter + pitcher payloads nested, so the front-end can render both."""
    hitter_payload  = _build_hitter_detail(mlb_id)
    pitcher_payload = _build_pitcher_detail(mlb_id)

    if hitter_payload and pitcher_payload:
        return {
            "player_type": "two_way",
            "mlb_id":      int(mlb_id),
            # pitcher table carries the player name; hitter table doesn't
            "player_name": pitcher_payload.get("player_name") or "",
            "team":        pitcher_payload.get("team") or hitter_payload.get("team") or "",
            "hand_bat":    hitter_payload.get("hand", ""),
            "hand_throw":  pitcher_payload.get("hand", ""),
            "games_count_bat":   hitter_payload.get("games_count", 0),
            "games_count_pitch": pitcher_payload.get("starts_count") or pitcher_payload.get("games_count", 0),
            "hitter":      hitter_payload,
            "pitcher":     pitcher_payload,
        }

    return hitter_payload or pitcher_payload


# ─────────────────────────────────────────────────────────────
#  PLAYER POWER RANKINGS — season-long production ranking, all
#  players + a 25-and-under cut. Rebuilt every cron tick by
#  player_rankings.py and written to outputs/player_rankings.json.
# ─────────────────────────────────────────────────────────────
def rankings_payload() -> dict:
    empty = {
        "available": False,
        "hitters": [],
        "pitchers": [],
        "note": "player_rankings.json not found — run "
                "player_rankings.py or wait for the next update.",
    }
    if not RANKINGS_PATH.exists():
        return empty
    try:
        import json
        bundle = json.loads(RANKINGS_PATH.read_text())
        bundle["available"] = True
        return bundle
    except Exception as e:
        return {
            "available": False,
            "hitters": [],
            "pitchers": [],
            "error": f"{type(e).__name__}: {str(e)[:200]}",
        }


# ───── ROUTES ───────────────────────────────────────────────
@app.get("/")
def index():
    if not INDEX_PATH.exists():
        raise HTTPException(status_code=500, detail="index.html missing")
    return FileResponse(INDEX_PATH)


@app.get("/favicon.svg")
def favicon_svg():
    if not FAVICON_PATH.exists():
        raise HTTPException(status_code=404, detail="favicon missing")
    return FileResponse(FAVICON_PATH, media_type="image/svg+xml")


@app.get("/favicon.ico")
def favicon_ico():
    # Browsers request /favicon.ico by default. Prefer the new mound PNG;
    # fall back to the old SVG if it's not on disk yet.
    png = BASE_DIR / "bullpenbet_logo_favicon.png"
    if png.exists():
        return FileResponse(png, media_type="image/png")
    if FAVICON_PATH.exists():
        return FileResponse(FAVICON_PATH, media_type="image/svg+xml")
    raise HTTPException(status_code=404, detail="favicon missing")


# Serve any bullpenbet_*.png|svg|jpg|gif|webp from the project root.
# Used by the topbar logo, the favicon, and any other brand assets the
# page embeds. Restricted to the bullpenbet_ prefix and rejects any
# path-traversal in the suffix.
@app.get("/bullpenbet_{rest:path}")
def serve_brand_asset(rest: str):
    if "/" in rest or ".." in rest:
        raise HTTPException(status_code=400, detail="bad path")
    fname = f"bullpenbet_{rest}"
    p = BASE_DIR / fname
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"{fname} not found")
    ext = p.suffix.lower()
    media = {".png": "image/png", ".svg": "image/svg+xml",
             ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
             ".gif": "image/gif", ".webp": "image/webp"}.get(ext)
    if not media:
        raise HTTPException(status_code=415, detail=f"unsupported type {ext}")
    return FileResponse(p, media_type=media)


@app.get("/api/games")
def api_games():
    return JSONResponse(_cached("games", games_payload))


@app.get("/api/pitchers")
def api_pitchers():
    return JSONResponse(_cached("pitchers", pitchers_payload))


@app.get("/api/hitters")
def api_hitters():
    return JSONResponse(_cached("hitters", hitters_payload))


@app.get("/api/accuracy")
def api_accuracy():
    return JSONResponse(_cached("accuracy", accuracy_payload))


@app.get("/api/player-accuracy")
def api_player_accuracy():
    return JSONResponse(_cached("player_accuracy", player_accuracy_payload))


@app.get("/api/calibration")
def api_calibration():
    """
    Returns the current bias-calibration block (rebuilt every cron tick by
    calibrate_projections.py). The accuracy page uses this to show the
    user which stats are being calibrated and by how much.
    """
    cal_path = BASE_DIR / "calibration.json"
    if not cal_path.exists():
        return JSONResponse({"available": False, "hitter": {}, "pitcher": {}})
    try:
        import json
        cal = json.loads(cal_path.read_text())
        cal["available"] = True
        return JSONResponse(cal)
    except Exception:
        return JSONResponse({"available": False, "hitter": {}, "pitcher": {}})


@app.get("/api/rankings")
def api_rankings():
    return JSONResponse(_cached("rankings", rankings_payload))


@app.get("/api/player/{mlb_id}")
def api_player_detail(mlb_id: int):
    """
    Detail endpoint for the per-player modal. Returns recent games + season
    / last-10 / last-5 aggregates split by opposing handedness.
    """
    payload = _cached(f"player_{mlb_id}", lambda: _player_detail_payload(mlb_id))
    if payload is None:
        raise HTTPException(status_code=404, detail=f"No game-data rows for mlb_id={mlb_id}")
    return JSONResponse(payload)


@app.get("/health")
def health():
    """
    Liveness probe + freshness indicators. The front-end footer reads this
    on page load and shows the user when the daily pipeline last produced
    output, so stale projections are visible at a glance.
    """
    info: dict[str, Any] = {
        "status": "ok",
        "ts": datetime.now(timezone.utc).isoformat(),
        "preds_csv_exists": PREDS_PATH.exists(),
        "proj_csv_exists": PROJ_PATH.exists(),
        "picks_csv_exists": PICKS_PATH.exists(),
        "player_acc_csv_exists": PLAYER_ACC_PATH.exists(),
    }

    # Pull the projection's stamped game_date — the most reliable signal of
    # "did today's pipeline actually run". The CSV stamps every row with its
    # target_date, so if game_date == today the cron updated successfully.
    # projection_status distinguishes why a projection isn't current so the
    # footer can show a useful message instead of pasting a null into a
    # sentence and saying "projection dated —, today is X".
    proj_game_date = None
    proj_row_count = None
    proj_status = "missing_file"
    server_today_et = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
    if PROJ_PATH.exists():
        try:
            df = pd.read_csv(PROJ_PATH, usecols=["game_date"], low_memory=False)
            if df.empty:
                proj_status = "empty_file"
            else:
                dates = df["game_date"].dropna()
                if dates.empty:
                    proj_status = "missing_game_date"
                else:
                    proj_game_date = str(dates.iloc[0])[:10]
                    proj_row_count = int(len(df))
                    proj_status = ("current" if proj_game_date == server_today_et
                                   else "stale")
        except ValueError:
            # usecols=['game_date'] raises ValueError when the column doesn't exist
            proj_status = "missing_game_date"
        except Exception:
            proj_status = "unreadable"
    info["projection_game_date"] = proj_game_date
    info["projection_rows"] = proj_row_count
    info["server_today_et"] = server_today_et
    info["projection_status"] = proj_status
    info["projection_is_today"] = (proj_status == "current")

    # Past-date snapshot coverage — how complete is the player accuracy log?
    snaps_dir = OUTPUT_DIR
    if snaps_dir.exists():
        try:
            import re
            pat = re.compile(r"hitterspitchers_(\d{4}-\d{2}-\d{2})\.csv$")
            n_present = 0
            n_populated = 0
            for f in snaps_dir.glob("hitterspitchers_*.csv"):
                m = pat.search(f.name)
                if not m:
                    continue
                n_present += 1
                try:
                    if f.stat().st_size >= 64:
                        n_populated += 1
                except OSError:
                    pass
            info["snapshots_present"] = n_present
            info["snapshots_populated"] = n_populated
        except Exception:
            info["snapshots_present"] = None
            info["snapshots_populated"] = None

    return info
