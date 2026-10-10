"""
rerun_season.py
===============
Re-runs the whole 2026 season for the website with models that never saw a
2026 game, so every graded pick and projection on the site is genuinely
out-of-sample.

Why a separate set of models: the deployed models are refit through June 2026.
Re-scoring March-June with them would be in-sample (the grader flags those
rows and the site hides them, and the game backfill skips those dates). The
"pre-season" models here use no 2026 game at all (player models train on 2025
rows; the game model on 2024-2025 games), exactly as they could have been built
on opening day, and are frozen for the whole season.

Nothing here touches the deployed models (models/, betting_model.pkl) or the
report's evaluation files. Pre-season models and a copy of the current live
ledgers go under backups/ (git-ignored).

Usage (run from the project folder, in order):
    python rerun_season.py --train      # 1. build the pre-season models (~10-20 min)
    python rerun_season.py --run        # 2. back up ledgers, re-score every 2026 date, grade (~1-2 h)
    python rerun_season.py --restore    # undo step 2 from the most recent backup

Options for --run:  --start 2026-03-25  --end YYYY-MM-DD (default yesterday ET)
                    --games-only / --players-only
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

HERE = Path(__file__).resolve().parent
ET = ZoneInfo("America/New_York")

SEASON_START = "2026-03-25"
PRE_DIR = HERE / "backups" / "preseason"
PRE_PLAYER_DIR = PRE_DIR / "models"
PRE_GAME_MODEL = PRE_DIR / "betting_model_preseason.pkl"
BACKUP_ROOT = HERE / "backups" / "live_ledgers"

PICKS_FILE = HERE / "2026_picks_accuracy.csv"
PLAYER_ACC_FILE = HERE / "2026_player_accuracy.csv"
OUTPUTS = HERE / "outputs"

# Pre-season protocol, mirroring the 2026 evaluation one season earlier:
# train < 2025-07-01 <= validation (model choice) < 2026-01-01; the chosen
# model is then refit on all of 2025 and frozen.
PRE_TRAIN_SEASON = 2025
PRE_VALID_START = "2025-07-01"
PRE_CUTOFF = "2026-01-01"


def _yesterday() -> str:
    return (datetime.now(ET) - timedelta(days=1)).strftime("%Y-%m-%d")


def _trained_through(path: Path) -> str | None:
    import pickle
    try:
        with open(path, "rb") as f:
            b = pickle.load(f)
        return b.get("trained_through") if isinstance(b, dict) else None
    except Exception:
        return None


# --------------------------------------------------------------------- train
def train() -> None:
    PRE_PLAYER_DIR.mkdir(parents=True, exist_ok=True)

    print("== Pre-season player models (2025 rows only) ==")
    subprocess.run([sys.executable, str(HERE / "hitterspitchers_train.py"),
                    "--model-dir", str(PRE_PLAYER_DIR),
                    "--train-seasons", str(PRE_TRAIN_SEASON),
                    "--valid-start", PRE_VALID_START], check=True, cwd=HERE)

    print("\n== Pre-season game model (games before 2026) ==")
    import train_game_model as tgm
    tgm.run(valid_start=PRE_VALID_START, test_start=PRE_CUTOFF, production=False,
            out=str(PRE_GAME_MODEL), promote=False, write_eval=False)

    pt = _trained_through(PRE_PLAYER_DIR / "pitcher_K.pkl")
    gt = _trained_through(PRE_GAME_MODEL)
    print(f"\nPlayer models trained through {pt}; game model trained through {gt}.")
    for name, d in (("player", pt), ("game", gt)):
        if not d or d >= PRE_CUTOFF:
            sys.exit(f"STOP: {name} model trained through {d}, which is not before {PRE_CUTOFF}.")
    print("Both are before the 2026 season. Next: python rerun_season.py --run")


# -------------------------------------------------------------------- backup
def _snapshot_files() -> list[Path]:
    return sorted(OUTPUTS.glob("hitterspitchers_2026-*.csv"))


def backup() -> Path:
    dest = BACKUP_ROOT / datetime.now(ET).strftime("%Y%m%d_%H%M%S")
    (dest / "outputs").mkdir(parents=True, exist_ok=True)
    for f in (PICKS_FILE, PLAYER_ACC_FILE):
        if f.exists():
            shutil.copy2(f, dest / f.name)
    for f in _snapshot_files():
        shutil.copy2(f, dest / "outputs" / f.name)
    print(f"Backed up live ledgers and {len(_snapshot_files())} player snapshots -> {dest}")
    return dest


def restore() -> None:
    runs = sorted(p for p in BACKUP_ROOT.glob("*") if p.is_dir()) if BACKUP_ROOT.exists() else []
    if not runs:
        sys.exit("No backup found under backups/live_ledgers.")
    src = runs[-1]
    for name in (PICKS_FILE.name, PLAYER_ACC_FILE.name):
        if (src / name).exists():
            shutil.copy2(src / name, HERE / name)
    for f in (src / "outputs").glob("*.csv"):
        shutil.copy2(f, OUTPUTS / f.name)
    print(f"Restored ledgers and snapshots from {src}")


# ----------------------------------------------------------------------- run
def rerun_games(start: str, end: str, old_picks: pd.DataFrame) -> None:
    import daily_mlb_model_runner as dmr

    tmp = PICKS_FILE.with_suffix(".rerun.csv")
    tmp.unlink(missing_ok=True)
    # backfill_season runs every date from start through yesterday that is not
    # already in the pick file; an empty file means every date is re-scored.
    dmr.backfill_season(season_start=start, model_path=str(PRE_GAME_MODEL),
                        history_path=str(HERE / "2025_model_data.csv"),
                        picks_file=str(tmp), sleep_seconds=0.3)
    new = pd.read_csv(tmp) if tmp.exists() else pd.DataFrame()
    if new.empty:
        tmp.unlink(missing_ok=True)
        sys.exit("Game re-run produced no picks; live pick log left unchanged.")
    new["game_date"] = pd.to_datetime(new["game_date"], errors="coerce")
    new = new[new["game_date"] <= pd.Timestamp(end)]

    # keep live rows outside the re-run range (today's slate, anything after --end)
    keep = pd.DataFrame()
    if len(old_picks):
        od = pd.to_datetime(old_picks["game_date"], errors="coerce")
        keep = old_picks[(od < pd.Timestamp(start)) | (od > pd.Timestamp(end))].copy()
    out = pd.concat([keep, new], ignore_index=True, sort=False)
    out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    out = out.sort_values("game_date", kind="mergesort")
    out.to_csv(PICKS_FILE, index=False)
    tmp.unlink(missing_ok=True)

    dmr.grade_saved_picks(picks_file=str(PICKS_FILE), output_file=str(PICKS_FILE))
    print(f"Game picks re-run: {len(new):,} games {start} -> {end} (model through "
          f"{_trained_through(PRE_GAME_MODEL)}); {len(keep):,} live rows kept outside that range.")


def rerun_players(start: str, end: str) -> None:
    import hitterspitchers_today as hpt
    import backfill_player_predictions as bpp

    hpt.MODEL_DIR = PRE_PLAYER_DIR          # every loader and the trained-through stamp read this
    print(f"Player models: {hpt.MODEL_DIR} (trained through {hpt.model_trained_through()})")
    bpp.backfill(start=start, end=end, force=True, grade=False, verbose=False)

    from grade_player_predictions import grade_player_predictions
    grade_player_predictions(snapshots_dir=str(OUTPUTS), output_file=str(PLAYER_ACC_FILE),
                             season_start=SEASON_START)
    g = pd.read_csv(PLAYER_ACC_FILE, low_memory=False)
    if "in_sample" in g.columns:
        d = pd.to_datetime(g["game_date"], errors="coerce")
        r = g[(d >= pd.Timestamp(start)) & (d <= pd.Timestamp(end))]
        n_in = int((pd.to_numeric(r["in_sample"], errors="coerce") == 1).sum())
        print(f"Player rows {start} -> {end}: {len(r):,}; flagged in-sample: {n_in:,} "
              "(should be 0, except dates whose lineup fetch failed and kept the old snapshot)")


def run(start: str, end: str, games: bool, players: bool) -> None:
    for p, label in ((PRE_GAME_MODEL, "game"), (PRE_PLAYER_DIR / "pitcher_K.pkl", "player")):
        if not p.exists():
            sys.exit(f"Missing pre-season {label} model ({p}). Run: python rerun_season.py --train")
    old_picks = pd.read_csv(PICKS_FILE, low_memory=False) if PICKS_FILE.exists() else pd.DataFrame()
    backup()
    if games:
        print("\n== Re-running game picks ==")
        rerun_games(start, end, old_picks)
    if players:
        print("\n== Re-running player projections ==")
        rerun_players(start, end)
    print("\nDone. Check the numbers, then push with the usual git sequence. "
          "Undo with: python rerun_season.py --restore")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--train", action="store_true")
    g.add_argument("--run", action="store_true")
    g.add_argument("--restore", action="store_true")
    ap.add_argument("--start", default=SEASON_START)
    ap.add_argument("--end", default=None, help="last date to re-run (default: yesterday ET)")
    only = ap.add_mutually_exclusive_group()
    only.add_argument("--games-only", action="store_true")
    only.add_argument("--players-only", action="store_true")
    a = ap.parse_args()
    if a.train:
        train()
    elif a.restore:
        restore()
    else:
        run(a.start, a.end or _yesterday(), games=not a.players_only, players=not a.games_only)


if __name__ == "__main__":
    main()
