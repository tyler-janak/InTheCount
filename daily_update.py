"""
daily_update.py
===============
Single entry point for the daily MLB pipeline. Run this once per cron tick
and it will:

    0. refresh_full_history + build_features: keep the previous two seasons
       cached, top up the current season (3-day overlap) and rebuild the
       current season's windowed feature tables (two-season history rule).
    1. Backfill any missing game-pick rows for completed dates.
    2. Run today's game model (win-probability projections).
    3. Re-grade the season pick log against final scores.
    4. Backfill any missing dated player-projection snapshots from past
       dates (uses MLB Stats API for actual lineups - no Rotowire scraping
       so this works for any past date).
    5. Generate today's hitter / pitcher projections (for the live site).
    6. Grade every player snapshot against MLB box scores -> rebuild
       2026_player_accuracy.csv.
    7. Recompute bias calibration from the graded log and apply it to
       today's projection (post-hoc fix for the systematic PA / IP
       under-projection observed in the 2025-data-only era).
    8. Rebuild season player power rankings.

Steps 0 and 4–6 are wrapped in their own try/except so a failure in any
sub-pipeline never blocks the others. The data refresh is also
non-blocking - if pybaseball is unavailable or the network is flaky, the
pipeline continues with whatever data files already exist on disk.

Outputs touched (must be committed by the cron workflow):
    - data/hitter_game_data.csv
    - data/pitcher_game_data.csv
    - data/team_batting_hand_context.csv
    - data/team_pitching_hand_context.csv
    - 2026_picks_accuracy.csv
    - 2026_player_accuracy.csv
    - outputs/today_predictions*.csv
    - outputs/hitterspitchers_today.csv
    - outputs/hitterspitchers_<date>.csv  (one per past date)
    - outputs/player_rankings.json
"""

import os
# Suppress sklearn's `joblib.delayed -> sklearn.utils.parallel.delayed`
# UserWarning that fires once per predict() call (joblib subprocess
# workers inherit PYTHONWARNINGS at startup). Must come before any
# sklearn import - see hitterspitchers_today.py for the long version.
os.environ.setdefault("PYTHONWARNINGS", "ignore")
import warnings
warnings.filterwarnings("ignore")

import pickle
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from daily_mlb_model_runner import backfill_season, grade_saved_picks, run

# Always run "today" in Eastern Time. GitHub Actions runners are UTC, but
# we want our day to roll over on ET so we never grade tomorrow before
# today's late West Coast games have actually finished.
ET = ZoneInfo("America/New_York")

SEASON_START     = "2026-03-25"
PICKS_FILE       = "2026_picks_accuracy.csv"
PLAYER_ACC_FILE  = "2026_player_accuracy.csv"
MODELS_DIR       = Path("models")


def _load_preset_params(prefix: str, targets: list[str]) -> dict:
    """Read previously-tuned XGB hyperparameters out of existing model pickles.

    Lets the daily refit reuse the expensive hyperparameter search from an
    earlier offline `hitterspitchers_train.py` run instead of re-searching on
    every cron tick - we refit on fresh data + recalibrate, but keep the tuned
    depth / n_estimators / regularisation. Returns {target: params}.
    """
    out: dict[str, dict] = {}
    for t in targets:
        p = MODELS_DIR / f"{prefix}_{t}.pkl"
        if not p.exists():
            continue
        try:
            with open(p, "rb") as fh:
                bundle = pickle.load(fh)
            bp = bundle.get("best_params") if isinstance(bundle, dict) else None
            if bp:
                out[t] = bp
        except Exception:
            continue
    return out


def _retrain_player_models(tune: bool) -> None:
    """Retrain the player model stack on the freshly-refreshed feature tables.

    Model family (RF / XGB / NN) is chosen on the most recent 15% of dates
    (chronological validation), then refit on everything. Calibration is
    always on (it's cheap and removes systematic bias). Tuning
    reuses persisted hyperparameters when available; a full randomized search
    only runs when `tune=True` AND no preset exists for a target. The two-stage
    pitcher, xHits, and team-PA hitter models are refit on all rows so today's
    projection AND the past-date backfill score off current-data models.
    """
    import pandas as pd
    import hitterspitchers_train as hpt
    from run_backtest import training_seasons

    MODELS_DIR.mkdir(exist_ok=True)
    # Multi-season windowed tables (data/features/*_games.csv); the earliest
    # cached season is history-only once three seasons are cached.
    pitcher_df, hitter_df = hpt.load_training_tables()
    pitcher_df = pitcher_df[pd.to_datetime(pitcher_df["game_date"]).dt.year.isin(training_seasons(pitcher_df))]
    hitter_df = hitter_df[pd.to_datetime(hitter_df["game_date"]).dt.year.isin(training_seasons(hitter_df))]

    hpt.validate_pitcher_training_data(pitcher_df)

    pitcher_presets = _load_preset_params("pitcher", hpt.PITCHER_TARGETS + hpt.PITCHER_RATE_TARGETS)
    hitter_presets  = _load_preset_params("hitter",  hpt.HITTER_TARGETS + hpt.HITTER_RATE_TARGETS)

    hpt.train_pitcher_models(pitcher_df, MODELS_DIR, tune=tune, calibrate=True,
                             preset_params=pitcher_presets)
    hpt.train_hitter_models(hitter_df, MODELS_DIR, tune=tune, calibrate=True,
                            preset_params=hitter_presets)

    # Advanced decomposition stack (two-stage per-9 / xHits / team-PA). These
    # are NOT used by run_projections anymore - the direct counting-stat models
    # above score better overall, so USE_DECOMPOSITION_MODELS is False in
    # hitterspitchers_today.py. We skip retraining them by default to save cron
    # time; set BULLPEN_TRAIN_DECOMP=1 to keep them fresh (e.g. if you flip the
    # projection flag back on).
    if os.environ.get("BULLPEN_TRAIN_DECOMP", "0") == "1":
        try:
            from train_pitcher_two_stage import train_two_stage
            train_two_stage(eval_holdout=0.0)
        except Exception as e:
            print(f"WARNING:  two-stage pitcher retrain failed: {e}")
        try:
            from train_pitcher_xhits import train_xhits
            train_xhits(eval_holdout=0.0)
        except Exception as e:
            print(f"WARNING:  xHits retrain failed: {e}")
        try:
            from train_hitter_team_pa import train_team_pa, train_rate_models
            train_team_pa(eval_holdout=0.0)
            train_rate_models(eval_holdout=0.0)
        except Exception as e:
            print(f"WARNING:  team-PA retrain failed: {e}")


def _models_behind_data() -> bool:
    """True when the feature data has games the player models were not trained
    on. With several early-morning ticks this keeps the retrain to once a day."""
    try:
        import pandas as pd
        from hitterspitchers_today import model_trained_through
        tt = model_trained_through()
        last = pd.to_datetime(pd.read_csv("data/pitcher_game_data.csv", usecols=["game_date"])["game_date"]).max()
        return tt is None or pd.Timestamp(tt) < last.normalize()
    except Exception:
        return True


def main():
    today = datetime.now(ET).strftime("%Y-%m-%d")
    print(f"\n========== Daily update for {today} (ET) ==========\n")

    # -- DATA REFRESH (Statcast -> per-game features) ---------------------
    # 0) refresh_full_history: make sure the two previous seasons are cached
    #    (downloaded once, then restored from the Actions cache), top up the
    #    current season with a 3-day overlap, then build_features rebuilds the
    #    current season's window (previous two seasons + current season before
    #    each game) and exports data/*_game_data.csv + data/game_model_data.csv.
    #    The team / lineup / true-talent enrichment runs INSIDE that windowed
    #    build - re-running the old file-level enrich_* passes on the exported
    #    current-season CSVs would recompute cumulative features from the
    #    current season only and silently drop the prior-season history.
    try:
        from refresh_2026_data import refresh as refresh_current_season
        print("-- Refreshing Statcast + windowed feature tables ------------")
        refresh_current_season(start=None, end=None, rebuild=False, skip_features=False)
    except Exception as e:
        # Non-blocking - projections fall back to the data/*.csv already on disk.
        print(f"WARNING:  data refresh failed: {e}")

    # 0c) RETRAIN the player models on the freshly-refreshed data so that both
    # today's projection (step 8) and the past-date backfill (step 7) score
    # off current-data, tuned + calibrated models. By default this runs only on
    # the early-morning (3 AM ET) cron tick so the midday/evening lineup-update
    # ticks stay fast; override with BULLPEN_RETRAIN=force / skip. A full
    # hyperparameter search runs only when BULLPEN_TUNE=1 (otherwise the refit
    # reuses the hyperparameters tuned in the last offline run).
    retrain_mode = os.environ.get("BULLPEN_RETRAIN", "auto").lower()
    do_retrain = (
        retrain_mode == "force"
        or (retrain_mode == "auto" and datetime.now(ET).hour < 9 and _models_behind_data())
    )
    if retrain_mode == "skip":
        do_retrain = False
    if do_retrain:
        tune = os.environ.get("BULLPEN_TUNE", "0") == "1"
        try:
            print(f"\n-- Retraining player models (tune={'ON' if tune else 'reuse-preset'}, "
                  f"calibrate=ON) ---")
            _retrain_player_models(tune=tune)
        except Exception as e:
            # Non-blocking - if retrain fails we fall back to the committed
            # pickles and projections still run.
            print(f"WARNING:  player model retrain failed (using existing models): {e}")
    else:
        print("   (skipping model retrain this tick - set BULLPEN_RETRAIN=force to override)")

    # -- GAME PIPELINE ----------------------------------------------------
    # 1) Backfill any missing completed dates through yesterday.
    backfill_season(
        season_start=SEASON_START,
        model_path="betting_model.pkl",
        history_path="2025_model_data.csv",
        picks_file=PICKS_FILE,
        sleep_seconds=0.3,
    )

    # 2) Run today's slate and save today's picks / outputs.
    run(
        date=today,
        model_path="betting_model.pkl",
        history_path="2025_model_data.csv",
        save_today_csv=True,
        save_pick_log=True,
        picks_file=PICKS_FILE,
    )

    # 2b) From 3 PM ET on, also build TOMORROW's slate (probable starters are
    #     posted the day before) into next_day_* files. The site switches to
    #     them at midnight ET, so it no longer shows yesterday's games, pitchers
    #     and lineups until the (often hours-late) morning GitHub run lands.
    #     No pick-log entry is written for tomorrow; the day-of run logs it.
    now_et = datetime.now(ET)
    if now_et.hour >= 15:
        tomorrow = (now_et + timedelta(days=1)).strftime("%Y-%m-%d")
        try:
            print(f"\n-- Building tomorrow's slate ({tomorrow}) for the overnight site ----")
            run(date=tomorrow, model_path="betting_model.pkl", history_path="2025_model_data.csv",
                save_today_csv=True, save_pick_log=False, picks_file=PICKS_FILE,
                alias_name="next_day_predictions.csv")
        except Exception as e:
            print(f"WARNING:  tomorrow's game slate failed: {e}")

    # 3) Re-grade the whole pick log so the season accuracy file always has
    #    fresh actual_winner / correct values for completed games.
    grade_saved_picks(
        picks_file=PICKS_FILE,
        output_file=PICKS_FILE,
    )

    # -- PLAYER PIPELINE --------------------------------------------------
    # 4) Backfill any missing past-date snapshots FIRST. The backfill loop
    #    calls run_projections(date, write_today_alias=False) for each past
    #    date, which only writes outputs/hitterspitchers_<date>.csv (no
    #    overwriting of the live "today" alias). Idempotent - dates that
    #    already have a populated snapshot are skipped.
    try:
        from backfill_player_predictions import backfill as backfill_player_predictions
        print("\n-- Backfilling past-date player snapshots ------------------")
        backfill_player_predictions(
            start=SEASON_START,
            end=None,           # auto = yesterday ET
            force=False,        # skip dates that already have a populated snapshot
            grade=False,        # we'll grade once at the end after today's run too
            verbose=False,
        )
    except Exception as e:
        # Backfill is non-blocking.
        print(f"WARNING:  Player backfill failed: {e}")

    # 5) Generate TODAY's hitter / pitcher projections last so the live
    #    "today" alias (outputs/hitterspitchers_today.csv) reflects the
    #    current slate - not whichever past date the backfill ended on.
    try:
        from hitterspitchers_today import run_projections
        print("\n-- Running today's player projections ----------------------")
        run_projections(today)
    except Exception as e:
        print(f"WARNING:  Today's player projections failed: {e}")
    if datetime.now(ET).hour >= 15:
        try:
            tomorrow = (datetime.now(ET) + timedelta(days=1)).strftime("%Y-%m-%d")
            print(f"\n-- Tomorrow's probable-starter projections ({tomorrow}) -------")
            run_projections(tomorrow, out_name="next_day_projections.csv")
        except Exception as e:
            print(f"WARNING:  tomorrow's player projections failed: {e}")

    # 6) Grade every snapshot (past + today) against MLB box scores and
    #    rebuild 2026_player_accuracy.csv.
    try:
        from grade_player_predictions import grade_player_predictions
        print("\n-- Grading player projections vs MLB box scores ------------")
        grade_player_predictions(
            snapshots_dir="outputs",
            output_file=PLAYER_ACC_FILE,
            season_start=SEASON_START,
        )
    except Exception as e:
        # Grading is non-blocking - game accuracy must still update even
        # if box-score endpoints are slow or rate-limited.
        print(f"WARNING:  Player grading failed: {e}")

    # 7) Rebuild calibration from the freshly-graded log, then apply it
    #    to the live "today" projection so users see bias-corrected
    #    numbers. The corrections are conservative - they only fire if
    #    we have ≥30 graded games for that stat AND |bias| ≥ 0.05.
    try:
        from hitterspitchers_today import RAW_MODEL_ONLY
    except Exception:
        RAW_MODEL_ONLY = False
    if RAW_MODEL_ONLY:
        print("\n-- Skipping display calibration (RAW_MODEL_ONLY: showing model output verbatim) --")
    else:
        try:
            from calibrate_projections import (
                compute_calibration, save_calibration,
                calibrate_today_csv, _print_calibration,
            )
            print("\n-- Rebuilding bias calibration from accuracy log -----------")
            cal = compute_calibration(min_n=30)
            _print_calibration(cal)
            save_calibration(cal)
            print("\n-- Applying calibration to today's projection --------------")
            calibrate_today_csv(cal=cal)
        except Exception as e:
            print(f"WARNING:  Calibration step failed: {e}")

    # 8) Rebuild season player power rankings from the freshly-graded
    #    accuracy log. Outputs outputs/player_rankings.json which the
    #    Power Rankings tab on the live site reads via /api/rankings.
    try:
        from player_rankings import build_rankings
        import json
        print("\n-- Rebuilding player power rankings -------------------------")
        bundle = build_rankings()
        out_path = Path("outputs") / "player_rankings.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(bundle, f, indent=2)
        print(f"   {len(bundle['hitters']):,} hitters, "
              f"{len(bundle['pitchers']):,} pitchers -> {out_path}")
    except Exception as e:
        print(f"WARNING:  Player power rankings step failed: {e}")

    print("\n[OK] Daily update complete")


if __name__ == "__main__":
    main()
