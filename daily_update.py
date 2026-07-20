"""
daily_update.py
===============
Single entry point for the daily MLB pipeline. Run this once per cron tick
and it will:

    0. Refresh 2026 Statcast pitch data via pybaseball and rebuild the four
       per-game feature CSVs in data/ (incremental — only pulls new dates).
    1. Backfill any missing game-pick rows for completed dates.
    2. Run today's game model (predictions + EV + bet log).
    3. Re-grade the season pick log against final scores.
    4. Rebuild the NRFI feature table, then backfill any missing NRFI
       predictions for past dates.
    5. Generate today's NRFI predictions and append to the picks log.
    6. Grade past NRFI picks against MLB first-inning linescores.
    7. Backfill any missing dated player-projection snapshots from past
       dates (uses MLB Stats API for actual lineups — no Rotowire scraping
       so this works for any past date).
    8. Generate today's hitter / pitcher projections (for the live site).
    9. Grade every player snapshot against MLB box scores → rebuild
       2026_player_accuracy.csv.
   10. Recompute bias calibration from the graded log and apply it to
       today's projection (post-hoc fix for the systematic PA / IP
       under-projection observed in the 2025-data-only era).

NOTE: player-model retraining has been removed from the daily flow. The
models in models/*.pkl are trained offline via hitterspitchers_train.py
and only refreshed when you run that manually. Daily runs score off the
committed pickles with freshly-rebuilt rolling features, which is where
the day-to-day signal actually lives.

Steps 0, 4–6, and 7–9 are wrapped in their own try/except so a failure in
any sub-pipeline never blocks the others.  The data refresh is also
non-blocking — if pybaseball is unavailable or the network is flaky, the
pipeline continues with whatever data files already exist on disk.

Outputs touched (must be committed by the cron workflow):
    - data/hitter_game_data.csv
    - data/pitcher_game_data.csv
    - data/team_batting_hand_context.csv
    - data/team_pitching_hand_context.csv
    - data/nrfi_game_data.csv
    - 2026_picks_accuracy.csv
    - 2026_player_accuracy.csv
    - 2026_nrfi_picks.csv
    - 2026_nrfi_accuracy.csv
    - outputs/today_predictions_with_ev*.csv
    - outputs/today_bets_to_make*.csv
    - outputs/hitterspitchers_today.csv
    - outputs/hitterspitchers_<date>.csv  (one per past date)
    - outputs/nrfi_today.csv
    - outputs/nrfi_status.json
"""

import os
# Suppress sklearn's `joblib.delayed → sklearn.utils.parallel.delayed`
# UserWarning that fires once per predict() call (joblib subprocess
# workers inherit PYTHONWARNINGS at startup). Must come before any
# sklearn import — see hitterspitchers_today.py for the long version.
os.environ.setdefault("PYTHONWARNINGS", "ignore")
import warnings
warnings.filterwarnings("ignore")

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from daily_mlb_model_runner import backfill_season, grade_saved_picks, run

# Always run "today" in Eastern Time. GitHub Actions runners are UTC, but
# we want our day to roll over on ET so we never grade tomorrow before
# today's late West Coast games have actually finished.
ET = ZoneInfo("America/New_York")

SEASON_START     = "2026-03-25"
PICKS_FILE       = "2026_picks_accuracy.csv"
PLAYER_ACC_FILE  = "2026_player_accuracy.csv"
NRFI_PICKS_FILE  = "2026_nrfi_picks.csv"
NRFI_ACC_FILE    = "2026_nrfi_accuracy.csv"


import subprocess
import sys

# Pitch-level Statcast cache maintained by refresh_2026_data. The loop
# picks the first path that exists — if none match, _rebuild_nrfi_features
# raises loudly with the full list so the mismatch shows up in the cron
# log instead of silently freezing nrfi_game_data.csv. If your
# refresh_2026_data writes somewhere else, add that path FIRST.
PITCH_CACHE_CANDIDATES = [
    "pitch_data_2026.csv",
    "data/pitch_data_2026.csv",
    "data/statcast_2026.csv",
    "statcast_2026.csv",
    "data/pitch_cache_2026.csv",
]


def _rebuild_nrfi_features() -> None:
    """Rebuild data/nrfi_game_data.csv by running nrfi_data.py as a
    subprocess (it's a CLI script with a required --input arg, so we
    shell out rather than import it).

    Must run AFTER step 0 (so the pitch cache and pitcher_game_data.csv
    are current) and BEFORE the NRFI backfill/today steps. Without this,
    run_nrfi() either stub-bails at its existence check or scores off a
    table frozen at the last manual nrfi_data.py run.
    """
    pitch_csv = next((p for p in PITCH_CACHE_CANDIDATES if Path(p).exists()), None)
    if pitch_csv is None:
        raise FileNotFoundError(
            "no pitch-level Statcast cache found — checked: "
            + ", ".join(PITCH_CACHE_CANDIDATES)
            + " (update PITCH_CACHE_CANDIDATES to match refresh_2026_data's output)"
        )
    print(f"   using pitch cache: {pitch_csv}")

    result = subprocess.run(
        [sys.executable, "nrfi_data.py", "--input", pitch_csv],
        capture_output=True,
        text=True,
        timeout=1800,   # 30 min ceiling so a hang can't stall the whole cron run
    )
    # Surface the script's own progress output in the cron log
    if result.stdout:
        print(result.stdout)
    if result.returncode != 0:
        raise RuntimeError(
            f"nrfi_data.py exited with code {result.returncode}:\n{result.stderr}"
        )

    # Freshness check: confirm the rebuild actually advanced the table.
    # A "successful" run that leaves the last game_date days behind means
    # the pitch cache itself is stale (step 0 failed or wrote elsewhere).
    _verify_nrfi_table_freshness()


def _verify_nrfi_table_freshness(max_staleness_days: int = 2) -> None:
    """Print the last game_date in data/nrfi_game_data.csv and warn loudly
    if it lags today by more than max_staleness_days. Non-fatal — the
    pipeline continues, but the cron log makes the problem obvious."""
    path = Path("data/nrfi_game_data.csv")
    if not path.exists():
        print("⚠️  nrfi_game_data.csv still missing after rebuild")
        return
    try:
        dates = pd.read_csv(path, usecols=["game_date"], low_memory=False)
        last_dt = pd.to_datetime(dates["game_date"], errors="coerce").max()
        today_et = pd.Timestamp(datetime.now(ET).strftime("%Y-%m-%d"))
        if pd.isna(last_dt):
            print("⚠️  nrfi_game_data.csv has no parseable game_date values")
            return
        staleness = (today_et - last_dt).days
        print(f"   nrfi_game_data.csv last game_date: {last_dt.date()} "
              f"({staleness} day(s) behind today ET)")
        if staleness > max_staleness_days:
            print(f"⚠️  NRFI feature table is STALE (> {max_staleness_days} days) — "
                  f"the pitch cache used for the rebuild is likely not being "
                  f"refreshed by step 0. Check refresh_2026_data's output path "
                  f"vs PITCH_CACHE_CANDIDATES.")
    except Exception as e:
        print(f"⚠️  could not verify NRFI table freshness: {e}")


def main():
    today = datetime.now(ET).strftime("%Y-%m-%d")
    print(f"\n========== Daily update for {today} (ET) ==========\n")

    # ── DATA REFRESH (Statcast → per-game features) ─────────────────────
    # 0) Pull any new 2026 pitch-level Statcast data via pybaseball and
    #    rebuild data/hitter_game_data.csv + data/pitcher_game_data.csv +
    #    the team hand-context CSVs. Incremental — only new dates are
    #    pulled, the full feature build runs on the combined cache.
    try:
        from refresh_2026_data import refresh as refresh_2026_data
        print("── Refreshing 2026 player history from Statcast ────────────")
        refresh_2026_data(
            start=None,         # resume from cache; first run uses 2026-03-01
            end=None,           # default = today ET
            rebuild=False,
            skip_features=False,
        )
    except Exception as e:
        # Non-blocking — projections will fall back to whatever's already in
        # data/*.csv if the Statcast pull fails (network / pybaseball issue).
        print(f"⚠️  2026 data refresh failed: {e}")

    # 0a) Post-process hitter_game_data.csv with team-level rolling features
    # (team_obp_*, team_pa_avg_*, opp_sp_ip_avg alias). Idempotent — re-runs
    # drop the prior enrichment columns before recomputing.
    try:
        from enrich_team_features import enrich as _enrich_team_features
        _enrich_team_features()
    except Exception as e:
        print(f"⚠️  team-feature enrichment failed: {e}")

    # 0b) Add batter-level lineup aggregations to pitcher_game_data.csv
    # (lineup_k_rate / lineup_bb_rate / lineup_avg_ev / …).
    try:
        from enrich_lineup_features import enrich as _enrich_lineup_features
        _enrich_lineup_features()
    except Exception as e:
        print(f"⚠️  lineup-feature enrichment failed: {e}")

    # 0b-ii) True-talent (empirical-Bayes shrunk) rates + log5 lineup matchup.
    # Honest, leakage-free opponent/skill signal used by the pitcher + hitter
    # models. Runs after lineup enrichment so hitter rows are present.
    try:
        from enrich_truetalent import enrich as _enrich_truetalent
        _enrich_truetalent()
    except Exception as e:
        print(f"⚠️  true-talent enrichment failed: {e}")

    # NOTE: model retraining removed. Models are trained offline via
    # hitterspitchers_train.py and committed as pickles; the daily run
    # only rebuilds features and scores.

    # ── GAME PIPELINE ────────────────────────────────────────────────────
    # 1) Backfill any missing completed dates through yesterday.
    backfill_season(
        season_start=SEASON_START,
        model_path="betting_model.pkl",
        history_path="2025_model_data.csv",
        picks_file=PICKS_FILE,
        sleep_seconds=0.3,
    )

    # 2) Run today's slate and save today's picks / outputs.
    # Env var ODDS_API_KEY wins so you can rotate without code changes;
    # the hardcoded value is the fallback for routine local runs.
    odds_key = os.environ.get("ODDS_API_KEY") or "68e6ecc1ec696c25142abba270265126"
    run(
        date=today,
        odds_api_key=odds_key,
        model_path="betting_model.pkl",
        history_path="2025_model_data.csv",
        min_ev=0.02,
        save_today_csv=True,
        save_pick_log=True,
        picks_file=PICKS_FILE,
    )

    # 3) Re-grade the whole pick log so the season accuracy file always has
    #    fresh actual_winner / correct values for completed games.
    grade_saved_picks(
        picks_file=PICKS_FILE,
        output_file=PICKS_FILE,
    )

    # ── NRFI PIPELINE ────────────────────────────────────────────────────
    # 4a) Rebuild the NRFI feature table from fresh data. nrfi_data.py is
    #     a CLI script, so we run it as a subprocess with the pitch cache
    #     as --input. This was the missing link: nothing in the daily flow
    #     ever built nrfi_game_data.csv, so run_nrfi() stub-bailed (or
    #     scored off a frozen table) and the picks log never grew. The
    #     rebuild now also verifies the table's last game_date advanced.
    try:
        print("\n── Rebuilding NRFI feature table ───────────────────────────")
        _rebuild_nrfi_features()
    except Exception as e:
        print(f"⚠️  NRFI feature rebuild failed: {e}")
    # 4b) Backfill any past dates not yet in the NRFI picks log. run_nrfi()
    #     already filters features to < target_date, so this is safe for any
    #     historical date once nrfi_game_data.csv exists.
    try:
        from backfill_nrfi import backfill_nrfi
        print("\n── Backfilling past NRFI predictions ───────────────────────")
        backfill_nrfi(
            start=SEASON_START,
            end=None,           # auto = yesterday ET
            picks_file=NRFI_PICKS_FILE,
            force=False,        # skip dates already in the log
            sleep_seconds=0.4,
            verbose=True,
        )
    except Exception as e:
        print(f"⚠️  NRFI backfill failed: {e}")

    # 5) Generate TODAY's NRFI predictions and append to the picks log.
    #    An empty result now logs loudly instead of silently no-opping —
    #    a stub-bail upstream used to look identical to an off-day here.
    #    (nrfi_today.py now writes a header-only CSV + nrfi_status.json on
    #    a bail, so the site shows "no predictions" instead of a phantom
    #    "No Team vs No Team" card.)
    try:
        from nrfi_today import run_nrfi
        from backfill_nrfi import append_nrfi_picks
        print("\n── Running today's NRFI predictions ────────────────────────")
        nrfi_results = run_nrfi(today)
        if nrfi_results is not None and not nrfi_results.empty:
            append_nrfi_picks(nrfi_results, NRFI_PICKS_FILE)
            print(f"  Appended {len(nrfi_results)} NRFI row(s) to {NRFI_PICKS_FILE}")
        else:
            print(f"⚠️  run_nrfi returned no rows for {today} — "
                  f"nothing appended to {NRFI_PICKS_FILE} "
                  f"(check for a stub-bail reason above)")
    except Exception as e:
        print(f"⚠️  Today's NRFI predictions failed: {e}")

    # 6) Grade every NRFI pick in the log against the MLB Stats API linescore
    #    and rebuild 2026_nrfi_accuracy.csv.
    try:
        from nrfi_grade import grade_nrfi_picks
        print("\n── Grading NRFI predictions vs MLB linescores ──────────────")
        grade_nrfi_picks(
            picks_file=NRFI_PICKS_FILE,
            output_file=NRFI_ACC_FILE,
        )
    except Exception as e:
        print(f"⚠️  NRFI grading failed: {e}")

    # ── PLAYER PIPELINE ──────────────────────────────────────────────────
    # 7) Backfill any missing past-date snapshots FIRST. The backfill loop
    #    calls run_projections(date, write_today_alias=False) for each past
    #    date, which only writes outputs/hitterspitchers_<date>.csv (no
    #    overwriting of the live "today" alias). Idempotent — dates that
    #    already have a populated snapshot are skipped.
    try:
        from backfill_player_predictions import backfill as backfill_player_predictions
        print("\n── Backfilling past-date player snapshots ──────────────────")
        backfill_player_predictions(
            start=SEASON_START,
            end=None,           # auto = yesterday ET
            force=False,        # skip dates that already have a populated snapshot
            grade=False,        # we'll grade once at the end after today's run too
            verbose=False,
        )
    except Exception as e:
        # Backfill is non-blocking.
        print(f"⚠️  Player backfill failed: {e}")

    # 8) Generate TODAY's hitter / pitcher projections last so the live
    #    "today" alias (outputs/hitterspitchers_today.csv) reflects the
    #    current slate — not whichever past date the backfill ended on.
    try:
        from hitterspitchers_today import run_projections
        print("\n── Running today's player projections ──────────────────────")
        run_projections(today)
    except Exception as e:
        print(f"⚠️  Today's player projections failed: {e}")

    # 9) Grade every snapshot (past + today) against MLB box scores and
    #    rebuild 2026_player_accuracy.csv.
    try:
        from grade_player_predictions import grade_player_predictions
        print("\n── Grading player projections vs MLB box scores ────────────")
        grade_player_predictions(
            snapshots_dir="outputs",
            output_file=PLAYER_ACC_FILE,
            season_start=SEASON_START,
        )
    except Exception as e:
        # Grading is non-blocking — game accuracy must still update even
        # if box-score endpoints are slow or rate-limited.
        print(f"⚠️  Player grading failed: {e}")

    # 10) Rebuild calibration from the freshly-graded log, then apply it
    #    to the live "today" projection so users see bias-corrected
    #    numbers. The corrections are conservative — they only fire if
    #    we have ≥30 graded games for that stat AND |bias| ≥ 0.05.
    try:
        from hitterspitchers_today import RAW_MODEL_ONLY
    except Exception:
        RAW_MODEL_ONLY = False
    if RAW_MODEL_ONLY:
        print("\n── Skipping display calibration (RAW_MODEL_ONLY: showing model output verbatim) ──")
    else:
        try:
            from calibrate_projections import (
                compute_calibration, save_calibration,
                calibrate_today_csv, _print_calibration,
            )
            print("\n── Rebuilding bias calibration from accuracy log ───────────")
            cal = compute_calibration(min_n=30)
            _print_calibration(cal)
            save_calibration(cal)
            print("\n── Applying calibration to today's projection ──────────────")
            calibrate_today_csv(cal=cal)
        except Exception as e:
            print(f"⚠️  Calibration step failed: {e}")

    # ── PROPS / EDGE ENGINE ─────────────────────────────────────────────
    # 11) Pull today's player-prop lines from The Odds API, join to the
    #     freshly-calibrated projections, compute edge/EV per side, and
    #     write outputs/today_props_with_ev.csv. Also appends each prop
    #     to 2026_props_log.csv with stage='open' so the close-line
    #     snapshot at 5pm and the post-game grader can find them.
    try:
        from props_fetch import fetch_today_props, compute_edge_today, log_clv_close
        print("\n── Fetching today's player props from The Odds API ─────────")
        props_df = fetch_today_props()
        edged = None
        if props_df is not None and not props_df.empty:
            edged = compute_edge_today()
            # On the 5pm cron tick (lineups locked, ~first pitch), also stamp
            # the props as "close" so CLV gets both bookends.
            try:
                hour_et = datetime.now(ET).hour
                if hour_et >= 17 and edged is not None and not edged.empty:
                    log_clv_close(edged)
                    print("   logged close lines for CLV tracking")
            except Exception:
                pass
    except Exception as e:
        print(f"⚠️  Props fetch / edge step failed: {e}")

    # 12) Grade past Value-flagged props against actual results from the
    #     player accuracy log. Builds 2026_props_accuracy.csv +
    #     2026_props_clv.csv summary.
    try:
        from props_grade import grade_props, compute_clv
        print("\n── Grading past prop picks vs actuals + computing CLV ──────")
        grade_props()
        compute_clv()
    except Exception as e:
        print(f"⚠️  Props grading step failed: {e}")

    # 13) Rebuild fantasy rankings from the freshly-graded accuracy log.
    #     Outputs outputs/fantasy_rankings.json which the Fantasy tab on
    #     the live site reads via /api/fantasy/rankings.
    try:
        from fantasy_rankings import build_rankings
        import json
        print("\n── Rebuilding fantasy rankings ─────────────────────────────")
        bundle = build_rankings()
        out_path = Path("outputs") / "fantasy_rankings.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(bundle, f, indent=2)
        print(f"   {len(bundle['hitters']):,} hitters, "
              f"{len(bundle['pitchers']):,} pitchers → {out_path}")
    except Exception as e:
        print(f"⚠️  Fantasy rankings step failed: {e}")

    print("\n✅ Daily update complete")


if __name__ == "__main__":
    main()
    main()
