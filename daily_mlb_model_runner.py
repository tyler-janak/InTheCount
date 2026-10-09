# ===============================
# DAILY MLB MODEL RUNNER — game outcome projections + pick tracker
# ===============================

import os
import pickle
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests


# -------------------------------
# LOAD MODEL
# -------------------------------
def load_model(path):
    with open(path, "rb") as f:
        data = pickle.load(f)

    if isinstance(data, dict):
        return data["model"], data["features"]
    return data, list(data.feature_names_in_)


# -------------------------------
# SCRAPE GAMES / RESULTS
# -------------------------------
def get_games_for_date(date):
    url = "https://statsapi.mlb.com/api/v1/schedule"
    params = {
        "sportId": 1,
        "date": date,
        "hydrate": "probablePitcher,team",
    }

    # Empty-frame fallback used both for "no games" and "API unreachable".
    EMPTY = pd.DataFrame(
        columns=[
            "game_date", "game_pk", "home_team", "away_team",
            "home_starter", "away_starter", "home_score", "away_score",
            "status", "home_win", "actual_winner", "commence_time",
        ]
    )

    # Retry with exponential backoff so a single SSL/timeout blip doesn't
    # kill the whole grading pass. If all retries fail, return EMPTY so the
    # caller skips this date instead of crashing the pipeline.
    import time as _time
    data = None
    last_err = None
    for attempt in range(4):              # 4 tries -> waits 1, 2, 4, 8 sec
        try:
            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()
            data = r.json()
            break
        except Exception as e:
            last_err = e
            if attempt < 3:
                _time.sleep(2 ** attempt)
    if data is None:
        print(f"  [mlb-api {date}] giving up after 4 attempts: {last_err}")
        return EMPTY

    if "dates" not in data or len(data["dates"]) == 0:
        return EMPTY

    games = []
    for d in data.get("dates", []):
        for g in d.get("games", []):
            teams = g.get("teams", {})
            home_info = teams.get("home", {})
            away_info = teams.get("away", {})

            home_team = home_info.get("team", {}) or {}
            away_team = away_info.get("team", {}) or {}

            home_prob = home_info.get("probablePitcher", {}) or {}
            away_prob = away_info.get("probablePitcher", {}) or {}

            home_score = home_info.get("score")
            away_score = away_info.get("score")
            status = g.get("status", {}).get("detailedState")
            commence_time = g.get("gameDate")

            if home_score is not None and away_score is not None:
                home_win = int(home_score > away_score)
                actual_winner = home_team.get("abbreviation") if home_win == 1 else away_team.get("abbreviation")
            else:
                home_win = np.nan
                actual_winner = np.nan

            games.append(
                {
                    "game_date": pd.to_datetime(date),
                    "game_pk": g.get("gamePk"),
                    "home_team": home_team.get("abbreviation"),
                    "away_team": away_team.get("abbreviation"),
                    "home_starter": home_prob.get("fullName"),
                    "away_starter": away_prob.get("fullName"),
                    "home_score": home_score,
                    "away_score": away_score,
                    "status": status,
                    "home_win": home_win,
                    "actual_winner": actual_winner,
                    "commence_time": commence_time,
                }
            )

    return pd.DataFrame(games)


# -------------------------------
# BUILD FEATURES FROM HISTORY
# -------------------------------
def build_features(games_df, hist_df, features):
    games_df = games_df.copy()
    hist_df = hist_df.copy()

    games_df["game_date"] = pd.to_datetime(games_df["game_date"], errors="coerce")
    hist_df["game_date"] = pd.to_datetime(hist_df["game_date"], errors="coerce")
    hist_df = hist_df.sort_values(["game_date", "game_pk"]).copy()

    if games_df["game_date"].notna().any():
        slate_date = games_df["game_date"].min()
        hist_df = hist_df[hist_df["game_date"] < slate_date].copy()

    if hist_df.empty:
        out = games_df.copy()
        for f in features:
            if f == "home_field":
                out[f] = 1
            else:
                out[f] = 0.0
        return out

    home_team_cols = [c for c in hist_df.columns if c.startswith("home_")]
    away_team_cols = [c for c in hist_df.columns if c.startswith("away_")]

    team_exclude = {
        "home_team",
        "away_team",
        "home_starter",
        "away_starter",
        "home_starter_id",
        "away_starter_id",
        "home_starter_throws",
        "away_starter_throws",
        "home_runs",
        "away_runs",
        "home_runs_allowed",
        "away_runs_allowed",
    }

    usable_home_team_cols = [c for c in home_team_cols if c not in team_exclude]
    usable_away_team_cols = [c for c in away_team_cols if c not in team_exclude]

    home_team_df = hist_df[["game_date", "home_team"] + usable_home_team_cols].copy()
    away_team_df = hist_df[["game_date", "away_team"] + usable_away_team_cols].copy()

    home_team_df = home_team_df.rename(columns={"home_team": "team"})
    away_team_df = away_team_df.rename(columns={"away_team": "team"})

    home_team_df = home_team_df.rename(columns={c: c.replace("home_", "", 1) for c in usable_home_team_cols})
    away_team_df = away_team_df.rename(columns={c: c.replace("away_", "", 1) for c in usable_away_team_cols})

    team_long = pd.concat([home_team_df, away_team_df], ignore_index=True, sort=False)
    team_long = team_long.sort_values(["team", "game_date"])
    latest_team = team_long.groupby("team", as_index=False).tail(1).copy()

    home_sp_cols = [c for c in hist_df.columns if c.startswith("home_SP_")]
    away_sp_cols = [c for c in hist_df.columns if c.startswith("away_SP_")]

    starter_frames = []

    if "home_starter" in hist_df.columns and len(home_sp_cols) > 0:
        temp = hist_df[["game_date", "home_starter"] + home_sp_cols].copy()
        temp = temp.rename(columns={"home_starter": "starter"})
        temp = temp.rename(columns={c: c.replace("home_", "", 1) for c in home_sp_cols})
        starter_frames.append(temp)

    if "away_starter" in hist_df.columns and len(away_sp_cols) > 0:
        temp = hist_df[["game_date", "away_starter"] + away_sp_cols].copy()
        temp = temp.rename(columns={"away_starter": "starter"})
        temp = temp.rename(columns={c: c.replace("away_", "", 1) for c in away_sp_cols})
        starter_frames.append(temp)

    if starter_frames:
        starter_long = pd.concat(starter_frames, ignore_index=True, sort=False)
        starter_long = starter_long.sort_values(["starter", "game_date"])
        latest_starter = starter_long.groupby("starter", as_index=False).tail(1).copy()
    else:
        latest_starter = pd.DataFrame(columns=["starter"])

    out = games_df.copy()

    home_snapshot = latest_team.rename(columns={"team": "home_team"}).copy()
    home_snapshot = home_snapshot.rename(
        columns={c: f"home_{c}" for c in home_snapshot.columns if c not in ["game_date", "home_team"]}
    )
    out = out.merge(home_snapshot.drop(columns=["game_date"], errors="ignore"), on="home_team", how="left")

    away_snapshot = latest_team.rename(columns={"team": "away_team"}).copy()
    away_snapshot = away_snapshot.rename(
        columns={c: f"away_{c}" for c in away_snapshot.columns if c not in ["game_date", "away_team"]}
    )
    out = out.merge(away_snapshot.drop(columns=["game_date"], errors="ignore"), on="away_team", how="left")

    if not latest_starter.empty:
        home_sp = latest_starter.rename(columns={"starter": "home_starter"}).copy()
        home_sp = home_sp.rename(
            columns={c: f"home_{c}" for c in home_sp.columns if c not in ["game_date", "home_starter"]}
        )
        out = out.merge(home_sp.drop(columns=["game_date"], errors="ignore"), on="home_starter", how="left")

        away_sp = latest_starter.rename(columns={"starter": "away_starter"}).copy()
        away_sp = away_sp.rename(
            columns={c: f"away_{c}" for c in away_sp.columns if c not in ["game_date", "away_starter"]}
        )
        out = out.merge(away_sp.drop(columns=["game_date"], errors="ignore"), on="away_starter", how="left")

    needed_home_raw = sorted({f"home_{f.replace('diff_', '', 1)}" for f in features if f.startswith("diff_")})
    needed_away_raw = sorted({f"away_{f.replace('diff_', '', 1)}" for f in features if f.startswith("diff_")})

    for col in needed_home_raw:
        if col not in out.columns:
            out[col] = np.nan

    for col in needed_away_raw:
        if col not in out.columns:
            out[col] = np.nan

    for f in features:
        if f.startswith("diff_"):
            base = f.replace("diff_", "", 1)
            home_col = f"home_{base}"
            away_col = f"away_{base}"

            if home_col not in out.columns:
                out[home_col] = np.nan
            if away_col not in out.columns:
                out[away_col] = np.nan

            out[home_col] = pd.to_numeric(out[home_col], errors="coerce")
            out[away_col] = pd.to_numeric(out[away_col], errors="coerce")

            home_fill = pd.to_numeric(hist_df[home_col], errors="coerce").median(skipna=True) if home_col in hist_df.columns else np.nan
            away_fill = pd.to_numeric(hist_df[away_col], errors="coerce").median(skipna=True) if away_col in hist_df.columns else np.nan

            if pd.isna(home_fill):
                home_fill = out[home_col].median(skipna=True)
            if pd.isna(away_fill):
                away_fill = out[away_col].median(skipna=True)

            if pd.isna(home_fill):
                home_fill = 0.0
            if pd.isna(away_fill):
                away_fill = 0.0

            out[f] = out[home_col].fillna(home_fill) - out[away_col].fillna(away_fill)

        elif f == "home_field":
            out[f] = 1

        else:
            if f in out.columns:
                out[f] = pd.to_numeric(out[f], errors="coerce")
            else:
                fallback = pd.to_numeric(hist_df[f], errors="coerce").median(skipna=True) if f in hist_df.columns else 0.0
                out[f] = 0.0 if pd.isna(fallback) else fallback

    return out


# -------------------------------
# SAFE PROBABILITY EXTRACTION
# -------------------------------
def _predict_home_prob(model, X):
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)

        if proba.ndim == 1:
            return proba

        if hasattr(model, "classes_"):
            classes = list(model.classes_)
            if 1 in classes:
                home_idx = classes.index(1)
                return proba[:, home_idx]

        if proba.shape[1] == 2:
            return proba[:, 1]

        return proba[:, -1]

    preds = model.predict(X)
    return np.asarray(preds, dtype=float)


# -------------------------------
# PREDICT
# -------------------------------
def predict_games(df, model, features):
    df = df.copy()

    for f in features:
        if f not in df.columns:
            df[f] = 0

    X = df[features].fillna(0)

    df["home_win_prob"] = _predict_home_prob(model, X)
    df["away_win_prob"] = 1 - df["home_win_prob"]

    df["predicted_home_win"] = (df["home_win_prob"] > 0.5).astype(int)
    df["predicted_winner"] = np.where(df["predicted_home_win"] == 1, df["home_team"], df["away_team"])

    return df


# -------------------------------
# SAVE TODAY'S PICKS
# -------------------------------
def append_today_picks(preds, picks_file="mlb_pick_log.csv"):
    picks = preds.copy()

    keep_cols = [
        "game_date",
        "game_pk",
        "commence_time",
        "away_team",
        "home_team",
        "away_starter",
        "home_starter",
        "predicted_winner",
        "home_win_prob",
        "away_win_prob",
    ]

    for c in keep_cols:
        if c not in picks.columns:
            picks[c] = np.nan

    picks = picks[keep_cols].copy()
    picks["game_date"] = pd.to_datetime(picks["game_date"], errors="coerce")

    if os.path.exists(picks_file):
        old = pd.read_csv(picks_file)
        if "game_date" in old.columns:
            old["game_date"] = pd.to_datetime(old["game_date"], errors="coerce")
        out = pd.concat([old, picks], ignore_index=True)
        out = out.drop_duplicates(subset=["game_pk"], keep="last")
    else:
        out = picks.copy()

    out = out.sort_values(["game_date", "game_pk"])
    out.to_csv(picks_file, index=False)
    return out


# -------------------------------
# GRADE SAVED PICKS
# -------------------------------
def grade_saved_picks(picks_file="mlb_pick_log.csv", output_file="mlb_pick_log_graded.csv"):
    if not os.path.exists(picks_file):
        print("Pick log does not exist.")
        return pd.DataFrame()

    picks = pd.read_csv(picks_file)
    if picks.empty:
        print("Pick log is empty.")
        return picks

    picks["game_date"] = pd.to_datetime(picks["game_date"], errors="coerce")

    all_dates = sorted(picks["game_date"].dropna().dt.strftime("%Y-%m-%d").unique().tolist())
    results_list = []

    for date_str in all_dates:
        day_results = get_games_for_date(date_str)
        if day_results.empty:
            continue
        results_list.append(day_results)

    if not results_list:
        print("No game results found yet.")
        return pd.DataFrame()

    results = pd.concat(results_list, ignore_index=True)
    results["game_date"] = pd.to_datetime(results["game_date"], errors="coerce")

    picks = picks.drop(columns=["actual_winner", "home_score", "away_score", "status", "correct"], errors="ignore")

    graded = picks.merge(
        results[["game_pk", "actual_winner", "home_score", "away_score", "status"]],
        on="game_pk",
        how="left",
    )

    graded["correct"] = np.where(
        graded["actual_winner"].notna(),
        graded["predicted_winner"] == graded["actual_winner"],
        np.nan,
    )

    graded = graded.sort_values(["game_date", "game_pk"]).reset_index(drop=True)

    graded.to_csv(output_file, index=False)
    return graded


# -------------------------------
# SEASON ACCURACY REPORT
# -------------------------------
def season_accuracy_report(picks_file="mlb_pick_log.csv"):
    if not os.path.exists(picks_file):
        print("Pick log does not exist. Run backfill_season first.")
        return pd.DataFrame()

    picks = pd.read_csv(picks_file)
    if picks.empty:
        print("Pick log is empty.")
        return picks

    picks["game_date"] = pd.to_datetime(picks["game_date"], errors="coerce")

    needs_result = picks[picks.get("actual_winner", pd.Series(dtype=str)).isna()].copy() if "actual_winner" in picks.columns else picks.copy()
    dates_to_fetch = sorted(needs_result["game_date"].dropna().dt.strftime("%Y-%m-%d").unique().tolist())

    results_list = []
    for date_str in dates_to_fetch:
        day_results = get_games_for_date(date_str)
        if not day_results.empty:
            results_list.append(day_results[["game_pk", "actual_winner", "home_score", "away_score", "status"]])

    if results_list:
        fresh_results = pd.concat(results_list, ignore_index=True)
        if "actual_winner" in picks.columns:
            picks = picks.drop(columns=["actual_winner", "home_score", "away_score", "status"], errors="ignore")
        picks = picks.merge(fresh_results, on="game_pk", how="left")
        picks.to_csv(picks_file, index=False)

    graded = picks.copy()
    graded["correct"] = np.where(
        graded["actual_winner"].notna(),
        graded["predicted_winner"] == graded["actual_winner"],
        np.nan,
    )

    finished = graded[graded["correct"].notna()].copy()

    if finished.empty:
        print("No finished games with results yet.")
        return graded

    total_games = len(finished)
    correct_games = finished["correct"].sum()
    overall_acc = correct_games / total_games

    print("\n" + "=" * 55)
    print("           SEASON ACCURACY REPORT")
    print("=" * 55)
    print(f"  Total graded games : {total_games}")
    print(f"  Overall accuracy   : {overall_acc:.1%}  ({int(correct_games)}-{total_games - int(correct_games)})")
    print("=" * 55)

    finished["month"] = finished["game_date"].dt.to_period("M")
    monthly = (
        finished.groupby("month")
        .agg(
            games=("correct", "count"),
            wins=("correct", "sum"),
        )
        .reset_index()
    )
    monthly["accuracy"] = monthly["wins"] / monthly["games"]

    print("\n  Monthly Breakdown:")
    print(f"  {'Month':<12} {'W-L':>8} {'Acc':>8}")
    print("  " + "-" * 30)
    for _, row in monthly.iterrows():
        w = int(row["wins"])
        l = int(row["games"] - row["wins"])
        print(f"  {str(row['month']):<12} {f'{w}-{l}':>8} {row['accuracy']:>7.1%}")

    print("=" * 55 + "\n")
    return graded


# -------------------------------
# BACKFILL SEASON
# -------------------------------
def backfill_season(
    season_start,
    model_path="betting_model.pkl",
    history_path="2025_model_data.csv",
    picks_file="mlb_pick_log.csv",
    sleep_seconds=1.0,
):
    model, features = load_model(model_path)

    hist = pd.read_csv(history_path)
    hist["game_date"] = pd.to_datetime(hist["game_date"], errors="coerce")
    hist = hist[hist["game_date"] >= pd.to_datetime("2025-01-01")].copy()

    if "home_win" not in hist.columns:
        raise ValueError("Historical file must contain a 'home_win' column.")

    already_done = set()
    if os.path.exists(picks_file):
        existing = pd.read_csv(picks_file)
        if "game_date" in existing.columns:
            existing["game_date"] = pd.to_datetime(existing["game_date"], errors="coerce")
            already_done = set(existing["game_date"].dropna().dt.strftime("%Y-%m-%d").unique())

    start_dt = pd.to_datetime(season_start).date()
    yesterday = (datetime.today() - timedelta(days=1)).date()

    date_range = []
    cur = start_dt
    while cur <= yesterday:
        date_range.append(cur.strftime("%Y-%m-%d"))
        cur += timedelta(days=1)

    to_run = [d for d in date_range if d not in already_done]

    if not to_run:
        print("All dates already in pick log. Nothing to backfill.")
        return

    print(f"Backfilling {len(to_run)} dates from {to_run[0]} to {to_run[-1]}...")

    for date_str in to_run:
        try:
            games = get_games_for_date(date_str)

            if games.empty:
                print(f"  {date_str} — no games found, skipping.")
                time.sleep(sleep_seconds)
                continue

            finished = games[games["home_win"].notna()].copy()
            if finished.empty:
                print(f"  {date_str} — no finished games, skipping.")
                time.sleep(sleep_seconds)
                continue

            feats = build_features(finished, hist, features)

            if "home_win" not in feats.columns:
                feats["home_win"] = finished["home_win"].values

            for f in features:
                if f not in feats.columns:
                    feats[f] = 0

            X = feats[features].fillna(0)

            preds = feats.copy()
            preds["home_win_prob"] = _predict_home_prob(model, X)
            preds["away_win_prob"] = 1 - preds["home_win_prob"]
            preds["predicted_home_win"] = (preds["home_win_prob"] > 0.5).astype(int)
            preds["predicted_winner"] = np.where(
                preds["predicted_home_win"] == 1,
                preds["home_team"],
                preds["away_team"]
            )

            preds["actual_winner"] = finished["actual_winner"].values
            preds["home_score"] = finished["home_score"].values
            preds["away_score"] = finished["away_score"].values
            preds["status"] = finished["status"].values

            keep_cols = [
                "game_date",
                "game_pk",
                "commence_time",
                "away_team",
                "home_team",
                "away_starter",
                "home_starter",
                "predicted_winner",
                "home_win_prob",
                "away_win_prob",
                "actual_winner",
                "home_score",
                "away_score",
                "status",
            ]

            for c in keep_cols:
                if c not in preds.columns:
                    preds[c] = np.nan

            preds = preds[keep_cols].copy()
            preds["game_date"] = pd.to_datetime(preds["game_date"], errors="coerce")

            if os.path.exists(picks_file):
                old = pd.read_csv(picks_file)
                old["game_date"] = pd.to_datetime(old["game_date"], errors="coerce")
                combined = pd.concat([old, preds], ignore_index=True)
                combined = combined.drop_duplicates(subset=["game_pk"], keep="last")
            else:
                combined = preds.copy()

            combined = combined.sort_values(["game_date", "game_pk"])
            combined.to_csv(picks_file, index=False)

            print(f"  {date_str} — {len(finished)} game(s) saved.")

        except Exception as e:
            print(f"  {date_str} — ERROR: {e}")

        time.sleep(sleep_seconds)

    print(f"\nBackfill complete. Pick log saved to: {picks_file}")


# -------------------------------
# DAILY RUN
# -------------------------------
def run(
    date,
    model_path="betting_model.pkl",
    history_path="2025_model_data.csv",
    save_today_csv=True,
    save_pick_log=True,
    picks_file="mlb_pick_log.csv",
):
    model, features = load_model(model_path)

    hist = pd.read_csv(history_path)
    hist["game_date"] = pd.to_datetime(hist["game_date"], errors="coerce")
    hist = hist[hist["game_date"] >= pd.to_datetime("2025-01-01")].copy()

    if "home_win" not in hist.columns:
        raise ValueError("Your historical file must contain a 'home_win' column.")

    today_games = get_games_for_date(date)

    if today_games.empty:
        print(f"No games found for {date}")
        return pd.DataFrame()

    today_features = build_features(today_games, hist, features)

    if "home_win" not in today_features.columns:
        today_features["home_win"] = np.nan

    for f in features:
        if f not in today_features.columns:
            today_features[f] = 0

    X_today = today_features[features].fillna(0)

    preds = today_features.copy()
    preds["home_win_prob"] = _predict_home_prob(model, X_today)
    preds["away_win_prob"] = 1 - preds["home_win_prob"]

    preds["predicted_home_win"] = (preds["home_win_prob"] > 0.5).astype(int)
    preds["predicted_winner"] = np.where(
        preds["predicted_home_win"] == 1, preds["home_team"], preds["away_team"]
    )

    if "game_pk" not in preds.columns and "game_pk" in today_games.columns:
        preds["game_pk"] = today_games["game_pk"].values
    if "commence_time" not in preds.columns and "commence_time" in today_games.columns:
        preds["commence_time"] = today_games["commence_time"].values

    if save_today_csv:
        output_dir = Path("outputs")
        output_dir.mkdir(parents=True, exist_ok=True)

        preds.to_csv(output_dir / "today_predictions.csv", index=False)
        preds.to_csv(output_dir / f"today_predictions_{date}.csv", index=False)

    if save_pick_log:
        append_today_picks(preds, picks_file=picks_file)

    print("\n=== TODAY ===")
    print(
        preds[
            [
                "away_team",
                "away_starter",
                "away_win_prob",
                "home_team",
                "home_starter",
                "home_win_prob",
                "predicted_winner",
            ]
        ]
    )

    return preds
