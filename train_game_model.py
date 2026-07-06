"""
train_game_model.py
====================
Retrain the game-ML moneyline classifier (betting_model.pkl) that
daily_mlb_model_runner.py consumes. Saves in the bundle format that
runner expects:   {"model": estimator, "features": list[str]}

Target column: home_win  (1 = home team won, 0 = away team won)

Input: 2025_model_data.csv (already has team + starter rolling features
       and the home_win label).

Holdout: chronological tail (default 20%) for honest evaluation. Reports
accuracy, log-loss, Brier score, and a naive home-team baseline.

Usage
-----
    python train_game_model.py
    python train_game_model.py --test-frac 0.25 --no-cal
    python train_game_model.py --model rf      # force RandomForest instead of XGB
"""

from __future__ import annotations

import argparse
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
HISTORY_CSV    = HERE / "2025_model_data.csv"
MODEL_PATH     = HERE / "betting_model.pkl"            # production
CANDIDATE_PATH = HERE / "betting_model_candidate.pkl"  # default write target

# Identifier / label / leakage columns that must NOT be passed as features.
NON_FEATURE_COLS = {
    "game_pk", "game_date", "commence_time",
    "home_team", "away_team",
    "home_starter", "away_starter",
    "home_starter_id", "away_starter_id",
    "home_starter_throws", "away_starter_throws",
    # Outcomes — these would leak the answer
    "home_runs", "away_runs",
    "home_runs_allowed", "away_runs_allowed",
    "home_win", "actual_winner", "winner",
    # Pre-loaded market odds — keep OUT so the model doesn't learn the line
    "home_ml", "away_ml", "home_decimal", "away_decimal",
    "home_implied", "away_implied",
}


def _select_features(df: pd.DataFrame) -> list[str]:
    """Numeric columns that aren't in the exclude set."""
    feats = []
    for c in df.columns:
        if c in NON_FEATURE_COLS:
            continue
        if pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c]):
            feats.append(c)
    return feats


def _build_estimator(kind: str):
    if kind == "xgb":
        try:
            from xgboost import XGBClassifier
        except ImportError:
            print("xgboost not installed — falling back to RandomForest.")
            return _build_estimator("rf")
        return XGBClassifier(
            n_estimators=600,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.7,
            reg_alpha=0.5,
            reg_lambda=2.0,
            min_child_weight=10,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
    # Random forest fallback
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(
        n_estimators=600,
        max_depth=6,
        min_samples_leaf=20,
        max_features=0.5,
        n_jobs=-1,
        random_state=42,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--history", default=str(HISTORY_CSV),
                    help="CSV with home/away rolling features + home_win label.")
    ap.add_argument("--out", default=str(CANDIDATE_PATH),
                    help="Where to save the trained candidate. Default writes to "
                         "betting_model_candidate.pkl (NEVER overwrites prod by default).")
    ap.add_argument("--test-frac", type=float, default=0.20,
                    help="Chronological tail fraction held out for evaluation.")
    ap.add_argument("--model", choices=("xgb", "rf"), default="xgb")
    ap.add_argument("--no-cal", action="store_true",
                    help="Skip isotonic probability calibration.")
    ap.add_argument("--promote", action="store_true",
                    help="After training, back up the existing betting_model.pkl "
                         "to betting_model_backup_<UTCstamp>.pkl, then copy the "
                         "candidate to betting_model.pkl. Skipped if the candidate "
                         "underperforms the current prod model on the holdout.")
    ap.add_argument("--force-promote", action="store_true",
                    help="Promote even if the candidate underperforms (you've been warned).")
    args = ap.parse_args()

    p = Path(args.history)
    if not p.exists():
        print(f"⚠️  {p} not found"); return

    print(f"Loading {p.name} …")
    df = pd.read_csv(p, low_memory=False)
    if "home_win" not in df.columns:
        raise ValueError("history must contain a 'home_win' column")

    # Clean labels + sort chronologically
    df["game_date"] = pd.to_datetime(df.get("game_date"), errors="coerce")
    df = df[df["home_win"].notna()].copy()
    df["home_win"] = pd.to_numeric(df["home_win"], errors="coerce").fillna(0).astype(int)
    df = df.sort_values(["game_date"]).reset_index(drop=True)

    feats = _select_features(df)
    print(f"  rows: {len(df):,}   numeric features: {len(feats)}")
    if not feats:
        raise RuntimeError("no usable numeric feature columns found")

    # Chronological split
    n_test = max(1, int(len(df) * args.test_frac))
    n_train = len(df) - n_test
    train = df.iloc[:n_train]
    test  = df.iloc[n_train:]
    print(f"  train: {len(train):,}   test: {len(test):,}   "
          f"(test span: {test['game_date'].min().date() if test['game_date'].notna().any() else '—'} "
          f"→ {test['game_date'].max().date() if test['game_date'].notna().any() else '—'})")

    X_tr = train[feats].apply(pd.to_numeric, errors="coerce").fillna(0)
    y_tr = train["home_win"].astype(int)
    X_te = test[feats].apply(pd.to_numeric, errors="coerce").fillna(0)
    y_te = test["home_win"].astype(int)

    # ── Fit ────────────────────────────────────────────────────────────────
    print(f"\nTraining {args.model.upper()} classifier …")
    model = _build_estimator(args.model)
    model.fit(X_tr, y_tr)

    # ── Optional probability calibration (isotonic via CV) ─────────────────
    if not args.no_cal:
        from sklearn.calibration import CalibratedClassifierCV
        print("Calibrating probabilities (isotonic, cv=3) …")
        model = CalibratedClassifierCV(_build_estimator(args.model),
                                       method="isotonic", cv=3)
        model.fit(X_tr, y_tr)

    # ── Evaluate ───────────────────────────────────────────────────────────
    proba = model.predict_proba(X_te)[:, 1]
    pred  = (proba > 0.5).astype(int)

    acc   = accuracy_score(y_te, pred)
    ll    = log_loss(y_te, proba, labels=[0, 1])
    brier = brier_score_loss(y_te, proba)
    naive = (y_te == 1).mean()  # always-home accuracy
    print("\n── Holdout metrics ────────────────────────────────────")
    print(f"  accuracy        : {acc*100:6.2f}%   (always-home baseline: {naive*100:.2f}%)")
    print(f"  log loss        : {ll:6.4f}     (random=0.6931)")
    print(f"  Brier score     : {brier:6.4f}     (lower=better, 0.25=random)")
    edge_vs_baseline = (acc - max(naive, 1 - naive)) * 100
    print(f"  edge vs baseline: {edge_vs_baseline:+.2f}%  (positive = model beats naive)")

    # ── Compare against current production model on the same holdout ───────
    prod_acc = prod_ll = prod_brier = None
    if MODEL_PATH.exists():
        try:
            with open(MODEL_PATH, "rb") as f:
                prod = pickle.load(f)
            prod_model = prod.get("model") if isinstance(prod, dict) else prod
            prod_feats = (prod.get("features") if isinstance(prod, dict)
                          else list(getattr(prod_model, "feature_names_in_", feats)))
            X_te_prod = test.reindex(columns=prod_feats).apply(
                pd.to_numeric, errors="coerce").fillna(0)
            prod_proba = prod_model.predict_proba(X_te_prod)[:, 1]
            prod_pred  = (prod_proba > 0.5).astype(int)
            prod_acc   = accuracy_score(y_te, prod_pred)
            prod_ll    = log_loss(y_te, prod_proba, labels=[0, 1])
            prod_brier = brier_score_loss(y_te, prod_proba)
            print("\n── Production model on the SAME holdout ───────────────")
            print(f"  accuracy        : {prod_acc*100:6.2f}%")
            print(f"  log loss        : {prod_ll:6.4f}")
            print(f"  Brier score     : {prod_brier:6.4f}")
            print(f"\n  Δ vs prod  acc: {(acc - prod_acc)*100:+.2f}%   "
                  f"logloss: {prod_ll - ll:+.4f}   brier: {prod_brier - brier:+.4f}")
        except Exception as e:
            print(f"\n  (couldn't evaluate prod model for comparison: {e})")

    # Save in the bundle format daily_mlb_model_runner.load_model expects.
    out_path = Path(args.out)
    with open(out_path, "wb") as f:
        pickle.dump({"model": model, "features": feats}, f)
    print(f"\nWrote candidate → {out_path}  ({len(feats)} features baked in)")
    if not args.promote:
        print("  (production model UNCHANGED — pass --promote to replace it after a backup)")

    # ── Promote candidate → production with timestamped backup ─────────────
    if args.promote:
        import shutil, datetime as _dt
        # Safety gate: don't promote if it's worse and --force-promote wasn't given
        if (prod_acc is not None and not args.force_promote
                and (acc < prod_acc - 0.001 or brier > prod_brier + 0.002)):
            print("\n⚠️  Candidate underperforms production on holdout — NOT promoting.")
            print("   (use --force-promote to override)")
            return
        if MODEL_PATH.exists():
            stamp  = _dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
            backup = MODEL_PATH.with_name(f"betting_model_backup_{stamp}.pkl")
            shutil.copy2(MODEL_PATH, backup)
            print(f"\nBacked up prod → {backup.name}")
        shutil.copy2(out_path, MODEL_PATH)
        print(f"Promoted candidate → {MODEL_PATH.name}")
        print("  daily_update.py will pick up the new model on its next run.")


if __name__ == "__main__":
    main()
