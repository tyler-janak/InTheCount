"""
train_game_model.py
===================
Train / evaluate the game win-probability classifier (betting_model.pkl)
consumed by daily_mlb_model_runner.py.

Target  : home_win (1 = home team won, 0 = away team won), one row per game.
Data    : data/game_model_data.csv from build_game_features.py (Statcast,
          pre-game features only, two-season window, regular season).
Features: the 18 features of the production model, unchanged in name and
          meaning (see FEATURES).
Model   : XGBoost classifier with isotonic calibration (the production
          configuration); RandomForest and uncalibrated variants are
          candidates.

Protocol (chronological, no shuffling)
--------------------------------------
    train       : game_date <  --valid-start      (default: every season before the latest)
    validation  : --valid-start <= date < --test-start   -> picks the candidate (log loss)
    test        : date >= --test-start            -> touched once, after selection
The selected candidate is refit on train + validation before scoring the
test window. With --production the test window is dropped and the selected
candidate is refit on every completed game (what the daily runner uses).

Outputs
-------
    betting_model_candidate.pkl                              (always)
    betting_model.pkl                                        (--promote, with backup)
    outputs/model_evaluation/data/game_test_predictions.csv  (evaluation mode)
    outputs/model_evaluation/metrics/game_model_selection.csv

Usage
-----
    python train_game_model.py                                   # evaluation run
    python train_game_model.py --valid-start 2026-03-01 --test-start 2026-07-01
    python train_game_model.py --production --promote            # refit on everything, ship
"""

from __future__ import annotations

import argparse
import pickle
import shutil
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

from leakage_guard import assert_pregame_features
import model_stacking as stk

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
HISTORY_CSV = HERE / "data" / "game_model_data.csv"
MODEL_PATH = HERE / "betting_model.pkl"
CANDIDATE_PATH = HERE / "betting_model_candidate.pkl"
EVAL_DIR = HERE / "outputs" / "model_evaluation"

FEATURES = [
    "diff_runs_scored_10g", "diff_runs_allowed_10g", "diff_OPS_10g", "diff_K_rate_10g",
    "diff_BB_rate_10g", "diff_SP_FIP_10", "diff_SP_KBB_pct_10", "diff_SP_HR9_10",
    "diff_SP_FIP_season", "diff_SP_KBB_pct_season", "diff_BP_ERA_proxy_14g",
    "diff_BP_WHIP_14g", "diff_BP_KBB_pct_14g", "diff_team_winpct", "diff_team_run_diff",
    "diff_season_OPS", "home_field", "diff_starter_hand",
]
ID_COLS = ["game_date", "game_pk", "home_team", "away_team", "home_starter", "away_starter",
           "prediction_season", "history_seasons"]
FEATURE_BUILDER = "statcast_v2"
SELECT_METRIC = "logloss"  # game-model selection, optimized stack weights and the v1 promotion gate
                           # (log loss: the site shows probabilities, so calibration matters; AUC was
                           #  considered and rejected - see the technical report, section 7)


def _xgb():
    from xgboost import XGBClassifier
    return XGBClassifier(n_estimators=600, max_depth=4, learning_rate=0.03, subsample=0.8,
                         colsample_bytree=0.7, reg_alpha=0.5, reg_lambda=2.0, min_child_weight=10,
                         objective="binary:logistic", eval_metric="logloss", random_state=42,
                         n_jobs=-1, verbosity=0)


def _rf():
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(n_estimators=600, max_depth=6, min_samples_leaf=20,
                                  max_features=0.5, n_jobs=-1, random_state=42)


def _calibrated(builder):
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import TimeSeriesSplit
    # Time-ordered folds: the isotonic map is always fit on games that come
    # AFTER the games its base model was trained on (the previous cv=3 used
    # unshuffled K-fold, i.e. later games predicting earlier ones).
    return CalibratedClassifierCV(builder(), method="isotonic", cv=TimeSeriesSplit(n_splits=3))


LOGIT_C = 0.001   # validation log loss 0.6893 at C=0.001 vs 0.6905 at C=0.1 (audit learning curve, training -> validation only)


def _logit():
    """Logistic regression on the same 18 standardised features (strong L2, C=LOGIT_C:
    the features are home-minus-away differences that overlap heavily, and a single
    game carries little signal, so heavy shrinkage toward the base rate wins on validation)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    return Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(C=LOGIT_C, max_iter=2000))])


CANDIDATES = {
    "xgb_isotonic": lambda: _calibrated(_xgb),    # production configuration
    # "xgb_raw" removed: uncalibrated XGBoost over-fits the training games (AUC 0.89 on training
    # vs 0.51 on validation) and had the worst validation log loss; the isotonic version stays.
    "rf_isotonic": lambda: _calibrated(_rf),
    "rf_raw": _rf,
    "logistic": _logit,
}


def _win_pct_logit():
    """Simple baseline: logistic regression on season win% difference + intercept
    (the intercept carries home-field advantage)."""
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(C=1.0)


def prob_metrics(y, p) -> dict:
    y = np.asarray(y, int); p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    out = {"n": int(len(y)), "accuracy": float(accuracy_score(y, (p > 0.5).astype(int))),
           "log_loss": float(log_loss(y, p, labels=[0, 1])), "brier": float(brier_score_loss(y, p))}
    out["roc_auc"] = float(roc_auc_score(y, p)) if len(set(y)) == 2 else float("nan")
    return out


def load_table(path=HISTORY_CSV, regular_season_only=True) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    df = df[df["home_win"].notna()].copy()
    if regular_season_only and "game_type" in df.columns:
        df = df[df["game_type"].astype(str) == "R"]
    df["home_win"] = df["home_win"].astype(int)
    return df.sort_values(["game_date", "game_pk"]).reset_index(drop=True)


def v1_live_metrics(games: pd.DataFrame, log_path=HERE / "2026_picks_accuracy.csv") -> dict | None:
    if not Path(log_path).exists() or games.empty:
        return None
    log = pd.read_csv(log_path)
    if "model_version" in log.columns:
        log = log[log["model_version"].fillna("v1").astype(str).str.startswith("v1")]
    log = log[log["correct"].notna() & log["game_pk"].notna()].drop_duplicates("game_pk", keep="last")
    j = games[["game_pk", "home_win", "home_team"]].merge(log[["game_pk", "home_win_prob", "home_team"]],
                                                          on="game_pk", suffixes=("", "_log"))
    j = j[j["home_team"] == j["home_team_log"]]
    return prob_metrics(j["home_win"], j["home_win_prob"]) if len(j) >= 100 else None


def X_of(df):
    return df.reindex(columns=FEATURES).apply(pd.to_numeric, errors="coerce").fillna(0)


def run(valid_start=None, test_start=None, production=False, history=HISTORY_CSV,
        out=CANDIDATE_PATH, promote=False, force_promote=False, write_eval=True) -> dict:
    assert_pregame_features(FEATURES, "game model")
    df = load_table(history)
    seasons = sorted(df["game_date"].dt.year.unique())
    latest = seasons[-1]
    valid_start = pd.Timestamp(valid_start or f"{latest}-03-01")
    if production:
        test_start = None
    else:
        test_start = pd.Timestamp(test_start or f"{latest}-07-01")
    train = df[df["game_date"] < valid_start]
    if test_start is not None:
        valid = df[(df["game_date"] >= valid_start) & (df["game_date"] < test_start)]
        test = df[df["game_date"] >= test_start]
    else:
        valid = df[df["game_date"] >= valid_start]
        test = df.iloc[0:0]
    print(f"rows: train {len(train):,} ({train.game_date.min().date()} -> {train.game_date.max().date()})  "
          f"valid {len(valid):,} ({valid.game_date.min().date()} -> {valid.game_date.max().date()})  "
          f"test {len(test):,}" + (f" ({test.game_date.min().date()} -> {test.game_date.max().date()})" if len(test) else ""))

    # ---- 1. candidate selection on validation ---------------------------
    sel_rows, valid_p = [], {}
    yv = valid["home_win"].to_numpy(float)
    for name, build in CANDIDATES.items():
        m = build()
        m.fit(X_of(train), train["home_win"])
        pv = m.predict_proba(X_of(valid))[:, 1]
        valid_p[name] = pv
        r = {"candidate": name, **{f"valid_{k}": v for k, v in prob_metrics(valid["home_win"], pv).items()}}
        sel_rows.append(r)
        print(f"  {name:<14} valid logloss={r['valid_log_loss']:.4f}  brier={r['valid_brier']:.4f}  "
              f"acc={r['valid_accuracy']:.3f}  auc={r['valid_roc_auc']:.3f}")

    # Stacking: even and optimised weights over the candidates' validation
    # probabilities. The optimised stack is scored with cross-fitted
    # predictions (weights never graded on the games they were fit on).
    base = list(CANDIDATES)
    P_va = np.column_stack([valid_p[n] for n in base])
    order = np.argsort(valid["game_date"].to_numpy(), kind="mergesort")    # chronological for cross-fitting
    stacks = {"stack_even": stk.even_weights(len(base)), "stack_opt": stk.fit_weights(P_va, yv, SELECT_METRIC)}
    for name, wts in stacks.items():
        if name == "stack_opt":
            cf = np.empty(len(yv)); cf[order] = stk.crossfit_predictions(P_va[order], yv[order], SELECT_METRIC)
            score = cf
        else:
            score = P_va @ wts
        r = {"candidate": name, "stack_weights": stk.describe(base, wts),
             **{f"valid_{k}": v for k, v in prob_metrics(valid["home_win"], np.clip(score, 1e-6, 1 - 1e-6)).items()}}
        sel_rows.append(r)
        print(f"  {name:<14} valid logloss={r['valid_log_loss']:.4f}  brier={r['valid_brier']:.4f}  "
              f"acc={r['valid_accuracy']:.3f}  auc={r['valid_roc_auc']:.3f}  [{r['stack_weights']}]")
    sel = pd.DataFrame(sel_rows)
    # Model selection: lowest validation log loss (a proper scoring rule for the
    # probabilities the site displays). AUC is reported, not used to choose.
    chosen = sel.sort_values("valid_log_loss").iloc[0]["candidate"]
    sel["selected"] = sel["candidate"] == chosen
    print(f"  -> selected on validation log loss: {chosen}")

    # Promotion gate: the deployed v1 model's LIVE picks on the same validation
    # games (genuinely out-of-sample - v1 uses no current-season data).
    v1 = v1_live_metrics(valid)
    if v1:
        sel = pd.concat([sel, pd.DataFrame([{"candidate": "v1_production_live_log",
                                             **{f"valid_{k}": v for k, v in v1.items()},
                                             "selected": False}])], ignore_index=True)
        print(f"  v1 production live log on validation: auc={v1['roc_auc']:.4f} logloss={v1['log_loss']:.4f} brier={v1['brier']:.4f} (n={v1['n']})")
    best_auc = float(sel.loc[sel["candidate"] == chosen, "valid_roc_auc"].iloc[0])
    best_ll = float(sel.loc[sel["candidate"] == chosen, "valid_log_loss"].iloc[0])
    beats_v1 = (not v1) or best_ll < v1["log_loss"]

    # ---- 2. refit on train + validation; score test once ----------------
    full = pd.concat([train, valid])
    full_models = {}

    def _full(name):                                     # base candidate refit on train + validation
        if name not in full_models:
            full_models[name] = CANDIDATES[name]().fit(X_of(full), full["home_win"])
        return full_models[name]

    if chosen in stacks:
        final = stk.StackedClassifier([_full(n) for n in base], stacks[chosen], base)
    else:
        final = _full(chosen)

    result = {"chosen": chosen, "selection": sel}
    if len(test) and write_eval:
        preds = test[[c for c in ID_COLS if c in test.columns]].copy()
        preds["actual_home_win"] = test["home_win"].to_numpy()
        for name in base:                                 # comparison only - never used to choose
            preds[f"p_{name}"] = _full(name).predict_proba(X_of(test))[:, 1]
        P_te = preds[[f"p_{n}" for n in base]].to_numpy(float)
        for name, wts in stacks.items():
            preds[f"p_{name}"] = np.clip(P_te @ wts, 0, 1)
        # baselines, fit on the same train + validation games
        home_rate = float(full["home_win"].mean())
        preds["p_baseline_home_rate"] = home_rate
        preds["p_baseline_coin_flip"] = 0.5
        bl = _win_pct_logit().fit(full[["diff_team_winpct"]].fillna(0), full["home_win"])
        preds["p_baseline_winpct_logit"] = bl.predict_proba(test[["diff_team_winpct"]].fillna(0))[:, 1]
        preds["selected_model"] = chosen
        preds["p_selected"] = preds[f"p_{chosen}"]
        preds["predicted_home_win"] = (preds["p_selected"] > 0.5).astype(int)
        preds["predicted_winner"] = np.where(preds["predicted_home_win"] == 1, preds["home_team"], preds["away_team"])
        preds["actual_winner"] = np.where(preds["actual_home_win"] == 1, preds["home_team"], preds["away_team"])
        preds["correct"] = (preds["predicted_home_win"] == preds["actual_home_win"]).astype(int)
        (EVAL_DIR / "data").mkdir(parents=True, exist_ok=True)
        (EVAL_DIR / "metrics").mkdir(parents=True, exist_ok=True)
        preds.to_csv(EVAL_DIR / "data" / "game_test_predictions.csv", index=False)
        sel.to_csv(EVAL_DIR / "metrics" / "game_model_selection.csv", index=False)
        m = prob_metrics(preds["actual_home_win"], preds["p_selected"])
        print(f"\n  TEST ({chosen}): n={m['n']}  acc={m['accuracy']:.4f}  logloss={m['log_loss']:.4f}  "
              f"brier={m['brier']:.4f}  auc={m['roc_auc']:.4f}")
        result["test_predictions"] = preds

    bundle = {"model": final, "features": FEATURES, "feature_builder": FEATURE_BUILDER,
              "candidate": chosen, "stack_weights": stk.describe(base, stacks[chosen]) if chosen in stacks else None,
              "trained_through": str(full["game_date"].max().date()),
              "train_window": [str(full["game_date"].min().date()), str(full["game_date"].max().date())],
              "selected_on": f"validation log loss {valid['game_date'].min().date()} -> {valid['game_date'].max().date()}",
              "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    with open(out, "wb") as f:
        pickle.dump(bundle, f)
    print(f"\nWrote candidate -> {out}")
    if promote and not beats_v1 and not force_promote:
        print(f"  NOT promoting: validation log loss {best_ll:.4f} does not beat the deployed v1 model's "
              f"live log ({v1['log_loss']:.4f}) on the same games. (--force-promote overrides.)")
        promote = False
    result["beats_v1_on_validation"] = beats_v1
    if promote:
        if test_start is not None and not force_promote:
            print("  (not promoting an evaluation-mode model: it was not refit on the test window. "
                  "Use --production --promote.)")
        else:
            if MODEL_PATH.exists():
                stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                backup = MODEL_PATH.with_name(f"betting_model_backup_{stamp}.pkl")
                shutil.copy2(MODEL_PATH, backup)
                print(f"  backed up production -> {backup.name}")
            shutil.copy2(out, MODEL_PATH)
            print(f"  promoted -> {MODEL_PATH.name}")
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--history", default=str(HISTORY_CSV))
    ap.add_argument("--valid-start", default=None)
    ap.add_argument("--test-start", default=None)
    ap.add_argument("--production", action="store_true",
                    help="no test window: select on validation, refit on all completed games")
    ap.add_argument("--out", default=str(CANDIDATE_PATH))
    ap.add_argument("--promote", action="store_true")
    ap.add_argument("--force-promote", action="store_true")
    args = ap.parse_args()
    run(args.valid_start, args.test_start, args.production, args.history, args.out,
        args.promote, args.force_promote)


if __name__ == "__main__":
    main()
