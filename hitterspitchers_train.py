"""
hitterspitchers_train.py
========================
Train pitcher and hitter models from the built game-level tables.
"""

import argparse
import pickle
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor

from model_calibration import (CalibratedRegressor, fit_linear_calibration,
                                fit_isotonic_calibration)

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    warnings.warn("xgboost not installed - skipping XGBoost models. pip install xgboost")

warnings.filterwarnings("ignore")


LOGIT_EPS = 1e-6


def _logit_transform(y):
    arr = np.asarray(y, dtype=float)
    arr = np.clip(arr, LOGIT_EPS, 1.0 - LOGIT_EPS)
    return np.log(arr / (1.0 - arr))


def _inv_logit_transform(z):
    arr = np.asarray(z, dtype=float)
    return 1.0 / (1.0 + np.exp(-arr))


def clean_hitter_training_rows(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    for c in ["PA", "H", "HR", "BB", "K", "2B", "3B", "SB",
              "h_rate", "hr_rate", "bb_rate", "k_rate"]:
        if c in work.columns:
            work[c] = pd.to_numeric(work[c], errors="coerce")

    # PA sanity: 1-7 per game is realistic
    if "PA" in work.columns:
        work = work[work["PA"].between(1, 7, inclusive="both") | work["PA"].isna()].copy()

    # Count sanity: non-negative, at most as large as PA
    if {"H", "PA"}.issubset(work.columns):
        work = work[(work["H"] <= work["PA"]) | work["H"].isna() | work["PA"].isna()].copy()
        work = work[(work["H"] >= 0) | work["H"].isna()].copy()
    if {"HR", "H"}.issubset(work.columns):
        work = work[(work["HR"] <= work["H"]) | work["HR"].isna() | work["H"].isna()].copy()
        work = work[(work["HR"] >= 0) | work["HR"].isna()].copy()
    # 2B/3B sanity: non-negative, and no single game can have more doubles (or
    # triples, or extra-base hits overall) than total hits.
    for xbh_col in ["2B", "3B"]:
        if xbh_col in work.columns:
            work = work[(work[xbh_col] >= 0) | work[xbh_col].isna()].copy()
            if "H" in work.columns:
                work = work[(work[xbh_col] <= work["H"]) | work[xbh_col].isna() | work["H"].isna()].copy()
    if {"2B", "3B", "HR", "H"}.issubset(work.columns):
        xbh_total = (pd.to_numeric(work["2B"], errors="coerce").fillna(0)
                     + pd.to_numeric(work["3B"], errors="coerce").fillna(0)
                     + pd.to_numeric(work["HR"], errors="coerce").fillna(0))
        work = work[(xbh_total <= work["H"]) | work["H"].isna()].copy()
    if "SB" in work.columns:
        work = work[(work["SB"] >= 0) | work["SB"].isna()].copy()
    for c in ["BB", "K"]:
        if c in work.columns and "PA" in work.columns:
            work = work[(work[c] <= work["PA"]) | work[c].isna() | work["PA"].isna()].copy()
            work = work[(work[c] >= 0) | work[c].isna()].copy()

    # Rate sanity (kept for feature columns that are still present)
    for c in ["h_rate", "hr_rate", "bb_rate", "k_rate"]:
        if c in work.columns:
            work = work[work[c].between(0, 1, inclusive="both") | work[c].isna()].copy()

    if {"hr_rate", "h_rate"}.issubset(work.columns):
        work = work[(work["hr_rate"] <= work["h_rate"]) | work["hr_rate"].isna() | work["h_rate"].isna()].copy()

    if {"h_rate", "bb_rate"}.issubset(work.columns):
        on_base = pd.to_numeric(work["h_rate"], errors="coerce") + pd.to_numeric(work["bb_rate"], errors="coerce")
        work = work[(on_base <= 0.75) | on_base.isna()].copy()

    return work


def report_hitter_feature_coverage(df: pd.DataFrame):
    cols = ["opp_sp_k_rate", "opp_sp_bb_rate", "opp_sp_hr_rate", "opp_sp_h_rate"]
    present = [c for c in cols if c in df.columns]
    if not present:
        return
    coverage = float(df[present].notna().all(axis=1).mean()) if len(df) else float("nan")
    print(f"  Hitter rows with opponent-starter core features: {coverage:.1%}")




# Predict counting stats directly - reduces compounding variance vs rate->count
# "R" = runs allowed per game (proxy for earned runs - Statcast has no ER/
# unearned split; see the comment on the R aggregation in
# hitterspitchers_data.py). Training R directly is what lets FIP/WAR be
# composed from genuine model outputs at scoring time instead of the old
# hand-tuned estimate_pitcher_runs() formula.
PITCHER_TARGETS = ["K", "BB", "HR", "H", "IP", "R"]
# "2B"/"3B" let wOBA be composed from real modeled extra-base-hit splits
# instead of only H/TB (TB can't distinguish "4 singles" from "1 double +
# 2 singles"). "SB" is a new stolen-base target - see the SB feature-
# engineering note in hitterspitchers_data.py (is_sb/is_cs). DRS was
# explicitly dropped (no free/scrapeable historical dataset exists) and OAA
# is NOT here - it needs fielding-chance data this per-batting-event
# pipeline doesn't have; the real Statcast OAA leaderboard fetch already
# used by player_rankings.py remains the source for OAA.
HITTER_TARGETS  = ["H", "HR", "BB", "K", "PA", "TB", "2B", "3B", "SB"]

# RATE TARGETS - trained alongside counts so hitterspitchers_today.py can
# ensemble them at scoring time:  final_K = 0.5 * direct_K + 0.5 * (K_per_9 * IP/9)
# The rate side leverages the fact that IP / PA predictions are already at
# book benchmark; the gap is per-opportunity efficiency, which a rate model
# learns more cleanly than a count model trying to fit volume + rate at once.
PITCHER_RATE_TARGETS = ["K_per_9", "BB_per_9", "H_per_9", "HR_per_9", "R_per_9"]
HITTER_RATE_TARGETS  = ["H_per_PA", "HR_per_PA", "BB_per_PA", "K_per_PA", "TB_per_PA",
                        "2B_per_PA", "3B_per_PA", "SB_per_PA"]

# Rate-target sanity caps (per-game). 1 IP × 5 K -> K9=45 destroys training.
# R_per_9 gets a slightly higher ceiling than the other rates - a short,
# disastrous outing (e.g. 1 IP, 6 R) is rarer than a hot-K game but still
# real, and runs allowed has no natural cap the way K/BB/H do per batter faced.
PITCHER_RATE_CAP = {"K_per_9": 27.0, "BB_per_9": 18.0,
                    "H_per_9": 27.0, "HR_per_9":  9.0,
                    "R_per_9": 30.0}

# Count targets - non-negative integers. XGB uses Poisson loss on these
# (correct loss for count data, especially sparse ones like HR where ~85% of
# rows are zero). Random forest still gets the log1p wrapper.
COUNT_TARGETS = {"H", "HR", "BB", "K",        # hitter counts
                 "PA",                         # plate appearances
                 "IP",                         # innings pitched
                 "TB",                         # total bases
                 "2B", "3B",                   # extra-base-hit splits (wOBA)
                 "SB",                         # stolen bases
                 "R",                          # runs allowed (ER proxy)
                 "BF", "outs", "pitches"}      # pitcher misc counts


def _attach_pitcher_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Add K_per_9, BB_per_9, H_per_9, HR_per_9 columns from raw counts + IP."""
    out = df.copy()
    ip = pd.to_numeric(out.get("IP"), errors="coerce").clip(lower=0.1)
    for raw, rate in [("K", "K_per_9"), ("BB", "BB_per_9"),
                      ("H", "H_per_9"), ("HR", "HR_per_9"),
                      ("R", "R_per_9")]:
        if raw in out.columns:
            v = pd.to_numeric(out[raw], errors="coerce") / ip * 9.0
            out[rate] = v.clip(lower=0.0, upper=PITCHER_RATE_CAP[rate])
    return out


def _attach_hitter_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Add H_per_PA, HR_per_PA, BB_per_PA, K_per_PA, TB_per_PA columns from
    raw counts + PA. TB rate clips up to 4 (max bases per PA = HR)."""
    out = df.copy()
    pa = pd.to_numeric(out.get("PA"), errors="coerce").clip(lower=1.0)
    for raw, rate, cap in [("H", "H_per_PA", 1.0), ("HR", "HR_per_PA", 1.0),
                            ("BB", "BB_per_PA", 1.0), ("K", "K_per_PA", 1.0),
                            ("TB", "TB_per_PA", 4.0),
                            ("2B", "2B_per_PA", 1.0), ("3B", "3B_per_PA", 1.0),
                            # SB isn't naturally "per PA" (steals happen per
                            # times-on-base opportunity, not per plate
                            # appearance) but per-PA keeps this consistent
                            # with every other rate target's architecture -
                            # the cap just needs to be generous since SB/PA
                            # is normally tiny (<0.05).
                            ("SB", "SB_per_PA", 1.0)]:
        if raw in out.columns:
            v = pd.to_numeric(out[raw], errors="coerce") / pa
            out[rate] = v.clip(lower=0.0, upper=cap)
    return out

PITCHER_FEATURES = [
    # -- season-to-date rates (stable baselines) --------------------------
    "K_rate_std", "BB_rate_std", "HR_rate_std", "H_rate_std",
    "IP_std", "avg_velocity_std", "avg_spin_std",

    # -- rate rolling windows ----------------------------------------------
    "K_rate_last5",  "BB_rate_last5",  "HR_rate_last5",  "H_rate_last5",  "IP_last5",
    "K_rate_last7",  "BB_rate_last7",  "HR_rate_last7",  "H_rate_last7",  "IP_last7",
    "K_rate_last10", "BB_rate_last10", "HR_rate_last10", "H_rate_last10", "IP_last10",
    "K_rate_last14", "BB_rate_last14", "HR_rate_last14", "H_rate_last14", "IP_last14",
    "K_rate_last21", "BB_rate_last21", "HR_rate_last21", "H_rate_last21", "IP_last21",
    "K_rate_last30", "BB_rate_last30", "HR_rate_last30", "H_rate_last30", "IP_last30",
    "avg_velocity_last5", "avg_velocity_last7", "avg_velocity_last10",

    # -- RAW COUNT rolling windows (direct target history) -----------------
    "K_std",   "K_last5",  "K_last7",  "K_last10", "K_last14", "K_last21", "K_last30",
    "BB_std",  "BB_last5", "BB_last7", "BB_last10","BB_last14","BB_last21","BB_last30",
    "HR_std",  "HR_last5", "HR_last7", "HR_last10","HR_last14","HR_last21","HR_last30",
    "H_std",   "H_last5",  "H_last7",  "H_last10", "H_last14", "H_last21", "H_last30",
    "R_std",   "R_last5",  "R_last7",  "R_last10", "R_last14", "R_last21", "R_last30",

    # -- workload / usage --------------------------------------------------
    "BF_std", "outs_std", "pitches_std",
    "BF_last3", "outs_last3", "pitches_last3",
    "BF_last5", "outs_last5", "pitches_last5",
    "BF_last7", "outs_last7", "pitches_last7",
    "BF_last10", "outs_last10", "pitches_last10",
    "BF_last14", "outs_last14", "pitches_last14",
    "BF_last21", "outs_last21", "pitches_last21",
    "BF_last30", "outs_last30", "pitches_last30",

    "BF_per_IP_std", "pitches_per_BF_std", "pitches_per_IP_std",
    "BF_per_IP_last5", "pitches_per_BF_last5", "pitches_per_IP_last5",
    "BF_per_IP_last10", "pitches_per_BF_last10", "pitches_per_IP_last10",

    "max_ip_last5", "max_ip_last10",
    "days_rest",
    "starter_pct_last5",
    "starter_pct_last10",

    # -- platoon splits ----------------------------------------------------
    # NO-WINDOW platoon splits removed - they hold the CURRENT game's per-hand
    # rate (leakage; see diagnose_leakage.py - corr ~0.5-0.6 with the target,
    # and they're overwritten with trailing values at serving so the model
    # falls apart live). Only the trailing _last5/_last10/_std windows are kept.
    "pitcher_k_rate_vs_hand_last5_R",  "pitcher_bb_rate_vs_hand_last5_R",
    "pitcher_hr_rate_vs_hand_last5_R", "pitcher_h_rate_vs_hand_last5_R",
    "pitcher_k_rate_vs_hand_last5_L",  "pitcher_bb_rate_vs_hand_last5_L",
    "pitcher_hr_rate_vs_hand_last5_L", "pitcher_h_rate_vs_hand_last5_L",

    "pitcher_k_rate_vs_hand_last10_R",  "pitcher_bb_rate_vs_hand_last10_R",
    "pitcher_hr_rate_vs_hand_last10_R", "pitcher_h_rate_vs_hand_last10_R",
    "pitcher_k_rate_vs_hand_last10_L",  "pitcher_bb_rate_vs_hand_last10_L",
    "pitcher_hr_rate_vs_hand_last10_L", "pitcher_h_rate_vs_hand_last10_L",

    "pitcher_k_rate_vs_hand_std_R",  "pitcher_bb_rate_vs_hand_std_R",
    "pitcher_hr_rate_vs_hand_std_R", "pitcher_h_rate_vs_hand_std_R",
    "pitcher_k_rate_vs_hand_std_L",  "pitcher_bb_rate_vs_hand_std_L",
    "pitcher_hr_rate_vs_hand_std_L", "pitcher_h_rate_vs_hand_std_L",

    # -- opponent team context ---------------------------------------------
    # NO-WINDOW team context removed - it's the CURRENT game's realized
    # opponent rate (leakage; corr ~0.7-0.8 with the target). Trailing kept.
    "team_k_rate_vs_hand_last5",  "team_bb_rate_vs_hand_last5",
    "team_hr_rate_vs_hand_last5", "team_h_rate_vs_hand_last5",
    "team_k_rate_vs_hand_last7",  "team_bb_rate_vs_hand_last7",
    "team_hr_rate_vs_hand_last7", "team_h_rate_vs_hand_last7",
    "team_k_rate_vs_hand_last10", "team_bb_rate_vs_hand_last10",
    "team_hr_rate_vs_hand_last10","team_h_rate_vs_hand_last10",
    "team_k_rate_vs_hand_last14", "team_bb_rate_vs_hand_last14",
    "team_hr_rate_vs_hand_last14","team_h_rate_vs_hand_last14",
    "team_k_rate_vs_hand_std",    "team_bb_rate_vs_hand_std",
    "team_hr_rate_vs_hand_std",   "team_h_rate_vs_hand_std",

    # -- true-talent (empirical-Bayes shrunk) + log5 lineup matchup --
    # Honest, leakage-free signal from enrich_truetalent.py: the pitcher's
    # sample-size-regressed rates, the opposing lineup's shrunk rates, and the
    # two combined via the log5 odds-ratio. This is the legitimate version of
    # the opponent signal the leaked team_*_rate_vs_hand columns were faking.
    "p_tt_k", "p_tt_bb", "p_tt_h", "p_tt_hr",
    "lineup_tt_k", "lineup_tt_bb", "lineup_tt_h", "lineup_tt_hr",
    "matchup_k", "matchup_bb", "matchup_h", "matchup_hr",

    "park_factor",
]

HITTER_FEATURES = [
    # -- lineup: today's batting-order slot (posted pre-game) + recent slots --
    "lineup_spot", "lineup_spot_last10",

    # -- season-to-date rates (stable baselines) --------------------------
    "h_rate_std", "hr_rate_std", "bb_rate_std", "k_rate_std",
    "PA_std", "avg_EV_std", "max_EV_std", "avg_LA_std", "avg_direction_std",

    # -- rate rolling windows ----------------------------------------------
    "h_rate_last5",  "hr_rate_last5",  "bb_rate_last5",  "k_rate_last5",  "PA_last5",
    "h_rate_last7",  "hr_rate_last7",  "bb_rate_last7",  "k_rate_last7",  "PA_last7",
    "h_rate_last10", "hr_rate_last10", "bb_rate_last10", "k_rate_last10", "PA_last10",
    "h_rate_last14", "hr_rate_last14", "bb_rate_last14", "k_rate_last14", "PA_last14",
    "h_rate_last21", "hr_rate_last21", "bb_rate_last21", "k_rate_last21", "PA_last21",
    "h_rate_last30", "hr_rate_last30", "bb_rate_last30", "k_rate_last30", "PA_last30",
    "avg_EV_last5", "max_EV_last5", "avg_LA_last5",
    "avg_EV_last10", "max_EV_last10", "avg_LA_last10",

    # -- RAW COUNT rolling windows (direct target history) -----------------
    "H_std",  "H_last5",  "H_last7",  "H_last10", "H_last14", "H_last21", "H_last30",
    "HR_std", "HR_last5", "HR_last7", "HR_last10","HR_last14","HR_last21","HR_last30",
    "BB_std", "BB_last5", "BB_last7", "BB_last10","BB_last14","BB_last21","BB_last30",
    "K_std",  "K_last5",  "K_last7",  "K_last10", "K_last14", "K_last21", "K_last30",
    "2B_std", "2B_last5", "2B_last7", "2B_last10","2B_last14","2B_last21","2B_last30",
    "3B_std", "3B_last5", "3B_last7", "3B_last10","3B_last14","3B_last21","3B_last30",
    "SB_std", "SB_last5", "SB_last7", "SB_last10","SB_last14","SB_last21","SB_last30",

    # -- PA convenience features -------------------------------------------
    "PA_last3",
    "max_hr_rate_last10", "max_h_rate_last10",
    "days_since_game",

    # -- batted-ball quality -----------------------------------------------
    "barrel_proxy_std", "hard_hit_proxy_std", "sweet_spot_proxy_std", "blast_proxy_std",
    "ev_la_interaction_std", "ev_spread_std",
    "times_on_base_rate_std", "xbh_proxy_rate_std",

    "barrel_proxy_last5", "hard_hit_proxy_last5", "sweet_spot_proxy_last5", "blast_proxy_last5",
    "ev_la_interaction_last5", "ev_spread_last5",
    "times_on_base_rate_last5", "xbh_proxy_rate_last5",

    "barrel_proxy_last10", "hard_hit_proxy_last10", "sweet_spot_proxy_last10", "blast_proxy_last10",
    "ev_la_interaction_last10", "ev_spread_last10",
    "times_on_base_rate_last10", "xbh_proxy_rate_last10",

    # -- platoon splits ----------------------------------------------------
    # NO-WINDOW platoon splits removed - current game's per-hand rate
    # (leakage, same pattern as the pitcher side). Trailing windows kept.
    "hitter_h_rate_vs_hand_last5_R",  "hitter_hr_rate_vs_hand_last5_R",
    "hitter_bb_rate_vs_hand_last5_R", "hitter_k_rate_vs_hand_last5_R",
    "hitter_h_rate_vs_hand_last5_L",  "hitter_hr_rate_vs_hand_last5_L",
    "hitter_bb_rate_vs_hand_last5_L", "hitter_k_rate_vs_hand_last5_L",

    "hitter_h_rate_vs_hand_last10_R",  "hitter_hr_rate_vs_hand_last10_R",
    "hitter_bb_rate_vs_hand_last10_R", "hitter_k_rate_vs_hand_last10_R",
    "hitter_h_rate_vs_hand_last10_L",  "hitter_hr_rate_vs_hand_last10_L",
    "hitter_bb_rate_vs_hand_last10_L", "hitter_k_rate_vs_hand_last10_L",

    "hitter_h_rate_vs_hand_std_R",  "hitter_hr_rate_vs_hand_std_R",
    "hitter_bb_rate_vs_hand_std_R", "hitter_k_rate_vs_hand_std_R",
    "hitter_h_rate_vs_hand_std_L",  "hitter_hr_rate_vs_hand_std_L",
    "hitter_bb_rate_vs_hand_std_L", "hitter_k_rate_vs_hand_std_L",

    # -- opponent pitching team context ------------------------------------
    # NO-WINDOW opponent-pitching context removed - current game's realized
    # rate (leakage). Trailing versions kept.
    "team_allowed_k_rate_vs_hand_last5",  "team_allowed_bb_rate_vs_hand_last5",
    "team_allowed_hr_rate_vs_hand_last5", "team_allowed_h_rate_vs_hand_last5",
    "team_allowed_k_rate_vs_hand_last7",  "team_allowed_bb_rate_vs_hand_last7",
    "team_allowed_hr_rate_vs_hand_last7", "team_allowed_h_rate_vs_hand_last7",
    "team_allowed_k_rate_vs_hand_last10", "team_allowed_bb_rate_vs_hand_last10",
    "team_allowed_hr_rate_vs_hand_last10","team_allowed_h_rate_vs_hand_last10",
    "team_allowed_k_rate_vs_hand_last14", "team_allowed_bb_rate_vs_hand_last14",
    "team_allowed_hr_rate_vs_hand_last14","team_allowed_h_rate_vs_hand_last14",
    "team_allowed_k_rate_vs_hand_std",    "team_allowed_bb_rate_vs_hand_std",
    "team_allowed_hr_rate_vs_hand_std",   "team_allowed_h_rate_vs_hand_std",

    # -- opposing starter --------------------------------------------------
    # AUDIT FIX: these five used to be the opposing starter's results in THIS
    # game (same game_pk merge) - direct leakage into every hitter target.
    # hitterspitchers_data.enrich_hitter_with_opp_starter now fills them with
    # the starter's PREVIOUS start, which is exactly what serving uses.
    "opp_sp_k_rate", "opp_sp_bb_rate", "opp_sp_hr_rate", "opp_sp_h_rate", "opp_sp_ip",
    "opp_sp_k_rate_last5",  "opp_sp_bb_rate_last5",  "opp_sp_hr_rate_last5",
    "opp_sp_h_rate_last5",  "opp_sp_ip_last5",
    "opp_sp_k_rate_last10", "opp_sp_bb_rate_last10", "opp_sp_hr_rate_last10",
    "opp_sp_h_rate_last10", "opp_sp_ip_last10",
    "opp_sp_k_rate_std",    "opp_sp_bb_rate_std",    "opp_sp_hr_rate_std",
    "opp_sp_h_rate_std",    "opp_sp_ip_std",

    # -- hitter true-talent (empirical-Bayes shrunk per-PA rates) --
    "h_tt_k", "h_tt_bb", "h_tt_h", "h_tt_hr",

    "park_factor",
]

TRAIN_FRAC = 0.75
CV_FOLDS = 5
RANDOM_STATE = 42

# Defaults for the new tuning + calibration behaviour. Both can be toggled
# from the CLI (--no-tune / --no-calibrate / --tune-iter).
TUNE_ITER_DEFAULT = 24      # randomized-search samples per (target, xgb)
CALIB_FRAC = 0.15           # tail of the train window held out to fit a + b

# Randomized-search space for XGBoost. Centred on the regularised defaults the
# project already uses (shallow trees, strong reg) - with ~1-5k pitcher rows
# and ~45k hitter rows, deep/under-regularised trees overfit hot/cold streaks,
# so the grid deliberately keeps depth low and reg high.
XGB_TUNE_GRID = {
    "n_estimators":    [300, 400, 600, 800],
    "max_depth":       [2, 3, 4, 5],
    "learning_rate":   [0.02, 0.03, 0.04, 0.05],
    "subsample":       [0.7, 0.8, 0.9],
    "colsample_bytree":[0.6, 0.7, 0.8],
    "reg_alpha":       [0.0, 0.5, 1.0, 2.0],
    "reg_lambda":      [1.0, 2.0, 3.0, 5.0],
    "min_child_weight":[5, 10, 20, 30],
}


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _tune_xgb(X_train, y_train, target: str, n_iter: int = TUNE_ITER_DEFAULT,
              seed: int = RANDOM_STATE) -> dict:
    """Randomized hyperparameter search for an XGB model on one target.

    Uses a CHRONOLOGICAL inner split of the (already date-sorted) training
    window - the last 20% of train rows act as the validation fold. Random
    K-fold is intentionally avoided: these are game logs, and shuffling lets
    a row's future leak into its own training set, producing optimistic params
    that fall apart out-of-sample.

    Returns the best parameter dict (empty if tuning can't run, in which case
    build_sklearn_model falls back to its hand-tuned defaults).
    """
    if not (HAS_XGB and target is not None):
        return {}
    n = len(X_train)
    cut = int(n * 0.8)
    if cut < 60 or (n - cut) < 25:
        return {}

    X_tr, X_val = X_train.iloc[:cut], X_train.iloc[cut:]
    y_tr, y_val = y_train.iloc[:cut], y_train.iloc[cut:]

    rng = random.Random(seed)
    keys = list(XGB_TUNE_GRID)
    best_params: dict = {}
    best_mae = float("inf")
    for _ in range(max(1, n_iter)):
        params = {k: rng.choice(XGB_TUNE_GRID[k]) for k in keys}
        try:
            m = build_sklearn_model("xgb", target_name=target, params=params)
            m.fit(X_tr, y_tr)
            mae = float(mean_absolute_error(y_val, m.predict(X_val)))
        except Exception:
            continue
        if mae < best_mae:
            best_mae, best_params = mae, params
    if best_params:
        print(f"      [tune {target}] inner-val MAE={best_mae:.4f}  "
              f"depth={best_params['max_depth']} n_est={best_params['n_estimators']} "
              f"lr={best_params['learning_rate']} reg_l={best_params['reg_lambda']}")
    return best_params


def chronological_split(df: pd.DataFrame, date_col: str, frac: float = TRAIN_FRAC):
    work = df.copy()
    work = work[work[date_col].notna()].sort_values(date_col).reset_index(drop=True)

    unique_dates = sorted(pd.to_datetime(work[date_col]).dt.normalize().unique())
    if len(unique_dates) < 2:
        return work.copy(), work.iloc[0:0].copy()

    cutoff_idx = max(1, int(len(unique_dates) * frac))
    cutoff_idx = min(cutoff_idx, len(unique_dates) - 1)

    train_dates = set(unique_dates[:cutoff_idx])
    test_dates = set(unique_dates[cutoff_idx:])

    train_df = work[pd.to_datetime(work[date_col]).dt.normalize().isin(train_dates)].copy()
    test_df = work[pd.to_datetime(work[date_col]).dt.normalize().isin(test_dates)].copy()
    return train_df, test_df


def select_features(df: pd.DataFrame, candidate_features: list) -> list:
    present = [f for f in candidate_features if f in df.columns]
    good = [f for f in present if df[f].notna().sum() > 50]
    return good


def build_sklearn_model(model_type: str = "rf", target_name: str | None = None,
                        params: dict | None = None) -> Pipeline:
    """Build an (imputer -> model) pipeline, optionally target-transformed.

    `params`, when given, overrides the default XGBoost hyperparameters - this
    is how the randomized search in `_tune_xgb` evaluates candidate configs and
    how the winning config is rebuilt for the final fit. It's ignored for rf/lin.
    """
    imputer = SimpleImputer(strategy="median")

    if model_type == "rf":
        if target_name == "PA":
            # PA is relatively smooth - slightly deeper tree is OK
            model = RandomForestRegressor(
                n_estimators=400,
                max_depth=5,
                min_samples_leaf=20,
                max_features=0.6,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
        else:
            # Count targets (H, HR, BB, K, IP): stay shallow to avoid
            # memorising hot/cold streaks from short rolling windows.
            # With ~1-5k training rows, max_depth > 5 leads to severe overfit.
            model = RandomForestRegressor(
                n_estimators=400,
                max_depth=4,
                min_samples_leaf=25,
                max_features=0.5,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
        base = Pipeline([("imputer", imputer), ("model", model)])

    elif model_type == "xgb" and HAS_XGB:
        if target_name == "PA":
            xgb_kwargs = dict(
                n_estimators=400,
                max_depth=3,
                learning_rate=0.04,
                subsample=0.8,
                colsample_bytree=0.7,
                reg_alpha=0.5,
                reg_lambda=2.0,
                min_child_weight=20,
                random_state=RANDOM_STATE,
                verbosity=0,
                n_jobs=-1,
            )
        else:
            # Strong regularisation: forces regression toward mean,
            # prevents learning "hot streak -> predict high counts".
            xgb_kwargs = dict(
                n_estimators=400,
                max_depth=3,
                learning_rate=0.03,
                subsample=0.75,
                colsample_bytree=0.65,
                reg_alpha=1.0,
                reg_lambda=3.0,
                min_child_weight=30,
                random_state=RANDOM_STATE,
                verbosity=0,
                n_jobs=-1,
            )
        # Count targets: use Poisson loss (natural for non-negative integer
        # counts). XGB minimises Poisson deviance, outputs exp(margin) so
        # predictions are already ≥ 0 with no log1p wrapper needed.
        # Continuous / rate targets keep the default reg:squarederror.
        if target_name in COUNT_TARGETS:
            xgb_kwargs["objective"] = "count:poisson"
            # Counts are integers ≥ 0 - tighter min_child_weight than the
            # default lets the model fit the long tail (e.g. 2-HR games)
            # without overfitting; Poisson loss already regularises strongly.
            xgb_kwargs.setdefault("max_delta_step", 0.7)
        # Tuned hyperparameters (from _tune_xgb) override the hand-picked
        # defaults above. random_state / verbosity / n_jobs are never tuned.
        if params:
            xgb_kwargs.update({k: v for k, v in params.items()
                               if k not in ("random_state", "verbosity", "n_jobs",
                                            "objective", "max_delta_step")})
        model = XGBRegressor(**xgb_kwargs)
        base = Pipeline([("imputer", imputer), ("model", model)])

    elif model_type == "lin":
        # Linear model. Every player target is a non-negative count or rate, so
        # this is a Poisson GLM (linear on the log scale): predictions can't go
        # negative and the loss matches the Poisson deviance used in evaluation.
        # Light L2 penalty on standardised features (many of them overlap).
        model = PoissonRegressor(alpha=0.05, max_iter=1000)
        base = Pipeline([
            ("imputer", imputer),
            ("scaler", StandardScaler()),
            ("model", model),
        ])
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Rate targets (legacy, kept for backward compat if rates are ever needed)
    if target_name in {"h_rate", "hr_rate", "bb_rate", "k_rate",
                       "K_rate", "BB_rate", "HR_rate", "H_rate"}:
        return TransformedTargetRegressor(
            regressor=base,
            func=_logit_transform,
            inverse_func=_inv_logit_transform,
            check_inverse=False,
        )

    # Counting-stat targets - non-negative integers.
    # XGB with objective='count:poisson' already outputs ≥ 0 in count space
    # (the link function does the exp internally), so NO log1p wrapper.
    # RF has no Poisson option, so it keeps the log1p wrapper as
    # a way to enforce non-negativity and stabilise variance.
    if target_name in COUNT_TARGETS:
        if (model_type == "xgb" and HAS_XGB) or model_type == "lin":
            return base   # Poisson XGB / Poisson GLM - predictions already in count space
        return TransformedTargetRegressor(
            regressor=base,
            func=np.log1p,
            inverse_func=np.expm1,
            check_inverse=False,
        )

    return base


def feature_importance(pipeline, feature_names: list) -> pd.DataFrame:
    fitted = pipeline
    if isinstance(fitted, CalibratedRegressor):
        fitted = fitted.base
    if isinstance(fitted, TransformedTargetRegressor):
        fitted = fitted.regressor_

    if hasattr(fitted, "named_steps"):
        model = fitted.named_steps.get("model")
    else:
        model = fitted

    if hasattr(model, "feature_importances_") or hasattr(model, "coef_"):
        # linear model: |coefficient| on standardised features
        imp = model.feature_importances_ if hasattr(model, "feature_importances_") else np.abs(model.coef_)
        return (
            pd.DataFrame({"feature": feature_names, "importance": imp})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
    return pd.DataFrame()


def validate_pitcher_training_data(df: pd.DataFrame):
    work = df.copy()

    if "IP" not in work.columns:
        raise ValueError("Pitcher training data is missing IP column.")

    work["IP"] = pd.to_numeric(work["IP"], errors="coerce")
    mean_ip = float(work["IP"].mean())
    median_ip = float(work["IP"].median())

    print("\nPitcher training data check:")
    print(f"  Rows:      {len(work):,}")
    print(f"  Mean IP:   {mean_ip:.3f}")
    print(f"  Median IP: {median_ip:.3f}")

    if "is_actual_starter" in work.columns:
        starter_rate = float(pd.to_numeric(work["is_actual_starter"], errors="coerce").fillna(0).mean())
        print(f"  Starter flag mean: {starter_rate:.3f}")

    if mean_ip < 3.5:
        raise ValueError(
            f"Pitcher training data still looks relief-heavy (mean IP={mean_ip:.3f}). "
            "Rebuild pitcher_game_data.csv before training."
        )


# ---------------------------------------------------------------------------
# Chronological train / validation / test protocol
# ---------------------------------------------------------------------------
# Audit changes (see docs in outputs/model_evaluation/report):
#   * The model family (RF / XGB / NN) used to be chosen by TEST-set RMSE, so
#     the reported "test" error was the minimum over three looks at the test
#     set. Selection now happens on a chronological VALIDATION window; the
#     test window (when one is requested) is touched once, after selection.
#   * cross_val_score() with default (unshuffled K-fold) trained on future
#     folds to predict past ones. It was diagnostic-only and is replaced by
#     the validation window.
#   * The production models were fit on the first 75% of dates only and never
#     saw the most recent quarter of data. After selection the chosen family
#     is now refit on train + validation (everything before the test window,
#     or everything available in production).
#   * Every feature list passes leakage_guard.assert_pregame_features().
from leakage_guard import assert_pregame_features
import model_stacking as stk

VALID_FRAC = 0.15
# The MLP ("nn") was removed: weakest family on validation for every target and a
# large share of training time (see the technical report, section 7).
MODEL_TYPES = ["rf"] + (["xgb"] if HAS_XGB else []) + ["lin"]

PITCHER_ID_COLS = ["game_date", "game_pk", "pitcher", "pitcher_name", "team", "opponent_team",
                   "prediction_season", "history_seasons"]
HITTER_ID_COLS = ["game_date", "game_pk", "batter", "batter_name", "team", "pitcher_team",
                  "opp_sp_name", "prediction_season", "history_seasons"]


def date_split(df: pd.DataFrame, date_col: str, valid_start=None, test_start=None,
               valid_frac: float = VALID_FRAC):
    """Split by calendar date: train < valid_start <= valid < test_start <= test.

    Without explicit dates the validation window is the most recent
    `valid_frac` of distinct dates before the test window (production mode:
    no test window)."""
    work = df[df[date_col].notna()].copy()
    work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
    work = work.sort_values(date_col, kind="mergesort").reset_index(drop=True)
    d = work[date_col].dt.normalize()
    if test_start is not None:
        test_mask = d >= pd.Timestamp(test_start)
    else:
        test_mask = pd.Series(False, index=work.index)
    pool = work[~test_mask]
    if valid_start is None:
        dates = sorted(pool[date_col].dt.normalize().unique())
        cut = dates[max(1, int(len(dates) * (1 - valid_frac)))] if len(dates) > 2 else dates[-1]
        valid_start = cut
    vs = pd.Timestamp(valid_start)
    train = work[(d < vs) & ~test_mask].copy()
    valid = work[(d >= vs) & ~test_mask].copy()
    test = work[test_mask].copy()
    return train, valid, test


def _fit_calibrated(X_train, y_train, target, model_type, best_params, calibrate):
    """Fit one model family exactly as the original pipeline did (optional
    tail-of-train linear/isotonic calibration, then a full fit)."""
    calib_a, calib_b, calib_iso = 0.0, 1.0, None
    floor = 0.0
    if calibrate and len(X_train) >= 200:
        k = int(len(X_train) * (1.0 - CALIB_FRAC))
        if k >= 60 and (len(X_train) - k) >= 30:
            cal_fit = build_sklearn_model(model_type, target_name=target, params=best_params)
            cal_fit.fit(X_train.iloc[:k], y_train.iloc[:k])
            cal_pred = cal_fit.predict(X_train.iloc[k:])
            y_holdout = y_train.iloc[k:]
            calib_a, calib_b = fit_linear_calibration(y_holdout, cal_pred)
            mae_raw = float(np.mean(np.abs(y_holdout - cal_pred)))
            mae_lin = float(np.mean(np.abs(y_holdout - np.clip(calib_a + calib_b * cal_pred, floor, None))))
            iso = fit_isotonic_calibration(y_holdout, cal_pred)
            if iso is not None:
                iso_pred = np.clip(iso.predict(cal_pred), floor, None)
                mae_iso = float(np.mean(np.abs(y_holdout - iso_pred)))
                if mae_iso < min(mae_lin, mae_raw) - 0.002:
                    calib_iso = iso
                    calib_a, calib_b = 0.0, 1.0
    pipe = build_sklearn_model(model_type, target_name=target, params=best_params)
    pipe.fit(X_train, y_train)
    model = CalibratedRegressor(pipe, a=calib_a, b=calib_b, floor=floor, iso=calib_iso)
    kind = "iso" if calib_iso is not None else ("lin" if (calib_a, calib_b) != (0.0, 1.0) else "none")
    return model, pipe, {"a": calib_a, "b": calib_b, "kind": kind}


def _metric_block(y, p, prefix):
    y = np.asarray(y, float); p = np.asarray(p, float)
    if len(y) == 0:
        return {}
    return {f"{prefix}_mae": float(np.mean(np.abs(p - y))),
            f"{prefix}_rmse": float(np.sqrt(np.mean((p - y) ** 2))),
            f"{prefix}_bias": float(np.mean(p - y)),
            f"{prefix}_n": int(len(y)),
            f"{prefix}_actual_mean": float(np.mean(y)),
            f"{prefix}_pred_mean": float(np.mean(p))}


def train_one_target(train_df, valid_df, test_df, features, target, id_cols,
                     model_types=None, tune=False, n_iter=TUNE_ITER_DEFAULT,
                     calibrate=False, preset_params=None, compare_on_test=False) -> dict:
    model_types = model_types or MODEL_TYPES
    if target not in train_df.columns:
        print(f"    [skip] '{target}' not in data")
        return {}
    feats = select_features(train_df, features)
    if not feats:
        return {}
    assert_pregame_features(feats, context=f"target {target}")

    tr = train_df[train_df[target].notna()]
    va = valid_df[valid_df[target].notna()]
    te = test_df[test_df[target].notna()] if test_df is not None and len(test_df) else test_df
    if tr.empty or va.empty:
        print(f"    [skip] '{target}' has empty train/validation")
        return {}
    X_tr, y_tr = tr[feats], tr[target]
    X_va, y_va = va[feats], va[target]

    best_params = {}
    if HAS_XGB and "xgb" in model_types:
        if preset_params:
            best_params = dict(preset_params)
        elif tune:
            best_params = _tune_xgb(X_tr, y_tr, target, n_iter=n_iter)   # train window only

    # 1) model-family selection on the validation window
    per_type = {}
    has_test = te is not None and len(te) > 0
    for mt in model_types:
        try:
            m, _, cal = _fit_calibrated(X_tr, y_tr, target, mt, best_params if mt == "xgb" else {}, calibrate)
            pv = np.asarray(m.predict(X_va), float)
            per_type[mt] = {"valid_pred": pv, "calibration": cal, **_metric_block(y_va, pv, "valid")}
            if compare_on_test and has_test:
                # family comparison on the test window with every family fit
                # identically on the TRAIN window only (reporting, never selection)
                per_type[mt]["test_pred_trainonly"] = np.asarray(m.predict(te[feats]), float)
                per_type[mt].update(_metric_block(te[target], per_type[mt]["test_pred_trainonly"], "test_trainonly"))
        except Exception as e:
            print(f"    [{target}/{mt}] failed: {e}")
    if not per_type:
        return {}

    # 1b) stacking: even-weight and optimised-weight averages of the families.
    # Weights come from validation predictions only; the optimised stack is
    # scored for selection with cross-fitted (out-of-fold) predictions.
    fams = [mt for mt in model_types if mt in per_type]
    if len(fams) >= 2:
        yv = y_va.to_numpy(float)
        P_va = np.column_stack([per_type[mt]["valid_pred"] for mt in fams])
        w_opt = stk.fit_weights(P_va, yv, "rmse")
        for name, w, pv_sel in (("stack_even", stk.even_weights(len(fams)), None),
                                ("stack_opt", w_opt, stk.crossfit_predictions(P_va, yv, "rmse"))):
            pv = np.clip(P_va @ w, 0, None)
            score = pv if pv_sel is None else np.clip(pv_sel, 0, None)
            per_type[name] = {"valid_pred": pv, "members": fams, "weights": w,
                              "calibration": {"kind": "stack", "weights": stk.describe(fams, w)},
                              **_metric_block(yv, score, "valid")}
            if compare_on_test and has_test and all("test_pred_trainonly" in per_type[mt] for mt in fams):
                T = np.column_stack([per_type[mt]["test_pred_trainonly"] for mt in fams])
                per_type[name]["test_pred_trainonly"] = np.clip(T @ w, 0, None)
                per_type[name].update(_metric_block(te[target], per_type[name]["test_pred_trainonly"], "test_trainonly"))

    chosen = min(per_type, key=lambda k: per_type[k]["valid_rmse"])

    # 2) refit the chosen model (or every member of a chosen stack) on
    #    train + validation; score the test window once
    full = pd.concat([tr, va])
    X_full, y_full = full[feats], full[target]
    test_preds = {}
    if chosen.startswith("stack_"):
        members, pipes = [], []
        for mt in per_type[chosen]["members"]:
            m, pipe, _ = _fit_calibrated(X_full, y_full, target, mt, best_params if mt == "xgb" else {}, calibrate)
            members.append(m); pipes.append(pipe)
        w = per_type[chosen]["weights"]
        final_model = stk.StackedRegressor(members, w, per_type[chosen]["members"])
        final_pipe = pipes[int(np.argmax(w))]                 # importance from the heaviest member
        final_cal = per_type[chosen]["calibration"]
    else:
        final_model, final_pipe, final_cal = _fit_calibrated(
            X_full, y_full, target, chosen, best_params if chosen == "xgb" else {}, calibrate)
    if te is not None and len(te):
        test_preds[chosen] = np.asarray(final_model.predict(te[feats]), float)
        per_type[chosen].update(_metric_block(te[target], test_preds[chosen], "test"))

    metrics_rows = []
    for mt, r in per_type.items():
        row = {"target": target, "model": mt, "selected": mt == chosen,
               "selection_metric": "valid_rmse", "n_train": int(len(tr)),
               "n_train_plus_valid": int(len(full)), "tuned": bool(best_params) and mt == "xgb",
               "calibration": r["calibration"]["kind"],
               "stack_weights": r["calibration"].get("weights", ""),
               "selection_score": "cross-fitted" if mt == "stack_opt" else "validation",
               "train_start": str(tr["game_date"].min().date()), "train_end": str(tr["game_date"].max().date()),
               "valid_start": str(va["game_date"].min().date()), "valid_end": str(va["game_date"].max().date())}
        row.update({k: v for k, v in r.items() if k.startswith(("valid_", "test_")) and k not in ("valid_pred", "test_pred_trainonly")})
        metrics_rows.append(row)
        flag = " <= selected" if mt == chosen else ""
        tmsg = f"  test MAE={r['test_mae']:.4f}" if "test_mae" in r else (
            f"  test MAE(train-only fit)={r['test_trainonly_mae']:.4f}" if "test_trainonly_mae" in r else "")
        print(f"    {target:<12} {mt.upper():<10} valid RMSE={r['valid_rmse']:.4f} MAE={r['valid_mae']:.4f}{tmsg}{flag}")

    pred_df = None
    if te is not None and len(te) and test_preds:
        keep = [c for c in id_cols if c in te.columns]
        pred_df = te[keep].copy()
        pred_df["target"] = target
        pred_df["actual"] = te[target].to_numpy(float)
        for mt, r in per_type.items():
            if "test_pred_trainonly" in r:
                pred_df[f"pred_{mt}"] = r["test_pred_trainonly"]     # train-window fit, comparison only
        pred_df["pred_selected"] = test_preds[chosen]                # train+validation refit
        pred_df["selected_model"] = chosen

    return {"pipeline": final_model, "raw_pipeline": final_pipe, "features": feats,
            "model_type": chosen, "calibration": final_cal, "best_params": best_params,
            "metrics_rows": metrics_rows, "predictions": pred_df,
            "importance": feature_importance(final_pipe, feats),
            "trained_through": str(full["game_date"].max().date())}


def _save_bundle(path: Path, res: dict, extra: dict | None = None):
    bundle = {"pipeline": res["pipeline"], "features": res["features"],
              "model_type": res["model_type"], "calibration": res["calibration"],
              "best_params": res["best_params"], "trained_through": res["trained_through"],
              "selected_on": "chronological validation window",
              "history_rule": "previous two seasons + current season before game date"}
    bundle.update(extra or {})
    with open(path, "wb") as f:
        pickle.dump(bundle, f)


def _train_group(df, model_dir, prefix, count_targets, rate_targets, attach_rates, features,
                 id_cols, tune, n_iter, calibrate, preset_params, valid_start, test_start,
                 save_models, compare_on_test, rate_kind):
    date_col = "game_date"
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    train_df, valid_df, test_df = date_split(df, date_col, valid_start, test_start)
    print(f"  Train: {len(train_df):,} ({train_df[date_col].min().date()} -> {train_df[date_col].max().date()})  "
          f"|  Valid: {len(valid_df):,} ({valid_df[date_col].min().date()} -> {valid_df[date_col].max().date()})  "
          f"|  Test: {len(test_df):,}")
    results, metrics, preds = {}, [], []
    for target_list, kind in ((count_targets, None), (rate_targets, rate_kind)):
        if kind:
            train_df, valid_df, test_df = attach_rates(train_df), attach_rates(valid_df), attach_rates(test_df)
            print(f"\n  -- RATE TARGETS ({kind}) --")
        for target in target_list:
            print(f"\n  Target: {target}")
            res = train_one_target(train_df, valid_df, test_df, features, target, id_cols,
                                   tune=tune, n_iter=n_iter, calibrate=calibrate,
                                   preset_params=(preset_params or {}).get(target),
                                   compare_on_test=compare_on_test)
            if not res:
                continue
            results[target] = res
            metrics.extend(res["metrics_rows"])
            if res["predictions"] is not None:
                preds.append(res["predictions"])
            if save_models:
                _save_bundle(model_dir / f"{prefix}_{target}.pkl", res,
                             {"target_kind": kind} if kind else None)
    if save_models and metrics:
        pd.DataFrame(metrics).to_csv(model_dir / f"{prefix}_metrics.csv", index=False)
        imp = [r["importance"].assign(target=t) for t, r in results.items() if not r["importance"].empty]
        if imp:
            pd.concat(imp, ignore_index=True).to_csv(model_dir / f"{prefix}_importance.csv", index=False)
    return {"results": results, "metrics": pd.DataFrame(metrics),
            "predictions": pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()}


def train_pitcher_models(df: pd.DataFrame, model_dir: Path, tune: bool = False,
                         n_iter: int = TUNE_ITER_DEFAULT, calibrate: bool = False,
                         preset_params: dict | None = None, valid_start=None, test_start=None,
                         save_models: bool = True, compare_on_test: bool = False) -> dict:
    print("\n" + "=" * 60 + "\nPITCHER MODELS\n" + "=" * 60)
    df = df.copy()
    if "is_actual_starter" in df.columns:
        df = df[pd.to_numeric(df["is_actual_starter"], errors="coerce").fillna(0) == 1]
    return _train_group(df, Path(model_dir), "pitcher", PITCHER_TARGETS, PITCHER_RATE_TARGETS,
                        _attach_pitcher_rates, PITCHER_FEATURES, PITCHER_ID_COLS, tune, n_iter,
                        calibrate, preset_params, valid_start, test_start, save_models,
                        compare_on_test, "rate_per_9")


def train_hitter_models(df: pd.DataFrame, model_dir: Path, tune: bool = False,
                        n_iter: int = TUNE_ITER_DEFAULT, calibrate: bool = False,
                        preset_params: dict | None = None, valid_start=None, test_start=None,
                        save_models: bool = True, compare_on_test: bool = False) -> dict:
    print("\n" + "=" * 60 + "\nHITTER MODELS\n" + "=" * 60)
    df = clean_hitter_training_rows(df)
    return _train_group(df, Path(model_dir), "hitter", HITTER_TARGETS, HITTER_RATE_TARGETS,
                        _attach_hitter_rates, HITTER_FEATURES, HITTER_ID_COLS, tune, n_iter,
                        calibrate, preset_params, valid_start, test_start, save_models,
                        compare_on_test, "rate_per_PA")


def load_training_tables(seasons: list[int] | None = None):
    """Feature tables for training. Prefers the windowed multi-season build
    (data/features/*_games.csv from build_features.py); falls back to the
    committed current-season CSVs."""
    feat_dir = Path("data/features")
    pf, hf = feat_dir / "pitcher_games.csv", feat_dir / "hitter_games.csv"
    if pf.exists() and hf.exists():
        p, h = pd.read_csv(pf, low_memory=False), pd.read_csv(hf, low_memory=False)
    elif (feat_dir / "pitcher_games.parquet").exists() and (feat_dir / "hitter_games.parquet").exists():  # legacy cache
        p, h = pd.read_parquet(feat_dir / "pitcher_games.parquet"), pd.read_parquet(feat_dir / "hitter_games.parquet")
    else:
        p = pd.read_csv("data/pitcher_game_data.csv", low_memory=False)
        h = pd.read_csv("data/hitter_game_data.csv", low_memory=False)
    if seasons:
        p = p[pd.to_datetime(p["game_date"], errors="coerce").dt.year.isin(seasons)]
        h = h[pd.to_datetime(h["game_date"], errors="coerce").dt.year.isin(seasons)]
    return p, h


def main():
    parser = argparse.ArgumentParser(description="Train MLB hitter + pitcher projection models")
    parser.add_argument("--model-dir", default="models", help="Directory to save trained models")
    parser.add_argument("--tune", action="store_true",
                        help="Fresh randomized XGB search on the TRAIN window (default: reuse tuned presets).")
    parser.add_argument("--no-calibrate", action="store_true")
    parser.add_argument("--tune-iter", type=int, default=TUNE_ITER_DEFAULT)
    parser.add_argument("--train-seasons", nargs="+", type=int, default=None,
                        help="Seasons whose rows are used as training examples (default: all built).")
    parser.add_argument("--valid-start", default=None, help="YYYY-MM-DD (default: last 15%% of dates)")
    args = parser.parse_args()

    model_dir = Path(args.model_dir); model_dir.mkdir(parents=True, exist_ok=True)
    pitcher_df, hitter_df = load_training_tables(args.train_seasons)
    if args.train_seasons is None:
        from run_backtest import training_seasons      # earliest of >=3 cached seasons = history only
        pitcher_df = pitcher_df[pd.to_datetime(pitcher_df["game_date"]).dt.year.isin(training_seasons(pitcher_df))]
        hitter_df = hitter_df[pd.to_datetime(hitter_df["game_date"]).dt.year.isin(training_seasons(hitter_df))]
    validate_pitcher_training_data(pitcher_df)
    from daily_update import _load_preset_params
    pp = None if args.tune else _load_preset_params("pitcher", PITCHER_TARGETS + PITCHER_RATE_TARGETS)
    hp = None if args.tune else _load_preset_params("hitter", HITTER_TARGETS + HITTER_RATE_TARGETS)
    train_pitcher_models(pitcher_df, model_dir, tune=args.tune, n_iter=args.tune_iter,
                         calibrate=not args.no_calibrate, preset_params=pp, valid_start=args.valid_start)
    train_hitter_models(hitter_df, model_dir, tune=args.tune, n_iter=args.tune_iter,
                        calibrate=not args.no_calibrate, preset_params=hp, valid_start=args.valid_start)
    print("\nDone.")


if __name__ == "__main__":
    main()
