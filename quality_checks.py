"""
quality_checks.py
=================
Automated integrity checks run by evaluate_models.py before ANY metric is
computed.  A critical failure raises `QualityCheckError` (the evaluation
stops) instead of quietly producing a misleading number.

Checks
------
prediction validity   probabilities in [0,1]; counts finite and >= 0;
                      required columns; no missing predictions; no duplicates
actual matching       one actual per game / player-game-target; actual values
                      plausible for the target; winners consistent with labels
temporal integrity    every test row on/after the test-window start; models
                      trained only on data before it (backtest metadata);
                      feature tables respect the two-season window
player history        history_seasons of every row is a subset of S-2..S and
                      never contains a later season
metric sanity         enough rows for each reported metric
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import history_window as hw


class QualityCheckError(AssertionError):
    pass


class Report:
    def __init__(self):
        self.rows = []

    def add(self, area, name, passed, detail, critical=True):
        self.rows.append({"area": area, "check": name, "passed": bool(passed),
                          "critical": bool(critical), "detail": str(detail)})

    @property
    def failures(self):
        return [r for r in self.rows if r["critical"] and not r["passed"]]

    def raise_if_failed(self):
        if self.failures:
            msg = "\n".join(f"  [{r['area']}] {r['check']}: {r['detail']}" for r in self.failures)
            raise QualityCheckError("critical quality checks failed:\n" + msg)

    def to_frame(self):
        return pd.DataFrame(self.rows)


# plausible single-game maxima (anything above is a join / parsing error)
PLAUSIBLE_MAX = {"pitcher": {"K": 20, "BB": 12, "HR": 8, "H": 20, "IP": 12, "R": 20},
                 "hitter": {"H": 6, "HR": 4, "BB": 6, "K": 6, "PA": 9, "TB": 20, "2B": 4, "3B": 3, "SB": 5}}


def check_game_predictions(df: pd.DataFrame, test_start: str | None, rep: Report):
    req = ["game_date", "game_pk", "home_team", "away_team", "actual_home_win", "p_selected",
           "predicted_home_win", "correct"]
    miss = [c for c in req if c not in df.columns]
    rep.add("game", "required_columns", not miss, miss or "ok")
    if miss:
        return
    pcols = [c for c in df.columns if c.startswith("p_")]
    bad = {c: int(((df[c] < 0) | (df[c] > 1) | df[c].isna()).sum()) for c in pcols}
    rep.add("game", "probabilities_in_unit_interval", sum(bad.values()) == 0, bad)
    rep.add("game", "no_duplicate_games", not df["game_pk"].duplicated().any(),
            f"{int(df['game_pk'].duplicated().sum())} duplicate game_pk")
    rep.add("game", "binary_target", set(df["actual_home_win"].unique()) <= {0, 1},
            sorted(df["actual_home_win"].unique().tolist()))
    rep.add("game", "predicted_label_matches_probability",
            bool(((df["p_selected"] > 0.5).astype(int) == df["predicted_home_win"]).all()), "threshold 0.5")
    rep.add("game", "correct_flag_consistent",
            bool((df["correct"] == (df["predicted_home_win"] == df["actual_home_win"]).astype(int)).all()), "ok")
    if "actual_winner" in df.columns:
        exp = np.where(df["actual_home_win"] == 1, df["home_team"], df["away_team"])
        rep.add("game", "winner_matches_label", bool((df["actual_winner"] == exp).all()), "ok")
    rep.add("game", "home_away_distinct", bool((df["home_team"] != df["away_team"]).all()), "ok")
    if test_start:
        early = int((pd.to_datetime(df["game_date"]) < pd.Timestamp(test_start)).sum())
        rep.add("game", "rows_inside_test_window", early == 0, f"{early} rows before {test_start}")
    rep.add("game", "sample_size", len(df) >= 200, f"n={len(df)}", critical=False)


def check_player_predictions(df: pd.DataFrame, group: str, test_start: str | None, rep: Report):
    area = f"player:{group}"
    req = ["game_date", "game_pk", "player_id", "target", "actual", "pred_final", "error", "abs_error"]
    miss = [c for c in req if c not in df.columns]
    rep.add(area, "required_columns", not miss, miss or "ok")
    if miss:
        return
    dups = int(df.duplicated(["game_pk", "player_id", "target"]).sum())
    rep.add(area, "no_duplicate_player_game_target", dups == 0, f"{dups} duplicates")
    pred_cols = [c for c in df.columns if c.startswith(("pred_", "baseline_"))]
    nonfinite = {c: int((~np.isfinite(pd.to_numeric(df[c], errors="coerce"))).sum()) for c in pred_cols}
    # rate x opportunity is NaN by design for targets without a rate model (IP, PA)
    nonfinite.pop("pred_rate_x_opportunity", None)
    rep.add(area, "predictions_finite", sum(nonfinite.values()) == 0, {k: v for k, v in nonfinite.items() if v})
    neg = {c: int((pd.to_numeric(df[c], errors="coerce") < -1e-9).sum()) for c in pred_cols}
    rep.add(area, "predictions_non_negative", sum(neg.values()) == 0, {k: v for k, v in neg.items() if v})
    rep.add(area, "actuals_present", int(df["actual"].isna().sum()) == 0, f"{int(df['actual'].isna().sum())} missing")
    caps = PLAUSIBLE_MAX[group]
    implausible = {t: int(((g["actual"] < 0) | (g["actual"] > caps.get(t, 99))).sum()) for t, g in df.groupby("target")}
    rep.add(area, "actuals_plausible", sum(implausible.values()) == 0, {k: v for k, v in implausible.items() if v})
    err = (df["pred_final"] - df["actual"] - df["error"]).abs().max()
    rep.add(area, "residual_convention_predicted_minus_actual", err < 1e-3, f"max deviation {err:.2e}")
    if test_start:
        early = int((pd.to_datetime(df["game_date"]) < pd.Timestamp(test_start)).sum())
        rep.add(area, "rows_inside_test_window", early == 0, f"{early} rows before {test_start}")
    if "history_seasons" in df.columns and "prediction_season" in df.columns:
        check_history_column(df, area, rep)
    if group == "hitter" and {"H", "PA"} <= set(df["target"].unique()):
        w = df.pivot_table(index=["game_pk", "player_id"], columns="target", values="actual")
        bad = int((w["H"] > w["PA"]).sum())
        rep.add(area, "hits_not_above_pa", bad == 0, f"{bad} player-games with H > PA")
    for t, g in df.groupby("target"):
        rep.add(area, f"sample_size_{t}", len(g) >= 300, f"n={len(g)}", critical=False)


def check_history_column(df: pd.DataFrame, area: str, rep: Report):
    bad = 0
    for (S, hs), _ in df.groupby(["prediction_season", "history_seasons"]):
        used = {int(x) for x in str(hs).split("+") if x}
        allowed = set(hw.eligible_seasons(int(S)))
        if not used <= allowed or max(used) > int(S):
            bad += 1
    rep.add(area, "two_season_history_rule", bad == 0,
            f"{bad} (season, history) combinations outside S-2..S")
    yr = pd.to_datetime(df["game_date"]).dt.year
    rep.add(area, "prediction_season_matches_game_year", bool((yr == df["prediction_season"]).all()), "ok")


def check_feature_tables(feat_dir: Path, rep: Report, sample_players: int = 25, seed: int = 7):
    """Recompute a sample of trailing features from raw per-game rows and
    confirm they exclude the current game (and older-than-window seasons)."""
    import hitterspitchers_data as hpd
    # The hitter table keeps EVERY batter-game (the pitcher table keeps starts
    # only, while its trailing windows also count relief outings), so it is
    # the one whose trailing features can be recomputed exactly from its rows.
    p = feat_dir / "hitter_games.csv"
    if not p.exists() and (feat_dir / "hitter_games.parquet").exists():   # legacy cache
        p = feat_dir / "hitter_games.parquet"
    if not p.exists():
        rep.add("features", "hitter_table_present", False, f"{p} missing", critical=False)
        return
    cols = ["batter", "game_date", "game_pk", "H", "H_last10", "H_std", "prediction_season", "history_seasons"]
    df = pd.read_csv(p, low_memory=False)[cols] if p.suffix == ".csv" else pd.read_parquet(p, columns=cols)
    df = df.rename(columns={"batter": "pitcher", "H": "K", "H_last10": "K_last10", "H_std": "K_std"})
    df["game_date"] = pd.to_datetime(df["game_date"])
    check_history_column(df, "features:hitter", rep)
    rng = np.random.default_rng(seed)
    ids = rng.choice(df["pitcher"].unique(), size=min(sample_players, df["pitcher"].nunique()), replace=False)
    worst_l10 = worst_std = 0.0
    for pid in ids:
        g = df[df["pitcher"] == pid].sort_values(["game_date", "game_pk"]).reset_index(drop=True)
        season = g["game_date"].dt.year.to_numpy()
        k = g["K"].to_numpy(float)
        for i in range(len(g)):
            S = season[i]
            # eligible history: this pitcher's starts in S-2..S strictly before row i
            hist = k[:i][(season[:i] >= S - 2) & (season[:i] <= S)]
            if len(hist):
                exp = hist[-10:].mean()
                worst_l10 = max(worst_l10, abs(exp - g["K_last10"].iloc[i]))
            if len(hist):
                worst_std = max(worst_std, abs(hist.mean() - g["K_std"].iloc[i]))
            elif not np.isnan(g["K_std"].iloc[i]):
                worst_std = max(worst_std, 1.0)      # no eligible history -> must be NaN
    rep.add("features", "H_last10_excludes_current_game_and_pre_window_seasons", worst_l10 < 1e-6,
            f"max |recomputed - stored| = {worst_l10:.2e} over {len(ids)} hitters")
    rep.add("features", "H_std_is_prior_games_in_window", worst_std < 1e-6,
            f"max |recomputed - stored| = {worst_std:.2e}")


def check_backtest_metadata(meta_path: Path, selection_paths: list[Path], rep: Report):
    if not meta_path.exists():
        rep.add("temporal", "backtest_metadata_present", False, f"{meta_path} missing", critical=False)
        return None
    meta = json.loads(meta_path.read_text())
    ts = pd.Timestamp(meta["test_start"])
    for p in selection_paths:
        if not p.exists():
            continue
        sel = pd.read_csv(p)
        if "valid_end" in sel.columns:
            late = sel[pd.to_datetime(sel["valid_end"]) >= ts]
            rep.add("temporal", f"{p.stem}_fit_data_before_test", late.empty,
                    f"{len(late)} model fits with data on/after {ts.date()}")
        if "selected" in sel.columns and "selection_metric" in sel.columns:
            rep.add("temporal", f"{p.stem}_selected_on_validation",
                    bool((sel["selection_metric"].astype(str).str.startswith("valid")).all()), "ok")
    return meta
