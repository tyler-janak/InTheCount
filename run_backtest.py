"""
run_backtest.py
===============
Honest out-of-sample backtest of the PLAYER projection models.

Trains the existing pitcher / hitter model stack with the chronological
protocol in hitterspitchers_train.py:

    train       : training-season rows before --valid-start
    validation  : --valid-start <= date < --test-start   (model family chosen here)
    test        : date >= --test-start                   (scored once)

and saves every test prediction - direct count model, rate x opportunity,
the production blend actually shown on the site, every candidate family and
simple baselines - so evaluate_models.py can recompute all metrics and
figures without retraining.

Nothing here overwrites the production models in models/ (save_models=False).

Training rows: the earliest cached season is used as HISTORY ONLY when three or
more seasons are cached (its rows have no prior-season history inside their
own window, so their trailing features are systematically thinner than the
rows they would be teaching the model to predict).

Usage
-----
    python run_backtest.py                                    # defaults below
    python run_backtest.py --valid-start 2026-03-01 --test-start 2026-07-01
"""

from __future__ import annotations

import argparse
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import hitterspitchers_train as hpt

HERE = Path(__file__).resolve().parent
FEAT = HERE / "data" / "features"
OUT = HERE / "outputs" / "model_evaluation"

# Production blend weights (hitterspitchers_today.py) - fixed, not tuned here.
PITCHER_BLEND_WEIGHTS = {"K": 0.50, "BB": 0.25, "H": 0.20, "HR": 0.30, "R": 0.30}
HITTER_BLEND_WEIGHTS = {"H": 0.55, "HR": 0.50, "BB": 0.50, "K": 0.50, "TB": 0.55,
                        "2B": 0.50, "3B": 0.50, "SB": 0.50}
PITCHER_RATE_OF = {"K": "K_per_9", "BB": "BB_per_9", "H": "H_per_9", "HR": "HR_per_9", "R": "R_per_9"}
HITTER_RATE_OF = {"H": "H_per_PA", "HR": "HR_per_PA", "BB": "BB_per_PA", "K": "K_per_PA",
                  "TB": "TB_per_PA", "2B": "2B_per_PA", "3B": "3B_per_PA", "SB": "SB_per_PA"}


def _presets(prefix, targets):
    out = {}
    for t in targets:
        p = HERE / "models" / f"{prefix}_{t}.pkl"
        if p.exists():
            try:
                b = pickle.load(open(p, "rb"))
                if isinstance(b, dict) and b.get("best_params"):
                    out[t] = b["best_params"]
            except Exception:
                pass
    return out


def training_seasons(df: pd.DataFrame) -> list[int]:
    yrs = sorted(pd.to_datetime(df["game_date"]).dt.year.unique().tolist())
    return yrs[1:] if len(yrs) >= 3 else yrs


def _baselines(test_rows: pd.DataFrame, target: str, league_mean: float, kind: str) -> dict:
    """Simple, pre-game baselines already present in the feature table."""
    def col(c):
        return pd.to_numeric(test_rows[c], errors="coerce") if c in test_rows.columns else pd.Series(np.nan, index=test_rows.index)
    if target == "TB" and kind == "hitter":
        std = col("H_std") + col("2B_std") + 2 * col("3B_std") + 3 * col("HR_std")
        l10 = col("H_last10") + col("2B_last10") + 2 * col("3B_last10") + 3 * col("HR_last10")
        l30 = col("H_last30") + col("2B_last30") + 2 * col("3B_last30") + 3 * col("HR_last30")
    else:
        std, l10, l30 = col(f"{target}_std"), col(f"{target}_last10"), col(f"{target}_last30")
    return {"baseline_season_avg": std.fillna(l30).fillna(league_mean).to_numpy(),
            "baseline_last10_avg": l10.fillna(league_mean).to_numpy(),
            "baseline_league_avg": np.full(len(test_rows), league_mean)}


def _key(df, id_col):
    # original-methodology tables have no game_pk: a player-game is player + date
    return ["game_pk", id_col] if "game_pk" in df.columns else ["game_date", id_col]


def _assemble(group: str, res: dict, df: pd.DataFrame, valid_start, test_start) -> pd.DataFrame:
    """Long table: one row per (player-game, count target) with every prediction."""
    preds = res["predictions"]
    if preds.empty:
        return preds
    id_col = "pitcher" if group == "pitcher" else "batter"
    name_col = "pitcher_name" if group == "pitcher" else "batter_name"
    opp_col = "opponent_team" if group == "pitcher" else "pitcher_team"
    key = _key(df, id_col)
    rate_of = PITCHER_RATE_OF if group == "pitcher" else HITTER_RATE_OF
    weights = PITCHER_BLEND_WEIGHTS if group == "pitcher" else HITTER_BLEND_WEIGHTS
    opp_target = "IP" if group == "pitcher" else "PA"
    scale = (1 / 9.0) if group == "pitcher" else 1.0

    wide = {t: g.set_index(key) for t, g in preds.groupby("target")}
    opp = wide[opp_target]["pred_selected"].clip(lower=0) if opp_target in wide else None

    d = df.copy()
    d["game_date"] = pd.to_datetime(d["game_date"])
    full = d[d["game_date"] < pd.Timestamp(test_start)]
    test_rows = d[d["game_date"] >= pd.Timestamp(test_start)]
    if key[0] == "game_date":
        test_rows = test_rows.drop_duplicates(key)
    test_rows = test_rows.set_index(key)

    targets = hpt.PITCHER_TARGETS if group == "pitcher" else hpt.HITTER_TARGETS
    out = []
    for t in targets:
        if t not in wide:
            continue
        w = wide[t].copy()
        rate_t = rate_of.get(t)
        if rate_t in wide and opp is not None:
            r = wide[rate_t]["pred_selected"].clip(lower=0).reindex(w.index)
            w["pred_rate_x_opportunity"] = r * opp.reindex(w.index) * scale
            a = weights.get(t, 0.5)
            w["pred_production_blend"] = (1 - a) * w["pred_selected"] + a * w["pred_rate_x_opportunity"]
            w["blend_weight_rate"] = a
        else:
            w["pred_rate_x_opportunity"] = np.nan
            w["pred_production_blend"] = w["pred_selected"]
            w["blend_weight_rate"] = 0.0
        league_mean = float(pd.to_numeric(full[t], errors="coerce").mean()) if t in full.columns else np.nan
        tr = test_rows.reindex(w.index)
        for k, v in _baselines(tr, t, league_mean, group).items():
            w[k] = v
        w = w.reset_index()
        w = w.rename(columns={id_col: "player_id", name_col: "player_name", opp_col: "opponent",
                              "pred_selected": "pred_direct_selected"})
        w["player_type"] = group
        w["model_name"] = w["selected_model"].map(lambda m: f"{group}_{t}_{m}")
        out.append(w)
    long = pd.concat(out, ignore_index=True)
    long["pred_final"] = long["pred_production_blend"]
    long["error"] = long["pred_final"] - long["actual"]          # residual = predicted - actual
    long["abs_error"] = long["error"].abs()
    cols = ["game_date", "game_pk", "player_type", "player_id", "player_name", "team", "opponent",
            "target", "actual", "pred_final", "error", "abs_error", "model_name", "selected_model",
            "pred_direct_selected", "pred_rate_x_opportunity", "pred_production_blend", "blend_weight_rate",
            "pred_rf", "pred_xgb", "pred_lin", "pred_stack_even", "pred_stack_opt", "baseline_season_avg", "baseline_last10_avg",
            "baseline_league_avg", "prediction_season", "history_seasons"]
    long = long[[c for c in cols if c in long.columns]]
    # original-methodology tables carry no game_pk
    return long.sort_values([c for c in ["game_date", "game_pk", "player_id", "target"] if c in long.columns])


def _load_table(feat_dir: Path, group: str) -> pd.DataFrame:
    for name in (f"{group}_games.csv", f"{group}_games.parquet"):   # parquet: legacy cache
        f = feat_dir / name
        if f.exists():
            return pd.read_csv(f, low_memory=False) if f.suffix == ".csv" else pd.read_parquet(f)
    return pd.read_csv(feat_dir / f"{group}_game_data.csv", low_memory=False)


def run(valid_start="2026-03-01", test_start="2026-07-01", calibrate=True, groups=("pitcher", "hitter"),
        feat_dir: Path = FEAT, out_dir: Path = OUT):
    global OUT
    OUT = Path(out_dir)
    (OUT / "data").mkdir(parents=True, exist_ok=True)
    (OUT / "metrics").mkdir(parents=True, exist_ok=True)
    meta = {"run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "valid_start": valid_start, "test_start": test_start, "calibrate": calibrate,
            "protocol": "model family selected on validation RMSE; chosen family refit on train+validation; test scored once",
            "hyperparameters": "XGB presets reused from models/*.pkl (tuned on data ending before the test window)"}
    for group in groups:
        df = _load_table(Path(feat_dir), group)
        seasons = training_seasons(df)
        df = df[pd.to_datetime(df["game_date"]).dt.year.isin(seasons)]
        meta[f"{group}_training_seasons"] = seasons
        prefix = group
        targets = (hpt.PITCHER_TARGETS + hpt.PITCHER_RATE_TARGETS) if group == "pitcher" else (hpt.HITTER_TARGETS + hpt.HITTER_RATE_TARGETS)
        fn = hpt.train_pitcher_models if group == "pitcher" else hpt.train_hitter_models
        # SB is excluded from evaluation (all-zero target, see evaluate_models.EXCLUDED_TARGETS);
        # skipping it here saves two model fits per family.
        if group == "hitter":
            hpt.HITTER_TARGETS = [t for t in hpt.HITTER_TARGETS if t != "SB"]
            hpt.HITTER_RATE_TARGETS = [t for t in hpt.HITTER_RATE_TARGETS if t != "SB_per_PA"]
        res = fn(df, HERE / "models", tune=False, calibrate=calibrate, preset_params=_presets(prefix, targets),
                 valid_start=valid_start, test_start=test_start, save_models=False, compare_on_test=True)
        res["metrics"].to_csv(OUT / "metrics" / f"{group}_model_selection.csv", index=False)
        clean = hpt.clean_hitter_training_rows(df) if group == "hitter" else df
        if group == "pitcher" and "is_actual_starter" in clean.columns:
            clean = clean[clean["is_actual_starter"] == 1]
        long = _assemble(group, res, clean, valid_start, test_start)
        long.to_csv(OUT / "data" / f"{group}_test_predictions.csv", index=False, float_format="%.5f")
        print(f"\n  wrote {group}_test_predictions.csv: {len(long):,} rows")
    path = OUT / "metrics" / "backtest_run.json"
    prev = json.loads(path.read_text()) if path.exists() else {}
    if prev.get("valid_start") == valid_start and prev.get("test_start") == test_start:
        prev.update(meta)          # keep the other group's entries from a split run
        meta = prev
    path.write_text(json.dumps(meta, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--valid-start", default="2026-03-01")
    ap.add_argument("--test-start", default="2026-07-01")
    ap.add_argument("--groups", nargs="+", default=["pitcher", "hitter"])
    ap.add_argument("--no-calibrate", action="store_true")
    ap.add_argument("--features-dir", default=str(FEAT),
                    help="feature tables (*_games.csv from build_features.py, or *_game_data.csv)")
    ap.add_argument("--out-dir", default=str(OUT))
    a = ap.parse_args()
    run(a.valid_start, a.test_start, not a.no_calibrate, tuple(a.groups), Path(a.features_dir), Path(a.out_dir))


if __name__ == "__main__":
    main()
