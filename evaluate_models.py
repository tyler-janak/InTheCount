"""
evaluate_models.py
==================
Reusable evaluation of saved OUT-OF-SAMPLE predictions. Never retrains.

    1. load   outputs/model_evaluation/data/{game,pitcher,hitter}_test_predictions.csv
    2. check  quality_checks.py (fails loudly on critical problems)
    3. score  metrics per model / baseline / target
    4. plot   figures/*.png (300 dpi) + the data behind every figure (data/figure_data/*.csv)
    5. save   metrics/metrics_long.csv + metrics/metrics.json (one record per metric)
    6. report generate_report.py (unless --no-report)

Usage
-----
    python evaluate_models.py
    python evaluate_models.py --no-report
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, brier_score_loss, confusion_matrix, f1_score,
                             log_loss, precision_score, recall_score, roc_auc_score, roc_curve)

import quality_checks as qc

HERE = Path(__file__).resolve().parent
EVAL = HERE / "outputs" / "model_evaluation"
DATA = EVAL / "data"
FIGDATA = DATA / "figure_data"
METRICS = EVAL / "metrics"
FIG = EVAL / "figures"

# ---------------------------------------------------------------------------
# Visual style (validated reference palette; static figures for the report)
# ---------------------------------------------------------------------------
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 300, "font.family": "DejaVu Sans", "font.size": 10,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 12,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": INK2, "ytick.color": INK2, "legend.frameon": False, "figure.facecolor": "white",
    "axes.axisbelow": True,
})

PITCHER_MAJOR = ["K", "BB", "H", "HR", "IP", "R"]
HITTER_MAJOR = ["H", "HR", "TB", "BB", "K", "PA"]
EXCLUDED_TARGETS = {"SB": "Statcast pitch events carry no stolen-base rows; SB actuals are all zero"}
LABEL = {"K": "Strikeouts", "BB": "Walks", "H": "Hits", "HR": "Home runs", "IP": "Innings pitched",
         "R": "Runs allowed", "PA": "Plate appearances", "TB": "Total bases", "2B": "Doubles", "3B": "Triples"}
METRIC_RECORDS: list[dict] = []


def _rec(metric, value, model, dataset, target, n, period):
    METRIC_RECORDS.append({"metric": metric, "value": None if value is None or (isinstance(value, float) and np.isnan(value)) else float(value),
                           "model": model, "dataset": dataset, "target": target, "n": int(n),
                           "period_start": period[0], "period_end": period[1]})


def _caption(fig, text):
    """Footnote placed BELOW everything already drawn (axes, tick labels, axis
    labels, legends), so it can never overlap them; the tight save keeps it."""
    from matplotlib.text import Annotation
    from matplotlib.transforms import Bbox

    def _below(renderer):
        boxes = [a.get_tightbbox(renderer) for a in fig.axes if a.get_visible() and a.axison]
        boxes = [b for b in boxes if b is not None and b.width > 0]
        return Bbox.union(boxes) if boxes else fig.bbox

    fig.add_artist(Annotation(text, xy=(0, 0), xycoords=_below, xytext=(0, -8), textcoords="offset points",
                              ha="left", va="top", fontsize=7.5, color=MUTED, annotation_clip=False))


def _save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return f"figures/{name}"


def _period(df):
    d = pd.to_datetime(df["game_date"])
    return (str(d.min().date()), str(d.max().date()))


# ===========================================================================
# GAME MODEL
# ===========================================================================
def game_metrics(y, p) -> dict:
    y = np.asarray(y, int); p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    yhat = (p > 0.5).astype(int)
    return {"n": len(y), "accuracy": accuracy_score(y, yhat), "precision": precision_score(y, yhat, zero_division=0),
            "recall": recall_score(y, yhat, zero_division=0), "f1": f1_score(y, yhat, zero_division=0),
            "roc_auc": roc_auc_score(y, p) if len(set(y)) == 2 else np.nan,
            "log_loss": log_loss(y, p, labels=[0, 1]), "brier": brier_score_loss(y, p)}


def evaluate_game(df: pd.DataFrame, live_log: Path | None) -> dict:
    period = _period(df)
    y = df["actual_home_win"].to_numpy(int)
    chosen = df["selected_model"].iloc[0]
    models = {c[2:]: df[c].to_numpy(float) for c in df.columns if c.startswith("p_") and c != "p_selected"}
    rows = []
    for name, p in models.items():
        m = game_metrics(y, p)
        if "coin_flip" in name:            # constant 0.5 -> threshold metrics meaningless
            for k in ("accuracy", "precision", "recall", "f1", "roc_auc"):
                m[k] = np.nan
        if "home_rate" in name:            # constant probability -> AUC undefined
            m["roc_auc"] = np.nan
        m["model"] = name
        m["role"] = "selected" if name == chosen else ("baseline" if name.startswith("baseline") else "candidate")
        rows.append(m)
        for k, v in m.items():
            if k not in ("model", "role", "n"):
                _rec(k, v, name, "game_test", "home_win", m["n"], period)

    # Existing production model (v1, trained on 2025_model_data.csv) - its LIVE
    # 2026 log on exactly the same games. Genuinely out-of-sample: the v1 model
    # and its feature snapshots contain no 2026 information at all.
    v1 = None
    if live_log is not None and live_log.exists():
        log = pd.read_csv(live_log)
        log = log[log["correct"].notna() & log["game_pk"].notna()].drop_duplicates("game_pk", keep="last")
        j = df[["game_pk", "actual_home_win", "home_team"]].merge(log[["game_pk", "home_win_prob", "home_team"]],
                                                                  on="game_pk", suffixes=("", "_log"))
        j = j[j["home_team"] == j["home_team_log"]]
        if len(j) > 50:
            m = game_metrics(j["actual_home_win"], j["home_win_prob"])
            m.update(model="v1_production_live_log", role="existing production (live log)")
            rows.append(m)
            v1 = j
            for k, v in m.items():
                if k not in ("model", "role", "n"):
                    _rec(k, v, "v1_production_live_log", "game_test_overlap", "home_win", m["n"], period)
            # the selected model on the same overlapping games, for a like-for-like comparison
            sub = df[df["game_pk"].isin(j["game_pk"])]
            m2 = game_metrics(sub["actual_home_win"], sub["p_selected"])
            m2.update(model=f"{chosen} (same games as v1 log)", role="selected, overlap subset")
            rows.append(m2)
    table = pd.DataFrame(rows)
    cols = ["model", "role", "n", "accuracy", "precision", "recall", "f1", "roc_auc", "log_loss", "brier"]
    table = table[cols]
    METRICS.mkdir(parents=True, exist_ok=True)
    table.to_csv(METRICS / "game_model_comparison.csv", index=False)

    p = df["p_selected"].to_numpy(float)
    figs = {}
    figs["roc"] = fig_roc(y, p, chosen, len(y))
    figs["confusion"] = fig_confusion(y, (p > 0.5).astype(int), chosen)
    figs["prob_vs_outcome"] = fig_prob_by_outcome(y, p, chosen)
    figs["buckets"] = fig_buckets(df, chosen)
    figs["over_time"] = fig_over_time(df, chosen)
    figs["comparison"] = fig_game_comparison(table)
    return {"table": table, "chosen": chosen, "period": period, "figures": figs,
            "n": len(df), "home_win_rate": float(y.mean())}


def fig_roc(y, p, model, n):
    fpr, tpr, thr = roc_curve(y, p)
    FIGDATA.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thr}).to_csv(FIGDATA / "game_roc_curve.csv", index=False)
    auc = roc_auc_score(y, p)
    fig, ax = plt.subplots(figsize=(6, 5.4))
    ax.plot([0, 1], [0, 1], ls="--", lw=1.2, color=MUTED, label="No skill (AUC 0.500)")
    ax.plot(fpr, tpr, lw=2, color=C1, label=f"Game model (AUC {auc:.3f})")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.01); ax.set_aspect("equal")
    ax.set_xlabel("False positive rate (away wins called as home wins)")
    ax.set_ylabel("True positive rate (home wins called)")
    ax.set_title("Game model ROC curve - out-of-sample test")
    ax.legend(loc="lower right")
    _caption(fig, f"n = {n:,} test games  |  positive class = home win  |  model: {model}")
    return _save(fig, "game_roc_curve.png")


def fig_confusion(y, yhat, model):
    cm = confusion_matrix(y, yhat, labels=[0, 1])
    pd.DataFrame(cm, index=["actual_away_win", "actual_home_win"],
                 columns=["pred_away_win", "pred_home_win"]).to_csv(FIGDATA / "game_confusion_matrix.csv")
    norm = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.grid(False)
    names = [["True negative", "False positive"], ["False negative", "True positive"]]
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{names[i][j]}\n{cm[i, j]:,}\n{norm[i, j]:.1%} of row",
                    ha="center", va="center", fontsize=10, color="white" if norm[i, j] > 0.55 else INK)
    ax.set_xticks([0, 1], ["Predicted away win", "Predicted home win"])
    ax.set_yticks([0, 1], ["Actual away win", "Actual home win"])
    ax.set_title("Game model confusion matrix (threshold 0.50)")
    _caption(fig, f"n = {cm.sum():,} test games  |  model: {model}  |  shading = share of actual class")
    return _save(fig, "game_confusion_matrix.png")


def fig_prob_by_outcome(y, p, model):
    bins = np.linspace(0, 1, 41)
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.hist(p[y == 1], bins=bins, alpha=0.55, color=C1, label=f"Home team won (n={int((y == 1).sum()):,})")
    ax.hist(p[y == 0], bins=bins, alpha=0.55, color=C2, label=f"Away team won (n={int((y == 0).sum()):,})")
    ax.axvline(0.5, color=INK2, lw=1, ls=":")
    ax.set_xlim(max(0, p.min() - 0.05), min(1, p.max() + 0.05))
    ax.set_xlabel("Predicted home-win probability")
    ax.set_ylabel("Games")
    ax.set_title("Predicted probability by actual outcome")
    ax.legend(loc="upper left")
    pd.DataFrame({"p_home": p, "home_won": y}).to_csv(FIGDATA / "game_probability_by_outcome.csv", index=False)
    _caption(fig, f"Separation between the two distributions is what AUC measures  |  model: {model}")
    return _save(fig, "game_probability_by_outcome.png")


def fig_buckets(df, model):
    """Bucket by the probability of the PICKED side (max(p, 1-p))."""
    p = df["p_selected"].to_numpy(float)
    conf = np.maximum(p, 1 - p)
    picked_won = np.where(p > 0.5, df["actual_home_win"], 1 - df["actual_home_win"]).astype(int)
    edges = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 1.0001]
    labels = ["50-55%", "55-60%", "60-65%", "65-70%", "70-75%", "75%+"]
    b = pd.cut(conf, edges, labels=labels, right=False)
    t = pd.DataFrame({"bucket": b, "conf": conf, "won": picked_won}).groupby("bucket", observed=False).agg(
        n=("won", "size"), avg_predicted=("conf", "mean"), actual_win_pct=("won", "mean")).reset_index()
    t["accuracy"] = t["actual_win_pct"]
    t["predicted_minus_actual"] = t["avg_predicted"] - t["actual_win_pct"]
    t["se"] = np.sqrt(t["actual_win_pct"] * (1 - t["actual_win_pct"]) / t["n"].clip(lower=1))
    t.to_csv(METRICS / "game_probability_buckets.csv", index=False)
    t.to_csv(FIGDATA / "game_probability_buckets.csv", index=False)
    tt = t[t["n"] >= 10]
    small = t[(t["n"] > 0) & (t["n"] < 10)]
    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.plot([0.5, 0.8], [0.5, 0.8], ls="--", lw=1.2, color=MUTED, label="Predicted = actual")
    ax.errorbar(tt["avg_predicted"], tt["actual_win_pct"], yerr=1.96 * tt["se"], fmt="o", color=C1, ms=7,
                capsize=3, label="Bucket (95% CI)")
    # bucket labels sit under the x-axis at each bucket's x position (never on the
    # error bars or the diagonal), staggered in two rows when buckets are close
    xs = tt["avg_predicted"].to_numpy(float)
    for i, (_, r) in enumerate(tt.iterrows()):
        close = i > 0 and xs[i] - xs[i - 1] < 0.045
        row = 1 if close and not getattr(ax, "_last_row", 0) else 0
        ax._last_row = row
        ax.annotate(f"{r['bucket']}  n={int(r['n'])}", (r["avg_predicted"], 0), xycoords=("data", "axes fraction"),
                    xytext=(0, 6 + 13 * row), textcoords="offset points", ha="center", va="bottom", fontsize=7.5,
                    color=INK2, bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.9))
    ax.set_xlim(0.48, 0.82)
    lo = max(0.0, float((tt["actual_win_pct"] - 1.96 * tt["se"]).min()) - 0.12)   # room for the labels
    hi = min(1.0, float((tt["actual_win_pct"] + 1.96 * tt["se"]).max()) + 0.08)
    ax.set_ylim(min(lo, 0.45), max(hi, 0.8))
    ax.set_xlabel("Average predicted win probability of the picked team")
    ax.set_ylabel("Picked team's actual win %")
    ax.set_title("Probability buckets: predicted vs actual win %")
    ax.legend(loc="upper left")
    note = ("  |  not plotted (n<10): " + ", ".join(f"{r['bucket']} n={int(r['n'])}" for _, r in small.iterrows())) if len(small) else ""
    _caption(fig, f"Buckets by confidence in the pick, max(p, 1-p)  |  model: {model}{note}")
    return _save(fig, "game_probability_buckets.png")


def fig_over_time(df, model, min_games=100):
    d = df.copy()
    d["month"] = pd.to_datetime(d["game_date"]).dt.to_period("M").astype(str)
    rows = []
    for mth, g in d.groupby("month"):
        m = game_metrics(g["actual_home_win"], g["p_selected"])
        rows.append({"month": mth, **m})
    t = pd.DataFrame(rows)
    t.to_csv(METRICS / "game_performance_by_month.csv", index=False)
    t.to_csv(FIGDATA / "game_performance_by_month.csv", index=False)
    t = t[t["n"] >= min_games]
    if len(t) < 2:
        return None
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    for ax, (k, lab, ref) in zip(axes, [("accuracy", "Accuracy", 0.5), ("brier", "Brier score", 0.25), ("roc_auc", "ROC AUC", 0.5)]):
        ax.plot(t["month"], t[k], "o-", color=C1, lw=2, ms=6)
        ax.axhline(ref, color=MUTED, ls="--", lw=1)
        ax.set_title(lab)
        vals = np.r_[t[k].to_numpy(float), ref]       # exact values are in the by-month table
        pad = 0.15 * (vals.max() - vals.min() or 0.01)
        ax.set_ylim(vals.min() - pad, vals.max() + pad)
        ax.margins(x=0.08)
    fig.suptitle("Game model performance by month (test window)", x=0.01, ha="left", fontweight="bold", color=INK)
    _caption(fig, "Dashed line = no-skill reference (0.5 accuracy / 0.25 Brier / 0.5 AUC); months with <100 games omitted  |  model: " + model)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    return _save(fig, "game_performance_over_time.png")


def fig_game_comparison(table):
    t = table[~table["model"].str.contains("same games")].dropna(subset=["log_loss"]).sort_values("log_loss")
    fig, ax = plt.subplots(figsize=(7.5, 0.45 * len(t) + 1.4))
    colors = [C1 if r == "selected" else (C2 if "production" in str(r) else (MUTED if r == "baseline" else "#9fc3ee")) for r in t["role"]]
    ax.barh(t["model"], t["log_loss"], color=colors, height=0.6)
    ax.axvline(np.log(2), color=INK2, ls="--", lw=1)
    # values in one column to the right of every bar AND the coin-flip line
    xcol = max(float(t["log_loss"].max()), np.log(2)) + 0.002
    for i, (v, n) in enumerate(zip(t["log_loss"], t["n"])):
        ax.text(xcol, i, f"{v:.4f}  (n={n})", va="center", ha="left", fontsize=8, color=INK2)
    ax.set_xlim(min(0.66, t["log_loss"].min() - 0.01), xcol + 0.022)
    ax.invert_yaxis()
    ax.set_xlabel("Test log loss (lower is better)")
    ax.set_title("Game model vs candidates and baselines")
    _caption(fig, "Blue = selected on validation; orange = existing production model's live 2026 log; grey = baselines; "
                  "dashed line = coin flip (0.693)")
    return _save(fig, "game_model_comparison.png")


# ===========================================================================
# PLAYER MODELS
# ===========================================================================
def reg_metrics(y, p) -> dict:
    y = np.asarray(y, float); p = np.asarray(p, float)
    e = p - y
    ss_res = float(np.sum(e ** 2)); ss_tot = float(np.sum((y - y.mean()) ** 2))
    pc = np.clip(p, 1e-9, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        dev = 2 * np.where(y > 0, y * np.log(y / pc), 0.0) - 2 * (y - pc)
    return {"n": len(y), "mean_error": float(e.mean()), "median_error": float(np.median(e)),
            "mae": float(np.abs(e).mean()), "rmse": float(np.sqrt((e ** 2).mean())),
            "r2": 1 - ss_res / ss_tot if ss_tot > 0 else np.nan,
            "poisson_deviance": float(np.mean(dev)),
            "corr": float(np.corrcoef(y, p)[0, 1]) if np.std(p) > 0 and np.std(y) > 0 else np.nan,
            "actual_mean": float(y.mean()), "pred_mean": float(p.mean())}


VARIANTS = [("pred_final", "Production blend (published)"), ("pred_direct_selected", "Direct count model"),
            ("pred_rate_x_opportunity", "Rate x predicted opportunity"), ("pred_rf", "Random forest (train-window fit)"),
            ("pred_xgb", "XGBoost (train-window fit)"),
            ("pred_lin", "Linear: Poisson regression (train-window fit)"),
            ("pred_stack_even", "Stack: even weights (train-window fit)"),
            ("pred_stack_opt", "Stack: optimized weights (train-window fit)"),
            ("baseline_season_avg", "Baseline: history avg (_std)"), ("baseline_last10_avg", "Baseline: last-10 avg"),
            ("baseline_league_avg", "Baseline: league avg")]


def evaluate_players(df: pd.DataFrame, group: str) -> dict:
    period = _period(df)
    major = PITCHER_MAJOR if group == "pitcher" else HITTER_MAJOR
    rows = []
    for t, g in df.groupby("target"):
        if t in EXCLUDED_TARGETS:
            continue
        for col, lab in VARIANTS:
            if col not in g.columns or g[col].isna().all():
                continue
            gg = g[g[col].notna()]
            m = reg_metrics(gg["actual"], gg[col])
            m.update(group=group, target=t, variant=lab, column=col)
            rows.append(m)
            for k in ("mae", "rmse", "r2", "mean_error", "median_error", "poisson_deviance"):
                _rec(k, m[k], f"{group}_{t}:{col}", f"{group}_test", t, m["n"], period)
    table = pd.DataFrame(rows)
    # skill vs the strongest simple baseline (lower MAE is better)
    best_base = (table[table["column"].str.startswith("baseline")].groupby("target")["mae"].min().rename("best_baseline_mae"))
    table = table.merge(best_base, on="target", how="left")
    table["mae_skill_vs_best_baseline"] = 1 - table["mae"] / table["best_baseline_mae"]
    table.to_csv(METRICS / f"{group}_model_comparison.csv", index=False)

    figs = {}
    final = df[~df["target"].isin(EXCLUDED_TARGETS)]
    figs["pred_vs_actual"] = {t: fig_pred_vs_actual(final[final["target"] == t], group, t) for t in major if t in set(final["target"])}
    figs["residuals"] = fig_residuals(final, group, major)
    figs["error_dist"] = fig_error_distribution(final, group, major)
    rng = projection_ranges(final, group, major)
    figs["ranges"] = fig_ranges(rng, group, major)
    figs["comparison"] = fig_player_comparison(table, group, major)
    figs["binned"] = fig_binned(final, group, major)
    figs["window_totals"] = fig_window_totals(final, group, major)
    return {"table": table, "ranges": rng, "figures": figs, "period": period,
            "n_player_games": int(df.drop_duplicates(["game_pk", "player_id"]).shape[0])}


def fig_pred_vs_actual(g, group, t):
    y, p = g["actual"].to_numpy(float), g["pred_final"].to_numpy(float)
    m = reg_metrics(y, p)
    rng = np.random.default_rng(0)
    discrete = t != "IP"
    xj = y + (rng.uniform(-0.18, 0.18, len(y)) if discrete else 0)
    fig, ax = plt.subplots(figsize=(6, 5.4))
    ax.scatter(xj, p, s=6, alpha=0.18 if len(y) > 3000 else 0.35, color=C1, lw=0, label="Player-game (x jittered)" if discrete else "Player-game")
    if discrete:
        agg = g.groupby("actual")["pred_final"].agg(["mean", "size"]).reset_index()
        agg = agg[agg["size"] >= 10]
        ax.plot(agg["actual"], agg["mean"], "o-", color=C2, lw=2, ms=6, label="Mean prediction at each actual")
    lim = (min(y.min(), p.min()) - 0.3, max(np.quantile(y, 0.999), p.max()) + 0.5)
    ax.plot(lim, lim, ls="--", color=INK2, lw=1.2, label="Perfect prediction (45°)")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel(f"Actual {LABEL.get(t, t).lower()}")
    ax.set_ylabel(f"Predicted {LABEL.get(t, t).lower()}")
    ax.set_title(f"{group.title()} {LABEL.get(t, t)}: predicted vs actual")
    ax.legend(loc="upper left", fontsize=8)
    _caption(fig, f"n = {m['n']:,}  |  MAE {m['mae']:.3f}  |  RMSE {m['rmse']:.3f}  |  R² {m['r2']:.3f}  |  "
                  f"Poisson dev {m['poisson_deviance']:.3f}  |  production blend, out-of-sample")
    return _save(fig, f"{group}_pred_vs_actual_{t}.png")


def fig_residuals(df, group, major, n_bins=10):
    targets = [t for t in major if t in set(df["target"])]
    ncol = 3; nrow = int(np.ceil(len(targets) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(12, 3.6 * nrow), squeeze=False)
    rows = []
    for ax, t in zip(axes.ravel(), targets):
        g = df[df["target"] == t]
        p, r = g["pred_final"].to_numpy(float), g["error"].to_numpy(float)
        ax.scatter(p, r, s=4, alpha=0.12, color=C1, lw=0)
        q = pd.qcut(p, n_bins, duplicates="drop")
        b = pd.DataFrame({"p": p, "r": r, "q": q}).groupby("q", observed=True).agg(
            pred_mean=("p", "mean"), resid_mean=("r", "mean"), resid_sd=("r", "std"), n=("r", "size")).reset_index(drop=True)
        b["target"] = t
        rows.append(b)
        ax.plot(b["pred_mean"], b["resid_mean"], "o-", color=C2, lw=2, ms=5, label="Binned mean residual")
        ax.axhline(0, color=INK2, lw=1, ls="--")
        lo, hi = np.quantile(r, [0.005, 0.995])
        ax.set_ylim(lo - 0.2, hi + 0.2)
        ax.set_title(LABEL.get(t, t), fontsize=10)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Residual (pred - actual)")
    for ax in axes.ravel()[len(targets):]:
        ax.axis("off")
    axes.ravel()[0].legend(loc="upper left", fontsize=8)
    fig.suptitle(f"{group.title()} residuals vs predicted (residual = predicted - actual)", x=0.01, ha="left",
                 fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    pd.concat(rows).to_csv(FIGDATA / f"{group}_residual_bins.csv", index=False)
    return _save(fig, f"{group}_residuals.png")


def fig_error_distribution(df, group, major):
    targets = [t for t in major if t in set(df["target"])]
    ncol = 3; nrow = int(np.ceil(len(targets) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(12, 3.3 * nrow), squeeze=False)
    for ax, t in zip(axes.ravel(), targets):
        e = df.loc[df["target"] == t, "error"].to_numpy(float)
        lo, hi = np.quantile(e, [0.002, 0.998])
        ax.hist(e, bins=np.linspace(lo, hi, 45), color=C1, alpha=0.8)
        ax.axvline(0, color=INK2, lw=1, ls="--")
        ax.axvline(e.mean(), color=C2, lw=2, label="Mean error")
        ax.axvline(np.median(e), color=C3, lw=2, label="Median error")
        # values go in the panel title, so no legend box sits on the bars or lines
        ax.set_title(f"{LABEL.get(t, t)}   mean {e.mean():+.2f} | median {np.median(e):+.2f}", fontsize=10)
        ax.set_xlabel("Error (pred - actual)")
    for ax in axes.ravel()[len(targets):]:
        ax.axis("off")
    h, l = axes.ravel()[0].get_legend_handles_labels()
    fig.suptitle(f"{group.title()} error distributions (out-of-sample)", x=0.01, ha="left", fontweight="bold", color=INK)
    fig.legend(h, l, loc="upper right", ncol=2, fontsize=9, frameon=False, bbox_to_anchor=(0.99, 0.995))
    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    return _save(fig, f"{group}_error_distributions.png")


def projection_ranges(df, group, major):
    rows = []
    for t in major:
        g = df[df["target"] == t]
        if g.empty:
            continue
        try:
            q = pd.qcut(g["pred_final"], 3, labels=["Low", "Middle", "High"])
        except ValueError:
            continue
        edges = g.groupby(q, observed=True)["pred_final"].agg(["min", "max"])
        for lab, gg in g.groupby(q, observed=True):
            rows.append({"group": group, "target": t, "projection_range": str(lab),
                         "range_min": float(edges.loc[lab, "min"]), "range_max": float(edges.loc[lab, "max"]),
                         "n": len(gg), "mean_predicted": gg["pred_final"].mean(), "mean_actual": gg["actual"].mean(),
                         "mae": gg["abs_error"].mean(), "mean_error": gg["error"].mean()})
    t = pd.DataFrame(rows)
    t.to_csv(METRICS / f"{group}_projection_ranges.csv", index=False)
    return t


def fig_ranges(rng, group, major):
    targets = [t for t in major if t in set(rng["target"])]
    ncol = 3; nrow = int(np.ceil(len(targets) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(12, 3.6 * nrow), squeeze=False)
    for ax, t in zip(axes.ravel(), targets):
        g = rng[rng["target"] == t]
        x = np.arange(len(g))
        ax.bar(x - 0.18, g["mean_predicted"], 0.34, color=C1, label="Mean predicted")
        ax.bar(x + 0.18, g["mean_actual"], 0.34, color=C2, label="Mean actual")
        # bias / MAE live in the tick labels under each group, never on the bars
        ticks = [f"{r}\nbias {me:+.2f}\nMAE {mae:.2f}" for r, me, mae in
                 zip(g["projection_range"], g["mean_error"], g["mae"])]
        ax.set_xticks(x, ticks, fontsize=8); ax.set_title(LABEL.get(t, t), fontsize=10)
        ax.set_ylim(0, max(g["mean_predicted"].max(), g["mean_actual"].max()) * 1.12)
    for ax in axes.ravel()[len(targets):]:
        ax.axis("off")
    for ax in axes.ravel():
        ax.grid(axis="x", visible=False)
    h, l = axes.ravel()[0].get_legend_handles_labels()
    fig.suptitle(f"{group.title()} accuracy by projection range (terciles of the prediction)", x=0.01, ha="left",
                 fontweight="bold", color=INK)
    fig.legend(h, l, loc="upper right", ncol=2, fontsize=9, frameon=False, bbox_to_anchor=(0.99, 0.995))
    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    return _save(fig, f"{group}_projection_ranges.png")


def fig_player_comparison(table, group, major):
    """MAE improvement over the league-average baseline, per target (positive = better)."""
    keep = ["pred_final", "pred_direct_selected", "baseline_season_avg", "baseline_last10_avg"]
    t = table[table["target"].isin(major)]
    mae = t.pivot_table(index="target", columns="column", values="mae").reindex([m for m in major if m in set(t["target"])])
    mae.to_csv(FIGDATA / f"{group}_mae_comparison.csv")
    rel = pd.DataFrame({c: 100 * (1 - mae[c] / mae["baseline_league_avg"]) for c in keep if c in mae.columns})
    rel.to_csv(FIGDATA / f"{group}_mae_improvement_vs_league_avg.csv")
    colors = {"pred_final": C1, "pred_direct_selected": "#9fc3ee", "baseline_season_avg": C2, "baseline_last10_avg": C4}
    names = dict(VARIANTS)
    fig, ax = plt.subplots(figsize=(9, 0.75 * len(rel) + 1.8))
    y = np.arange(len(rel)); h = 0.2
    for i, c in enumerate(rel.columns):
        ax.barh(y + (i - 1.5) * h, rel[c], h, color=colors[c], label=names[c])
    for yi, v in zip(y, rel["pred_final"]):
        ax.text(v + (0.3 if v >= 0 else -0.3), yi - 1.5 * h, f"{v:+.1f}%", va="center",
                ha="left" if v >= 0 else "right", fontsize=8, color=INK2)
    ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks(y, [LABEL.get(i, i) for i in rel.index]); ax.invert_yaxis()
    ax.set_xlabel("MAE improvement over league-average baseline (%) - higher is better")
    ax.set_title(f"{group.title()} models vs simple baselines")
    ax.grid(axis="y", visible=False)
    ax.legend(fontsize=8, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.14 - 0.02 * (6 - len(rel))))
    fig.tight_layout()
    return _save(fig, f"{group}_model_comparison.png")


def fig_binned(df, group, major, n_bins=10):
    """Aggregate view: projections grouped into deciles of the projection;
    mean projected vs mean actual per bin (45-degree line = perfect)."""
    targets = [t for t in major if t in set(df["target"])]
    ncol = 3; nrow = int(np.ceil(len(targets) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(12, 3.8 * nrow), squeeze=False)
    rows = []
    for ax, t in zip(axes.ravel(), targets):
        g = df[df["target"] == t]
        q = pd.qcut(g["pred_final"].rank(method="first"), n_bins, labels=False)
        b = g.groupby(q).agg(n=("actual", "size"), mean_pred=("pred_final", "mean"),
                             mean_actual=("actual", "mean"), mae=("abs_error", "mean")).reset_index(drop=True)
        b["se_actual"] = g.groupby(q)["actual"].std().to_numpy() / np.sqrt(b["n"])
        b.insert(0, "bin", range(1, len(b) + 1)); b.insert(0, "target", t)
        rows.append(b)
        lo = min(b["mean_pred"].min(), b["mean_actual"].min()); hi = max(b["mean_pred"].max(), b["mean_actual"].max())
        pad = 0.08 * (hi - lo if hi > lo else 1)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], ls="--", color=MUTED, lw=1.2)
        ax.errorbar(b["mean_pred"], b["mean_actual"], yerr=1.96 * b["se_actual"], fmt="o", color=C1, ms=6, capsize=3)
        ax.set_xlim(lo - pad, hi + pad); ax.set_ylim(lo - pad, hi + pad)
        ax.set_title(f"{LABEL.get(t, t)}  (n={len(g):,}, ~{len(g) // n_bins:,}/bin)", fontsize=10)
        ax.set_xlabel("Mean projected"); ax.set_ylabel("Mean actual")
    for ax in axes.ravel()[len(targets):]:
        ax.axis("off")
    fig.suptitle(f"{group.title()} projections by decile: mean projected vs mean actual (95% CI)", x=0.01,
                 ha="left", fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    pd.concat(rows).to_csv(FIGDATA / f"{group}_binned_deciles.csv", index=False)
    pd.concat(rows).to_csv(METRICS / f"{group}_binned_deciles.csv", index=False)
    return _save(fig, f"{group}_binned_pred_vs_actual.png")


def fig_window_totals(df, group, major, min_games=None):
    """Second view of the same predictions: each player's summed projection over
    the whole test window vs his actual total (players with enough games)."""
    min_games = min_games or (5 if group == "pitcher" else 20)
    targets = [t for t in major if t in set(df["target"])]
    ncol = 3; nrow = int(np.ceil(len(targets) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(12, 3.8 * nrow), squeeze=False)
    rows, summ = [], []
    for ax, t in zip(axes.ravel(), targets):
        g = df[df["target"] == t]
        name = {"player_name": ("player_name", "first")} if "player_name" in g.columns else {}
        tot = g.groupby("player_id").agg(**name, games=("actual", "size"),
                                          projected_total=("pred_final", "sum"), actual_total=("actual", "sum")).reset_index()
        tot = tot[tot["games"] >= min_games]
        tot["target"] = t
        rows.append(tot)
        y, p = tot["actual_total"].to_numpy(float), tot["projected_total"].to_numpy(float)
        m = reg_metrics(y, p)
        summ.append({"group": group, "target": t, "players": len(tot), "min_games": min_games,
                     "mae_total": m["mae"], "rmse_total": m["rmse"], "r2_total": m["r2"], "corr_total": m["corr"],
                     "mean_error_total": m["mean_error"]})
        _rec("r2_test_window_total", m["r2"], f"{group}_{t}:pred_final", f"{group}_test_window_totals", t, len(tot), _period(g))
        lim = (0, max(y.max(), p.max()) * 1.05)
        ax.plot(lim, lim, ls="--", color=MUTED, lw=1.2)
        ax.scatter(y, p, s=12, alpha=0.5, color=C1, lw=0)
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_title(f"{LABEL.get(t, t)}  ({len(tot)} players)  r={m['corr']:.2f}  R²={m['r2']:.2f}", fontsize=10)
        ax.set_xlabel("Actual total"); ax.set_ylabel("Projected total")
    for ax in axes.ravel()[len(targets):]:
        ax.axis("off")
    fig.suptitle(f"{group.title()} test-window totals per player (min {min_games} games): projected vs actual",
                 x=0.01, ha="left", fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    pd.concat(rows).to_csv(FIGDATA / f"{group}_test_window_totals.csv", index=False)
    pd.DataFrame(summ).to_csv(METRICS / f"{group}_test_window_totals.csv", index=False)
    return _save(fig, f"{group}_test_window_totals.png")


COMPARE_DIR = EVAL / "comparison" / "original_methodology"


def compare_original(group: str, updated: pd.DataFrame) -> pd.DataFrame | None:
    """Original methodology vs updated pipeline on the SAME test player-games
    (matched on player + date; doubleheader dates dropped because the original
    tables merge both games into one row)."""
    p = COMPARE_DIR / "data" / f"{group}_test_predictions.csv"
    if not p.exists():
        return None
    orig = pd.read_csv(p)
    keys = ["player_id", "game_date", "target"]
    for d in (orig, updated):
        d["game_date"] = pd.to_datetime(d["game_date"]).dt.strftime("%Y-%m-%d")
    dh = updated.groupby(["player_id", "game_date", "target"])["game_pk"].transform("size") > 1
    u = updated[~dh]
    j = u.merge(orig[keys + ["actual", "pred_final", "pred_direct_selected"]], on=keys, suffixes=("", "_orig"))
    j = j[np.isclose(j["actual"], j["actual_orig"])]          # same game outcome on both sides
    rows = []
    for t, g in j.groupby("target"):
        if t in EXCLUDED_TARGETS:
            continue
        mo, mu = reg_metrics(g["actual"], g["pred_final_orig"]), reg_metrics(g["actual"], g["pred_final"])
        rows.append({"group": group, "target": t, "n_matched": len(g),
                     "mae_original": mo["mae"], "mae_updated": mu["mae"],
                     "rmse_original": mo["rmse"], "rmse_updated": mu["rmse"],
                     "r2_original": mo["r2"], "r2_updated": mu["r2"],
                     "bias_original": mo["mean_error"], "bias_updated": mu["mean_error"],
                     "poisson_dev_original": mo["poisson_deviance"], "poisson_dev_updated": mu["poisson_deviance"]})
    t = pd.DataFrame(rows)
    t["mae_change_pct"] = 100 * (t["mae_updated"] / t["mae_original"] - 1)
    t.to_csv(METRICS / f"{group}_original_vs_updated.csv", index=False)
    return t


LINEUP_BASE_DIR = EVAL / "comparison" / "without_lineup"


def compare_lineup(group: str, updated: pd.DataFrame) -> pd.DataFrame | None:
    """Controlled experiment: the same pipeline, data, protocol and test games
    with vs without the lineup (batting-order) features. Matched on game_pk +
    player + target."""
    p = LINEUP_BASE_DIR / "data" / f"{group}_test_predictions.csv"
    if not p.exists():
        return None
    base = pd.read_csv(p)
    keys = ["game_pk", "player_id", "target"]
    # Same model family on both sides (rf/xgb/nn exist in both runs), so the
    # difference is the lineup features only - not the families/stacks added
    # since the baseline was saved. pred_final is reported too, labelled.
    cols = [c for c in ["pred_rf", "pred_xgb", "pred_final"] if c in base.columns and c in updated.columns]
    j = updated.merge(base[keys + ["actual"] + cols], on=keys, suffixes=("", "_base"))
    j = j[np.isclose(j["actual"], j["actual_base"])]
    rows = []
    for t, c in [(t, c) for t in sorted(j["target"].unique()) for c in cols]:
        g = j[j["target"] == t]
        if t in EXCLUDED_TARGETS:
            continue
        mb, ml = reg_metrics(g["actual"], g[f"{c}_base"]), reg_metrics(g["actual"], g[c])
        rows.append({"group": group, "target": t, "model": {"pred_rf": "random forest", "pred_xgb": "XGBoost",
                     "pred_final": "published blend*"}[c], "n_matched": len(g),
                     "mae_without_lineup": mb["mae"], "mae_with_lineup": ml["mae"],
                     "rmse_without_lineup": mb["rmse"], "rmse_with_lineup": ml["rmse"],
                     "poisson_dev_without_lineup": mb["poisson_deviance"], "poisson_dev_with_lineup": ml["poisson_deviance"],
                     "bias_without_lineup": mb["mean_error"], "bias_with_lineup": ml["mean_error"]})
    t = pd.DataFrame(rows)
    if t.empty:
        return None
    t["mae_change_pct"] = 100 * (t["mae_with_lineup"] / t["mae_without_lineup"] - 1)
    t.to_csv(METRICS / f"{group}_lineup_experiment.csv", index=False)
    return t


# ===========================================================================
def run(write_report: bool = True) -> dict:
    METRICS.mkdir(parents=True, exist_ok=True); FIGDATA.mkdir(parents=True, exist_ok=True)
    rep = qc.Report()
    meta = qc.check_backtest_metadata(METRICS / "backtest_run.json",
                                      [METRICS / "pitcher_model_selection.csv", METRICS / "hitter_model_selection.csv"], rep)
    test_start = (meta or {}).get("test_start")
    out = {"generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}

    gpath = DATA / "game_test_predictions.csv"
    game = pd.read_csv(gpath) if gpath.exists() else None
    if game is not None:
        sel = pd.read_csv(METRICS / "game_model_selection.csv") if (METRICS / "game_model_selection.csv").exists() else None
        qc.check_game_predictions(game, str(pd.to_datetime(game["game_date"]).min().date()), rep)
    players = {}
    for grp in ("pitcher", "hitter"):
        p = DATA / f"{grp}_test_predictions.csv"
        if p.exists():
            players[grp] = pd.read_csv(p)
            qc.check_player_predictions(players[grp], grp, test_start, rep)
    qc.check_feature_tables(HERE / "data" / "features", rep)
    rep.to_frame().to_csv(METRICS / "quality_checks.csv", index=False)
    print(rep.to_frame().to_string())
    rep.raise_if_failed()

    if game is not None:
        out["game"] = evaluate_game(game, HERE / "2026_picks_accuracy.csv")
        print(out["game"]["table"].to_string())
    for grp, df in players.items():
        out[grp] = evaluate_players(df, grp)
        cmp = compare_original(grp, df.copy())
        if cmp is not None:
            out[grp]["original_vs_updated"] = cmp
            print(cmp.to_string())
        lx = compare_lineup(grp, df.copy())
        if lx is not None:
            out[grp]["lineup_experiment"] = lx
            print(lx.to_string())
        print(out[grp]["table"][out[grp]["table"]["column"].isin(["pred_final", "baseline_season_avg"])]
              [["target", "variant", "n", "mae", "rmse", "r2", "mean_error"]].to_string())

    long = pd.DataFrame(METRIC_RECORDS)
    long.to_csv(METRICS / "metrics_long.csv", index=False)
    (METRICS / "metrics.json").write_text(json.dumps({"generated_utc": out["generated_utc"],
                                                      "records": METRIC_RECORDS}, indent=1))
    manifest = {"generated_utc": out["generated_utc"], "excluded_targets": EXCLUDED_TARGETS,
                "figures": {k: v.get("figures") for k, v in out.items() if isinstance(v, dict) and "figures" in v}}
    (METRICS / "figure_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    if write_report:
        import generate_report
        generate_report.main([])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-report", action="store_true")
    a = ap.parse_args()
    run(write_report=not a.no_report)


if __name__ == "__main__":
    main()
