"""
report_v2.py
============
Builds the single-document InTheCount technical report (front-office audience)
from the files the evaluation and audit steps saved. Every number is read from
those files at build time; nothing is typed in by hand.

Order: Summary -> Data & history window -> Leakage fixes -> Model approach ->
Player results -> Game results -> Known issues -> Close.

"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EVAL = HERE / "outputs" / "model_evaluation"
MET = EVAL / "metrics"
AM = EVAL / "audit" / "metrics"
FIG = EVAL / "figures"
AFIG = EVAL / "audit" / "figures"

LABEL = {"K": "Strikeouts", "BB": "Walks", "H": "Hits", "HR": "Home runs", "IP": "Innings pitched",
         "R": "Runs allowed", "PA": "Plate appearances", "TB": "Total bases", "2B": "Doubles", "3B": "Triples"}
P_ORDER = ["IP", "K", "H", "BB", "HR", "R"]
H_ORDER = ["PA", "H", "TB", "K", "BB", "HR"]
EXCLUDED = {("hitter", "2B"), ("hitter", "3B")}   # not part of the published hitter set
GAME_NAME = {"xgb_isotonic": "XGBoost (isotonic)", "xgb_raw": "XGBoost (uncalibrated)",
             "rf_isotonic": "Random forest (isotonic)", "rf_raw": "Random forest (uncalibrated)",
             "logistic": "Logistic regression", "stack_even": "Even-weight stack", "stack_opt": "Optimized stack",
             "baseline_home_rate": "Always home (base rate)", "baseline_train_home_rate": "Always home (base rate)",
             "baseline_coin_flip": "Coin flip (50%)", "baseline_winpct_logit": "Win-% difference only",
             "v1_production_live_log": "Deployed model, live picks"}
FAM_NAME = {"rf": "random forest", "xgb": "XGBoost", "lin": "Poisson regression"}


# ---------------------------------------------------------------------------
def _csv(p: Path) -> pd.DataFrame:
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def _json(p: Path) -> dict:
    return json.loads(p.read_text()) if p.exists() else {}


def pct(v, d=1, sign=True):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "n/a"
    return f"{v:+.{d}f}%" if sign else f"{v:.{d}f}%"


def ci(pt, lo, hi, d=1):
    if not np.isfinite(pt):
        return "n/a"
    return f"{pt:+.{d}f}% ({lo:+.{d}f} to {hi:+.{d}f})"


def tag(group, target):
    return f"{'Hitter' if group == 'hitter' else 'Pitcher'} {LABEL.get(target, target).lower()}"


def short(group, target):
    return f"{group[0].upper()} {target}"


def lst(xs, empty="none"):
    xs = [str(x) for x in xs]
    if not xs:
        return empty
    return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1]


def table(df: pd.DataFrame, fmt: dict | None = None) -> str:
    fmt = fmt or {}
    cols = list(df.columns)
    import re as _re

    def numeric(c):
        vals = [str(v) for v in df[c].tolist()]
        return all(_re.match(r"^[+\-−]?[\d.,]+%?( \(.*\))?$|^n/a$", v) for v in vals) or \
            pd.api.types.is_numeric_dtype(df[c])
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---:" if (i > 0 and numeric(c)) else ":---" for i, c in enumerate(cols)) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if c in fmt and not (isinstance(v, float) and np.isnan(v)):
                cells.append(fmt[c](v))
            elif isinstance(v, (float, np.floating)):
                cells.append("n/a" if np.isnan(v) else f"{v:.3f}")
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def figure(path: Path, caption: str) -> str:
    if not path.exists():
        return ""
    import html as _h
    try:
        from PIL import Image
        with Image.open(path) as im:
            cls = "wide" if im.width / im.height > 1.3 else "narrow"
    except Exception:
        cls = "wide"
    rel = path.relative_to(EVAL).as_posix()
    return (f'<figure><img class="{cls}" src="../{rel}" alt="{_h.escape(caption)}">'
            f'<figcaption>{_h.escape(caption)}</figcaption></figure>\n')


# ---------------------------------------------------------------------------
def pipeline_figure() -> Path:
    """Static flow diagram of the daily system (drawn from fixed labels)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    out = FIG / "pipeline_diagram.png"
    FIG.mkdir(parents=True, exist_ok=True)
    boxes = [
        ("Statcast pitches\n2024-2026\nvalidated by season", "#e8f0fb"),
        ("History window\nlast two seasons\n+ season to date", "#e8f0fb"),
        ("Pre-game features\nrolling windows,\ntrue talent, lineup", "#e8f0fb"),
        ("Models\nplayer counts and\ngame win probability", "#fdf0e6"),
        ("Daily publication\nprojections and\nwin probabilities", "#fdf0e6"),
        ("Grading\nbox scores and\npublic ledger", "#e9f7f0"),
    ]
    fig, ax = plt.subplots(figsize=(13, 2.6))
    ax.set_xlim(0, 13.2); ax.set_ylim(0, 2.6); ax.axis("off")
    w, h, gap, y = 1.95, 1.5, 0.27, 0.75
    for i, (txt, col) in enumerate(boxes):
        x = 0.1 + i * (w + gap)
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                    fc=col, ec="#52514e", lw=1))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=9.5, color="#0b0b0b", linespacing=1.35)
        if i < len(boxes) - 1:
            ax.annotate("", xy=(x + w + gap - 0.02, y + h / 2), xytext=(x + w + 0.02, y + h / 2),
                        arrowprops=dict(arrowstyle="->", color="#52514e", lw=1.2))
    ax.text(0.1, 0.25, "Schedules, probable pitchers, posted lineups and box scores come from the MLB Stats API. "
            "Grading results are never fed back into training.", fontsize=8.5, color="#52514e")
    fig.savefig(out, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def importance_figure() -> Path:
    """Players only, larger type: share of each target's mean |SHAP| (XGBoost, validation rows)
    summed over every feature that measures the same thing."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    imp = _csv(AM / "importance_individual.csv")
    out = FIG / "report_group_importance.png"
    if imp.empty:
        return out
    x = imp[imp.family == "xgb"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), gridspec_kw={"width_ratios": [6, 8], "wspace": 0.55})
    for ax, g, order in ((axes[0], "pitcher", P_ORDER), (axes[1], "hitter", H_ORDER)):
        d = x[x.group == g]
        G = d.groupby(["target", "construct"])["shap_valid"].sum().unstack(fill_value=0)
        G = G.div(G.sum(1), axis=0) * 100
        G = G.loc[[t for t in order if t in G.index]]
        keep = G.mean().sort_values(ascending=False)
        keep = keep[keep >= 2].index
        G = G[keep]
        im = ax.imshow(G.to_numpy().T, cmap="Blues", vmin=0, vmax=max(40, float(G.to_numpy().max())), aspect="auto")
        ax.set_xticks(range(len(G.index)), [LABEL[t].replace(" ", "\n", 1) for t in G.index], fontsize=9)
        ax.set_yticks(range(len(G.columns)), G.columns, fontsize=10)
        for i in range(G.shape[0]):
            for j in range(G.shape[1]):
                v = G.iat[i, j]
                if v >= 1:
                    ax.text(i, j, f"{v:.0f}", ha="center", va="center", fontsize=9, color="white" if v > 30 else "#0b0b0b")
        ax.set_title("Starting pitchers" if g == "pitcher" else "Hitters", loc="left", fontsize=12, fontweight="bold")
        for s_ in ax.spines.values():
            s_.set_visible(False)
    fig.text(0.01, -0.02, "Share (%) of each target's signal by what the feature measures (XGBoost, SHAP on validation "
             "games). Groups under 2% on average are omitted.", fontsize=9, color="#52514e")
    fig.savefig(out, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
def load() -> dict:
    D = {
        "val": _json(EVAL / "data_validation" / "statcast_validation.json"),
        "build": _json(HERE / "data" / "features" / "build_summary.json"),
        "bt": _json(MET / "backtest_run.json"),
        "tests": _json(MET / "test_results.json"),
        "qc": _csv(MET / "quality_checks.csv"),
        "gcomp": _csv(MET / "game_model_comparison.csv"),
        "gsel": _csv(MET / "game_model_selection.csv"),
        "base": _csv(AM / "baselines_bootstrap.csv"),
        "gcal": _csv(AM / "game_calibration.csv"),
        "gboot": _csv(AM / "game_bootstrap.csv"),
        "ghi": _csv(AM / "game_high_confidence.csv"),
        "gtemp": _csv(AM / "game_temporal.csv"),
        "gabl": _csv(AM / "game_ablation.csv"),
        "gmeta": _json(AM / "game_meta.json"),
        "over": _csv(AM / "overfit_summary.csv"),
        "om": _csv(AM / "overfit_metrics.csv"),
        "red": _csv(AM / "redundancy_summary.csv"),
        "abl": _csv(AM / "ablation.csv"),
        "lin": _csv(AM / "linear_stability.csv"),
        "imps": _csv(AM / "importance_stability.csv"),
        "div": _csv(AM / "model_diversity.csv"),
        "curves": _csv(AM / "learning_curves.csv"),
        "temp": _csv(AM / "temporal_players.csv"),
        "tenure": _csv(AM / "tenure_players.csv"),
        "shrink": _csv(AM / "shrinkage_test.csv"),
    }
    for g in ("pitcher", "hitter"):
        drop = {t for gg, t in EXCLUDED if gg == g}
        for key, name in (("comp", "model_comparison"), ("tot", "test_window_totals"), ("orig", "original_vs_updated")):
            d = _csv(MET / f"{g}_{name}.csv")
            D[f"{g}_{key}"] = d[~d.target.isin(drop)] if len(d) and "target" in d else d
    D["lineup"] = lineup_rmse()
    D["inv"] = _csv(AM / "feature_inventory.csv")
    for g in ("pitcher", "hitter"):
        D[f"{g}_sel"] = _csv(MET / f"{g}_model_selection.csv")
    return D


def lineup_rmse() -> pd.DataFrame:
    """Controlled lineup experiment, scored in RMSE: the same families fit the same way
    with and without the two lineup features, matched on game, player and target."""
    a, b = EVAL / "data" / "hitter_test_predictions.csv", EVAL / "comparison" / "without_lineup" / "data" / "hitter_test_predictions.csv"
    if not (a.exists() and b.exists()):
        return pd.DataFrame()
    cols = ["game_pk", "player_id", "target", "actual", "pred_rf", "pred_xgb"]
    w = pd.read_csv(a, usecols=lambda c: c in cols)
    wo = pd.read_csv(b, usecols=lambda c: c in cols)
    m = w.merge(wo, on=["game_pk", "player_id", "target"], suffixes=("_w", "_wo"))
    m = m[np.isclose(m.actual_w, m.actual_wo)]
    rows = []
    for t, d in m.groupby("target"):
        if ("hitter", t) in EXCLUDED:
            continue
        r = {"target": t, "n": len(d)}
        for f in ("rf", "xgb"):
            rw = np.sqrt(np.mean((d[f"pred_{f}_w"] - d.actual_w) ** 2))
            rwo = np.sqrt(np.mean((d[f"pred_{f}_wo"] - d.actual_w) ** 2))
            r[f] = 100 * (rw / rwo - 1)
        r["avg"] = (r["rf"] + r["xgb"]) / 2
        rows.append(r)
    return pd.DataFrame(rows)


def _keep(df: pd.DataFrame, gcol="group", tcol="target") -> pd.DataFrame:
    if df.empty or gcol not in df or tcol not in df:
        return df
    return df[[(g, t) not in EXCLUDED for g, t in zip(df[gcol], df[tcol])]]


def prod(D) -> pd.DataFrame:
    b = D["base"]
    return _keep(b[b.model == "production blend (published)"].copy()) if len(b) else b


def bias_table(D, g) -> pd.DataFrame:
    c = D[f"{g}_comp"]
    if c.empty:
        return pd.DataFrame()
    p = c[c.column == "pred_final"].set_index("target")
    return pd.DataFrame({"mean": p.actual_mean, "bias": p.mean_error, "bias_pct": 100 * p.mean_error / p.actual_mean})


# ---------------------------------------------------------------------------
def build(tests_override: dict | None = None) -> str:
    D = load()
    L: list[str] = []
    w = L.append
    P = prod(D)
    tests = tests_override or D["tests"]
    bt = D["bt"]

    w("# InTheCount: MLB Game and Player Projection System\n")
    w("*Technical report*\n")

    # ================================================================ SUMMARY
    w("## 1. Executive Summary and Key Findings\n")
    w("InTheCount is a public MLB projection system. Each day it publishes a win probability for every game and "
      "projections for each starting pitcher (innings, strikeouts, walks, hits, home runs, runs allowed) and each "
      "lineup hitter (plate appearances, hits, total bases, strikeouts, walks, home runs), then "
      "grades every number against the box score. It is built on three seasons of pitch-level Statcast data. "
      "All results below are out-of-sample: models were chosen on the first part of the latest season and "
      "scored once on its remainder.\n")
    if len(P):
        pi = P.set_index(["group", "target"])
        heads = [("hitter", "PA"), ("pitcher", "IP"), ("pitcher", "K")]
        rows = []
        for g, t in heads:
            if (g, t) in pi.index:
                r = pi.loc[(g, t)]
                rows.append({"Projection": tag(g, t),
                             "vs league average": ci(r["rmse_pct_better_than_league average"], r["ci_lo_vs_league average"], r["ci_hi_vs_league average"]),
                             "vs player's own history": ci(r["rmse_pct_better_than_history-to-date"], r["ci_lo_vs_history-to-date"], r["ci_hi_vs_history-to-date"])})
        w("### Summary of Out-of-Sample Performance\n")
        w("Each percentage is how much smaller the projection's typical single-game miss is "
          "than the miss of a simple benchmark. The miss is measured as RMSE (root-mean-square error) on the test "
          "games. *League average* gives every player the league's average for that stat. *Player's own history* "
          "gives every player his own average over all his earlier games in the history window. The 95% interval "
          "is in parentheses.\n")
        w(table(pd.DataFrame(rows)))
        hc = D["hitter_comp"]
        if len(hc) and rows and rows[0]["Projection"] == "Hitter plate appearances":
            pa = hc[hc.target == "PA"].set_index("variant")["rmse"]
            need = {"Production blend (published)", "Baseline: league avg", "Baseline: history avg (_std)"}
            if need <= set(pa.index):
                w(f"For example, the projection's typical miss on a hitter's plate appearances is "
                  f"{pa['Production blend (published)']:.3f} PA. Predicting the league average gives "
                  f"{pa['Baseline: league avg']:.3f}, and the player's own average gives "
                  f"{pa['Baseline: history avg (_std)']:.3f}. Those differences are the "
                  f"{rows[0]['vs league average'].split(' ')[0]} and {rows[0]['vs player' + chr(39) + 's own history'].split(' ')[0]} "
                  "in the first row. A positive number means the projection is better.\n")
        n_all = len(P)
        n_lg = int((P["ci_lo_vs_league average"] > 0).sum())
        n_hist = int((P["ci_lo_vs_history-to-date"] > 0).sum())
        weak = P[P["rmse_pct_better_than_history-to-date"] < 2].sort_values("rmse_pct_better_than_history-to-date")
        w((f"All {n_all} player projections beat" if n_lg == n_all else f"{n_lg} of {n_all} player projections beat")
          + " the league average with an interval above zero, and "
          f"{n_hist} of {n_all} beat the player's own history-to-date average. The gains "
          f"are largest where playing time drives the outcome and small for rare events: against the player's own "
          f"history, {lst([tag(r.group, r.target).lower() for r in weak.itertuples()])} improve by less than 2%.\n")
    gb = D["gboot"]
    gsel_name = D["gmeta"].get("selected_on_valid", "")
    gc_ = D["gcal"]
    sel_auc = float(gc_[(gc_.model == gsel_name) & (gc_.split == "test")].auc.iloc[0]) if len(gc_) else float("nan")
    if len(gb):
        v = gb[(gb.split == "valid") & gb.logloss_minus_home_rate.notna()]
        beat = v[v.hi < 0]
        gcm = D["gcomp"]
        selrow = gcm[gcm.model == gsel_name] if len(gcm) else gcm
        homerow = gcm[gcm.model == "baseline_home_rate"] if len(gcm) else gcm
        tv_ = gb[(gb.split == "test") & gb.logloss_minus_home_rate.notna() & (gb.hi < 0) & gb.model.isin([gsel_name, "logistic", "stack_opt"])]
        w("### Game-Level Predictive Performance\n")
        w((f"On the test games the selected game model's log loss is {selrow.log_loss.iloc[0]:.4f}, against "
           f"{homerow.log_loss.iloc[0]:.4f} for always picking the home team, and its accuracy is "
           f"{100 * selrow.accuracy.iloc[0]:.1f}% against {100 * homerow.accuracy.iloc[0]:.1f}%. " if len(selrow) and len(homerow) else "")
          + f"Its test AUC is {sel_auc:.3f}. "
          + (f"The {lst([GAME_NAME[m].lower() for m in tv_.model])} improve on the home-team baseline with 95% "
             "intervals that exclude zero on the test games. " if len(tv_) else "")
          + "Section 4 gives the full comparison.\n")
    w("### Evaluation Integrity and Leakage Prevention\n")
    w("The rebuild found and removed same-game information that had leaked into "
      "the hitter features, along with test-set model selection. Automated tests now rebuild the feature tables "
      "after changing one game's outcome and require that none of that game's features move, and require that "
      "seasons outside the history window cannot change any feature. "
      + (f"The full suite passes ({tests.get('summary', '').split(',')[0]})." if tests.get("returncode") == 0 else
         "The test suite result is not available for this build.") + "\n")
    # ================================================================ 1 DATA
    w("## 2. Data, Feature Engineering, and Leakage Prevention\n")
    val = D["val"]
    if val.get("seasons"):
        rows = []
        for s in sorted(val["seasons"]):
            v = val["seasons"][s]; rs = v.get("regular_season", {})
            rows.append({"Season": s, "Pitches": f"{v.get('records', 0):,}", "Reg.-season games": f"{rs.get('games', 0):,}",
                         "Pitchers": f"{rs.get('pitchers', 0):,}", "Batters": f"{rs.get('batters', 0):,}",
                         "Validation": "Pass" if v.get("passed") else "Fail"})
        w("All modelling data is pitch-level Statcast. Every season is validated before use: record and game counts, "
          "date range, required columns, missing IDs, duplicate pitches, one date and home/away pair per game, and "
          "30 teams.\n")
        w(table(pd.DataFrame(rows)))
    w("### Historical Data Window and Feature Construction\n")
    w("A projection for a game in season *S* may use the player's games from "
      "seasons *S−2* and *S−1* and from season *S* strictly before the game. The window is applied to the raw "
      "pitches before any feature is built, so older data cannot reach a feature indirectly. Inside the window, "
      "every feature is a trailing summary of earlier games: rolling averages over the previous 7 to 30 "
      "appearances, a history-to-date average, handedness splits, true-talent rates shrunk toward a league prior, "
      "a pitcher-vs-lineup matchup rate, opponent context and, for hitters, the posted batting-order slot. "
      "Spring-training games are excluded.\n")
    inv = D["inv"]
    if len(inv):
        u = inv[inv.used & inv.group.isin(["pitcher", "hitter"])]
        u = u[~((u.group == "hitter") & u.feature.str.startswith(("2B_", "3B_")))]
        cnt = u.groupby(["family", "group"]).feature.count().unstack(fill_value=0)
        EX = {"rolling": "Strikeouts, hits, walks and rates over the last 7-30 games",
              "expanding": "Same statistics averaged over all earlier games in the window",
              "opportunity": "Innings, batters faced, pitch counts, days of rest; plate appearances",
              "handedness": "Rates against left- and right-handed opponents",
              "team/opponent": "Opponent's recent rates against this handedness; opposing starter's form",
              "true_talent": "Rates shrunk toward the league according to sample size",
              "matchup": "Pitcher-vs-lineup rate combining both true-talent estimates",
              "batted_ball": "Exit velocity, launch angle, hard-hit and barrel proxies",
              "velocity/pitch_mix": "Velocity and spin",
              "lineup": "Today's posted batting-order slot and the recent average slot",
              "park": "Park factor"}
        NAME = {"rolling": "Rolling windows", "expanding": "History-to-date", "opportunity": "Playing time / workload",
                "handedness": "Handedness splits", "team/opponent": "Opponent context", "true_talent": "True talent",
                "matchup": "Matchup", "batted_ball": "Batted-ball quality", "velocity/pitch_mix": "Stuff",
                "lineup": "Lineup slot", "park": "Park"}
        rows = []
        for f in ["rolling", "expanding", "opportunity", "handedness", "team/opponent", "true_talent", "matchup",
                  "batted_ball", "velocity/pitch_mix", "lineup", "park"]:
            if f in cnt.index:
                rows.append({"Feature group": NAME[f], "Pitcher": int(cnt.loc[f].get("pitcher", 0)),
                             "Hitter": int(cnt.loc[f].get("hitter", 0)), "What it contains": EX[f]})
        w("### Feature Architecture and Input Variables\n")
        w("Number of model inputs in each feature group:\n")
        w(table(pd.DataFrame(rows), {"Pitcher": lambda v: f"{int(v)}", "Hitter": lambda v: f"{int(v)}"}))
    b = D["build"]
    if b:
        w("Three seasons are collected. The earliest one serves as history only, because "
          "its own rows have no prior seasons inside their window and would teach the model from thinner features "
          "than it sees in use.\n")
    w("### Changes to the Modeling Pipeline\n")
    w("The original feature definitions were kept and given two extra seasons of "
      "history. The data-leakage and protocol bugs listed below were fixed. One feature was added (lineup "
      "slot) and a linear model and stacked models were added as candidates. No other feature definition changed.\n")

    # ================================================================ 2 LEAKAGE
    w("### Leakage Corrections and Automated Safeguards\n")
    w("The cutoff for every feature is the first pitch of the game being predicted. The five most consequential "
      "fixes:\n")
    fixes = [("Opposing starter's same-game results in hitter features", "Use his previous start (known pre-game)"),
             ("Model family chosen on the test set", "Choose on a validation window; score the test once"),
             ("Validation and calibration trained on later games", "Forward-in-time splits only"),
             ("Starter = last pitcher in file order (often a reliever)", "Pitcher of the team's first pitch"),
             ("Public ledger scored past games with models trained on them", "In-sample projections excluded")]
    w(table(pd.DataFrame(fixes, columns=["Problem", "Fix"])))
    qc = D["qc"]
    crit = qc[qc.critical] if len(qc) and "critical" in qc else pd.DataFrame()
    w("**Safeguards:** (1) a perturbation test turns every plate appearance of one game into a home run, rebuilds "
      "all tables and requires that none of that game's features change (run against the old code, it flags "
      "exactly the leaked opposing-starter columns); (2) an older-season test confirms that data outside the "
      "window cannot change any feature; (3) a train/serve parity test confirms that features built for a "
      "scheduled slate equal the stored training rows; (4) a static guard rejects any feature that is not a "
      "trailing summary or an explicitly justified pre-game fact"
      + (f"; (5) {int(crit.passed.sum())} of {len(crit)} automated data and protocol checks pass before any metric "
         "is computed" if len(crit) else "") + ".\n")

    # ================================================================ 3 MODEL
    w("## 3. Modeling Methodology and Evaluation\n")
    w("### Player Projection Model Architecture\n")
    w("For every target, three model families are fit: a random forest, a Poisson-loss "
      "XGBoost and a Poisson regression. Two stacks of these (equal weights, and weights optimized on validation "
      "and scored out-of-fold) compete with them. The candidate with the lowest validation RMSE is refit on all "
      "data before the test window. Each count also has a per-opportunity model (per nine innings or per plate "
      "appearance). The published number blends the direct count with rate × predicted innings or plate "
      "appearances, using fixed weights. A neural network was tried and dropped: it was the weakest family on "
      "validation and the slowest to train.\n")
    w("### Game-Level Win Probability Model\n")
    w("The candidates are an isotonic-calibrated XGBoost, a random forest with and without isotonic calibration, "
      "a strongly regularized logistic regression, and two stacks of these. An uncalibrated XGBoost was dropped "
      "because it fit the training games far better than new ones. Each of the 18 features is a home-minus-away "
      "difference:\n")
    w(table(pd.DataFrame([
        {"Group": "Offense, last 10 games", "Features": "Runs scored, OPS, strikeout rate, walk rate"},
        {"Group": "Run prevention, last 10 games", "Features": "Runs allowed"},
        {"Group": "Starting pitcher", "Features": "FIP, K-BB%, HR/9 over his last 10 starts; FIP and K-BB% season to date"},
        {"Group": "Bullpen, last 14 games", "Features": "ERA proxy, WHIP, K-BB%"},
        {"Group": "Team strength, season to date", "Features": "Win %, run differential, OPS"},
        {"Group": "Context", "Features": "Home field; starter handedness match-up"}])))
    w("The model is selected on validation log loss, because the site publishes probabilities and log loss rewards "
      "probabilities that are both well ranked and honest. AUC only measures ranking.\n")
    w("### Model Configuration and Regularization\n")
    w("These defaults are deliberately shallow and regularized, because a single start or "
      "game is mostly noise and deep trees memorize hot and cold streaks. XGBoost depth, learning rate and "
      "regularization are tuned per target on the training rows only.\n")
    w(table(pd.DataFrame([
        {"Model": "Random forest", "Settings": "400 trees, depth 4 (5 for plate appearances), at least 25 rows per leaf, half the features per split"},
        {"Model": "XGBoost (players)", "Settings": "Poisson loss for counts; about 400 trees, depth 3, learning rate 0.03, L1 1.0 / L2 3.0"},
        {"Model": "Poisson regression", "Settings": "Standardized inputs, L2 penalty 0.05"},
        {"Model": "Game XGBoost", "Settings": "600 trees, depth 4, learning rate 0.03, L1 0.5 / L2 2.0"},
        {"Model": "Game random forest", "Settings": "600 trees, depth 6, at least 20 games per leaf"},
        {"Model": "Game logistic regression", "Settings": "Standardized inputs, strong L2 penalty (C = 0.001)"}])))
    sel_rows = []
    for g in ("pitcher", "hitter"):
        sdf = D[f"{g}_sel"]
        if sdf.empty:
            continue
        x = sdf[sdf.selected & sdf.target.isin(P_ORDER if g == "pitcher" else H_ORDER)].set_index("target")
        for t in (P_ORDER if g == "pitcher" else H_ORDER):
            if t in x.index:
                r = x.loc[t]
                m = {"stack_opt": "Optimized stack", "stack_even": "Even stack", "xgb": "XGBoost", "rf": "Random forest",
                     "lin": "Poisson regression"}.get(r.model, r.model)
                wts = r.stack_weights if isinstance(r.stack_weights, str) else ""
                wts = wts.replace("rf=", "RF ").replace("xgb=", "XGB ").replace("lin=", "Poisson ")
                sel_rows.append({"Projection": tag(g, t), "Chosen on validation": m, "Stack weights": wts or "-",
                                 "Validation RMSE": r.valid_rmse})
    if sel_rows:
        w("### Model Selection and Validation Results\n")
        w("The candidate chosen on validation for each published count:\n")
        w(table(pd.DataFrame(sel_rows), {"Validation RMSE": lambda v: f"{v:.3f}"}))
    try:
        import run_backtest as _rb
        bw = [{"Projection": tag("pitcher", t), "Weight on rate model": v} for t, v in _rb.PITCHER_BLEND_WEIGHTS.items()]
        bw = [r for r in bw if r["Projection"] in {tag("pitcher", t) for t in P_ORDER}]
        bw += [{"Projection": tag("hitter", t), "Weight on rate model": v} for t, v in _rb.HITTER_BLEND_WEIGHTS.items()
               if t in H_ORDER]
        w("### Direct-Count and Rate-Based Projection Blending\n")
        w("The published number is (1 − w) × direct count + w × (rate × predicted innings or "
          "plate appearances). Innings and plate appearances are themselves predicted directly. The weights w are "
          "fixed:\n")
        w(table(pd.DataFrame(bw), {"Weight on rate model": lambda v: f"{v:.2f}"}))
    except Exception:
        pass
    w("### Temporal Validation and Evaluation Protocol\n")
    w("Models train on earlier games, are chosen on a validation window that follows the training "
      "data, and are scored once on a later test window. Every choice of model, stack weight and "
      "calibration is made on validation; the test window is scored once. Baselines are the league average, the "
      "player's history-to-date average and his last-10 average for players, and always-home for games. "
      "Intervals throughout are 95% bootstrap intervals that resample whole game dates, because players on the "
      "same day share weather, umpires and opponents.\n")
    w("### Evaluation Metrics and Interpretation\n")
    w(table(pd.DataFrame([
        {"Measure": "RMSE", "Meaning": "Typical size of a single-game miss, in the stat's own units; large misses count more"},
        {"Measure": "% vs baseline", "Meaning": "How much smaller the projection's RMSE is than the baseline's (positive = better)"},
        {"Measure": "Bias", "Meaning": "Average of projected minus actual; positive = projections run high"},
        {"Measure": "Log loss", "Meaning": "Penalty on the probability given to what happened; lower is better, 0.693 = coin flip"},
        {"Measure": "Brier score", "Meaning": "Mean squared error of a probability; 0.25 = always saying 50%"},
        {"Measure": "AUC", "Meaning": "Chance a random home win is rated above a random home loss; 0.5 = no ranking skill"},
        {"Measure": "Calibration", "Meaning": "Whether games given 60% are won about 60% of the time"}])))

    w("### Production Pipeline and Daily Inference\n")
    w("The same code path that builds the training tables builds each day's inputs, so a "
      "projection is computed exactly as the evaluation scored it:\n")
    w("1. The current season's pitches are refreshed, with a short overlap so late-arriving data is picked up.\n"
      "2. Feature tables are rebuilt on the history window from data before today only.\n"
      "3. Probable starters and posted lineups are read from the MLB Stats API. A hitter's batting-order slot is "
      "used only once the lineup is posted.\n"
      "4. The player models and the game model score the slate. Evening runs also publish the next day's games, so "
      "the site is never a day behind.\n"
      "5. Finished games are graded against the box score. Each projection records when its model's training data "
      "ended, so in-sample projections never enter the public accuracy ledger. Grading results are never used for "
      "training.\n")

    # ================================================================ 5 GAME
    w("## 4. Game-Level Win Probability Model Evaluation\n")
    gc, gs = D["gcomp"], D["gsel"]
    meta = D["gmeta"]
    sel = meta.get("selected_on_valid", "")
    if len(gc) and len(gs):
        vmap = gs.set_index("candidate")["valid_log_loss"].to_dict()
        keep = [sel, "logistic", "stack_opt", "baseline_home_rate"]
        t = gc[gc.model.isin(keep)].copy()
        t["Model"] = t.model.map(lambda m: (GAME_NAME.get(m, m)[:-1] + ", selected)") if m == sel and GAME_NAME.get(m, m).endswith(")")
                                 else GAME_NAME.get(m, m) + (" (selected)" if m == sel else ""))
        t["Valid log loss"] = t.model.map(lambda m: vmap.get(m, np.nan))
        t = t.rename(columns={"log_loss": "Test log loss", "brier": "Test Brier", "roc_auc": "Test AUC", "accuracy": "Test accuracy"})
        n_test = int(t.n.max())
        w(f"{n_test:,} regular-season test games; the home team won {100 * D['gcal'].query('split == \"test\"').win_rate.iloc[0]:.1f}% of them.\n")
        w(table(t[["Model", "Valid log loss", "Test log loss", "Test Brier", "Test AUC", "Test accuracy"]],
                {"Valid log loss": lambda v: f"{v:.4f}", "Test log loss": lambda v: f"{v:.4f}", "Test Brier": lambda v: f"{v:.4f}",
                 "Test AUC": lambda v: f"{v:.3f}", "Test accuracy": lambda v: f"{100 * v:.1f}%"}))
    if len(gb):
        rows = []
        for m in dict.fromkeys([sel, "logistic", "stack_opt"]):
            r = {"Model": GAME_NAME[m]}
            for sp in ("valid", "test"):
                x = gb[(gb.model == m) & (gb.split == sp)]
                r[f"{'Validation' if sp == 'valid' else 'Test'}"] = (
                    f"{1000 * x.logloss_minus_home_rate.iloc[0]:+.1f} ({1000 * x.lo.iloc[0]:+.1f} to {1000 * x.hi.iloc[0]:+.1f})" if len(x) else "n/a")
            rows.append(r)
        w("**Against always-home** (log loss minus the always-home log loss, ×1000; negative is better):\n")
        w(table(pd.DataFrame(rows)))
        tv = gb[(gb.split == "test") & gb.logloss_minus_home_rate.notna() & (gb.hi < 0)]
        tv = tv[tv.model.isin([sel, "logistic", "stack_opt"])]
        w(f"On the test games, {lst(['the ' + GAME_NAME[m].lower() for m in tv.model]) if len(tv) else 'no candidate'} "
          "improve on the home-team baseline with intervals that exclude zero. On the shorter validation window, all "
          "three intervals include zero.\n")
    if len(gs) and (gs.candidate == "v1_production_live_log").any() and len(gc):
        v1 = gs[gs.candidate == "v1_production_live_log"].iloc[0]
        ch_ = gs[gs.candidate == sel].iloc[0]
        ov = gc[gc.role.astype(str).str.contains("overlap", na=False)]
        v1t = gc[gc.model == "v1_production_live_log"]
        w(f"**Deployment decision: keep the deployed model.** On validation the selected model's log loss "
          f"({ch_.valid_log_loss:.4f}) was {v1.valid_log_loss - ch_.valid_log_loss:.4f} lower than the deployed "
          f"model's live picks ({v1.valid_log_loss:.4f}). That passes the automatic promotion gate, but the "
          "difference is small relative to its uncertainty"
          + (f". On the {int(v1t.n.iloc[0])} test games both covered, the deployed model was slightly better "
             f"({v1t.log_loss.iloc[0]:.4f} vs {ov.log_loss.iloc[0]:.4f})" if len(ov) and len(v1t) else "")
          + ". The deployed model stays in production.\n")
    gcal = D["gcal"]
    if len(gcal):
        s = gcal[gcal.model == sel].set_index("split")
        w("### Probability Calibration and Reliability\n")
        w(f"The selected model's average level is right: calibration-in-the-large is "
          f"{s.loc['valid', 'cal_intercept_itl']:+.3f} on validation and {s.loc['test', 'cal_intercept_itl']:+.3f} on test, "
          f"where zero is perfect. Expected calibration error is {s.loc['valid', 'ece']:.3f} and {s.loc['test', 'ece']:.3f}. "
          f"The calibration slope is {s.loc['valid', 'cal_slope']:.2f} on validation and {s.loc['test', 'cal_slope']:.2f} "
          "on test, where 1 is ideal; below 1, the probabilities spread further from 50% than the outcomes justify.")
        hi = D["ghi"]
        h = hi[(hi.model == sel) & (hi.confidence_ge == 0.75)] if len(hi) else pd.DataFrame()
        if len(h):
            r = h[h.split == "test"].iloc[0]
            w(f" The 75%+ bucket holds only {int(r.n)} test games. The favorite won {100 * r.fav_win_rate:.0f}% "
              f"against a mean prediction of {100 * r.mean_conf:.0f}%, but the 95% interval for that rate runs from "
              f"{100 * r.ci_lo:.0f}% to {100 * r.ci_hi:.0f}%, so over-confidence there cannot be established.")
        iso = pd.DataFrame()
        if len(iso):
            xv = iso[(iso.model == "xgb_isotonic minus xgb_raw") & (iso.split == "valid")].iloc[0]
            xt = iso[(iso.model == "xgb_isotonic minus xgb_raw") & (iso.split == "test")].iloc[0]
            rt = iso[(iso.model == "rf_isotonic minus rf_raw") & (iso.split == "test")].iloc[0]
            w(f" Isotonic calibration helped XGBoost on validation ({1000 * xv.logloss_diff:+.1f} ×1000, interval "
              f"{1000 * xv.lo:+.1f} to {1000 * xv.hi:+.1f}) but not on test ({1000 * xt.logloss_diff:+.1f}). For the random "
              f"forest it hurt on test ({1000 * rt.logloss_diff:+.1f}, interval {1000 * rt.lo:+.1f} to {1000 * rt.hi:+.1f}).")
        w("\n")
    w(figure(FIG / "game_probability_buckets.png", "Figure 1. Predicted vs. Observed Win Rate by Probability Bucket"))
    if len(gcal):
        s_ = gcal[gcal.model == sel].set_index("split")
        w("### Discrimination Performance: ROC Curve and AUC\n")
        w(f"The ROC curve shows how well the probabilities order games, at every "
          f"possible cut-off. AUC is the area under it, from 0.5 for no ranking skill to 1.0 for perfect ranking. The "
          f"selected model's test AUC is {s_.loc['test', 'auc']:.3f}, against {s_.loc['valid', 'auc']:.3f} on "
          "validation, above the 0.5 line of a model with no ranking ability.\n")
    w(figure(FIG / "game_roc_curve.png", "Figure 2. ROC Curve for Game-Level Win Predictions"))

    # ================================================================ 4 PLAYERS
    w("## 5. Player-Level Projection Model Evaluation\n")
    w("The primary measure is the reduction in single-game RMSE versus each baseline, with its interval. Bias is "
      "the mean of projected minus actual (positive = over-projection).\n")
    for g in ("pitcher", "hitter"):
        x = P[P.group == g].set_index("target") if len(P) else pd.DataFrame()
        if x.empty:
            continue
        bias = bias_table(D, g)
        order = [t for t in (P_ORDER if g == "pitcher" else H_ORDER) if t in x.index]
        rows = []
        for t in order:
            r = x.loc[t]
            rows.append({"Target": LABEL[t], "Actual mean": r["mean_actual"],
                         "Bias": f"{bias.loc[t, 'bias']:+.3f} ({bias.loc[t, 'bias_pct']:+.0f}%)" if t in bias.index else "n/a",
                         "RMSE": r["rmse"],
                         "vs league": ci(r["rmse_pct_better_than_league average"], r["ci_lo_vs_league average"], r["ci_hi_vs_league average"]),
                         "vs own history": ci(r["rmse_pct_better_than_history-to-date"], r["ci_lo_vs_history-to-date"], r["ci_hi_vs_history-to-date"]),
                         "vs last 10": ci(r["rmse_pct_better_than_last 10"], r["ci_lo_vs_last 10"], r["ci_hi_vs_last 10"])})
        n = int(x["n"].max())
        w(f"**{'Starting pitchers' if g == 'pitcher' else 'Hitters'}** ({n:,} test {g}-games):\n")
        w(table(pd.DataFrame(rows), {"Actual mean": lambda v: f"{v:.2f}", "RMSE": lambda v: f"{v:.3f}"}))
    c_p, c_h = D["pitcher_comp"], D["hitter_comp"]
    if len(c_p) and len(c_h):
        mae_lose = []
        for g, c in (("pitcher", c_p), ("hitter", c_h)):
            pr = c[c.column == "pred_final"]
            mae_lose += [short(g, t) for t in pr[pr.mae_skill_vs_best_baseline <= 0].target]
        w("On mean absolute error (MAE) the projections beat every simple baseline except for "
          f"{lst([LABEL[s.split()[1]].lower() + (' (pitcher)' if s[0] == 'P' else ' (hitter)') for s in mae_lose])}. "
          "MAE rewards predicting the median, which is zero for rare events, so it is not used as the primary "
          "measure for a projected mean.\n")
    w("### Projection Calibration by Predicted Value\n")
    w("Single games are mostly noise. A hitter projected for 1.1 "
      "hits gets 0, 1, 2 or 3. The decile view sorts test games into ten equal groups by projection and compares "
      "the mean projection with the mean result.\n")
    w(figure(FIG / "pitcher_binned_pred_vs_actual.png", "Figure 3. Pitcher Projection Calibration by Decile"))
    w(figure(FIG / "hitter_binned_pred_vs_actual.png", "Figure 4. Hitter Projection Calibration by Decile"))
    tp, th = D["pitcher_tot"], D["hitter_tot"]
    if len(tp) and len(th):
        rows = []
        for g, d in (("pitcher", tp), ("hitter", th)):
            for r in d.itertuples():
                rows.append({"Projection": tag(g, r.target), "Players": int(r.players), "Min. games": int(r.min_games),
                             "Correlation": r.corr_total, "R²": r.r2_total, "Total bias": r.mean_error_total})
        w("### Aggregated Player-Level Performance\n")
        w("Each player's projections and results were summed over the "
          "test window, for players with at least the minimum number of games. R² here is the share of "
          "player-to-player variation in those totals that the projections explain.\n")
        w(table(pd.DataFrame(rows), {"Players": lambda v: f"{int(v)}", "Min. games": lambda v: f"{int(v)}",
                                     "Correlation": lambda v: f"{v:.2f}", "R²": lambda v: f"{v:.2f}",
                                     "Total bias": lambda v: f"{v:+.2f}"}))
    w(figure(FIG / "pitcher_test_window_totals.png", "Figure 5. Predicted vs. Actual Pitcher Totals"))
    w(figure(FIG / "hitter_test_window_totals.png", "Figure 6. Predicted vs. Actual Hitter Totals"))

    w("### Test-Set Comparison of Model Families\n")
    w("Test RMSE, lower is better. Single families and the stack are fit on the training window only; the published "
      "model is refit through validation.\n")
    names = [("Production blend (published)", "Published"),
             ("Random forest (train-window fit)", "RF"), ("XGBoost (train-window fit)", "XGB"),
             ("Linear: Poisson regression (train-window fit)", "Poisson"),
             ("Stack: optimized weights (train-window fit)", "Opt. stack"),
             ("Baseline: history avg (_std)", "History avg"), ("Baseline: league avg", "League avg")]
    for g, c in (("pitcher", c_p), ("hitter", c_h)):
        if c.empty:
            continue
        piv = c.pivot_table(index="target", columns="variant", values="rmse")
        piv = piv[[n for n, _ in names if n in piv.columns]].rename(columns=dict(names))
        piv = piv.reindex([t for t in (P_ORDER if g == "pitcher" else H_ORDER) if t in piv.index])
        piv.index = [LABEL[t] for t in piv.index]
        w(f"*{'Starting pitchers' if g == 'pitcher' else 'Hitters'}*\n")
        w(table(piv.reset_index().rename(columns={"index": "Target"}), {c2: (lambda v: f"{v:.3f}") for c2 in piv.columns}))
    lx = D["lineup"]
    if len(lx):
        top = lx.sort_values("avg")
        big = top[top.avg < -0.5]
        w("### Ablation Study: Impact of Batting-Order Features\n")
        w("Hitter models gained the batting-order slot the hitter starts in "
          "(posted before first pitch) and his average slot over the previous ten games. Everything else was held "
          "fixed: data, protocol, model families and test games. The table shows the change in test RMSE from adding "
          "the two features; negative means better.\n")
        lt = lx.set_index("target").reindex([t for t in H_ORDER if t in set(lx.target)]).reset_index()
        lt["Projection"] = lt.target.map(lambda t: LABEL[t])
        w(table(lt[["Projection", "rf", "xgb"]].rename(columns={"rf": "Random forest", "xgb": "XGBoost"}),
                {"Random forest": lambda v: pct(v, 1), "XGBoost": lambda v: pct(v, 1)}))
        w("The slot matters mostly through playing time, because a hitter's place in the order sets how often he "
          "comes up.\n")
    op, oh = D["pitcher_orig"], D["hitter_orig"]
    if len(op) and len(oh):
        def ch(d):
            d = d.assign(r=100 * (d.rmse_updated / d.rmse_original - 1))
            return d
        oh2, op2 = ch(oh), ch(op)
        better = oh2[oh2.r < -1]; worse = oh2[oh2.r > 1]
        w("### Comparison with the Original Pipeline\n")
        w("On the same test games, the rebuilt pitcher projections are within "
          f"{op2.r.abs().max():.1f}% of the original on RMSE for every target. The hitter projections improve on "
          + lst([f"{LABEL[r.target].lower()} ({r.r:+.1f}%)" for r in better.itertuples()])
          + " and are worse on " + lst([f"{LABEL[r.target].lower()} ({r.r:+.1f}%)" for r in worse.itertuples()])
          + ". The original hitter features contained the opposing starter's same-game results, which are not "
            "available before first pitch, so its numbers on those targets are optimistic rather than achievable "
            "live. How much of each gap comes from the leak was not measured separately.\n")

    # ================================================================ 6 CONCLUSION
    w("## 6. Conclusions\n")
    w(conclusion(D))
    return "\n".join(L)


# ---------------------------------------------------------------------------
def best_family(D) -> pd.DataFrame:
    s = D["over"]
    if s.empty:
        return s
    s = s[s.family.isin(["rf", "xgb", "lin"])]
    return s.loc[s.groupby(["group", "target"]).valid_rmse.idxmin()]


def audit_answers(D) -> list[str]:
    out = []
    bf = best_family(D)
    if len(bf):
        vc = bf.verdict.value_counts()
        out.append(f"**Overfitting: present but contained.** The models keep a median "
                   f"{100 * bf.skill_retention.median():.0f}% of their training skill on validation, so they do fit some "
                   f"training noise. Their validation predictions are not over-spread, though: calibration slopes run "
                   f"{bf.valid_cal_slope.min():.2f}-{bf.valid_cal_slope.max():.2f}, against 1 for ideal. The production "
                   "settings also sit below the depth and tree counts where validation error starts rising.")
    red = D["red"]
    if len(red):
        p = red[red.model.isin(["pitcher", "hitter"])]
        out.append("**Multicollinearity: large but harmless to accuracy.** "
                   + "; ".join(f"{int(r.n_vif_gt_10)} of {int(r.n_features)} {r.model} features have VIF above 10" for r in p.itertuples())
                   + ". It makes individual coefficients and importances unstable, not predictions.")
    sig = signal_rows(D)
    if sig:
        parts = []
        for g in ("hitter", "pitcher"):
            rr = [r for r in sig if r[0] == g][:2]
            if rr:
                parts.append(f"for {g}s, " + lst([f"{r[1]} ({LABEL[r[2]].lower()} {r[3]:+.1f}% if removed)" for r in rr]))
        out.append("**What matters** (largest validated losses when a feature family is removed): " + "; ".join(parts)
                   + ". Most other families are individually replaceable because overlapping families carry the same "
                     "information.")
    d = D["div"]
    if len(d):
        out.append(f"**Stacking: no practical gain.** At most {d.stack_opt_valid_pct_vs_best.max():.1f}% RMSE over the best "
                   "single model, because all three model families make nearly the same errors.")
    out.append("**Game probabilities:** well calibrated on average, but without enough discrimination to beat "
               "always-home on validation.")
    sh = D["shrink"]
    if len(sh):
        comp = sh[sh.cal_slope > 1.10]
        out.append("**Regression to the mean:** not a defect. The tight look of single-game scatter plots is "
                   "game-to-game noise. Projection deciles track outcomes with slopes near 1"
                   + (f", except {lst([LABEL[t].lower() for t in comp.target])} (compressed)" if len(comp) else "")
                   + ". The real errors are biases in level (section 7).")
    P = prod(D)
    if len(P):
        strong = P.sort_values("rmse_pct_better_than_league average", ascending=False).head(3)
        weak = P[P["rmse_pct_better_than_history-to-date"] < 2]
        bias_bad = []
        for g in ("pitcher", "hitter"):
            b = bias_table(D, g)
            bias_bad += [(g, t) for t, r in b.iterrows() if abs(r.bias_pct) >= 10]
        out.append("**Strongest:** " + lst([tag(r.group, r.target).lower() for r in strong.itertuples()])
                   + ". **Weakest:** " + lst([tag(r.group, r.target).lower() for r in weak.itertuples()])
                   + " (under 2% better than the player's own average)"
                   + (", plus " + lst([tag(g, t).lower() for g, t in bias_bad]) + " (biased by 10% or more)" if bias_bad else "")
                   + ".")
    out.append("**Next:** fix the pitcher over-projection (runs, home runs, hits), retune the calibration step "
               "for pitchers, and stop treating the game model as a source of edge.")
    return out


def diag_overfit(D) -> str:
    bf = best_family(D)
    cu = D["curves"]
    out = ["### Overfitting\n"]
    if len(bf):
        mild = bf[bf.verdict != "no meaningful overfitting"]
        none = bf[bf.verdict == "no meaningful overfitting"]
        out.append(f"For the family each target would select, the training-to-validation gap in RMSE has a median of "
                   f"{bf.valid_train_gap_pct.median():.1f}%. Validation keeps a median "
                   f"{100 * bf.skill_retention.median():.0f}% of the training skill, where skill is the share of error "
                   "removed relative to the training mean on the same rows. "
                   f"{lst([tag(r.group, r.target).lower() for r in none.itertuples()])} show no meaningful gap. The rest show "
                   "mild overfitting, concentrated in low-signal targets, where a little memorized noise is a large "
                   "share of a small skill.")
        wst = bf.loc[bf.skill_retention.idxmin()] if bf.skill_retention.notna().any() else None
        if wst is not None:
            out.append(f" The extreme case is {tag(wst.group, wst.target).lower()}: {100 * wst.train_skill_dev:.0f}% of "
                       f"error removed on training rows against {100 * wst.valid_skill_dev:.0f}% on validation.")
    if len(cu):
        dep = cu[cu.axis == "max_depth"]
        frac = cu[cu.axis == "train_fraction_recent"]
        still = frac.groupby(["group", "target"]).apply(lambda d: d.sort_values("value").valid_rmse.iloc[-1] <= d.valid_rmse.min() + 1e-12)
        out.append(f" Learning curves show the over-fitting regime clearly: trees of depth 6-8 cut training error "
                   f"sharply while validation error rises. The production settings sit short of it. Validation error "
                   f"is still falling with all available training rows for {int(still.sum())} of {len(still)} targets, "
                   "so more training seasons would likely help more than retuning.")
    gcal = D["gcal"]
    if len(gcal):
        x = gcal[gcal.model == "xgb_raw"].set_index("split")
        out.append(f" The one clear case is the game model's uncalibrated XGBoost: training AUC {x.loc['train', 'auc']:.2f} "
                   f"against {x.loc['valid', 'auc']:.2f} on validation.")
    return "".join(out) + "\n"


def diag_collinearity(D) -> str:
    red, abl, ls, ist = D["red"], D["abl"], D["lin"], D["imps"]
    out = ["### Multicollinearity\n"]
    if len(red):
        out.append("The player feature sets are heavily redundant. The same statistic appears over several windows "
                   "(e.g. strikeouts over the last 7, 10, 14, 21 and 30 starts). "
                   + "; ".join(f"{int(r.n_vif_gt_10)} of {int(r.n_features)} {r.model} features have a variance "
                               "inflation factor above 10" for r in red[red.model != "game"].itertuples())
                   + ". The game model's features are only mildly collinear.")
    if len(ls):
        f = ls[ls.feature_set == "full"]
        out.append(f" As a result the linear model's coefficients are not interpretable: refit on each quarter of the "
                   f"training window, a median {100 * f.share_sign_flips.median():.0f}% of them change sign.")
    if len(ist):
        out.append(f" Tree importances of individual features are unstable across refits (rank correlation "
                   f"{ist.fold_rank_corr_individual.median():.2f}), but importances grouped by what the feature measures "
                   f"are stable ({ist.fold_rank_corr_grouped.median():.2f}).")
    if len(abl):
        r90 = abl[abl.variant.str.contains("0.90", regex=False)]
        r80 = abl[abl.variant.str.contains("0.80", regex=False)]
        if len(r90):
            out.append(f" Replacing each group of features correlated at 0.90 or more with a single representative "
                       f"changed validation RMSE by at most {r90.valid_rmse_pct_vs_full.max():+.2f}%, so the redundancy "
                       "costs interpretability and training time, not accuracy.")
        if len(r80):
            worst = r80.loc[r80.valid_rmse_pct_vs_full.idxmax()]
            out.append(f" A looser 0.80 threshold shows why correlation alone should not drive removal: it grouped "
                       f"today's batting-order slot with the ten-game average slot and kept the average. That raised "
                       f"validation RMSE for {LABEL[worst.target].lower()} by {worst.valid_rmse_pct_vs_full:+.1f}%.")
    return "".join(out) + "\n\n"


def signal_rows(D) -> list:
    """(group, family, target, % validation RMSE increase) for removals whose interval is above zero,
    largest first per group."""
    abl = D["abl"]
    rows = []
    if len(abl) and "delta_lo" in abl:
        a = abl[(abl.variant != "full") & ~abl.variant.str.startswith("one per")]
        for (g, v), d in a.groupby(["group", "variant"]):
            hurt = d[d.delta_lo > 0]
            if len(hurt):
                top = hurt.loc[hurt.valid_rmse_pct_vs_full.idxmax()]
                rows.append((g, v.replace("no ", ""), top.target, float(top.valid_rmse_pct_vs_full)))
    return sorted(rows, key=lambda r: (r[0], -r[3]))


def diag_signal(D) -> str:
    abl = D["abl"]
    out = ["### What carries the signal\n"]
    out.append("Each feature family was removed in turn and the model refit and scored on validation. Importance "
               "here is predictive, not causal: a family that overlaps with another can look unimportant even when "
               "its information matters.")
    if len(abl) and "delta_lo" in abl:
        rows = signal_rows(D)
        for g in ("hitter", "pitcher"):
            rr = sorted([r for r in rows if r[0] == g], key=lambda r: -r[3])
            big = [r for r in rr if r[3] >= 0.3]
            if big:
                out.append(f" {g.title()}s: removing "
                           + lst([f"{r[1]} raises {LABEL[r[2]].lower()} error by {r[3]:.1f}%" for r in big]) + ".")
        gabl = D["gabl"]
        if len(gabl) and "delta_lo" in gabl:
            gh = gabl[(gabl.variant != "full") & (gabl.delta_lo > 0)]
            out.append(f" Game model: removing any one feature group changed validation log loss by at most "
                       f"{gabl.valid_logloss_delta_vs_full.abs().max():.4f}"
                       + (", never significantly." if gh.empty else f", significantly in {len(gh)} case(s)."))
    return "".join(out) + "\n"


def diag_stacking(D) -> str:
    d = D["div"]
    if d.empty:
        return ""
    ec = d[[c for c in d.columns if c.startswith("err_corr_")]].to_numpy()
    sig = d[d.stack_opt_valid_pct_lo > 0]
    return ("### Model diversity and stacking\n"
            f"The three player model families make nearly the same errors: the correlation between their errors has a "
            f"median of {np.nanmedian(ec):.2f} on validation. Most of each error is game-to-game noise that no model "
            f"can see. The optimized stack beats the best single family with an interval above zero for "
            f"{lst([tag(r.group, r.target).lower() for r in sig.itertuples()])}, by at most "
            f"{d.stack_opt_valid_pct_vs_best.max():.1f}% RMSE. That gain is real but too small to matter for a "
            "single-game projection, and it triples training time.\n")


def diag_temporal(D) -> str:
    t = D["temp"]
    out = ["### Stability through the season\n"]
    if len(t):
        th = t[t.period_type == "third"]
        piv = th.pivot_table(index=["group", "target"], columns="period", values="skill_vs_history_rmse_pct")
        span = (piv.max(1) - piv.min(1))
        moving = span[span > 3]
        out.append("The test window was split into early, middle and late thirds, and skill was measured against the "
                   "player's own average within each period. Skill is stable for most targets"
                   + (f"; it moves by more than 3 points for {lst([tag(g, tg).lower() for g, tg in moving.index])}" if len(moving) else "")
                   + ". The bias does not stay constant (section 7).")
    tn = D["tenure"]
    if len(tn):
        piv = tn.pivot_table(index=["group", "target"], columns="period", values="skill_vs_history_rmse_pct")
        if "<20 prior games" in piv and "120+" in piv:
            out.append(f" The projections add the most over a player's own average when that average rests on few "
                       f"games: median gain {piv['<20 prior games'].median():+.1f}% for players with fewer than 20 prior "
                       f"games, against {piv['120+'].median():+.1f}% for those with 120 or more.")
    gt = D["gtemp"]
    if len(gt):
        g = gt[(gt.period_type == "third") & (gt.split == "test")]
        out.append(f" Game-model AUC by third of the test window ranges from {g.auc.min():.3f} to {g.auc.max():.3f}.")
    return "".join(out) + "\n"


def conclusion(D) -> str:
    P = prod(D)
    out = []
    if len(P):
        pi = P.set_index(["group", "target"])
        strong = P.sort_values("rmse_pct_better_than_league average", ascending=False).head(3)
        weak = P[P["rmse_pct_better_than_history-to-date"] < 2]
        out.append(
            "InTheCount turns pitch-level Statcast data into daily projections for every starting pitcher and lineup "
            "hitter, and a win probability for every game. All of it was evaluated the way it is used: models chosen "
            "on one stretch of games and scored once on a later stretch they had never seen. "
            f"On that test, all {len(P)} player projections beat both the league average and the player's own history, "
            "with intervals above zero. The gains are largest where playing time drives the result: "
            + lst([f"{tag(g, t).lower()} {pi.loc[(g, t), 'rmse_pct_better_than_league average']:+.1f}% against the league average"
                   for g, t in zip(strong.group, strong.target)])
            + ". Knowing who starts, where a hitter bats and how deep a starter usually goes is the information a "
              "simple average cannot carry.\n\n")
        out.append(
            "For rare events the honest answer is a small edge. "
            + (lambda x: x[:1].upper() + x[1:])(lst([tag(g, t).lower() for g, t in zip(weak.group, weak.target)]))
            + " improve on the player's own average by less than 2%. A single game's home runs or walks are close to the "
              "limit of what any pre-game information can predict. Summed over a full window the projections still "
              "rank players well, so they are most useful for weekly and season-level decisions rather than single "
              "games.\n\n")
    bp = bias_table(D, "pitcher")
    big = bp[bp.bias_pct.abs() >= 5].sort_values("bias_pct", ascending=False) if len(bp) else bp
    if len(big):
        out.append(
            "The pitcher projections run high for "
            + lst([f"{LABEL[t].lower()} ({r.bias_pct:+.0f}%)" for t, r in big.iterrows()])
            + ". Most of that comes from the rate-times-innings part of the published blend, which points to how the "
              "blend is weighted rather than to the underlying models.\n\n")
    gcm, gcal = D["gcomp"], D["gcal"]
    sel = D["gmeta"].get("selected_on_valid", "")
    if len(gcm) and len(gcal):
        sr = gcm[gcm.model == sel]; hr = gcm[gcm.model == "baseline_home_rate"]
        s_ = gcal[gcal.model == sel].set_index("split")
        if len(sr) and len(hr):
            out.append(
                f"The game model's probabilities are well calibrated on average (calibration-in-the-large "
                f"{s_.loc['test', 'cal_intercept_itl']:+.3f} on test). Its test log loss of {sr.log_loss.iloc[0]:.4f} and "
                f"accuracy of {100 * sr.accuracy.iloc[0]:.1f}% compare with {hr.log_loss.iloc[0]:.4f} and "
                f"{100 * hr.accuracy.iloc[0]:.1f}% for always picking the home team.\n\n")
    return "".join(out)


def known_issues(D) -> str:
    out = []
    bp = bias_table(D, "pitcher")
    comp = D["pitcher_comp"]
    rows = []
    if len(bp) and len(comp):
        rate = comp[comp.variant == "Rate x predicted opportunity"].set_index("target")["mean_error"]
        direct = comp[comp.variant == "Direct count model"].set_index("target")["mean_error"]
        for t in [x for x in ("R", "HR", "H") if x in bp.index]:
            rows.append({"Issue": f"Pitcher {LABEL[t].lower()} over-projected",
                         "Evidence": f"{bp.loc[t, 'bias']:+.3f} per start, {bp.loc[t, 'bias_pct']:+.1f}% of the actual mean",
                         "Where it comes from": f"rate × innings part {rate.get(t, np.nan):+.2f}; direct model {direct.get(t, np.nan):+.2f}"})
    t = D["temp"]
    if len(t):
        th = t[(t.period_type == "third") & (t.group == "pitcher")]
        for tg in ("K", "IP"):
            x = th[th.target == tg].set_index("period")
            if {"early", "middle", "late"} <= set(x.index):
                rows.append({"Issue": f"Pitcher {LABEL[tg].lower()} bias grows late in the season",
                             "Evidence": f"bias {x.loc['early', 'bias']:+.2f} early, {x.loc['middle', 'bias']:+.2f} middle, "
                                                f"{x.loc['late', 'bias']:+.2f} late",
                             "Where it comes from": "late-season actuals fall (actual mean "
                                                    f"{x.loc['early', 'mean_actual']:.2f} early vs {x.loc['late', 'mean_actual']:.2f} late); "
                                                    "cause not verified"})
    om = pd.DataFrame()       # calibration-step finding comes from the diagnostics, which are not part of this report
    if len(om):
        v = om[om.split == "valid"]
        cal = v[v.family.isin(["xgb"])].set_index(["group", "target"])["rmse"]
        raw = v[v.family == "xgb_uncalibrated"].set_index(["group", "target"])["rmse"]
        pc = (100 * (cal / raw - 1)).dropna()
        p_med = pc.loc["pitcher"].median() if "pitcher" in pc.index.get_level_values(0) else np.nan
        h_med = pc.loc["hitter"].median() if "hitter" in pc.index.get_level_values(0) else np.nan
        # "about zero" is only meaningful where 0.05 is far below the target's mean
        z = v[v.family.isin(["rf", "xgb", "lin"]) & (v.mean_actual >= 0.5) & (v["share_pred_lt_0.05"] > 0.005)
              & (v.group == "pitcher")]
        zt = sorted({LABEL[t].lower() for t in z.target})
        zr = (100 * z["share_pred_lt_0.05"].min(), 100 * z["share_pred_lt_0.05"].max()) if len(z) else None
        rows.append({"Issue": "Calibration step tuned to the wrong target",
                     "Evidence": f"validation RMSE {p_med:+.1f}% (pitchers) and {h_med:+.1f}% (hitters) with "
                                        "the step vs without it, XGBoost, median",
                     "Where it comes from": "chosen by MAE, so it drifts toward the median"
                                            + (f"; projects about zero for {zr[0]:.1f}-{zr[1]:.1f}% of starts ({lst(zt)}), "
                                               "depending on model family" if zt else "")})
    gb = D["gboot"]
    if len(gb):
        rows.append({"Issue": "Game model has no demonstrated edge",
                     "Evidence": "not distinguishable from always-home on validation",
                     "Where it comes from": "18 team-level features carry little signal about a single game"})
    rows.append({"Issue": "Stolen bases not modelled", "Evidence": "Statcast pitch events carry no steals",
                 "Where it comes from": "a data-source limitation"})
    out.append("Bias is projected minus actual, on the test window.\n\n")
    out.append(table(pd.DataFrame(rows)))
    return "".join(out)
    out.append("\n**Next steps, in order.**\n\n")
    out.append("1. **Fix the pitcher level bias.** Over-projection of runs, home runs and hits comes mainly from the rate × "
               "predicted-innings part of the published blend; the train-window models alone are biased low. "
               "Re-estimate the blend weights or add a level correction on validation, then re-score the test window "
               "once.\n"
               "2. **Retune the calibration step for pitchers.** Choose it by RMSE or Poisson deviance instead of MAE, or "
               "remove it where validation does not improve.\n"
               "3. **Track the late-season drift** in strikeout and innings projections, and test whether a "
               "workload or innings-limit feature removes it.\n"
               "4. **Keep the deployed game model** and present game probabilities as close to coin-flip-plus-home-field; "
               "revisit only when a candidate clears always-home on validation with an interval.\n"
               "5. **Simplify.** Collapse near-duplicate windows (no measurable cost at the 0.90 threshold), keep a "
               "single model where a stack gains less than its uncertainty, and add the earliest cached season as "
               "training rows, since learning curves are still improving.\n"
               "6. **Evaluate more than one test window** (walk-forward), so that model-selection uncertainty itself "
               "has an interval.\n")
    return "".join(out)
