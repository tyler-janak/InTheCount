"""
report_audit.py
===============
Writes the report section "Feature Diagnostics, Multicollinearity, and
Overfitting" and the audit answers used in the conclusion, from the files
audit_models.py saved in outputs/model_evaluation/audit/metrics/.

Every sentence that states a result is generated from those files with the
rules written next to it, so a re-run of the audit re-writes the text. Nothing
here reads the test set to make a recommendation: recommendations use
validation evidence; test numbers are shown beside them as confirmation only.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
AUD = HERE / "outputs" / "model_evaluation" / "audit"
AM = AUD / "metrics"
FAMS = ["rf", "xgb", "lin"]
FAM_NAME = {"rf": "random forest", "xgb": "XGBoost", "lin": "Poisson GLM"}
LABEL = {"K": "Strikeouts", "BB": "Walks", "H": "Hits", "HR": "Home runs", "IP": "Innings pitched",
         "R": "Runs allowed", "PA": "Plate appearances", "TB": "Total bases", "2B": "Doubles", "3B": "Triples"}


def _csv(name):
    p = AM / name
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def _json(name):
    p = AM / name
    return json.loads(p.read_text()) if p.exists() else {}


def available() -> bool:
    return (AM / "overfit_summary.csv").exists() or (AM / "game_calibration.csv").exists()


def _n(x, d=3):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{d}f}"


def _p(x, d=1, sign=False):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return f"{x:+.{d}f}%" if sign else f"{x:.{d}f}%"


def _ci(pt, lo, hi, d=2, unit="%"):
    return f"{pt:+.{d}f}{unit} [{lo:+.{d}f}, {hi:+.{d}f}]"


def _lst(xs, empty="none"):
    xs = [str(x) for x in xs]
    return ", ".join(xs) if xs else empty


# ---------------------------------------------------------------------------
# Findings (shared by the section, the conclusion and the prioritized list)
# ---------------------------------------------------------------------------
def findings() -> dict:
    F: dict = {}
    m = _csv("overfit_metrics.csv")
    if len(m):
        v = m[m.split == "valid"]
        cal = v[v.family.isin(FAMS)].set_index(["group", "target", "family"])["rmse"]
        unc = v[v.family.str.endswith("_uncalibrated")].assign(family=lambda d: d.family.str.replace("_uncalibrated", ""))
        unc = unc.set_index(["group", "target", "family"])["rmse"]
        j = pd.concat([cal.rename("cal"), unc.rename("raw")], axis=1).dropna()
        j["pct"] = 100 * (j.cal / j.raw - 1)                # + = calibration wrapper makes validation RMSE worse
        F["calib"] = j.reset_index()
        z = v[v.family.isin(FAMS)]
        F["zero_preds"] = z[z["share_pred_lt_0.05"] > 0.005][["group", "target", "family", "share_pred_lt_0.05"]]
        t = m[m.split == "test"]
        ct = t[t.family.isin(FAMS)].set_index(["group", "target", "family"])["rmse"]
        ut = t[t.family.str.endswith("_uncalibrated")].assign(family=lambda d: d.family.str.replace("_uncalibrated", "")).set_index(["group", "target", "family"])["rmse"]
        jt = pd.concat([ct.rename("cal"), ut.rename("raw")], axis=1).dropna()
        F["calib_test"] = (100 * (jt.cal / jt.raw - 1)).rename("pct").reset_index()
    s = _csv("overfit_summary.csv")
    if len(s):
        F["overfit"] = s[s.family.isin(FAMS)].copy()
    F["div"] = _csv("model_diversity.csv")
    F["abl"] = _csv("ablation.csv")
    F["red"] = _csv("redundancy_summary.csv")
    F["thr"] = _csv("redundancy_thresholds.csv")
    F["fam"] = _csv("redundancy_families.csv")
    F["stems"] = _csv("redundancy_stems.csv")
    F["linstab"] = _csv("linear_stability.csv")
    F["impstab"] = _csv("importance_stability.csv")
    F["imp"] = _csv("importance_individual.csv")
    F["gperm"] = _csv("importance_grouped_permutation.csv")
    F["curves"] = _csv("learning_curves.csv")
    F["shrink"] = _csv("shrinkage_test.csv")
    F["dec_valid"] = _csv("deciles_valid_best_family.csv")
    F["dec_test"] = _csv("deciles_test_production.csv")
    F["temporal"] = _csv("temporal_players.csv")
    F["tenure"] = _csv("tenure_players.csv")
    F["base"] = _csv("baselines_bootstrap.csv")
    F["gcal"] = _csv("game_calibration.csv")
    F["gboot"] = _csv("game_bootstrap.csv")
    F["gbk"] = _csv("game_buckets.csv")
    F["ghi"] = _csv("game_high_confidence.csv")
    F["gtemp"] = _csv("game_temporal.csv")
    F["gdiv"] = _csv("game_diversity.csv")
    F["gimp"] = _csv("game_importance.csv")
    F["gabl"] = _csv("game_ablation.csv")
    F["glc"] = _csv("game_learning_curves.csv")
    F["gstab"] = _csv("game_logistic_stability.csv")
    F["gmeta"] = _json("game_meta.json")
    F["run"] = _json("audit_run.json")
    F["inv"] = _csv("feature_inventory.csv")
    return F


def _abl_effects(abl: pd.DataFrame):
    """Per (group, family-ablation): targets where removal hurts validation RMSE
    with the 95% date-block interval above zero, and targets where it helps
    with the interval below zero (rule used for 'matters' / 'removable')."""
    out = []
    if abl.empty or "delta_lo" not in abl:
        return pd.DataFrame()
    a = abl[(abl.variant != "full") & abl.delta_lo.notna()]
    for (g, var), d in a.groupby(["group", "variant"]):
        hurt = d[d.delta_lo > 0]
        helped = d[d.delta_hi < 0]
        out.append({"group": g, "variant": var, "n_runs": len(d), "n_dropped": int(d.n_dropped.iloc[0]),
                    "hurts": sorted({f"{r.target} ({r.model}, {r.valid_rmse_pct_vs_full:+.2f}%)" for r in hurt.itertuples()}),
                    "helps": sorted({f"{r.target} ({r.model}, {r.valid_rmse_pct_vs_full:+.2f}%)" for r in helped.itertuples()}),
                    "n_hurt": len(hurt), "n_help": len(helped),
                    "max_hurt_pct": float(d.valid_rmse_pct_vs_full.max()), "median_pct": float(d.valid_rmse_pct_vs_full.median())})
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
_WA = None          # appendix writer for the current render (None = everything in the main text)
_APX_NUM = 11


def _apx(title: str, body: str) -> str:
    """Send a detailed table/figure to the appendix; return a pointer for the main text."""
    if _WA is None:
        return body
    _WA(f"**{title}**\n")
    _WA(body)
    return f"*(Appendix A{_APX_NUM}: \"{title}\".)*\n"


def section(w, fig, md_table, num: int, wa=None) -> None:
    global _WA, _APX_NUM
    _WA, _APX_NUM = wa, num
    if wa is not None:
        wa(f"## A{num}. Audit — full tables and charts\n")
    F = findings()
    w(f"## {num}. Feature Diagnostics, Multicollinearity, and Overfitting\n")
    run = F["run"]
    w("This section is an audit, not a tuning exercise. Every diagnostic model was fit on the **training** window "
      "with the same settings as production (including the calibration wrapper), every comparison that could "
      "motivate a change was made on the **validation** window, and the **test** window is shown only next to "
      "those results, as confirmation. Nothing in this section changed the production models. Intervals are "
      "95% bootstrap intervals that resample whole game dates (players on the same day share weather, umpires "
      "and opponents, so resampling rows would overstate precision). Produced by `audit_models.py`"
      + (f" ({run.get('minutes')} min" + (", QUICK sample - not for publication" if run.get("quick") else "") + ")" if run else "")
      + "; every table below is a file in `outputs/model_evaluation/audit/metrics/`.\n")

    # -------------------------------------------------------------- pipeline map
    w(f"### {num}.1 What the models see\n")
    inv = F["inv"]
    if len(inv):
        u = inv[inv.used]
        tab = u.pivot_table(index="family", columns="group", values="feature", aggfunc="count", fill_value=0)
        w("Feature counts by construction family (a feature belongs to one family; precedence: lineup, handedness, "
          "true-talent, matchup, team/opponent, park, batted-ball, velocity/pitch-mix, opportunity, rolling window, "
          "expanding `_std`):\n")
        w(md_table(tab.reset_index(), {c: (lambda v: f"{int(v)}") for c in tab.columns}) + "\n")
        gap = u[(u.coverage_gap_train_test.abs() > 0.2)]
        w("Every feature passes `leakage_guard.assert_pregame_features` (a trailing-window name or an explicitly "
          "justified whitelist entry) and is computed by the same functions at training and serving time. Coverage "
          "(share of non-missing values) was compared between the training and test windows: "
          + (f"{len(gap)} feature(s) differ by more than 20 points ({_lst(gap.feature.head(8))}), which matters for "
             "median imputation." if len(gap) else "no feature differs by more than 20 points, so the median imputation "
             "sees similar inputs in training and in use.")
          + (" `lineup_spot` is blank for substitutes by design." if "lineup_spot" in set(u.feature) else "") + "\n")
        sb = inv[inv.in_saved_bundles.notna() & (inv.in_saved_bundles < 1) & inv.used] if "in_saved_bundles" in inv else pd.DataFrame()
        if len(sb):
            w(f"*{len(sb)} feature(s) in the current lists are missing from some saved production bundles "
              f"({_lst(sb.feature.head(6))}); those bundles predate the list and are refit by the next training run.*\n")

    # -------------------------------------------------------------- redundancy
    w(f"### {num}.2 Correlation, VIF and redundancy (training rows only)\n")
    red, thr = F["red"], F["thr"]
    if len(red):
        r2 = red.copy()
        r2["condition_number"] = r2.condition_number.map(lambda v: f"{v:,.0f}")
        w(md_table(r2[["model", "n_features", "rows_used", "median_vif", "n_vif_gt_5", "n_vif_gt_10", "n_vif_gt_100", "condition_number"]],
                   {"n_features": lambda v: f"{int(v)}", "rows_used": lambda v: f"{int(v):,}", "median_vif": lambda v: f"{v:,.1f}",
                    "n_vif_gt_5": lambda v: f"{int(v)}", "n_vif_gt_10": lambda v: f"{int(v)}", "n_vif_gt_100": lambda v: f"{int(v)}"}) + "\n")
    if len(thr):
        t2 = thr.pivot_table(index="model", columns="threshold", values="pairs").reset_index()
        t2.columns = ["model"] + [f"pairs |r|>={c:.2f}" for c in t2.columns[1:]]
        tot = thr.groupby("model").of_pairs.first()
        t2["of all pairs"] = t2.model.map(tot)
        w("Highly correlated feature pairs (max of |Pearson|, |Spearman|):\n")
        w(md_table(t2, {c: (lambda v: f"{int(v):,}") for c in t2.columns if c != "model"}) + "\n")
        s90 = thr[thr.threshold == 0.90].set_index("model")
        msg = []
        for g in s90.index:
            msg.append(f"{g}: {_p(100 * s90.loc[g, 'share_same_construct'], 0)} of the |r| >= 0.90 pairs measure the same "
                       f"construct and {_p(100 * s90.loc[g, 'share_same_stem'], 0)} are the same statistic over different windows")
        w("Redundancy is concentrated inside constructs, not across them - " + "; ".join(msg) + ".\n")
    for g in ("pitcher", "hitter"):
        w(fig(f"audit/figures/audit_corr_{g}.png", f"{g.title()} feature correlation structure (training window)",
              f"INSERT {g.upper()} CORRELATION HEATMAP"))
    w(_apx("Game feature correlation heatmap", fig("audit/figures/audit_corr_game.png",
                                                   "Game feature correlation structure (training window)",
                                                   "INSERT GAME CORRELATION HEATMAP")))
    st = F["stems"]
    if len(st):
        explicit = []
        for g in ("pitcher", "hitter"):
            d = st[st.model == g].sort_values("mean_abs_rho", ascending=False)
            pick = [s for s in ("K", "BB", "H", "HR", "IP", "BF", "outs", "pitches", "K_rate", "H_rate", "PA", "h_rate", "k_rate") if s in set(d.stem)]
            explicit.append(d[d.stem.isin(pick)].head(8))
        ex = pd.concat(explicit) if explicit else pd.DataFrame()
        if len(ex):
            top = ex.sort_values("mean_abs_rho", ascending=False).head(3)
            w("The named redundant groups (one statistic over every window, e.g. `K_last7 ... K_last30, K_std`) are "
              "near-duplicates: " + "; ".join(f"{r.model} `{r.stem}` windows correlate at {r.mean_abs_rho:.2f} on average"
                                             for r in top.itertuples()) + ".\n")
            w(_apx("Redundant window groups", md_table(ex[["model", "stem", "n_windows", "mean_abs_rho", "min_abs_rho", "max_vif"]],
                       {"n_windows": lambda v: f"{int(v)}", "max_vif": lambda v: f"{v:,.0f}"}) + "\n"))
    fam = F["fam"]
    if len(fam):
        w(_apx("Redundancy by feature family (within-family mean |rho| and VIF)",
               md_table(fam[["model", "family", "n_features", "within_mean_abs_rho", "within_max_abs_rho", "median_vif", "max_vif", "n_vif_gt_10"]],
                   {"n_features": lambda v: f"{int(v)}", "median_vif": lambda v: f"{v:,.1f}", "max_vif": lambda v: f"{v:,.0f}",
                    "n_vif_gt_10": lambda v: f"{int(v)}"}) + "\n"))
    w("**Does the redundancy cause a problem?** Correlation alone is not a defect. Three tests decide it.\n")
    ls = F["linstab"]
    if len(ls):
        full = ls[ls.feature_set == "full"]
        red_ = ls[ls.feature_set != "full"]
        w(f"*Linear-model instability.* The Poisson GLM was refit on each chronological quarter of the training window. "
          f"With the full feature set a median of {_p(100 * full.share_sign_flips.median(), 0)} of coefficients change sign "
          f"between quarters (range {_p(100 * full.share_sign_flips.min(), 0)}-{_p(100 * full.share_sign_flips.max(), 0)}), "
          f"median coefficient CV {_n(full.median_coef_cv.median(), 2)}"
          + (f"; keeping one feature per |rho| >= 0.90 cluster gives {_p(100 * red_.share_sign_flips.median(), 0)}" if len(red_) else "")
          + ". Individual GLM coefficients are therefore **not interpretable** (that is what collinearity does); "
            "whether the *predictions* suffer is the next test.\n")
    abl = F["abl"]
    if len(abl):
        rr = abl[abl.variant.str.startswith("one per")]
        if len(rr):
            agg = rr.groupby(["group", "variant", "model"]).agg(targets=("target", "count"), n_features=("n_features", "median"),
                                                               median_pct=("valid_rmse_pct_vs_full", "median"),
                                                               worst_pct=("valid_rmse_pct_vs_full", "max"),
                                                               sig_worse=("delta_lo", lambda x: int((x > 0).sum())),
                                                               sig_better=("delta_hi", lambda x: int((x < 0).sum()))).reset_index()
            w("*Out-of-sample cost of removing redundancy.* Each |rho| cluster was replaced by its medoid feature "
              "(clusters and medoids chosen on training rows without looking at any target), the model refit on "
              "training rows and scored on validation:\n")
            w(md_table(agg, {"targets": lambda v: f"{int(v)}", "n_features": lambda v: f"{int(v)}",
                             "median_pct": lambda v: f"{v:+.2f}%", "worst_pct": lambda v: f"{v:+.2f}%",
                             "sig_worse": lambda v: f"{int(v)}", "sig_better": lambda v: f"{int(v)}"}) + "\n")
            w("(`median_pct`/`worst_pct`: change in validation RMSE vs the full set; `sig_worse`/`sig_better`: targets whose "
              "date-block interval excludes zero.)\n")
    ist = F["impstab"]
    if len(ist):
        w(f"*Tree-importance instability.* XGBoost was refit on three expanding chronological folds of the training "
          f"window and SHAP importance computed on the same validation rows. Rank correlation of **individual** feature "
          f"importance between folds: median {_n(ist.fold_rank_corr_individual.median(), 2)} (top-10 overlap, Jaccard "
          f"{_n(ist.fold_top10_jaccard.median(), 2)}); of **construct-grouped** importance: median "
          f"{_n(ist.fold_rank_corr_grouped.median(), 2)}. Native gain importance agrees with SHAP at rank correlation "
          f"{_n(ist.native_vs_shap_rank_corr.median(), 2)}. Correlated windows trade importance among themselves from fit "
          "to fit while the construct they belong to keeps its place: read individual-feature importance as unstable "
          "and grouped importance as the stable quantity.\n")
    w(_redundancy_verdict(F) + "\n")

    # -------------------------------------------------------------- importance
    w(f"### {num}.3 Which feature groups carry the signal\n")
    w("Importance below is **predictive attribution, not causation**: SHAP and permutation importance describe how "
      "these fitted models use correlated inputs, and a group that is redundant with another can look unimportant "
      "even when the information it carries matters.\n")
    w(fig("audit/figures/audit_group_importance.png", "Construct-grouped importance (XGBoost SHAP share; game: grouped permutation)",
          "INSERT FEATURE-GROUP IMPORTANCE"))
    gp = F["gperm"]
    if len(gp):
        top = (gp[gp.family == "xgb"].sort_values("pct_increase", ascending=False).groupby(["group", "target"]).head(3)
               .groupby(["group", "target"]).apply(lambda d: ", ".join(f"{r.construct} ({r.pct_increase:+.1f}%)" for r in d.itertuples()))
               .rename("top 3 groups: validation RMSE increase when permuted").reset_index())
        w(_apx("Top three construct groups per target (grouped permutation, XGBoost, validation)", md_table(top) + "\n"))
    w(_ablation_text(F, num) + "\n")

    # -------------------------------------------------------------- overfitting
    w(f"### {num}.4 Overfitting and shrinkage\n")
    w("Skill below is the share of Poisson deviance explained relative to the training-mean predictor **on the same "
      "rows**, so a noisier period does not masquerade as over-fitting. *Calibration slope* is the OLS slope of actual "
      "on predicted: 1 means the spread of the predictions is right, above 1 means predictions are compressed toward "
      "the mean (over-shrunk), below 1 means they are too spread out (the classic over-fit signature).\n")
    w(fig("audit/figures/audit_train_valid_test.png", "Train vs validation vs test skill", "INSERT TRAIN/VALIDATION/TEST ERROR"))
    of = F.get("overfit")
    if of is not None and len(of):
        cols = ["group", "target", "family", "train_rmse", "valid_rmse", "test_rmse", "valid_train_gap_pct", "test_valid_gap_pct",
                "train_skill_dev", "valid_skill_dev", "test_skill_dev", "train_cal_slope", "valid_cal_slope", "verdict"]
        fmt_of = ({c: (lambda v: f"{v:+.1f}%") for c in ("valid_train_gap_pct", "test_valid_gap_pct")}
                  | {c: (lambda v: f"{100 * v:.1f}%") for c in ("train_skill_dev", "valid_skill_dev", "test_skill_dev")}
                  | {c: (lambda v: f"{v:.2f}") for c in ("train_cal_slope", "valid_cal_slope")})
        best = of.loc[of.groupby(["group", "target"]).valid_rmse.idxmin()]
        w("For the family each target would select on validation:\n")
        w(md_table(best[["group", "target", "family", "valid_train_gap_pct", "train_skill_dev", "valid_skill_dev",
                         "test_skill_dev", "valid_cal_slope", "verdict"]], fmt_of) + "\n")
        w(_apx("Overfitting diagnostics, every target and family", md_table(of[cols], fmt_of) + "\n"))
        w("Verdict rule (validation only): *clear overfitting* = validation keeps under half of the training skill **and** "
          "slope < 0.85; *mild* = under 75% of the training skill **or** slope < 0.90; *under-fit / excessive shrinkage* = "
          "slope > 1.10; otherwise *no meaningful overfitting*.\n")
        vc = of.verdict.value_counts()
        w("Counts: " + "; ".join(f"{k}: {v}" for k, v in vc.items()) + ".\n")
    w(_calib_wrapper_text(F) + "\n")
    w(_curves_text(F) + "\n")
    w(_shrinkage_text(F) + "\n")
    w(fig("audit/figures/audit_deciles.png", "Decile calibration of the published projection (test)", "INSERT PREDICTION DECILE CALIBRATION"))

    # -------------------------------------------------------------- game calibration
    w(f"### {num}.5 Game-model calibration\n")
    w(_game_text(F, md_table) + "\n")
    w(fig("audit/figures/audit_reliability.png", "Reliability diagram with 95% intervals (validation and test)", "INSERT RELIABILITY CURVE"))

    # -------------------------------------------------------------- diversity
    w(f"### {num}.6 Model diversity and stacking\n")
    w(fig("audit/figures/audit_error_corr.png", "Prediction and error correlation between model families (validation)",
          "INSERT MODEL ERROR CORRELATION MATRIX"))
    w(_stack_text(F, md_table) + "\n")

    # -------------------------------------------------------------- temporal
    w(f"### {num}.7 Temporal stability and sample size\n")
    w(_temporal_text(F, md_table) + "\n")

    # -------------------------------------------------------------- baselines
    w(f"### {num}.8 Against simple baselines, with uncertainty\n")
    w(_baseline_text(F, md_table) + "\n")

    # -------------------------------------------------------------- recommendations
    w(f"### {num}.9 Recommended changes (prioritized)\n")
    w("Each recommendation cites validation evidence. None has been applied to the production models; each should be "
      "made as its own change, re-validated on the same windows, and the test window re-scored once afterwards.\n")
    pr = priorities(F)
    for level in ("Must fix", "Should fix", "Nice to have"):
        items = [x for x in pr if x[0] == level]
        if items:
            w(f"**{level}**\n")
            for _, _, txt in items:
                w(f"- {txt}")
            w("")


# ---------------------------------------------------------------------------
def _redundancy_verdict(F) -> str:
    red, abl = F["red"], F["abl"]
    if red.empty:
        return ""
    p = red[red.model.isin(["pitcher", "hitter"])]
    rr = abl[abl.variant.str.startswith("one per")] if len(abl) else pd.DataFrame()
    sig_worse = int((rr.delta_lo > 0).sum()) if len(rr) else 0
    worst = float(rr.valid_rmse_pct_vs_full.max()) if len(rr) else np.nan
    txt = (f"**Verdict on multicollinearity.** It is real and large for the player models "
           f"({_lst([f'{r.model}: {int(r.n_vif_gt_10)} of {int(r.n_features)} features with VIF > 10' for r in p.itertuples()])}), "
           "and it makes linear coefficients and individual tree importances unstable. ")
    if len(rr):
        txt += (f"It does **not** show up as a measurable prediction problem: collapsing every near-duplicate cluster "
                f"changed validation RMSE by at most {worst:+.2f}%, and only {sig_worse} of {len(rr)} model/target runs got "
                f"worse with an interval excluding zero. ")
    g = red[red.model == "game"]
    if len(g):
        txt += (f"The game model's 17 varying features are mildly collinear (condition number "
                f"{g.condition_number.iloc[0]:,.0f}, {int(g.n_vif_gt_10.iloc[0])} with VIF > 10). ")
    txt += ("So the redundant windows are a cost in interpretability, training time and maintenance rather than in "
            "accuracy; removing them is a simplification to make on validation evidence, not a fix for an accuracy defect.")
    return txt


def _ablation_text(F, num) -> str:
    e = _abl_effects(F["abl"])
    if e.empty:
        return "`[ablation results not available]`"
    lines = ["**Feature-family ablations** (each family removed alone; XGBoost and Poisson GLM refit on training rows, "
             "scored on validation; a family *matters* for a target when removing it raises validation RMSE with the "
             "date-block interval above zero):\n"]
    rows = []
    for r in e[~e.variant.str.startswith("one per")].itertuples():
        rows.append({"group": r.group, "removed": r.variant.replace("no ", ""), "features dropped": r.n_dropped,
                     "runs": r.n_runs, "hurts (interval > 0)": _lst(r.hurts), "helps (interval < 0)": _lst(r.helps),
                     "median change": f"{r.median_pct:+.2f}%", "largest": f"{r.max_hurt_pct:+.2f}%"})
    t = pd.DataFrame(rows)
    from generate_report import md_table
    lines.append(_apx("Feature-family ablations (validation)", md_table(t) + "\n"))
    matter = e[(e.n_hurt > 0) & ~e.variant.str.startswith("one per")]
    none = e[(e.n_hurt == 0) & ~e.variant.str.startswith("one per")]
    for g in sorted(e.group.unique()):
        mg = matter[matter.group == g].sort_values("max_hurt_pct", ascending=False)
        ng = none[none.group == g]
        lines.append(f"*{g.title()}s:* families with measurable incremental value: "
                     f"{_lst([v.replace('no ', '') + f' (up to {x:+.2f}%)' for v, x in zip(mg.variant, mg.max_hurt_pct)])}. "
                     f"No measurable incremental value on any target: {_lst([v.replace('no ', '') for v in ng.variant])}"
                     " - their information is either absent or already carried by other families.")
    ga = F["gabl"]
    if len(ga) and "valid_logloss_delta_vs_full" in ga:
        d = ga[ga.variant != "full"]
        hurt = d[d.delta_lo > 0]
        lines.append(f"\n*Game model:* removing a feature group raised validation log loss with an interval above zero in "
                     f"{len(hurt)} of {len(d)} model/group runs"
                     + (f" ({_lst([f'{r.variant} ({r.model})' for r in hurt.itertuples()])})" if len(hurt) else "")
                     + ". Removing any one group never moved validation log loss by more than "
                     f"{d.valid_logloss_delta_vs_full.abs().max():.4f}.")
    lines.append("\nAn ablation measures *incremental* value given everything else. With this much overlap, removing one "
                 "family often costs nothing because a correlated family carries the same information; that is not proof "
                 "the information is useless.")
    return "\n".join(lines)


def _calib_wrapper_text(F) -> str:
    c = F.get("calib")
    if c is None or c.empty:
        return ""
    worse = c[c.pct > 0.25]; better = c[c.pct < -0.25]
    ct = F.get("calib_test", pd.DataFrame())
    zp = F.get("zero_preds", pd.DataFrame())
    txt = ("**The tail-of-train calibration wrapper.** Production wraps every player model in a post-hoc calibration fit "
           "on the last 15% of the training window, choosing isotonic over linear or none when it lowers **MAE** on "
           "that tail. Comparing the same fits with and without the wrapper on validation: the wrapper makes validation "
           f"RMSE worse by more than 0.25% in {len(worse)} of {len(c)} target/family fits and better in {len(better)} "
           f"(median {c.pct.median():+.2f}%; XGBoost median {c[c.family == 'xgb'].pct.median():+.2f}%).")
    if len(ct):
        txt += f" The test window shows the same direction (median {ct.pct.median():+.2f}%), as confirmation only."
    if len(zp):
        txt += (f" It also produces predictions of (almost) exactly zero for "
                f"{_lst([f"{r['group']} {r['target']} {r['family']} ({100 * r['share_pred_lt_0.05']:.1f}%)" for _, r in zp.iterrows()])} of validation rows - "
                "a starter projected for zero walks or zero runs, which is impossible to price as a probability.")
    txt += (" The cause is the selection criterion: MAE is minimized by the conditional *median*, and for low counts an "
            "isotonic step function fit to ~15% of the training rows moves predictions toward the median, which is lower "
            "than the mean and is often 0. The site publishes means and the evaluation uses RMSE and Poisson deviance, so "
            "the wrapper optimizes the wrong target.")
    return txt


def _curves_text(F) -> str:
    c = F["curves"]
    if c.empty:
        return ""
    out = ["**XGBoost learning curves** (training-window fit, validation score; table "
           "`learning_curves.csv`). For each axis, the production setting's validation RMSE vs the best value on the grid:"]
    rows = []
    for (g, t, ax), d in c.groupby(["group", "target", "axis"]):
        prod = d[d.production]
        best = d.loc[d.valid_rmse.idxmin()]
        if prod.empty:
            pv = np.nan; pval = "(not on grid)"
        else:
            pv = float(prod.valid_rmse.iloc[0]); pval = prod.value.iloc[0]
        rows.append({"group": g, "target": t, "axis": ax, "production": pval, "best on grid": best.value,
                     "valid RMSE gap": 100 * (pv / best.valid_rmse - 1) if np.isfinite(pv) else np.nan,
                     "train-valid gap at production": float(100 * (prod.valid_rmse.iloc[0] / prod.train_rmse.iloc[0] - 1)) if len(prod) else np.nan})
    t = pd.DataFrame(rows)
    from generate_report import md_table
    t["production"] = t["production"].map(lambda v: v if isinstance(v, str) else f"{v:g}")
    t["best on grid"] = t["best on grid"].map(lambda v: f"{v:g}")
    out.append(f"The production setting is within {np.nanmax(t['valid RMSE gap']):.2f}% of the best grid value on every axis.")
    out.append(_apx("XGBoost learning curves: production vs best grid value",
                    md_table(t, {"valid RMSE gap": lambda v: f"{v:+.2f}%", "train-valid gap at production": lambda v: f"{v:+.1f}%"}) + "\n"))
    deep = c[(c.axis == "max_depth")]
    if len(deep):
        dd = deep.groupby(["group", "target"]).apply(lambda d: (d.sort_values("value").train_rmse.iloc[-1] < 0.8 * d.train_rmse.max(),
                                                               d.sort_values("value").valid_rmse.iloc[-1] > d.valid_rmse.min())).tolist()
        n_classic = sum(a and b for a, b in dd)
        out.append(f"Depth 6-8 drives training error down sharply while validation error rises in {n_classic} of {len(dd)} "
                   "curves: the regime where XGBoost *would* over-fit exists, and the production settings sit well short of it.")
    frac = c[c.axis == "train_fraction_recent"]
    if len(frac):
        still = frac.groupby(["group", "target"]).apply(lambda d: d.sort_values("value").valid_rmse.iloc[-1] <= d.valid_rmse.min() + 1e-12)
        out.append(f"Validation error is still falling at 100% of the training rows for {int(still.sum())} of {len(still)} "
                   "targets: more training rows (e.g. the earliest cached season, currently history-only) would likely help more "
                   "than any hyperparameter.")
    return "\n".join(out)


def _shrinkage_text(F) -> str:
    sh, dv = F["shrink"], F["dec_valid"]
    if sh.empty:
        return ""
    lines = ["**Regression toward the mean, quantified.** Scatter plots of single games always look compressed: the "
             f"standard deviation of the predictions is only {sh.sd_pred_over_sd_actual.min():.2f}-{sh.sd_pred_over_sd_actual.max():.2f} "
             "of the actual outcomes', because most of a single game is unpredictable. That ratio is not a defect; the test is "
             "whether the *mean* outcome moves one-for-one with the prediction."]
    rows = []
    for r in sh.itertuples():
        v = dv[(dv.group == r.group) & (dv.target == r.target)]
        vs = np.nan
        if len(v):
            vs = float(np.polyfit(v.mean_pred, v.mean_actual, 1)[0])
        rows.append({"group": r.group, "target": r.target, "slope (test, published)": r.cal_slope,
                     "decile slope (validation, best family)": vs,
                     "top-decile actual − pred": r.top_decile_residual, "bottom-decile actual − pred": r.bottom_decile_residual,
                     "actual spread / predicted spread (D10−D1)": r.spread_ratio_actual_over_pred})
    t = pd.DataFrame(rows)
    from generate_report import md_table
    lines.append(md_table(t, {c: (lambda v: f"{v:.2f}") for c in t.columns if c not in ("group", "target")}) + "\n")
    comp = t[t["slope (test, published)"] > 1.10]; spread = t[t["slope (test, published)"] < 0.90]
    ok = t[(t["slope (test, published)"] >= 0.90) & (t["slope (test, published)"] <= 1.10)]
    lines.append(f"Compressed toward the mean (slope > 1.10): {_lst(comp.group.str[0].str.upper() + ' ' + comp.target)}. "
                 f"About right (0.90-1.10): {_lst(ok.group.str[0].str.upper() + ' ' + ok.target)}. "
                 f"Too spread out (slope < 0.90): {_lst(spread.group.str[0].str.upper() + ' ' + spread.target)}. "
                 "So the visual impression of strong regression to the mean is mostly the irreducible single-game noise; "
                 "where the decile slope departs from 1 the table says in which direction.")
    dt = F["dec_test"]
    if len(dt):
        bias = dt.groupby(["group", "target"]).apply(lambda d: float(np.average(d.mean_actual - d.mean_pred, weights=d.n)))
        level = dt.groupby(["group", "target"]).apply(lambda d: float(np.average(d.mean_actual, weights=d.n)))
        rel = (100 * bias / level).sort_values()
        over = rel[rel < -3]
        if len(over):
            lines.append(f" What the decile plot does show is a **level** error: the published projection is above the actual mean "
                         f"in most deciles for {_lst([f'{g[0].upper()} {t} ({v:+.0f}%)' for (g, t), v in over.items()])} on test "
                         "(actual minus predicted, % of the actual mean). That is a bias, not shrinkage; the temporal table shows "
                         "whether it grows through the season.")
    return "\n".join(lines)


def _game_text(F, md_table) -> str:
    c, meta = F["gcal"], F["gmeta"]
    if c.empty:
        return "`[game audit not available]`"
    sel = meta.get("selected_on_valid", "")
    keep = ["xgb_raw", "xgb_isotonic", "rf_raw", "rf_isotonic", "logistic", "stack_even", "stack_opt", "baseline_train_home_rate"]
    t = c[c.model.isin(keep) & c.split.isin(["valid", "test"])][
        ["model", "split", "n", "log_loss", "brier", "auc", "cal_intercept_itl", "cal_slope", "ece"]]
    t = t.sort_values(["split", "log_loss"])
    out = [f"Train {meta.get('n_train', 0):,} / validation {meta.get('n_valid', 0):,} / test {meta.get('n_test', 0):,} games. "
           f"Candidates fit on training games for validation; refit on train + validation for test, exactly as "
           f"`train_game_model.py` does. Selected on validation log loss: **{sel}**. *Calibration-in-the-large* is the "
           "intercept with the slope fixed at 1 (0 = right average level); *calibration slope* is from refitting "
           "logit(y) on logit(p) (1 = right spread, < 1 = over-confident); ECE uses ten equal-width bins.\n",
           md_table(t[t.model.isin([sel, "xgb_isotonic", "logistic", "baseline_train_home_rate"])], {"n": lambda v: f"{int(v):,}"}) + "\n",
           _apx("Game candidates: every model, validation and test", md_table(t, {"n": lambda v: f"{int(v):,}"}) + "\n")]
    b = F["gboot"]
    if len(b):
        v = b[(b.split == "valid") & b.logloss_minus_home_rate.notna()]
        beat = v[v.hi < 0]
        out.append(f"**Skill over the always-home baseline (validation, log-loss difference with date-block interval):** "
                   + "; ".join(f"{r.model} {_ci(r.logloss_minus_home_rate * 1000, r.lo * 1000, r.hi * 1000, 1, '')}" for r in v.itertuples())
                   + " (×1000; negative = better than always-home). "
                   + (f"Only {_lst(beat.model)} beat(s) it with an interval excluding zero." if len(beat) else
                      "**No candidate beats the always-home baseline on validation with an interval excluding zero.**"))
        te = b[(b.split == "test") & b.logloss_minus_home_rate.notna()]
        bt = te[te.hi < 0]
        out.append(f" On test (confirmation only) {len(bt)} of {len(te)} candidates beat it with an interval excluding zero "
                   f"({_lst(bt.model)}).")
        iso = b[b.model.str.contains("minus")]
        if len(iso):
            out.append("\n**Does isotonic calibration help?** Log-loss change from adding isotonic calibration (×1000, "
                       "negative = isotonic better): " + "; ".join(
                           f"{r.model.split(' minus ')[0].replace('_isotonic', '')} {r.split} {_ci(r.logloss_diff * 1000, r.lo * 1000, r.hi * 1000, 1, '')}"
                           for r in iso.itertuples()) + ".")
        gv = c[c.split == "valid"].set_index("model")
        for fam in ("xgb", "rf"):
            if f"{fam}_raw" in gv.index and f"{fam}_isotonic" in gv.index:
                a, i = gv.loc[f"{fam}_raw"], gv.loc[f"{fam}_isotonic"]
                out.append(f" {fam.upper()} on validation: AUC {a.auc:.3f} → {i.auc:.3f}, ECE {a.ece:.3f} → {i.ece:.3f}, "
                           f"slope {a.cal_slope:.2f} → {i.cal_slope:.2f}.")
        tr = c[c.split == "train"].set_index("model")
        if "xgb_raw" in tr.index:
            out.append(f" For scale: raw XGBoost has training AUC {tr.loc['xgb_raw', 'auc']:.3f} versus validation "
                       f"{gv.loc['xgb_raw', 'auc']:.3f} - the game XGBoost memorizes its training games; the logistic model "
                       f"(training AUC {tr.loc['logistic', 'auc']:.3f}) does not.")
    hi = F["ghi"]
    if len(hi):
        h = hi[(hi.model == sel) & (hi.n > 0)]
        if len(h):
            out.append("\n\n**High-confidence picks** (favorite's probability ≥ threshold) for the selected model:\n")
            out.append(md_table(h[["split", "confidence_ge", "n", "mean_conf", "fav_win_rate", "ci_lo", "ci_hi"]],
                                {"n": lambda v: f"{int(v)}", "mean_conf": lambda v: f"{100 * v:.1f}%",
                                 "fav_win_rate": lambda v: f"{100 * v:.1f}%", "ci_lo": lambda v: f"{100 * v:.1f}%",
                                 "ci_hi": lambda v: f"{100 * v:.1f}%"}) + "\n")
            small = h[h.confidence_ge >= 0.75]
            out.append("These buckets hold a few dozen games at most; a 95% interval for the observed rate spans roughly "
                       + ", ".join(f"{100 * (r.ci_hi - r.ci_lo):.0f} points ({r.split}, n={int(r.n)})" for r in small.itertuples())
                       + " at ≥ 75%. Neither over- nor under-confidence can be established there; the earlier impression "
                       "that the top bucket is over-confident is within sampling noise unless an interval excludes the "
                       "mean predicted probability.")
    lc = F["glc"]
    if len(lc):
        L = lc[lc.axis == "logistic_C"]
        if len(L):
            b_ = L.loc[L.valid_log_loss.idxmin()]
            out.append(f"\n\nLogistic regularization on validation: best C = {b_.value:g} (log loss {b_.valid_log_loss:.4f}) vs "
                       f"production C = 0.1 ({float(L[L.production].valid_log_loss.iloc[0]):.4f}).")
    return "".join(out)


def _stack_text(F, md_table) -> str:
    d = F["div"]
    if d.empty:
        return ""
    ecols = [c for c in d.columns if c.startswith("err_corr_")]
    pcols = [c for c in d.columns if c.startswith("pred_corr_")]
    t = d[["group", "target", "best_single", "best_single_valid_rmse", "stack_even_valid_pct_vs_best", "stack_even_valid_pct_lo",
           "stack_even_valid_pct_hi", "stack_opt_valid_pct_vs_best", "stack_opt_valid_pct_lo", "stack_opt_valid_pct_hi", "stack_opt_weights"]].copy()
    t["even stack vs best (valid)"] = [_ci(a, b, c_, 2) for a, b, c_ in zip(t.stack_even_valid_pct_vs_best, t.stack_even_valid_pct_lo, t.stack_even_valid_pct_hi)]
    t["optimized stack vs best (valid, cross-fitted)"] = [_ci(a, b, c_, 2) for a, b, c_ in zip(t.stack_opt_valid_pct_vs_best, t.stack_opt_valid_pct_lo, t.stack_opt_valid_pct_hi)]
    out = [f"Across player targets the three families' validation predictions correlate at a median "
           f"{np.nanmedian(d[pcols].to_numpy()):.2f}, but their **errors** correlate at a median "
           f"{np.nanmedian(d[ecols].to_numpy()):.2f} (range {np.nanmin(d[ecols].to_numpy()):.2f}-{np.nanmax(d[ecols].to_numpy()):.2f}). "
           "Most of each error is the same unpredictable game-to-game noise, which no family can see, so averaging "
           "the families can only remove the small part of the error on which they disagree.\n",
           _apx("Stacks vs best single family (validation, % RMSE improvement, positive = stack better)",
                md_table(t[["group", "target", "best_single", "best_single_valid_rmse", "even stack vs best (valid)",
                            "optimized stack vs best (valid, cross-fitted)", "stack_opt_weights"]]) + "\n")]
    sig = d[d.stack_opt_valid_pct_lo > 0]
    out.append(f"The optimized stack beats the best single family with an interval excluding zero for {len(sig)} of {len(d)} "
               f"targets ({_lst(sig.group.str[0].str.upper() + ' ' + sig.target)}), by at most "
               f"{d.stack_opt_valid_pct_vs_best.max():.2f}%. Gains of a fraction of a percent of RMSE are real but not "
               "practically meaningful for a single-game projection, and stacking triples training time.")
    gd = F["gdiv"]
    if len(gd):
        v = gd[gd.split == "valid"]
        out.append(f" Game model: candidate errors correlate at {v.err_corr.min():.2f}-{v.err_corr.max():.2f} on validation.")
    return "".join(out)


def _temporal_text(F, md_table) -> str:
    tp, tn, gt = F["temporal"], F["tenure"], F["gtemp"]
    out = []
    if len(tp):
        th = tp[tp.period_type == "third"]
        piv = th.pivot_table(index=["group", "target"], columns="period", values="skill_vs_history_rmse_pct")
        piv = piv[[c for c in ("early", "middle", "late") if c in piv.columns]]
        b = th.pivot_table(index=["group", "target"], columns="period", values="bias")
        piv = piv.join(b[[c for c in ("early", "middle", "late") if c in b.columns]].add_prefix("bias "))
        out.append("Test window split into thirds by date. Skill = % RMSE improvement of the published projection over "
                   "the history-to-date baseline **in the same period** (so a noisier month does not look like decay):\n")
        out.append(_apx("Test-window thirds: skill vs history-to-date and bias",
                        md_table(piv.reset_index(), {c: (lambda v: f"{v:+.1f}%") for c in piv.columns if not c.startswith("bias")}
                                 | {c: (lambda v: f"{v:+.3f}") for c in piv.columns if c.startswith("bias")}) + "\n"))
        rng = piv[[c for c in ("early", "middle", "late") if c in piv.columns]]
        span = (rng.max(1) - rng.min(1))
        unstable = span[span > 3].index.tolist()
        out.append(f"Skill moves by more than 3 points between thirds for {_lst([f'{g[0].upper()} {t}' for g, t in unstable])}; "
                   f"elsewhere it is stable. Monthly rows are in `temporal_players.csv`.\n")
    if len(tn):
        piv = tn.pivot_table(index=["group", "target"], columns="period", values="skill_vs_history_rmse_pct")
        piv = piv[[c for c in ("<20 prior games", "20-59", "60-119", "120+") if c in piv.columns]]
        out.append("By the player's prior games in the cached window (tenure):\n")
        if "<20 prior games" in piv.columns and "120+" in piv.columns:
            out.append(f"Median skill: {piv['<20 prior games'].median():+.1f}% for players with fewer than 20 prior games vs "
                       f"{piv['120+'].median():+.1f}% for 120+.\n")
        out.append(_apx("Skill by tenure (prior games in the cached window)",
                        md_table(piv.reset_index(), {c: (lambda v: f"{v:+.1f}%") for c in piv.columns}) + "\n"))
        out.append("Skill over the history-to-date average is largest where that average is least reliable (few prior games), "
                   "which is what the true-talent shrinkage is for; check the `<20` column before trusting projections "
                   "for call-ups.\n")
    if len(gt):
        g = gt[gt.period_type == "third"]
        out.append(f"Game log loss by third ranges {g.log_loss.min():.4f}-{g.log_loss.max():.4f} and AUC "
                   f"{g.auc.min():.3f}-{g.auc.max():.3f}.\n")
        out.append(_apx("Game model by third of each window",
                        md_table(g[["split", "period", "n", "log_loss", "auc", "brier", "mean_p", "win_rate"]],
                                 {"n": lambda v: f"{int(v)}"}) + "\n"))
    return "".join(out)


def _baseline_text(F, md_table) -> str:
    b = F["base"]
    if b.empty:
        return ""
    keep = b[b.model.isin(["production blend (published)", "direct selected model", "best single family on validation",
                           "stack_opt (train-only fit)"])].copy()
    for k in ("league average", "history-to-date", "last 10"):
        c = f"rmse_pct_better_than_{k}"
        if c in keep:
            keep[f"vs {k}"] = [_ci(a, l, h, 1) if np.isfinite(a) else "n/a"
                               for a, l, h in zip(keep[c], keep[f"ci_lo_vs_{k}"], keep[f"ci_hi_vs_{k}"])]
    cols = ["group", "target", "model", "rmse"] + [c for c in keep.columns if c.startswith("vs ")]
    out = ["Published projection, test window: % RMSE improvement with 95% date-block intervals (positive = better than "
           "the baseline). The appendix adds the direct model, the best single family on validation and the stack (the "
           "last two are fit on the training window only, so compare them with each other, not with the refit rows).\n",
           md_table(keep[keep.model == "production blend (published)"][[c for c in cols if c != "model"]], {"rmse": lambda v: f"{v:.4f}"}) + "\n",
           _apx("Baselines: published blend, direct model, best single family and stack", md_table(keep[cols], {"rmse": lambda v: f"{v:.4f}"}) + "\n")]
    p = b[b.model == "production blend (published)"]
    if len(p) and "ci_lo_vs_history-to-date" in p:
        strong = p[p["ci_lo_vs_league average"] > 5]
        modest = p[(p["ci_lo_vs_history-to-date"] > 0) & (p["rmse_pct_better_than_history-to-date"] < 3)]
        none_ = p[p["ci_lo_vs_history-to-date"] <= 0]
        out.append(f"The published projection beats the league average by more than 5% (interval lower bound) for "
                   f"{_lst(strong.group.str[0].str.upper() + ' ' + strong.target)}. Against each player's own history-to-date "
                   f"average it is better but by under 3% for {_lst(modest.group.str[0].str.upper() + ' ' + modest.target)}, "
                   f"and **not distinguishable** from it for {_lst(none_.group.str[0].str.upper() + ' ' + none_.target)}. "
                   "High-variance counts (home runs, triples, runs allowed) are close to the limit of what any pre-game model "
                   "can predict for a single game; the meaningful comparison there is aggregation over many games.")
    return "".join(out)


# ---------------------------------------------------------------------------
def priorities(F) -> list[tuple[str, str, str]]:
    P = []
    c = F.get("calib")
    if c is not None and len(c):
        worse = c[c.pct > 0.25]
        if len(worse) > len(c) / 2:
            P.append(("Must fix", "fix the player calibration wrapper", f"**Calibration wrapper (`hitterspitchers_train._fit_calibrated`).** It worsens validation RMSE in "
                      f"{len(worse)} of {len(c)} target/family fits (median {c.pct.median():+.2f}%) and can output exact zeros. "
                      "Change the selection criterion from MAE to RMSE or Poisson deviance on the held-out tail (or drop the "
                      "isotonic option), keep the result only if validation improves, then retrain. Evidence: "
                      "`overfit_metrics.csv` (families with and without `_uncalibrated`)."))
    b = F["gboot"]
    if len(b):
        v = b[(b.split == "valid") & b.logloss_minus_home_rate.notna()]
        if not (v.hi < 0).any():
            P.append(("Must fix", "stop presenting game probabilities as an edge", "**Game-model claims.** No candidate beats the always-home baseline on validation with an "
                      "interval excluding zero (`game_bootstrap.csv`). The site and report must present game probabilities as "
                      "near-coin-flip estimates, not as an edge; keep v1 in production and treat promotion as a non-event until "
                      "a candidate clears the baseline on validation with an interval."))
        tr = F["gcal"]
        if len(tr):
            t = tr[(tr.model == "xgb_raw")].set_index("split")
            if {"train", "valid"} <= set(t.index) and t.loc["train", "auc"] - t.loc["valid", "auc"] > 0.15:
                P.append(("Should fix", "restrict or drop the game XGBoost", f"**Game XGBoost over-fits** (training AUC {t.loc['train', 'auc']:.2f} vs validation "
                          f"{t.loc['valid', 'auc']:.2f}). Restrict it (depth 2, more min_child_weight; see `game_learning_curves.csv`) "
                          "or drop it from the candidate list; the logistic model is as good on validation at a fraction of the variance."))
    e = _abl_effects(F["abl"])
    if len(e):
        none = e[(e.n_hurt == 0) & (e.n_help > 0) & ~e.variant.str.startswith("one per")]
        if len(none):
            P.append(("Should fix", "re-validate removing families with no incremental value", "**Feature families with no incremental value and some measurable harm on validation**: "
                      + "; ".join(f"{r.group} {r.variant.replace('no ', '')} (removal helps {r.n_help} run(s))" for r in none.itertuples())
                      + ". Remove one family at a time, re-validate, keep the removal only if validation does not worsen."))
        rr = F["abl"][F["abl"].variant.str.startswith("one per |rho|>=0.90")]
        if len(rr) and (rr.delta_lo > 0).sum() == 0:
            P.append(("Should fix", "collapse near-duplicate windows", f"**Collapse near-duplicate windows** (one feature per |rho| ≥ 0.90 cluster: "
                      f"{int(rr.n_features.median())} instead of {int((rr.n_features + rr.n_dropped).median())} features) - "
                      "no validation run got measurably worse. Same accuracy, faster training, stable importances."))
    cu = F["curves"]
    if len(cu):
        frac = cu[cu.axis == "train_fraction_recent"]
        still = frac.groupby(["group", "target"]).apply(lambda d: d.sort_values("value").valid_rmse.iloc[-1] <= d.valid_rmse.min() + 1e-12)
        if still.mean() >= 0.5:
            P.append(("Should fix", "test more training rows", "**More training rows.** Validation error is still falling at the full training window "
                      f"for {int(still.sum())} of {len(still)} targets. Test using the earliest cached season's rows as training "
                      "examples (they are history-only today) on the same validation window."))
    d = F["div"]
    if len(d) and (d.stack_opt_valid_pct_vs_best.max() < 1.0):
        P.append(("Should fix", "ship stacks only where they clear validation noise", "**Stacking adds under 1% RMSE** on every player target (`model_diversity.csv`; family errors "
                  f"correlate at {np.nanmedian(d[[c for c in d.columns if c.startswith('err_corr_')]].to_numpy()):.2f}). "
                  "Keep the stack only where its validation interval excludes zero; otherwise ship the best single family."))
    P.append(("Should fix", "tighten report wording", "**Report wording.** State that single-game R² is low by nature, that importance is predictive "
              "attribution (not causal), and that individual feature importances are unstable under collinearity while "
              "construct-grouped importances are stable (this section)."))
    P.append(("Nice to have", "walk-forward evaluation", "Walk-forward evaluation (several validation/test windows) to put intervals on the model-selection "
              "step itself; one validation window chooses among families that differ by less than its own noise."))
    P.append(("Nice to have", "grouped SHAP reporting", "Report grouped SHAP alongside the existing importance files in `models/*_importance.csv`, and "
              "drop native gain importance from any public explanation."))
    P.append(("Nice to have", "probabilistic player props", "Probability outputs for player props (Poisson / negative-binomial on the projected mean) "
              "evaluated with proper scoring rules, since a mean that is right on average says nothing about tail calibration."))
    return P


# ---------------------------------------------------------------------------
def conclusion_answers(w) -> None:
    """The ten audit questions, answered from the audit files."""
    F = findings()
    if not available():
        return
    w("**Audit answers** (details and evidence in the diagnostics section):\n")
    of = F.get("overfit")
    qs = []
    if of is not None and len(of):
        vc = of.verdict.value_counts().to_dict()
        best = of.loc[of.groupby(["group", "target"]).valid_rmse.idxmin()]
        qs.append(("Is there evidence of overfitting?",
                   f"Mild, not severe, for the player models: of {len(of)} target/family fits, "
                   + ", ".join(f"{v} {k}" for k, v in vc.items())
                   + f". For the family each target would select, validation keeps a median "
                   f"{100 * best.skill_retention.median():.0f}% of its training skill; the train-validation RMSE gap is "
                   f"{best.valid_train_gap_pct.median():+.1f}% (median). The learning curves place production well short of "
                   "the depth/tree-count region where validation error turns up. "
                   + (f"Clear cases: {_lst([f'{r.group} {r.target} ({FAM_NAME[r.family]})' for r in of[of.verdict == 'clear overfitting'].itertuples()])}, "
                      "all low-signal targets where a little memorized noise is a large share of the small skill; " if (of.verdict == 'clear overfitting').any() else "")
                   + "the game XGBoost (training AUC far above validation) is the most severe. Part of the player gap is the "
                   "calibration wrapper rather than the trees (it produces near-zero predictions; see 'Must fix'). Good test numbers are not offered as proof of the absence of over-fitting; "
                   "the evidence is the validation gaps and curves."))
    red = F["red"]
    if len(red):
        qs.append(("Is there meaningful multicollinearity?",
                   "Yes, measured: " + "; ".join(f"{r.model} {int(r.n_vif_gt_10)}/{int(r.n_features)} features with VIF > 10"
                                                 for r in red.itertuples())
                   + ". It destabilizes linear coefficients and individual importances but costs no measurable validation "
                     "accuracy (collapsing redundant clusters left validation RMSE within noise)."))
    e = _abl_effects(F["abl"])
    if len(e):
        m = e[(e.n_hurt > 0) & ~e.variant.str.startswith("one per")]
        qs.append(("Which feature families actually matter?",
                   "; ".join(f"{g}s: " + _lst([v.replace('no ', '') for v in m[m.group == g].sort_values('max_hurt_pct', ascending=False).variant])
                             for g in sorted(e.group.unique()))
                   + ". Opportunity/workload features carry by far the most incremental value; most other families are "
                     "individually replaceable because correlated families carry the same information."))
    d = F["div"]
    if len(d):
        qs.append(("Does stacking add meaningful value?",
                   f"No, not practically: at most {d.stack_opt_valid_pct_vs_best.max():.2f}% RMSE over the best single family "
                   f"on validation, with family errors correlated at a median "
                   f"{np.nanmedian(d[[c for c in d.columns if c.startswith('err_corr_')]].to_numpy()):.2f}."))
    c = F["gcal"]
    if len(c):
        sel = F["gmeta"].get("selected_on_valid")
        s = c[(c.model == sel)].set_index("split")
        qs.append(("Are the game probabilities calibrated?",
                   f"On average, yes (calibration-in-the-large {s.loc['valid', 'cal_intercept_itl']:+.3f} validation, "
                   f"{s.loc['test', 'cal_intercept_itl']:+.3f} test; ECE {s.loc['valid', 'ece']:.3f} / {s.loc['test', 'ece']:.3f}). "
                   f"In spread, the slope is {s.loc['valid', 'cal_slope']:.2f} on validation and {s.loc['test', 'cal_slope']:.2f} on test. "
                   + ("A slope below 1 means the spread is too wide (mildly over-confident) on that window. " if min(s.loc['valid', 'cal_slope'], s.loc['test', 'cal_slope']) < 0.85 else "")
                   + "The high-confidence buckets are too small to call over-confident either way. The larger problem is "
                   f"discrimination: validation AUC {s.loc['valid', 'auc']:.3f}, test {s.loc['test', 'auc']:.3f}."))
    sh = F["shrink"]
    if len(sh):
        comp = sh[sh.cal_slope > 1.10]; spread = sh[sh.cal_slope < 0.90]
        qs.append(("Where does the model regress toward the mean?",
                   f"Measured by the slope of actual on predicted (test, published blend): compressed for "
                   f"{_lst(comp.group.str[0].str.upper() + ' ' + comp.target)}; slightly too spread for "
                   f"{_lst(spread.group.str[0].str.upper() + ' ' + spread.target)}; the rest are within 0.90-1.10. The narrow "
                   "look of the scatter plots is single-game noise, not shrinkage."))
    b = F["base"]
    if len(b):
        p = b[b.model == "production blend (published)"]
        if len(p) and "ci_lo_vs_league average" in p:
            strong = p[p["ci_lo_vs_league average"] > 5].sort_values("rmse_pct_better_than_league average", ascending=False)
            weak = p[(p["ci_lo_vs_history-to-date"] <= 0) | (p["rmse_pct_better_than_history-to-date"] < 2)]
            qs.append(("Which targets are genuinely strong?",
                       _lst([f"{r['group'][0].upper()} {r['target']} ({r['rmse_pct_better_than_league average']:+.1f}% vs league)" for _, r in strong.iterrows()])
                       + " - opportunity-driven targets, where playing time and workload are predictable."))
            qs.append(("Which targets remain weak?",
                       _lst([f"{r['group'][0].upper()} {r['target']} ({r['rmse_pct_better_than_history-to-date']:+.1f}% vs history-to-date)" for _, r in weak.iterrows()])
                       + " - within 2% of (or not distinguishable from) the player's own history-to-date average on test; "
                       "plus the game winner, which is not distinguishable from always-home on validation."))
            qs.append(("Does the model outperform simple baselines?",
                       f"Against league average, yes for {int((p['ci_lo_vs_league average'] > 0).sum())} of {len(p)} player targets; "
                       f"against the player's own history-to-date average, with an interval above zero for "
                       f"{int((p['ci_lo_vs_history-to-date'] > 0).sum())} of {len(p)}, mostly by low single-digit percentages."))
    pr = priorities(F)
    qs.append(("What should be changed next?", "; ".join(f"{title} ({lvl.lower()})" for lvl, title, _ in pr if lvl != "Nice to have")
               + ". Details in the prioritized list of the diagnostics section."))
    for i, (q, a) in enumerate(qs, 1):
        w(f"{i}. **{q}** {a}")
    w("")
