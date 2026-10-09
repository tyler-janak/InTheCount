"""
audit_models.py
===============
Statistical / modeling audit of the InTheCount projection system.

Diagnostics only. Nothing here writes to models/, betting_model*.pkl or the
production prediction files, and no result is fed back into training.

Protocol (identical to run_backtest.py / train_game_model.py)
-------------------------------------------------------------
    train       game_date <  VALID_START                (every diagnostic model is fit here)
    validation  VALID_START <= game_date < TEST_START   (every comparison that could inform a decision)
    test        game_date >= TEST_START                 (reported only, never used to choose anything)

Steps (each writes outputs/model_evaluation/audit/metrics/*.csv and is cached,
so an interrupted run resumes; --force reruns everything)

    map          feature inventory: family, construct, coverage by split, saved-bundle check
    redundancy   Pearson / Spearman / VIF / |r| thresholds / clusters      (train rows only)
    fits         per target x family (rf, xgb, lin) fit on train exactly as production
                 (calibration wrapper included): train / valid / test RMSE, MAE, Poisson
                 deviance, calibration slope, native + SHAP + grouped permutation importance
                 (validation), importance across chronological folds, linear-coefficient
                 stability, prediction / error correlation between families, stacks
    curves       XGBoost learning curves: trees, depth, min_child_weight, reg_lambda,
                 training-sample size                                      (train -> validation)
    ablation     feature-family ablations + redundancy-reduced feature sets (train -> validation;
                 test shown as a report-only column)
    game         the same diagnostics for the game win-probability model, plus calibration
                 (intercept, slope, ECE, reliability buckets with Wilson intervals)
    saved        deciles / shrinkage, temporal stability, tenure groups and baseline comparisons
                 with date-block bootstrap intervals, from the saved TEST predictions
    figures      the eight audit figures

Usage
-----
    python audit_models.py                      # everything (resumes from cache)
    python audit_models.py --steps game saved figures
    python audit_models.py --groups pitcher     # player steps for one group only
    python audit_models.py --quick              # 35% row sample of hitter train (smoke test only)
    python audit_models.py --targets K H PA     # a subset of player targets (faster)
    python audit_models.py --force
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import evaluate_models as ev                      # noqa: E402  (shared figure style)
import hitterspitchers_train as hpt               # noqa: E402
import model_stacking as stk                      # noqa: E402
import run_backtest as rb                         # noqa: E402

import matplotlib                                 # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                   # noqa: E402

HERE = Path(__file__).resolve().parent
EVAL = HERE / "outputs" / "model_evaluation"
AUD = EVAL / "audit"
AM, AF, AC = AUD / "metrics", AUD / "figures", AUD / "cache"
SEED = 42
N_BOOT = 1000

_run = {}
try:
    _run = json.loads((EVAL / "metrics" / "backtest_run.json").read_text())
except Exception:
    pass
VALID_START = _run.get("valid_start", "2026-03-01")
TEST_START = _run.get("test_start", "2026-07-01")

PITCHER_COUNTS = ["K", "BB", "H", "HR", "IP", "R"]
HITTER_COUNTS = ["H", "HR", "TB", "BB", "K", "PA", "2B", "3B"]
CURVE_TARGETS = {"pitcher": ["K", "IP", "H"], "hitter": ["H", "TB", "PA"]}
FAMILIES = ["rf", "xgb", "lin"]
QUICK = False
ONLY_TARGETS: list[str] | None = None


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _out(name, df):
    """Write a result table. A file held open by Excel or locked by OneDrive sync
    raises PermissionError on Windows: retry for ~20 s, then stop with a clear
    message (everything computed so far is cached, so rerunning resumes)."""
    AM.mkdir(parents=True, exist_ok=True)
    for attempt in range(10):
        try:
            df.to_csv(AM / name, index=False)
            return df
        except PermissionError:
            if attempt == 0:
                log(f"  {name} is locked (open in Excel, or OneDrive syncing) - retrying for 20 s")
            time.sleep(2)
    raise SystemExit(f"\nCould not write {AM / name}: close it in Excel (or pause OneDrive), then rerun the same "
                     "command - finished steps are cached and will not be recomputed.")


def _cached(name, force):
    p = AM / name
    return (not force) and p.exists()


# ---------------------------------------------------------------------------
# Losses and small statistics
# ---------------------------------------------------------------------------
def pois_dev_rows(y, p):
    y = np.asarray(y, float); p = np.clip(np.asarray(p, float), 1e-6, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(y > 0, y * np.log(y / p), 0.0)
    return 2.0 * (t - (y - p))


def reg_block(y, p, prefix=""):
    y = np.asarray(y, float); p = np.asarray(p, float)
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], p[ok]
    if len(y) < 3:
        return {}
    e = p - y
    out = {"n": len(y), "rmse": float(np.sqrt(np.mean(e ** 2))), "mae": float(np.mean(np.abs(e))),
           "poisson_dev": float(np.mean(pois_dev_rows(y, p))), "bias": float(np.mean(e)),
           "mean_actual": float(np.mean(y)), "mean_pred": float(np.mean(p)),
           "share_pred_lt_0.05": float(np.mean(p < 0.05))}
    out["cal_slope"] = cal_slope(y, p)
    return {f"{prefix}{k}": v for k, v in out.items()}


def cal_slope(y, p):
    """OLS slope of actual on predicted. 1 = spread right; >1 = predictions too
    compressed toward the mean (under-dispersed / over-shrunk); <1 = too spread
    out (the classic over-fitting signature)."""
    y = np.asarray(y, float); p = np.asarray(p, float)
    v = np.var(p)
    return float(np.cov(p, y, bias=True)[0, 1] / v) if v > 1e-12 else float("nan")


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def logloss_rows(y, p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6); y = np.asarray(y, float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def prob_block(y, p, prefix=""):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y, float); p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    out = {"n": len(y), "log_loss": float(np.mean(logloss_rows(y, p))),
           "brier": float(np.mean((p - y) ** 2)),
           "auc": float(roc_auc_score(y, p)) if len(set(y)) == 2 else float("nan"),
           "accuracy": float(np.mean((p > 0.5) == (y == 1))),
           "mean_p": float(np.mean(p)), "win_rate": float(np.mean(y))}
    a, b = logistic_recal(y, p)
    out.update({"cal_intercept_itl": cal_in_the_large(y, p), "cal_slope": b, "cal_intercept": a,
                "ece": ece(y, p)})
    return {f"{prefix}{k}": v for k, v in out.items()}


def logistic_recal(y, p, iters=50):
    """Fit y ~ sigmoid(a + b*logit(p)) by Newton's method; returns (a, b)."""
    z = logit(p); X = np.column_stack([np.ones_like(z), z]); w = np.array([0.0, 1.0])
    for _ in range(iters):
        q = sigmoid(X @ w); g = X.T @ (y - q); H = (X * (q * (1 - q))[:, None]).T @ X
        try:
            step = np.linalg.solve(H + 1e-9 * np.eye(2), g)
        except np.linalg.LinAlgError:
            break
        w = w + step
        if np.max(np.abs(step)) < 1e-8:
            break
    return float(w[0]), float(w[1])


def cal_in_the_large(y, p, iters=50):
    """Intercept a with slope fixed at 1 (offset logit(p)): 0 = average level right."""
    z = logit(p); a = 0.0
    for _ in range(iters):
        q = sigmoid(a + z); g = np.sum(y - q); h = np.sum(q * (1 - q))
        if h <= 0:
            break
        a += g / h
        if abs(g / h) < 1e-9:
            break
    return float(a)


def ece(y, p, bins=10):
    y = np.asarray(y, float); p = np.asarray(p, float)
    edges = np.linspace(0, 1, bins + 1); idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    tot = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            tot += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(tot)


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    ph = k / n; den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den; h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return (c - h, c + h)


def block_boot(dates, rows: dict, stat, n=N_BOOT, seed=SEED):
    """Bootstrap by resampling whole game dates (rows on the same day share
    weather, umpires, lineups ...). `rows` maps name -> per-row arrays; `stat`
    gets a dict of per-date-summed arrays plus 'cnt' and returns a scalar.
    Returns (point, lo, hi)."""
    d = pd.Series(pd.to_datetime(dates)).dt.normalize().to_numpy()
    codes, uniq = pd.factorize(d)
    D = len(uniq)
    sums = {k: np.bincount(codes, weights=np.asarray(v, float), minlength=D) for k, v in rows.items()}
    sums["cnt"] = np.bincount(codes, minlength=D).astype(float)
    point = stat({k: v.sum() for k, v in sums.items()})
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, D, size=(n, D))
    reps = np.array([stat({k: v[i].sum() for k, v in sums.items()}) for i in idx])
    return float(point), float(np.nanpercentile(reps, 2.5)), float(np.nanpercentile(reps, 97.5))


def boot_rmse_diff(dates, y, pa, pb, n=N_BOOT):
    """RMSE(a) - RMSE(b) and % improvement of a over b, date-block bootstrap."""
    ea, eb = (np.asarray(pa) - y) ** 2, (np.asarray(pb) - y) ** 2
    diff = block_boot(dates, {"a": ea, "b": eb}, lambda s: np.sqrt(s["a"] / s["cnt"]) - np.sqrt(s["b"] / s["cnt"]), n)
    pct = block_boot(dates, {"a": ea, "b": eb}, lambda s: 100 * (1 - np.sqrt(s["a"] / s["cnt"]) / np.sqrt(s["b"] / s["cnt"])), n)
    return diff, pct


# ---------------------------------------------------------------------------
# Feature taxonomy
# ---------------------------------------------------------------------------
OPP_STEMS = ("pa", "ip", "bf", "outs", "pitches", "bf_per_ip", "pitches_per_bf", "pitches_per_ip",
             "max_ip", "starter_pct", "days_rest", "days_since_game")
BATTED = ("avg_ev", "max_ev", "avg_la", "avg_direction", "barrel", "hard_hit", "sweet_spot", "blast",
          "ev_la", "ev_spread", "times_on_base", "xbh_proxy", "avg_ev_allowed")
STUFF = ("avg_velocity", "avg_spin", "whiff", "csw", "zone_pct", "f_strike", "fb_pct", "br_pct",
         "off_pct", "pitch_pct", "fb_velo", "br_velo", "off_velo", "velo_sep")
WINDOW_RE = re.compile(r"_(last\d+|std)(_[LR])?$")


def stem(f):
    return WINDOW_RE.sub("", f)


def family(f: str) -> str:
    """Construction family (used for ablations). Single primary family with
    precedence lineup > handedness > true-talent > matchup > opponent/team >
    park > batted-ball > velocity/pitch-mix > opportunity > rolling > expanding."""
    s = f.lower()
    if s.startswith("lineup_spot"):
        return "lineup"
    if "_vs_hand" in s:
        return "team/opponent" if s.startswith("team_") else "handedness"
    if s.startswith(("p_tt_", "h_tt_")):
        return "true_talent"
    if s.startswith(("lineup_tt_", "matchup_")):
        return "matchup"
    if s.startswith(("opp_sp_", "team_", "lineup_")):
        return "team/opponent"
    if s == "park_factor":
        return "park"
    if any(s.startswith(b) for b in BATTED):
        return "batted_ball"
    if any(s.startswith(b) for b in STUFF):
        return "velocity/pitch_mix"
    st = stem(s)
    if st in OPP_STEMS or st.startswith(("max_ip", "starter_pct")):
        return "opportunity"
    if re.search(r"_last\d+", s):
        return "rolling"
    if s.endswith("_std"):
        return "expanding"
    return "other"


def construct(f: str) -> str:
    """What the feature measures (used for grouped importance)."""
    s = f.lower()
    if s.startswith("lineup_spot"):
        return "lineup slot"
    if s == "park_factor":
        return "park"
    if s.startswith(("opp_sp_", "team_", "lineup_tt_", "lineup_")) and not s.startswith("lineup_spot"):
        return "opponent"
    if s.startswith("matchup_"):
        return "matchup (log5)"
    st = stem(s)
    if st in OPP_STEMS or st.startswith(("max_ip", "starter_pct")):
        return "playing time / workload"
    if any(s.startswith(b) for b in STUFF):
        return "stuff / pitch mix" if not s.startswith(("whiff", "csw")) else "strikeout skill"
    if any(s.startswith(b) for b in BATTED):
        return "contact quality"
    toks = set(re.split(r"_", s))
    if "k" in toks:
        return "strikeout skill"
    if "bb" in toks:
        return "walk skill"
    if "hr" in toks:
        return "home-run skill"
    if toks & {"2b", "3b"}:
        return "extra-base hits"
    if "sb" in toks:
        return "stolen bases"
    if "h" in toks:
        return "hit skill"
    if "r" in toks:
        return "runs allowed"
    return "other"


GAME_FAMILY = {  # construction family for the game model
    "diff_runs_scored_10g": "rolling", "diff_runs_allowed_10g": "rolling", "diff_OPS_10g": "rolling",
    "diff_K_rate_10g": "rolling", "diff_BB_rate_10g": "rolling", "diff_SP_FIP_10": "rolling",
    "diff_SP_KBB_pct_10": "rolling", "diff_SP_HR9_10": "rolling", "diff_BP_ERA_proxy_14g": "rolling",
    "diff_BP_WHIP_14g": "rolling", "diff_BP_KBB_pct_14g": "rolling",
    "diff_SP_FIP_season": "expanding", "diff_SP_KBB_pct_season": "expanding",
    "diff_team_winpct": "expanding", "diff_team_run_diff": "expanding", "diff_season_OPS": "expanding",
    "home_field": "other", "diff_starter_hand": "handedness",
}


def game_construct(f):
    if "_SP_" in f:
        return "starting pitcher"
    if "_BP_" in f:
        return "bullpen"
    if f in ("diff_team_winpct", "diff_team_run_diff", "diff_season_OPS"):
        return "team strength (season)"
    if f in ("home_field", "diff_starter_hand"):
        return "home field / hand"
    if f == "diff_runs_allowed_10g":
        return "run prevention (10g)"
    return "offense (10g)"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
_DATA: dict = {}


def load_group(group: str) -> dict:
    if group in _DATA:
        return _DATA[group]
    log(f"loading {group} table")
    df = rb._load_table(rb.FEAT, group)
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    idc = "pitcher" if group == "pitcher" else "batter"
    df = df.sort_values(["game_date"] + (["game_pk"] if "game_pk" in df.columns else []), kind="mergesort")
    df["n_prior_games"] = df.groupby(idc).cumcount()          # games inside the cached history window
    seasons = rb.training_seasons(df)
    df = df[df["game_date"].dt.year.isin(seasons)]
    if group == "hitter":
        df = hpt.clean_hitter_training_rows(df)
        feats_all, counts, attach = hpt.HITTER_FEATURES, HITTER_COUNTS, hpt._attach_hitter_rates
    else:
        if "is_actual_starter" in df.columns:
            df = df[pd.to_numeric(df["is_actual_starter"], errors="coerce").fillna(0) == 1]
        feats_all, counts, attach = hpt.PITCHER_FEATURES, PITCHER_COUNTS, hpt._attach_pitcher_rates
    tr, va, te = hpt.date_split(df, "game_date", VALID_START, TEST_START)
    if QUICK and group == "hitter":
        tr = tr.sample(frac=0.35, random_state=SEED).sort_values("game_date", kind="mergesort")
    tr, va, te = attach(tr), attach(va), attach(te)
    feats = hpt.select_features(tr, feats_all)
    for d in (tr, va, te):
        d[feats] = d[feats].apply(pd.to_numeric, errors="coerce")
    counts = [t for t in counts if t in tr.columns and (ONLY_TARGETS is None or t in ONLY_TARGETS)]
    _DATA[group] = {"train": tr.reset_index(drop=True), "valid": va.reset_index(drop=True),
                    "test": te.reset_index(drop=True), "features": feats, "listed": feats_all,
                    "targets": counts, "seasons": seasons, "id": idc}
    log(f"  {group}: train {len(tr):,}  valid {len(va):,}  test {len(te):,}  features {len(feats)}")
    return _DATA[group]


def presets(group, target):
    p = rb._presets(group, [target])
    return p.get(target, {})


# ---------------------------------------------------------------------------
# 1. Pipeline map
# ---------------------------------------------------------------------------
def step_map(groups, force):
    prev = pd.read_csv(AM / "feature_inventory.csv") if (AM / "feature_inventory.csv").exists() and not force else pd.DataFrame()
    if len(prev) and set(groups) <= set(prev.group):
        return
    rows = prev[~prev.group.isin(groups + ["game"])].to_dict("records") if len(prev) else []
    for g in groups:
        D = load_group(g)
        bundles = {}
        for t in D["targets"]:
            p = HERE / "models" / f"{g}_{t}.pkl"
            if p.exists():
                try:
                    bundles[t] = set(pickle.load(open(p, "rb")).get("features", []))
                except Exception:
                    pass
        for f in D["listed"]:
            r = {"group": g, "feature": f, "family": family(f), "construct": construct(f), "stem": stem(f),
                 "used": f in D["features"]}
            for split in ("train", "valid", "test"):
                d = D[split]
                r[f"coverage_{split}"] = float(d[f].notna().mean()) if f in d.columns else 0.0
            r["coverage_gap_train_test"] = r["coverage_train"] - r["coverage_test"]
            r["in_saved_bundles"] = (sum(f in b for b in bundles.values()) / len(bundles)) if bundles else np.nan
            rows.append(r)
    import train_game_model as tgm
    gd = tgm.load_table()
    vs, ts = pd.Timestamp(VALID_START), pd.Timestamp(TEST_START)
    for f in tgm.FEATURES:
        r = {"group": "game", "feature": f, "family": GAME_FAMILY.get(f, "other"), "construct": game_construct(f),
             "stem": f, "used": True}
        for split, m in (("train", gd.game_date < vs), ("valid", (gd.game_date >= vs) & (gd.game_date < ts)),
                         ("test", gd.game_date >= ts)):
            r[f"coverage_{split}"] = float(gd.loc[m, f].notna().mean()) if f in gd.columns else 0.0
        r["coverage_gap_train_test"] = r["coverage_train"] - r["coverage_test"]
        rows.append(r)
    _out("feature_inventory.csv", pd.DataFrame(rows))
    log("map: feature_inventory.csv")


# ---------------------------------------------------------------------------
# 2. Redundancy (train rows only)
# ---------------------------------------------------------------------------
def _vif(X: pd.DataFrame) -> pd.Series:
    Z = X.fillna(X.median())
    Z = Z.loc[:, Z.std() > 1e-12]
    Z = (Z - Z.mean()) / Z.std()
    R = np.corrcoef(Z.to_numpy().T)
    inv = np.linalg.pinv(R, hermitian=True)
    v = pd.Series(np.diag(inv), index=Z.columns).clip(lower=1.0)
    ev_ = np.linalg.eigvalsh(R)
    v.attrs["condition_number"] = float(np.sqrt(ev_.max() / max(ev_.min(), 1e-12)))
    return v


def _clusters(S: pd.DataFrame, thr: float, coverage: pd.Series) -> dict:
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    A = S.abs().fillna(0).to_numpy().copy(); np.fill_diagonal(A, 1.0)
    Dm = np.clip(1 - A, 0, None); np.fill_diagonal(Dm, 0)
    lab = fcluster(linkage(squareform(Dm, checks=False), "complete"), t=1 - thr, criterion="distance")
    out = {}
    cols = list(S.columns)
    for c in np.unique(lab):
        mem = [cols[i] for i in np.where(lab == c)[0]]
        if len(mem) == 1:
            rep = mem[0]
        else:   # medoid, ties broken by coverage
            sub = S.loc[mem, mem].abs()
            score = sub.mean() + 1e-3 * coverage.reindex(mem).fillna(0)
            rep = score.idxmax()
        out[int(c)] = {"members": mem, "rep": rep}
    return out


def redundancy_one(name, X: pd.DataFrame, fam_of, cons_of):
    n = min(len(X), 60000)
    Xs = X.sample(n, random_state=SEED) if len(X) > n else X
    P = Xs.corr(method="pearson", min_periods=200)
    S = Xs.rank().corr(method="pearson", min_periods=200)        # Spearman = Pearson on ranks
    P.to_csv(AM / f"{name}_corr_pearson.csv"); S.to_csv(AM / f"{name}_corr_spearman.csv")
    cols = list(P.columns)
    iu = np.triu_indices(len(cols), 1)
    pairs = pd.DataFrame({"feature_a": np.array(cols)[iu[0]], "feature_b": np.array(cols)[iu[1]],
                          "pearson": P.to_numpy()[iu], "spearman": S.to_numpy()[iu]})
    pairs["abs_r"] = pairs[["pearson", "spearman"]].abs().max(axis=1)
    pairs["family_a"] = pairs.feature_a.map(fam_of); pairs["family_b"] = pairs.feature_b.map(fam_of)
    pairs["same_stem"] = pairs.feature_a.map(stem) == pairs.feature_b.map(stem)
    pairs["same_construct"] = pairs.feature_a.map(cons_of) == pairs.feature_b.map(cons_of)
    pairs = pairs.sort_values("abs_r", ascending=False)
    pairs.head(400).to_csv(AM / f"{name}_top_pairs.csv", index=False)
    thr = []
    for t in (0.70, 0.80, 0.90, 0.95):
        m = pairs.abs_r >= t
        thr.append({"model": name, "threshold": t, "pairs": int(m.sum()), "of_pairs": len(pairs),
                    "features_involved": int(len(set(pairs.feature_a[m]) | set(pairs.feature_b[m]))),
                    "share_same_stem": float(pairs.same_stem[m].mean()) if m.any() else np.nan,
                    "share_same_construct": float(pairs.same_construct[m].mean()) if m.any() else np.nan})
    vif = _vif(X)
    vt = pd.DataFrame({"feature": vif.index, "vif": vif.values})
    vt["family"] = vt.feature.map(fam_of); vt["construct"] = vt.feature.map(cons_of)
    vt = vt.sort_values("vif", ascending=False)
    vt.to_csv(AM / f"{name}_vif.csv", index=False)
    cov = X.notna().mean()
    rows = []
    for thr_c in (0.80, 0.90, 0.95):
        cl = _clusters(S, thr_c, cov)
        for cid, c in cl.items():
            if len(c["members"]) > 1:
                rows.append({"model": name, "threshold": thr_c, "cluster": cid, "size": len(c["members"]),
                             "representative": c["rep"], "members": " | ".join(c["members"])})
        (AC / f"{name}_clusters_{int(thr_c*100)}.json").write_text(json.dumps({str(k): v for k, v in cl.items()}))
    pd.DataFrame(rows).to_csv(AM / f"{name}_clusters.csv", index=False)
    # family summary: within-family mean |r| and the VIF picture
    fam_rows = []
    fams = pd.Series({c: fam_of(c) for c in cols})
    A = S.abs()
    for f, mem in fams.groupby(fams).groups.items():
        mem = list(mem)
        sub = A.loc[mem, mem].to_numpy()
        within = sub[np.triu_indices(len(mem), 1)] if len(mem) > 1 else np.array([np.nan])
        fam_rows.append({"model": name, "family": f, "n_features": len(mem),
                         "within_mean_abs_rho": float(np.nanmean(within)),
                         "within_max_abs_rho": float(np.nanmax(within)) if len(mem) > 1 else np.nan,
                         "median_vif": float(vt[vt.family == f].vif.median()),
                         "max_vif": float(vt[vt.family == f].vif.max()),
                         "n_vif_gt_10": int((vt[vt.family == f].vif > 10).sum())})
    # stem summary (K_last7 / K_last10 / ... / K_std)
    stem_rows = []
    stems = pd.Series({c: stem(c) for c in cols})
    for s_, mem in stems.groupby(stems).groups.items():
        mem = list(mem)
        if len(mem) < 2:
            continue
        sub = A.loc[mem, mem].to_numpy()
        w = sub[np.triu_indices(len(mem), 1)]
        stem_rows.append({"model": name, "stem": s_, "n_windows": len(mem), "mean_abs_rho": float(np.nanmean(w)),
                          "min_abs_rho": float(np.nanmin(w)), "max_vif": float(vt[vt.feature.isin(mem)].vif.max())})
    summ = {"model": name, "n_features": len(cols), "rows_used": int(len(Xs)),
            "condition_number": vif.attrs.get("condition_number"),
            "n_vif_gt_5": int((vt.vif > 5).sum()), "n_vif_gt_10": int((vt.vif > 10).sum()),
            "n_vif_gt_100": int((vt.vif > 100).sum()), "median_vif": float(vt.vif.median())}
    return thr, fam_rows, stem_rows, summ


def step_redundancy(groups, force):
    """Per-model results are cached separately (cache/redundancy_<model>.pkl),
    then every cached model is combined into the summary tables."""
    AC.mkdir(parents=True, exist_ok=True)
    for g in groups + ["game"]:
        cp = AC / f"redundancy_{g}.pkl"
        if cp.exists() and not force:
            continue
        if g == "game":
            import train_game_model as tgm
            gd = tgm.load_table()
            gtr = gd[gd.game_date < pd.Timestamp(VALID_START)]
            feats = [f for f in tgm.FEATURES if f != "home_field"]          # home_field is a constant (=1)
            r = redundancy_one("game", gtr[feats].apply(pd.to_numeric, errors="coerce"),
                               lambda f: GAME_FAMILY.get(f, "other"), game_construct)
        else:
            D = load_group(g)
            r = redundancy_one(g, D["train"][D["features"]], family, construct)
        pickle.dump(r, open(cp, "wb"))
        log(f"redundancy: {g} done")
    thr, fam, stems, summ = [], [], [], []
    for cp in sorted(AC.glob("redundancy_*.pkl")):
        r = pickle.load(open(cp, "rb"))
        thr += r[0]; fam += r[1]; stems += r[2]; summ.append(r[3])
    _out("redundancy_thresholds.csv", pd.DataFrame(thr))
    _out("redundancy_families.csv", pd.DataFrame(fam))
    _out("redundancy_stems.csv", pd.DataFrame(stems))
    _out("redundancy_summary.csv", pd.DataFrame(summ))
    log("redundancy: written")


# ---------------------------------------------------------------------------
# 3-4, 7. Per-target family fits: overfitting, importance, stability, diversity
# ---------------------------------------------------------------------------
def _inner(pipe):
    """(imputer, estimator) from a production pipeline (possibly target-wrapped)."""
    from sklearn.compose import TransformedTargetRegressor
    p = pipe.regressor_ if isinstance(pipe, TransformedTargetRegressor) else pipe
    return p.named_steps["imputer"], p.named_steps["model"], p


def _imputed(pipe, X):
    imp, _, _ = _inner(pipe)
    return imp.transform(X)


def native_importance(pipe, feats, fam):
    _, m, p = _inner(pipe)
    if fam == "xgb":
        sc = m.get_booster().get_score(importance_type="gain")
        v = np.array([sc.get(f"f{i}", 0.0) for i in range(len(feats))])
    elif hasattr(m, "feature_importances_"):
        v = m.feature_importances_
    else:
        v = np.abs(m.coef_)
    v = np.asarray(v, float)
    return v / v.sum() if v.sum() > 0 else v


def shap_xgb(pipe, X, feats):
    import xgboost as xgb
    _, m, _ = _inner(pipe)
    c = m.get_booster().predict(xgb.DMatrix(_imputed(pipe, X)), pred_contribs=True)[:, :-1]
    return np.abs(c).mean(axis=0)


def grouped_perm(predict, X: pd.DataFrame, y, groups: dict, repeats=3, kind="rmse"):
    """Increase in validation loss when every column of a group is permuted
    together (same row shuffle), which keeps within-group correlation intact and
    so is not fooled by correlated features splitting importance."""
    rng = np.random.default_rng(SEED)
    y = np.asarray(y, float)
    lossf = (lambda p: float(np.sqrt(np.mean((p - y) ** 2)))) if kind == "rmse" else (lambda p: float(np.mean(logloss_rows(y, p))))
    base = lossf(predict(X))
    out = {}
    for g, cols in groups.items():
        cols = [c for c in cols if c in X.columns]
        if not cols:
            continue
        inc = []
        for _ in range(repeats):
            Xp = X.copy()
            perm = rng.permutation(len(X))
            Xp[cols] = X[cols].to_numpy()[perm]
            inc.append(lossf(predict(Xp)) - base)
        out[g] = float(np.mean(inc))
    return base, out


def fit_target(group, target, D):
    feats = D["features"]
    tr = D["train"][D["train"][target].notna()]
    va = D["valid"][D["valid"][target].notna()]
    te = D["test"][D["test"][target].notna()]
    Xtr, ytr = tr[feats], tr[target].to_numpy(float)
    Xva, yva = va[feats], va[target].to_numpy(float)
    Xte, yte = te[feats], te[target].to_numpy(float)
    params = presets(group, target)
    null = float(np.mean(ytr))
    res = {"metrics": [], "importance": [], "folds": [], "lin_stability": [], "preds": {}}
    for split, y in (("train", ytr), ("valid", yva), ("test", yte)):
        res["metrics"].append({"group": group, "target": target, "family": "null_train_mean", "split": split,
                               **reg_block(y, np.full(len(y), null))})
    cons = pd.Series({f: construct(f) for f in feats})
    cgroups = {g: list(m) for g, m in cons.groupby(cons).groups.items()}
    pipes = {}
    for fam in FAMILIES:
        t0 = time.time()
        model, pipe, cal = hpt._fit_calibrated(Xtr, pd.Series(ytr), target, fam, params if fam == "xgb" else {}, True)
        pipes[fam] = pipe
        P = {s: np.asarray(model.predict(X), float) for s, X in (("train", Xtr), ("valid", Xva), ("test", Xte))}
        res["preds"][fam] = P
        for split, y in (("train", ytr), ("valid", yva), ("test", yte)):
            res["metrics"].append({"group": group, "target": target, "family": fam, "split": split,
                                   "calibration": cal["kind"], **reg_block(y, P[split])})
            # the same fit without the tail-of-train calibration wrapper
            res["metrics"].append({"group": group, "target": target, "family": f"{fam}_uncalibrated", "split": split,
                                   "calibration": "none", **reg_block(y, np.asarray(pipe.predict(
                                       {"train": Xtr, "valid": Xva, "test": Xte}[split]), float))})
        # importance (validation for permutation / SHAP; native from the train fit)
        nat = native_importance(pipe, feats, fam)
        imp = pd.DataFrame({"group": group, "target": target, "family": fam, "feature": feats,
                            "construct": cons.values, "feature_family": [family(f) for f in feats],
                            "native": nat})
        if fam == "xgb":
            imp["shap_valid"] = shap_xgb(pipe, Xva, feats)
            imp["shap_train"] = shap_xgb(pipe, Xtr.sample(min(len(Xtr), 30000), random_state=SEED), feats)
        sidx = np.random.default_rng(SEED).permutation(len(Xva))[:15000]      # permutation on <=15k validation rows
        base, gp = grouped_perm(model.predict, Xva.iloc[sidx], yva[sidx], cgroups)
        res["importance"].append(imp)
        res.setdefault("grouped_perm", []).extend(
            {"group": group, "target": target, "family": fam, "construct": g, "valid_rmse_increase": v,
             "valid_rmse": base, "pct_increase": 100 * v / base} for g, v in gp.items())
        log(f"    {group}/{target}/{fam}: valid RMSE {np.sqrt(np.mean((P['valid']-yva)**2)):.4f} ({time.time()-t0:.0f}s)")

    # importance across chronological folds of TRAIN (xgb, expanding windows)
    dates = tr["game_date"].dt.normalize()
    qs = dates.quantile([1 / 3, 2 / 3]).tolist()
    fold_imp = {}
    for k, cut in enumerate(qs + [None], 1):
        m = (dates < cut) if cut is not None else np.ones(len(tr), bool)
        if m.sum() < 300:
            continue
        p = hpt.build_sklearn_model("xgb", target_name=target, params=params)
        p.fit(Xtr[m], ytr[m])
        fold_imp[f"fold{k}"] = shap_xgb(p, Xva, feats)
    for k, v in fold_imp.items():
        res["folds"].append(pd.DataFrame({"group": group, "target": target, "fold": k, "feature": feats,
                                          "construct": cons.values, "shap_valid": v}))

    # linear-coefficient stability: four chronological quarters of train, full vs
    # one-representative-per-|rho|>=0.90-cluster feature set
    cl = json.loads((AC / f"{group}_clusters_90.json").read_text()) if (AC / f"{group}_clusters_90.json").exists() else None
    sets = {"full": feats}
    if cl:
        sets["reduced_090"] = [c["rep"] for c in cl.values() if c["rep"] in feats]
    qq = dates.quantile([0.25, 0.5, 0.75]).tolist()
    edges = [dates.min() - pd.Timedelta(days=1)] + qq + [dates.max() + pd.Timedelta(days=1)]
    for sname, fs in sets.items():
        coefs = []
        for a, b in zip(edges[:-1], edges[1:]):
            m = (dates > a) & (dates <= b)
            if m.sum() < 200:
                continue
            p = hpt.build_sklearn_model("lin", target_name=target)
            p.fit(tr.loc[m.values, fs], ytr[m.values])
            coefs.append(_inner(p)[1].coef_)
        if len(coefs) < 2:
            continue
        C = np.vstack(coefs)
        sign_flip = (np.sign(C) != np.sign(np.median(C, axis=0))).any(axis=0)
        mag = np.abs(C).mean(axis=0)
        cv = C.std(axis=0) / np.maximum(np.abs(C.mean(axis=0)), 1e-9)
        big = mag >= np.quantile(mag, 0.5)
        res["lin_stability"].append({"group": group, "target": target, "feature_set": sname, "n_features": len(fs),
                                     "blocks": len(coefs), "share_sign_flips": float(sign_flip.mean()),
                                     "share_sign_flips_top_half": float(sign_flip[big].mean()),
                                     "median_coef_cv": float(np.median(cv)),
                                     "max_abs_coef": float(np.abs(C).max())})

    # diversity and stacks (validation decides; test reported)
    fams = FAMILIES
    Pv = np.column_stack([res["preds"][f]["valid"] for f in fams])
    Pt = np.column_stack([res["preds"][f]["test"] for f in fams])
    div = {"group": group, "target": target}
    for i in range(3):
        for j in range(i + 1, 3):
            a, b = fams[i], fams[j]
            div[f"pred_corr_{a}_{b}"] = float(np.corrcoef(Pv[:, i], Pv[:, j])[0, 1])
            div[f"err_corr_{a}_{b}"] = float(np.corrcoef(Pv[:, i] - yva, Pv[:, j] - yva)[0, 1])
            div[f"test_err_corr_{a}_{b}"] = float(np.corrcoef(Pt[:, i] - yte, Pt[:, j] - yte)[0, 1])
    rm = {f: float(np.sqrt(np.mean((Pv[:, k] - yva) ** 2))) for k, f in enumerate(fams)}
    best = min(rm, key=rm.get)
    order = np.argsort(va["game_date"].to_numpy(), kind="mergesort")
    w_opt = stk.fit_weights(Pv, yva, "rmse")
    cf = np.empty(len(yva)); cf[order] = stk.crossfit_predictions(Pv[order], yva[order], "rmse")
    even = np.clip(Pv @ stk.even_weights(3), 0, None)
    ib = fams.index(best)
    for name, pv, pt in (("stack_even", even, np.clip(Pt @ stk.even_weights(3), 0, None)),
                         ("stack_opt", np.clip(cf, 0, None), np.clip(Pt @ w_opt, 0, None))):
        (d, lo, hi), (pct, plo, phi) = boot_rmse_diff(va["game_date"].values, yva, pv, Pv[:, ib])
        div.update({f"{name}_valid_rmse": float(np.sqrt(np.mean((pv - yva) ** 2))),
                    f"{name}_valid_pct_vs_best": pct, f"{name}_valid_pct_lo": plo, f"{name}_valid_pct_hi": phi,
                    f"{name}_test_rmse": float(np.sqrt(np.mean((pt - yte) ** 2)))})
    div.update({"best_single": best, "best_single_valid_rmse": rm[best],
                "best_single_test_rmse": float(np.sqrt(np.mean((Pt[:, ib] - yte) ** 2))),
                "stack_opt_weights": stk.describe(fams, w_opt)})
    res["diversity"] = div
    # validation deciles of the best single family (decision-side shrinkage check)
    res["valid_deciles"] = deciles(yva, Pv[:, ib]).assign(group=group, target=target, family=best, split="valid")
    res.pop("preds")
    return res


def step_fits(groups, force):
    AC.mkdir(parents=True, exist_ok=True)
    # per-target fits are cached in AC; the summary tables are always rebuilt from every cached fit
    # (cheap), so an interrupted write can never leave a stale summary behind
    for g in groups:
        D = load_group(g)
        for t in D["targets"]:
            cp = AC / f"fit_{g}_{t}.pkl"
            if cp.exists() and not force:
                continue
            log(f"fits: {g} {t}")
            r = fit_target(g, t, D)
            pickle.dump(r, open(cp, "wb"))
    collect_fits()


def collect_fits():
    met, imp, gp, folds, lst, div, dec = [], [], [], [], [], [], []
    for cp in sorted(AC.glob("fit_*.pkl")):
        r = pickle.load(open(cp, "rb"))
        met += r["metrics"]; imp += r["importance"]; gp += r.get("grouped_perm", [])
        folds += r["folds"]; lst += r["lin_stability"]; div.append(r["diversity"]); dec.append(r["valid_deciles"])
    m = pd.DataFrame(met)
    _out("overfit_metrics.csv", m)
    _out("importance_individual.csv", pd.concat(imp, ignore_index=True))
    _out("importance_grouped_permutation.csv", pd.DataFrame(gp))
    if folds:
        _out("importance_folds.csv", pd.concat(folds, ignore_index=True))
    _out("linear_stability.csv", pd.DataFrame(lst))
    _out("model_diversity.csv", pd.DataFrame(div))
    _out("deciles_valid_best_family.csv", pd.concat(dec, ignore_index=True))
    _out("overfit_summary.csv", overfit_summary(m))
    _out("importance_stability.csv", importance_stability())
    log("fits: collected")


def overfit_summary(m: pd.DataFrame) -> pd.DataFrame:
    """Skill = 1 - deviance / deviance of the train-mean predictor on the SAME
    split (so a harder or easier period does not masquerade as over-fitting)."""
    rows = []
    for (g, t), d in m.groupby(["group", "target"]):
        null = d[d.family == "null_train_mean"].set_index("split")
        for fam, dd in d[d.family != "null_train_mean"].groupby("family"):
            dd = dd.set_index("split")
            r = {"group": g, "target": t, "family": fam,
                 "calibration": dd["calibration"].iloc[0] if "calibration" in dd else ""}
            for s in ("train", "valid", "test"):
                if s not in dd.index:
                    continue
                for k in ("rmse", "mae", "poisson_dev", "cal_slope", "bias", "share_pred_lt_0.05"):
                    r[f"{s}_{k}"] = dd.loc[s, k]
                r[f"{s}_skill_dev"] = 1 - dd.loc[s, "poisson_dev"] / null.loc[s, "poisson_dev"]
                r[f"{s}_skill_rmse"] = 1 - dd.loc[s, "rmse"] / null.loc[s, "rmse"]
            r["valid_train_gap_rmse"] = r["valid_rmse"] - r["train_rmse"]
            r["test_valid_gap_rmse"] = r.get("test_rmse", np.nan) - r["valid_rmse"]
            r["valid_train_gap_pct"] = 100 * r["valid_train_gap_rmse"] / r["train_rmse"]
            r["test_valid_gap_pct"] = 100 * r["test_valid_gap_rmse"] / r["valid_rmse"]
            r["skill_retention"] = r["valid_skill_dev"] / r["train_skill_dev"] if r["train_skill_dev"] > 1e-6 else np.nan
            r["verdict"] = overfit_verdict(r)
            rows.append(r)
    return pd.DataFrame(rows)


def overfit_verdict(r) -> str:
    """Rules (validation evidence only; test is not consulted):
       clear overfitting   : validation keeps < 50% of the train skill AND slope < 0.85
       mild overfitting    : validation keeps < 75% of the train skill OR slope < 0.90
       under-fit / shrunk  : validation slope > 1.10 (predictions too compressed)
       otherwise           : no meaningful overfitting"""
    ret, sl = r.get("skill_retention", np.nan), r.get("valid_cal_slope", np.nan)
    if np.isfinite(ret) and ret < 0.5 and np.isfinite(sl) and sl < 0.85:
        return "clear overfitting"
    if (np.isfinite(ret) and ret < 0.75) or (np.isfinite(sl) and sl < 0.90):
        return "mild overfitting"
    if np.isfinite(sl) and sl > 1.10:
        return "under-fit / excessive shrinkage"
    return "no meaningful overfitting"


def importance_stability() -> pd.DataFrame:
    from scipy.stats import spearmanr
    rows = []
    p = AM / "importance_folds.csv"
    imp = pd.read_csv(AM / "importance_individual.csv")
    folds = pd.read_csv(p) if p.exists() else pd.DataFrame()
    for (g, t), d in imp[imp.family == "xgb"].groupby(["group", "target"]):
        r = {"group": g, "target": t}
        a, b = d["shap_train"].to_numpy(), d["shap_valid"].to_numpy()
        r["train_vs_valid_rank_corr_individual"] = float(spearmanr(a, b).correlation)
        ga = d.groupby("construct")["shap_train"].sum(); gb = d.groupby("construct")["shap_valid"].sum()
        r["train_vs_valid_rank_corr_grouped"] = float(spearmanr(ga, gb.reindex(ga.index)).correlation)
        r["native_vs_shap_rank_corr"] = float(spearmanr(d["native"], d["shap_valid"]).correlation)
        f = folds[(folds.group == g) & (folds.target == t)] if len(folds) else folds
        if len(f) and f.fold.nunique() >= 2:
            W = f.pivot(index="feature", columns="fold", values="shap_valid")
            Gw = f.groupby(["construct", "fold"])["shap_valid"].sum().unstack()
            ci, cg, jac = [], [], []
            cols = list(W.columns)
            for i in range(len(cols)):
                for j in range(i + 1, len(cols)):
                    ci.append(spearmanr(W[cols[i]], W[cols[j]]).correlation)
                    cg.append(spearmanr(Gw[cols[i]], Gw[cols[j]]).correlation)
                    ta = set(W[cols[i]].nlargest(10).index); tb = set(W[cols[j]].nlargest(10).index)
                    jac.append(len(ta & tb) / len(ta | tb))
            r.update({"fold_rank_corr_individual": float(np.mean(ci)), "fold_rank_corr_grouped": float(np.mean(cg)),
                      "fold_top10_jaccard": float(np.mean(jac)),
                      "fold_top_construct_same": bool(Gw.idxmax().nunique() == 1)})
        top = d.nlargest(10, "shap_valid")
        r["top10_distinct_constructs"] = int(top.construct.nunique())
        r["top10_distinct_stems"] = int(top.feature.map(stem).nunique())
        rows.append(r)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4b. XGBoost learning curves (train -> validation)
# ---------------------------------------------------------------------------
def _xgb_with(group, target, **over):
    params = dict(presets(group, target)); params.update(over)
    return hpt.build_sklearn_model("xgb", target_name=target, params=params)


def _tr_va(D, target):
    tr = D["train"][D["train"][target].notna()]; va = D["valid"][D["valid"][target].notna()]
    return tr, va


def step_curves(groups, force):
    prev = pd.read_csv(AM / "learning_curves.csv") if (AM / "learning_curves.csv").exists() and not force else pd.DataFrame()
    if len(prev) and set(groups) <= set(prev.group):
        return
    rows = prev[~prev.group.isin(groups)].to_dict("records") if len(prev) else []
    for g in groups:
        D = load_group(g); feats = D["features"]
        for t in CURVE_TARGETS[g]:
            if t not in D["targets"]:
                continue
            tr, va = _tr_va(D, t)
            Xtr, ytr, Xva, yva = tr[feats], tr[t].to_numpy(float), va[feats], va[t].to_numpy(float)
            base = _xgb_with(g, t)
            m0 = base.named_steps["model"] if hasattr(base, "named_steps") else base.regressor.named_steps["model"]
            prod = m0.get_params()
            log(f"curves: {g} {t} (prod n_est={prod['n_estimators']} depth={prod['max_depth']} "
                f"mcw={prod['min_child_weight']} lambda={prod['reg_lambda']})")

            def score(pipe, X_fit=Xtr, y_fit=ytr):
                pipe.fit(X_fit, y_fit)
                return reg_block(y_fit, pipe.predict(X_fit), "train_") | reg_block(yva, pipe.predict(Xva), "valid_")

            # trees: one long fit, staged predictions at checkpoints
            from xgboost import XGBRegressor
            from sklearn.impute import SimpleImputer
            imp = SimpleImputer(strategy="median").fit(Xtr)
            A, B = imp.transform(Xtr), imp.transform(Xva)
            kw = {k: v for k, v in prod.items() if v is not None}
            kw["n_estimators"] = max(1500, 3 * prod["n_estimators"])
            xm = XGBRegressor(**kw).fit(A, ytr)
            for n in sorted({25, 50, 100, 200, 300, 400, 600, 800, 1000, 1200, 1500, prod["n_estimators"]}):
                if n > kw["n_estimators"]:
                    continue
                pa = xm.predict(A, iteration_range=(0, n)); pb = xm.predict(B, iteration_range=(0, n))
                rows.append({"group": g, "target": t, "axis": "n_estimators", "value": n,
                             "production": n == prod["n_estimators"],
                             **reg_block(ytr, pa, "train_"), **reg_block(yva, pb, "valid_")})
            for ax, vals in (("max_depth", [1, 2, 3, 4, 6, 8]), ("min_child_weight", [1, 5, 10, 30, 100, 300]),
                             ("reg_lambda", [0.0, 1.0, 3.0, 10.0, 30.0, 100.0])):
                for v in vals:
                    rows.append({"group": g, "target": t, "axis": ax, "value": v, "production": v == prod[ax],
                                 **score(_xgb_with(g, t, **{ax: v}))})
            dts = tr["game_date"].dt.normalize()
            for frac in (0.1, 0.25, 0.5, 0.75, 1.0):
                cut = dts.quantile(1 - frac) if frac < 1 else dts.min()
                m = (dts >= cut).to_numpy()
                rows.append({"group": g, "target": t, "axis": "train_fraction_recent", "value": frac,
                             "production": frac == 1.0, "n_train_rows": int(m.sum()),
                             **score(_xgb_with(g, t), Xtr[m], ytr[m])})
    _out("learning_curves.csv", pd.DataFrame(rows))
    log("curves: written")


# ---------------------------------------------------------------------------
# 10. Ablations (train -> validation; test is a report-only column)
# ---------------------------------------------------------------------------
ABLATIONS = [("no rolling windows", {"rolling"}), ("no expanding/std", {"expanding"}),
             ("no handedness splits", {"handedness"}), ("no true-talent", {"true_talent"}),
             ("no matchup", {"matchup"}), ("no lineup", {"lineup"}), ("no batted-ball", {"batted_ball"}),
             ("no velocity/pitch-mix", {"velocity/pitch_mix"}), ("no opportunity", {"opportunity"}),
             ("no team/opponent context", {"team/opponent"})]


def step_ablation(groups, force, families=("xgb", "lin")):
    if _cached("ablation.csv", force) and set(groups) <= set(pd.read_csv(AM / "ablation.csv").group):
        return
    rows = []
    cache = AC / "ablation_partial.csv"
    done = pd.read_csv(cache) if cache.exists() and not force else pd.DataFrame()
    seen = set(zip(done.get("group", []), done.get("target", []), done.get("model", []), done.get("variant", [])))
    rows = done.to_dict("records")
    for g in groups:
        D = load_group(g); feats = D["features"]
        fam = {f: family(f) for f in feats}
        variants = [("full", feats)]
        for name, drop in ABLATIONS:
            keep = [f for f in feats if fam[f] not in drop]
            if len(keep) < len(feats):
                variants.append((name, keep))
        for thr in (90, 80):
            p = AC / f"{g}_clusters_{thr}.json"
            if p.exists():
                cl = json.loads(p.read_text())
                variants.append((f"one per |rho|>={thr/100:.2f} cluster", [c["rep"] for c in cl.values() if c["rep"] in feats]))
        for t in D["targets"]:
            tr, va = _tr_va(D, t)
            te = D["test"][D["test"][t].notna()]
            ytr, yva, yte = tr[t].to_numpy(float), va[t].to_numpy(float), te[t].to_numpy(float)
            for mdl in families:
                full_pred = None
                for vname, fs in variants:
                    if (g, t, mdl, vname) in seen:
                        continue
                    p = (_xgb_with(g, t) if mdl == "xgb" else hpt.build_sklearn_model("lin", target_name=t))
                    p.fit(tr[fs], ytr)
                    pv, pt = p.predict(va[fs]), p.predict(te[fs])
                    r = {"group": g, "target": t, "model": mdl, "variant": vname, "n_features": len(fs),
                         "n_dropped": len(feats) - len(fs), **reg_block(yva, pv, "valid_"), **reg_block(yte, pt, "test_")}
                    if vname == "full":
                        full_pred = pv
                        np.save(AC / f"abl_full_{g}_{t}_{mdl}.npy", pv)
                    else:
                        if full_pred is None and (AC / f"abl_full_{g}_{t}_{mdl}.npy").exists():
                            full_pred = np.load(AC / f"abl_full_{g}_{t}_{mdl}.npy")
                        if full_pred is not None:
                            (d, lo, hi), _ = boot_rmse_diff(va["game_date"].values, yva, pv, full_pred, n=500)
                            r.update({"valid_rmse_delta_vs_full": d, "delta_lo": lo, "delta_hi": hi})
                    rows.append(r)
                    pd.DataFrame(rows).to_csv(cache, index=False)
                log(f"ablation: {g} {t} {mdl}")
    a = pd.DataFrame(rows)
    full = a[a.variant == "full"].set_index(["group", "target", "model"])
    for k in ("valid_rmse", "valid_poisson_dev", "test_rmse"):
        a[f"{k}_pct_vs_full"] = 100 * (a[k] / a.set_index(["group", "target", "model"]).index.map(full[k]).to_numpy() - 1)
    _out("ablation.csv", a)
    log("ablation: written")


# ---------------------------------------------------------------------------
# 6. Game model: calibration, overfit, importance, diversity, ablation, curves
# ---------------------------------------------------------------------------
BUCKETS = [0.0, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 1.0]


def buckets(y, p, model, split):
    y = np.asarray(y, float); p = np.asarray(p, float)
    idx = np.clip(np.digitize(p, BUCKETS) - 1, 0, len(BUCKETS) - 2)
    out = []
    for b in range(len(BUCKETS) - 1):
        m = idx == b
        n = int(m.sum())
        if n == 0:
            continue
        k = int(y[m].sum()); lo, hi = wilson(k, n)
        from scipy.stats import binomtest
        mp = float(p[m].mean())
        out.append({"model": model, "split": split, "bucket": f"{BUCKETS[b]:.2f}-{BUCKETS[b+1]:.2f}", "n": n,
                    "mean_p": mp, "win_rate": k / n, "ci_lo": lo, "ci_hi": hi, "gap": k / n - mp,
                    "p_value_vs_mean_p": float(binomtest(k, n, min(max(mp, 1e-6), 1 - 1e-6)).pvalue)})
    return out


def step_game(force):
    if _cached("game_meta.json", force):
        return
    import train_game_model as tgm
    from sklearn.base import clone  # noqa: F401
    df = tgm.load_table()
    vs, ts = pd.Timestamp(VALID_START), pd.Timestamp(TEST_START)
    tr, va, te = df[df.game_date < vs], df[(df.game_date >= vs) & (df.game_date < ts)], df[df.game_date >= ts]
    full = pd.concat([tr, va])
    ytr, yva, yte = (d["home_win"].to_numpy(float) for d in (tr, va, te))
    log(f"game: train {len(tr)} valid {len(va)} test {len(te)}")
    names = list(tgm.CANDIDATES)
    Ptr, Pva, Pte, models_tr = {}, {}, {}, {}
    for n in names:
        m = tgm.CANDIDATES[n]().fit(tgm.X_of(tr), tr["home_win"])
        models_tr[n] = m
        Ptr[n] = m.predict_proba(tgm.X_of(tr))[:, 1]
        Pva[n] = m.predict_proba(tgm.X_of(va))[:, 1]
        mf = tgm.CANDIDATES[n]().fit(tgm.X_of(full), full["home_win"])          # = the reported test protocol
        Pte[n] = mf.predict_proba(tgm.X_of(te))[:, 1]
    P_va = np.column_stack([Pva[n] for n in names]); P_te = np.column_stack([Pte[n] for n in names])
    P_tr = np.column_stack([Ptr[n] for n in names])
    w_opt = stk.fit_weights(P_va, yva, "logloss")
    order = np.argsort(va.game_date.to_numpy(), kind="mergesort")
    cf = np.empty(len(yva)); cf[order] = stk.crossfit_predictions(P_va[order], yva[order], "logloss")
    for sname, w in (("stack_even", stk.even_weights(len(names))), ("stack_opt", w_opt)):
        Pva[sname] = np.clip(cf if sname == "stack_opt" else P_va @ w, 1e-6, 1 - 1e-6)
        Pte[sname] = np.clip(P_te @ w, 1e-6, 1 - 1e-6)
        Ptr[sname] = np.clip(P_tr @ w, 1e-6, 1 - 1e-6)
    base_rate = float(ytr.mean())
    sel = min(Pva, key=lambda k: np.mean(logloss_rows(yva, Pva[k])))
    cal_rows, bk, boot = [], [], []
    for n in list(Pva):
        for split, y, p in (("train", ytr, Ptr[n]), ("valid", yva, Pva[n]), ("test", yte, Pte[n])):
            cal_rows.append({"model": n, "split": split, "selected_on_valid": n == sel, **prob_block(y, p)})
        for split, y, p in (("valid", yva, Pva[n]), ("test", yte, Pte[n])):
            bk += buckets(y, p, n, split)
    for split, y, d in (("valid", yva, va), ("test", yte, te)):
        cal_rows.append({"model": "baseline_train_home_rate", "split": split,
                         **prob_block(y, np.full(len(y), base_rate))})
        P = Pva if split == "valid" else Pte
        for n in P:
            ll = logloss_rows(y, P[n]); llb = logloss_rows(y, np.full(len(y), base_rate))
            pt, lo, hi = block_boot(d.game_date.values, {"a": ll, "b": llb},
                                    lambda s: s["a"] / s["cnt"] - s["b"] / s["cnt"])
            boot.append({"model": n, "split": split, "logloss_minus_home_rate": pt, "lo": lo, "hi": hi})
        # isotonic vs raw (calibration and discrimination), bootstrap by date
        for raw, iso in [(r_, i_) for r_, i_ in (("xgb_raw", "xgb_isotonic"), ("rf_raw", "rf_isotonic")) if r_ in P and i_ in P]:
            la, lb = logloss_rows(y, P[iso]), logloss_rows(y, P[raw])
            pt, lo, hi = block_boot(d.game_date.values, {"a": la, "b": lb}, lambda s: s["a"] / s["cnt"] - s["b"] / s["cnt"])
            boot.append({"model": f"{iso} minus {raw}", "split": split, "logloss_diff": pt, "lo": lo, "hi": hi})
    _out("game_calibration.csv", pd.DataFrame(cal_rows))
    _out("game_buckets.csv", pd.DataFrame(bk))
    _out("game_bootstrap.csv", pd.DataFrame(boot))
    hi_rows = []
    for split, y, P in (("valid", yva, Pva), ("test", yte, Pte)):
        for n in P:
            p = P[n]; m = (p >= 0.70) | (p <= 0.30)
            for thr in (0.65, 0.70, 0.75):
                m = (p >= thr) | (p <= 1 - thr)
                if m.sum() == 0:
                    hi_rows.append({"model": n, "split": split, "confidence_ge": thr, "n": 0}); continue
                fav = np.where(p[m] >= 0.5, p[m], 1 - p[m]); won = np.where(p[m] >= 0.5, y[m], 1 - y[m])
                k, nn = int(won.sum()), int(m.sum()); lo, hi = wilson(k, nn)
                hi_rows.append({"model": n, "split": split, "confidence_ge": thr, "n": nn, "mean_conf": float(fav.mean()),
                                "fav_win_rate": k / nn, "ci_lo": lo, "ci_hi": hi})
    _out("game_high_confidence.csv", pd.DataFrame(hi_rows))
    # temporal (test thirds + months; validation months)
    trow = []
    for split, d, P in (("valid", va, Pva), ("test", te, Pte)):
        dd = d[["game_date"]].copy(); dd["y"] = d.home_win.to_numpy(float); dd["p"] = P[sel]
        dd["month"] = dd.game_date.dt.strftime("%Y-%m")
        u = np.sort(dd.game_date.dt.normalize().unique()); cuts = [u[len(u) // 3], u[2 * len(u) // 3]]
        dd["third"] = np.where(dd.game_date < cuts[0], "early", np.where(dd.game_date < cuts[1], "middle", "late"))
        for col in ("month", "third"):
            for k, x in dd.groupby(col):
                trow.append({"split": split, "period_type": col, "period": k, "model": sel, **prob_block(x.y, x.p)})
    _out("game_temporal.csv", pd.DataFrame(trow))
    # diversity
    div = []
    for split, y, P in (("valid", yva, Pva), ("test", yte, Pte)):
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                div.append({"split": split, "a": a, "b": b, "pred_corr": float(np.corrcoef(P[a], P[b])[0, 1]),
                            "err_corr": float(np.corrcoef(y - P[a], y - P[b])[0, 1])})
    _out("game_diversity.csv", pd.DataFrame(div))
    # importance: grouped permutation on validation (log loss), SHAP for xgb_raw
    feats = tgm.FEATURES
    cg = {}
    for f in feats:
        cg.setdefault(game_construct(f), []).append(f)
    gp = []
    for n in [x for x in ("xgb_raw", "rf_raw", "logistic") if x in models_tr]:
        m = models_tr[n]
        base, out = grouped_perm(lambda X: m.predict_proba(X)[:, 1], tgm.X_of(va), yva, cg, repeats=5, kind="logloss")
        gp += [{"model": n, "construct": k, "valid_logloss_increase": v, "valid_logloss": base} for k, v in out.items()]
        base, out = grouped_perm(lambda X: m.predict_proba(X)[:, 1], tgm.X_of(va), yva, {f: [f] for f in feats},
                                 repeats=5, kind="logloss")
        gp += [{"model": n, "feature": k, "valid_logloss_increase": v, "valid_logloss": base} for k, v in out.items()]
    import xgboost as xgb
    xshap = models_tr["xgb_raw"] if "xgb_raw" in models_tr else tgm._xgb().fit(tgm.X_of(tr), tr["home_win"])   # diagnostic only
    sh = xshap.get_booster().predict(xgb.DMatrix(tgm.X_of(va)), pred_contribs=True)[:, :-1]
    shp = pd.DataFrame({"feature": feats, "shap_valid": np.abs(sh).mean(0), "construct": [game_construct(f) for f in feats]})
    co = models_tr["logistic"].named_steps["model"].coef_[0]
    shp["logistic_std_coef"] = co
    _out("game_importance.csv", pd.concat([pd.DataFrame(gp), shp.assign(model="xgb_raw_shap")], ignore_index=True))
    # logistic coefficient stability across train quarters (+ sign flips)
    dts = tr.game_date
    qs = [dts.min() - pd.Timedelta(days=1)] + dts.quantile([0.25, 0.5, 0.75]).tolist() + [dts.max() + pd.Timedelta(days=1)]
    C = []
    for a, b in zip(qs[:-1], qs[1:]):
        m = (dts > a) & (dts <= b)
        C.append(tgm._logit().fit(tgm.X_of(tr[m]), tr.loc[m, "home_win"]).named_steps["model"].coef_[0])
    C = np.vstack(C)
    _out("game_logistic_stability.csv", pd.DataFrame({"feature": feats, "coef_mean": C.mean(0), "coef_sd": C.std(0),
                                                      "sign_flip": (np.sign(C) != np.sign(np.median(C, 0))).any(0)}))
    # ablation on validation (logistic + rf_raw + xgb_raw), test report-only
    ab = []
    fam_groups = {}
    for f in feats:
        fam_groups.setdefault(GAME_FAMILY.get(f, "other"), []).append(f)
    variants = [("full", feats)] + [(f"no {k}", [f for f in feats if f not in v]) for k, v in fam_groups.items() if k != "other"]
    variants += [(f"no {k}", [f for f in feats if f not in v]) for k, v in cg.items()]
    gfeat = [f for f in feats if f != "home_field"]
    p = AC / "game_clusters_90.json"
    if p.exists():
        cl = json.loads(p.read_text()); variants.append(("one per |rho|>=0.90 cluster", [c["rep"] for c in cl.values()] + ["home_field"]))
    p = AC / "game_clusters_80.json"
    if p.exists():
        cl = json.loads(p.read_text()); variants.append(("one per |rho|>=0.80 cluster", [c["rep"] for c in cl.values()] + ["home_field"]))
    for n in [x for x in ("logistic", "rf_raw", "xgb_raw") if x in tgm.CANDIDATES]:
        fullp = None
        for vname, fs in variants:
            Xt = tr.reindex(columns=fs).apply(pd.to_numeric, errors="coerce").fillna(0)
            Xv = va.reindex(columns=fs).apply(pd.to_numeric, errors="coerce").fillna(0)
            Xe = te.reindex(columns=fs).apply(pd.to_numeric, errors="coerce").fillna(0)
            m = tgm.CANDIDATES[n]().fit(Xt, tr["home_win"])
            pv, pe = m.predict_proba(Xv)[:, 1], m.predict_proba(Xe)[:, 1]
            r = {"model": n, "variant": vname, "n_features": len(fs), **prob_block(yva, pv, "valid_"), **prob_block(yte, pe, "test_")}
            if vname == "full":
                fullp = pv
            else:
                pt, lo, hi = block_boot(va.game_date.values, {"a": logloss_rows(yva, pv), "b": logloss_rows(yva, fullp)},
                                        lambda s: s["a"] / s["cnt"] - s["b"] / s["cnt"], n=500)
                r.update({"valid_logloss_delta_vs_full": pt, "delta_lo": lo, "delta_hi": hi})
            ab.append(r)
    _out("game_ablation.csv", pd.DataFrame(ab))
    # learning curves for xgb_raw
    lc = []
    from xgboost import XGBClassifier
    prod = tgm._xgb().get_params()
    kw = {k: v for k, v in prod.items() if v is not None}; kw["n_estimators"] = 1500
    xm = XGBClassifier(**kw).fit(tgm.X_of(tr), ytr)
    for n in (25, 50, 100, 200, 300, 400, 600, 800, 1000, 1500):
        pa = xm.predict_proba(tgm.X_of(tr), iteration_range=(0, n))[:, 1]; pb = xm.predict_proba(tgm.X_of(va), iteration_range=(0, n))[:, 1]
        lc.append({"axis": "n_estimators", "value": n, "production": n == prod["n_estimators"],
                   "train_log_loss": float(np.mean(logloss_rows(ytr, pa))), "valid_log_loss": float(np.mean(logloss_rows(yva, pb))),
                   "valid_auc": prob_block(yva, pb)["auc"]})
    for ax, vals in (("max_depth", [1, 2, 3, 4, 6, 8]), ("min_child_weight", [1, 5, 10, 30, 100, 300]),
                     ("reg_lambda", [0.0, 1.0, 2.0, 10.0, 30.0, 100.0])):
        for v in vals:
            k2 = dict(kw); k2["n_estimators"] = prod["n_estimators"]; k2[ax] = v
            m = XGBClassifier(**k2).fit(tgm.X_of(tr), ytr)
            pa, pb = m.predict_proba(tgm.X_of(tr))[:, 1], m.predict_proba(tgm.X_of(va))[:, 1]
            lc.append({"axis": ax, "value": v, "production": v == prod[ax],
                       "train_log_loss": float(np.mean(logloss_rows(ytr, pa))), "valid_log_loss": float(np.mean(logloss_rows(yva, pb))),
                       "valid_auc": prob_block(yva, pb)["auc"]})
    for C_ in (0.0001, 0.0003, 0.001, 0.003, 0.01, 0.1, 1.0):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        m = Pipeline([("s", StandardScaler()), ("m", LogisticRegression(C=C_, max_iter=2000))]).fit(tgm.X_of(tr), ytr)
        pa, pb = m.predict_proba(tgm.X_of(tr))[:, 1], m.predict_proba(tgm.X_of(va))[:, 1]
        lc.append({"axis": "logistic_C", "value": C_, "production": abs(C_ - tgm._logit().named_steps["model"].C) < 1e-12,
                   "train_log_loss": float(np.mean(logloss_rows(ytr, pa))), "valid_log_loss": float(np.mean(logloss_rows(yva, pb))),
                   "valid_auc": prob_block(yva, pb)["auc"]})
    _out("game_learning_curves.csv", pd.DataFrame(lc))
    json.dump({"selected_on_valid": sel, "stack_opt_weights": stk.describe(names, w_opt),
               "n_train": len(tr), "n_valid": len(va), "n_test": len(te), "train_home_rate": base_rate},
              open(AM / "game_meta.json", "w"), indent=2)
    log(f"game: written (selected on validation: {sel})")


# ---------------------------------------------------------------------------
# 5, 8, 9. Saved TEST predictions: deciles, temporal, tenure, baselines
# ---------------------------------------------------------------------------
def deciles(y, p, q=10):
    y = np.asarray(y, float); p = np.asarray(p, float)
    r = pd.Series(p).rank(method="first")
    b = pd.qcut(r, q, labels=False)
    out = []
    for k in range(q):
        m = (b == k).to_numpy()
        out.append({"decile": k + 1, "n": int(m.sum()), "mean_pred": float(p[m].mean()), "mean_actual": float(y[m].mean()),
                    "mean_residual": float((y[m] - p[m]).mean()), "mae": float(np.abs(y[m] - p[m]).mean()),
                    "rmse": float(np.sqrt(((y[m] - p[m]) ** 2).mean())), "poisson_dev": float(pois_dev_rows(y[m], p[m]).mean())})
    return pd.DataFrame(out)


def shrink_stats(y, p):
    d = deciles(y, p)
    spread_pred = d.mean_pred.iloc[-1] - d.mean_pred.iloc[0]
    spread_act = d.mean_actual.iloc[-1] - d.mean_actual.iloc[0]
    return {"cal_slope": cal_slope(y, p), "decile_spread_pred": spread_pred, "decile_spread_actual": spread_act,
            "spread_ratio_actual_over_pred": spread_act / spread_pred if spread_pred > 0 else np.nan,
            "top_decile_residual": d.mean_residual.iloc[-1], "bottom_decile_residual": d.mean_residual.iloc[0],
            "sd_pred_over_sd_actual": float(np.std(p) / np.std(y)) if np.std(y) > 0 else np.nan}


def step_saved(groups, force):
    prev = {}
    if not force and (AM / "baselines_bootstrap.csv").exists():
        for n in ("deciles_test_production", "shrinkage_test", "temporal_players", "tenure_players", "baselines_bootstrap"):
            prev[n] = pd.read_csv(AM / f"{n}.csv")
        if set(groups) <= set(prev["baselines_bootstrap"].group):
            return
    dec_rows, shr_rows, tmp_rows, ten_rows, base_rows = [], [], [], [], []
    sel = {}
    for g in groups:
        p = EVAL / "metrics" / f"{g}_model_selection.csv"
        if p.exists():
            s = pd.read_csv(p)
            for t, d in s.groupby("target"):
                singles = d[d.model.isin(FAMILIES)]
                if len(singles):
                    sel[(g, t)] = singles.sort_values("valid_rmse").iloc[0]["model"]
        f = EVAL / "data" / f"{g}_test_predictions.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f, low_memory=False)
        df["game_date"] = pd.to_datetime(df.game_date)
        D = load_group(g)
        ten = D["test"][["game_pk", D["id"], "n_prior_games"]].rename(columns={D["id"]: "player_id"}).drop_duplicates(["game_pk", "player_id"])
        df = df.merge(ten, on=["game_pk", "player_id"], how="left")
        u = np.sort(df.game_date.dt.normalize().unique()); cuts = [u[len(u) // 3], u[2 * len(u) // 3]]
        df["third"] = np.where(df.game_date < cuts[0], "early", np.where(df.game_date < cuts[1], "middle", "late"))
        df["month"] = df.game_date.dt.strftime("%Y-%m")
        df["tenure"] = pd.cut(df.n_prior_games, [-1, 19, 59, 119, 10 ** 6], labels=["<20 prior games", "20-59", "60-119", "120+"])
        for t, d in df.groupby("target"):
            if t not in D["targets"]:
                continue
            y, p = d.actual.to_numpy(float), d.pred_final.to_numpy(float)
            dec_rows.append(deciles(y, p).assign(group=g, target=t, prediction="production blend"))
            shr_rows.append({"group": g, "target": t, "split": "test", **shrink_stats(y, p)})
            for col in ("third", "month", "tenure"):
                for k, x in d.groupby(col, observed=True):
                    if len(x) < 30:
                        continue
                    yy, pp = x.actual.to_numpy(float), x.pred_final.to_numpy(float)
                    bb = x.baseline_season_avg.to_numpy(float)
                    rb_ = reg_block(yy, pp)
                    rb_["skill_vs_history_rmse_pct"] = 100 * (1 - rb_["rmse"] / np.sqrt(np.mean((bb - yy) ** 2)))
                    (ten_rows if col == "tenure" else tmp_rows).append({"group": g, "target": t, "period_type": col, "period": str(k), **rb_})
            cands = {"production blend (published)": "pred_final", "direct selected model": "pred_direct_selected",
                     "stack_even (train-only fit)": "pred_stack_even", "stack_opt (train-only fit)": "pred_stack_opt"}
            for fam in FAMILIES:
                cands[f"{fam} (train-only fit)"] = f"pred_{fam}"
            if (g, t) in sel:
                cands["best single family on validation"] = f"pred_{sel[(g, t)]}"
            bases = {"league average": "baseline_league_avg", "history-to-date": "baseline_season_avg",
                     "last 10": "baseline_last10_avg"}
            for cname, ccol in cands.items():
                if ccol not in d.columns or d[ccol].isna().all():
                    continue
                pc = d[ccol].to_numpy(float)
                r = {"group": g, "target": t, "model": cname, **reg_block(y, pc)}
                for bname, bcol in bases.items():
                    if bcol not in d.columns:
                        continue
                    pb = d[bcol].to_numpy(float)
                    (_, _, _), (pct, lo, hi) = boot_rmse_diff(d.game_date.values, y, pc, pb, n=N_BOOT)
                    r[f"rmse_pct_better_than_{bname}"] = pct
                    r[f"ci_lo_vs_{bname}"] = lo; r[f"ci_hi_vs_{bname}"] = hi
                    mae_pct = block_boot(d.game_date.values, {"a": np.abs(pc - y), "b": np.abs(pb - y)},
                                         lambda s: 100 * (1 - s["a"] / s["b"]), n=300)
                    r[f"mae_pct_better_than_{bname}"] = mae_pct[0]
                base_rows.append(r)
            for bname, bcol in bases.items():
                if bcol in d.columns:
                    base_rows.append({"group": g, "target": t, "model": f"baseline: {bname}", **reg_block(y, d[bcol].to_numpy(float))})
        log(f"saved: {g} done")
    new = {"deciles_test_production": pd.concat(dec_rows, ignore_index=True), "shrinkage_test": pd.DataFrame(shr_rows),
           "temporal_players": pd.DataFrame(tmp_rows), "tenure_players": pd.DataFrame(ten_rows),
           "baselines_bootstrap": pd.DataFrame(base_rows)}
    for n, d in new.items():
        old = prev.get(n)
        if old is not None and len(old):
            d = pd.concat([old[~old.group.isin(groups)], d], ignore_index=True)
        _out(f"{n}.csv", d)
    log("saved: written")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
FAM_COLORS = {"rolling": "#2a78d6", "expanding": "#1baf7a", "rate": "#5b9bd5", "opportunity": "#eda100",
              "handedness": "#eb6834", "velocity/pitch_mix": "#8e5bd6", "batted_ball": "#c2185b",
              "true_talent": "#00897b", "matchup": "#6d4c41", "team/opponent": "#546e7a", "lineup": "#f06292",
              "park": "#9e9d24", "other": "#bdbdbd"}


def _save(fig, name):
    AF.mkdir(parents=True, exist_ok=True)
    fig.savefig(AF / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_game_corr():
    p = AM / "game_corr_pearson.csv"
    if not p.exists():
        return
    P = pd.read_csv(p, index_col=0)
    lab = [c.replace("diff_", "") for c in P.columns]
    fig, ax = plt.subplots(figsize=(9.5, 8))
    im = ax.imshow(P.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(lab)), lab, rotation=90, fontsize=8); ax.set_yticks(range(len(lab)), lab, fontsize=8)
    ax.grid(False)
    for i in range(len(lab)):
        for j in range(len(lab)):
            v = P.iat[i, j]
            if i != j and np.isfinite(v) and abs(v) >= 0.5:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5, color="white" if abs(v) > 0.75 else ev.INK)
    fig.colorbar(im, ax=ax, shrink=0.75, label="Pearson r (training games)")
    ax.set_title("Game model: home-minus-away feature correlations")
    ev._caption(fig, "Training window only. Values shown where |r| >= 0.50. home_field is a constant and is omitted.")
    _save(fig, "audit_corr_game.png")


def fig_player_corr(group):
    p = AM / f"{group}_corr_spearman.csv"
    if not p.exists():
        return
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import squareform
    S = pd.read_csv(p, index_col=0).abs().fillna(0)
    A = S.to_numpy().copy(); np.fill_diagonal(A, 1)
    order = leaves_list(linkage(squareform(np.clip(1 - A, 0, None), checks=False), "average"))
    cols = S.columns[order]
    fams = pd.Series([family(c) for c in cols])
    fig = plt.figure(figsize=(13, 6.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1], wspace=0.35)
    ax = fig.add_subplot(gs[0])
    im = ax.imshow(S.loc[cols, cols].to_numpy(), cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    from matplotlib.colors import ListedColormap
    fl = sorted(fams.unique())
    strip = np.array([[fl.index(f) for f in fams]])
    ax2 = ax.inset_axes([0, 1.01, 1, 0.035])
    ax2.imshow(strip, aspect="auto", cmap=ListedColormap([FAM_COLORS.get(f, "#999") for f in fl]), interpolation="nearest")
    ax2.set_axis_off()
    fig.colorbar(im, ax=ax, shrink=0.7, label="|Spearman rho|")
    ax.set_title(f"{group.title()} features, clustered ({len(cols)} features)", pad=18)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=FAM_COLORS.get(f, "#999"), label=f) for f in fl], loc="upper center",
              bbox_to_anchor=(0.5, -0.02), ncol=4, fontsize=7.5)
    # family x family mean |rho|
    axb = fig.add_subplot(gs[1])
    fam_all = pd.Series({c: family(c) for c in S.columns})
    M = pd.DataFrame(index=fl, columns=fl, dtype=float)
    for a in fl:
        for b in fl:
            ia, ib = fam_all[fam_all == a].index, fam_all[fam_all == b].index
            sub = S.loc[ia, ib].to_numpy()
            if a == b:
                sub = sub[np.triu_indices(len(ia), 1)] if len(ia) > 1 else np.array([np.nan])
            M.loc[a, b] = np.nanmean(sub)
    im2 = axb.imshow(M.to_numpy(float), cmap="viridis", vmin=0, vmax=1)
    axb.set_xticks(range(len(fl)), fl, rotation=60, ha="right", fontsize=8); axb.set_yticks(range(len(fl)), fl, fontsize=8)
    axb.grid(False)
    for i in range(len(fl)):
        for j in range(len(fl)):
            v = M.iat[i, j]
            if np.isfinite(v):
                axb.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5, color="white" if v < 0.6 else ev.INK)
    axb.set_title("Mean |rho| within / between families")
    ev._caption(fig, "Training rows only (sample of up to 60,000). Left: every feature, ordered by hierarchical clustering; "
                     "bright blocks are groups of near-duplicate features. Right: diagonal = average |rho| inside a family.")
    _save(fig, f"audit_corr_{group}.png")


def fig_group_importance():
    p = AM / "importance_individual.csv"
    if not p.exists():
        return
    imp = pd.read_csv(p)
    x = imp[imp.family == "xgb"]
    panels = [g for g in ("pitcher", "hitter") if g in x.group.unique()]
    gi = AM / "game_importance.csv"
    fig, axes = plt.subplots(1, len(panels) + (1 if gi.exists() else 0), figsize=(16, 5.6),
                             gridspec_kw={"width_ratios": [1.0] * len(panels) + ([0.55] if gi.exists() else []),
                                          "wspace": 0.45})
    axes = np.atleast_1d(axes)
    for ax, g in zip(axes, panels):
        d = x[x.group == g]
        G = d.groupby(["target", "construct"])["shap_valid"].sum().unstack(fill_value=0)
        G = G.div(G.sum(1), axis=0) * 100
        G = G.loc[[t for t in (PITCHER_COUNTS if g == "pitcher" else HITTER_COUNTS) if t in G.index]]
        G = G[G.mean().sort_values(ascending=False).index]
        im = ax.imshow(G.to_numpy().T, cmap="Blues", vmin=0, vmax=max(40, G.to_numpy().max()), aspect="auto")
        ax.set_xticks(range(len(G.index)), G.index); ax.set_yticks(range(len(G.columns)), G.columns, fontsize=8.5)
        ax.grid(False)
        for i in range(G.shape[0]):
            for j in range(G.shape[1]):
                v = G.iat[i, j]
                if v >= 1:
                    ax.text(i, j, f"{v:.0f}", ha="center", va="center", fontsize=7.5, color="white" if v > 30 else ev.INK)
        ax.set_title(f"{g.title()} (XGBoost, % of mean |SHAP|)")
    if gi.exists():
        ax = axes[-1]
        d = pd.read_csv(gi)
        d = d[d.model.isin(["xgb_raw", "rf_raw", "logistic"]) & d.construct.notna() & d.feature.isna()]
        piv = d.pivot(index="construct", columns="model", values="valid_logloss_increase").fillna(0) * 1000
        piv = piv.loc[piv.mean(1).sort_values().index]
        yy = np.arange(len(piv))
        for k, (m, c) in enumerate(zip(piv.columns, [ev.C1, ev.C2, ev.C3])):
            ax.barh(yy + (k - 1) * 0.26, piv[m], height=0.26, color=c, label=m)
        ax.set_yticks(yy, piv.index, fontsize=8.5); ax.axvline(0, color=ev.MUTED, lw=0.8)
        ax.set_xlabel("validation log-loss increase x1000 when permuted")
        ax.set_title("Game model (grouped permutation)")
        ax.legend(fontsize=8, loc="lower right")
    ev._caption(fig, "Players: share of each target's mean |SHAP| (validation rows, log-count scale for Poisson targets) summed "
                     "over every feature in a construct group, so correlated windows are not split. Game: grouped permutation "
                     "on validation; negative or near-zero = no measurable contribution.")
    _save(fig, "audit_group_importance.png")


def fig_train_valid_test():
    p = AM / "overfit_summary.csv"
    if not p.exists():
        return
    s = pd.read_csv(p)
    s = s[s.family.isin(FAMILIES)]
    best = s.loc[s.groupby(["group", "target"]).valid_rmse.idxmin()]
    gc = AM / "game_calibration.csv"
    groups = [g for g in ("pitcher", "hitter") if g in best.group.unique()]
    fig, axes = plt.subplots(1, len(groups) + (1 if gc.exists() else 0), figsize=(15, 4.6),
                             gridspec_kw={"width_ratios": [6, 8][:len(groups)] + ([2.6] if gc.exists() else [])})
    axes = np.atleast_1d(axes)
    for ax, g in zip(axes, groups):
        d = best[best.group == g].set_index("target")
        d = d.loc[[t for t in (PITCHER_COUNTS if g == "pitcher" else HITTER_COUNTS) if t in d.index]]
        xx = np.arange(len(d))
        for k, (sp, c) in enumerate((("train", ev.MUTED), ("valid", ev.C1), ("test", ev.C2))):
            ax.bar(xx + (k - 1) * 0.27, 100 * d[f"{sp}_skill_dev"], width=0.27, color=c, label=sp)
        ax.set_xticks(xx, [f"{t}\n{f}" for t, f in zip(d.index, d.family)], fontsize=8.5)
        ax.set_ylabel("% Poisson deviance explained\nvs train-mean predictor")
        ax.axhline(0, color=ev.INK2, lw=0.8)
        ax.set_title(f"{g.title()}: train / validation / test skill")
        ax.legend(fontsize=8, ncol=3, loc="upper left")
    if gc.exists():
        ax = axes[-1]
        c = pd.read_csv(gc)
        meta = json.loads((AM / "game_meta.json").read_text())
        base = c[c.model == "baseline_train_home_rate"].set_index("split")["log_loss"]
        d = c[c.model == meta["selected_on_valid"]].set_index("split")
        vals = [100 * (1 - d.loc[sp, "log_loss"] / (base.get(sp, np.nan) if sp != "train" else
                                                    float(-(meta["train_home_rate"] * np.log(meta["train_home_rate"]) + (1 - meta["train_home_rate"]) * np.log(1 - meta["train_home_rate"])))))
                for sp in ("train", "valid", "test")]
        ax.bar([0, 1, 2], vals, color=[ev.MUTED, ev.C1, ev.C2])
        ax.set_xticks([0, 1, 2], ["train", "valid", "test"]); ax.axhline(0, color=ev.INK2, lw=0.8)
        ax.set_ylabel("% log loss reduction vs home-win rate")
        ax.set_title(f"Game: {meta['selected_on_valid']}")
    ev._caption(fig, "Each player target shows the family with the lowest validation RMSE, fit on the training window only "
                     "(same settings and calibration wrapper as production). Skill is measured against the training-mean "
                     "predictor on the same rows, so a harder period does not look like over-fitting.")
    _save(fig, "audit_train_valid_test.png")


def fig_reliability():
    p = AM / "game_buckets.csv"
    if not p.exists():
        return
    b = pd.read_csv(p)
    meta = json.loads((AM / "game_meta.json").read_text())
    show = [meta["selected_on_valid"], "xgb_isotonic", "logistic"]
    show = list(dict.fromkeys(show))
    cols = dict(zip(show, [ev.C1, ev.C2, ev.C4, ev.C3]))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), sharey=True)
    for ax, sp in zip(axes, ("valid", "test")):
        ax.plot([0.25, 0.85], [0.25, 0.85], color=ev.MUTED, lw=1, ls="--")
        for k, m in enumerate(show):
            d = b[(b.model == m) & (b.split == sp) & (b.n >= 5)]
            off = (k - 1) * 0.006
            ax.errorbar(d.mean_p + off, d.win_rate, yerr=[d.win_rate - d.ci_lo, d.ci_hi - d.win_rate], fmt="o-",
                        color=cols[m], ms=4, lw=1.2, capsize=2, label=m)
        d = b[(b.model == show[0]) & (b.split == sp)]
        for i, (_, r) in enumerate(d.iterrows()):
            ax.annotate(f"n={r.n}", (r.mean_p, 0.205 if i % 2 else 0.235), fontsize=7, ha="center", color=ev.INK2)
        ax.set_xlim(0.2, 0.9); ax.set_ylim(0.18, 0.92)
        ax.set_xlabel("mean predicted home-win probability in bucket")
        ax.set_title("Validation (Mar-Jun 2026)" if sp == "valid" else "Test (Jul 2026 on, untouched)")
    axes[0].set_ylabel("observed home-win rate (95% Wilson interval)")
    axes[0].legend(fontsize=8, loc="upper left")
    ev._caption(fig, f"Bucket edges {', '.join(f'{x:.2f}' for x in BUCKETS[1:-1])}; buckets with fewer than 5 games are not drawn. "
                     f"Counts shown for {show[0]}. Error bars are 95% "
                     "intervals for the observed rate; a bucket whose interval covers the diagonal is consistent with "
                     "calibration at this sample size.")
    _save(fig, "audit_reliability.png")


def fig_deciles():
    p = AM / "deciles_test_production.csv"
    if not p.exists():
        return
    d = pd.read_csv(p)
    groups = [g for g in ("pitcher", "hitter") if g in d.group.unique()]
    fig, axes = plt.subplots(1, len(groups), figsize=(14, 4.8), gridspec_kw={"wspace": 0.35})
    axes = np.atleast_1d(axes)
    pal = [ev.C1, ev.C2, ev.C3, ev.C4, "#8e5bd6", "#c2185b", "#546e7a", "#6d4c41"]
    for ax, g in zip(axes, groups):
        x = d[d.group == g]
        for k, (t, dd) in enumerate(x.groupby("target", sort=False)):
            ratio = dd.mean_actual / dd.mean_pred
            ax.plot(dd.decile, ratio, "o-", ms=3.5, lw=1.3, color=pal[k % len(pal)], label=t)
        ax.axhline(1, color=ev.INK2, lw=0.9)
        ax.set_xticks(range(1, 11)); ax.set_xlabel("decile of predicted value (1 = lowest)")
        ax.set_ylabel("mean actual / mean predicted")
        ax.set_title(f"{g.title()}: decile calibration on test (published blend)")
        ax.legend(fontsize=8, ncol=1, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ev._caption(fig, "Above 1 in the top deciles and below 1 in the bottom deciles = predictions compressed toward the mean "
                     "(regression to the mean); the opposite pattern would indicate over-confident, over-fit predictions.")
    _save(fig, "audit_deciles.png")


def fig_error_corr():
    p = AM / "model_diversity.csv"
    if not p.exists():
        return
    d = pd.read_csv(p)
    d["label"] = d.group.str[0].str.upper() + ": " + d.target
    pairs = [("rf", "xgb"), ("rf", "lin"), ("xgb", "lin")]
    E = d[[f"err_corr_{a}_{b}" for a, b in pairs]].to_numpy()
    Pm = d[[f"pred_corr_{a}_{b}" for a, b in pairs]].to_numpy()
    g = AM / "game_diversity.csv"
    labels = list(d.label)
    if g.exists():
        gd = pd.read_csv(g)
        gd = gd[gd.split == "valid"].set_index(["a", "b"])
        gp = [("rf_raw", "xgb_raw"), ("rf_raw", "logistic"), ("xgb_raw", "logistic")]
        E = np.vstack([E, [gd.loc[k, "err_corr"] if k in gd.index else np.nan for k in gp]])
        Pm = np.vstack([Pm, [gd.loc[k, "pred_corr"] if k in gd.index else np.nan for k in gp]])
        labels.append("Game (win prob)")
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.33 * len(labels) + 2.2))
    for ax, M, ttl in ((axes[0], Pm, "Prediction correlation"), (axes[1], E, "Error correlation")):
        im = ax.imshow(M, cmap="magma_r", vmin=0.5, vmax=1, aspect="auto")
        ax.set_xticks(range(3), ["RF-XGB", "RF-linear", "XGB-linear"]); ax.set_yticks(range(len(labels)), labels, fontsize=8.5)
        ax.grid(False)
        for i in range(M.shape[0]):
            for j in range(3):
                if np.isfinite(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7.5, color="white" if M[i, j] > 0.85 else ev.INK)
        ax.set_title(ttl)
    fig.colorbar(im, ax=axes, shrink=0.6)
    ev._caption(fig, "Validation window; each family fit on the training window only. Error correlation near 1 means the "
                     "families make the same mistakes, so averaging them cannot remove much error.")
    _save(fig, "audit_error_corr.png")


def step_figures(force=True):
    for f in (fig_game_corr, lambda: fig_player_corr("pitcher"), lambda: fig_player_corr("hitter"), fig_group_importance,
              fig_train_valid_test, fig_reliability, fig_deciles, fig_error_corr):
        try:
            f()
        except Exception as e:                       # a missing input skips one figure, not the run
            log(f"figure failed: {e!r}")
    log("figures: written")


# ---------------------------------------------------------------------------
def main():
    global QUICK, FAMILIES, ONLY_TARGETS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", nargs="+", default=["map", "redundancy", "fits", "curves", "ablation", "game", "saved", "figures"])
    ap.add_argument("--groups", nargs="+", default=["pitcher", "hitter"])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--targets", nargs="+", default=None, help="limit player steps to these targets (e.g. K H PA)")
    a = ap.parse_args()
    QUICK = a.quick
    ONLY_TARGETS = a.targets
    for d in (AM, AF, AC):
        d.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    log(f"audit: valid {VALID_START}, test {TEST_START}, steps {a.steps}, groups {a.groups}{' (QUICK)' if QUICK else ''}")
    if "map" in a.steps: step_map(a.groups, a.force)
    if "redundancy" in a.steps: step_redundancy(a.groups, a.force)
    if "game" in a.steps: step_game(a.force)
    if "fits" in a.steps: step_fits(a.groups, a.force)
    if "saved" in a.steps: step_saved(a.groups, a.force)
    if "ablation" in a.steps: step_ablation(a.groups, a.force)
    if "curves" in a.steps: step_curves(a.groups, a.force)
    if "figures" in a.steps: step_figures()
    json.dump({"run_utc": pd.Timestamp.utcnow().isoformat(timespec="seconds"), "valid_start": VALID_START,
               "test_start": TEST_START, "quick": QUICK, "groups": a.groups, "minutes": round((time.time() - t0) / 60, 1)},
              open(AM / "audit_run.json", "w"), indent=2)
    log(f"audit finished in {(time.time()-t0)/60:.1f} min -> {AUD}")


if __name__ == "__main__":
    main()
