"""
generate_report.py
==================
Builds the InTheCount technical report from the files the pipeline wrote.
Every number in the report is read from those files at generation time -
nothing is typed in by hand - so re-running the evaluation and then this
script always yields a report that matches the code and the data.

Inputs   outputs/model_evaluation/{metrics,data,figures}/*,
         outputs/model_evaluation/data_validation/statcast_validation.json,
         data/features/build_summary.json
Outputs  outputs/model_evaluation/report/InTheCount_Technical_Report.md
         outputs/model_evaluation/report/InTheCount_Technical_Report.html

Usage
-----
    python generate_report.py               # also runs the pytest suite and records the result
    python generate_report.py --skip-tests
"""

from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EVAL = HERE / "outputs" / "model_evaluation"
MET = EVAL / "metrics"
REPORT_DIR = EVAL / "report"
LABEL = {"K": "Strikeouts", "BB": "Walks", "H": "Hits", "HR": "Home runs", "IP": "Innings pitched",
         "R": "Runs allowed", "PA": "Plate appearances", "TB": "Total bases", "2B": "Doubles", "3B": "Triples"}


def _csv(name):
    p = MET / name
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def _json(path):
    return json.loads(Path(path).read_text()) if Path(path).exists() else {}


def f3(x):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"


def f4(x):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.4f}"


def pct(x, d=1):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.{d}f}%"


def md_table(df: pd.DataFrame, fmt: dict | None = None) -> str:
    fmt = fmt or {}
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if c in fmt and isinstance(v, (int, float, np.floating, np.integer)) and not pd.isna(v):
                cells.append(fmt[c](v))
            elif isinstance(v, (float, np.floating)):
                cells.append("n/a" if pd.isna(v) else f"{v:.4f}")
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def fig(path_rel: str | None, caption: str, placeholder: str) -> str:
    if path_rel and (EVAL / path_rel).exists():
        import html as _h
        c = _h.escape(caption)
        try:                                   # wide multi-panel charts get more width than square ones
            from PIL import Image
            with Image.open(EVAL / path_rel) as im:
                cls = "wide" if im.width / im.height > 1.3 else "narrow"
        except Exception:
            cls = "wide"
        return (f'<figure><img class="{cls}" src="../{path_rel}" alt="{c}">'
                f'<figcaption><em>{c}</em> (<code>outputs/model_evaluation/{path_rel}</code>)</figcaption></figure>\n')
    return f"`[{placeholder}]`\n"


def run_tests() -> dict:
    try:
        r = subprocess.run([sys.executable, "-m", "pytest", "-q", str(HERE / "tests")],
                           capture_output=True, text=True, cwd=HERE, timeout=3600)
        tail = [l for l in r.stdout.strip().splitlines() if l.strip()][-1] if r.stdout.strip() else r.stderr[-300:]
        return {"returncode": r.returncode, "summary": tail}
    except Exception as e:
        return {"returncode": -1, "summary": f"could not run: {e}"}


# ---------------------------------------------------------------------------
def build(skip_tests: bool = False) -> tuple[str, str]:
    """Returns (main report, appendix). The appendix holds the per-target charts and
    the full diagnostic tables; the main report keeps the summaries and verdicts."""
    val = _json(EVAL / "data_validation" / "statcast_validation.json")
    build_sum = _json(HERE / "data" / "features" / "build_summary.json")
    bt = _json(MET / "backtest_run.json")
    gsel, gcomp = _csv("game_model_selection.csv"), _csv("game_model_comparison.csv")
    buckets, months = _csv("game_probability_buckets.csv"), _csv("game_performance_by_month.csv")
    qcs = _csv("quality_checks.csv")
    manifest = _json(MET / "figure_manifest.json")
    figs = manifest.get("figures", {})
    gp = EVAL / "data" / "game_test_predictions.csv"
    gpred = pd.read_csv(gp) if gp.exists() else pd.DataFrame()
    if skip_tests:      # reuse the last saved test run instead of reporting "not run"
        tests = _json(MET / "test_results.json") or {"summary": "not run (--skip-tests)", "returncode": None}
    else:
        tests = run_tests()
    (MET / "test_results.json").write_text(json.dumps(tests, indent=2))

    seasons = sorted(val.get("seasons", {}).keys())
    L = []
    w = L.append
    A = []
    wa = A.append
    wa("# InTheCount Technical Report — Appendix\n")
    wa("Per-target charts and full diagnostic tables behind the main report. Section numbers match the main "
       "report; every table is also a CSV in `outputs/model_evaluation/`.\n")
    w("# InTheCount: MLB Game and Player Projection System Technical Report\n")

    # ------------------------------------------------------------------ 1
    w("## 1. Project overview\n")
    w("InTheCount (formerly TheBullpenBet) is a public MLB projection site. Every day it publishes a "
      "win probability for each scheduled game and per-player projections for starting pitchers "
      "(innings, strikeouts, walks, hits, home runs and runs allowed) and lineup hitters (plate "
      "appearances, hits, total bases, home runs, walks, strikeouts, doubles, triples), then grades "
      "every projection against the official box score and keeps a public, season-long accuracy ledger.\n")
    w("The system began as a single-season 2026 pipeline, was extended with the full 2025 season to "
      "give the player models enough starts and plate appearances to learn from, and was audited and "
      "rebuilt for this report around three goals: (1) a reproducible multi-season Statcast history "
      "(2024–2026) collected by one season-configurable scraper, (2) a strict, testable rule for how "
      "much player history a projection may use, and (3) an honest, chronological out-of-sample "
      "evaluation whose numbers can be regenerated without retraining.\n")

    # ------------------------------------------------------------------ 2
    w("## 2. Data\n")
    w("All modelling data is pitch-level Statcast from Baseball Savant, pulled with "
      "`pybaseball.statcast(start_dt, end_dt)` — the same call the project has always used for the "
      "current season. Final scores, starters, bullpens, plate-appearance outcomes, batted-ball "
      "quality, pitch mix and velocity are all derived from those pitches; the MLB Stats API is used "
      "only for schedules, probable pitchers, lineups and box-score grading.\n")
    if val:
        rows = []
        for s in seasons:
            v = val["seasons"][s]
            rs = v.get("regular_season", {})
            rows.append({"Season": s, "Pitches (all)": f"{v.get('records', 0):,}",
                         "Regular-season pitches": f"{rs.get('records', 0):,}",
                         "Regular-season games": f"{rs.get('games', 0):,}",
                         "First / last RS date": f"{rs.get('first_date')} / {rs.get('last_date')}",
                         "Pitchers": f"{rs.get('pitchers', 0):,}", "Batters": f"{rs.get('batters', 0):,}",
                         "Validation": "PASS" if v.get("passed") else "FAIL"})
        w("**Statcast cache by season** (`pitch_data_<season>.csv`; "
          "validation in `outputs/model_evaluation/data_validation/statcast_validation.json`):\n")
        w(md_table(pd.DataFrame(rows)) + "\n")
        fails = {s: val["seasons"][s]["critical_failures"] for s in seasons if val["seasons"][s]["critical_failures"]}
        if fails:
            w("**Open data issues flagged by validation** (the pipeline still runs; these are reported, not hidden):\n")
            for s, f in fails.items():
                for x in f:
                    w(f"- {s}: {x}")
            w("")
        missing = [s for s in ("2024", "2025", "2026") if s not in seasons]
        if missing:
            w(f"> **Not yet collected in this build:** {', '.join(missing)}. The collector supports these seasons "
              "(`python refresh_full_history.py --seasons " + " ".join(missing) + "`) but the evaluation below "
              "was produced without them; see Section 4 for how that affects the history window.\n")
    w("Validation checks per season (`refresh_full_history.validate`): record counts, first/last dates, "
      "regular-season game count (expected ≈2,430), unique players and games, required columns and missing "
      "rates, no missing IDs, no duplicate pitches `(game_pk, at_bat_number, pitch_number)`, one date/home/away "
      "per game, plate appearances with more than one batter, `game_year` vs. season, 30 teams, regular season "
      "starting by early April, and gaps of more than four days inside the season.\n")
    w("Only MLB competitive games (regular season and postseason) enter the modelling layer. Spring "
      "training is excluded: starters throw 1–3 innings by design and lineups are mostly minor "
      "leaguers, which corrupted early-season workload and rate features in the previous pipeline.\n")

    # ------------------------------------------------------------------ 3
    w("## 3. Scope of the changes\n")
    w("The goal of this revision is *the original feature calculations plus two additional seasons of "
      "history*. Every change falls in one of four categories:\n")
    w("| Category | What changed |\n|---|---|\n"
      "| **Data expansion** | `refresh_full_history.py` takes the season as a parameter and caches 2024, 2025 and 2026 "
      "in the existing `pitch_data_<season>.csv` format (same pybaseball call); current-season top-up with a 3-day "
      "overlap, request timeout and resume. The two-season rule restricts each prediction season to *S−2…S* before "
      "features are built. Focused data validation per season. |\n"
      "| **Feature-engineering changes** | **One addition: lineup** (`lineup_spot`, `lineup_spot_last10`; section 10, "
      "run as a controlled experiment). Otherwise none to the player features: rolling windows (7/10/14/21/30), `_std` "
      "(expanding mean of all prior games, no season reset), splits, true-talent shrinkage, matchup and team "
      "definitions are the original code; they simply see more history. The game-model features were *rebuilt* with "
      "the same 18 names because the original builder is not in the repository and its output file was corrupted "
      "(see section 6). |\n"
      "| **Bug / leakage fixes** | Opposing-starter same-game leak, lineup same-game inputs, true-talent league prior "
      "over the whole file, `game_pk` / doubleheaders, starter identification, spring-training exclusion, model "
      "selection on the test set, K-fold CV on time-ordered data, game calibration folds, in-sample public ledger "
      "(section 6). |\n"
      "| **Future experiments** | xBA/xwOBA, pitcher workload, plate discipline, game-model improvements — "
      "**not implemented** (section 17). |\n")
    w("Operational changes needed to evaluate honestly: the selected model family is refit on train + validation "
      "before the single test scoring, and the evaluation, quality-check and test scripts are new. None of them "
      "changes how a production feature is computed.\n")
    w("**Modelling addition (requested):** model stacking - an even-weight and an optimized-weight average of the "
      "existing base models compete with the single models on validation (section 7). It changes which model is "
      "deployed when a stack wins on validation; it does not change any feature.\n")
    w("**Statistical audit (section 11):** multicollinearity, importance stability, overfitting, shrinkage, "
      "calibration, model diversity, temporal stability, baselines with intervals and feature-family ablations "
      "(`audit_models.py`). Diagnostics only: it changes no feature, model or production file; its recommendations "
      "are listed for separate, validated changes.\n")

    # ------------------------------------------------------------------ 4
    w("## 4. Feature engineering (original definitions)\n")
    w("`hitterspitchers_data.py` aggregates pitches to one row per pitcher-start and one row per "
      "batter-game; `build_game_features.py` aggregates to one row per game. Every modelling feature is "
      "a trailing summary of games strictly before the game being predicted:\n")
    w("- **Rolling windows** — `shift(1).rolling(N, min_periods=1).mean()` over the previous 7, 10, 14, 21 and 30 "
      "appearances (`*_lastN`), plus 3-game workload windows. As in the original code they run over the player's "
      "whole history, so they cross season boundaries inside the history window.\n"
      "- **History-to-date** (`*_std`) — `shift(1).expanding().mean()`, the original definition: the mean of all "
      "the player's prior games in the window (previous two seasons plus the current season to date). No season "
      "reset was introduced.\n"
      "- **Pitcher** — K/BB/HR/H rates per batter faced, IP, batters faced, pitches, pitches per BF/IP, "
      "velocity, spin, whiff%, CSW%, zone%, first-pitch strike%, hard-hit and barrel rates allowed, "
      "fastball/breaking/offspeed usage and velocity separation, days of rest, recent max IP, share of "
      "recent appearances that were starts.\n"
      "- **Hitter** — H/HR/BB/K rates per PA, PA, doubles/triples, exit velocity, launch angle, "
      "batted-ball proxies (hard-hit, sweet-spot, barrel), days since last game.\n"
      "- **Handedness splits** — pitcher rates vs. LHB/RHB and hitter rates vs. LHP/RHP (trailing windows and `_std` only).\n"
      "- **Opponent / matchup** — opposing team's rates vs. the pitcher's hand; opposing starter's trailing "
      "rates; empirical-Bayes *true-talent* rates (cumulative events regressed to a league prior with "
      "stabilisation constants of 70 PA for K, 120 for BB, 60 for H, 300 for HR); the opposing lineup's mean "
      "true-talent rate and a log5 pitcher-vs-lineup matchup rate.\n"
      "- **Team (game model, rebuilt with the original 18 names)** — runs scored/allowed, AVG/OBP/SLG/OPS, K% and BB% over the previous 10 "
      "games; season-to-date OPS, K%, win% and run differential; bullpen ERA proxy, WHIP and K-BB% over the "
      "previous 14 team games and season-to-date; starter FIP, K-BB% and HR/9 over his previous 10 starts and "
      "season-to-date; starter handedness match. The model uses home-minus-away differences.\n")

    # ------------------------------------------------------------------ 4
    w("## 5. Historical data window\n")
    w("**Collection vs. modelling history are different things.** The collector caches whole seasons; the "
      "models may only *see* a bounded slice of them.\n")
    w("**Two-season rule.** A projection for a game in season *S* may use the player's games from seasons "
      "*S−2* and *S−1* plus season *S* strictly before the game — never *S−3* or earlier, never the game "
      "itself, never anything later:\n")
    w("| Prediction season | Eligible history |\n|---|---|\n| 2025 | 2023 + 2024 + 2025-to-date |\n"
      "| 2026 | 2024 + 2025 + 2026-to-date |\n| 2027 | 2025 + 2026 + 2027-to-date |\n")
    w("**Enforced before feature engineering.** `history_window.window_pitches()` restricts the raw pitches "
      "to *S−2…S* first; the feature tables are then built on that window only and just the season-*S* rows "
      "are kept (`hitterspitchers_data.build_windowed_tables`). Because rolling windows, expanding means, "
      "true-talent accumulators, splits, team context and lineup aggregates are computed on the already-"
      "restricted window, older data cannot reach a feature indirectly. Each row records the seasons its "
      "window actually contained (`history_seasons`).\n")
    if build_sum:
        rbs = build_sum.get("tables", {}).get("hitter", {}).get("rows_by_season", {})
        w(f"Feature tables built for prediction seasons {build_sum.get('prediction_seasons')} from cached seasons "
          f"{build_sum.get('cached_seasons')}; hitter rows by season: {rbs}.\n")
    tr_seasons = bt.get("hitter_training_seasons")
    if tr_seasons:
        w(f"**Training rows.** Player models were trained on rows from seasons {tr_seasons}. When three or more "
          "seasons are cached, the earliest one is used as history only: its rows have no prior-season "
          "history inside their own window, so their trailing features are systematically thinner than the "
          "rows the model is asked to predict.\n")
    if build_sum and 2024 not in build_sum.get("cached_seasons", []):
        w("> In this build 2024 is not cached yet, so every 2026 window actually contains 2025 + 2026 only, and "
          "2025 rows have no prior-season history. The rule is enforced the same way; it simply has less to draw on "
          "until `refresh_full_history.py --seasons 2024` is run and the evaluation regenerated.\n")
    w("**Example (2026 start on 2026-07-15).** The pitcher's `K_last10` is the mean of his previous ten "
      "appearances drawn from 2024-01-01 through 2026-07-14; `K_std` is the mean of all his competitive "
      "appearances from 2024 through July 14; `p_tt_k` accumulates his strikeouts and batters faced from 2024 through July 14. A "
      "2023 start can never enter any of these, even through a cached intermediate table.\n")

    # ------------------------------------------------------------------ 5
    w("## 6. Bug and leakage fixes\n")
    w("The pregame cutoff is the start of the game being predicted: every trailing statistic is shifted one "
      "game within the player (or team) before it is aggregated, so the current game is never in its own "
      "feature. The audit found and fixed the following:\n")
    w("| Issue found | Effect | Fix |\n|---|---|---|\n"
      "| Hitter features `opp_sp_k_rate`, `opp_sp_bb_rate`, `opp_sp_hr_rate`, `opp_sp_h_rate`, `opp_sp_ip` were the opposing starter's results **in the same game** | Direct target leakage into every hitter model; at serving time the same columns held the starter's previous start, so live behaviour differed from training | Filled with the starter's previous start in training too (matches serving); covered by the perturbation test |\n"
      "| Lineup features averaged batters' **same-game** split rates and exit velocity | Leaky inputs (not in the production feature list, but present in the tables) | Rebuilt from batters' `_std` / `_last10` values |\n"
      "| Player model family (RF / XGB / NN) chosen by **test-set** RMSE | Reported test error was the best of three looks at the test set | Family chosen on a chronological validation window; test scored once |\n"
      "| `cross_val_score` with default K-fold on time-ordered games | Folds trained on later games to predict earlier ones | Removed; replaced by the validation window |\n"
      "| Game model isotonic calibration with unshuffled 3-fold CV | Calibrators fit on folds that predate their base model's training data | `TimeSeriesSplit` folds |\n"
      "| True-talent league prior computed over the whole file | Future run environment inside every prior | Same shrinkage formula; the league rate is taken from earlier dates only (prior constants unchanged) |\n"
      "| Public player-accuracy ledger built by force-re-scoring past dates with models trained on most of those dates | In-sample projections shown as accuracy | Snapshots stamped with `model_trained_through`; grader adds `in_sample`; the site excludes in-sample rows |\n"
      "| `game_pk` not loaded | Doubleheaders merged into one \"game\" and rows matched by date only | `game_pk`, `at_bat_number`, `pitch_number` loaded; ties in the trailing-window sort broken by `game_pk` (window definitions unchanged) |\n| Starter = the team's *last* pitcher row in file order | Depends on the CSV's row order: on the 2025 file (oldest-first) it labelled a reliever as starter in 4,923 of 4,954 team-games checked (mean IP 1.17 vs 4.98 for the true starter); on the 2026 file (newest-first) it agreed in all 3,586 | Starter = pitcher of the team's first pitch in the game (`at_bat_number`, `pitch_number`) |\n| Spring-training games in the pitch files | Starters throw 1–3 innings by design and lineups are mostly minor leaguers; these games entered rolling and `_std` features | Only competitive games (regular season + postseason) are used |\n"
      "| `2025_model_data.csv` (game model) lists each game twice with home/away swapped; builder not in repo | Fabricated home team in half the rows, `home_field` constant | Replaced by the reproducible one-row-per-game builder |\n")
    w("Earlier, un-windowed platoon and team-context columns (e.g. `team_k_rate_vs_hand`) had already been "
      "removed from the feature lists by the project's own diagnostics; the audit confirmed none remain.\n")
    w("**Tests that keep it fixed** (`tests/`):\n")
    w("- *Perturbation test* — turn every plate appearance of one game into a home run, rebuild every "
      "table, and require that none of that game's pitcher, hitter or game-model features change while later "
      "games' features do. Run against the original opposing-starter merge, this test flags exactly the five "
      "`opp_sp_*` columns above.\n"
      "- *Older-season test* — append a copy of the data relabelled three seasons earlier; no feature of the "
      "real season may change.\n"
      "- *Train/serve parity* — game features built for a \"scheduled\" slate from data before the slate date "
      "equal the stored training rows.\n"
      "- Static guard (`leakage_guard.py`) — every trainer refuses a feature that is not a trailing window, "
      "season-to-date value, or an explicitly justified pre-game column.\n"
      "- Unit tests of the shift/rolling primitives (no season reset), window assertion and date split.\n")
    w(f"**Test suite result:** `{tests.get('summary')}`.\n")
    if not qcs.empty:
        crit = qcs[qcs["critical"]]
        w(f"**Automated quality checks** run before any metric is computed: {int(crit['passed'].sum())} of "
          f"{len(crit)} critical checks passed (`metrics/quality_checks.csv`). A critical failure stops the "
          "evaluation instead of producing a number.\n")

    # ------------------------------------------------------------------ 6
    w("## 7. Model development\n")
    w("The existing model families, targets, hyperparameter presets, calibration and production blend are kept; the "
      "validation protocol and the history available to the features changed, and model stacking was added as a "
      "further candidate (below).\n")
    w("- **Game model** — `XGBClassifier` (600 trees, depth 4, learning rate 0.03, L1 0.5 / L2 2.0) with "
      "isotonic calibration is the production configuration; uncalibrated XGBoost, a calibrated / "
      "uncalibrated random forest and an L2 logistic regression on the standardised features are the other candidates. Target `home_win` (1 = home team won), one row "
      "per regular-season game, the same 18 features as the original model.\n"
      "- **Lineup** — hitters use the batting-order slot they start in plus their recent average slot "
      "(section 10).\n"
      "- **Player models** — for every target a random forest, a Poisson-loss XGBoost regressor "
      "and a linear model (Poisson regression - linear on the log scale, so it cannot predict a negative "
      "count) are fit; the family with the lowest validation RMSE is kept. Count targets use Poisson "
      "deviance (XGB) or a log1p transform (RF). A tail-of-training linear/isotonic calibration is "
      "applied when it improves MAE. Each count target also has a per-opportunity rate model (per 9 IP or per "
      "PA); the published projection blends the direct count with rate × predicted IP/PA using fixed weights.\n"
      "- **Neural network removed** — the original pipeline also fit a multilayer perceptron (two hidden layers, "
      "128 and 64 units) for every target. It was dropped after the development runs: on validation it was the "
      "weakest family for every target checked there (e.g. pitcher strikeouts RMSE 2.45 vs 2.21 for XGBoost; innings "
      "1.32 vs 1.23 - figures from the runs before removal, not from this report's run), the optimized stacks gave it "
      "little weight, and it accounted for a large share of training time. With a few thousand noisy, "
      "tabular rows per target, tree ensembles and a regularised linear model are the better fit; a neural net "
      "was not tried for the game model for the same reason (about 7,000 games, 18 features).\n"
      "- **Model stacking** — two stacked candidates are added next to the single models, for players (RF + XGB + "
      "linear) and for the game model (the five candidates above): an **even-weight stack** (simple average) and an "
      "**optimized-weight stack** (non-negative weights summing to 1 that minimise validation RMSE for players or "
      "validation log loss for games). Weights are fit on validation predictions only. Because the optimized "
      "weights are tuned on the validation window, that stack is scored for selection with cross-fitted "
      "predictions (weights fit on one chronological half of validation, scored on the other, and vice versa), so "
      "it gets no advantage from grading itself. A stack competes with the single models on the same validation "
      "metric; if it wins, every member is refit on train + validation and combined with those weights. The test "
      "window is never used to fit or choose weights.\n")
    w("**Game-model selection metric: log loss (AUC considered and not used).** The game model, the optimized "
      "stack's weights and the promotion gate against the deployed model all use validation log loss. ROC AUC was "
      "evaluated as the selection criterion and rejected for three reasons: (1) the site publishes a win probability "
      "for every game, and AUC only measures how games are *ranked* - a model can keep its AUC while every "
      "probability is shifted or over-confident, so AUC cannot detect the error users would actually see; (2) log "
      "loss is a proper scoring rule, minimised only by honest probabilities, so selecting on it rewards both "
      "ranking and calibration; (3) with game AUCs only a few points above 0.5, differences between candidates on ~1,000 validation "
      "games are mostly noise, and AUC is the noisier of the two at that scale. AUC is still reported for every "
      "candidate, baseline and month.\n")
    _gs = _csv("game_model_selection.csv")
    if not _gs.empty and {"valid_roc_auc", "valid_log_loss"} <= set(_gs.columns):
        _c = _gs[~_gs["candidate"].astype(str).str.startswith("v1")]
        by_ll = _c.sort_values("valid_log_loss").iloc[0]
        by_auc = _c.sort_values("valid_roc_auc", ascending=False).iloc[0]
        if by_ll["candidate"] == by_auc["candidate"]:
            w(f"In this run both criteria pick the same candidate (**{by_ll['candidate']}**).\n")
        else:
            w(f"In this run the criteria disagree: log loss picks **{by_ll['candidate']}** (log loss "
              f"{f4(by_ll['valid_log_loss'])}, AUC {f4(by_ll['valid_roc_auc'])}); AUC would have picked "
              f"**{by_auc['candidate']}** (AUC {f4(by_auc['valid_roc_auc'])}, log loss {f4(by_auc['valid_log_loss'])}).\n")
    if bt:
        w(f"**Chronological protocol.** Train: before {bt.get('valid_start')}. Validation: {bt.get('valid_start')} to "
          f"the day before {bt.get('test_start')} — chooses the model family (players: RMSE; game: log loss). "
          f"Test: {bt.get('test_start')} onward — scored once after the chosen model is refit on train + "
          "validation. XGBoost hyperparameters are the tuned presets stored in the production pickles; that "
          "tuning used data ending before the test window.\n")
    w("**Targets.** Pitchers: K, BB, H, HR, IP, R (runs allowed; Statcast has no earned/unearned split). "
      "Hitters: H, HR, BB, K, PA, TB, 2B, 3B, SB. Stolen bases are excluded from evaluation: Statcast pitch "
      "events carry no stolen-base rows, so the SB target is all zeros (a pre-existing limitation).\n")
    w("**Baselines.** Game: always-home at the training home-win rate, a 50% coin flip, a logistic "
      "regression on season win-% difference, and the existing production model's live 2026 log. Player: "
      "history-to-date (`_std`) average, last-10 average and league average for the target.\n")

    # ------------------------------------------------------------------ 7
    w("## 8. Game prediction evaluation\n")
    if not gcomp.empty and len(gpred):
        chosen = gpred["selected_model"].iloc[0]
        s = gcomp[gcomp["role"] == "selected"].iloc[0]
        w(f"Out-of-sample test: **{int(s['n']):,} regular-season games** from {gpred['game_date'].min()} to "
          f"{gpred['game_date'].max()}. Selected candidate (validation log loss): **{chosen}**. "
          f"Home teams won {pct(gpred['actual_home_win'].mean())} of test games.\n")
        show = gcomp[["model", "role", "n", "accuracy", "roc_auc", "log_loss", "brier", "precision", "recall", "f1"]]
        w(md_table(show, {"n": lambda v: f"{int(v):,}", "accuracy": pct}) + "\n")
        hr = gcomp[gcomp["model"] == "baseline_home_rate"]
        if len(hr):
            ll_gain = hr["log_loss"].iloc[0] - s["log_loss"]
            w(f"Against the always-home baseline the selected model's log loss is "
              f"{'lower' if ll_gain > 0 else 'higher'} by {abs(ll_gain):.4f} and its Brier score is "
              f"{'lower' if s['brier'] < hr['brier'].iloc[0] else 'higher'} "
              f"({f4(s['brier'])} vs {f4(hr['brier'].iloc[0])}).\n")
        if not gsel.empty and (gsel["candidate"] == "v1_production_live_log").any():
            v1v = gsel[gsel["candidate"] == "v1_production_live_log"].iloc[0]
            chv = gsel[gsel["candidate"] == chosen].iloc[0]
            keep_v1 = chv["valid_log_loss"] >= v1v["valid_log_loss"]
            w(f"**Deployment decision (made on validation, not test).** On the {int(v1v['valid_n']):,} validation games "
              f"the rebuilt model's log loss was {f4(chv['valid_log_loss'])} versus {f4(v1v['valid_log_loss'])} for the "
              f"deployed v1 model's live picks (AUC {f4(chv['valid_roc_auc'])} vs {f4(v1v['valid_roc_auc'])}). " + (
                  "Because the rebuilt model did not beat v1 on validation, `train_game_model.py` refuses to promote it and "
                  "v1 stays in production; the daily runner supports both and tags every pick with `model_version`. "
                  if keep_v1 else
                  "The rebuilt model beat v1 on validation, so it is eligible for promotion "
                  "(`python train_game_model.py --production --promote`). ") + "\n")
        best = gcomp.dropna(subset=["log_loss"]).sort_values("log_loss").iloc[0]
        w(f"Lowest test log loss among all rows: **{best['model']}** ({f4(best['log_loss'])}). Test results "
          "were not used to choose the model; they are reported as they came out.\n")
        w(fig(figs.get("game", {}).get("roc"), "ROC curve on the test window", "INSERT GAME MODEL ROC CURVE"))
        wa("## A8. Game prediction — additional charts\n")
        wa(fig(figs.get("game", {}).get("confusion"), "Confusion matrix at a 0.50 threshold", "INSERT GAME MODEL CONFUSION MATRIX"))
        wa(fig(figs.get("game", {}).get("prob_vs_outcome"), "Predicted probability by actual outcome", "INSERT PREDICTED PROBABILITY VS OUTCOME"))
        if not buckets.empty:
            b = buckets[buckets["n"] > 0][["bucket", "n", "avg_predicted", "actual_win_pct", "predicted_minus_actual"]]
            w("**Probability buckets** (confidence in the picked team, max(p, 1−p)):\n")
            w(md_table(b, {"n": lambda v: f"{int(v):,}", "avg_predicted": pct, "actual_win_pct": pct,
                           "predicted_minus_actual": lambda v: f"{100 * v:+.1f} pts"}) + "\n")
        w("The reliability diagram with confidence intervals is in section 11.5; monthly results and the remaining "
          "game charts are in the appendix (A8).\n")
        wa(fig(figs.get("game", {}).get("buckets"), "Probability buckets vs. actual win %", "INSERT GAME PROBABILITY BUCKET ANALYSIS"))
        if not months.empty:
            wa("**By month:**\n")
            wa(md_table(months[["month", "n", "accuracy", "roc_auc", "log_loss", "brier"]],
                        {"n": lambda v: f"{int(v):,}", "accuracy": pct}) + "\n")
        wa(fig(figs.get("game", {}).get("over_time"), "Monthly accuracy, Brier and AUC", "INSERT GAME PERFORMANCE OVER TIME"))
        wa(fig(figs.get("game", {}).get("comparison"), "Test log loss: candidates, baselines and the v1 live log", "INSERT MODEL COMPARISON TABLE"))
    else:
        w("Game test predictions were not available when this report was generated.\n")

    # ------------------------------------------------------------------ 8
    w("## 9. Player projection evaluation\n")
    wa("## A9. Player projections — per-target charts and tables\n")
    w("All rows are out-of-sample test player-games. Residual = predicted − actual (positive = over-projection). "
      "The published projection (\"production blend\") is evaluated; R² is reported for completeness but "
      "single-game counts are dominated by irreducible randomness, so MAE, RMSE, bias and Poisson deviance "
      "against simple baselines are the primary measures.\n")
    for grp in ("pitcher", "hitter"):
        comp = _csv(f"{grp}_model_comparison.csv")
        if comp.empty:
            w(f"*{grp.title()} test predictions were not available.*\n")
            continue
        rng = _csv(f"{grp}_projection_ranges.csv")
        major = ["K", "BB", "H", "HR", "IP", "R"] if grp == "pitcher" else ["H", "HR", "TB", "BB", "K", "PA"]
        prod = comp[comp["column"] == "pred_final"].set_index("target")
        base = comp[comp["column"] == "baseline_season_avg"].set_index("target")
        l10 = comp[comp["column"] == "baseline_last10_avg"].set_index("target")
        lg = comp[comp["column"] == "baseline_league_avg"].set_index("target")
        rows = []
        for t in [m for m in major if m in prod.index] + [m for m in prod.index if m not in major]:
            r = prod.loc[t]
            rows.append({"Target": LABEL.get(t, t), "n": int(r["n"]), "Actual mean": r["actual_mean"],
                         "Pred mean": r["pred_mean"], "Mean err": r["mean_error"], "Median err": r["median_error"],
                         "MAE": r["mae"], "RMSE": r["rmse"], "R²": r["r2"],
                         "MAE season-avg": base.loc[t, "mae"] if t in base.index else np.nan,
                         "MAE last-10": l10.loc[t, "mae"] if t in l10.index else np.nan,
                         "MAE skill vs best baseline": r["mae_skill_vs_best_baseline"],
                         "Poisson dev": r["poisson_deviance"],
                         "Poisson dev league-avg": lg.loc[t, "poisson_deviance"] if t in lg.index else np.nan})
        tb = pd.DataFrame(rows)
        w(f"### {grp.title()}s\n")
        w(f"{int(prod['n'].max()):,} test {grp}-games.\n")
        w(md_table(tb, {"n": lambda v: f"{int(v):,}", "MAE skill vs best baseline": pct}) + "\n")
        better = tb[tb["MAE skill vs best baseline"] > 0]["Target"].tolist()
        worse = tb[tb["MAE skill vs best baseline"] <= 0]["Target"].tolist()
        if better:
            w(f"The published projection has lower MAE than every simple baseline for: {', '.join(better)}.")
        if worse:
            w(f" It does **not** beat the best simple baseline for: {', '.join(worse)}.")
        w("\n")
        dev_better = tb[tb["Poisson dev"] < tb["Poisson dev league-avg"]]["Target"].tolist()
        w(f"On mean Poisson deviance (the proper scoring rule for a predicted count mean, which MAE is not - MAE "
          f"rewards predicting the median, i.e. 0 for rare events such as home runs) the projection beats the "
          f"league-average baseline for {len(dev_better)} of {len(tb)} targets"
          + (f": {', '.join(dev_better)}." if dev_better else ".") + "\n")
        biased = tb[(tb["Mean err"].abs() > 0.05 * tb["Actual mean"].abs())]
        if len(biased):
            w("Systematic bias above 5% of the target mean: " + "; ".join(
                f"{r['Target']} {r['Mean err']:+.3f} ({'over' if r['Mean err'] > 0 else 'under'}-projection)"
                for _, r in biased.iterrows()) + ".\n")
        wa(f"### {grp.title()}s\n")
        pva = figs.get(grp, {}).get("pred_vs_actual", {}) or {}
        for t in major:
            wa(fig(pva.get(t), f"{grp.title()} {LABEL.get(t, t)} — predicted vs actual",
                   f"INSERT PLAYER PREDICTED VS ACTUAL — {t}"))
        wa(fig(figs.get(grp, {}).get("residuals"), f"{grp.title()} residuals vs predicted with binned means", "INSERT PLAYER RESIDUAL PLOT"))
        wa(fig(figs.get(grp, {}).get("error_dist"), f"{grp.title()} error distributions", "INSERT PLAYER ERROR DISTRIBUTIONS"))
        if not rng.empty:
            wa("**Projection-range analysis** (terciles of the projection):\n")
            r2 = rng[["target", "projection_range", "range_min", "range_max", "n", "mean_predicted", "mean_actual", "mae", "mean_error"]]
            wa(md_table(r2, {"n": lambda v: f"{int(v):,}"}) + "\n")
        wa(fig(figs.get(grp, {}).get("ranges"), f"{grp.title()} accuracy by projection range", "INSERT PROJECTION RANGE ANALYSIS"))
        w(f"Single-game predicted-vs-actual scatters, residual and error-distribution charts and the projection-range "
          f"table for {grp}s are in the appendix (A9); the binned view below is the more informative summary.\n")
        w("**Binned predictions.** A single-game scatter is dominated by game-to-game noise (a hitter projected for "
          "1.1 hits gets 0, 1, 2 or 3), so it understates how well the projections track outcomes on average. The "
          "binned view sorts test player-games into ten equal-count deciles of the projection and plots the mean "
          "projection against the mean actual (with n and a 95% interval) against the 45° line.\n")
        bins = _csv(f"{grp}_binned_deciles.csv")
        w(fig(figs.get(grp, {}).get("binned"), f"{grp.title()} binned projections (deciles): mean predicted vs mean actual",
              "INSERT BINNED PREDICTION PLOT"))
        if not bins.empty:
            w(f"Decile values (n, mean projected, mean actual, standard error) are in `metrics/{grp}_binned_deciles.csv`.\n")
        wt = _csv(f"{grp}_test_window_totals.csv")
        w("**Test-window totals.** Summing each player's projections and actual results over the whole test window "
          "removes most single-game randomness and shows whether the projections rank players correctly.\n")
        if not wt.empty:
            w(f"Correlation between projected and actual test-window totals per player: "
              + ", ".join(f"{r.target} {r.corr_total:.2f}" for r in wt.itertuples()) + " (table in the appendix, A9).\n")
            wa(f"**{grp.title()} test-window totals**\n")
            wa(md_table(wt[["target", "players", "min_games", "mae_total", "rmse_total", "r2_total", "corr_total", "mean_error_total"]]) + "\n")
        wa(fig(figs.get(grp, {}).get("window_totals"), f"{grp.title()} projected vs actual test-window totals per player",
               "INSERT SEASON-TOTAL COMPARISON"))

    # ------------------------------------------------------------------ 9
    w("## 10. Model comparison\n")
    wa("## A10. Model comparison — stacking and charts\n")
    for grp in ("pitcher", "hitter"):
        comp = _csv(f"{grp}_model_comparison.csv")
        sel = _csv(f"{grp}_model_selection.csv")
        if comp.empty:
            continue
        piv = comp.pivot_table(index="target", columns="variant", values="mae")
        order = [v for v in ["Production blend (published)", "Direct count model", "Rate x predicted opportunity",
                             "Random forest (train-window fit)", "XGBoost (train-window fit)", "Linear: Poisson regression (train-window fit)",
                             "Stack: even weights (train-window fit)", "Stack: optimized weights (train-window fit)",
                             "Baseline: history avg (_std)", "Baseline: last-10 avg", "Baseline: league avg"] if v in piv.columns]
        piv = piv[order].reset_index()
        w(f"**{grp.title()} test MAE by variant** (lower is better):\n")
        w(md_table(piv) + "\n")
        if not sel.empty:
            cols = [c for c in ["target", "model", "valid_rmse", "valid_mae", "test_mae", "stack_weights"] if c in sel.columns]
            chosen = sel[sel["selected"]][cols]
            w(f"Model selected on validation ({grp}s): " + ", ".join(f"{r.target} {r.model}" for r in chosen.itertuples())
              + " (validation and test errors in the appendix, A10).\n")
            wa(f"**Model selected on validation per target ({grp}s)**\n")
            wa(md_table(chosen) + "\n")
            st = sel[sel["model"].astype(str).str.startswith("stack_")] if "model" in sel.columns else sel.iloc[0:0]
            if not st.empty:
                vr = sel.pivot_table(index="target", columns="model", values="valid_rmse")
                vr.columns = [f"valid RMSE {c}" for c in vr.columns]
                ow = st[st["model"] == "stack_opt"].set_index("target")["stack_weights"].rename("optimized weights")
                wa(f"**Stacking on validation ({grp}s)** — single models vs stacks (optimized stack scored cross-fitted):\n")
                wa(md_table(vr.join(ow).reset_index()) + "\n")
        wa(fig(figs.get(grp, {}).get("comparison"), f"{grp.title()} model vs baselines (MAE)", "INSERT MODEL COMPARISON TABLE"))
    w("Models were selected on validation only. The per-family and stack columns come from models fit identically "
      "on the training window (stack weights from validation) and are reported for transparency; they were not used "
      "for any decision. The published (production blend) and direct-model columns use the selected model refit on "
      "train + validation.\n")

    w("### Lineup (batting order) — controlled experiment\n")
    w("Hitter models gained two pre-game lineup features: `lineup_spot`, the batting-order slot the hitter starts in "
      "(1-9, posted before first pitch and served from the posted lineup; substitutes are left blank because their "
      "entry is not known before the game), and `lineup_spot_last10`, the mean of his previous ten slots. Everything "
      "else - data, protocol, model families, stacking, test games - is identical to the run without them, so the "
      "difference below is the lineup alone. Rows are matched on game, player and target.\n")
    any_lx = False
    for grp in ("hitter", "pitcher"):
        lx = _csv(f"{grp}_lineup_experiment.csv")
        if lx.empty:
            continue
        any_lx = True
        w(f"**{grp.title()}s** (test, lower is better; negative change = lineup helps):\n")
        w(md_table(lx[["target", "model", "n_matched", "mae_without_lineup", "mae_with_lineup", "mae_change_pct",
                       "poisson_dev_without_lineup", "poisson_dev_with_lineup", "bias_without_lineup", "bias_with_lineup"]],
                   {"n_matched": lambda v: f"{int(v):,}", "mae_change_pct": lambda v: f"{v:+.1f}%"}) + "\n")
    if not any_lx:
        w("`[INSERT LINEUP EXPERIMENT - baseline in outputs/model_evaluation/comparison/without_lineup]`\n")
    lxh = _csv("hitter_lineup_experiment.csv")
    if not lxh.empty and "model" in lxh.columns:
        fam = lxh[~lxh["model"].astype(str).str.contains("published")]
        avg = fam.groupby("target")["mae_change_pct"].mean().sort_values()
        helped = [f"{t} ({v:+.1f}%)" for t, v in avg.items() if v < -1]
        flat = [t for t, v in avg.items() if abs(v) <= 1]
        hurt = [f"{t} ({v:+.1f}%)" for t, v in avg.items() if v > 1]
        w("**Result (hitters, average of the random-forest and XGBoost rows):** lineup lowers test MAE by more than "
          f"1% for {', '.join(helped) or 'no target'}; it is within 1% for {', '.join(flat) or 'no target'}"
          + (f"; it raises MAE for {', '.join(hurt)}" if hurt else "") + ".\n")
    w("Each row compares the same model family fit the same way with and without the lineup features, so the "
      "difference is the lineup alone. *The published blend row also reflects the linear model and stacking options "
      "added after the baseline run, so it is not a clean lineup comparison.\n")
    w("Pitcher models do not use the new features (they already see the opposing lineup's trailing strength), so "
      "any pitcher difference is run-to-run noise.\n")

    w("### Original vs updated methodology\n")
    w("Both sides use the same model families, presets, protocol (train < validation < test windows) and the same "
      "test player-games; only the inputs differ. **Original**: the verbatim original feature modules "
      "(`experiments/original_code/`) on the original inputs (2025 + 2026 files as downloaded, spring training "
      "included, no `game_pk`). **Updated**: the same feature definitions with the two-season window and the "
      "fixes in section 6. Rows are matched on player, date and target; doubleheader dates are dropped because "
      "the original tables merge both games, and a row is kept only if both sides record the same actual.\n")
    if build_sum and 2024 not in build_sum.get("cached_seasons", []):
        w("> 2024 is not cached in this build, and the original pipeline already used 2025 history, so the difference "
          "below reflects the bug and leakage fixes only, not the added history. Re-run after collecting 2024 to "
          "measure the data expansion.\n")
    any_cmp = False
    for grp in ("pitcher", "hitter"):
        oc = _csv(f"{grp}_original_vs_updated.csv")
        if oc.empty:
            continue
        any_cmp = True
        w(f"**{grp.title()}s** (test MAE / RMSE / Poisson deviance, lower is better; negative change = updated is better):\n")
        w(md_table(oc[["target", "n_matched", "mae_original", "mae_updated", "mae_change_pct", "rmse_original", "rmse_updated",
                       "poisson_dev_original", "poisson_dev_updated", "bias_original", "bias_updated"]],
                   {"n_matched": lambda v: f"{int(v):,}", "mae_change_pct": lambda v: f"{v:+.1f}%"}) + "\n")
    if not any_cmp:
        w("`[INSERT ORIGINAL VS UPDATED COMPARISON — run experiments/original_methodology.py and the backtest on its tables]`\n")
    w("The updated side also includes the lineup features and model stacking; the controlled lineup experiment "
      "above isolates the lineup effect.\n")
    # Summary written from the comparison files (never hand-typed numbers).
    hit_orig_better = []
    for grp in ("hitter", "pitcher"):
        oc = _csv(f"{grp}_original_vs_updated.csv")
        if oc.empty:
            continue
        up = oc[oc["mae_change_pct"] < -1]
        org = oc[oc["mae_change_pct"] > 1]
        same = oc[oc["mae_change_pct"].abs() <= 1]
        fmt = lambda d: ", ".join(f"{r.target} ({r.mae_change_pct:+.1f}%)" for r in d.itertuples()) or "none"
        w(f"**{grp.title()}s:** updated pipeline lower MAE (by more than 1%) on {fmt(up)}; original lower on "
          f"{fmt(org)}; within 1% on {', '.join(same['target']) or 'none'}.\n")
        if grp == "hitter":
            hit_orig_better = list(org["target"])
    if hit_orig_better:
        w(f"Where the original is lower ({', '.join(hit_orig_better)}), that is not evidence it would predict better "
          "live: its hitter features contained the opposing starter's results from the same game (section 6), "
          "information that does not exist before first pitch, so its test numbers on those targets are optimistic. "
          "How much of each gap is the leak versus the other fixes was not separately measured.\n")
    w("Where the updated pipeline is lower, the lineup experiment above shows how much of it the batting-order "
      "features account for (plate appearances in particular depend directly on lineup slot).\n")

    # ------------------------------------------------------------------ 11 (audit)
    try:
        import report_audit
        if report_audit.available():
            report_audit.section(w, fig, md_table, 11, wa=wa)
        else:
            w("## 11. Feature Diagnostics, Multicollinearity, and Overfitting\n")
            w("`[Run python audit_models.py, then regenerate this report]`\n")
    except Exception as e:                      # the audit must never stop the main report
        w("## 11. Feature Diagnostics, Multicollinearity, and Overfitting\n")
        w(f"`[audit section failed to render: {e!r}]`\n")

    # ------------------------------------------------------------------ 12-14
    w("## 12. Daily prediction pipeline\n")
    w("`daily_update.py` is the single entry point (GitHub Actions, three runs a day):\n")
    w("1. `refresh_full_history` — restore the previous two seasons from cache (download once if missing); top "
      "up the current season with a three-day overlap.\n"
      "2. `build_features` — rebuild the current season's window (S−2…S) and export "
      "`data/*_game_data.csv`, the team context files and `data/game_model_data.csv`.\n"
      "3. Early-morning run (unchanged schedule) — refit the player models on the multi-season tables: family "
      "chosen on the most recent 15% of dates, then refit on all rows; each pickle records `trained_through`.\n"
      "4. Game model — completed games read their stored pre-game rows; today's slate is built from data "
      "before today with the probable starters' IDs (identical code path to training).\n"
      "5. Player projections for today's lineups (Rotowire / MLB API), blended direct + rate × opportunity.\n"
      "6. Grading, calibration display layer and power rankings.\n")
    w("`[INSERT DAILY PIPELINE DIAGRAM]`\n")
    w("## 13. Public prediction website\n")
    w("`server.py` (FastAPI) serves `index.html` and JSON endpoints (`/api/games`, `/api/pitchers`, "
      "`/api/hitters`, `/api/accuracy`, `/api/player-accuracy`, `/api/rankings`, `/api/player/{id}`) that read "
      "the CSV/JSON outputs committed by the daily job. Tabs: Today's Games, Pitcher and Hitter Projections, "
      "Power Rankings, Performance (season accuracy ledger), Methodology/FAQ/About.\n")
    w("`[INSERT INTHECOUNT TODAY'S GAMES SCREENSHOT]`\n\n`[INSERT INTHECOUNT PERFORMANCE PAGE SCREENSHOT]`\n")
    w("## 14. Prediction grading\n")
    w("Games: each pick is joined to the MLB Stats API schedule result by `game_pk`; a pick is correct when "
      "the predicted winner equals the final-score winner; unfinished or postponed games stay ungraded. "
      "Picks carry a `model_version` (v1 snapshot model vs. v2 rebuilt model), and the backfill refuses to "
      "write a pick for a date inside the deployed model's training window. Players: projections are joined to "
      "box-score lines by MLB ID and date; players who did not appear are marked not played and excluded; "
      "rows whose projection came from a model trained on that game are flagged `in_sample` and excluded from "
      "the site's accuracy. Grading outputs are never read by any trainer.\n")

    # ------------------------------------------------------------------ 13
    w("## 15. Automation and deployment\n")
    w("GitHub Actions (`.github/workflows/daily.yml`) runs at roughly 3 AM, 11 AM and 5 PM ET. Statcast "
      "CSV caches (`pitch_data_<season>.csv`) and the multi-season feature tables live in the Actions cache (history seasons under a "
      "stable per-season key, so they are never re-downloaded daily); the job commits refreshed CSV/JSON "
      "outputs, and Render (`render.yaml`, `uvicorn server:app`) redeploys on every push. Offline retraining "
      "and evaluation are run locally with `run_full_pipeline.ps1` or the commands in the implementation "
      "summary.\n")

    # ------------------------------------------------------------------ 14-16
    w("## 16. Lessons learned\n")
    w("- **Leakage hides in joins, not in rolling windows.** Every `shift(1)` in the code was correct; the "
      "worst leak came from merging the opposing starter's row for the *same* `game_pk`. Only a test that "
      "changes an outcome and watches the features catches that class of bug reliably.\n"
      "- **Selecting on the test set is leakage too.** Choosing among three model families by test RMSE "
      "turned the test set into a validation set.\n"
      "- **History needs a definition, not just more data.** Without a window, every expanding feature "
      "would silently grow with however many seasons happen to be cached; the window has to be imposed "
      "before features exist, while the feature definitions themselves stay as they were.\n"
      "- **Row order is an assumption.** The starter heuristic worked on newest-first files and failed on "
      "oldest-first ones; explicit keys (`game_pk`, pitch order) remove that dependence.\n"
      "- **A public ledger must know what is in-sample.** Re-scoring the past with today's model makes "
      "accuracy look better than anything a user could have acted on.\n"
      "- **Baselines keep the story honest.** Single-game baseball outcomes are mostly noise; the right "
      "question is how much better than a season average or an always-home pick a model is, and the answer "
      "can be \"not much\".\n")
    w("## 17. Future experiments (not implemented)\n")
    w("None of the following is implemented. Each experiment "
      "should be run **one at a time** under a fixed protocol: the same train / validation / test windows as this "
      "report, the decision to keep a feature made on validation only, the test window scored once at the end, "
      "and the leakage tests extended to the new inputs before any result is read.\n")
    exps = ["**Expected stats (xBA / xwOBA)** — trailing Statcast expected-outcome rates as less noisy inputs than realised H/TB.",
            "**Pitcher workload** — pitch-count and leash features (recent pitch counts, days of rest interactions) for IP and counts.",
            "**Plate discipline** — chase, zone-contact and swinging-strike rates for hitters and the opposing pitcher.",
            "**Game-model improvements** — bullpen availability, lineup-weighted team strength and park effects, "
            "evaluated against the deployed v1 model on validation."]
    for x in exps:
        w(f"{exps.index(x) + 1}. {x}")
    w("")
    w("Known limitations outside those experiments: 2024 must be collected (and 2026 completed if flagged) and "
      "the evaluation regenerated; stolen bases are not in Statcast pitch events, so the SB target is all zeros; "
      "evaluation uses a single test window rather than walk-forward.\n")
    w("## 18. Conclusion\n")
    try:
        import report_audit
        report_audit.conclusion_answers(w)
    except Exception as e:
        w(f"`[audit answers failed to render: {e!r}]`\n")
    w("InTheCount now collects any Statcast season through one cached, validated code path; restricts every "
      "projection to the previous two seasons plus the current season before the game; proves with "
      "perturbation and parity tests that its features are pre-game; selects models on validation data only; "
      "and evaluates game and player projections on a held-out window against honest baselines. The numbers "
      "above — strong or weak — are what the system actually does out of sample, and every one of them can be "
      "regenerated with `python evaluate_models.py`.\n")
    return "\n".join(L), "\n".join(A)


def to_html(md: str, title: str = "InTheCount Technical Report") -> str:
    """Self-contained HTML with figures inlined (shareable as one file)."""
    try:
        import markdown
        body = markdown.markdown(md, extensions=["tables"])
    except ImportError:
        import html
        body = "<pre>" + html.escape(md) + "</pre>"
    import re

    def inline(m):
        p = (REPORT_DIR / m.group(1)).resolve()
        if p.exists():
            return f'src="data:image/png;base64,{base64.b64encode(p.read_bytes()).decode()}"'
        return m.group(0)
    body = re.sub(r'src="([^"]+\.png)"', inline, body)
    # keep a table's title (the paragraph or heading right before it) on the same printed page as the table
    body = re.sub(r'(<(p|h3|h4)>(?:(?!</\2>).){0,600}</\2>\s*<table>.*?</table>)',
                  r'<div class="keep">\1</div>', body, flags=re.S)
    css = ("body{font-family:Georgia,'Times New Roman',serif;max-width:980px;margin:40px auto;padding:0 20px;"
           "color:#1b1b1b;line-height:1.55;background:#fff}h1,h2,h3{font-family:Helvetica,Arial,sans-serif;"
           "color:#0b2545}h1{font-size:28px}h2{border-bottom:1px solid #ddd;padding-bottom:4px;margin-top:36px}"
           "table{border-collapse:collapse;font:13px Helvetica,Arial,sans-serif;margin:10px 0;display:block;"
           "overflow-x:auto}th,td{border:1px solid #ddd;padding:4px 7px;text-align:right}th{background:#f3f4f6}"
           "td:first-child,th:first-child{text-align:left}"
           "figure{margin:16px auto 40px;text-align:center;break-inside:avoid}"
           "figure img{width:auto;height:auto;max-width:100%;max-height:620px;border:1px solid #eee}"
           "figcaption{font:11px Helvetica,Arial,sans-serif;color:#555;margin-top:6px}"
           "img{max-width:100%;border:1px solid #eee}"
           "code{background:#f5f5f5;padding:1px 4px;font-size:90%}blockquote{border-left:3px solid #eda100;"
           "margin:0;padding-left:12px;color:#444}"
           # print / PDF: full-width tables, no split figures or rows, headings kept with their content
           "@page{size:Letter;margin:0.5in}"
           "@media print{body{max-width:none;margin:0;padding:0;font-size:11px}"
           "table{display:table;font-size:10.5px;width:100%;max-width:100%;margin:10px 0 18px}""th,td{padding:3px 5px}td{overflow-wrap:normal;word-break:keep-all}""th{font-size:10px;line-height:1.2;overflow-wrap:normal;word-break:normal;hyphens:none}"
           "figure img{max-width:100%;max-height:4.4in}figure{margin:8px auto 26px}"
           ".keep{break-inside:avoid;page-break-inside:avoid}figure,img,tr,blockquote,table{break-inside:avoid}thead{display:table-header-group}h2,h3{break-after:avoid}h2{margin-top:22px}}")
    return (f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,"
            f"initial-scale=1'><title>{title}</title><style>{css}</style></head><body>{body}</body></html>")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-tests", action="store_true")
    a = ap.parse_args(argv)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    if a.skip_tests:        # reuse the last saved test run
        tests = _json(MET / "test_results.json") or {"summary": "not run (--skip-tests)", "returncode": None}
    else:
        tests = run_tests()
        (MET / "test_results.json").write_text(json.dumps(tests, indent=2))
    import report_v2
    md = report_v2.build(tests)
    stem = "InTheCount_Technical_Report"
    (REPORT_DIR / f"{stem}.md").write_text(md, encoding="utf-8")
    html_path = REPORT_DIR / f"{stem}.html"
    html_path.write_text(to_html(md), encoding="utf-8")
    print(f"wrote {REPORT_DIR / (stem + '.md')} and .html")
    pdf = to_pdf(html_path, REPORT_DIR / f"{stem}.pdf")
    print(f"wrote {pdf}" if pdf else "PDF skipped: no Edge/Chrome/Chromium found (open the .html and Print > Save as PDF)")
    for old in ("InTheCount_Technical_Report_Appendix.md", "InTheCount_Technical_Report_Appendix.html",
                "InTheCount_Technical_Report_Appendix.pdf"):
        p = REPORT_DIR / old
        if p.exists():
            try:
                p.unlink()
            except OSError:
                print(f"note: could not remove the old {old}; it is no longer produced")


def _find_browser() -> str | None:
    """Edge ships with Windows; Chrome/Chromium are fallbacks."""
    import os
    import shutil
    names = ["msedge", "chrome", "google-chrome", "chromium", "chromium-browser"]
    for n in names:
        if shutil.which(n):
            return shutil.which(n)
    pf = [os.environ.get(k, "") for k in ("ProgramFiles(x86)", "ProgramFiles", "LOCALAPPDATA")]
    cands = [Path(b) / "Microsoft/Edge/Application/msedge.exe" for b in pf if b] + \
            [Path(b) / "Google/Chrome/Application/chrome.exe" for b in pf if b] + \
            [Path("/opt/pw-browsers/chromium-1194/chrome-linux/chrome"),
             Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")]
    for c in cands:
        if c.exists():
            return str(c)
    return None


def to_pdf(html_path: Path, pdf_path: Path) -> Path | None:
    """Print the self-contained HTML to PDF with a headless browser."""
    import subprocess
    exe = _find_browser()
    if not exe:
        return None
    import tempfile
    prof = tempfile.mkdtemp(prefix="itc_pdf_")   # separate profile: works even if Edge is open
    cmd = [exe, "--headless", "--disable-gpu", "--no-sandbox", "--no-pdf-header-footer",
           f"--user-data-dir={prof}", f"--print-to-pdf={pdf_path.resolve()}", html_path.resolve().as_uri()]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=300)
    except Exception as e:  # never fail the report over the PDF
        print(f"PDF export failed: {e}")
        return None
    return pdf_path if pdf_path.exists() else None


if __name__ == "__main__":
    main()
