# InTheCount audit — implementation summary (2026-10-06)

Numbers below come from `outputs/model_evaluation/metrics/` (backtest: train < 2026-03-01 ≤ validation < 2026-07-01 ≤ test).
They were produced **without 2024 data** (see "Remaining limitations"). Regenerate after collecting 2024.

## Scope (per the narrowed brief: original feature calculations + two more seasons of history)
| Category | Changes |
|---|---|
| Data expansion | Season-configurable `refresh_full_history.py` (2024/2025/2026 CSV caches, current-season top-up, 3-day overlap, timeout, resume); two-season window applied before feature building; focused per-season validation |
| Feature-engineering changes | None to player features (rolling 7/10/14/21/30, `_std` expanding mean with no season reset, splits, true-talent, matchup, team: original code). Game features rebuilt with the same 18 names because the original builder is missing and `2025_model_data.csv` duplicates every game with home/away swapped |
| Bug / leakage fixes | opp-starter same-game leak; lineup same-game inputs; true-talent league prior from earlier dates only; `game_pk`/doubleheaders; starter identification (row-order bug); spring training excluded; family selection on validation, not test; K-fold CV removed; TimeSeriesSplit calibration; in-sample rows excluded from the public ledger |
| Future experiments (not implemented) | batting order, xBA/xwOBA, pitcher workload, plate discipline, game-model improvements — one at a time, same train/validation/test, test untouched |

Out-of-scope changes kept, and why: selected family refit on train+validation before the single test scoring (otherwise the deployed model would ignore the newest two months); new evaluation/QC/test scripts (required to measure without touching production).
No reliability curve is produced. The evaluation adds binned (decile) and test-window-total plots alongside the existing ones, and an Original-vs-Updated comparison (`experiments/original_methodology.py`, `metrics/*_original_vs_updated.csv`).

## Files inspected
All Python modules at the project root, `.github/workflows/*.yml`, `render.yaml`, `Procfile`, `DEPLOY.md`, `README.md`,
`run_full_pipeline.ps1`, `requirements*.txt`, `data/*.csv`, `models/*.pkl` (+ metrics/importance), `2025_model_data.csv`,
`2026_picks_accuracy.csv`, `2026_player_accuracy.csv`, `betting_model.pkl`, the Aug 28 training log, `pitch_data_2025/2026.csv`.

## Files created
| File | Purpose |
|---|---|
| `experiments/original_methodology.py` + `experiments/original_code/` | Rebuilds features with the verbatim original modules for the Original-vs-Updated comparison |
| `model_stacking.py` | Even-weight and optimized-weight (simplex, validation-fit, cross-fitted for selection) stacks; `StackedRegressor` / `StackedClassifier` wrappers for the pickles |
| `tests/test_stacking.py` | Weights on the simplex, cross-fitting uses only the other half, wrappers predict correctly |
| `history_window.py` | Two-season rule (S-2, S-1, S-before-game), competitive-game filter, window assertion |
| `build_features.py` | Builds every modelling table from the cached seasons, season by season under the window |
| `build_game_features.py` | Reproducible one-row-per-game feature table (same 18 feature names) + slate builder for serving |
| `leakage_guard.py` | Static check that every model feature is pre-game |
| `run_backtest.py` | Out-of-sample player backtest → saved test predictions (never overwrites production models) |
| `quality_checks.py` | Integrity checks; critical failure stops the evaluation |
| `evaluate_models.py` | Metrics, figures, machine-readable outputs from saved predictions (no retraining) |
| `generate_report.py` | Technical report (Markdown + self-contained HTML) built only from output files; runs pytest |
| `tests/test_leakage.py` | Perturbation tests (player + game), older-season test, train/serve parity, primitives, guard, date split |
| `tests/test_quality_checks.py` | Checks fail loudly on bad predictions / history violations |
| `tests/test_serving_smoke.py` | Offline serving tests of the player scorer and the v2 game runner |
| `docs/daily.yml.proposed` | Updated GitHub Actions workflow (the remote tools could not write `.github/`) |
| `outputs/model_evaluation/**` | Test predictions, metrics, figures, figure data, validation, report |
| `backups/pre_audit_20261006/` | Copy of the previous models, data CSVs, betting_model.pkl and accuracy logs |

## Files modified
| File | Change |
|---|---|
| `hitterspitchers_data.py` | Rolling/`_std`/split code is the original (only `game_pk` added as a sort tie-breaker). Loads `game_pk`/order columns (doubleheaders), drops spring training, starter = team's first pitcher (row-order bug), opp-starter features use the PREVIOUS start (leak fix), windowed build entry points |
| `enrich_lineup_features.py` | Lineup features from batters' trailing values (were same-game); in-memory `enrich_frame` |
| `enrich_team_features.py` | Definitions unchanged; in-memory `enrich_frame` so it runs inside the windowed build |
| `enrich_truetalent.py` | League prior from earlier dates only; per-`game_pk` lineup matching; `enrich_frames` |
| `hitterspitchers_train.py` | Model family chosen on a chronological validation window (was test RMSE); refit on train+validation; K-fold CV removed; leakage guard; `trained_through` in pickles; test predictions returned |
| `train_game_model.py` | New data source, 18 production features fixed, train/validation/test by date, TimeSeriesSplit calibration, baselines, saved test predictions, promotion gate vs v1 live log |
| `daily_mlb_model_runner.py` | v2 feature path (stored pre-game rows / slate builder), starter IDs, spring training skipped, `model_version` on picks, refuses in-sample backfill |
| `daily_update.py` | `refresh_full_history` + windowed build replace the 2026-only refresh and file-level enrich passes; retrain schedule unchanged, now on multi-season tables |
| `refresh_full_history.py` | The only scraper (no separate collector module). Same pybaseball approach with the season as a parameter (2024/2025/2026 → `pitch_data_<season>.csv`), 3-day overlap for the current season, front-gap fill, request timeout + sequential retry, per-season validation JSON, then the windowed feature build |
| `refresh_2026_data.py` | Daily wrapper: ensures the two previous seasons are cached, tops up the current season, rebuilds its window |
| `hitterspitchers_today.py`, `backfill_player_predictions.py` | Snapshots stamped with `model_trained_through` |
| `grade_player_predictions.py`, `server.py` | `in_sample` flag; site accuracy excludes in-sample rows |
| `requirements.txt`, `README.md`, `run_full_pipeline.ps1`, `.gitignore` | New deps, docs, pipeline steps, cache paths ignored |
| `models/*.pkl` (pitcher/hitter) | Retrained with the fixed features/protocol (trained through 2026-08-27) |
| `data/*_game_data.csv`, team context CSVs, `data/game_model_data.csv` | Rebuilt under the window |

`betting_model.pkl` (game model) was **not** replaced: the rebuilt model did not beat the deployed v1 model on validation.

## Commands
```
python refresh_full_history.py --seasons 2024 2026 --skip-features   # one-time: pull 2024 + fill 2026 (must run on your PC; Savant is unreachable from the cloud)
python refresh_full_history.py                            # collect / update 2024-2026 (+ validation + features)
python build_features.py                                   # feature engineering (windowed)
python hitterspitchers_train.py                            # production player models
python train_game_model.py --production --promote          # production game model (gated)
python daily_update.py                                     # daily predictions + grading
python train_game_model.py                                 # game test predictions
python run_backtest.py                                     # player test predictions
python evaluate_models.py                                  # checks, metrics, figures, report
python generate_report.py                                  # report only
python -m pytest -q tests
```

## Cleanup (2026-10-06)
Moved to `_to_delete/compact_20261006/` (unreferenced by any script, the site or the daily workflow): `__pycache__/`,
`.streamlit/`, `pitch_data_combined.csv` (696 MB), `training_run_*.log`, `test_file.svg`, `test_logo.svg`,
root `today_bets_to_make.csv` / `today_predictions_with_ev.csv`, the retired NRFI / props / market-consensus CSVs and
`models/nrfi_*`, `outputs/` betting/props/NRFI/fantasy leftovers, stale `data/*_importance.csv` / `data/*_metrics.csv`,
`data/game_data_2025.csv`, `data/nrfi_game_data.csv`, `collect_statcast_history.bat`. `streamlit` removed from
`requirements.txt` (nothing imports it). Unused logos and `Video Output/` moved to `branding/` (kept, not runtime).
Still to remove by hand (protected path): `.github/workflows/retrain_nrfi.yml` and `nrfi_rebackfill.yml` run
`nrfi_train.py`, which no longer exists.

## Model stacking (added 2026-10-07, on request)
Player models: `stack_even` (RF/XGB/MLP averaged 1/3 each) and `stack_opt` (non-negative weights summing to 1,
minimising validation RMSE) compete with the single families on validation RMSE; the optimized stack's selection
score is cross-fitted (weights from one chronological half of validation, scored on the other). Game model: the
same two stacks over the four candidates on validation log loss. A winning stack is refit member-by-member on
train + validation. Test predictions for both stacks are saved (`pred_stack_even`, `pred_stack_opt`,
`p_stack_even`, `p_stack_opt`) and appear in the report's model comparison. The game promotion gate vs v1 is unchanged.

## Lineup, next-day slate, tables (2026-10-07)
- **Lineup features (hitters):** `lineup_spot` = batting-order slot the hitter starts in (first nine distinct batters of
  a team-game by first plate appearance; substitutes blank, since their entry is not known pre-game; pipelines impute
  the median with no missing-indicator) and `lineup_spot_last10` (trailing mean, shift(1)). Served from the posted
  lineup. Whitelisted in `leakage_guard.py`; perturbation/older-season tests pass. Evaluated as a controlled experiment
  against the previous run (`outputs/model_evaluation/comparison/without_lineup/`, `metrics/*_lineup_experiment.csv`).
- **Site one day behind:** GitHub starts scheduled runs 4-6 h late, so the morning slate landed ~10 AM ET. Fixes:
  evening runs (>= 3 PM ET) also write tomorrow's games and probable-starter projections to
  `outputs/next_day_predictions.csv` / `outputs/next_day_projections.csv` (no pick-log entry); `server.py` serves the
  file whose `game_date` is today (ET); extra cron ticks + `concurrency` in `docs/daily.yml.proposed`; lineups come from
  the MLB Stats API by date (Rotowire only fills gaps for today's slate after 6 AM ET - its page carries no date);
  default slate date and "yesterday" use ET instead of the runner's UTC clock; the player-model retrain only runs when
  the data has games the models have not seen.
- **Report:** tables never split across pages; charts two per page.
- **GitHub:** `sync_to_github.ps1` commits local work, merges GitHub (daily data + index.html edits kept), runs checks,
  pushes. Run with `-DryRun` first.

## Linear models (2026-10-07)
Player models gain a fourth family, `lin` = Poisson regression (`PoissonRegressor`, L2 alpha 0.05, standardised
features): every target is a non-negative count/rate, so a log-link GLM is the appropriate linear model (OLS could
predict negatives). The game model gains a `logistic` candidate (L2, C = 0.1, standardised). Both compete on
validation like the other families and are members of the even- and optimized-weight stacks; test predictions are
saved as `pred_lin` / `p_logistic`.

## Game-model selection metric (2026-10-07)
Selection, optimized-stack weights and the promotion gate use validation **log loss**. ROC AUC was tried as the
criterion and reverted: the site shows probabilities and AUC ignores calibration; log loss is a proper scoring rule;
AUC differences near 0.53 on ~1,000 games are mostly noise. The report (section 7) explains this and states which
candidate AUC would have chosen in each run. `model_stacking.fit_weights(..., "auc")` remains available.

## Neural network removed (2026-10-08)
The player MLP (`nn`) is gone from `hitterspitchers_train.py` (families: rf, xgb, lin + the two stacks), from the
backtest/evaluation columns and from the report's comparison tables. Reason, stated in the report: weakest family on
validation for every target checked, small stack weight, large share of training time.
