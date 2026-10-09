# InTheCount — MLB Research & Analytics

## What This Is

A free, public dashboard for a machine-learning MLB projection system. No accounts, no paywall, no odds — just daily projections and a season-long accuracy ledger anyone can audit.

- **Today's Games** — Model win-probability for every scheduled game, projected winner, starting pitchers; tap a card to drill into the full matchup (pitching lines + lineup projections)
- **Pitcher Projections** — Sortable table of projected IP, K, BB, H, ER for each starter
- **Hitter Projections** — Lineup cards per matchup or full sortable table (PA, H, HR, K, BB, R)
- **Power Rankings** — Season-long player production ranking, plus a 25-and-under view
- **Performance** — Cumulative game and player projection accuracy, graded against actual results
- **Methodology, FAQ, About** — How the pipeline works and who's behind it

The site is served by `server.py` (FastAPI), reading CSV/JSON output produced by the daily pipeline in `daily_update.py`. See [DEPLOY.md](DEPLOY.md) for hosting instructions.

The technical report (data, history window, leakage controls, out-of-sample evaluation) is generated into
`outputs/model_evaluation/report/InTheCount_Technical_Report.md` (and a self-contained `.html`).

---

## Data flow

```
pybaseball.statcast()  ──>  refresh_full_history.py  ──>  pitch_data_<season>.csv
                             (any season, cached,            + data_validation/statcast_validation.json
                              3-day overlap, validated)
        │
        ▼  history_window.py: prediction season S may see S-2, S-1 and S-before-the-game only
build_features.py ──> hitterspitchers_data.build_windowed_tables()   data/features/{pitcher,hitter}_games.csv
                  └─> build_game_features.build_game_table()         data/game_model_data.csv
                  └─> current-season exports for serving              data/*_game_data.csv, team context CSVs
        │
        ├─> hitterspitchers_train.py   player models (RF / XGB / MLP, chosen on validation)  models/*.pkl
        ├─> train_game_model.py        game model (XGB + isotonic etc., chosen on validation) betting_model.pkl
        │
        ├─> daily_mlb_model_runner.py + hitterspitchers_today.py   today's predictions  outputs/
        ├─> grade_player_predictions.py + grade_saved_picks         2026_*_accuracy.csv
        └─> server.py / index.html                                  the public site

Evaluation (never touches production models):
run_backtest.py + train_game_model.py ──> outputs/model_evaluation/data/*_test_predictions.csv
evaluate_models.py (quality_checks.py) ──> metrics/, figures/, report/ (generate_report.py)
tests/ (pytest)                         ──> perturbation leakage tests, two-season rule, train/serve parity
```

## Commands

```bash
pip install -r requirements.txt

# data
python refresh_full_history.py --seasons 2024 2026 --skip-features   # one-time 2024 pull (run on your own PC)
python refresh_full_history.py                            # collect 2024-2026 (history cached once) + validate + features
python refresh_full_history.py --seasons 2024 --skip-features   # just collect one season
python refresh_full_history.py --validate                 # validation report only
python build_features.py                                  # windowed feature tables, all cached seasons

# production models
python hitterspitchers_train.py                           # player models (validation-selected, refit on all)
python train_game_model.py --production --promote         # game model, backs up betting_model.pkl first

# out-of-sample evaluation (train < 2026-03-01 <= validation < 2026-07-01 <= test)
python train_game_model.py                                # game test predictions
python run_backtest.py                                    # player test predictions
python evaluate_models.py                                 # checks + metrics + figures + report
python generate_report.py                                 # report only (re-runs pytest)
python -m pytest -q tests

# daily
python daily_update.py
```

## Running the site locally

```bash
python -m venv .venv
.venv\Scripts\activate         # source .venv/bin/activate on macOS/Linux
pip install -r requirements-server.txt
uvicorn server:app --reload --port 8000
```

Open **http://localhost:8000**. If every tab is empty, run `python daily_update.py` once first.

---

## Project layout

```
refresh_full_history.py        season-configurable pybaseball collector (pitch_data_<season>.csv) + validation
history_window.py              the two-season player-history rule
build_features.py              builds every modelling table from the cached seasons
hitterspitchers_data.py        per-game pitcher / hitter features (windowed build)
enrich_*.py                    team, lineup and true-talent features (run inside the windowed build)
build_game_features.py         one-row-per-game features for the win-probability model
leakage_guard.py               static check that every model feature is pre-game
hitterspitchers_train.py       player model training (chronological train / validation / test)
train_game_model.py            game model training / evaluation / promotion
run_backtest.py                out-of-sample player backtest -> saved test predictions
quality_checks.py              integrity checks run before any metric
evaluate_models.py             metrics, figures, machine-readable outputs
generate_report.py             technical report from the evaluation outputs
daily_update.py                full daily pipeline (single entry point)
daily_mlb_model_runner.py      game predictions + pick log + grading
hitterspitchers_today.py       player projection runner
grade_player_predictions.py    player grading against MLB box scores
server.py / index.html         public site
data/                          feature CSVs served by the site (+ features/ cache, gitignored)
models/                        trained model .pkl files
outputs/                       daily predictions, snapshots, rankings
outputs/model_evaluation/      test predictions, metrics, figures, report, data validation
tests/                         pytest suite
```
