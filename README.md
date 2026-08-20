# TheBullpenBet — MLB Research & Analytics

## What This Is

A free, public dashboard for a machine-learning MLB projection system. No accounts, no paywall, no odds — just daily projections and a season-long accuracy ledger anyone can audit.

- **Today's Games** — Model win-probability for every scheduled game, projected winner, starting pitchers; tap a card to drill into the full matchup (pitching lines + lineup projections)
- **Pitcher Projections** — Sortable table of projected IP, K, BB, H, ER for each starter
- **Hitter Projections** — Lineup cards per matchup or full sortable table (PA, H, HR, K, BB, R)
- **Power Rankings** — Season-long player production ranking, plus a 25-and-under view
- **Performance** — Cumulative game and player projection accuracy, graded against actual results
- **Methodology, FAQ, About** — How the pipeline works and who's behind it

The site is served by `server.py` (FastAPI), reading CSV/JSON output produced by the daily pipeline in `daily_update.py`. See [DEPLOY.md](DEPLOY.md) for hosting instructions.

---

## Running locally

```bash
python -m venv .venv
source .venv/bin/activate      # or .venv\Scripts\activate on Windows
pip install -r requirements-server.txt
uvicorn server:app --reload --port 8000
```

Open **http://localhost:8000**.

If every tab is empty, run the pipeline once first:

```bash
pip install -r requirements.txt
python daily_update.py
```

---

## Daily pipeline

`daily_update.py` is the single entry point. It refreshes Statcast data, runs today's game and player projection models, grades yesterday's picks against final results, recalibrates projections, and rebuilds the season player power rankings. It's meant to run on a schedule (see `.github/workflows/daily.yml`) so the live site always reflects the current slate.

```bash
python daily_update.py
```

---

## Project layout

```
server.py                     ← FastAPI backend serving index.html + /api/*
index.html                    ← Single-file front-end
daily_update.py                ← Full daily pipeline (single entry point)
daily_mlb_model_runner.py     ← Game win-probability model runner
hitterspitchers_today.py       ← Player projection runner
hitterspitchers_data.py        ← Data helpers
hitterspitchers_train.py       ← Player model training
player_rankings.py             ← Season power rankings builder
train_game_model.py            ← Game-outcome model training
data/                          ← Per-game feature CSVs (Statcast-derived)
models/                        ← Trained model .pkl files
outputs/                       ← Daily-refreshed CSV/JSON the site reads
```
