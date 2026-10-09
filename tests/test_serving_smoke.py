"""
Offline smoke tests of the serving paths (no network): the player projection
scorer and the v2 game runner are fed a past slate reconstructed from the
cached Statcast data instead of the MLB Stats API.
"""
import numpy as np
import pandas as pd
import pytest

import build_game_features as bgf
import daily_mlb_model_runner as runner
import refresh_full_history as sc
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _slate(day):
    p = sc.read_season(int(day[:4]), columns=list(dict.fromkeys(bgf.NEEDED + ["stand"])))
    if p.empty:
        pytest.skip("no cached Statcast")
    return p[(pd.to_datetime(p["game_date"]) == day) & (p["game_type"] == "R")]


def test_player_projection_scoring_offline(monkeypatch):
    import backfill_player_predictions as bpp
    data = ROOT / "data" / "pitcher_game_data.csv"
    if not data.exists() or not (ROOT / "models" / "pitcher_K.pkl").exists():
        pytest.skip("feature CSVs / models not present")
    pg = pd.read_csv(data, usecols=["game_date"])
    season = int(pd.to_datetime(pg["game_date"]).max().year)
    gt = sc.read_season(season, columns=["game_date", "game_type"])
    # a recent REGULAR-season date (the cache can end in the postseason)
    reg = gt.loc[gt["game_type"] == "R", "game_date"]
    day = sorted(pd.to_datetime(reg).dt.strftime("%Y-%m-%d").unique())[-3]
    p = _slate(day)
    p = bgf._prep(p)
    lines = bgf.pitching_lines(p)
    games = p.groupby("game_pk")[["home_team", "away_team"]].first()
    pit, hit = [], []
    for _, r in lines[lines["is_starter"] == 1].iterrows():
        g = games.loc[r["game_pk"]]
        opp = g["away_team"] if r["team"] == g["home_team"] else g["home_team"]
        pit.append({"player_type": "pitcher", "team": r["team"], "opponent": opp, "player_name": r["name"],
                    "mlb_id": int(r["pitcher"]), "is_actual_starter": True})
    bat = p.dropna(subset=["batter"]).sort_values("at_bat_number").drop_duplicates(["game_pk", "batter"])
    for gpk, g in bat.groupby("game_pk"):
        for team, gg in g.groupby("bat_team"):
            opp = games.loc[gpk, "away_team"] if team == games.loc[gpk, "home_team"] else games.loc[gpk, "home_team"]
            for spot, (_, h) in enumerate(gg.head(9).iterrows(), start=1):
                hit.append({"player_type": "hitter", "team": team, "opponent": opp, "player_name": str(int(h["batter"])),
                            "mlb_id": int(h["batter"]), "lineup_spot": spot, "pos": "DH",
                            "lineup_status": "Confirmed", "norm_name": "", "roster_name": ""})
    monkeypatch.setattr(bpp, "_build_past_inputs", lambda d, sleep_seconds=0.2: (pd.DataFrame(pit), pd.DataFrame(hit)))
    monkeypatch.chdir(ROOT)
    out = bpp._project_past_date(day)
    assert out is not None and len(out)
    pitchers = out[out["player_type"] == "pitcher"]
    hitters = out[out["player_type"] == "hitter"]
    assert len(pitchers) >= 0.8 * len(pit), f"only {len(pitchers)}/{len(pit)} pitchers projected"
    assert len(hitters) >= 0.8 * len(hit), f"only {len(hitters)}/{len(hit)} hitters projected"
    for c in ("proj_strikeouts", "proj_ip"):
        v = pd.to_numeric(pitchers[c], errors="coerce")
        assert v.notna().all() and (v >= 0).all()
    assert "model_trained_through" in out.columns


def test_v2_game_runner_offline(monkeypatch, tmp_path):
    cand = ROOT / "betting_model_candidate.pkl"
    if not cand.exists() or not (ROOT / "data" / "game_model_data.csv").exists():
        pytest.skip("candidate model / game table not present")
    t = pd.read_csv(ROOT / "data" / "game_model_data.csv")
    day = t["game_date"].max()
    rows = t[t["game_date"] == day]
    fake = rows[["game_pk", "game_date", "home_team", "away_team", "home_starter", "away_starter",
                 "home_starter_id", "away_starter_id"]].copy()
    fake["game_pk"] = fake["game_pk"] + 9_000_000          # unknown games -> forces the slate builder
    fake["game_date"] = pd.to_datetime(day) + pd.Timedelta(days=1)
    for c in ("home_score", "away_score", "status", "home_win", "actual_winner", "commence_time"):
        fake[c] = np.nan
    monkeypatch.setattr(runner, "get_games_for_date", lambda d: fake.copy())
    monkeypatch.chdir(ROOT)
    preds = runner.run(date=str(fake["game_date"].iloc[0].date()), model_path=str(cand),
                       save_today_csv=False, save_pick_log=False)
    assert len(preds) == len(fake)
    assert preds["home_win_prob"].between(0, 1).all()
    assert preds["model_version"].str.startswith("v2").all()
    feats = runner.load_bundle(cand)["features"]
    assert preds[[f for f in feats if f.startswith("diff_SP")]].notna().mean().mean() > 0.7
