"""
build_game_features.py
======================
Reproducible, leakage-audited game-level feature table for the moneyline /
win-probability model, built directly from the cached Statcast pitches.

Why this exists
---------------
The production game model was trained on `2025_model_data.csv`, whose builder
script is not in the repository.  The audit found that file

  * lists every game TWICE with home/away swapped (5,346 rows for 2,626
    game_pks), so `home_field` is a constant 1 and half the rows carry a
    fabricated home team,
  * includes spring-training games (first row is 2025-03-15),
  * and, at serving time, every 2026 game was scored from end-of-2025
    snapshots because that file is the only history the runner reads.

This module rebuilds the SAME feature families with the SAME column names
(`diff_runs_scored_10g`, `diff_OPS_10g`, `diff_SP_FIP_10`, `diff_BP_WHIP_14g`,
`diff_team_winpct`, ...) so `train_game_model.py` and
`daily_mlb_model_runner.py` keep their interfaces, but every value is
computed from games strictly before the one being predicted:

  team offence / run features   prior 10 games or season-to-date, current season
  bullpen features              prior 14 team games or season-to-date, current season
  starter features              starter's prior 10 starts (two-season window) or
                                season-to-date starts
  win% / run differential       season-to-date before the game

One row per game_pk from the true home team's perspective. Target:
home_win (1 = home team won).

Serving: `build_slate_features()` appends today's scheduled games (with the
probable starters' MLBAM ids) as placeholder rows with no outcome, then runs
the identical code path, so a live prediction sees exactly the information a
training row saw.

Usage
-----
    python build_game_features.py                       # all cached seasons -> data/game_model_data.csv
    python build_game_features.py --seasons 2025 2026
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import history_window as hw

HERE = Path(__file__).resolve().parent
OUT_CSV = HERE / "data" / "game_model_data.csv"
FIP_CONSTANT = 3.10

NEEDED = ["game_pk", "game_date", "game_type", "game_year", "home_team", "away_team",
          "inning_topbot", "pitcher", "batter", "p_throws", "events", "at_bat_number",
          "pitch_number", "bat_score", "post_bat_score", "post_home_score", "post_away_score",
          "player_name"]

HIT = {"single": 1, "double": 2, "triple": 3, "home_run": 4}
WALK = {"walk", "intent_walk"}
HBP = {"hit_by_pitch"}
SF = {"sac_fly", "sac_fly_double_play"}
SH = {"sac_bunt", "sac_bunt_double_play"}
CI = {"catcher_interf"}
STRIKEOUT = {"strikeout", "strikeout_double_play"}
OUTS = {"field_out": 1, "force_out": 1, "fielders_choice_out": 1, "fielders_choice": 1,
        "sac_fly": 1, "sac_bunt": 1, "strikeout": 1, "other_out": 1,
        "grounded_into_double_play": 2, "double_play": 2, "strikeout_double_play": 2,
        "sac_fly_double_play": 2, "sac_bunt_double_play": 2, "triple_play": 3,
        "caught_stealing_2b": 1, "caught_stealing_3b": 1, "caught_stealing_home": 1,
        "pickoff_1b": 1, "pickoff_2b": 1, "pickoff_3b": 1,
        "pickoff_caught_stealing_2b": 1, "pickoff_caught_stealing_3b": 1,
        "pickoff_caught_stealing_home": 1}

TEAM_ALIASES = {"ARI": "AZ", "OAK": "ATH", "CHW": "CWS", "WSN": "WSH", "KCR": "KC",
                "SDP": "SD", "SFG": "SF", "TBR": "TB", "LA": "LAD", "ANA": "LAA"}


def norm_team(x):
    if pd.isna(x):
        return x
    x = str(x).strip().upper()
    return TEAM_ALIASES.get(x, x)


# ---------------------------------------------------------------------------
# Per-game aggregates (outcomes - raw material, never features)
# ---------------------------------------------------------------------------
def _prep(p: pd.DataFrame) -> pd.DataFrame:
    p = p[[c for c in NEEDED if c in p.columns]].copy()
    p["game_date"] = pd.to_datetime(p["game_date"]).dt.normalize()
    for c in ("home_team", "away_team"):
        p[c] = p[c].map(norm_team)
    top = p["inning_topbot"].astype(str).str.lower().eq("top")
    p["bat_team"] = np.where(top, p["away_team"], p["home_team"])
    p["fld_team"] = np.where(top, p["home_team"], p["away_team"])
    ev = p["events"].astype("object").fillna("").astype(str).str.lower()
    p["pa"] = (ev != "").astype(int) * (~ev.str.startswith(("caught_stealing", "pickoff", "stolen_base",
                                                             "wild_pitch", "passed_ball"))).astype(int)
    p["h"] = ev.isin(HIT.keys()).astype(int)
    p["tb"] = ev.map(HIT).fillna(0).astype(int)
    p["hr"] = (ev == "home_run").astype(int)
    p["bb"] = ev.isin(WALK).astype(int)
    p["hbp"] = ev.isin(HBP).astype(int)
    p["sf"] = ev.isin(SF).astype(int)
    p["sh"] = ev.isin(SH).astype(int)
    p["ci"] = ev.isin(CI).astype(int)
    p["k"] = ev.isin(STRIKEOUT).astype(int)
    p["outs"] = ev.map(OUTS).fillna(0).astype(int)
    p["runs"] = (pd.to_numeric(p["post_bat_score"], errors="coerce")
                 - pd.to_numeric(p["bat_score"], errors="coerce")).clip(lower=0).fillna(0)
    return p


def game_results(p: pd.DataFrame) -> pd.DataFrame:
    g = p.groupby("game_pk").agg(game_date=("game_date", "first"), game_type=("game_type", "first"),
                                 home_team=("home_team", "first"), away_team=("away_team", "first"),
                                 home_runs=("post_home_score", "max"), away_runs=("post_away_score", "max"))
    g = g.reset_index()
    g["home_win"] = np.where(g["home_runs"] > g["away_runs"], 1,
                             np.where(g["home_runs"] < g["away_runs"], 0, np.nan))
    return g


def team_batting(p: pd.DataFrame) -> pd.DataFrame:
    cols = ["pa", "h", "tb", "hr", "bb", "hbp", "sf", "sh", "ci", "k"]
    t = p.groupby(["game_pk", "bat_team"])[cols].sum().reset_index().rename(columns={"bat_team": "team"})
    t["ab"] = (t["pa"] - t["bb"] - t["hbp"] - t["sf"] - t["sh"] - t["ci"]).clip(lower=0)
    t["obp_den"] = t["ab"] + t["bb"] + t["hbp"] + t["sf"]
    t["obp_num"] = t["h"] + t["bb"] + t["hbp"]
    return t


def pitching_lines(p: pd.DataFrame) -> pd.DataFrame:
    """One row per (game, pitcher): counting stats + starter flag + hand."""
    first = (p.sort_values(["game_pk", "fld_team", "at_bat_number", "pitch_number"], kind="mergesort")
              .drop_duplicates(["game_pk", "fld_team"])[["game_pk", "fld_team", "pitcher"]]
              .rename(columns={"pitcher": "starter"}))
    agg = p.groupby(["game_pk", "fld_team", "pitcher"]).agg(
        bf=("pa", "sum"), h=("h", "sum"), hr=("hr", "sum"), bb=("bb", "sum"), hbp=("hbp", "sum"),
        k=("k", "sum"), outs=("outs", "sum"), r=("runs", "sum"),
        hand=("p_throws", "first"), name=("player_name", "first")).reset_index()
    agg = agg.merge(first, on=["game_pk", "fld_team"], how="left")
    agg["is_starter"] = (agg["pitcher"] == agg["starter"]).astype(int)
    return agg.drop(columns="starter").rename(columns={"fld_team": "team"})


# ---------------------------------------------------------------------------
# Lagged rolling helpers (current row always excluded)
# ---------------------------------------------------------------------------
def _prior_sum(df, keys, cols, window=None):
    """Sum over PRIOR rows of each key group (rolling `window` or expanding)."""
    shifted = df.groupby(keys, sort=False)[cols].shift(1)
    g = shifted.groupby([df[k] for k in keys], sort=False)
    r = g.rolling(window, min_periods=1).sum() if window else g.expanding().sum()
    r = r.reset_index(level=list(range(len(keys))), drop=True).reindex(df.index)
    # rows with no prior history -> NaN (not 0)
    n = df.groupby(keys, sort=False).cumcount()
    r[n == 0] = np.nan
    return r


def _ratio(num, den, scale=1.0):
    num = np.asarray(num, float); den = np.asarray(den, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(den > 0, scale * num / den, np.nan)
    return out


def team_features(team_games: pd.DataFrame) -> pd.DataFrame:
    """team_games: one row per (game_pk, team) with outcome columns.
    Returns pregame features per (game_pk, team)."""
    t = team_games.sort_values(["team", "game_date", "game_pk"], kind="mergesort").reset_index(drop=True)
    t["season"] = t["game_date"].dt.year
    keys_s = ["team", "season"]                       # *_season / win% / run diff: this season only
    t["win"] = (t["runs_scored"] > t["runs_allowed"]).astype(float)
    t.loc[t["runs_scored"].isna(), "win"] = np.nan
    t["game_ct"] = t["runs_scored"].notna().astype(float)
    t["rd"] = t["runs_scored"] - t["runs_allowed"]
    bat = ["runs_scored", "runs_allowed", "game_ct", "h", "tb", "ab", "obp_num", "obp_den", "k", "bb", "pa"]
    s10 = _prior_sum(t, ["team"], bat, 10)          # last 10 games, may span the prior season
    sea = _prior_sum(t, keys_s, bat + ["win", "rd"], None)
    out = t[["game_pk", "team", "game_date"]].copy()
    out["runs_scored_10g"] = _ratio(s10["runs_scored"], s10["game_ct"])
    out["runs_allowed_10g"] = _ratio(s10["runs_allowed"], s10["game_ct"])
    out["AVG_10g"] = _ratio(s10["h"], s10["ab"])
    out["OBP_10g"] = _ratio(s10["obp_num"], s10["obp_den"])
    out["SLG_10g"] = _ratio(s10["tb"], s10["ab"])
    out["OPS_10g"] = out["OBP_10g"] + out["SLG_10g"]
    out["K_rate_10g"] = _ratio(s10["k"], s10["pa"])
    out["BB_rate_10g"] = _ratio(s10["bb"], s10["pa"])
    out["season_OPS"] = _ratio(sea["obp_num"], sea["obp_den"]) + _ratio(sea["tb"], sea["ab"])
    out["season_K_rate"] = _ratio(sea["k"], sea["pa"])
    out["team_run_diff"] = sea["rd"].to_numpy()
    out["team_winpct"] = _ratio(sea["win"], sea["game_ct"])
    # bullpen
    bp = ["bp_outs", "bp_r", "bp_h", "bp_bb", "bp_k", "bp_bf"]
    b14 = _prior_sum(t, ["team"], bp, 14)
    bse = _prior_sum(t, keys_s, bp, None)
    for tag, src in (("14g", b14), ("season", bse)):
        ip = src["bp_outs"] / 3.0
        out[f"BP_ERA_proxy_{tag}"] = _ratio(src["bp_r"], ip, 9.0)
        out[f"BP_WHIP_{tag}"] = _ratio(src["bp_h"] + src["bp_bb"], ip)
        out[f"BP_KBB_pct_{tag}"] = _ratio(src["bp_k"] - src["bp_bb"], src["bp_bf"])
    return out


def starter_features(starts: pd.DataFrame) -> pd.DataFrame:
    """starts: one row per (game_pk, starter). Prior 10 starts may span the
    two-season window; *_season resets each season."""
    s = starts.sort_values(["pitcher", "game_date", "game_pk"], kind="mergesort").reset_index(drop=True)
    s["season"] = s["game_date"].dt.year
    cols = ["k", "bb", "hbp", "hr", "outs", "bf"]
    l10 = _prior_sum(s, ["pitcher"], cols, 10)
    sea = _prior_sum(s, ["pitcher", "season"], cols, None)
    out = s[["game_pk", "team", "pitcher"]].copy()
    for tag, src in (("10", l10), ("season", sea)):
        ip = src["outs"] / 3.0
        out[f"SP_FIP_{tag}"] = _ratio(13 * src["hr"] + 3 * (src["bb"] + src["hbp"]) - 2 * src["k"], ip) + FIP_CONSTANT
        out[f"SP_KBB_pct_{tag}"] = _ratio(src["k"] - src["bb"], src["bf"])
        out[f"SP_HR9_{tag}"] = _ratio(src["hr"], ip, 9.0)
    return out


# ---------------------------------------------------------------------------
# Assemble one window
# ---------------------------------------------------------------------------
TEAM_FEATS = ["runs_scored_10g", "runs_allowed_10g", "AVG_10g", "OBP_10g", "SLG_10g", "OPS_10g",
              "K_rate_10g", "BB_rate_10g", "BP_ERA_proxy_season", "BP_WHIP_season",
              "BP_KBB_pct_season", "BP_ERA_proxy_14g", "BP_WHIP_14g", "BP_KBB_pct_14g",
              "season_OPS", "season_K_rate", "team_run_diff", "team_winpct"]
SP_FEATS = ["SP_FIP_10", "SP_KBB_pct_10", "SP_HR9_10", "SP_FIP_season", "SP_KBB_pct_season", "SP_HR9_season"]


def _build_window(p: pd.DataFrame, slate: pd.DataFrame | None = None) -> pd.DataFrame:
    p = _prep(p)
    games = game_results(p)
    bat = team_batting(p)
    lines = pitching_lines(p)
    starters = lines[lines["is_starter"] == 1].copy()
    pen = lines[lines["is_starter"] == 0].groupby(["game_pk", "team"]).agg(
        bp_outs=("outs", "sum"), bp_r=("r", "sum"), bp_h=("h", "sum"), bp_bb=("bb", "sum"),
        bp_k=("k", "sum"), bp_bf=("bf", "sum")).reset_index()

    long_rows = []
    for side, opp in (("home", "away"), ("away", "home")):
        x = games[["game_pk", "game_date", f"{side}_team", f"{side}_runs", f"{opp}_runs"]].rename(
            columns={f"{side}_team": "team", f"{side}_runs": "runs_scored", f"{opp}_runs": "runs_allowed"})
        long_rows.append(x)
    team_games = pd.concat(long_rows, ignore_index=True)
    team_games = team_games.merge(bat, on=["game_pk", "team"], how="left").merge(pen, on=["game_pk", "team"], how="left")
    bp_cols = ["bp_outs", "bp_r", "bp_h", "bp_bb", "bp_k", "bp_bf"]
    team_games[bp_cols] = team_games[bp_cols].fillna(0)        # complete game = 0 bullpen work
    starters = starters.merge(games[["game_pk", "game_date"]], on="game_pk", how="left")

    if slate is not None and len(slate):
        sl = slate.copy()
        sl["game_date"] = pd.to_datetime(sl["game_date"]).dt.normalize()
        for side in ("home", "away"):
            team_games = pd.concat([team_games, pd.DataFrame({
                "game_pk": sl["game_pk"], "game_date": sl["game_date"], "team": sl[f"{side}_team"].map(norm_team)})],
                ignore_index=True)
            st = sl[["game_pk", "game_date", f"{side}_team", f"{side}_starter_id"]].rename(
                columns={f"{side}_team": "team", f"{side}_starter_id": "pitcher"})
            st["team"] = st["team"].map(norm_team)
            st = st[st["pitcher"].notna()]
            st["pitcher"] = st["pitcher"].astype("int64")
            starters = pd.concat([starters, st], ignore_index=True)
        games = pd.concat([games, sl[["game_pk", "game_date", "home_team", "away_team"]].assign(
            home_team=lambda d: d["home_team"].map(norm_team), away_team=lambda d: d["away_team"].map(norm_team),
            game_type="R")], ignore_index=True)

    tf = team_features(team_games)
    sf = starter_features(starters)
    hand = (lines[lines["is_starter"] == 1].dropna(subset=["hand"])
            .groupby("pitcher")["hand"].agg(lambda s: s.mode().iat[0]))
    sname = lines.dropna(subset=["name"]).groupby("pitcher")["name"].last()

    rows = games.copy()
    for side in ("home", "away"):
        tfs = tf.rename(columns={c: f"{side}_{c}" for c in TEAM_FEATS}).drop(columns="game_date")
        rows = rows.merge(tfs, left_on=["game_pk", f"{side}_team"], right_on=["game_pk", "team"], how="left").drop(columns="team")
        sfs = sf.rename(columns={c: f"{side}_{c}" for c in SP_FEATS})
        sfs = sfs.rename(columns={"pitcher": f"{side}_starter_id"})
        rows = rows.merge(sfs, left_on=["game_pk", f"{side}_team"], right_on=["game_pk", "team"], how="left").drop(columns="team")
        rows[f"{side}_starter_throws"] = rows[f"{side}_starter_id"].map(hand)
        rows[f"{side}_starter"] = rows[f"{side}_starter_id"].map(sname)
    rows["home_field"] = 1
    rows["same_starter_hand"] = (rows["home_starter_throws"] == rows["away_starter_throws"]).astype(int)
    rows["diff_starter_hand"] = 1 - rows["same_starter_hand"]
    for c in TEAM_FEATS + SP_FEATS:
        rows[f"diff_{c}"] = rows[f"home_{c}"] - rows[f"away_{c}"]
    return rows


def build_game_table(pitches: pd.DataFrame, prediction_seasons: list[int] | None = None) -> pd.DataFrame:
    """Training/evaluation table: one row per competitive game, windowed by season."""
    seasons_avail = sorted(set(hw.game_season(pitches).unique().tolist()))
    prediction_seasons = prediction_seasons or seasons_avail
    parts = []
    for S in prediction_seasons:
        win = hw.window_pitches(pitches, S)
        hw.assert_window(win, S)
        if not (hw.game_season(win) == S).any():
            continue
        rows = _build_window(win)
        rows = rows[rows["game_date"].dt.year == S].copy()
        rows["prediction_season"] = S
        rows["history_seasons"] = "+".join(str(s) for s in hw.describe_window(S, seasons_avail)["seasons_available"])
        parts.append(rows)
        print(f"  game table {S}: {len(rows):,} games")
    out = pd.concat(parts, ignore_index=True).sort_values(["game_date", "game_pk"]).reset_index(drop=True)
    return out


def build_slate_features(pitches: pd.DataFrame, slate: pd.DataFrame) -> pd.DataFrame:
    """Pre-game feature rows for scheduled games (no outcomes). `slate` needs
    game_pk, game_date, home_team, away_team, home_starter_id, away_starter_id."""
    if slate is None or slate.empty:
        return pd.DataFrame()
    d0 = pd.to_datetime(slate["game_date"]).min().normalize()
    S = int(d0.year)
    hist = pitches[pd.to_datetime(pitches["game_date"]) < d0]      # strictly before the slate date
    win = hw.window_pitches(hist, S)
    hw.assert_window(win, S)
    rows = _build_window(win, slate=slate)
    return rows[rows["game_pk"].isin(slate["game_pk"])].reset_index(drop=True)


def main():
    import refresh_full_history as rfh
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seasons", nargs="+", type=int, default=None,
                    help="prediction seasons to build (default: every cached season)")
    ap.add_argument("--out", default=str(OUT_CSV))
    args = ap.parse_args()
    pitches = rfh.load_seasons(rfh.cached_seasons(), columns=NEEDED)
    table = build_game_table(pitches, args.seasons)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False, float_format="%.5f")
    print(f"wrote {args.out}: {len(table):,} games")


if __name__ == "__main__":
    main()
