"""
Code-level leakage tests.

The core idea is a PERTURBATION test: change the outcome of one game in the
raw pitch data, rebuild every feature table, and require that

  * no pre-game feature of that game moves (nothing in the row can depend on
    the game's own result), while
  * features of later games DO move (the perturbation is real and propagates
    forward, so the test is not vacuous).

Plus unit tests of the trailing-window primitives and of the two-season rule
(adding much older data must not change any feature).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import build_game_features as bgf
import history_window as hw
import hitterspitchers_data as hpd
import hitterspitchers_train as hpt
import refresh_full_history as sc
import train_game_model as tgm
from leakage_guard import LeakageError, assert_pregame_features

RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "statcast"


def _slice(season=2025, start="04-01", end="04-24"):
    p = sc.read_season(season, columns=sorted(hpd.NEEDED_COLS | set(bgf.NEEDED)))
    if p.empty:
        pytest.skip(f"no cached Statcast for {season}")
    d = pd.to_datetime(p["game_date"])
    p = p[(d >= f"{season}-{start}") & (d <= f"{season}-{end}") & (p["game_type"] == "R")]
    return p.reset_index(drop=True)


def _perturb(p, game_pk):
    """Turn every plate-appearance result of one game into a home run."""
    q = p.copy()
    m = (q["game_pk"] == game_pk) & q["events"].notna() & (q["events"].astype(str) != "")
    q.loc[m, "events"] = "home_run"
    q.loc[m, "launch_speed"] = 115.0
    q.loc[m, "post_bat_score"] = q.loc[m, "bat_score"] + 4
    q.loc[q["game_pk"] == game_pk, "post_home_score"] = 30
    return q


def _same(a, b):
    a = pd.to_numeric(a, errors="coerce").to_numpy(float)
    b = pd.to_numeric(b, errors="coerce").to_numpy(float)
    return np.allclose(a, b, equal_nan=True, atol=1e-9)


@pytest.fixture(scope="module")
def perturbed_tables():
    p = _slice()
    games = p.groupby("game_pk")["game_date"].first().sort_values()
    target = games.index[len(games) // 2]          # a mid-window game
    base = hpd.build_tables(p[[c for c in p.columns if c in hpd.NEEDED_COLS]])
    pert = hpd.build_tables(_perturb(p, target)[[c for c in p.columns if c in hpd.NEEDED_COLS]])
    return target, games, base, pert, p


@pytest.mark.parametrize("kind,features,key", [
    ("pitcher", hpt.PITCHER_FEATURES, "pitcher"),
    ("hitter", hpt.HITTER_FEATURES, "batter"),
])
def test_player_features_do_not_see_own_game(perturbed_tables, kind, features, key):
    target, games, base, pert, _ = perturbed_tables
    b = base[kind].set_index(["game_pk", key]).sort_index()
    q = pert[kind].set_index(["game_pk", key]).sort_index()
    feats = [f for f in features if f in b.columns]
    assert len(feats) > 50
    rows = b.index.get_level_values(0) == target
    assert rows.sum() > 0
    moved = [f for f in feats if not _same(b.loc[rows, f], q.loc[rows, f])]
    assert not moved, f"features of game {target} changed when only its outcome changed: {moved}"


def test_player_perturbation_propagates_forward(perturbed_tables):
    target, games, base, pert, _ = perturbed_tables
    later = games[games > games[target]].index
    b = base["hitter"].set_index(["game_pk", "batter"]).sort_index()
    q = pert["hitter"].set_index(["game_pk", "batter"]).sort_index()
    rows = b.index.get_level_values(0).isin(later)
    assert not _same(b.loc[rows, "HR_last10"], q.loc[rows, "HR_last10"]), "perturbation had no downstream effect - test is vacuous"


def test_game_features_do_not_see_own_game():
    p = _slice()
    games = p.groupby("game_pk")["game_date"].first().sort_values()
    target = games.index[len(games) // 2]
    base = bgf._build_window(p).set_index("game_pk")
    pert = bgf._build_window(_perturb(p, target)).set_index("game_pk")
    feats = [f for f in tgm.FEATURES if f in base.columns]
    moved = [f for f in feats if not _same(base.loc[[target], f], pert.loc[[target], f])]
    assert not moved, f"game features moved: {moved}"
    assert pert.loc[target, "home_runs"] == 30            # the outcome itself did change
    later = games[games > games[target]].index
    assert not _same(base.loc[later, "diff_runs_scored_10g"], pert.loc[later, "diff_runs_scored_10g"])


def test_older_seasons_cannot_enter_indirectly():
    """Add a copy of the data relabelled as a season 3 years earlier: under the
    two-season rule the features of the real season must not change at all."""
    p = _slice(start="04-01", end="04-15")
    p = p[[c for c in p.columns if c in hpd.NEEDED_COLS]]
    old = p.copy()
    old["game_date"] = pd.to_datetime(old["game_date"]) - pd.DateOffset(years=3)
    old["game_year"] = old["game_year"] - 3
    old["game_pk"] = old["game_pk"] + 10_000_000
    a = hpd.build_windowed_tables(p, [2025])
    b = hpd.build_windowed_tables(pd.concat([old, p], ignore_index=True), [2025])
    for kind, key, feats in (("pitcher", "pitcher", hpt.PITCHER_FEATURES), ("hitter", "batter", hpt.HITTER_FEATURES)):
        x = a[kind].set_index(["game_pk", key]).sort_index()
        y = b[kind].set_index(["game_pk", key]).sort_index()
        assert x.index.equals(y.index)
        moved = [f for f in feats if f in x.columns and not _same(x[f], y[f])]
        assert not moved, f"{kind}: 3-years-old data leaked into {moved}"


def test_window_and_assertion():
    assert hw.eligible_seasons(2026) == [2024, 2025, 2026]
    assert hw.eligible_seasons(2027) == [2025, 2026, 2027]
    df = pd.DataFrame({"game_date": pd.to_datetime(["2023-05-01", "2024-05-01", "2026-05-01", "2027-05-01"]),
                       "game_type": ["R", "R", "S", "R"]})
    w = hw.window_pitches(df, 2026)
    assert set(w["game_date"].dt.year) == {2024}          # 2023 too old, 2027 future, spring dropped
    with pytest.raises(AssertionError):
        hw.assert_window(df, 2026)


def test_trailing_primitives_exclude_current_row_and_reset_std():
    df = pd.DataFrame({"pid": [1] * 6, "game_date": pd.to_datetime(
        ["2025-09-01", "2025-09-02", "2025-09-03", "2026-04-01", "2026-04-02", "2026-04-03"]),
        "x": [1.0, 2.0, 3.0, 10.0, 20.0, 30.0]})
    r = hpd.add_rolling(df, "pid", ["x"], windows=[2])
    assert np.isnan(r["x_last2"].iloc[0])
    assert r["x_last2"].tolist()[1:] == [1.0, 1.5, 2.5, 6.5, 15.0]       # spans the season boundary
    s = hpd.season_to_date(df, "pid", ["x"])
    assert np.isnan(s["x_std"].iloc[0])                                   # never includes the current row
    assert s["x_std"].tolist()[1:] == [1.0, 1.5, 2.0, 4.0, 7.2]


def test_static_guard_rejects_same_game_columns():
    with pytest.raises(LeakageError):
        assert_pregame_features(["K_rate_last10", "team_k_rate_vs_hand"])
    with pytest.raises(LeakageError):
        assert_pregame_features(["hitter_h_rate_vs_hand_R"])
    assert_pregame_features(hpt.PITCHER_FEATURES)
    assert_pregame_features(hpt.HITTER_FEATURES)
    assert_pregame_features(tgm.FEATURES)


def test_date_split_is_chronological():
    df = pd.DataFrame({"game_date": pd.date_range("2025-04-01", periods=100, freq="D"), "y": range(100)})
    tr, va, te = hpt.date_split(df, "game_date", "2025-06-01", "2025-06-20")
    assert tr["game_date"].max() < va["game_date"].min() <= va["game_date"].max() < te["game_date"].min()


def test_game_serving_matches_training_rows():
    """Train/serve parity: features built for a 'scheduled' slate from data
    strictly before the slate date equal the stored training rows."""
    p = sc.load_seasons([s for s in (2025, 2026) if sc.season_path(s).exists()], columns=bgf.NEEDED)
    if p.empty:
        pytest.skip("no cached Statcast")
    table = bgf.build_game_table(p, [2026] if (p["game_year"] == 2026).any() else [2025])
    day = table["game_date"].sort_values().unique()[-20]
    rows = table[table["game_date"] == day]
    slate = rows[["game_pk", "game_date", "home_team", "away_team", "home_starter_id", "away_starter_id"]]
    served = bgf.build_slate_features(p, slate).set_index("game_pk").loc[rows["game_pk"]]
    stored = rows.set_index("game_pk")
    for f in tgm.FEATURES:
        assert _same(served[f], stored[f]), f"train/serve mismatch on {f}"
