import numpy as np
import pandas as pd
import pytest

import quality_checks as qc


def _games(n=300):
    rng = np.random.default_rng(0)
    p = rng.uniform(0.3, 0.7, n)
    y = (rng.uniform(size=n) < p).astype(int)
    df = pd.DataFrame({"game_date": pd.date_range("2026-07-01", periods=n, freq="h"), "game_pk": range(n),
                       "home_team": "NYY", "away_team": "BOS", "actual_home_win": y, "p_selected": p})
    df["predicted_home_win"] = (p > 0.5).astype(int)
    df["correct"] = (df["predicted_home_win"] == y).astype(int)
    return df


def test_clean_game_predictions_pass():
    r = qc.Report(); qc.check_game_predictions(_games(), "2026-07-01", r); r.raise_if_failed()


@pytest.mark.parametrize("mutate", [
    lambda d: d.assign(p_selected=d["p_selected"].where(d.index != 3, 1.3)),
    lambda d: pd.concat([d, d.iloc[[0]]]),
    lambda d: d.assign(game_date=d["game_date"].where(d.index != 0, pd.Timestamp("2026-06-01"))),
])
def test_bad_game_predictions_fail_loudly(mutate):
    r = qc.Report(); qc.check_game_predictions(mutate(_games()), "2026-07-01", r)
    with pytest.raises(qc.QualityCheckError):
        r.raise_if_failed()


def test_player_history_rule_violation_detected():
    df = pd.DataFrame({"game_date": ["2026-07-01"], "game_pk": [1], "player_id": [5], "target": ["H"],
                       "actual": [1.0], "pred_final": [1.2], "error": [0.2], "abs_error": [0.2],
                       "prediction_season": [2026], "history_seasons": ["2023+2025+2026"]})
    r = qc.Report(); qc.check_player_predictions(df, "hitter", "2026-07-01", r)
    with pytest.raises(qc.QualityCheckError):
        r.raise_if_failed()
