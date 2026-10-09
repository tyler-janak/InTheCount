"""
experiments/original_methodology.py
===================================
Rebuilds the player feature tables EXACTLY the way the pre-audit pipeline did
(the original modules are kept verbatim in experiments/original_code/), so the
updated pipeline can be compared against the original methodology on the same
train / validation / test dates.

Original pipeline reproduced here (refresh_full_history.py before the audit):
  * input  = 2025 + 2026 pitch caches concatenated (spring training included),
             rows in Baseball Savant order (newest first) - the order the original
             "last row = starter" heuristic relies on
  * game_pk not loaded (player-game = player + date)
  * features = original hitterspitchers_data + enrich_team / enrich_lineup /
               enrich_truetalent, unchanged (including the same-game opp_sp_* columns)

Output: experiments/original_methodology/data/{pitcher,hitter}_game_data.csv
Then:   python run_backtest.py --features-dir experiments/original_methodology/data \
                               --out-dir outputs/model_evaluation/comparison/original_methodology
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE / "original_code"))
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

import refresh_full_history as rfh  # noqa: E402

OUT = HERE / "original_methodology" / "data"


def main(seasons=(2025, 2026)):
    import enrich_lineup_features_orig as el
    import enrich_team_features_orig as et
    import enrich_truetalent_orig as ett
    import hitterspitchers_data_orig as hpd

    OUT.mkdir(parents=True, exist_ok=True)
    p = rfh.load_seasons(list(seasons))
    p = p.sort_values(["game_date", "game_pk", "at_bat_number", "pitch_number"],
                      ascending=[False, False, False, False])          # Savant order
    tmp = OUT / "_pitch_data_combined.csv"
    p.to_csv(tmp, index=False)
    del p

    df = hpd.load_data(str(tmp))
    df = hpd.event_flags(df)
    df = hpd.mark_actual_starters(df)
    pf = hpd.load_park_factors(str(ROOT / "data" / "park_factors.csv"))
    tb = hpd.build_team_batting_hand_context(df)
    tp = hpd.build_team_pitching_hand_context(df)
    pitcher = hpd.build_pitcher_games(df, tb)
    hitter = hpd.build_hitter_games(df, tp)
    hitter = hpd.enrich_hitter_with_opp_starter(hitter, pitcher)
    pitcher = hpd.merge_park_factors(pitcher, pf)
    hitter = hpd.merge_park_factors(hitter, pf)
    pc, hc = OUT / "pitcher_game_data.csv", OUT / "hitter_game_data.csv"
    pitcher.to_csv(pc, index=False, float_format="%.4f")
    hitter.to_csv(hc, index=False, float_format="%.4f")
    et.enrich(hc)
    el.enrich(pc, hc)
    ett.enrich(pc, hc)
    tmp.unlink()
    print(f"original-methodology tables -> {OUT}")


if __name__ == "__main__":
    main()
