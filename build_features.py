"""
build_features.py
=================
One entry point that turns the cached Statcast seasons into every modelling
table, under the two-season history rule (history_window.py).

    pitch_data_<season>.csv                          (refresh_full_history.py)
            |
            v   per prediction season S: window = S-2..S, built BEFORE any feature
    data/features/pitcher_games.csv       all seasons, training / evaluation
    data/features/hitter_games.csv
    data/pitcher_game_data.csv            current season only  (served by the site,
    data/hitter_game_data.csv             read by hitterspitchers_today.py)
    data/team_batting_hand_context.csv
    data/team_pitching_hand_context.csv
    data/game_model_data.csv              game-level table (build_game_features.py)

Usage
-----
    python build_features.py                     # every cached season
    python build_features.py --seasons 2026      # rebuild only the 2026 window (daily cron)
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import build_game_features as bgf
import hitterspitchers_data as hpd
import refresh_full_history as rfh

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
FEAT = DATA / "features"


def cached_seasons() -> list[int]:
    return rfh.cached_seasons()


def seasons_to_build(current: int) -> list[int]:
    """Daily mode: always the current season's window, plus any cached season
    whose rows are missing from data/features (fresh CI runner, new season)."""
    avail = cached_seasons()
    have = set()
    p = FEAT / "hitter_games.csv"
    if p.exists():
        have = set(pd.to_datetime(pd.read_csv(p, usecols=["game_date"])["game_date"]).dt.year.unique().tolist())
    return sorted({current} | {s for s in avail if s not in have and s <= current})


def _merge_season_rows(path: Path, new: pd.DataFrame, seasons: list[int]) -> pd.DataFrame:
    """Replace the rows of `seasons` in an existing feature table with `new`."""
    if path.exists() and not new.empty:
        old = pd.read_csv(path, low_memory=False)
        yr = pd.to_datetime(old["game_date"]).dt.year
        old = old[~yr.isin(seasons)]
        return pd.concat([old, new], ignore_index=True, sort=False)
    return new


def build(seasons: list[int] | None = None, export_current: bool = True) -> dict:
    avail = cached_seasons()
    if not avail:
        raise SystemExit("no cached Statcast seasons - run refresh_full_history.py first")
    seasons = seasons or avail
    # every season that can sit inside one of the requested windows
    need = sorted({s for S in seasons for s in range(S - 2, S + 1) if s in avail})
    print(f"cached seasons: {avail}  | building prediction seasons {seasons} from {need}")
    pitches = rfh.load_seasons(need, columns=sorted(hpd.NEEDED_COLS | set(bgf.NEEDED)))

    FEAT.mkdir(parents=True, exist_ok=True)
    tables = hpd.build_windowed_tables(pitches[[c for c in pitches.columns if c in hpd.NEEDED_COLS]], seasons)
    summary = {"built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "prediction_seasons": seasons, "cached_seasons": avail, "tables": {}}
    for name in ("pitcher", "hitter"):
        path = FEAT / f"{name}_games.csv"
        merged = _merge_season_rows(path, tables[name], seasons)
        merged.to_csv(path, index=False)
        yr = pd.to_datetime(merged["game_date"]).dt.year
        summary["tables"][name] = {"rows": int(len(merged)), "rows_by_season": {int(k): int(v) for k, v in yr.value_counts().sort_index().items()}}
        print(f"  {path.relative_to(HERE)}: {len(merged):,} rows {summary['tables'][name]['rows_by_season']}")

    game = bgf.build_game_table(pitches[[c for c in pitches.columns if c in bgf.NEEDED]], seasons)
    gpath = DATA / "game_model_data.csv"
    if gpath.exists():
        old = pd.read_csv(gpath, low_memory=False)
        old = old[~pd.to_datetime(old["game_date"]).dt.year.isin(seasons)]
        game = pd.concat([old, game], ignore_index=True, sort=False)
    game = game.sort_values(["game_date", "game_pk"])
    game.to_csv(gpath, index=False, float_format="%.5f")
    summary["tables"]["game"] = {"rows": int(len(game))}
    print(f"  {gpath.relative_to(HERE)}: {len(game):,} games")

    if export_current:
        cur = max(avail)
        if cur in seasons:
            # the live site + hitterspitchers_today read these (current season only,
            # same shape as before - now computed with the 2-season window)
            for name, t in (("pitcher", tables["pitcher"]), ("hitter", tables["hitter"])):
                rows = t[pd.to_datetime(t["game_date"]).dt.year == cur]
                rows.to_csv(DATA / f"{name}_game_data.csv", index=False, float_format="%.4f")
            for name in ("team_batting_hand_ctx", "team_pitching_hand_ctx"):
                t = tables[name]
                fname = "team_batting_hand_context.csv" if "batting" in name else "team_pitching_hand_context.csv"
                rows = t[pd.to_datetime(t["game_date"]).dt.year == cur]
                rows.to_csv(DATA / fname, index=False, float_format="%.4f")
            print(f"  exported current-season ({cur}) CSVs for serving")
    (FEAT / "build_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seasons", nargs="+", type=int, default=None)
    ap.add_argument("--no-export", action="store_true")
    args = ap.parse_args()
    build(args.seasons, export_current=not args.no_export)


if __name__ == "__main__":
    main()
