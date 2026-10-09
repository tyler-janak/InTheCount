"""
refresh_2026_data.py
====================
Current-season refresh used by the daily pipeline (kept under its original
name and CLI so existing commands / workflows keep working).

Since the history-window audit this is a thin wrapper around the shared,
season-configurable modules:

    refresh_full_history.refresh_season() incremental pybaseball pull with a 3-day
                                          overlap -> pitch_data_<year>.csv
    refresh_full_history.refresh_season() for the two previous seasons, ONLY if their
                                          cache is missing (downloaded once, then cached)
    build_features.build()                rebuilds the current season's window
                                          (previous two seasons + current season)
                                          and exports data/*_game_data.csv

Run:
    python refresh_2026_data.py
    python refresh_2026_data.py --end 2026-09-30
    python refresh_2026_data.py --rebuild          # re-pull the current season
    python refresh_2026_data.py --skip-features
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

import refresh_full_history as rfh

ET = ZoneInfo("America/New_York")


def current_season() -> int:
    return datetime.now(ET).year


def refresh_pitch_cache(start: str | None = None, end: str | None = None, rebuild: bool = False):
    """Ensure history seasons exist, then top up the current season."""
    season = current_season()
    for s in (season - 2, season - 1):
        if not rfh.season_path(s).exists():
            print(f"  history season {s} not cached - collecting once")
            try:
                rfh.refresh_season(s)
            except Exception as e:      # history is best-effort; never blocks the current season
                print(f"WARNING:  could not collect history season {s}: {e}")
    return rfh.refresh_season(season, rebuild=rebuild, end=end)


def build_features(pitch_cache=None) -> bool:
    import build_features as bf
    bf.build(seasons=bf.seasons_to_build(current_season()))
    return True


def refresh(start: str | None = None, end: str | None = None, rebuild: bool = False,
            skip_features: bool = False) -> bool:
    print(f"\n========== Refreshing {current_season()} Statcast + features ==========")
    try:
        cache = refresh_pitch_cache(start=start, end=end, rebuild=rebuild)
    except Exception as e:                         # network / pybaseball problems never block the day
        print(f"WARNING:  Statcast refresh failed ({e}); using cached data")
        cache = rfh.season_path(current_season())
    if cache is None or not rfh.season_path(current_season()).exists():
        print("WARNING:  no current-season pitch data - leaving data/*.csv untouched.")
        return False
    if skip_features:
        return True
    return build_features()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    p.add_argument("--rebuild", action="store_true")
    p.add_argument("--skip-features", action="store_true")
    a = p.parse_args()
    sys.exit(0 if refresh(a.start, a.end, a.rebuild, a.skip_features) else 1)


if __name__ == "__main__":
    main()
