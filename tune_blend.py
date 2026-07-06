"""
tune_blend.py
==============
Find the optimal ensemble blend weight α per stat by grid-searching graded
historical projections. Reads `2026_player_accuracy.csv` (which now carries
the `direct_*` and `rate_*` component columns written by
hitterspitchers_today.score_pitchers / score_hitters) and, for each stat,
computes the α that minimises MAE of:

    blended = (1 - α) * direct_count + α * (rate_model_count)

vs the actual outcome column from the MLB box score.

Why this exists
---------------
The per-stat weights in PITCHER_BLEND_WEIGHTS / HITTER_BLEND_WEIGHTS are
hand-tuned rules of thumb. This script replaces them with values fit
empirically from live grading data. As more games accumulate, the optimal α
typically drifts a bit; re-run periodically (every few hundred new picks).

Usage
-----
    python tune_blend.py
    python tune_blend.py --since 2026-05-01      # window the data
    python tune_blend.py --min-rows 50           # skip stats with too few rows

Output
------
For each stat: best α, MAE at that α, current α's MAE, and the projected
improvement. Prints the corrected PITCHER_BLEND_WEIGHTS / HITTER_BLEND_WEIGHTS
dict you can paste straight into hitterspitchers_today.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ACC_CSV = HERE / "2026_player_accuracy.csv"

# Stat name -> (player_type, direct col, rate col, actual col, current α default).
PITCHER_STATS = [
    ("K",  "pitcher", "direct_K",  "rate_K",  "actual_strikeouts",   0.50),
    ("BB", "pitcher", "direct_BB", "rate_BB", "actual_walks",        0.25),
    ("H",  "pitcher", "direct_H",  "rate_H",  "actual_hits_allowed", 0.20),
    ("HR", "pitcher", "direct_HR", "rate_HR", None,                  0.30),  # actual_hr_allowed not in CSV
]
HITTER_STATS = [
    ("H",  "hitter", "direct_H",  "rate_H",  "actual_hits",        0.55),
    ("HR", "hitter", "direct_HR", "rate_HR", "actual_hr",          0.50),
    ("BB", "hitter", "direct_BB", "rate_BB", "actual_walks",       0.50),
    ("K",  "hitter", "direct_K",  "rate_K",  "actual_strikeouts",  0.50),
    ("TB", "hitter", "direct_TB", "rate_TB", "actual_total_bases", 0.55),
]


def _grid_search(direct: pd.Series, rate: pd.Series, actual: pd.Series,
                 alpha_grid=np.linspace(0.0, 1.0, 21)) -> tuple[float, float]:
    """Return (best_alpha, best_mae). Uses 5%-step grid by default."""
    d = pd.to_numeric(direct, errors="coerce").to_numpy()
    r = pd.to_numeric(rate,   errors="coerce").to_numpy()
    y = pd.to_numeric(actual, errors="coerce").to_numpy()
    mask = np.isfinite(d) & np.isfinite(r) & np.isfinite(y)
    if mask.sum() < 30:
        return float("nan"), float("nan")
    d, r, y = d[mask], r[mask], y[mask]
    best_a, best_mae = 0.5, float("inf")
    for a in alpha_grid:
        blended = (1 - a) * d + a * r
        mae = float(np.mean(np.abs(blended - y)))
        if mae < best_mae:
            best_mae, best_a = mae, float(a)
    return best_a, best_mae


def _block(df: pd.DataFrame, stats: list, ptype: str,
           min_rows: int) -> dict[str, float]:
    print(f"\n── {ptype.upper()} ─────────────────────────────────────────────────")
    print(f"{'Stat':<6}{'n':>6}  {'cur α':>7} {'cur MAE':>9}  {'best α':>7} "
          f"{'best MAE':>10}  {'Δ MAE':>9}  {'direct MAE':>11} {'rate MAE':>10}")
    print("-" * 88)
    fitted: dict[str, float] = {}
    sub = df[df["player_type"].astype(str).str.lower() == ptype]
    for name, _ptype, dcol, rcol, acol, current_a in stats:
        if acol is None or acol not in sub.columns or dcol not in sub.columns or rcol not in sub.columns:
            print(f"{name:<6}{'(missing)':>20}")
            continue
        rows = sub.dropna(subset=[dcol, rcol, acol])
        n = len(rows)
        if n < min_rows:
            print(f"{name:<6}{n:>6}  (too few rows; need ≥ {min_rows})")
            continue

        d = pd.to_numeric(rows[dcol], errors="coerce").to_numpy()
        r = pd.to_numeric(rows[rcol], errors="coerce").to_numpy()
        y = pd.to_numeric(rows[acol], errors="coerce").to_numpy()

        direct_mae = float(np.mean(np.abs(d - y)))
        rate_mae   = float(np.mean(np.abs(r - y)))
        cur_blend  = (1 - current_a) * d + current_a * r
        cur_mae    = float(np.mean(np.abs(cur_blend - y)))

        best_a, best_mae = _grid_search(rows[dcol], rows[rcol], rows[acol])
        delta = cur_mae - best_mae
        fitted[name] = round(best_a, 2)
        print(f"{name:<6}{n:>6}  {current_a:>7.2f} {cur_mae:>9.3f}  "
              f"{best_a:>7.2f} {best_mae:>10.3f}  {delta:>+9.3f}  "
              f"{direct_mae:>11.3f} {rate_mae:>10.3f}")
    return fitted


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--acc-csv", default=str(ACC_CSV))
    ap.add_argument("--since", default=None,
                    help="Only use rows with game_date on/after this date.")
    ap.add_argument("--min-rows", type=int, default=50,
                    help="Skip stats with fewer than this many gradable rows.")
    args = ap.parse_args()

    p = Path(args.acc_csv)
    if not p.exists():
        print(f"⚠️  {p} not found"); return
    df = pd.read_csv(p, low_memory=False)

    if args.since:
        df["game_date"] = pd.to_datetime(df.get("game_date"), errors="coerce")
        df = df[df["game_date"] >= pd.to_datetime(args.since)]

    if "direct_K" not in df.columns:
        print("⚠️  Accuracy CSV is missing the direct_*/rate_* component columns.")
        print("    The current snapshots predate the blend-component logging.")
        print("    Re-run a force-backfill so new snapshots include them, then re-grade:")
        print("        python -c \"from backfill_player_predictions import backfill; "
              "backfill(start='2026-03-25', end=None, force=True, grade=False, verbose=True)\"")
        print("        python daily_update.py")
        return

    print(f"\nLoaded {len(df):,} graded rows from {p.name}")
    if args.since:
        print(f"(filtered to game_date >= {args.since})")

    pitcher_weights = _block(df, PITCHER_STATS, "pitcher", args.min_rows)
    hitter_weights  = _block(df, HITTER_STATS,  "hitter",  args.min_rows)

    print("\n── Recommended config (paste into hitterspitchers_today.py) ──────────")
    if pitcher_weights:
        print("PITCHER_BLEND_WEIGHTS = {")
        for k, v in pitcher_weights.items():
            print(f"    {k!r:<5}: {v:.2f},")
        print("}")
    if hitter_weights:
        print("HITTER_BLEND_WEIGHTS = {")
        for k, v in hitter_weights.items():
            print(f"    {k!r:<5}: {v:.2f},")
        print("}")
    print()


if __name__ == "__main__":
    main()
