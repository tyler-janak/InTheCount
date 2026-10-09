"""
backfill_blend_components.py
=============================
Quick backfill of the direct_*/rate_* blend-component columns into existing
snapshots, without redoing the slow full backfill (no lineup fetching, no
MLB Stats API, no Rotowire scraping). Just pandas + model inference.

What it does
------------
For every outputs/hitterspitchers_<YYYY-MM-DD>.csv on disk:
  • For each pitcher row, look up the pitcher's feature row as-of (date - 1)
    from data/pitcher_game_data.csv and run BOTH the direct count model and
    the per-9 rate model. Compute rate_count = rate_pred × proj_ip / 9.
  • For each hitter row, same with data/hitter_game_data.csv and proj_pa.
  • Write the 8 component columns (direct_K, rate_K, direct_BB, rate_BB,
    direct_H, rate_H, direct_HR, rate_HR) back into the snapshot.

Idempotent — rows that already have a finite direct_K are skipped. Run after
adding new rate models or after changing PITCHER_RATE_TARGETS / HITTER_RATE_TARGETS,
WITHOUT having to re-do a full backfill_player_predictions force run.

Usage
-----
    python backfill_blend_components.py
    python backfill_blend_components.py --since 2026-05-01
    python backfill_blend_components.py --force          # overwrite existing components
"""

from __future__ import annotations

import argparse
import glob
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

import hitterspitchers_today as hpt

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "outputs"
DATA_DIR = HERE / "data"
DATE_RX = re.compile(r"hitterspitchers_(\d{4}-\d{2}-\d{2})\.csv$")


def _latest_row(df: pd.DataFrame, id_col: str, mlb_id, before_date) -> pd.Series:
    """Return the latest row in `df` for `mlb_id` strictly before `before_date`."""
    if mlb_id is None or pd.isna(mlb_id):
        return pd.Series(dtype=object)
    try:
        mid = int(mlb_id)
    except Exception:
        return pd.Series(dtype=object)
    sub = df[(pd.to_numeric(df[id_col], errors="coerce") == mid)
             & (df["game_date"] < before_date)]
    if sub.empty:
        return pd.Series(dtype=object)
    return sub.sort_values("game_date").iloc[-1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--since", default=None,
                    help="Only process snapshots on/after this date (YYYY-MM-DD).")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite existing direct_*/rate_* values.")
    args = ap.parse_args()

    print("Loading models …")
    pitcher_count = hpt.load_models_count_only("pitcher", hpt.PITCHER_TARGETS)
    hitter_count  = hpt.load_models_count_only("hitter",  hpt.HITTER_TARGETS)
    pitcher_rate  = hpt.load_models_count_only("pitcher", hpt.PITCHER_RATE_TARGETS)
    hitter_rate   = hpt.load_models_count_only("hitter",  hpt.HITTER_RATE_TARGETS)
    print(f"  pitcher count: {list(pitcher_count.keys())}")
    print(f"  pitcher rate : {list(pitcher_rate.keys())}")
    print(f"  hitter count : {list(hitter_count.keys())}")
    print(f"  hitter rate  : {list(hitter_rate.keys())}")

    print("Loading feature tables …")
    pgd = pd.read_csv(DATA_DIR / "pitcher_game_data.csv", low_memory=False)
    hgd = pd.read_csv(DATA_DIR / "hitter_game_data.csv",  low_memory=False)
    pgd["game_date"] = pd.to_datetime(pgd["game_date"], errors="coerce")
    hgd["game_date"] = pd.to_datetime(hgd["game_date"], errors="coerce")

    pitcher_id_col = next((c for c in ("pitcher", "pitcher_id", "mlb_id", "player_id") if c in pgd.columns), None)
    hitter_id_col  = next((c for c in ("batter", "batter_id", "mlb_id", "player_id") if c in hgd.columns), None)
    if not (pitcher_id_col and hitter_id_col):
        print(f"⚠️  missing id columns: pitcher_id_col={pitcher_id_col}, hitter_id_col={hitter_id_col}")
        return
    print(f"  pitcher id col: {pitcher_id_col}   hitter id col: {hitter_id_col}")

    snaps = sorted(glob.glob(str(OUT_DIR / "hitterspitchers_*.csv")))
    if args.since:
        cutoff = pd.to_datetime(args.since).date()
        keep = []
        for p in snaps:
            m = DATE_RX.search(os.path.basename(p))
            if m and pd.to_datetime(m.group(1)).date() >= cutoff:
                keep.append(p)
        snaps = keep
    if not snaps:
        print("No snapshots match."); return
    print(f"\nProcessing {len(snaps)} snapshot file(s) …")

    total_rows = 0
    total_filled = 0
    for path in snaps:
        m = DATE_RX.search(os.path.basename(path))
        if not m:
            continue
        snap_date = pd.to_datetime(m.group(1))
        # Skip zero-byte / no-column files cleanly (these happen when an
        # earlier backfill ran while the lineup API was down — empty CSV
        # got written instead of being skipped). Don't try to read them.
        try:
            if os.path.getsize(path) < 50:    # smaller than even a header row
                print(f"  {os.path.basename(path):<35}  (empty file — skipping)")
                continue
            df = pd.read_csv(path, low_memory=False)
        except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
            print(f"  {os.path.basename(path):<35}  (unreadable: {e})")
            continue
        if df.empty:
            continue
        # Ensure component columns exist
        for c in ("direct_K","rate_K","direct_BB","rate_BB",
                  "direct_H","rate_H","direct_HR","rate_HR",
                  "direct_TB","rate_TB"):
            if c not in df.columns:
                df[c] = np.nan

        filled = 0
        for idx, row in df.iterrows():
            # Only skip if BOTH the direct and rate sides are already populated.
            # (An earlier run could have filled only direct_K when the rate
            # models weren't loading.)
            if (not args.force
                and pd.notna(row.get("direct_K"))
                and pd.notna(row.get("rate_K"))):
                continue
            ptype = str(row.get("player_type", "")).lower()
            mid = row.get("mlb_id")

            if ptype == "pitcher":
                feat = _latest_row(pgd, pitcher_id_col, mid, snap_date)
                if feat.empty:
                    continue
                ip = float(row.get("proj_ip", 0.0) or 0.0)
                ip9 = ip / 9.0 if ip > 0 else 0.0
                for target, count_var in (("K","direct_K"),("BB","direct_BB"),
                                           ("H","direct_H"),("HR","direct_HR")):
                    b = pitcher_count.get(target)
                    if b is None: continue
                    v = hpt.predict_model(b, feat)
                    if v is not None: df.at[idx, count_var] = float(v)
                for rate_key, rate_var in (("K_per_9","rate_K"),("BB_per_9","rate_BB"),
                                            ("H_per_9","rate_H"),("HR_per_9","rate_HR")):
                    b = pitcher_rate.get(rate_key)
                    if b is None or ip9 <= 0: continue
                    v = hpt.predict_model(b, feat)
                    if v is not None: df.at[idx, rate_var] = max(0.0, float(v)) * ip9
                filled += 1

            elif ptype == "hitter":
                feat = _latest_row(hgd, hitter_id_col, mid, snap_date)
                if feat.empty:
                    continue
                pa = float(row.get("proj_pa", 0.0) or 0.0)
                for target, count_var in (("K","direct_K"),("BB","direct_BB"),
                                           ("H","direct_H"),("HR","direct_HR"),
                                           ("TB","direct_TB")):
                    b = hitter_count.get(target)
                    if b is None: continue
                    v = hpt.predict_model(b, feat)
                    if v is not None: df.at[idx, count_var] = float(v)
                for rate_key, rate_var in (("K_per_PA","rate_K"),("BB_per_PA","rate_BB"),
                                            ("H_per_PA","rate_H"),("HR_per_PA","rate_HR"),
                                            ("TB_per_PA","rate_TB")):
                    b = hitter_rate.get(rate_key)
                    if b is None or pa <= 0: continue
                    v = hpt.predict_model(b, feat)
                    if v is not None: df.at[idx, rate_var] = max(0.0, float(v)) * pa
                filled += 1

        # ── RECOMPUTE proj_* from the (now-populated) components using
        # the CURRENT PITCHER_BLEND_WEIGHTS / HITTER_BLEND_WEIGHTS. This is
        # how a tune_blend.py weight update propagates into snapshots without
        # having to do the slow full backfill (no model inference needed —
        # just math on already-filled columns).
        def _blend(d, r, w):
            d = pd.to_numeric(d, errors="coerce")
            r = pd.to_numeric(r, errors="coerce")
            # When rate_* is missing, fall back to direct only (no blend).
            r_filled = r.where(r.notna(), d)
            return (1 - w) * d + w * r_filled

        is_p = df.get("player_type", "").astype(str).str.lower() == "pitcher"
        is_h = df.get("player_type", "").astype(str).str.lower() == "hitter"

        for stat, proj_col, w in (
            ("K",  "proj_strikeouts",   hpt.PITCHER_BLEND_WEIGHTS.get("K",  0.5)),
            ("BB", "proj_walks",        hpt.PITCHER_BLEND_WEIGHTS.get("BB", 0.5)),
            ("H",  "proj_hits_allowed", hpt.PITCHER_BLEND_WEIGHTS.get("H",  0.5)),
            ("HR", "proj_hr_allowed",   hpt.PITCHER_BLEND_WEIGHTS.get("HR", 0.5)),
        ):
            if proj_col not in df.columns:
                df[proj_col] = np.nan
            mask = is_p
            if mask.any():
                df.loc[mask, proj_col] = _blend(
                    df.loc[mask, f"direct_{stat}"],
                    df.loc[mask, f"rate_{stat}"],
                    w,
                ).round(2)

        for stat, proj_col, w in (
            ("H",  "proj_hits",       hpt.HITTER_BLEND_WEIGHTS.get("H",  0.5)),
            ("HR", "proj_hr",         hpt.HITTER_BLEND_WEIGHTS.get("HR", 0.5)),
            ("BB", "proj_walks",      hpt.HITTER_BLEND_WEIGHTS.get("BB", 0.5)),
            ("K",  "proj_strikeouts", hpt.HITTER_BLEND_WEIGHTS.get("K",  0.5)),
            ("TB", "proj_tb",         hpt.HITTER_BLEND_WEIGHTS.get("TB", 0.55)),
        ):
            if proj_col not in df.columns:
                df[proj_col] = np.nan
            mask = is_h
            if mask.any():
                df.loc[mask, proj_col] = _blend(
                    df.loc[mask, f"direct_{stat}"],
                    df.loc[mask, f"rate_{stat}"],
                    w,
                ).round(2)

        df.to_csv(path, index=False)
        total_rows += len(df)
        total_filled += filled
        print(f"  {os.path.basename(path):<35}  {len(df):>4} rows  "
              f"{filled:>4} filled  + proj_* recomputed")

    print(f"\nDone. {total_filled:,} rows updated across {len(snaps)} snapshots "
          f"({total_rows:,} total rows scanned).")
    print("All snapshots now also have proj_* recomputed from current "
          "PITCHER_BLEND_WEIGHTS / HITTER_BLEND_WEIGHTS.")
    print("Next: regrade and check the dashboard.")


if __name__ == "__main__":
    main()
