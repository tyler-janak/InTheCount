"""
enrich_lineup_features.py
=========================
Adds per-pitcher-game lineup-level aggregation columns to
`data/pitcher_game_data.csv`.

Why this exists
---------------
The current pitcher features include `team_k_rate_vs_hand_last10` - the
entire opposing team's K-rate against the pitcher's hand. That's the team
average, which is useful but coarse: pitchers face a specific 9-batter
lineup each night, and there's a huge spread between a Judge-Soto-Stanton
top of order and a Maldonado-Wong-Ruiz bottom. Aggregating the **actual
lineup that faced this pitcher** gives a sharper signal - and crucially,
one that's NOT correlated with the pitcher's own K_rate, so XGBoost can't
substitute it.

What gets added
---------------
For each pitcher-game in pitcher_game_data.csv, we look up the 9 batters
who faced that pitcher in that game (from hitter_game_data.csv, matched by
team + game_date + opp_pitcher_name), pull each batter's pre-game
season-to-date and last-10 rates vs the pitcher's handedness, then aggregate:

    lineup_k_rate              - mean of hitter K-rate vs pitcher hand
    lineup_k_rate_last10       - mean of hitter last-10 K-rate vs pitcher hand
    lineup_bb_rate             - mean of hitter walk rate vs pitcher hand
    lineup_bb_rate_last10
    lineup_h_rate              - mean of hitter hit rate vs pitcher hand
    lineup_h_rate_last10
    lineup_hr_rate             - mean of hitter HR rate vs pitcher hand
    lineup_hr_rate_last10
    lineup_avg_ev              - mean of hitter avg EV (contact quality)
    lineup_hard_hit_pct        - mean of hitter hard-hit rate
    lineup_n_batters           - how many batters we actually matched (≤9)

The lineup_* columns are explicitly NOT a copy or correlate of any pitcher
feature - they describe the opponent. This is the information channel that
unlocked the most gain in industry models.

When this runs
--------------
After `refresh_full_history.py` builds the per-game tables, before
training. Idempotent: re-runs drop the prior lineup_* columns first.

Usage
-----
    python enrich_lineup_features.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"

LINEUP_COLS = [
    "lineup_k_rate", "lineup_k_rate_last10",
    "lineup_bb_rate", "lineup_bb_rate_last10",
    "lineup_h_rate",  "lineup_h_rate_last10",
    "lineup_hr_rate", "lineup_hr_rate_last10",
    "lineup_avg_ev",  "lineup_hard_hit_pct",
    "lineup_n_batters",
]


def _pick_hand_col(df: pd.DataFrame, base: str, hand: str) -> pd.Series:
    """Return df[base + '_' + hand] series; falls back to 'R' if 'L' missing."""
    col = f"{base}_{hand}"
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    other = "L" if hand == "R" else "R"
    fallback = f"{base}_{other}"
    if fallback in df.columns:
        return pd.to_numeric(df[fallback], errors="coerce")
    return pd.Series(np.nan, index=df.index)


def enrich_frame(pdf: pd.DataFrame, hdf: pd.DataFrame) -> pd.DataFrame:
    """Add lineup_* columns to a pitcher-game frame (in memory).

    LEAKAGE FIX: the previous version averaged each batter's SAME-GAME
    rate vs hand (hitter_k_rate_vs_hand_*) and same-game avg_EV, i.e. the
    outcome of the very game being predicted. Every input below is now a
    trailing, pre-game value (`_std` season-to-date or `_last10`).
    The lineup itself is the nine batters with the most PA for that team in
    that game - the realised lineup, a close proxy for the posted lineup
    that is known before first pitch (pinch hitters can shift membership)."""
    pdf = pdf.drop(columns=[c for c in LINEUP_COLS if c in pdf.columns]).copy()
    hdf = hdf.copy()
    pdf["game_date"] = pd.to_datetime(pdf["game_date"], errors="coerce")
    hdf["game_date"] = pd.to_datetime(hdf["game_date"], errors="coerce")

    pitcher_team_col = next((c for c in ("team", "pitcher_team") if c in pdf.columns), None)
    hitter_opponent_col = next((c for c in ("opponent", "opponent_team", "pitcher_team") if c in hdf.columns), None)
    if pitcher_team_col is None or hitter_opponent_col is None:
        print("WARNING:  lineup enrichment: missing team/opponent columns - skipping")
        for c in LINEUP_COLS:
            pdf[c] = np.nan
        return pdf
    if pitcher_team_col != "team":
        pdf = pdf.rename(columns={pitcher_team_col: "team"})
    if hitter_opponent_col != "opponent":
        hdf = hdf.rename(columns={hitter_opponent_col: "opponent"})

    use_pk = "game_pk" in pdf.columns and "game_pk" in hdf.columns
    gkey = ["game_pk", "opponent"] if use_pk else ["game_date", "opponent"]
    hitter_groups = hdf.groupby(gkey, sort=False)

    spec = [
        ("lineup_k_rate",         "hitter_k_rate_vs_hand_std"),
        ("lineup_k_rate_last10",  "hitter_k_rate_vs_hand_last10"),
        ("lineup_bb_rate",        "hitter_bb_rate_vs_hand_std"),
        ("lineup_bb_rate_last10", "hitter_bb_rate_vs_hand_last10"),
        ("lineup_h_rate",         "hitter_h_rate_vs_hand_std"),
        ("lineup_h_rate_last10",  "hitter_h_rate_vs_hand_last10"),
        ("lineup_hr_rate",        "hitter_hr_rate_vs_hand_std"),
        ("lineup_hr_rate_last10", "hitter_hr_rate_vs_hand_last10"),
    ]
    rows = []
    matched = 0
    for _, prow in pdf.iterrows():
        phand = str(prow.get("pitcher_hand", "")).strip().upper() or "R"
        if phand not in ("R", "L"):
            phand = "R"
        key = (prow.get("game_pk"), prow.get("team")) if use_pk else (prow.get("game_date"), prow.get("team"))
        try:
            sub = hitter_groups.get_group(key)
        except KeyError:
            rows.append({c: np.nan for c in LINEUP_COLS})
            continue
        if "PA" in sub.columns and len(sub) > 9:
            sub = sub.nlargest(9, "PA")
        matched += 1
        out_row = {}
        for key_out, base in spec:
            series = _pick_hand_col(sub, base, phand)
            out_row[key_out] = float(series.mean()) if series.notna().any() else np.nan
        ev_col = "avg_EV_std" if "avg_EV_std" in sub.columns else None
        if ev_col:
            ev = pd.to_numeric(sub[ev_col], errors="coerce")
            out_row["lineup_avg_ev"] = float(ev.mean()) if ev.notna().any() else np.nan
        else:
            out_row["lineup_avg_ev"] = np.nan
        if "hard_hit_proxy_std" in sub.columns:
            hh = pd.to_numeric(sub["hard_hit_proxy_std"], errors="coerce")
            out_row["lineup_hard_hit_pct"] = float(hh.mean()) if hh.notna().any() else np.nan
        else:
            out_row["lineup_hard_hit_pct"] = np.nan
        out_row["lineup_n_batters"] = int(len(sub))
        rows.append(out_row)
    pdf = pd.concat([pdf, pd.DataFrame(rows, index=pdf.index)], axis=1)
    print(f"  lineup features: matched {matched:,}/{len(pdf):,} pitcher games")
    return pdf


def enrich(
    pitcher_csv: Path = DATA_DIR / "pitcher_game_data.csv",
    hitter_csv: Path = DATA_DIR / "hitter_game_data.csv",
    write_back: bool = True,
) -> pd.DataFrame:
    if not pitcher_csv.exists() or not hitter_csv.exists():
        print(f"WARNING:  Missing input - pitcher: {pitcher_csv.exists()}, hitter: {hitter_csv.exists()}")
        return pd.DataFrame()
    print(f"\n-- Enriching lineup features in {pitcher_csv.name} --")
    pdf = pd.read_csv(pitcher_csv, low_memory=False)
    hdf = pd.read_csv(hitter_csv, low_memory=False)
    pdf = enrich_frame(pdf, hdf)
    if write_back:
        pdf.to_csv(pitcher_csv, index=False, float_format="%.4f")
        print(f"  Wrote {len(pdf):,} rows back -> {pitcher_csv}")
    return pdf


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pitcher-csv", default=str(DATA_DIR / "pitcher_game_data.csv"))
    ap.add_argument("--hitter-csv",  default=str(DATA_DIR / "hitter_game_data.csv"))
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args()
    enrich(Path(args.pitcher_csv), Path(args.hitter_csv), write_back=not args.no_write)


if __name__ == "__main__":
    main()
