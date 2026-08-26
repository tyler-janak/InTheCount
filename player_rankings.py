from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
ACC_CSV = HERE / "2026_player_accuracy.csv"
OUT_JSON = HERE / "outputs" / "player_rankings.json"
AGE_CACHE = HERE / "data" / "player_ages.json"
MODELS_DIR = HERE / "models"
DEFENSE_MODEL_CACHE = MODELS_DIR / "defense_run_value_model.pkl"
BASERUNNING_WEIGHTS_CACHE = MODELS_DIR / "baserunning_re_weights.json"
ET = ZoneInfo("America/New_York")

DEFAULT_SEASON_GAMES = 162
AGE_CUTOFF = 25
TOP_N_HITTERS_FOR_AGE = 400
TOP_N_PITCHERS_FOR_AGE = 300
TOP_N_HITTERS_FOR_BASERUNNING = 400

RUNS_PER_WIN = 10.0
REPLACEMENT_RUNS_PER_600PA = 20.0
PITCHER_REPLACEMENT_RA9_MULTIPLIER = 1.28

# Regression controls
HITTER_REGRESSION_PA = 250.0
PITCHER_REGRESSION_IP = 100.0
DEFENSE_REGRESSION_GAMES = 120.0
BASERUNNING_REGRESSION_GAMES = 100.0
PLAYING_TIME_PRIOR_GAMES = 30.0

PITCHER_WEIGHTS = {"IP": 3.0, "K": 1.0, "ER": -1.0, "H": -0.5, "BB": -0.5}

BATTING_LINEAR_WEIGHTS = {
    "BB": 0.69,
    "1B": 0.89,
    "2B": 1.27,
    "3B": 1.62,
    "HR": 2.10,
    "OUT": -0.25,
}

FALLBACK_SB_RUN = 0.20
FALLBACK_CS_RUN = -0.40

POSITION_NAMES = {
    1: "P", 2: "C", 3: "1B", 4: "2B", 5: "3B",
    6: "SS", 7: "LF", 8: "CF", 9: "RF",
}

# Runs per 600 PA relative to a neutral position.
# Kept moderate because the defense model already contains some positional context.
POSITION_ADJUSTMENT_PER_600PA = {
    "C": 10.0,
    "1B": -12.5,
    "2B": 2.5,
    "3B": 2.5,
    "SS": 7.5,
    "LF": -7.5,
    "CF": 2.5,
    "RF": -7.5,
    "DH": -17.5,
    "P": 0.0,
}

_HIT_EVENT_WEIGHT_KEY = {
    "single": "1B",
    "double": "2B",
    "triple": "3B",
    "home_run": "HR",
}

_OUT_EVENTS = {
    "field_out",
    "force_out",
    "grounded_into_double_play",
    "double_play",
    "fielders_choice_out",
    "sac_fly",
    "sac_bunt",
    "sac_fly_double_play",
}

_DEFENSE_FEATURE_COLS = ["launch_speed", "launch_angle", "hit_location"]
_DEFENSE_RAW_COLS = [
    "events", "bb_type", "hit_location", "launch_speed", "launch_angle",
    "pitcher", "fielder_2", "fielder_3", "fielder_4", "fielder_5",
    "fielder_6", "fielder_7", "fielder_8", "fielder_9",
]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _pitch_data_csv(year: int) -> Path:
    return HERE / f"pitch_data_{year}.csv"


def _resolve_retrain(explicit: bool) -> bool:
    return explicit or os.environ.get(
        "BULLPEN_RETRAIN_DEFENSE", "auto"
    ).lower() == "force"


def _safe(row, col: str) -> float:
    value = row.get(col, 0)
    return 0.0 if pd.isna(value) else float(value)


def _blend(actual: float, projected: float, sample: float, prior: float) -> float:
    """
    Blends actual performance with model projection.
    Small samples lean toward projection; large samples trust actual results.
    """
    if sample <= 0:
        return projected

    weight_actual = sample / (sample + prior)
    return weight_actual * actual + (1 - weight_actual) * projected


def _team_games_played(df: pd.DataFrame) -> dict[str, int]:
    if "game_pk" not in df.columns:
        return {}

    return {
        team: int(sub["game_pk"].nunique())
        for team, sub in df.dropna(subset=["team"]).groupby("team")
    }


def _remaining_games(
    player_games: int,
    team_games: int,
    season_games: int,
) -> float:
    """
    Smoothed playing-time projection.
    Prevents tiny samples from producing extreme participation rates.
    """
    if team_games <= 0:
        return 0.0

    remaining_team_games = max(0, season_games - team_games)

    participation = (
        player_games + PLAYING_TIME_PRIOR_GAMES * 0.35
    ) / (
        team_games + PLAYING_TIME_PRIOR_GAMES
    )

    participation = max(0.02, min(0.95, participation))

    return remaining_team_games * participation


# ---------------------------------------------------------------------
# Hitter rates and totals
# ---------------------------------------------------------------------

def _hitter_totals(group: pd.DataFrame) -> dict:
    played = group[group["actual_pa"].notna()]

    cols = {
        "H": "actual_hits",
        "HR": "actual_hr",
        "BB": "actual_walks",
        "K": "actual_strikeouts",
        "R": "actual_runs",
        "RBI": "actual_rbi",
        "2B": "actual_doubles",
        "3B": "actual_triples",
        "PA": "actual_pa",
    }

    return {
        key: float(played[col].fillna(0).sum()) if col in played else 0.0
        for key, col in cols.items()
    }


def _projected_hitter_per_game(group: pd.DataFrame) -> dict:
    """
    Uses the latest pipeline projection as the ROS talent estimate.
    Falls back to current-season rate only if projection data is unavailable.
    """
    last = group.sort_values("game_date").iloc[-1]
    played = group[group["actual_pa"].notna()]

    games = len(played)
    actual = _hitter_totals(group)
    actual_pa = actual["PA"]

    actual_rates = {
        k: actual[k] / games if games else 0.0
        for k in actual if k != "PA"
    }
    actual_rates["PA"] = actual_pa / games if games else 0.0

    proj = {
        "H": _safe(last, "proj_hits"),
        "HR": _safe(last, "proj_hr"),
        "BB": _safe(last, "proj_walks"),
        "K": _safe(last, "proj_strikeouts"),
        "R": _safe(last, "proj_runs"),
        "RBI": _safe(last, "proj_rbi"),
        "PA": _safe(last, "proj_pa"),
    }

    # Projection file currently does not have projected doubles/triples.
    # Use current rates for these, but regress strongly toward zero.
    proj["2B"] = actual_rates.get("2B", 0.0) * 0.5
    proj["3B"] = actual_rates.get("3B", 0.0) * 0.5

    if proj["PA"] <= 0:
        proj = actual_rates

    # Blend season results with projected talent using PA sample.
    rate = {}
    for key in proj:
        rate[key] = _blend(
            actual_rates.get(key, 0.0),
            proj.get(key, 0.0),
            actual_pa,
            HITTER_REGRESSION_PA,
        )

    rate["games_played"] = games
    return rate


# ---------------------------------------------------------------------
# Pitcher rates and totals
# ---------------------------------------------------------------------

def _pitcher_totals(group: pd.DataFrame) -> dict:
    played = group[group["actual_ip"].notna()]

    cols = {
        "IP": "actual_ip",
        "K": "actual_strikeouts",
        "BB": "actual_walks",
        "H": "actual_hits_allowed",
        "ER": "actual_runs_allowed",
    }

    return {
        key: float(played[col].fillna(0).sum()) if col in played else 0.0
        for key, col in cols.items()
    }


def _projected_pitcher_per_game(group: pd.DataFrame) -> dict:
    last = group.sort_values("game_date").iloc[-1]
    played = group[group["actual_ip"].notna()]

    games = len(played)
    actual = _pitcher_totals(group)
    actual_ip = actual["IP"]

    actual_rates = {
        key: actual[key] / games if games else 0.0
        for key in actual
    }

    proj = {
        "IP": _safe(last, "proj_ip"),
        "K": _safe(last, "proj_strikeouts"),
        "BB": _safe(last, "proj_walks"),
        "H": _safe(last, "proj_hits_allowed"),
        "ER": _safe(last, "proj_runs_allowed"),
    }

    if proj["IP"] <= 0:
        proj = actual_rates

    rate = {
        key: _blend(
            actual_rates.get(key, 0.0),
            proj.get(key, 0.0),
            actual_ip,
            PITCHER_REGRESSION_IP,
        )
        for key in proj
    }

    rate["games_played"] = games
    return rate


# ---------------------------------------------------------------------
# Batting WAR
# ---------------------------------------------------------------------

def _batting_runs_per_pa(rate: dict) -> float:
    pa = rate.get("PA", 0.0)
    if pa <= 0:
        return 0.0

    h = rate.get("H", 0.0)
    doubles = rate.get("2B", 0.0)
    triples = rate.get("3B", 0.0)
    hr = rate.get("HR", 0.0)
    bb = rate.get("BB", 0.0)

    singles = max(0.0, h - doubles - triples - hr)
    outs = max(0.0, pa - h - bb)

    w = BATTING_LINEAR_WEIGHTS

    return (
        w["BB"] * bb
        + w["1B"] * singles
        + w["2B"] * doubles
        + w["3B"] * triples
        + w["HR"] * hr
        + w["OUT"] * outs
    ) / pa


def _league_avg_batting_runs_per_pa(rates: list[dict]) -> float:
    total_runs = 0.0
    total_pa = 0.0

    for rate in rates:
        pa = rate.get("PA", 0.0)
        if pa <= 0:
            continue
        total_runs += _batting_runs_per_pa(rate) * pa
        total_pa += pa

    return total_runs / total_pa if total_pa else 0.0


def _position_adjustment(position: str | None, pa: float) -> float:
    return POSITION_ADJUSTMENT_PER_600PA.get(
        position or "",
        0.0,
    ) * pa / 600.0


# ---------------------------------------------------------------------
# Defense
# ---------------------------------------------------------------------

def _load_pitch_data(year: int, columns: list[str]) -> pd.DataFrame | None:
    path = _pitch_data_csv(year)

    if not path.exists():
        return None

    try:
        return pd.read_csv(path, usecols=columns, low_memory=False)
    except Exception:
        return None


def _load_multi_year_pitch_data(
    years: list[int],
    columns: list[str],
) -> pd.DataFrame | None:
    frames = [
        data
        for year in years
        if (data := _load_pitch_data(year, columns)) is not None
    ]

    return pd.concat(frames, ignore_index=True) if frames else None


def _outcome_run_value(event) -> float | None:
    if event in _HIT_EVENT_WEIGHT_KEY:
        return BATTING_LINEAR_WEIGHTS[_HIT_EVENT_WEIGHT_KEY[event]]

    if event == "field_error":
        return BATTING_LINEAR_WEIGHTS["1B"]

    if event in _OUT_EVENTS:
        return BATTING_LINEAR_WEIGHTS["OUT"]

    return None


def _prep_batted_ball_rows(df: pd.DataFrame) -> pd.DataFrame:
    bip = df[df["bb_type"].notna()].copy()
    bip["outcome_value"] = bip["events"].apply(_outcome_run_value)

    return bip.dropna(
        subset=["outcome_value", *_DEFENSE_FEATURE_COLS]
    )


def _train_defense_model(
    years: list[int],
    min_rows: int = 500,
):
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
    except Exception:
        return None

    raw = _load_multi_year_pitch_data(years, _DEFENSE_RAW_COLS)

    if raw is None:
        return None

    train = _prep_batted_ball_rows(raw)

    if len(train) < min_rows:
        return None

    model = HistGradientBoostingRegressor(
        max_depth=5,
        random_state=0,
    )

    model.fit(
        train[_DEFENSE_FEATURE_COLS],
        train["outcome_value"],
    )

    try:
        MODELS_DIR.mkdir(exist_ok=True)

        import joblib

        joblib.dump(
            {
                "model": model,
                "meta": {
                    "train_years": years,
                    "trained_at": datetime.now(ET).isoformat(),
                },
            },
            DEFENSE_MODEL_CACHE,
        )
    except Exception:
        pass

    return model


def build_defense_model(
    season_year: int,
    retrain: bool = False,
) -> tuple[dict[int, float], dict[int, str]]:

    df = _load_pitch_data(season_year, _DEFENSE_RAW_COLS)

    if df is None:
        return {}, {}

    model = None

    if not _resolve_retrain(retrain) and DEFENSE_MODEL_CACHE.exists():
        try:
            import joblib
            model = joblib.load(DEFENSE_MODEL_CACHE)["model"]
        except Exception:
            model = None

    if model is None:
        model = _train_defense_model(
            [season_year - 1, season_year]
        )

    if model is None:
        return {}, {}

    scored = _prep_batted_ball_rows(df)

    if scored.empty:
        return {}, {}

    scored["expected_value"] = model.predict(
        scored[_DEFENSE_FEATURE_COLS]
    )

    scored["fielder_run_value"] = (
        scored["expected_value"]
        - scored["outcome_value"]
    )

    def get_fielder(row):
        pos = int(row["hit_location"])
        return row.get("pitcher") if pos == 1 else row.get(f"fielder_{pos}")

    scored["fielder_id"] = scored.apply(get_fielder, axis=1)
    scored = scored.dropna(subset=["fielder_id"])

    scored["fielder_id"] = scored["fielder_id"].astype(int)
    scored["position_num"] = scored["hit_location"].astype(int)

    defense = scored.groupby("fielder_id")[
        "fielder_run_value"
    ].sum().to_dict()

    chances = scored.groupby([
        "fielder_id",
        "position_num",
    ]).size()

    positions = {}

    for player_id, sub in chances.groupby(level=0):
        pos = sub.loc[player_id].idxmax()
        positions[int(player_id)] = POSITION_NAMES.get(
            int(pos),
            "?",
        )

    return (
        {int(k): float(v) for k, v in defense.items()},
        positions,
    )


# ---------------------------------------------------------------------
# Baserunning
# ---------------------------------------------------------------------

def build_run_expectancy_and_baserunning_weights(
    season_year: int,
    retrain: bool = False,
) -> tuple[float, float, bool]:

    if (
        not _resolve_retrain(retrain)
        and BASERUNNING_WEIGHTS_CACHE.exists()
    ):
        try:
            cached = json.loads(
                BASERUNNING_WEIGHTS_CACHE.read_text()
            )

            return (
                cached["sb_run_value"],
                cached["cs_run_value"],
                cached.get("built_from_data", True),
            )
        except Exception:
            pass

    cols = [
        "game_pk", "inning", "inning_topbot",
        "at_bat_number", "events",
        "on_1b", "on_2b", "on_3b",
        "outs_when_up", "bat_score", "post_bat_score",
    ]

    df = _load_multi_year_pitch_data(
        [season_year - 1, season_year],
        cols,
    )

    if df is None:
        return FALLBACK_SB_RUN, FALLBACK_CS_RUN, False

    pa = df[df["events"].notna()].copy()

    if pa.empty:
        return FALLBACK_SB_RUN, FALLBACK_CS_RUN, False

    pa = pa.sort_values([
        "game_pk",
        "inning",
        "inning_topbot",
        "at_bat_number",
    ])

    pa["runs_on_play"] = (
        pa["post_bat_score"] - pa["bat_score"]
    ).clip(lower=0)

    pa["base_state"] = (
        pa["on_1b"].notna().astype(int).astype(str)
        + pa["on_2b"].notna().astype(int).astype(str)
        + pa["on_3b"].notna().astype(int).astype(str)
    )

    pa["runs_remaining"] = pa.groupby([
        "game_pk",
        "inning",
        "inning_topbot",
    ])["runs_on_play"].transform(
        lambda x: x.iloc[::-1].cumsum().iloc[::-1]
    )

    re = pa.groupby([
        "base_state",
        "outs_when_up",
    ])["runs_remaining"].mean()

    sb_values = []
    cs_values = []

    for outs in (0, 1, 2):
        first = re.get(("100", outs))
        second = re.get(("010", outs))
        after_cs = 0.0 if outs == 2 else re.get(("000", outs + 1))

        if pd.notna(first) and pd.notna(second):
            sb_values.append(second - first)

        if pd.notna(first) and pd.notna(after_cs):
            cs_values.append(after_cs - first)

    if not sb_values or not cs_values:
        return FALLBACK_SB_RUN, FALLBACK_CS_RUN, False

    sb = float(sum(sb_values) / len(sb_values))
    cs = float(sum(cs_values) / len(cs_values))

    try:
        MODELS_DIR.mkdir(exist_ok=True)

        BASERUNNING_WEIGHTS_CACHE.write_text(
            json.dumps(
                {
                    "sb_run_value": sb,
                    "cs_run_value": cs,
                    "built_from_data": True,
                    "train_years": [
                        season_year - 1,
                        season_year,
                    ],
                },
                indent=2,
            )
        )
    except Exception:
        pass

    return sb, cs, True


def fetch_baserunning(
    ids: set[int],
    season: int,
) -> dict[int, dict]:

    out = {}

    for player_id in ids:
        url = (
            f"https://statsapi.mlb.com/api/v1/people/{player_id}/stats"
            f"?stats=season&group=hitting&season={season}"
        )

        try:
            data = requests.get(
                url,
                timeout=8,
            ).json()

            splits = (
                (data.get("stats") or [{}])[0]
                .get("splits")
                or []
            )

            if splits:
                stat = splits[0].get("stat", {})

                out[player_id] = {
                    "sb": int(stat.get("stolenBases") or 0),
                    "cs": int(stat.get("caughtStealing") or 0),
                }

        except Exception:
            pass

        time.sleep(0.03)

    return out


# ---------------------------------------------------------------------
# Pitcher WAR
# ---------------------------------------------------------------------

def _ra9(stats: dict) -> float | None:
    ip = stats.get("IP", 0.0)

    return stats.get("ER", 0.0) / ip * 9 if ip > 0 else None


def _league_avg_ra9(rates: list[dict]) -> float:
    ip = sum(x.get("IP", 0.0) for x in rates)
    runs = sum(x.get("ER", 0.0) for x in rates)

    return runs / ip * 9 if ip else 0.0


def _pitching_runs(stats: dict, league_ra9: float) -> float:
    ip = stats.get("IP", 0.0)
    ra9 = _ra9(stats)

    if not ip or ra9 is None:
        return 0.0

    return (league_ra9 - ra9) * ip / 9


def _pitching_replacement(stats: dict, league_ra9: float) -> float:
    return (
        league_ra9
        * (PITCHER_REPLACEMENT_RA9_MULTIPLIER - 1)
        * stats.get("IP", 0.0)
        / 9
    )


def _pitcher_points(stats: dict) -> float:
    return sum(
        stats.get(k, 0.0) * weight
        for k, weight in PITCHER_WEIGHTS.items()
    )


# ---------------------------------------------------------------------
# Ages
# ---------------------------------------------------------------------

def _load_age_cache() -> dict:
    try:
        return json.loads(AGE_CACHE.read_text())
    except Exception:
        return {}


def _save_age_cache(cache: dict) -> None:
    AGE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    AGE_CACHE.write_text(json.dumps(cache, indent=2))


def get_player_ages(ids: set[int]) -> dict[int, dict]:
    cache = _load_age_cache()
    out = {}

    for player_id in ids:
        key = str(player_id)

        if key in cache:
            out[player_id] = cache[key]
            continue

        try:
            data = requests.get(
                f"https://statsapi.mlb.com/api/v1/people/{player_id}",
                timeout=8,
            ).json()

            person = (data.get("people") or [])[0]
            birth = person.get("birthDate")

            if birth:
                bd = datetime.strptime(birth, "%Y-%m-%d")
                today = datetime.now()

                age = (
                    today.year - bd.year
                    - ((today.month, today.day) < (bd.month, bd.day))
                )

                out[player_id] = {"age": age}
                cache[key] = out[player_id]

        except Exception:
            pass

    _save_age_cache(cache)
    return out


def _attach_ages(rows: list[dict], top_n: int) -> None:
    ids = {
        r["mlb_id"]
        for r in rows[:top_n]
        if r.get("mlb_id") is not None
    }

    ages = get_player_ages(ids)

    for row in rows:
        row["age"] = (
            ages.get(row.get("mlb_id"), {}).get("age")
        )


# ---------------------------------------------------------------------
# Main rankings
# ---------------------------------------------------------------------

def build_rankings(
    season_games: int = DEFAULT_SEASON_GAMES,
    as_of: str | None = None,
    fetch_ages: bool = True,
    fetch_defense: bool = True,
    fetch_baserunning_data: bool = True,
    retrain_models: bool = False,
) -> dict:

    if not ACC_CSV.exists():
        raise FileNotFoundError(ACC_CSV)

    df = pd.read_csv(ACC_CSV, low_memory=False)
    df["game_date"] = pd.to_datetime(
        df["game_date"],
        errors="coerce",
    )

    if as_of:
        df = df[
            df["game_date"] <= pd.to_datetime(as_of)
        ]

    team_games = _team_games_played(df)

    season_year = (
        int(df["game_date"].dt.year.max())
        if df["game_date"].notna().any()
        else datetime.now().year
    )

    # ================================================================
    # HITERS
    # ================================================================

    hitters = df[
        df["player_type"].astype(str).str.lower() == "hitter"
    ]

    prelim = []

    for (mid, name), group in hitters.groupby([
        "mlb_id",
        "player_name",
    ]):

        team = (
            str(group["team"].dropna().iloc[-1])
            if group["team"].notna().any()
            else ""
        )

        rate = _projected_hitter_per_game(group)
        gp = rate.pop("games_played")
        totals = _hitter_totals(group)

        team_gp = team_games.get(team, max(gp, 1))
        gr = _remaining_games(
            gp,
            team_gp,
            season_games,
        )

        ros = {
            key: value * gr
            for key, value in rate.items()
        }

        full = {
            key: totals.get(key, 0.0) + ros.get(key, 0.0)
            for key in rate
        }

        prelim.append({
            "mlb_id": int(mid),
            "name": name,
            "team": team,
            "gp": gp,
            "gr": gr,
            "totals": totals,
            "rate": rate,
            "ros": ros,
            "full": full,
        })

    league_bat_rate = _league_avg_batting_runs_per_pa(
        [p["rate"] for p in prelim]
    )

    for p in prelim:
        ros_rate = _batting_runs_per_pa(p["rate"])

        actual_rate = _batting_runs_per_pa(p["totals"])

        p["ros_bat_runs"] = (
            ros_rate - league_bat_rate
        ) * p["ros"]["PA"]

        # Actual season value uses actual stats.
        # Future value uses projected/blended ROS rate.
        p["full_bat_runs"] = (
            (actual_rate - league_bat_rate)
            * p["totals"]["PA"]
            + p["ros_bat_runs"]
        )

    prelim.sort(
        key=lambda x: x["ros_bat_runs"],
        reverse=True,
    )

    baserunning_data = {}
    sb_value = FALLBACK_SB_RUN
    cs_value = FALLBACK_CS_RUN
    re_matrix_built = False

    if fetch_baserunning_data:
        ids = {
            p["mlb_id"]
            for p in prelim[:TOP_N_HITTERS_FOR_BASERUNNING]
        }

        baserunning_data = fetch_baserunning(
            ids,
            season_year,
        )

        sb_value, cs_value, re_matrix_built = (
            build_run_expectancy_and_baserunning_weights(
                season_year,
                retrain_models,
            )
        )

    defense = {}
    positions = {}

    if fetch_defense:
        defense, positions = build_defense_model(
            season_year,
            retrain_models,
        )

    hitter_rows = []

    for p in prelim:
        mid = p["mlb_id"]
        gp = p["gp"]
        gr = p["gr"]
        ros = p["ros"]
        full = p["full"]
        totals = p["totals"]

        position = positions.get(mid)

        # Regress current defensive value heavily toward zero.
        season_def = defense.get(mid, 0.0)

        def_per_game = (
            season_def / (gp + DEFENSE_REGRESSION_GAMES)
            if gp > 0
            else 0.0
        )

        ros_def = def_per_game * gr
        full_def = season_def + ros_def

        # Regress SB/CS rates.
        br = baserunning_data.get(mid, {})

        sb = br.get("sb", 0)
        cs = br.get("cs", 0)

        if gp > 0:
            sb_rate = sb / (gp + BASERUNNING_REGRESSION_GAMES)
            cs_rate = cs / (gp + BASERUNNING_REGRESSION_GAMES)

            ros_bsr = (
                sb_value * sb_rate
                + cs_value * cs_rate
            ) * gr

            full_bsr = (
                sb_value * sb
                + cs_value * cs
                + ros_bsr
            )
        else:
            ros_bsr = 0.0
            full_bsr = 0.0

        ros_pos = _position_adjustment(
            position,
            ros["PA"],
        )

        full_pos = _position_adjustment(
            position,
            full["PA"],
        )

        ros_replacement = (
            REPLACEMENT_RUNS_PER_600PA
            * ros["PA"] / 600
        )

        full_replacement = (
            REPLACEMENT_RUNS_PER_600PA
            * full["PA"] / 600
        )

        ros_war = (
            p["ros_bat_runs"]
            + ros_def
            + ros_bsr
            + ros_pos
            + ros_replacement
        ) / RUNS_PER_WIN

        full_war = (
            p["full_bat_runs"]
            + full_def
            + full_bsr
            + full_pos
            + full_replacement
        ) / RUNS_PER_WIN

        hitter_rows.append({
            "mlb_id": mid,
            "name": p["name"],
            "team": p["team"],
            "position": position,
            "games_played": gp,
            "games_remaining": round(gr, 1),

            "ros_h": round(ros["H"], 1),
            "full_h": round(full["H"], 1),
            "ros_2b": round(ros["2B"], 1),
            "full_2b": round(full["2B"], 1),
            "ros_3b": round(ros["3B"], 1),
            "full_3b": round(full["3B"], 1),
            "ros_hr": round(ros["HR"], 1),
            "full_hr": round(full["HR"], 1),
            "ros_r": round(ros["R"], 1),
            "full_r": round(full["R"], 1),
            "ros_rbi": round(ros["RBI"], 1),
            "full_rbi": round(full["RBI"], 1),
            "ros_bb": round(ros["BB"], 1),
            "full_bb": round(full["BB"], 1),
            "ros_k": round(ros["K"], 1),
            "full_k": round(full["K"], 1),
            "ros_pa": round(ros["PA"], 1),
            "full_pa": round(full["PA"], 1),

            "ros_bat_runs": round(p["ros_bat_runs"], 1),
            "full_bat_runs": round(p["full_bat_runs"], 1),
            "ros_def_runs": round(ros_def, 1),
            "full_def_runs": round(full_def, 1),
            "ros_bsr_runs": round(ros_bsr, 1),
            "full_bsr_runs": round(full_bsr, 1),
            "ros_pos_runs": round(ros_pos, 1),
            "full_pos_runs": round(full_pos, 1),
            "ros_replacement_runs": round(ros_replacement, 1),
            "full_replacement_runs": round(full_replacement, 1),

            "ros_war": round(ros_war, 1),
            "full_war": round(full_war, 1),

            "has_baserunning_data": mid in baserunning_data,
            "has_defense_data": mid in defense,
        })

    hitter_rows.sort(
        key=lambda x: x["ros_war"],
        reverse=True,
    )

    for rank, row in enumerate(hitter_rows, 1):
        row["rank"] = rank

    # ================================================================
    # PITCHERS
    # ================================================================

    pitchers = df[
        df["player_type"].astype(str).str.lower() == "pitcher"
    ]

    prelim_p = []

    for (mid, name), group in pitchers.groupby([
        "mlb_id",
        "player_name",
    ]):

        team = (
            str(group["team"].dropna().iloc[-1])
            if group["team"].notna().any()
            else ""
        )

        rate = _projected_pitcher_per_game(group)
        gp = rate.pop("games_played")
        totals = _pitcher_totals(group)

        team_gp = team_games.get(team, max(gp, 1))

        gr = _remaining_games(
            gp,
            team_gp,
            season_games,
        )

        ros = {
            key: value * gr
            for key, value in rate.items()
        }

        full = {
            key: totals.get(key, 0.0) + ros.get(key, 0.0)
            for key in rate
        }

        prelim_p.append({
            "mlb_id": int(mid),
            "name": name,
            "team": team,
            "gp": gp,
            "gr": gr,
            "rate": rate,
            "ros": ros,
            "full": full,
        })

    league_ra9 = _league_avg_ra9(
        [p["rate"] for p in prelim_p]
    )

    pitcher_rows = []

    for p in prelim_p:
        ros = p["ros"]
        full = p["full"]

        ros_pitching_runs = _pitching_runs(
            ros,
            league_ra9,
        )

        full_pitching_runs = _pitching_runs(
            full,
            league_ra9,
        )

        ros_replacement = _pitching_replacement(
            ros,
            league_ra9,
        )

        full_replacement = _pitching_replacement(
            full,
            league_ra9,
        )

        ros_war = (
            ros_pitching_runs + ros_replacement
        ) / RUNS_PER_WIN

        full_war = (
            full_pitching_runs + full_replacement
        ) / RUNS_PER_WIN

        pitcher_rows.append({
            "mlb_id": p["mlb_id"],
            "name": p["name"],
            "team": p["team"],
            "starts_made": p["gp"],
            "starts_remaining": round(p["gr"], 1),

            "ros_ip": round(ros["IP"], 1),
            "full_ip": round(full["IP"], 1),
            "ros_k": round(ros["K"], 1),
            "full_k": round(full["K"], 1),
            "ros_bb": round(ros["BB"], 1),
            "full_bb": round(full["BB"], 1),
            "ros_h": round(ros["H"], 1),
            "full_h": round(full["H"], 1),
            "ros_er": round(ros["ER"], 1),
            "full_er": round(full["ER"], 1),

            "ros_ra9": (
                round(_ra9(ros), 2)
                if _ra9(ros) is not None
                else None
            ),

            "full_ra9": (
                round(_ra9(full), 2)
                if _ra9(full) is not None
                else None
            ),

            "ros_pitching_runs": round(
                ros_pitching_runs,
                1,
            ),

            "full_pitching_runs": round(
                full_pitching_runs,
                1,
            ),

            "ros_replacement_runs": round(
                ros_replacement,
                1,
            ),

            "full_replacement_runs": round(
                full_replacement,
                1,
            ),

            "ros_war": round(ros_war, 1),
            "full_war": round(full_war, 1),
            "ros_power_score": round(
                _pitcher_points(ros),
                1,
            ),

            "full_power_score": round(
                _pitcher_points(full),
                1,
            ),
        })

    pitcher_rows.sort(
        key=lambda x: x["ros_war"],
        reverse=True,
    )

    for rank, row in enumerate(pitcher_rows, 1):
        row["rank"] = rank

    # ================================================================
    # AGES
    # ================================================================

    if fetch_ages:
        _attach_ages(
            hitter_rows,
            TOP_N_HITTERS_FOR_AGE,
        )

        _attach_ages(
            pitcher_rows,
            TOP_N_PITCHERS_FOR_AGE,
        )

    else:
        for row in hitter_rows + pitcher_rows:
            row["age"] = None

    return {
        "as_of": (
            as_of
            or datetime.now(ET).strftime("%Y-%m-%d")
        ),

        "season_games": season_games,
        "season_year": season_year,
        "age_cutoff": AGE_CUTOFF,

        "data_availability": {
            "defense": bool(defense),
            "baserunning": bool(baserunning_data),
            "baserunning_weights_from_own_data": (
                re_matrix_built
            ),
        },

        "scoring": {
            "type": "war_v4_projected",

            "hitter": {
                "model": (
                    "Projected Bat + Regressed Def + "
                    "Regressed BsR + Position + Replacement"
                ),

                "runs_per_win": RUNS_PER_WIN,

                "replacement_runs_per_600pa": (
                    REPLACEMENT_RUNS_PER_600PA
                ),

                "hitter_regression_pa": (
                    HITTER_REGRESSION_PA
                ),

                "defense_regression_games": (
                    DEFENSE_REGRESSION_GAMES
                ),

                "baserunning_regression_games": (
                    BASERUNNING_REGRESSION_GAMES
                ),

                "batting_linear_weights": (
                    BATTING_LINEAR_WEIGHTS
                ),

                "baserunning_weights_used": {
                    "SB": round(sb_value, 3),
                    "CS": round(cs_value, 3),
                },
            },

            "pitcher": {
                "model": (
                    "(RA9 runs above average + replacement) / runs per win"
                ),

                "league_ra9_used": round(
                    league_ra9,
                    3,
                ),

                "replacement_ra9_multiplier": (
                    PITCHER_REPLACEMENT_RA9_MULTIPLIER
                ),

                "runs_per_win": RUNS_PER_WIN,
            },
        },

        "team_games_played": team_games,
        "hitters": hitter_rows,
        "pitchers": pitcher_rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--season-games",
        type=int,
        default=DEFAULT_SEASON_GAMES,
    )

    ap.add_argument("--as-of")
    ap.add_argument("--out", default=str(OUT_JSON))
    ap.add_argument("--no-ages", action="store_true")
    ap.add_argument("--no-defense", action="store_true")
    ap.add_argument("--no-baserunning", action="store_true")
    ap.add_argument(
        "--retrain-defense",
        action="store_true",
    )

    args = ap.parse_args()

    bundle = build_rankings(
        season_games=args.season_games,
        as_of=args.as_of,
        fetch_ages=not args.no_ages,
        fetch_defense=not args.no_defense,
        fetch_baserunning_data=not args.no_baserunning,
        retrain_models=args.retrain_defense,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w") as f:
        json.dump(bundle, f, indent=2)

    print(f"Wrote -> {out}")
    print(
        f"{len(bundle['hitters'])} hitters / "
        f"{len(bundle['pitchers'])} pitchers"
    )

    print("\nTop 10 hitters:")
    for r in bundle["hitters"][:10]:
        print(
            f"{r['rank']:>2}. "
            f"{r['name']:<25} "
            f"{r['team']:<4} "
            f"ROS {r['ros_war']:>5.1f} "
            f"FULL {r['full_war']:>5.1f}"
        )

    print("\nTop 10 pitchers:")
    for r in bundle["pitchers"][:10]:
        print(
            f"{r['rank']:>2}. "
            f"{r['name']:<25} "
            f"{r['team']:<4} "
            f"ROS {r['ros_war']:>5.1f} "
            f"FULL {r['full_war']:>5.1f}"
        )


if __name__ == "__main__":
    main()
