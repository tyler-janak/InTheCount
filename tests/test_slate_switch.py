"""The site shows the slate dated today (ET): the day-of file once it exists,
otherwise last evening's next-day file - never yesterday's. Official MLB
lineups are parsed per game and date."""
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

import hitterspitchers_today as ht
import server


def _write(p, d):
    pd.DataFrame({"game_date": [d], "home_team": ["NYY"], "away_team": ["BOS"]}).to_csv(p, index=False)


def test_slate_path_prefers_file_dated_today(tmp_path):
    today = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
    yday = (pd.Timestamp(today) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    t, n = tmp_path / "today.csv", tmp_path / "next.csv"
    _write(t, yday); _write(n, today)
    assert server._slate_path(t, n) == n            # morning run not landed yet -> next-day file
    _write(t, today)
    assert server._slate_path(t, n) == t            # morning run landed -> day-of file
    n.unlink(); _write(t, yday)
    assert server._slate_path(t, n) == t            # nothing better -> fall back (footer flags it stale)


def test_parse_mlb_lineups():
    js = {"dates": [{"games": [{"gamePk": 1, "teams": {
        "away": {"team": {"abbreviation": "BOS"}}, "home": {"team": {"abbreviation": "NYY"}}},
        "lineups": {"awayPlayers": [{"id": 10, "fullName": "A One", "primaryPosition": {"abbreviation": "CF"}},
                                    {"id": 11, "fullName": "A Two", "primaryPosition": {"abbreviation": "SS"}}],
                    "homePlayers": []}}]}]}
    df = ht.parse_mlb_lineups(js)
    assert list(df["lineup_spot"]) == [1, 2]
    assert set(df["team"]) == {ht.team_to_abbr("BOS")} and set(df["opponent"]) == {ht.team_to_abbr("NYY")}
    assert list(df["mlb_id"]) == [10, 11] and (df["lineup_status"] == "Confirmed Lineup").all()
