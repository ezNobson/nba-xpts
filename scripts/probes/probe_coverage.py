"""Check whether a league-wide shotchartdetail call returns the full season."""
from nba_api.stats.endpoints import shotchartdetail

NOP = 1610612740


def fetch(team_id):
    return shotchartdetail.ShotChartDetail(
        team_id=team_id, player_id=0, season_nullable="2014-15",
        season_type_all_star="Regular Season",
        context_measure_simple="FGA", timeout=120,
    ).get_data_frames()[0]


league = fetch(0)
print("league rows:", len(league))
print("games:", league["GAME_ID"].nunique(), "(expected 1230)")
print("dates:", league["GAME_DATE"].min(), "->", league["GAME_DATE"].max())
print("shots per game:", round(len(league) / league["GAME_ID"].nunique(), 1), "(expected ~166)")
print("teams:", league["TEAM_ID"].nunique())

team = fetch(NOP)
print("NOP in league call:", (league["TEAM_ID"] == NOP).sum())
print("NOP team call:     ", len(team))
