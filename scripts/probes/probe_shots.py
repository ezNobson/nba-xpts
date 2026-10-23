"""One-off probe: fetch all league shots for a single season and inspect them."""
import time

from nba_api.stats.endpoints import shotchartdetail

t0 = time.time()
resp = shotchartdetail.ShotChartDetail(
    team_id=0,                    # 0 = all teams
    player_id=0,                  # 0 = all players
    season_nullable="2014-15",
    season_type_all_star="Regular Season",
    context_measure_simple="FGA",  # default "PTS" returns made shots only!
    timeout=120,
)
df = resp.get_data_frames()[0]
print(f"rows={len(df):,}  time={time.time() - t0:.1f}s")
print(df.columns.tolist())
print(df["SHOT_MADE_FLAG"].value_counts(normalize=True))
print(df.head(3).T)
