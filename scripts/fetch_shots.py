"""Download all regular-season shots (FGA) from stats.nba.com, one file per team-season.

League-wide calls (team_id=0) are silently truncated at ~100k rows,
so each team is queried separately. Re-running skips finished files.
"""
import random
import time
from pathlib import Path

from nba_api.stats.endpoints import shotchartdetail
from nba_api.stats.static import teams
from tqdm import tqdm

SEASONS = [f"{y}-{str(y + 1)[-2:]}" for y in range(2014, 2026)]  # 2014-15 .. 2025-26
OUT_DIR = Path("data/raw/shots")
SLEEP_S = 1.0
MAX_RETRIES = 5


def fetch_team_season(team_id, season):
    for attempt in range(MAX_RETRIES):
        try:
            df = shotchartdetail.ShotChartDetail(
                team_id=team_id, player_id=0, season_nullable=season,
                season_type_all_star="Regular Season",
                context_measure_simple="FGA",  # default "PTS" = made shots only
                timeout=60,
            ).get_data_frames()[0]
            if df.empty:
                raise ValueError("empty response")
            return df
        except Exception as e:
            wait = 5 * 2 ** attempt + random.uniform(0, 2)  # exponential backoff
            tqdm.write(f"{season} {team_id}: {type(e).__name__}, retry in {wait:.0f}s")
            time.sleep(wait)
    raise RuntimeError(f"giving up on {season} {team_id}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    team_ids = sorted(t["id"] for t in teams.get_teams())
    jobs = [(s, t) for s in SEASONS for t in team_ids]

    for season, team_id in tqdm(jobs, desc="team-seasons"):
        path = OUT_DIR / f"shots_{season}_{team_id}.parquet"
        if path.exists():
            continue
        df = fetch_team_season(team_id, season)
        df["SEASON"] = season
        tmp = path.with_suffix(".tmp")
        df.to_parquet(tmp, index=False)
        tmp.replace(path)  # atomic: no half-written files after a crash
        time.sleep(SLEEP_S + random.uniform(0, 0.5))  # be polite to the API


if __name__ == "__main__":
    main()
