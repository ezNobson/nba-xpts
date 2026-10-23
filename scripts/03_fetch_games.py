"""Download regular-season team box scores (one row per team per game) for each season."""
import random
import time
from pathlib import Path

from nba_api.stats.endpoints import leaguegamelog

SEASONS = [f"{y}-{str(y + 1)[-2:]}" for y in range(2014, 2026)]  # 2014-15 .. 2025-26
OUT_DIR = Path("data/raw/games")
MAX_RETRIES = 5


def fetch_season(season):
    for attempt in range(MAX_RETRIES):
        try:
            df = leaguegamelog.LeagueGameLog(
                season=season,
                season_type_all_star="Regular Season",
                player_or_team_abbreviation="T",  # team-level rows
                timeout=60,
            ).get_data_frames()[0]
            if df.empty:
                raise ValueError("empty response")
            return df
        except Exception as e:
            wait = 5 * 2 ** attempt + random.uniform(0, 2)
            print(f"{season}: {type(e).__name__}, retry in {wait:.0f}s")
            time.sleep(wait)
    raise RuntimeError(f"giving up on {season}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for season in SEASONS:
        path = OUT_DIR / f"games_{season}.parquet"
        if path.exists():
            continue
        df = fetch_season(season)
        df["SEASON"] = season
        df.to_parquet(path, index=False)
        print(f"{season}: {len(df)} team-games, {df['GAME_ID'].nunique()} games")
        time.sleep(1.0 + random.uniform(0, 0.5))


if __name__ == "__main__":
    main()
