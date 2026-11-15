"""Download regular-season pull-up shot totals per team from NBA tracking data (camera-based, not play-by-play labels)."""
import random
import time
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import leaguedashptstats

SEASONS = [f"{y}-{str(y + 1)[-2:]}" for y in range(2014, 2025)]  # 2014-15 .. 2024-25, hold-out not needed
OUT_PATH = Path("data/raw/tracking/pullup_team.parquet")
MAX_RETRIES = 5


def fetch_season(season):
    for attempt in range(MAX_RETRIES):
        try:
            df = leaguedashptstats.LeagueDashPtStats(
                season=season,
                season_type_all_star="Regular Season",
                pt_measure_type="PullUpShot",
                player_or_team="Team",
                per_mode_simple="Totals",
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
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    frames = []
    for season in SEASONS:
        df = fetch_season(season).assign(SEASON=season)
        frames.append(df)
        print(f"{season}: {df['PULL_UP_FGA'].sum():,} pull-up FGA")
        time.sleep(1.0 + random.uniform(0, 0.5))
    pd.concat(frames, ignore_index=True).to_parquet(OUT_PATH, index=False)


if __name__ == "__main__":
    main()
