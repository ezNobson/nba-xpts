"""Download the listed position of every NBA player (one row per player, nba_api PlayerIndex).

The position is the one listed by NBA.com for the player's career (G, F, C or a combination such as G-F),
not per season. The primary position is its first letter.
"""
import random
import time
from pathlib import Path

from nba_api.stats.endpoints import playerindex

OUT = Path("data/raw/players/positions.parquet")
MAX_RETRIES = 5


def fetch():
    for attempt in range(MAX_RETRIES):
        try:
            df = playerindex.PlayerIndex(season="2025-26", historical_nullable=1, timeout=60).get_data_frames()[0]
            if df.empty:
                raise ValueError("empty response")
            return df
        except Exception as e:
            wait = 5 * 2 ** attempt + random.uniform(0, 2)
            print(f"{type(e).__name__}, retry in {wait:.0f}s")
            time.sleep(wait)
    raise RuntimeError("giving up")


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df = fetch()
    df = df.rename(columns={"PERSON_ID": "PLAYER_ID"})[
        ["PLAYER_ID", "PLAYER_FIRST_NAME", "PLAYER_LAST_NAME", "POSITION", "HEIGHT", "WEIGHT", "FROM_YEAR", "TO_YEAR"]]
    df["POS"] = df["POSITION"].str[0]  # primary position: G, F or C
    df.to_parquet(OUT, index=False)
    print(f"{len(df)} players, primary position: {df['POS'].value_counts(dropna=False).to_dict()}")


if __name__ == "__main__":
    main()
