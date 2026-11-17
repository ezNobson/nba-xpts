"""Download regular-season play-by-play (stats.nba.com format) from the shufinskiy/nba_data archive.

One tar.xz per season (https://github.com/shufinskiy/nba_data, Apache-2.0), saved as one parquet per season.
Events join to shots by GAME_ID + EVENTNUM (= GAME_EVENT_ID in shotchartdetail).
The hold-out season 2025-26 is not in the archive and is not needed before the final evaluation.
"""
import io
import tarfile
import time
from pathlib import Path

import pandas as pd
import requests

URL = "https://github.com/shufinskiy/nba_data/raw/main/datasets/nbastats_{year}.tar.xz"
YEARS = range(2014, 2025)  # 2014-15 .. 2024-25
OUT_DIR = Path("data/raw/pbp")
COLS = ["GAME_ID", "EVENTNUM", "EVENTMSGTYPE", "EVENTMSGACTIONTYPE", "PERIOD", "PCTIMESTRING",
        "HOMEDESCRIPTION", "NEUTRALDESCRIPTION", "VISITORDESCRIPTION", "SCORE", "SCOREMARGIN",
        "PLAYER1_ID", "PLAYER1_TEAM_ID", "PLAYER2_ID", "PLAYER2_TEAM_ID", "PLAYER3_ID", "PLAYER3_TEAM_ID"]
MAX_RETRIES = 4


def fetch_season(year: int) -> pd.DataFrame:
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(URL.format(year=year), timeout=120)
            r.raise_for_status()
            break
        except requests.RequestException as e:
            wait = 5 * 2 ** attempt
            print(f"{year}: {type(e).__name__}, retry in {wait}s")
            time.sleep(wait)
    else:
        raise RuntimeError(f"giving up on {year}")
    # read the csv straight from the archive, nothing is extracted to disk
    with tarfile.open(fileobj=io.BytesIO(r.content), mode="r:xz") as tar:
        member = next(m for m in tar.getmembers() if m.name.endswith(".csv"))
        df = pd.read_csv(tar.extractfile(member), usecols=COLS, dtype={"SCOREMARGIN": "string", "SCORE": "string"})
    df["GAME_ID"] = df["GAME_ID"].astype(str).str.zfill(10)  # same format as in the shot data
    return df


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for year in YEARS:
        season = f"{year}-{str(year + 1)[-2:]}"
        out = OUT_DIR / f"pbp_{season}.parquet"
        if out.exists():
            print(f"{season}: already downloaded")
            continue
        df = fetch_season(year)
        df.to_parquet(out, index=False)
        print(f"{season}: {df['GAME_ID'].nunique():,} games, {len(df):,} events")


if __name__ == "__main__":
    main()
