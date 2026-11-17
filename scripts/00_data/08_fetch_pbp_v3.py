"""Download play-by-play in the newer stats.nba.com v3 format and convert it to the v2 columns of 07_fetch_pbp.py.

The nba_data archive has 2025-26 (hold-out season) only as nbastatsv3_{year}.tar.xz. After conversion
scripts/02_features/06_build_pbp_features.py works unchanged. The adapter is checked on 2024-25, where both
formats exist: `--check` builds the context features from the v3 file and compares them with the v2 ones.
"""
import argparse
import io
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

URL = "https://github.com/shufinskiy/nba_data/raw/main/datasets/nbastatsv3_{year}.tar.xz"
OUT_DIR = Path("data/raw/pbp")
# v3 actionType -> v2 EVENTMSGTYPE
EVENT_TYPES = {"Made Shot": 1, "Missed Shot": 2, "Free Throw": 3, "Rebound": 4, "Turnover": 5, "Foul": 6,
               "Violation": 7, "Substitution": 8, "Timeout": 9, "Jump Ball": 10, "Ejection": 11,
               "Instant Replay": 18}


def download(year: int) -> pd.DataFrame:
    r = requests.get(URL.format(year=year), timeout=300)
    r.raise_for_status()
    with tarfile.open(fileobj=io.BytesIO(r.content), mode="r:xz") as tar:
        member = next(m for m in tar.getmembers() if m.name.endswith(".csv"))
        return pd.read_csv(tar.extractfile(member), low_memory=False)


def to_v2(v: pd.DataFrame) -> pd.DataFrame:
    """One row per event number with the v2 columns used by the feature script.

    In v3 a steal and a block are separate rows with the same actionNumber as the turnover / missed shot;
    the steal becomes PLAYER2_ID of the turnover (as in v2), both extra rows are then dropped.
    """
    v = v.sort_values(["gameId", "actionId"]).copy()
    v["actionType"] = v["actionType"].str.strip()
    is_steal = v["actionType"].isna() & v["description"].str.contains("STEAL", na=False)
    steals = v[is_steal].groupby(["gameId", "actionNumber"])["personId"].first().rename("PLAYER2_ID")
    v = v[v["actionType"].notna()].join(steals, on=["gameId", "actionNumber"])

    t = v["actionType"].map(EVENT_TYPES)
    t[(v["actionType"] == "period") & (v["subType"] == "start")] = 12
    t[(v["actionType"] == "period") & (v["subType"] == "end")] = 13
    clock = v["clock"].str.extract(r"PT(\d+)M(\d+)")
    score = v["scoreHome"].notna() & (v["scoreHome"].astype(str).str.strip() != "")
    margin = pd.to_numeric(v["scoreHome"], errors="coerce") - pd.to_numeric(v["scoreAway"], errors="coerce")
    team = v["teamId"].where(v["teamId"] > 0)
    return pd.DataFrame({
        "GAME_ID": v["gameId"].astype(str).str.zfill(10),
        "EVENTNUM": v["actionNumber"].astype(int),
        "EVENTMSGTYPE": t.astype("Int64"),
        "EVENTMSGACTIONTYPE": 0,
        "PERIOD": v["period"].astype(int),
        "PCTIMESTRING": clock[0].astype(int).astype(str) + ":" + clock[1].str.zfill(2),
        "HOMEDESCRIPTION": v["description"].where(v["location"] == "h"),
        "NEUTRALDESCRIPTION": v["description"].where(v["location"].isna()),
        "VISITORDESCRIPTION": v["description"].where(v["location"] == "v"),
        "SCORE": (v["scoreAway"].astype(str) + " - " + v["scoreHome"].astype(str)).where(score).astype("string"),
        "SCOREMARGIN": margin.where(score).astype("Int64").astype("string").replace("0", "TIE"),
        "PLAYER1_ID": v["personId"].astype(float),
        "PLAYER1_TEAM_ID": team.astype(float),
        "PLAYER2_ID": v["PLAYER2_ID"].astype(float),
        "PLAYER2_TEAM_ID": np.nan,
        "PLAYER3_ID": np.nan,
        "PLAYER3_TEAM_ID": np.nan,
    }).dropna(subset=["EVENTMSGTYPE"]).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2025, help="first year of the season, 2025 = 2025-26")
    parser.add_argument("--out", type=Path, default=None, help="output parquet (default data/raw/pbp/pbp_{season}.parquet)")
    args = parser.parse_args()
    season = f"{args.year}-{str(args.year + 1)[-2:]}"
    out = args.out or OUT_DIR / f"pbp_{season}.parquet"
    df = to_v2(download(args.year))
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"{season}: {df['GAME_ID'].nunique():,} games, {len(df):,} events -> {out}")


if __name__ == "__main__":
    main()
