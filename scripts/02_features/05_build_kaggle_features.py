"""Join Kaggle tracking context (defender, shot clock, dribbles, touch time) to matched nba_api shots.

Only per-row cleaning rules are used (no statistics fitted on data). Location, shot value and the label
come from nba_api; Kaggle outcome columns are not loaded.
"""
from pathlib import Path

import pandas as pd

pd.set_option("display.width", 250)
OUT = Path("data/processed/kaggle_features.parquet")

match = pd.read_parquet("data/processed/kaggle_nba_match.parquet")
nba = pd.read_parquet("data/processed/shots_features.parquet")
kg = pd.read_parquet(
    "data/interim/kaggle_shot_logs.parquet",
    columns=["GAME_CLOCK", "SHOT_CLOCK", "DRIBBLES", "TOUCH_TIME", "CLOSE_DEF_DIST", "CLOSEST_DEFENDER_PLAYER_ID"],
)
kg["k_row"] = range(len(kg))
mins, secs = kg["GAME_CLOCK"].str.split(":", expand=True).astype(int).T.values
kg["kaggle_secs_left"] = mins * 60 + secs

df = (match[["k_row", "GAME_ID", "GAME_EVENT_ID"]]
      .merge(nba, on=["GAME_ID", "GAME_EVENT_ID"], how="left", validate="one_to_one")
      .merge(kg.drop(columns="GAME_CLOCK"), on="k_row", how="left", validate="one_to_one"))

# shot clock is turned off when less than 24 s are left in the period
df["shot_clock_off"] = (df["SHOT_CLOCK"].isna() & df["kaggle_secs_left"].lt(24)).astype(int)
df["shot_clock"] = df["SHOT_CLOCK"].mask(df["shot_clock_off"].eq(1), df["kaggle_secs_left"])
df["shot_clock_missing"] = df["shot_clock"].isna().astype(int)
df["touch_time"] = df["TOUCH_TIME"].mask(df["TOUCH_TIME"].lt(0))  # negative values are recording errors
df["close_def_dist"] = df["CLOSE_DEF_DIST"]
df["dribbles"] = df["DRIBBLES"]
df["is_catch_shoot"] = (df["dribbles"].eq(0) & df["touch_time"].le(2)).astype(int)
df = df.drop(columns=["SHOT_CLOCK", "TOUCH_TIME", "CLOSE_DEF_DIST", "DRIBBLES"])

df.to_parquet(OUT, index=False)
print("saved:", OUT, df.shape)
print("\ndtypes and NaN counts:")
print(pd.DataFrame({"dtype": df.dtypes.astype(str), "n_nan": df.isna().sum()}).to_string())

new = ["close_def_dist", "shot_clock", "dribbles", "touch_time", "kaggle_secs_left"]
print("\nnew columns:")
print(df[new].describe().round(2).to_string())
print("\nshare shot_clock_off:", round(df["shot_clock_off"].mean(), 4),
      " shot_clock_missing:", round(df["shot_clock_missing"].mean(), 4),
      " is_catch_shoot:", round(df["is_catch_shoot"].mean(), 4))

# quick sanity check only (Kaggle covers part of 2014-15, no hold-out data involved)
def_bin = pd.cut(df["close_def_dist"], [0, 2, 4, 6, float("inf")], right=False, labels=["0-2", "2-4", "4-6", "6+"])
print("\nFG% by close_def_dist (ft):")
print(df.groupby(def_bin, observed=True)["SHOT_MADE_FLAG"].agg(n="size", fg="mean").round(3).to_string())
print("\nFG% by is_catch_shoot:")
print(df.groupby("is_catch_shoot")["SHOT_MADE_FLAG"].agg(n="size", fg="mean").round(3).to_string())
