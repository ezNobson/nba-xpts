"""Compare per team-game shot counts from shotchartdetail with official box scores."""
from pathlib import Path

import pandas as pd

shots = pd.read_parquet("data/interim/shots.parquet")
games = pd.concat(
    [pd.read_parquet(f) for f in sorted(Path("data/raw/games").glob("games_*.parquet"))],
    ignore_index=True,
)

shots["is_3"] = shots["SHOT_TYPE"].eq("3PT Field Goal")
shots["made_3"] = shots["is_3"] & shots["SHOT_MADE_FLAG"].eq(1)
from_shots = shots.groupby(["SEASON", "GAME_ID", "TEAM_ID"]).agg(
    FGA_s=("SHOT_MADE_FLAG", "size"),
    FGM_s=("SHOT_MADE_FLAG", "sum"),
    FG3A_s=("is_3", "sum"),
    FG3M_s=("made_3", "sum"),
).reset_index()

box = games[["SEASON", "GAME_ID", "TEAM_ID", "FGA", "FGM", "FG3A", "FG3M"]].copy()
box["GAME_ID"] = box["GAME_ID"].astype(str)

m = box.merge(from_shots, on=["SEASON", "GAME_ID", "TEAM_ID"], how="outer", indicator=True)
print("team-games only in one source:", (m["_merge"] != "both").sum())
m = m[m["_merge"] == "both"]

for col in ["FGA", "FGM", "FG3A", "FG3M"]:
    m[f"{col}_diff"] = m[f"{col}_s"] - m[col]

m["exact"] = (m[["FGA_diff", "FGM_diff", "FG3A_diff", "FG3M_diff"]] == 0).all(axis=1)
summary = m.groupby("SEASON").agg(
    team_games=("exact", "size"),
    exact_pct=("exact", "mean"),
    fga_diff=("FGA_diff", "sum"),
    fgm_diff=("FGM_diff", "sum"),
)
print(summary.round(4))
print(f"overall exact match: {m['exact'].mean():.2%}")

worst = m.loc[~m["exact"]].assign(abs_diff=lambda x: x["FGA_diff"].abs() + x["FGM_diff"].abs())
print(worst.nlargest(10, "abs_diff")[["SEASON", "GAME_ID", "TEAM_ID", "FGA", "FGA_s", "FGM", "FGM_s"]])
