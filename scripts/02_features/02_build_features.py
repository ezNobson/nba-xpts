"""Build per-shot features (row-wise transforms only, no fitted statistics)."""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("data/processed/shots_features.parquet")
ID_COLS = ["GAME_ID", "GAME_EVENT_ID", "PLAYER_ID", "PLAYER_NAME", "TEAM_ID",
           "SEASON", "GAME_DATE", "HTM", "VTM"]
# keyword rules applied in order, first match wins
ACTION_RULES = [
    (("Dunk",), "dunk"),
    (("Tip", "Putback"), "tip_putback"),
    (("Layup",), "layup"),
    (("Hook",), "hook"),
    (("Floating",), "floater"),
    (("Step Back",), "stepback_jumper"),
    (("Pull",), "pullup_jumper"),
    (("Fadeaway", "Turnaround"), "fadeaway_jumper"),
    (("Jump",), "jumper"),
]


def action_group(action):
    for keywords, group in ACTION_RULES:
        if any(k in action for k in keywords):
            return group
    return "other"


shots = pd.read_parquet("data/interim/shots.parquet")
missing_loc = shots[["LOC_X", "LOC_Y"]].isna().any(axis=1)
print("dropped shots with missing LOC_X/LOC_Y:", missing_loc.sum())
shots = shots[~missing_loc].reset_index(drop=True)

df = shots[ID_COLS + ["SHOT_MADE_FLAG"]].copy()
df["LOC_X"] = shots["LOC_X"]
df["LOC_Y"] = shots["LOC_Y"]
df["dist_ft"] = np.hypot(shots["LOC_X"], shots["LOC_Y"]) / 10
df["angle_deg"] = np.degrees(np.arctan2(shots["LOC_X"], shots["LOC_Y"]))
df["is_three"] = shots["SHOT_TYPE"].eq("3PT Field Goal").astype(int)
df["shot_value"] = 2 + df["is_three"]
df["is_corner3"] = (df["is_three"].eq(1) & shots["LOC_X"].abs().ge(220) & shots["LOC_Y"].le(92.5)).astype(int)
df["SHOT_ZONE_BASIC"] = shots["SHOT_ZONE_BASIC"].astype("category")
df["SHOT_ZONE_AREA"] = shots["SHOT_ZONE_AREA"].astype("category")
df["ACTION_TYPE"] = shots["ACTION_TYPE"]
df["action_group"] = shots["ACTION_TYPE"].map(action_group).astype("category")
df["PERIOD"] = shots["PERIOD"]
df["is_overtime"] = shots["PERIOD"].ge(5).astype(int)
df["secs_left_period"] = shots["MINUTES_REMAINING"] * 60 + shots["SECONDS_REMAINING"]
df["is_heave"] = (df["dist_ft"].ge(30) & df["secs_left_period"].le(3)).astype(int)

OUT.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(OUT, index=False)
print("saved:", OUT, df.shape)
print("\ndtypes:")
print(df.dtypes)
print("\nNaN counts:")
print(df.isna().sum())

# EDA only on training seasons (2025-26 is hold-out)
eda = df[df["SEASON"] != "2025-26"]
print("\nEDA rows (excluding 2025-26):", len(eda))
print("\nACTION_TYPE -> action_group (top 60):")
print(eda.groupby(["ACTION_TYPE", "action_group"], observed=True).size()
      .sort_values(ascending=False).head(60).to_string())
for col in ["action_group", "is_heave"]:
    print(f"\n{col}: n and FG%")
    print(eda.groupby(col, observed=True)["SHOT_MADE_FLAG"].agg(n="size", fg_pct="mean")
          .sort_values("n", ascending=False).round(3).to_string())
