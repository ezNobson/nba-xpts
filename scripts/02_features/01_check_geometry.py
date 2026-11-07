"""Check consistency of shot coordinates with SHOT_DISTANCE and SHOT_TYPE."""
import numpy as np
import pandas as pd

shots = pd.read_parquet(
    "data/interim/shots.parquet",
    columns=["SEASON", "LOC_X", "LOC_Y", "SHOT_DISTANCE", "SHOT_TYPE"],
)
shots = shots[shots["SEASON"] != "2025-26"]  # hold-out test season
print("shots (excluding 2025-26):", len(shots))

# (a) exact distance in feet vs reported SHOT_DISTANCE
dist = np.hypot(shots["LOC_X"], shots["LOC_Y"]) / 10
print("\n(a) exact distance - SHOT_DISTANCE:")
print((dist - shots["SHOT_DISTANCE"]).describe())
print("share matching round():", (np.round(dist) == shots["SHOT_DISTANCE"]).mean())
print("share matching floor():", (np.floor(dist) == shots["SHOT_DISTANCE"]).mean())

# (b) raw coordinates
print("\n(b) LOC_X / LOC_Y:")
print(shots[["LOC_X", "LOC_Y"]].describe())

# (c) suspicious shots vs 3PT line geometry (22 ft corner, 23.75 ft arc)
is_3 = shots["SHOT_TYPE"].eq("3PT Field Goal")
corner = shots["LOC_Y"] <= 92.5
print("\n(c) geometry anomalies:")
print("3PT with distance < 22 ft:", (is_3 & (dist < 22)).sum())
print("2PT beyond 23.75 ft outside corner:", (~is_3 & ~corner & (dist > 23.75)).sum())
print("2PT beyond 22 ft in corner:", (~is_3 & corner & (dist > 22)).sum())
print("LOC_Y < -52:", (shots["LOC_Y"] < -52).sum())
