"""Diagnostics of Kaggle NBA Shot Logs 2014-15 before building features."""
import pandas as pd

pd.set_option("display.width", 250)
kg = pd.read_parquet("data/interim/kaggle_shot_logs.parquet")

print("1. shape:", kg.shape)
print(pd.DataFrame({"dtype": kg.dtypes.astype(str), "n_nan": kg.isna().sum()}).to_string())

# 2. SHOT_CLOCK is expected to be missing when the shot clock is turned off (< 24 s left in period)
mins, secs = kg["GAME_CLOCK"].str.split(":", expand=True).astype(int).T.values
kg["game_secs_left"] = mins * 60 + secs
clock_off = kg["game_secs_left"] < 24
nan_sc = kg["SHOT_CLOCK"].isna()
print("\n2. SHOT_CLOCK")
print("NaN rate, game clock < 24 s:", round(nan_sc[clock_off].mean(), 3), "of", clock_off.sum())
print("NaN rate, game clock >= 24 s:", round(nan_sc[~clock_off].mean(), 3), "of", (~clock_off).sum())
print("NaN with game clock >= 24 s:", (nan_sc & ~clock_off).sum())
print(kg["SHOT_CLOCK"].describe().round(2).to_string())
print("SHOT_CLOCK == 24.0:", kg["SHOT_CLOCK"].eq(24.0).sum())

print("\n3. TOUCH_TIME")
print(kg["TOUCH_TIME"].describe().round(2).to_string())
print("< 0:", kg["TOUCH_TIME"].lt(0).sum(), " == 0:", kg["TOUCH_TIME"].eq(0).sum())
print(pd.crosstab(kg["TOUCH_TIME"].le(0).rename("TOUCH_TIME<=0"), kg["DRIBBLES"].eq(0).rename("DRIBBLES==0")))

print("\n4. CLOSE_DEF_DIST and DRIBBLES")
print(kg[["CLOSE_DEF_DIST", "DRIBBLES"]].describe().round(2).to_string())

print("\n5. PTS_TYPE vs SHOT_DIST")
print("3PT with SHOT_DIST < 22:", (kg["PTS_TYPE"].eq(3) & kg["SHOT_DIST"].lt(22)).sum())
print("2PT with SHOT_DIST > 23.75:", (kg["PTS_TYPE"].eq(2) & kg["SHOT_DIST"].gt(23.75)).sum())

print("\n6. Leakage columns (never features):", [c for c in ["W", "FINAL_MARGIN", "PTS", "SHOT_RESULT", "FGM"] if c in kg])
print(pd.crosstab(kg["SHOT_RESULT"], kg["FGM"]))
print("PTS == FGM * PTS_TYPE:", (kg["PTS"] == kg["FGM"] * kg["PTS_TYPE"]).mean())
