"""Check Kaggle NBA Shot Logs 2014-15 and save as parquet with nba_api-compatible GAME_ID."""
import pandas as pd

RAW = "data/raw/kaggle/shot_logs.csv"
OUT = "data/interim/kaggle_shot_logs.parquet"

df = pd.read_csv(RAW)
print(f"rows: {len(df):,}  columns: {df.shape[1]}")
print(df.columns.tolist())

# Kaggle stores GAME_ID as int (21400001); nba_api uses a 10-char string ("0021400001")
df["GAME_ID"] = df["GAME_ID"].astype(str).str.zfill(10)
df["GAME_DATE"] = pd.to_datetime(df["MATCHUP"].str.split(" - ").str[0], format="%b %d, %Y")

print("games:", df["GAME_ID"].nunique(), " dates:", df["GAME_DATE"].min().date(), "->", df["GAME_DATE"].max().date())
print("missing values:\n", df.isna().sum()[lambda s: s > 0])
print("negative TOUCH_TIME:", (df["TOUCH_TIME"] < 0).sum())

nba = pd.read_parquet("data/interim/shots.parquet", columns=["GAME_ID", "SEASON"])
nba_ids = set(nba.loc[nba["SEASON"] == "2014-15", "GAME_ID"])
print(f"Kaggle games found in nba_api 2014-15: {df['GAME_ID'].isin(nba_ids).mean():.2%}")

df.to_parquet(OUT, index=False)
print("saved", OUT)
