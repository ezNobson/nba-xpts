"""Merge per team-season raw files into one parquet and run sanity checks."""
from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw/shots")
OUT = Path("data/interim/shots.parquet")

files = sorted(RAW_DIR.glob("shots_*.parquet"))
print("files:", len(files), "(expected 360)")
df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

dups = df.duplicated(["GAME_ID", "GAME_EVENT_ID"]).sum()
print(f"rows: {len(df):,}   duplicates: {dups}")

summary = df.groupby("SEASON").agg(
    games=("GAME_ID", "nunique"),
    shots=("GAME_ID", "size"),
    fg_pct=("SHOT_MADE_FLAG", "mean"),
    first=("GAME_DATE", "min"),
    last=("GAME_DATE", "max"),
)
summary["shots_per_game"] = (summary["shots"] / summary["games"]).round(1)
print(summary.round(3))

df.to_parquet(OUT, index=False)
print(f"saved {OUT} ({OUT.stat().st_size / 1e6:.0f} MB)")
