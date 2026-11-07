"""Match Kaggle Shot Logs 2014-15 shots to nba_api shots (by game, player, period and game clock).

Kaggle GAME_CLOCK runs a few seconds behind nba_api (Kaggle shows more time left, typical offset +2 s),
so candidate pairs with a Kaggle - nba_api clock difference in [MIN_DIFF, MAX_DIFF] are accepted, pairs closest
to TYPICAL_DIFF first, and each shot is used at most once on both sides. Exact clock matches are not preferred,
because with the typical offset they often pair a shot with a neighbouring one (e.g. a putback).
Outcome (FGM) and shot type (PTS_TYPE) are used only to validate the result, never to match.
"""
import pandas as pd

pd.set_option("display.width", 250)
OUT = "data/processed/kaggle_nba_match.parquet"
KEY = ["GAME_ID", "PLAYER_ID", "PERIOD"]
MIN_DIFF, MAX_DIFF, TYPICAL_DIFF = -1, 5, 2

kg = pd.read_parquet("data/interim/kaggle_shot_logs.parquet")
kg["k_row"] = range(len(kg))
mins, secs = kg["GAME_CLOCK"].str.split(":", expand=True).astype(int).T.values
kg["secs"] = mins * 60 + secs
k = kg.rename(columns={"player_id": "PLAYER_ID"})[KEY + ["secs", "SHOT_NUMBER", "k_row"]]

nba = pd.read_parquet(
    "data/processed/shots_features.parquet",
    columns=["SEASON", "GAME_ID", "GAME_EVENT_ID", "PLAYER_ID", "PERIOD", "secs_left_period",
             "dist_ft", "shot_value", "SHOT_MADE_FLAG"],
)
nba = nba[nba["SEASON"] == "2014-15"].rename(columns={"secs_left_period": "secs"})
n = nba[KEY + ["secs", "GAME_EVENT_ID"]]

# candidate pairs: same game, player and period, clock offset within [MIN_DIFF, MAX_DIFF]
cand = k.merge(n, on=KEY, suffixes=("_k", "_n"))
cand["secs_diff"] = cand["secs_k"] - cand["secs_n"]
cand = cand[cand["secs_diff"].between(MIN_DIFF, MAX_DIFF)]
cand = cand.assign(gap=(cand["secs_diff"] - TYPICAL_DIFF).abs()).sort_values(["gap", "SHOT_NUMBER", "GAME_EVENT_ID"])
# greedy one-to-one assignment, most typical offsets first
pairs = []
while len(cand):
    best = cand.drop_duplicates("k_row").drop_duplicates(["GAME_ID", "GAME_EVENT_ID"])
    pairs.append(best)
    cand = cand[~cand["k_row"].isin(best["k_row"])
                & ~cand.set_index(["GAME_ID", "GAME_EVENT_ID"]).index.isin(best.set_index(["GAME_ID", "GAME_EVENT_ID"]).index)]
match = pd.concat(pairs, ignore_index=True)[["k_row", "GAME_ID", "GAME_EVENT_ID", "secs_diff"]]
match.to_parquet(OUT, index=False)
print("saved:", OUT, match.shape)

print("\nKaggle shots:", len(kg), " nba_api 2014-15 shots:", len(nba))
print("matched:", len(match), f"({len(match) / len(kg):.2%})")
print("secs_diff:", match["secs_diff"].value_counts().sort_index().to_dict())

unmatched = kg[~kg["k_row"].isin(match["k_row"])]
print("\nunmatched Kaggle shots:", len(unmatched))
print(unmatched[["k_row", "GAME_ID", "player_name", "player_id", "PERIOD", "GAME_CLOCK", "SHOT_DIST", "PTS_TYPE"]]
      .sample(min(8, len(unmatched)), random_state=0).to_string(index=False))
print("unmatched whose player has no nba_api shot in that game:",
      (~unmatched.set_index(["GAME_ID", "player_id"]).index
       .isin(nba.set_index(["GAME_ID", "PLAYER_ID"]).index)).sum())

print("\nduplicates: k_row", match["k_row"].duplicated().sum(),
      " (GAME_ID, GAME_EVENT_ID)", match.duplicated(["GAME_ID", "GAME_EVENT_ID"]).sum())

# validation only: outcome and shot type were not used for matching
v = (match.merge(kg[["k_row", "FGM", "PTS_TYPE", "SHOT_DIST"]], on="k_row")
     .merge(nba[["GAME_ID", "GAME_EVENT_ID", "SHOT_MADE_FLAG", "shot_value", "dist_ft"]], on=["GAME_ID", "GAME_EVENT_ID"]))
print("\nvalidation on matched pairs")
print("FGM == SHOT_MADE_FLAG:", round((v["FGM"] == v["SHOT_MADE_FLAG"]).mean(), 4))
print("PTS_TYPE == shot_value:", round((v["PTS_TYPE"] == v["shot_value"]).mean(), 4))
print(v.groupby("secs_diff").apply(lambda d: pd.Series({
    "n": len(d), "FGM agree": (d["FGM"] == d["SHOT_MADE_FLAG"]).mean()}), include_groups=False).round(4).to_string())
print("\nSHOT_DIST - dist_ft:")
print((v["SHOT_DIST"] - v["dist_ft"]).describe().round(2).to_string())
