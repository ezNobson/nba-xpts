"""Build per-shot game-context features from play-by-play.

Every feature uses only events that happened BEFORE the shot in the same game (no leakage):
- is_home: shooter plays at home;
- prev_event: how the play before the shot ended (last event before the shot, substitutions and replays skipped);
- secs_since_prev: seconds between that event and the shot (same period);
- score_margin: shooter's team score minus opponent score before the shot.
Seasons without a play-by-play file (hold-out 2025-26) are skipped.
"""
from pathlib import Path

import numpy as np
import pandas as pd

PBP_DIR = Path("data/raw/pbp")
OUT = Path("data/processed/shots_context.parquet")
KEY = ["GAME_ID", "GAME_EVENT_ID"]
MAKE, MISS, FT, REBOUND, TURNOVER, FOUL, VIOLATION, SUB, TIMEOUT, JUMP_BALL, START, REPLAY = 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 18
DEADBALL = {FOUL: "foul", VIOLATION: "violation", TIMEOUT: "timeout", JUMP_BALL: "jump_ball", START: "period_start",
            FT: "free_throw"}


def season_context(pbp: pd.DataFrame, shots: pd.DataFrame) -> pd.DataFrame:
    """Context features for the shots of one season; pbp rows are in chronological order."""
    pbp = pbp.reset_index(drop=True)
    mins, secs = pbp["PCTIMESTRING"].str.split(":", expand=True).astype(int).T.to_numpy()
    pbp["secs"] = mins * 60 + secs
    # team of the event: team rebounds / turnovers carry the team id in PLAYER1_ID
    pbp["team"] = pbp["PLAYER1_TEAM_ID"].where(pbp["PLAYER1_TEAM_ID"].notna(),
                                               pbp["PLAYER1_ID"].where(pbp["PLAYER1_ID"] > 1.6e9))

    # home team of each game: team of made shots described in HOMEDESCRIPTION
    home_makes = pbp[(pbp["EVENTMSGTYPE"] == MAKE) & pbp["HOMEDESCRIPTION"].notna() & pbp["VISITORDESCRIPTION"].isna()]
    home_team = home_makes.groupby("GAME_ID")["team"].agg(lambda s: s.mode().iloc[0])

    # score margin (home - visitor) BEFORE each event: last known margin of the earlier rows
    margin = pd.to_numeric(pbp["SCOREMARGIN"].replace("TIE", "0"), errors="coerce")
    pbp["home_margin_before"] = margin.groupby(pbp["GAME_ID"]).transform(lambda s: s.ffill().shift()).fillna(0)

    # previous event in the same period, substitutions and replays skipped
    ev = pbp[~pbp["EVENTMSGTYPE"].isin([SUB, REPLAY, 13])].copy()
    g = ev.groupby(["GAME_ID", "PERIOD"])
    for col in ["EVENTMSGTYPE", "team", "secs", "PLAYER2_ID"]:
        ev["prev_" + col] = g[col].shift()
    ev["prev2_team"] = g["team"].shift(2)  # for a rebound: the team that missed

    out = shots[KEY + ["TEAM_ID"]].merge(
        ev.drop_duplicates(["GAME_ID", "EVENTNUM"]).rename(columns={"EVENTNUM": "GAME_EVENT_ID"})[
            KEY + ["EVENTMSGTYPE", "secs", "home_margin_before", "prev_EVENTMSGTYPE", "prev_team", "prev_secs",
                   "prev_PLAYER2_ID", "prev2_team"]],
        on=KEY, how="left")
    matched = out["EVENTMSGTYPE"].isin([MAKE, MISS])

    is_home = out["TEAM_ID"] == out["GAME_ID"].map(home_team)
    own = out["prev_team"] == out["TEAM_ID"]
    t = out["prev_EVENTMSGTYPE"]
    prev = t.map(DEADBALL).astype(object)
    prev[(t == REBOUND) & own & (out["prev2_team"] == out["TEAM_ID"])] = "off_rebound"
    prev[(t == REBOUND) & own & (out["prev2_team"] != out["TEAM_ID"])] = "def_rebound"
    prev[(t == REBOUND) & ~own] = "other"
    prev[(t == TURNOVER) & (out["prev_PLAYER2_ID"] > 0)] = "steal"
    prev[(t == TURNOVER) & ~(out["prev_PLAYER2_ID"] > 0)] = "dead_turnover"
    prev[t == MAKE] = "made_shot"
    prev[t == MISS] = "other"  # miss without a recorded rebound
    prev[~matched | t.isna()] = np.nan

    return pd.DataFrame({
        "GAME_ID": out["GAME_ID"], "GAME_EVENT_ID": out["GAME_EVENT_ID"],
        "is_home": is_home.astype(float).where(matched),
        "prev_event": prev,
        "secs_since_prev": (out["prev_secs"] - out["secs"]).where(matched),
        "score_margin": (out["home_margin_before"] * np.where(is_home, 1, -1)).where(matched),
    })


def main():
    shots = pd.read_parquet("data/processed/shots_features.parquet", columns=KEY + ["TEAM_ID", "SEASON"])
    frames = []
    for season, s in shots.groupby("SEASON"):
        path = PBP_DIR / f"pbp_{season}.parquet"
        if not path.exists():
            print(f"{season}: no play-by-play file, skipped")
            continue
        pbp = pd.read_parquet(path)
        time_up = (pbp.assign(m=pbp["PCTIMESTRING"].str.split(":").str[0].astype(int) * 60
                              + pbp["PCTIMESTRING"].str.split(":").str[1].astype(int))
                   .groupby(["GAME_ID", "PERIOD"])["m"].diff() > 0).sum()
        ctx = season_context(pbp, s)
        frames.append(ctx)
        print(f"{season}: {len(ctx):,} shots, matched {ctx['prev_event'].notna().mean():.4f}, "
              f"rows out of time order: {time_up}")
    out = pd.concat(frames, ignore_index=True)
    out["prev_event"] = out["prev_event"].astype("category")
    assert not out.duplicated(KEY).any()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    print("saved:", OUT, out.shape)


if __name__ == "__main__":
    main()
