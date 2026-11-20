"""Monte Carlo of played games: the real shots of both teams are kept, only their results are drawn.

Shots are grouped into **chains**: a shot and the putbacks that followed it after offensive rebounds of the same team
(the putback exists only because the shot before it missed). In one simulation a chain scores the value of its first
made shot and stops; the later shots of the chain are dropped. If a chain whose last shot was made in reality misses in
the simulation, the team gets an offensive rebound with the league rate rho and a generic putback (2 points with
probability p_pb), repeated at most MAX_EXTRA times. Chains of different possessions are independent.

Free throws are added as observed (FTM from the box score), so the simulated margin differs from the real one only by
the luck of field goal results (and by model error in p). The distribution of a team's points can also be computed
exactly (chain by chain, then convolution), which is used to check the Monte Carlo.
`chains=False` gives the naive model in which every shot is independent and the shot counts are fixed.
"""
import glob

import numpy as np
import pandas as pd
from scipy.stats import norm

KEY = ["GAME_ID", "GAME_EVENT_ID"]
MAX_EXTRA = 3


def load_games(pred_path, features_path, context_path, games_dir, model: str = "gbm_w2_ctx"):
    """Shots with p, value and chain, and one row per game (home and away team, FTM, actual points).

    Team-games whose field goal points + FTM differ from the box score PTS (shots missing in shotchartdetail) are
    dropped with their whole game. A shot continues the chain of the previous shot of the game if it follows an
    offensive rebound and the previous shot was a miss by the same team in the same period.
    Returns (shots, games, number of dropped games); shots carry `side` = "home" / "away".
    """
    pred = pd.read_parquet(pred_path, columns=KEY + ["SEASON", "y", model]).rename(columns={model: "p"})
    feat = pd.read_parquet(features_path, columns=KEY + ["TEAM_ID", "PLAYER_ID", "PERIOD", "shot_value", "is_heave"])
    ctx = pd.read_parquet(context_path, columns=KEY + ["prev_event"])
    shots = pred.merge(feat, on=KEY, how="left", validate="1:1").merge(ctx, on=KEY, how="left", validate="1:1")
    shots = add_chains(shots)

    box = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{games_dir}/games_*.parquet"))])
    box = box[box["GAME_ID"].isin(shots["GAME_ID"].unique())]
    fg = (shots.assign(fg_pts=shots["y"] * shots["shot_value"])
               .groupby(["GAME_ID", "TEAM_ID"]).agg(fga=("y", "size"), fg_pts=("fg_pts", "sum")).reset_index())
    tg = box.merge(fg, on=["GAME_ID", "TEAM_ID"], how="left")
    tg["ok"] = tg["fg_pts"] + tg["FTM"] == tg["PTS"]
    bad = tg.loc[~tg["ok"], "GAME_ID"].unique()
    tg = tg[~tg["GAME_ID"].isin(bad)].copy()
    tg["side"] = np.where(tg["MATCHUP"].str.contains("vs."), "home", "away")

    cols = ["TEAM_ID", "TEAM_ABBREVIATION", "FGA", "OREB", "TOV", "FTM", "FTA", "PTS", "WL"]
    home = tg[tg["side"] == "home"].set_index("GAME_ID")[["SEASON", "GAME_DATE"] + cols]
    away = tg[tg["side"] == "away"].set_index("GAME_ID")[cols]
    games = home.join(away, lsuffix="_home", rsuffix="_away", how="inner").reset_index()
    games["GAME_DATE"] = pd.to_datetime(games["GAME_DATE"])
    games["margin"] = games["PTS_home"] - games["PTS_away"]
    games["home_win"] = (games["margin"] > 0).astype(int)

    shots = shots[shots["GAME_ID"].isin(games["GAME_ID"])].copy()
    side = tg.set_index(["GAME_ID", "TEAM_ID"])["side"]
    shots["side"] = side.reindex(pd.MultiIndex.from_frame(shots[["GAME_ID", "TEAM_ID"]])).to_numpy()
    return shots.sort_values(["GAME_ID", "side", "GAME_EVENT_ID"]).reset_index(drop=True), games, len(bad)


def add_chains(shots: pd.DataFrame) -> pd.DataFrame:
    """Chain id (consecutive integers), and for every chain whether its last real shot was made."""
    s = shots.sort_values(["GAME_ID", "GAME_EVENT_ID"]).copy()
    g = s.groupby("GAME_ID")
    cont = ((s["prev_event"] == "off_rebound") & (g["TEAM_ID"].shift() == s["TEAM_ID"])
            & (g["y"].shift() == 0) & (g["PERIOD"].shift() == s["PERIOD"]))
    s["cont"] = cont.to_numpy()
    s["chain"] = np.cumsum(~s["cont"].to_numpy())
    s["chain_len"] = s.groupby("chain")["y"].transform("size")
    s["chain_last_made"] = s.groupby("chain")["y"].transform("last").astype(int)
    return s


def putback_params(features_path, context_path) -> pd.DataFrame:
    """Per season: rho = putback shots per missed field goal, p_pb = points per putback / 2 (real results)."""
    feat = pd.read_parquet(features_path, columns=KEY + ["SEASON", "TEAM_ID", "PERIOD", "SHOT_MADE_FLAG", "shot_value"])
    ctx = pd.read_parquet(context_path, columns=KEY + ["prev_event"])
    s = add_chains(feat.merge(ctx, on=KEY, how="left").rename(columns={"SHOT_MADE_FLAG": "y"}))
    out = s.groupby("SEASON").apply(lambda q: pd.Series({
        "rho": q["cont"].sum() / (q["y"] == 0).sum(),
        "p_pb": (q.loc[q["cont"], "y"] * q.loc[q["cont"], "shot_value"]).mean() / 2,
        "share_cont": q["cont"].mean()}), include_groups=False)
    return out


def extra_pmf(rho: float, p_pb: float, depth: int = MAX_EXTRA) -> np.ndarray:
    """Points from extra putbacks after a simulated miss of a chain that was made in reality."""
    tail = np.array([1.0])
    for _ in range(depth):
        new = np.zeros(len(tail) + 2)
        new[0] = 1 - rho
        new[2] += rho * p_pb
        new[:len(tail)] += rho * (1 - p_pb) * tail
        tail = new
    return tail


def team_pmf(p, v, chain, last_made, rho=0.0, p_pb=0.0) -> np.ndarray:
    """Exact distribution of a team's field goal points (index = points).

    chain: chain id per shot (shots of a chain consecutive), last_made: per shot, the real result of its chain's last shot.
    """
    p, v, chain = np.asarray(p, float), np.asarray(v, int), np.asarray(chain)
    starts = np.r_[0, np.flatnonzero(np.diff(chain)) + 1, len(chain)]
    extra = extra_pmf(rho, p_pb) if rho > 0 else np.array([1.0])
    total = np.array([1.0])
    for a, b in zip(starts[:-1], starts[1:]):
        pmf = extra.copy() if last_made[b - 1] and rho > 0 else np.array([1.0])
        for i in range(b - 1, a - 1, -1):
            new = np.zeros(max(len(pmf), v[i] + 1))
            new[:len(pmf)] = (1 - p[i]) * pmf
            new[v[i]] += p[i]
            pmf = new
        total = np.convolve(total, pmf)
    return total


def margin_dist(pmf_h, pmf_a, ft_diff: int):
    """Distribution of home - away points: values and probabilities."""
    d = np.convolve(pmf_h, pmf_a[::-1])
    return np.arange(len(d)) - (len(pmf_a) - 1) + ft_diff, d


def simulate_team(p, v, chain, last_made, rng, n_sims, rho=0.0, p_pb=0.0, u=None) -> np.ndarray:
    """Simulated field goal points of a team (n_sims values). u: optional uniforms (n_sims x shots)."""
    p, v, chain = np.asarray(p, np.float32), np.asarray(v, np.int16), np.asarray(chain)
    if u is None:
        u = rng.random((n_sims, len(p)), dtype=np.float32)
    made = u < p
    starts = np.r_[0, np.flatnonzero(np.diff(chain)) + 1]
    cid = np.cumsum(np.isin(np.arange(len(p)), starts)) - 1          # chain index 0..k-1 per shot
    cum = np.cumsum(made, axis=1)
    before = np.where(starts > 0, cum[:, np.maximum(starts - 1, 0)], 0)  # makes before each chain
    within = cum - before[:, cid]
    first = made & (within == 1)
    pts = first @ v
    if rho > 0:
        ends = np.r_[starts[1:], len(p)] - 1
        elig = np.asarray(last_made)[ends].astype(bool)
        no_make = within[:, ends[elig]] == 0                            # sims where an eligible chain missed
        extra = np.zeros(no_make.shape, np.int16)
        alive = no_make.copy()
        for _ in range(MAX_EXTRA):
            reb = rng.random(alive.shape) < rho
            mk = rng.random(alive.shape) < p_pb
            extra += 2 * (alive & reb & mk)
            alive &= reb & ~mk
        pts = pts + extra.sum(axis=1)
    return pts


def run_games(shots, games, params=None, p_col: str = "p", chains: bool = True, n_sims: int = 10_000, seed: int = 0,
              mc: bool = True, ft_prob=None, antithetic: bool = False) -> pd.DataFrame:
    """Simulate every game; one row per game with P(home win) and the luck percentile of the actual margin.

    params: DataFrame indexed by SEASON with rho and p_pb (putbacks; use the previous season's values).
    chains=False: naive model, every shot independent, no putbacks. mc=False: exact results only.
    ft_prob: DataFrame indexed by GAME_ID with q_home, q_away; then each real free throw attempt is drawn (exact only).
    A tie counts as half a win.
    """
    rng = np.random.default_rng(seed)
    g = games.set_index("GAME_ID")
    shots = shots[shots["GAME_ID"].isin(g.index)]
    out = []
    for gid, gs in shots.groupby("GAME_ID", sort=False):
        row = g.loc[gid]
        rho, p_pb = (params.loc[row["SEASON"], "rho"], params.loc[row["SEASON"], "p_pb"]) if chains else (0.0, 0.0)
        teams = {}
        for side in ("home", "away"):
            t = gs[gs["side"] == side]
            ch = t["chain"].to_numpy() if chains else np.arange(len(t))
            lm = t["chain_last_made"].to_numpy() if chains else np.zeros(len(t), int)
            teams[side] = (t[p_col].to_numpy(), t["shot_value"].to_numpy(), ch, lm)
        ft = int(row["FTM_home"] - row["FTM_away"])
        pmf_h = team_pmf(*teams["home"], rho, p_pb)
        pmf_a = team_pmf(*teams["away"], rho, p_pb)
        if ft_prob is not None:
            q = ft_prob.loc[gid]
            pmf_h = np.convolve(pmf_h, _binom_pmf(int(row["FTA_home"]), q["q_home"]))
            pmf_a = np.convolve(pmf_a, _binom_pmf(int(row["FTA_away"]), q["q_away"]))
            ft = 0
        x, d = margin_dist(pmf_h, pmf_a, ft)
        mu = np.sum(x * d)
        sd = np.sqrt(np.sum((x - mu) ** 2 * d))
        margin = int(row["margin"])
        res = {"GAME_ID": gid, "xmargin": mu, "sd_margin": sd,
               "p_win_exact": d[x > 0].sum() + 0.5 * d[x == 0].sum(),
               "luck_pct_exact": d[x < margin].sum() + 0.5 * d[x == margin].sum(),
               "p_win_normal": float(norm.sf(0, mu, sd))}
        if mc:
            uh = ua = None
            if antithetic:
                half = n_sims // 2
                uh = rng.random((half, len(teams["home"][0])), dtype=np.float32)
                ua = rng.random((half, len(teams["away"][0])), dtype=np.float32)
                uh, ua = np.vstack([uh, 1 - uh]), np.vstack([ua, 1 - ua])
            m = (simulate_team(*teams["home"], rng, n_sims, rho, p_pb, uh)
                 - simulate_team(*teams["away"], rng, n_sims, rho, p_pb, ua) + ft)
            res["p_win_mc"] = (m > 0).mean() + 0.5 * (m == 0).mean()
            res["luck_pct_mc"] = (m < margin).mean() + 0.5 * (m == margin).mean()
        out.append(res)
    return games.merge(pd.DataFrame(out), on="GAME_ID")


def _binom_pmf(n: int, q: float) -> np.ndarray:
    from scipy.stats import binom
    return binom.pmf(np.arange(n + 1), n, q)
