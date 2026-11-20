"""Pre-game forecast of a game by Monte Carlo, using only games played before it.

Profile of a team = its last K games before the game date (they may reach into the previous season):
- attack: possessions with a shot (chains) per game, the pool of first shots and of putbacks (p, value), offensive
  rebound rate rho (putbacks per missed shot), FTM per game, shot making (points per shot above xPTS, shrunk);
- defence: the same quantities allowed to opponents.
The two profiles are combined additively around the league average of the previous season, e.g.
chains_A = attack_A + allowed_B - league. One simulation uses the expected number of chains (stochastic rounding; both teams have about the same number of
possessions, so independent random counts would exaggerate the spread), uses the expected FTM in the same way, draws the shots of each chain from the pools, makes with p + shift / v, and putbacks after misses with probability rho.
The home team gets h points (home advantage, estimated on development seasons).
"""
import numpy as np
import pandas as pd

MAX_PUTBACKS = 3


def team_game_table(shots: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """One row per team and game with attack and allowed (opponent) totals.

    shots come from simulation.load_games (with chains); r = v (y - p) is the residual against xPTS.
    """
    s = shots.assign(r=shots["shot_value"] * (shots["y"] - shots["p"]),
                     var=shots["shot_value"] ** 2 * shots["p"] * (1 - shots["p"]),
                     miss=1 - shots["y"])
    s = s.merge(games[["GAME_ID", "TEAM_ID_home", "TEAM_ID_away"]], on="GAME_ID")
    att = s.groupby(["GAME_ID", "TEAM_ID"]).agg(shots=("y", "size"), chains=("cont", lambda c: (~c).sum()),
                                                cont=("cont", "sum"), miss=("miss", "sum"), r=("r", "sum"),
                                                var=("var", "sum")).reset_index()
    rows = []
    for side, opp_side in (("home", "away"), ("away", "home")):
        t = games[["GAME_ID", "SEASON", "GAME_DATE", f"TEAM_ID_{side}", f"TEAM_ID_{opp_side}", f"FTM_{side}",
                   f"FTM_{opp_side}", f"PTS_{side}", f"PTS_{opp_side}"]].copy()
        t.columns = ["GAME_ID", "SEASON", "GAME_DATE", "TEAM_ID", "OPP_ID", "ftm", "ftm_allowed", "pts", "pts_allowed"]
        t["is_home"] = int(side == "home")
        rows.append(t)
    tg = pd.concat(rows, ignore_index=True)
    tg = tg.merge(att, on=["GAME_ID", "TEAM_ID"], how="left")
    opp = att.rename(columns={"TEAM_ID": "OPP_ID", **{c: f"{c}_allowed" for c in ["shots", "chains", "cont", "miss", "r", "var"]}})
    tg = tg.merge(opp, on=["GAME_ID", "OPP_ID"], how="left")
    tg["win"] = (tg["pts"] > tg["pts_allowed"]).astype(int)
    return tg.sort_values(["TEAM_ID", "GAME_DATE", "GAME_ID"]).reset_index(drop=True)


def rolling_profiles(tg: pd.DataFrame, K: int) -> pd.DataFrame:
    """Profile of each team before each of its games: sums over its previous K games (shifted, no current game)."""
    cols = ["chains", "chains_allowed", "ftm", "ftm_allowed", "cont", "miss", "cont_allowed", "miss_allowed",
            "r", "var", "shots", "r_allowed", "var_allowed", "shots_allowed", "win", "pts", "pts_allowed"]
    g = tg.groupby("TEAM_ID")[cols]
    prof = g.transform(lambda x: x.shift(1).rolling(K, min_periods=1).sum())
    prof["n_games"] = tg.groupby("TEAM_ID")["GAME_ID"].transform(lambda x: x.shift(1).rolling(K, min_periods=1).count())
    prof[["GAME_ID", "TEAM_ID", "SEASON", "GAME_DATE"]] = tg[["GAME_ID", "TEAM_ID", "SEASON", "GAME_DATE"]]
    return prof


def league_means(tg: pd.DataFrame) -> pd.DataFrame:
    """League averages per team-game by season (used with a one-season lag)."""
    g = tg.groupby("SEASON")
    return pd.DataFrame({"chains": g["chains"].mean(), "ftm": g["ftm"].mean(),
                         "rho": g["cont"].sum() / g["miss"].sum()})


def window_games(tg: pd.DataFrame, K: int) -> dict:
    """For every (GAME_ID, TEAM_ID) the GAME_IDs of the team's previous K games."""
    out = {}
    for tid, t in tg.groupby("TEAM_ID", sort=False):
        ids = t["GAME_ID"].to_numpy()
        for i, gid in enumerate(ids):
            out[(gid, tid)] = ids[max(0, i - K):i]
    return out


def simulate_side(pool_first, pool_pb, lam_chains, lam_ftm, rho, shift, rng, n_sims):
    """Simulated points of one team: expected number of chains and FTM, shots drawn from the pools."""
    p1, v1 = pool_first
    pp, vp = pool_pb
    lam_chains = max(lam_chains, 1.0)
    n = (np.floor(lam_chains) + (rng.random(n_sims) < lam_chains % 1)).astype(int)  # stochastic rounding
    m = n.max()
    alive_chain = np.arange(m)[None, :] < n[:, None]
    idx = rng.integers(len(p1), size=(n_sims, m))
    pr = np.clip(p1[idx] + shift / v1[idx], 0.001, 0.999)
    made = (rng.random((n_sims, m)) < pr) & alive_chain
    pts = (made * v1[idx]).sum(axis=1)
    open_ = alive_chain & ~made
    if len(pp):
        for _ in range(MAX_PUTBACKS):
            reb = open_ & (rng.random((n_sims, m)) < rho)
            j = rng.integers(len(pp), size=(n_sims, m))
            prj = np.clip(pp[j] + shift / vp[j], 0.001, 0.999)
            mk = reb & (rng.random((n_sims, m)) < prj)
            pts = pts + (mk * vp[j]).sum(axis=1)
            open_ = reb & ~mk
    lam_ftm = max(lam_ftm, 0.0)
    return pts + np.floor(lam_ftm) + (rng.random(n_sims) < lam_ftm % 1)  # expected FTM, as for the chains


def forecast_games(target: pd.DataFrame, shots: pd.DataFrame, tg: pd.DataFrame, K: int, k_shrink: float,
                   h: float = 0.0, n_sims: int = 10_000, seed: int = 0) -> pd.DataFrame:
    """P(home win), expected and sd of the margin for every game in `target`, from the previous K games only.

    k_shrink: shots at which team shot making gets weight 1/2 (shrinkage toward 0). h: home advantage in points.
    """
    rng = np.random.default_rng(seed)
    prof = rolling_profiles(tg, K).set_index(["GAME_ID", "TEAM_ID"])
    lg = league_means(tg)
    prev = dict(zip(sorted(lg.index)[1:], sorted(lg.index)[:-1]))
    win = window_games(tg, K)
    first = shots[~shots["cont"]]
    pb = shots[shots["cont"]]
    fp = {gid: (g["p"].to_numpy(), g["shot_value"].to_numpy()) for gid, g in first.groupby(["GAME_ID", "TEAM_ID"])}
    pbp = {gid: (g["p"].to_numpy(), g["shot_value"].to_numpy()) for gid, g in pb.groupby(["GAME_ID", "TEAM_ID"])}

    def pool(gids, tid, d):
        parts = [d[(g, tid)] for g in gids if (g, tid) in d]
        if not parts:
            return np.array([]), np.array([])
        return np.concatenate([a for a, _ in parts]), np.concatenate([b for _, b in parts]).astype(float)

    out = []
    for r in target.itertuples(index=False):
        L = lg.loc[prev.get(r.SEASON, r.SEASON)]
        res = {}
        sides = {"home": (r.TEAM_ID_home, r.TEAM_ID_away), "away": (r.TEAM_ID_away, r.TEAM_ID_home)}
        sim = {}
        for side, (a, b) in sides.items():
            pa, pb_ = prof.loc[(r.GAME_ID, a)], prof.loc[(r.GAME_ID, b)]
            na, nb = max(pa["n_games"], 1), max(pb_["n_games"], 1)
            lam_c = pa["chains"] / na + pb_["chains_allowed"] / nb - L["chains"]
            lam_f = pa["ftm"] / na + pb_["ftm_allowed"] / nb - L["ftm"]
            rho = (pa["cont"] / max(pa["miss"], 1) + pb_["cont_allowed"] / max(pb_["miss_allowed"], 1) - L["rho"])
            sm_a = pa["r"] / (pa["shots"] + k_shrink)            # = raw SM * n / (n + k)
            sm_b = pb_["r_allowed"] / (pb_["shots_allowed"] + k_shrink)
            ids = win[(r.GAME_ID, a)]
            sim[side] = simulate_side(pool(ids, a, fp), pool(ids, a, pbp), lam_c, lam_f, float(np.clip(rho, 0, 0.6)),
                                      sm_a + sm_b, rng, n_sims)
            res[f"x_{side}"] = sim[side].mean()
        m = sim["home"] - sim["away"] + h
        res.update({"GAME_ID": r.GAME_ID, "p_home": (m > 0).mean() + 0.5 * (m == 0).mean(), "xmargin": m.mean(),
                    "sd_margin": m.std()})
        out.append(res)
    return target.merge(pd.DataFrame(out), on="GAME_ID")


def elo_ratings(games: pd.DataFrame, k: float = 20.0, home: float = 100.0, carry: float = 0.75) -> pd.DataFrame:
    """Pre-game Elo probability of a home win for every game (FiveThirtyEight-style, without margin of victory).

    Ratings start at 1500, move by k * (result - expected) after each game, and regress to the mean between seasons
    (r = carry * r + (1 - carry) * 1505). Only results of earlier games are used.
    """
    g = games.sort_values(["GAME_DATE", "GAME_ID"])
    rating, season = {}, None
    p_out = []
    for r in g.itertuples(index=False):
        if r.SEASON != season:
            rating = {t: carry * v + (1 - carry) * 1505 for t, v in rating.items()}
            season = r.SEASON
        rh, ra = rating.get(r.TEAM_ID_home, 1500.0), rating.get(r.TEAM_ID_away, 1500.0)
        p = 1 / (1 + 10 ** (-(rh + home - ra) / 400))
        p_out.append(p)
        res = 1.0 if r.PTS_home > r.PTS_away else 0.0
        rating[r.TEAM_ID_home] = rh + k * (res - p)
        rating[r.TEAM_ID_away] = ra - k * (res - p)
    return pd.DataFrame({"GAME_ID": g["GAME_ID"].to_numpy(), "p_elo": p_out})
