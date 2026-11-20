"""Player shot making: residuals of out-of-sample xPTS and empirical Bayes shrinkage.

Shot making of a player is SM = (PTS - xPTS) / n, the mean of the per-shot residuals r = v * (y - p), where
v is the shot value (2 or 3) and p the out-of-sample make probability. Under the null "the player is an average
shooter" r has mean 0 and variance v^2 p (1 - p), so the sampling variance of SM is known for each player.

Shrinkage uses the normal-normal model
    SM_i | theta_i ~ N(theta_i, s_i^2),   theta_i ~ N(m_i, tau^2),   m_i = X_i @ beta,
which is the large-n form of the beta-binomial model. The prior mean m_i is the league level (X = 1), a position
level (X = position dummies) or a regression on the previous season. beta and tau^2 are fitted by maximising the
marginal likelihood SM_i ~ N(m_i, tau^2 + s_i^2) (empirical Bayes). The posterior mean is
    theta_i = w_i SM_i + (1 - w_i) m_i,   w_i = tau^2 / (tau^2 + s_i^2) ~ n_i / (n_i + k),   k = sigma^2 / tau^2.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

KEY = ["GAME_ID", "GAME_EVENT_ID"]
SHOT_COLS = ["PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "GAME_DATE", "shot_value", "is_three", "action_group",
             "dist_ft", "is_heave"]


def load_player_shots(pred_path, features_path, positions_path=None, model: str = "gbm_w2_ctx") -> pd.DataFrame:
    """Out-of-sample predictions joined with shot data; heaves removed.

    Adds p (make probability of `model`), xpts = v p, r = v (y - p), var = v^2 p (1 - p) and r_c, the residual
    centred on the league mean of its season (the model's season bias is not shot making).
    """
    pred = pd.read_parquet(pred_path, columns=KEY + ["SEASON", "y", model])
    shots = pd.read_parquet(features_path, columns=KEY + SHOT_COLS)
    df = pred.merge(shots, on=KEY, how="left", validate="1:1")
    df = df[df["is_heave"] == 0].drop(columns="is_heave").rename(columns={model: "p"})
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], format="%Y%m%d")
    df["pts"] = df["y"] * df["shot_value"]
    df["xpts"] = df["p"] * df["shot_value"]
    df["r"] = df["pts"] - df["xpts"]
    df["var"] = df["shot_value"] ** 2 * df["p"] * (1 - df["p"])
    df["r_c"] = df["r"] - df.groupby("SEASON")["r"].transform("mean")
    if positions_path is not None:
        pos = pd.read_parquet(positions_path, columns=["PLAYER_ID", "POS"])
        df = df.merge(pos, on="PLAYER_ID", how="left", validate="m:1")
    return df


def player_table(shots: pd.DataFrame, by=("SEASON", "PLAYER_ID"), resid: str = "r_c") -> pd.DataFrame:
    """One row per group: shots n, points, xPTS, shot making sm = mean residual and its sampling variance s2."""
    g = shots.groupby(list(by), observed=True)
    t = pd.DataFrame({"n": g.size(), "pts": g["pts"].sum(), "xpts": g["xpts"].sum(),
                      "sm": g[resid].mean(), "s2": g["var"].sum() / g.size() ** 2})
    return t.reset_index()


def fit_prior(sm, s2, X=None) -> dict:
    """Empirical Bayes: beta and tau^2 maximising the marginal likelihood sm ~ N(X beta, tau^2 + s2).

    For a given tau^2, beta is weighted least squares; tau^2 is found by a 1-D search (profile likelihood).
    Returns beta, tau2, the log-likelihood and k = mean per-shot variance / tau^2 (shots at which w = 0.5).
    """
    sm, s2 = np.asarray(sm, float), np.asarray(s2, float)
    X = np.ones((len(sm), 1)) if X is None else np.asarray(X, float)

    def beta_ll(tau2):
        w = 1 / (tau2 + s2)
        beta = np.linalg.solve(X.T @ (X * w[:, None]), X.T @ (w * sm))
        res = sm - X @ beta
        return beta, -0.5 * np.sum(np.log(2 * np.pi / w) + w * res ** 2)

    opt = minimize_scalar(lambda t: -beta_ll(np.exp(t))[1], bounds=(-20, 2), method="bounded")
    tau2 = float(np.exp(opt.x))
    beta, ll = beta_ll(tau2)
    return {"beta": beta, "tau2": tau2, "loglik": ll, "n_params": X.shape[1] + 1}


def shrink(sm, s2, prior_mean, tau2):
    """Posterior mean and variance of theta for the normal-normal model."""
    sm, s2 = np.asarray(sm, float), np.asarray(s2, float)
    w = tau2 / (tau2 + s2)
    return w * sm + (1 - w) * np.asarray(prior_mean, float), w * s2


def add_season_halves(shots: pd.DataFrame) -> pd.DataFrame:
    """Split every season at the median game date: half = 1 or 2.

    r_h is the residual centred on the league mean of its own half, so first-half numbers never use second-half shots.
    """
    shots = shots.copy()
    games = shots.drop_duplicates("GAME_ID")[["SEASON", "GAME_ID", "GAME_DATE"]]
    mid = games.groupby("SEASON")["GAME_DATE"].median()
    shots["half"] = np.where(shots["GAME_DATE"] <= shots["SEASON"].map(mid), 1, 2)
    shots["r_h"] = shots["r"] - shots.groupby(["SEASON", "half"])["r"].transform("mean")
    return shots


def prior_designs(t: pd.DataFrame, prev: pd.Series | None = None) -> dict:
    """Design matrices X of the prior mean for a player table t (one row per player).

    league: one common mean; position: one mean per primary position (G, F, C); previous: a + b * theta_prev for
    players with an estimate from the previous season, a separate mean for the others; pos_previous: both.
    prev maps PLAYER_ID to the shrunk shot making of the previous season.
    """
    one = np.ones((len(t), 1))
    pos = pd.get_dummies(t["POS"]).reindex(columns=["G", "F", "C"], fill_value=0).to_numpy(float)
    out = {"league": one, "position": pos}
    if prev is not None:
        th = t["PLAYER_ID"].map(prev)
        has = th.notna().to_numpy(float)
        out["previous"] = np.c_[one, has, th.fillna(0).to_numpy()]
        out["pos_previous"] = np.c_[pos, has, th.fillna(0).to_numpy()]
    return out


def cluster_bootstrap(df: pd.DataFrame, loss_cols, weight: str, base: str, cluster: str = "PLAYER_ID",
                      n_boot: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Weighted mean loss of each column and its difference to `base` with a 95% CI.

    Whole clusters (players, over all their seasons) are resampled, because the seasons of one player are not
    independent. diff < 0 means the column predicts better than base.
    """
    w = df[weight].to_numpy(float)
    g = df.assign(_w=w, **{c: df[c] * w for c in loss_cols}).groupby(cluster)[list(loss_cols) + ["_w"]].sum()
    idx = np.random.default_rng(seed).integers(len(g), size=(n_boot, len(g)))
    W = g["_w"].to_numpy()[idx].sum(axis=1)
    rows = {}
    for c in loss_cols:
        boot = (g[c].to_numpy()[idx].sum(axis=1) - g[base].to_numpy()[idx].sum(axis=1)) / W
        rows[c] = {"loss": g[c].sum() / g["_w"].sum(), "diff": (g[c].sum() - g[base].sum()) / g["_w"].sum(),
                   "ci_low": np.quantile(boot, 0.025), "ci_high": np.quantile(boot, 0.975)}
    return pd.DataFrame(rows).T


def season_estimates(t: pd.DataFrame, positions: pd.Series) -> pd.DataFrame:
    """Shrunk shot making per season with the previous-season prior (league prior in the first season).

    t is player_table(shots) (one row per season and player), positions maps PLAYER_ID to G/F/C.
    Each season uses only its own shots and the estimates of the season before.
    """
    seasons = sorted(t["SEASON"].unique())
    out, prev = [], None
    for season in seasons:
        g = t[t["SEASON"] == season].copy()
        g["POS"] = g["PLAYER_ID"].map(positions)
        designs = prior_designs(g, prev)
        X = designs["previous"] if prev is not None else designs["league"]
        f = fit_prior(g["sm"], g["s2"], X)
        g["prior"] = X @ f["beta"]
        g["theta"], g["post_var"] = shrink(g["sm"], g["s2"], g["prior"], f["tau2"])
        f0 = fit_prior(g["sm"], g["s2"])  # the previous-season input is the league-prior estimate (as in 10_players)
        prev = pd.Series(shrink(g["sm"], g["s2"], f0["beta"][0], f0["tau2"])[0], index=g["PLAYER_ID"])
        out.append(g)
    return pd.concat(out, ignore_index=True)
