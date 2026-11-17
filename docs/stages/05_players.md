# Stage 5: Player xPTS and shot making

This document describes how shot making of players is measured, how it is shrunk and which prior is used.
All numbers come from `notebooks/10_players.ipynb`; data files are not committed.

## 1. Purpose, inputs and outputs

Purpose: answer RQ2 (which players score more than the quality of their shots implies, and how much of it is skill).

| | File | Content |
|---|---|---|
| Input | `data/processed/oos_predictions.parquet` | out-of-sample predictions, test seasons 2015-16 to 2024-25 (column `gbm_w2_ctx`) |
| Input | `data/processed/holdout_predictions.parquet` | predictions for the hold-out season 2025-26 |
| Input | `data/processed/shots_features.parquet` | player, date and shot value of every shot |
| Input | `data/raw/players/positions.parquet` | listed position per player, `scripts/00_data/09_fetch_positions.py` |
| Code | `src/xpts/players.py` | `load_player_shots`, `player_table`, `fit_prior`, `shrink`, `add_season_halves`, `prior_designs`, `cluster_bootstrap` |
| Output | `reports/figures/05_*.pdf` | funnel plot, reliability, half-season test, top players, Curry |

## 2. Definitions

- Per shot: xPTS = v p, residual r = v (y - p), variance v^2 p (1 - p) under "average shooter".
- **SM = (PTS - xPTS) / n**, the mean residual; sampling variance s^2 = sum v^2 p (1 - p) / n^2 (known).
- Residuals are **centred on the league mean of the season** (season bias of the model: -0.023 to +0.033 points per shot).
- Heaves (`is_heave`, 0.4% of shots) are removed. 2,091,980 shots, 1,441 players.
- Positions: NBA.com listing via `nba_api` PlayerIndex, one per career; primary position = first letter (G, F, C).

## 3. Shrinkage (empirical Bayes)

SM_i | theta_i ~ N(theta_i, s_i^2), theta_i ~ N(X_i beta, tau^2). beta and tau^2 maximise the marginal likelihood
SM_i ~ N(X_i beta, tau^2 + s_i^2) (`fit_prior`, profile likelihood over tau^2). Posterior mean
theta = w SM + (1 - w) m, w = tau^2 / (tau^2 + s^2) ~ n / (n + k), k = sigma^2 / tau^2. This is the normal form of the
beta-binomial model (needed because 2PT and 3PT shots are mixed in one measure).

## 4. Results

| Quantity | Value |
|---|---|
| Player-seasons outside the 95% luck funnel (n >= 300) | 27.5% (5% expected), sd(z) = 1.85 |
| Split-half reliability (odd/even shots, Spearman-Brown) | 0.59 (n >= 100) to 0.75 (n >= 800) |
| tau (sd of true SM) | 0.071 points per shot (0.063-0.081 by season) |
| k (shots for 50% reliability) | about 260 (206-330 by season) |
| Year-to-year correlation of raw SM (n >= 300 both) | 0.58 |

Prior variants (full seasons): position means differ significantly (LR test, 2 df) in 7 of 10 seasons, but the effect
fades (centres +0.03 to +0.06, guards -0.02 to -0.04 in 2015-16 to 2018-19; not significant in 2020-21, 2023-24,
2024-25). Previous-season prior: slope 0.73-0.96, tau falls to about 0.051, LR 118-190 every season.

Half-season test (split at the median game date, everything estimated on the first half, target = second-half SM,
n2-weighted MSE, 95% CI from a bootstrap of players), difference to shrinkage with the league prior (x1000):

| Method | 2016-17 to 2024-25 (4,174) | 2025-26 (477) |
|---|---|---|
| raw first-half SM | +10.20 [8.53; 12.23] | +10.92 [5.77; 18.71] |
| everyone average (0) | +2.49 [1.92; 3.21] | +2.11 [1.06; 3.27] |
| position prior | -0.08 [-0.14; -0.01] | +0.18 [0.05; 0.33] |
| **previous-season prior** | **-0.72 [-0.91; -0.53]** | **-0.88 [-1.39; -0.40]** |
| position + previous | -0.74 [-0.93; -0.54] | -0.73 [-1.26; -0.25] |

Decision: **previous-season prior**; position priors are not justified (about 1% gain in development, zero for
players with >= 100 shots per half, worse on 2025-26).

Players: 2024-25 best Jokić (+0.23 [0.18; 0.28], about 305 points), Durant, Pritchard, Poeltl, Sabonis, Curry (+0.12).
2015-16 to 2024-25 pooled, 232 players with >= 3000 shots: 77 with interval above 0, 48 below. Top: Jokić +0.21,
Poeltl, Stephen Curry +0.14, Seth Curry, Joe Harris; bottom: Westbrook -0.12.
Curry: positive in all ten seasons (+0.08 to +0.15), 3PT 42.1% vs 37.3% expected; 2019-20 (80 shots, raw -0.07) shrunk
to +0.09.

Robustness: SM from the logit + context correlates 0.995 with LightGBM; centring per season x shot type: r = 0.989,
centres +0.037 -> +0.028, same top 10. SM vs mean p: Spearman -0.18. SM rises with volume.
Hold-out: 2024-25 shrunk estimates predict 2025-26 SM better than raw (MSE -4.9e-3 [-7.0e-3; -3.3e-3]; r 0.57 vs 0.49).

## 5. Open points

- No defender distance after 2014-15: part of SM is unobserved shot difficulty.
- Volume (log n) as a prior covariate raised the likelihood in a prototype but did not improve the n2-weighted test.
- One position per career; per-season positions would need team rosters (about 330 API calls).
- Stage 6 uses the same out-of-sample p for the Monte Carlo simulation.

## 6. Reproduction

```
python scripts/00_data/09_fetch_positions.py
# notebook: 10_players (needs oos_predictions.parquet from 06_main_model, holdout_predictions.parquet from 09_holdout)
```
