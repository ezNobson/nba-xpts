# Stage 6: Monte Carlo of played games

This document describes the retrospective simulation of played games (RQ3: how much of a result is shot quality and
how much is luck in making the shots). All numbers come from `notebooks/11_simulation.ipynb`; data files are not
committed. The pre-game forecast (Stage 6b) is a separate step.

## 1. Purpose, inputs and outputs

| | File | Content |
|---|---|---|
| Input | `data/processed/oos_predictions.parquet` | out-of-sample p per shot, 2015-16 to 2024-25 (`gbm_w2_ctx`) |
| Input | `data/processed/holdout_predictions.parquet` | p for the hold-out season 2025-26 |
| Input | `data/processed/shots_features.parquet`, `shots_context.parquet` | shot value, period, `prev_event` (putback chains) |
| Input | `data/raw/games/` | box scores (FTM, FTA, PTS, FGA, OREB, TOV) |
| Code | `src/xpts/simulation.py` | `load_games`, `add_chains`, `putback_params`, `team_pmf`, `simulate_team`, `run_games` |
| Code | `src/xpts/players.py` | `season_estimates` (shooter skill for the skill-adjusted variant) |
| Output | `data/processed/game_simulations.parquet`, `game_simulations_holdout.parquet` | per game: P(win) MC / exact / normal, luck percentile, expected margin |
| Output | `reports/figures/06_*.pdf` | example game, convergence, calibration + luck percentile, team wins |

## 2. Method

- Everything that happened in a game is kept; only field goal results are drawn, shot i made with p_i. 10,000
  simulations per game, a tie counts 1/2. Free throws = real FTM (decision of 2026-11-17); drawing them with the team's
  previous-season FT% is a sensitivity check.
- 93 games (0.8%) whose shots + FTM do not reproduce the official points are dropped; 11,881 games remain.
- **Chains.** A shot after an offensive rebound that follows a miss of the same team in the same period continues the
  chain of that miss (10.2% of shots). A chain scores the value of its first made shot; later shots are dropped. If the
  real last shot of a chain was made but misses in the simulation, an offensive rebound follows with the league rate rho
  of the previous season (0.18-0.20) and a 2-point putback with the previous season's putback FG% (53-57%), at most 3.
- **Why chains:** the naive model (independent shots, fixed counts) counts the extra shots created by misses as
  extra expected points: its expected margin correlates 0.57 with the OREB difference and only 0.27 with the actual
  margin. Chains: 0.38 and 0.39.
- **Exact distribution:** per chain backwards (v_j with p_j, else the rest of the chain), convolution over chains and
  teams. Used to check Monte Carlo; the normal approximation uses its mean and variance.
- **Skill-adjusted variant:** p' = p + theta/v for non-heaves, theta = shrunk shot making of the shooter in the
  previous season (Stage 5), newcomers get the newcomers' mean. 2016-17 to 2024-25.

## 3. Results (2015-16 to 2024-25)

| Quantity | Value |
|---|---|
| MC - exact P(win) | mean abs 0.0035, max 0.021, 95.5% within 1.96 SE |
| Normal - exact | mean abs 0.0003 |
| Antithetic variates | variance ratio 0.68 |
| sd of margin from FG luck | 14.2 points (11.7-17.6) |
| Favourite by shot quality wins | 63.3% [62.5; 64.2] (naive model 59.1%) |
| R2 of expected margin | 0.152 (by season 0.09-0.22; naive 0.074) |
| Brier, ECE of P(win) | 0.226, 0.05 (too extreme at both ends) |
| Luck percentile | mean 0.507, sd 0.282 (uniform 0.289), chi2 rejects (p < 1e-7), small deviation |
| Skill-adjusted (2016-17+) | favourite 63.1% -> 65.8%, R2 0.150 -> 0.201, Brier -0.012 [-0.013; -0.011] |
| FT drawn instead of real | favourite 61.3%, R2 0.118 |
| Team-season W - xW | sd 9.1 wins; pure luck 4.0 |
| Persistence of W - xW (halves) | r 0.65 [0.58; 0.71] |
| Skill-adjusted W - xW | sd 7.3 wins, corr(W, xW) 0.83, persistence r 0.53 [0.43; 0.61] |
| Team shot making, halves | attack sd 0.039 pts/shot, r 0.64; defence sd 0.032, r 0.51 |
| Predict 2nd-half W% | Pythagorean (exp 14) best, W% next, xW% worse (+0.005 MSE) |

Hold-out 2025-26 (1,225 games): favourite 69.7% [67.3; 72.2], R2 0.33, ECE 0.035, skill-adjusted 70.7%. Stronger than in
development; probable reason: the v3 play-by-play has offensive rebounds missing in the v2 archive, so chains are found
more completely (putback share 11.2%). Not tested.

## 4. Conclusions (RQ3)

- One game: field goal luck is large (sd about 14 points), the team with the better shots wins only 63%; shot quality
  explains 15% of the variance of margins (20% with shooter skill).
- A season: luck in this narrow sense is about +-4 wins; the larger gap between shot quality and wins (+-9) persists and
  is team shot making in attack and defence, which the model (no defender, no player identity) cannot see.

## 5. Open points and limits

- Shot counts and turnovers fixed; generic putback after a simulated miss; real offensive rebounds after real misses
  are kept as they happened.
- v2 vs v3 play-by-play difference in putback detection (hold-out).

## 6. Stage 6b: pre-game forecast (`notebooks/12_forecast.ipynb`, `src/xpts/forecast.py`)

- Profile of a team = its previous K games (may reach into the previous season): chains per game, pools of first shots
  and putbacks (p, value), putback rate, FTM per game, team shot making shrunk with k = 1,659 shots (EB on 2015-16 to
  2022-23: tau 0.032 attack, 0.025 defence); the same allowed to opponents. Attack of A and defence of B combined
  additively around the previous season's league mean. Expected chains and FTM (stochastic rounding: independent random
  counts made the margin too wide), shots drawn from the pools, p + (SM_A + SM_allowed_B)/v, home advantage h.
- Development 2023-24 and 2024-25: K = 40 (from 10, 20, 40, 82), h = 1.97 points; calibrated version = logistic
  regression of the result on the expected margin. Baselines: home constant, win share and point differential of the
  last K games (logistic, K chosen the same way), Elo (FiveThirtyEight-style, K = 20, home +100, carry-over 0.75; box
  scores from 2014-15, neutral-site games without a home team skipped).
- Development log loss: point differential 0.6156, win share 0.6171, Elo 0.6286, shot model 0.6306.

Hold-out 2025-26 (1,225 games, run once):

| Method | Log loss | Brier | Accuracy |
|---|---|---|---|
| Shot model Monte Carlo | 0.6125 | 0.2116 | 0.662 |
| Shot model, calibrated | 0.6109 | 0.2107 | 0.660 |
| Point differential (40 games) | 0.6088 | 0.2097 | 0.682 |
| Win share (40 games) | 0.6165 | 0.2132 | 0.675 |
| Elo | 0.6178 | 0.2136 | 0.667 |
| Home team | 0.6875 | 0.2472 | 0.554 |
| Retrospective with real shots (not a forecast) | 0.5759 | 0.1966 | 0.697 |

- Paired bootstrap (games): calibrated - Elo -0.0070 [-0.0222; 0.0093], - point differential +0.0021 [-0.0050; 0.0095],
  - home -0.077 [-0.099; -0.055]. No significant difference to the standard methods.
- Logistic regression on Elo + shot model (descriptive, in-sample): shot model coefficient 0.48, LR test p = 0.0008.
- Shot model well calibrated, Elo overrates the home team; shot model best in October-November.
- Limits: no injuries, trades or rest; expected possessions and FT; additive defence; one hold-out season.
  Next step: combine the shot model with a results-based rating.

## 7. Reproduction

```
# needs oos_predictions.parquet (06_main_model), holdout_predictions.parquet (09_holdout),
# shots_context.parquet (06_build_pbp_features), positions (09_fetch_positions)
# notebooks: 11_simulation (about 25 minutes on CPU), 12_forecast (about 20 minutes)
```
