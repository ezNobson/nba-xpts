# nba-xpts

Expected points (xPTS) model for NBA shots, player shot-making analysis
and Monte Carlo game simulation. Engineering thesis, AGH.

## Setup
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .   # the xpts package in src/

## Data pipeline
Run in order from the repo root (details in docs/stages/01_data.md):

| Script | What it does | Output |
|---|---|---|
| `scripts/00_data/01_fetch_shots.py` | Download every regular-season shot, one request per team-season | `data/raw/shots/` (360 files) |
| `scripts/00_data/02_merge_shots.py` | Concatenate raw shot files, check duplicates and games per season | `data/interim/shots.parquet` |
| `scripts/00_data/03_fetch_games.py` | Download team box scores (leaguegamelog) | `data/raw/games/` (12 files) |
| `scripts/00_data/04_validate_shots.py` | Compare shot counts with box scores | printed report |
| `scripts/00_data/05_prepare_kaggle.py` | Check Kaggle Shot Logs, align GAME_ID with nba_api | `data/interim/kaggle_shot_logs.parquet` |
| `scripts/00_data/06_fetch_pullup_tracking.py` | Download tracking pull-up totals per team-season (label drift check, Stage 3) | `data/raw/tracking/pullup_team.parquet` |
| `scripts/00_data/07_fetch_pbp.py` | Download play-by-play events per season (nba_data archive, Stage 4) | `data/raw/pbp/` (11 files) |
| `scripts/00_data/08_fetch_pbp_v3.py` | Download play-by-play in the v3 format and convert to v2 columns (hold-out 2025-26) | `data/raw/pbp/pbp_2025-26.parquet` |
| `scripts/00_data/09_fetch_positions.py` | Download the listed position of every player (Stage 5 priors) | `data/raw/players/positions.parquet` |
| `scripts/02_features/06_build_pbp_features.py` | Per-shot game-context features from play-by-play (pre-shot events only) | `data/processed/shots_context.parquet` |

`scripts/00_data/probes/` holds one-off API checks that motivated the per-team download.

## Models
Baseline models (null, zone, spline logistic regression) and their validation: `src/xpts/`,
`notebooks/04_baselines.ipynb`, details in docs/stages/03_baselines.md.

Main model (LightGBM with play-by-play context), calibration, SHAP, MLP comparison and the Kaggle tracking
experiment: `notebooks/05_lightgbm.ipynb` to `08_kaggle_defence.ipynb`, final hold-out evaluation in `09_holdout.ipynb`, details in docs/stages/04_main_model.md.

## Players
Shot making SM = (PTS - xPTS) / n from out-of-sample predictions, empirical Bayes shrinkage, prior variants
(league, position, previous season) tested by predicting the second half of a season, Stephen Curry case study:
`src/xpts/players.py`, `notebooks/10_players.ipynb`, details in docs/stages/05_players.md.

## Simulation
Retrospective Monte Carlo of played games (real shots and free throws, field goal results drawn from out-of-sample p,
shots grouped into putback chains, exact convolution as a check), luck percentile, calibration of P(win), expected vs
actual wins: `src/xpts/simulation.py`, `notebooks/11_simulation.ipynb`. Pre-game forecast of 2025-26 by Monte Carlo from
team shot profiles, compared with Elo and simple baselines: `src/xpts/forecast.py`, `notebooks/12_forecast.ipynb`.
Details in docs/stages/06_simulation.md.
