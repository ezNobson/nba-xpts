# nba-xpts

Expected points (xPTS) model for NBA shots, player shot-making analysis
and Monte Carlo game simulation. Engineering thesis, AGH.

## Setup
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

## Data pipeline
Run in order from the repo root (details in docs/data.md):

| Script | What it does | Output |
|---|---|---|
| `scripts/01_fetch_shots.py` | Download every regular-season shot, one request per team-season | `data/raw/shots/` (360 files) |
| `scripts/02_merge_shots.py` | Concatenate raw shot files, check duplicates and games per season | `data/interim/shots.parquet` |
| `scripts/03_fetch_games.py` | Download team box scores (leaguegamelog) | `data/raw/games/` (12 files) |
| `scripts/04_validate_shots.py` | Compare shot counts with box scores | printed report |

`scripts/probes/` holds one-off API checks that motivated the per-team download.
