# Data: NBA shot attempts (nba_api `shotchartdetail`)

This document describes the raw shot data, how it was downloaded, and how it was
turned into `data/interim/shots.parquet`. Data files are not committed to git;
they are fully reproducible with the scripts listed below.

## 1. Source

| Item | Value |
|---|---|
| Provider | stats.nba.com (official NBA statistics API) |
| Python client | `nba_api` 1.11 (`nba_api.stats.endpoints.shotchartdetail`) |
| Scope | Every field goal attempt (2PT and 3PT) in the regular season |
| Seasons | 2014-15 to 2025-26 (12 seasons) |
| Excluded | Playoffs, play-in, preseason, All-Star; free throws (not part of this endpoint) |
| Download date | 2026-10-17 |

### Request parameters

| Parameter | Value | Why |
|---|---|---|
| `team_id` | one team at a time | league-wide calls are truncated (see 2.1) |
| `player_id` | `0` | all players of the team |
| `season_nullable` | e.g. `"2014-15"` | one season per call |
| `season_type_all_star` | `"Regular Season"` | playoffs excluded |
| `context_measure_simple` | `"FGA"` | the default `"PTS"` returns **made shots only** |

## 2. Download (`scripts/fetch_shots.py`)

### 2.1 Why one request per team and season

A single league-wide request (`team_id=0`) for 2014-15 returned 106,496 shots
covering only games up to 2015-01-22 (640 of 1,230 games), without any error.
A request for one team (New Orleans, 2014-15) returned 6,795 shots, i.e. the
full 82 games. The API therefore silently caps the response size, so data is
requested per team and season: 30 teams x 12 seasons = 360 requests.
Team IDs are franchise IDs and are stable across all seasons in scope.

### 2.2 Rate limiting and robustness

- 1.0-1.5 s random pause between requests.
- On any error (timeout, connection reset, empty response): retry with
  exponential backoff (5, 10, 20, 40, 80 s + jitter), up to 5 attempts.
- One output file per team-season, written atomically (temporary file, then
  rename). Re-running the script skips files that already exist, so an
  interrupted download resumes where it stopped.

### 2.3 Output

`data/raw/shots/shots_{SEASON}_{TEAM_ID}.parquet`, 360 files. Each file holds all
shots **taken by that team** in that season (one row per shot); the opponent's
shots are in the opponent's file. The content is exactly the API response plus
one added column, `SEASON`. No filtering, type conversion or feature engineering
happens at this stage.

## 3. Processing to interim (`scripts/merge_shots.py`)

1. Read all 360 raw files and concatenate them row-wise (no aggregation, no rows
   or columns dropped).
2. Validate:
   - number of files = 360,
   - no duplicate shots on the key (`GAME_ID`, `GAME_EVENT_ID`),
   - games, shots, FG%, first/last game date and shots per game per season.
3. Save as `data/interim/shots.parquet` (26 MB).

### Validation results

Total: **2,525,326 shots**, **0 duplicates**.

| Season | Games | Shots | FG% | First game | Last game | Shots/game |
|---|---:|---:|---:|---|---|---:|
| 2014-15 | 1230 | 205,570 | 44.9% | 2014-10-28 | 2015-04-15 | 167.1 |
| 2015-16 | 1230 | 208,049 | 45.2% | 2015-10-27 | 2016-04-13 | 169.1 |
| 2016-17 | 1230 | 210,114 | 45.7% | 2016-10-25 | 2017-04-12 | 170.8 |
| 2017-18 | 1230 | 211,707 | 46.0% | 2017-10-17 | 2018-04-11 | 172.1 |
| 2018-19 | 1230 | 219,458 | 46.1% | 2018-10-16 | 2019-04-10 | 178.4 |
| 2019-20 | 1059 | 188,116 | 46.0% | 2019-10-22 | 2020-08-14 | 177.6 |
| 2020-21 | 1080 | 190,983 | 46.6% | 2020-12-22 | 2021-05-16 | 176.8 |
| 2021-22 | 1230 | 216,722 | 46.1% | 2021-10-19 | 2022-04-10 | 176.2 |
| 2022-23 | 1230 | 217,220 | 47.5% | 2022-10-18 | 2023-04-09 | 176.6 |
| 2023-24 | 1230 | 218,700 | 47.4% | 2023-10-24 | 2024-04-14 | 177.8 |
| 2024-25 | 1230 | 219,527 | 46.7% | 2024-10-22 | 2025-04-13 | 178.5 |
| 2025-26 | 1230 | 219,160 | 47.1% | 2025-10-21 | 2026-04-12 | 178.2 |

Every full season has 1,230 games (30 teams x 82 / 2). The two exceptions are
expected: 2019-20 was suspended by COVID-19 and finished in the Orlando bubble
(1,059 games), and 2020-21 was shortened to 72 games per team (1,080 games).

## 4. Columns

25 columns, one row per field goal attempt.

| Column | Type | Description |
|---|---|---|
| `GRID_TYPE` | str | Constant `"Shot Chart Detail"`. No information. |
| `GAME_ID` | str | Game identifier, e.g. `0021400001`: `002` = regular season, `14` = season start year, `00001` = game number. Must stay a string (leading zeros). |
| `GAME_EVENT_ID` | int | Event number within the game's play-by-play. (`GAME_ID`, `GAME_EVENT_ID`) uniquely identifies a shot. |
| `PLAYER_ID` | int | Shooter ID (stable across seasons). |
| `PLAYER_NAME` | str | Shooter name. |
| `TEAM_ID` | int | Shooter's team (franchise) ID. |
| `TEAM_NAME` | str | Team name at the time of the game. |
| `PERIOD` | int | Quarter 1-4; 5+ = overtime periods. |
| `MINUTES_REMAINING` | int | Minutes left in the period. |
| `SECONDS_REMAINING` | int | Seconds part of time left in the period. |
| `EVENT_TYPE` | str | `"Made Shot"` / `"Missed Shot"`. **Same information as the target: must never be used as a feature.** |
| `ACTION_TYPE` | str | Detailed shot type (e.g. `Jump Shot`, `Driving Layup Shot`, `Dunk Shot`). |
| `SHOT_TYPE` | str | `2PT Field Goal` / `3PT Field Goal`; gives the shot value (2 or 3). |
| `SHOT_ZONE_BASIC` | str | Court zone: `Restricted Area`, `In The Paint (Non-RA)`, `Mid-Range`, `Left Corner 3`, `Right Corner 3`, `Above the Break 3`, `Backcourt`. |
| `SHOT_ZONE_AREA` | str | Side of the court: `Center(C)`, `Left Side(L)`, `Left Side Center(LC)`, `Right Side(R)`, `Right Side Center(RC)`, `Back Court(BC)`. |
| `SHOT_ZONE_RANGE` | str | Distance band: `Less Than 8 ft.`, `8-16 ft.`, `16-24 ft.`, `24+ ft.`, `Back Court Shot`. |
| `SHOT_DISTANCE` | float | Distance to the hoop in feet, **rounded to whole feet**. |
| `LOC_X` | float | Horizontal position, tenths of a foot, hoop at 0 (range about -250 to 250). |
| `LOC_Y` | float | Position along the court, tenths of a foot, hoop at 0; baseline at about -52. |
| `SHOT_ATTEMPTED_FLAG` | int | Always 1 (every row is an attempt). No information. |
| `SHOT_MADE_FLAG` | int | **Target**: 1 = made, 0 = missed. |
| `GAME_DATE` | str | Game date as `YYYYMMDD`. |
| `HTM` | str | Home team abbreviation. |
| `VTM` | str | Visiting team abbreviation. |
| `SEASON` | str | Season, e.g. `2014-15`. **Added by `fetch_shots.py`**, not part of the API response. |

## 5. Known limitations

- **No free throws.** Only field goals are included. Game score simulation must
  add free throws from another source (e.g. the `leaguegamelog` endpoint).
- **Fouled missed shots are absent.** A missed shot on which the shooter is fouled
  is not counted as a field goal attempt, so it does not appear in the data (only
  the resulting free throws exist). The model does not see the value of drawing
  shooting fouls.
- **No defensive context.** There is no closest-defender distance, shot clock,
  dribbles or touch time. Shot quality is described by location, shot type and
  game time only.
- **Rounded distance.** `SHOT_DISTANCE` is in whole feet; the exact distance and
  angle should be computed from `LOC_X` / `LOC_Y`.
- **End-of-period heaves.** Long shots (30+ ft) are mostly desperation attempts
  at the buzzer and have very low FG%.
- **League drift.** Shots per game rose from 167 to 178 and FG% from 44.9% to
  47.1% between 2014-15 and 2025-26, so the data is not stationary over time.

## 6. Methodological decisions

- **Test set:** season 2025-26 is held out and is not used in EDA, feature
  design or model tuning. It is used only for the final evaluation.
- **Splitting:** train/validation/test splits are done by season or by
  `GAME_ID`, never by randomly sampling individual shots (shots from the same
  game would leak between sets).
- **Leakage:** `EVENT_TYPE` duplicates the target and must be dropped before
  modelling.

## 7. How to reproduce

```bash
source .venv/bin/activate
pip install -r requirements.txt
python scripts/fetch_shots.py   # ~15 min, resumable
python scripts/merge_shots.py   # writes data/interim/shots.parquet + prints checks
```
