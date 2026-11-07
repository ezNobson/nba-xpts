# Stage 2: Exploratory analysis and feature engineering

This document describes how the shot data from Stage 1 was checked, turned into model features, and explored.
It covers the nba_api shots (all seasons) and the Kaggle NBA Shot Logs 2014-15 (tracking context).
Data files are not committed; everything is reproducible with the scripts listed in section 9.

## 1. Purpose, inputs and outputs

Purpose: build per-shot features for the expected points (xPTS) model, check data quality, and use exploratory
analysis (EDA) to decide which features are worth modelling.

| | File | Content |
|---|---|---|
| Input | `data/interim/shots.parquet` | nba_api shots, 2014-15 to 2025-26, 2,525,326 rows (Stage 1) |
| Input | `data/interim/kaggle_shot_logs.parquet` | Kaggle Shot Logs 2014-15, 128,069 rows (Stage 1) |
| Output | `data/processed/shots_features.parquet` | nba_api features, 2,524,965 rows, 25 columns |
| Output | `data/processed/kaggle_nba_match.parquet` | Kaggle row ↔ nba_api shot mapping, 126,838 rows |
| Output | `data/processed/kaggle_features.parquet` | matched shots with nba_api and Kaggle features, 126,838 rows, 35 columns |
| Output | `notebooks/02_eda_features.ipynb`, `notebooks/03_eda_kaggle.ipynb` | EDA, step by step with observations |
| Output | `reports/figures/02_*.pdf` | thesis figures |

## 2. Leakage policy

- **Row-wise features only.** Every feature is a deterministic function of a single shot. No statistic is fitted on
  the data in this stage (no means, target or frequency encoding, scaling, imputation by averages). Anything that
  is fitted belongs to the model pipeline and is fitted on training data only. Because of this, the feature
  scripts can process all seasons at once, including the test season, without leakage.
- **Hold-out season.** 2025-26 is the final test set. It is never used in EDA, figures or feature decisions; all
  statistics in this document use the development seasons 2014-15 to 2024-25 only. (The Kaggle data covers only
  2014-15, so it never touches the test season.)
- **Outcome columns are never features:** nba_api `EVENT_TYPE`; Kaggle `SHOT_RESULT`, `FGM`, `PTS`, `W`,
  `FINAL_MARGIN`. The label is always nba_api `SHOT_MADE_FLAG`.
- **EDA informs feature design only.** No number computed in the notebooks is passed to a model.

## 3. Geometry check (`scripts/02_features/01_check_geometry.py`)

The nba_api coordinates are in tenths of a foot with the hoop at (0, 0); `LOC_X` runs across the court
(-250 to 250), `LOC_Y` towards half court (baseline at about -52). Development seasons, 2,306,166 shots:

| Check | Result |
|---|---|
| exact distance `hypot(LOC_X, LOC_Y) / 10` minus `SHOT_DISTANCE` | always in [0, 1), mean 0.48 |
| share where `floor(exact) == SHOT_DISTANCE` | 100% of shots with coordinates (`round` matches only 51%) |
| shots with missing `LOC_X` / `LOC_Y` | 361 |
| 3PT shots closer than 22 ft | 352 |
| 2PT shots beyond 23.75 ft outside the corner (`LOC_Y > 92.5`) | 101 |
| 2PT shots beyond 22 ft in the corner (`LOC_Y <= 92.5`) | 613 |
| shots with `LOC_Y < -52` | 0 |

Conclusions: `SHOT_DISTANCE` is the exact distance truncated to whole feet, so the exact distance is computed
from the coordinates instead. The 361 shots without coordinates are dropped. About 0.05% of shots disagree with
the 3PT line geometry; this is label noise in the official data and those shots are kept.

## 4. nba_api features (`scripts/02_features/02_build_features.py`)

Output `data/processed/shots_features.parquet`: 2,524,965 rows (all 12 seasons), 25 columns, no missing values.

### 4.1 Columns

| Column | Formula / source | Meaning |
|---|---|---|
| `GAME_ID`, `GAME_EVENT_ID` | nba_api | shot key; identifiers, not features |
| `PLAYER_ID`, `PLAYER_NAME`, `TEAM_ID` | nba_api | shooter and team; identifiers (player analysis) |
| `SEASON`, `GAME_DATE`, `HTM`, `VTM` | nba_api | season, date, home / visiting team; used for splits |
| `SHOT_MADE_FLAG` | nba_api | **label**: 1 made, 0 missed |
| `LOC_X`, `LOC_Y` | nba_api | raw shot position (tenths of a foot) |
| `dist_ft` | `hypot(LOC_X, LOC_Y) / 10` | exact distance to the hoop (ft) |
| `angle_deg` | `degrees(arctan2(LOC_X, LOC_Y))` | shot angle: 0 = straight on, ±90 = along the baseline through the hoop, beyond ±90 = behind the backboard line; the sign gives the side |
| `is_three` | `SHOT_TYPE == "3PT Field Goal"` | 3-point attempt (0/1) |
| `shot_value` | `2 + is_three` | points if made; xPTS = P(made) × `shot_value` |
| `is_corner3` | `is_three` and `abs(LOC_X) >= 220` and `LOC_Y <= 92.5` | corner 3 (straight part of the 3PT line, 22 ft) |
| `SHOT_ZONE_BASIC`, `SHOT_ZONE_AREA` | nba_api, categorical | official court zone (7 levels) and side (6 levels) |
| `ACTION_TYPE` | nba_api | raw shot description (64 values) |
| `action_group` | keyword rules, section 4.2 | shot type in 10 groups |
| `PERIOD` | nba_api | 1-4 quarters, 5+ overtime |
| `is_overtime` | `PERIOD >= 5` | overtime (0/1) |
| `secs_left_period` | `MINUTES_REMAINING * 60 + SECONDS_REMAINING` | seconds left in the period |
| `is_heave` | `dist_ft >= 30` and `secs_left_period <= 3` | desperation shot at the end of a period |

The first version of `is_corner3` used only `LOC_Y <= 92.5` and also flagged mislabelled "3s" near the hoop.
Adding `abs(LOC_X) >= 220` removed 607 shots (204,255 → 203,648; FG% unchanged at 38.7%).

### 4.2 `ACTION_TYPE` grouping

The 64 raw shot descriptions are mapped to 10 groups by fixed keyword rules (substring match, applied in this
order, first match wins). The rules are written by hand, not learned from data, so they cause no leakage.

| Order | Keyword(s) | Group |
|---|---|---|
| 1 | Dunk | `dunk` |
| 2 | Tip, Putback | `tip_putback` |
| 3 | Layup | `layup` |
| 4 | Hook | `hook` |
| 5 | Floating | `floater` |
| 6 | Step Back | `stepback_jumper` |
| 7 | Pull | `pullup_jumper` |
| 8 | Fadeaway, Turnaround | `fadeaway_jumper` |
| 9 | Jump | `jumper` |
| – | anything else | `other` |

The order matters: "Tip Dunk Shot" becomes `dunk`, "Turnaround Hook Shot" becomes `hook`, "Running Pull-Up Jump
Shot" becomes `pullup_jumper`. The group `other` (3,194 shots) is mostly "Driving Bank shot" and "Running Bank shot",
plus 81 "No Shot".

### 4.3 Removed columns

| Column | Reason |
|---|---|
| `EVENT_TYPE` | "Made Shot" / "Missed Shot": the label itself (leakage) |
| `GRID_TYPE`, `SHOT_ATTEMPTED_FLAG` | constant |
| `SHOT_DISTANCE` | truncated distance, replaced by `dist_ft` |
| `SHOT_TYPE` | replaced by `is_three` / `shot_value` |
| `SHOT_ZONE_RANGE` | distance band, redundant with `dist_ft` |
| `TEAM_NAME` | redundant with `TEAM_ID` |
| `MINUTES_REMAINING`, `SECONDS_REMAINING` | combined into `secs_left_period` |

## 5. EDA of nba_api features (`notebooks/02_eda_features.ipynb`)

Development seasons 2014-15 to 2024-25: 2,305,805 shots, FG% 46.2%, 1.053 points per shot.

| Figure | Key numbers | Consequence |
|---|---|---|
| `02_fg_vs_distance.pdf`: FG%, points per shot (2PT / 3PT) and shot counts by distance, 1-ft bins | FG% 75.7% at 0-1 ft, 45.4% at 3-4 ft, flat at about 40-43% from 4 to 20 ft. At the 3PT line FG% barely changes (2s at 21-22 ft 37.5%, 3s at 22-24 ft 38.7%) but points per shot do: the longest 2s give about 0.68-0.75 points, 3s from the same range about 1.16, mid-range 2s 0.79-0.85, shots at the rim 1.51 | the distance effect is strongly non-linear |
| `02_shot_chart.pdf`: hexbin maps of shot frequency and FG% | shots concentrate at the rim, in the corners and just behind the arc; FG% is high only near the hoop; court lines match the shot pattern | coordinates and conversion to feet are correct |
| `02_three_by_angle.pdf`: 3PT FG% by absolute angle, 5° bins | flat at 35.0-35.8% from 0 to 50°, rising to 39.4% at 90-95°; corner 3s 38.7% vs above-the-break 35.5% | keep `is_corner3` / `angle_deg` |
| `02_fg_by_action_group.pdf`: FG% by action group | dunk 89.5%, layup 56.0%, tip/putback 55.9%, hook 48.4%, floater 45.4%, fadeaway 43.1%, pull-up 42.2%, step-back 40.4%, jumper 35.3%, other 66.6% | strong feature, to be used together with distance |
| `02_end_of_period.pdf`: FG% in the last 24 s, shots under / over 30 ft | 30+ ft shots at 0 s: 5.6% (7,701 shots), at 1 s: 7.3%; shots under 30 ft also drop at the buzzer: 29.6% at 0 s, 32.4% at 1 s vs about 42-46% earlier | `is_heave` captures desperation shots; the buzzer effect also affects short shots |
| `02_league_trend.pdf`: 3PT share and 2PT / 3PT FG% by season | 3PT share 26.8% → 42.1%; 2PT FG% 48.5% → 54.5%; 3PT FG% stable at 35-37% | data is not stationary: validate by time |

## 6. Kaggle Shot Logs 2014-15

The Kaggle data adds tracking context that nba_api does not have: distance to the closest defender, shot clock,
dribbles and touch time. It covers 904 games of 2014-15 played up to 2015-03-04.

### 6.1 Diagnostics (`scripts/02_features/03_check_kaggle.py`)

| Check | Result |
|---|---|
| `SHOT_CLOCK` missing | 5,567 shots: 3,550 with less than 24 s left in the period (the shot clock is turned off; 64% of such shots have no value), 2,017 (1.6%) otherwise |
| `SHOT_CLOCK == 24.0` | 3,857 shots, mostly putbacks (median `SHOT_DIST` 2.3 ft, 90% without a dribble, FG% 60%): the clock had not restarted after an offensive rebound |
| `TOUCH_TIME` | 312 negative values (min -163.6 s; 109 below -1 s); 3,046 zeros, almost all with 0 dribbles |
| `CLOSE_DEF_DIST`, `DRIBBLES` | plausible (median 3.7 ft, max 53.2 ft; max 32 dribbles) |
| `PTS_TYPE` vs `SHOT_DIST` | 872 3s under 22 ft and 539 2s beyond 23.75 ft (1.1%, vs 0.05% in nba_api): Kaggle distance is less precise |
| outcome columns | `SHOT_RESULT` matches `FGM` exactly, `PTS == FGM × PTS_TYPE` for all rows |

### 6.2 Matching Kaggle shots to nba_api (`scripts/02_features/04_match_kaggle.py`)

The Kaggle file has no `GAME_EVENT_ID`, so shots are matched on game, shooter, period and game clock.

- **Clock offset.** The Kaggle game clock runs 1-4 s behind nba_api (Kaggle shows more time left; most common
  difference +2 s, e.g. Kaggle 10:38 vs nba_api 10:36). Matching on the exact clock or within ±1 s found only 12.5%
  of the shots.
- **Rule.** Candidate pairs: same `GAME_ID`, shooter (`player_id` = `PLAYER_ID`) and `PERIOD`, with a clock
  difference (Kaggle − nba_api) in [-1, 5] s. Pairs are assigned greedily, those closest to +2 s first, so that
  each shot is used at most once on both sides. Preferring exact clock matches made results worse (outcome
  agreement 92.6%), because with the typical offset an exact match often pairs a shot with a neighbouring one
  such as a putback.
- **Outcome is not a key.** `FGM` and `PTS_TYPE` are used only to validate the matching. Using the outcome to match
  (or to drop disagreeing pairs) would select rows based on the label and bias any model trained on them.
  Pairs with disagreeing outcome are kept.

| Result | Value |
|---|---|
| matched Kaggle shots | 126,838 of 128,069 (99.04%) |
| clock difference of matched pairs | -1 s: 352, 0: 2,842, 1: 11,927, 2: 55,568, 3: 45,036, 4: 8,893, 5: 2,220 |
| duplicates | 0 Kaggle rows and 0 nba_api shots used twice |
| `FGM == SHOT_MADE_FLAG` | 99.74% (lowest at the window edges: 91.8% at -1 s, 98.7% at 5 s) |
| `PTS_TYPE == shot_value` | 99.98% |
| `SHOT_DIST − dist_ft` | median +0.5 ft, IQR -0.6 to 2.0 ft, extremes -80.6 / +29.0 ft |
| unmatched | 1,231; all shooters do have nba_api shots in that game, only outside the clock window |

### 6.3 Kaggle features (`scripts/02_features/05_build_kaggle_features.py`)

Output `data/processed/kaggle_features.parquet`: 126,838 matched shots, all columns of `shots_features.parquet`
plus the columns below. Location, shot value and the label come from nba_api. Dropped Kaggle columns: `W`,
`FINAL_MARGIN`, `PTS`, `SHOT_RESULT`, `FGM` (outcome, leakage) and `SHOT_DIST`, `PTS_TYPE` (nba_api used instead).

| Column | Rule (per row) | Meaning |
|---|---|---|
| `close_def_dist` | `CLOSE_DEF_DIST` | distance to the closest defender (ft) |
| `shot_clock_off` | `SHOT_CLOCK` missing and Kaggle clock < 24 s left in the period | shot clock turned off (2.8% of shots) |
| `shot_clock` | `SHOT_CLOCK`; if `shot_clock_off`, the seconds left in the period | seconds left on the shot clock |
| `shot_clock_missing` | `shot_clock` still missing | missing for another reason (1,986 shots, 1.6%) |
| `dribbles` | `DRIBBLES` | dribbles before the shot |
| `touch_time` | `TOUCH_TIME`, values < 0 set to missing (282) | seconds the shooter held the ball |
| `is_catch_shoot` | `dribbles == 0` and `touch_time <= 2` (0 if `touch_time` missing) | catch-and-shoot (46.2% of shots) |
| `CLOSEST_DEFENDER_PLAYER_ID` | Kaggle | identifier only |
| `kaggle_secs_left` | parsed Kaggle `GAME_CLOCK` | Kaggle game clock (seconds left in the period) |
| `k_row` | row number in the Kaggle file | link to the source row |

`shot_clock == 24.0` is kept as is for 2s (real putbacks, see 6.4). The 68 3PT shots with `shot_clock == 24.0` are
almost certainly clock recording errors and are to be treated as missing at the modelling stage.

### 6.4 EDA of Kaggle features (`notebooks/03_eda_kaggle.ipynb`)

126,362 matched shots without heaves (476 removed), FG% 45.4%.

| Figure | Key numbers | Consequence |
|---|---|---|
| `02_kaggle_defender_by_distance.pdf`: FG% by closest defender distance within shot groups (2PT 0-4 ft, 4-14 ft, 14+ ft, 3PT) | FG% rises with defender distance in every group: at the rim 50.4% (defender within 1 ft) → 86.5% (6-10 ft) → 94.2% (10+ ft); short 2s 35.8% → 50.8%; long 2s 30.0% → 45.8%; 3s 20.5% → 38.1% (6-10 ft) → 41.5% (10+ ft) | strongest tracking feature, but only when shot distance is controlled for |
| (same figure, confounding) | without the split FG% hardly changes with defender distance (45.3%, 46.9%, 43.2%, 44.7% for 0-2, 2-4, 4-6, 6+ ft), because the median shot distance is 2.3 ft with the defender within 1 ft and 23.2 ft with the defender 6-10 ft away | defender distance must be modelled together with shot distance |
| `02_kaggle_shot_clock.pdf`: FG% by shot clock, 2PT / 3PT | 0-2 s left: 2PT 34.0%, 3PT 24.3% vs about 47% / 37% at 10-16 s; early-clock 2s 66.2% at 20-22 s (putbacks, fast breaks); at exactly 24.0 s 3,665 shots, median 1.2 ft, 54% tip/putback, 21% layup, 16% dunk | hurried late-clock shots are worse; 24.0 s values are real putbacks |
| `02_kaggle_dribbles.pdf`: FG% by dribbles, 2PT / 3PT | 2PT 54.5% with 0 dribbles, 47.6% with 1, about 43-45% with 2+; 3PT 36.6% with 0, 28.0% with 2 | main difference is 0 vs 1+ dribbles |
| `02_kaggle_catch_and_shoot.pdf`: catch-and-shoot vs other | 3PT 36.8% (n = 24,724) vs 31.9% (n = 8,462); 2PT jump shots from 10+ ft 42.3% vs 38.5% | about +4-5 percentage points off the catch |

## 7. Decisions

- Exact distance and angle computed from coordinates; `SHOT_DISTANCE` and `SHOT_TYPE` replaced.
- 361 shots without coordinates dropped; shots inconsistent with the 3PT line kept (label noise, about 0.05%).
- `is_corner3` requires `abs(LOC_X) >= 220` and `LOC_Y <= 92.5`.
- `ACTION_TYPE` grouped by fixed keyword rules into 10 groups; the raw value is kept for analysis. The small `other` group is kept as is: re-assigning it after seeing its FG% would fit the rules to the outcome.
- Heaves are kept in the data and flagged (`is_heave`), not removed.
- Kaggle shots are matched to nba_api by game, shooter, period and clock with an offset window of [-1, 5] s; the
  outcome is used only for validation and disagreeing pairs are kept.
- **The main xPTS model uses nba_api features only**, because the Kaggle features exist only for part of 2014-15
  and not for the test season. The Kaggle features are used in a separate comparison experiment (section 8).
- Splits are made by season / date or by game, never by randomly sampling individual shots.

## 8. Open issues for the modelling stage

- **End-of-period flag:** shots under 30 ft also drop in the last 1-2 s. A flag for `secs_left_period <= 1` is
  likely needed for logistic regression (tree models can use `secs_left_period` directly).
- **Logistic regression needs splines and interactions:** the effect of distance is strongly non-linear and differs
  by shot type (e.g. the logit of FG% falls steeply with distance for layups, but rises from 4 to 16 ft for
  jumpers). A plain linear term in `dist_ft` cannot represent this; use splines of `dist_ft` (e.g.
  `SplineTransformer`) and their interactions with `action_group`. Gradient boosting is expected to capture
  this directly.
- **Shot zones** (`SHOT_ZONE_BASIC`, `SHOT_ZONE_AREA`) partly duplicate distance and angle; test whether they add
  anything.
- **Time drift:** 2PT FG% rises over the seasons as the shot mix changes; test whether a season / time feature is
  needed with a time-based validation (train on older seasons, validate on newer ones).
- **Kaggle comparison experiment** on the matched 2014-15 shots, split by date (earlier games for training, later
  games for testing): model A with nba_api features only vs model B with nba_api + `close_def_dist`,
  `shot_clock`, `dribbles`, `touch_time`, `is_catch_shoot`; compare log-loss, Brier score, AUC and calibration,
  and how player shot-making estimates change. The 68 3PT shots with `shot_clock == 24.0` are treated as missing.

## 9. How to reproduce

```bash
source .venv/bin/activate
python scripts/02_features/01_check_geometry.py        # geometry checks (prints only)
python scripts/02_features/02_build_features.py        # -> data/processed/shots_features.parquet
python scripts/02_features/03_check_kaggle.py          # Kaggle diagnostics (prints only)
python scripts/02_features/04_match_kaggle.py          # -> data/processed/kaggle_nba_match.parquet
python scripts/02_features/05_build_kaggle_features.py # -> data/processed/kaggle_features.parquet
jupyter nbconvert --to notebook --execute --inplace notebooks/02_eda_features.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/03_eda_kaggle.ipynb
```
