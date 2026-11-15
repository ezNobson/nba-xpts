# Stage 3: Baseline models

This document describes the baseline models for the probability that a shot is made, how they are validated,
and the checks done on them. They are the reference for the main model in Stage 4 (gradient boosting).
All numbers come from `notebooks/04_baselines.ipynb`; data files are not committed (section 8).

## 1. Purpose, inputs and outputs

Purpose: set the bar that the main model has to beat, fix the validation scheme and the metrics, and find out
which features and design choices matter.

| | File | Content |
|---|---|---|
| Input | `data/processed/shots_features.parquet` | nba_api features (Stage 2), development seasons 2014-15 to 2024-25 only |
| Input | `data/raw/tracking/pullup_team.parquet` | tracking pull-up totals per team and season (only to check the label drift, section 5) |
| Code | `src/xpts/validation.py`, `src/xpts/metrics.py`, `src/xpts/baselines.py` | folds, metrics, models (package `xpts`, installed from `pyproject.toml`) |
| Output | `notebooks/04_baselines.ipynb` | all experiments, step by step with observations |
| Output | `reports/figures/03_calibration_zone.pdf`, `03_calibration_baselines.pdf` | thesis figures |

## 2. Validation and metrics

- **Walk-forward by season** (`walk_forward`): test on season S, train on the seasons before S. Ten folds,
  test seasons 2015-16 to 2024-25. `window=None` is the expanding window (all earlier seasons), `window=k` uses
  only the last k seasons. Shots are never split randomly, so no game or season is in both train and test.
- **Hold-out:** 2025-26 is never loaded in this stage.
- Everything that is fitted (zone table, spline basis, groups, coefficients) is fitted inside each fold on the
  training seasons only.
- **Metrics** (`evaluate`): log loss (main metric), Brier score, AUC. Reported as the mean over the ten test
  seasons. Calibration (`calibration_table`): mean prediction vs observed FG% in 20 quantile bins.
  Bias = observed FG% − mean prediction (positive = model too pessimistic).

## 3. Models

| Model | Code | Description |
|---|---|---|
| Null | notebook | every shot gets the league FG% of the training seasons |
| Zone | `ZoneBaseline` | FG% of the zone `SHOT_ZONE_BASIC` × `SHOT_ZONE_AREA` (16 zones), smoothed: `(makes + α p_league) / (attempts + α)`, α = 20 |
| Logit | `make_logit`, `ShotFeatures` | logistic regression, L2 penalty C = 1, on a sparse design matrix (below) |

Design matrix of the logit (`ShotFeatures`):
- cubic B-splines of `dist_ft` (clipped at 40 ft), knots `DIST_KNOTS` = 0, 1, 2, 3, 5, 8, 12, 16, 20, 22, 24, 26,
  30, 40 ft, chosen from the court geometry (dense near the rim and the 3PT line), not fitted on outcomes;
- one separate spline curve per `action_group`: spline basis × one-hot group, 10 × 16 = 160 columns;
- flags `is_three`, `is_corner3`, `is_heave`, `is_overtime`, `last_second` (`secs_left_period <= 1`);
- options: `group_col` (constant column = no shot type), `angle_knots` (spline of `angle_deg`, `ANGLE_KNOTS` every
  15°, clipped to ±90), `knots` (other distance knots).

## 4. Results

Mean over the ten test seasons:

| Model | Window | Log loss | Brier | AUC |
|---|---|---|---|---|
| Null | expanding | 0.6905 | 0.2487 | 0.500 |
| Zone | expanding | 0.6610 | 0.2342 | 0.625 |
| Logit without `action_group` | expanding | 0.6536 | 0.2308 | 0.640 |
| Logit | expanding | 0.6421 | 0.2264 | 0.653 |
| Logit | 3 seasons | 0.6409 | 0.2258 | 0.655 |
| **Logit** | **1 season** | **0.6402** | **0.2255** | **0.656** |

- The logit beats the zone model in every season. Distance and shot type carry much more information than the zones.
- The zone model under-predicts shots near the hoop (Restricted Area +2.3 pp, paint +1.7 pp), and its bias grows
  over time to about +2 pp, because 2PT FG% rose and the expanding window lags behind.
- With the expanding window the logit over-predicts 2015-19 (up to −1.2 pp) and under-predicts 2022-25
  (+1.2 to +1.7 pp). With a 1-season window the bias in 2023-25 is within ±0.2 pp. Exception: 2020-21
  (+1.6 pp with any window), the season after the COVID bubble.
- **Main baseline for Stage 4: logit with a 1-season window.**

## 5. Label drift of `ACTION_TYPE`

The sign change of the bias comes from the `ACTION_TYPE` labels: the same label covers a different mix of shots in
different seasons.

- `pullup_jumper`: 3.5% of shots in 2014-15 (FG 57.1%), 13.1% in 2024-25 (FG 37.6%). Plain `jumper`: 56.0% → 32.0%
  of shots, FG 33.7% → 37.5%.
- The label vocabulary changes abruptly: "Driving Jump shot" is 0.8% in 2014-15 and 0 afterwards, "Driving Floating
  Jump Shot" appears in 2015-16, all "Bank shot" variants disappear from 2017-18 on.
- Independent check with camera tracking (`scripts/00_data/06_fetch_pullup_tracking.py`, nba_api
  `LeagueDashPtStats`, `pt_measure_type="PullUpShot"`): share of pull-ups among all shots, 2014-15 vs 2024-25,
  95% CI for the difference of two proportions:

| Source | 2014-15 | 2024-25 | Difference | 95% CI |
|---|---|---|---|---|
| tracking | 23.91% | 23.95% | +0.04 pp | [−0.22; 0.29] |
| `pullup_jumper` label | 3.49% | 13.14% | +9.65 pp | [9.49; 9.81] |

  Players shoot off the dribble as often as before (tracking FG% stable at 37-39%); only the labelling changed.
  The tracking data are season totals and are used only for this check, never as model features.
- Consequences: `action_group` is kept (without it the log loss rises to 0.654), the model is trained on a short
  window, and the drift affects every model that uses `action_group`, including LightGBM.

## 6. Sensitivity checks (logit, 1-season window)

| Check | Variants | Log loss | Conclusion |
|---|---|---|---|
| Angle | none vs spline of `angle_deg` | 0.6402 vs 0.6403 | no gain, not used; kept as a candidate for LightGBM |
| Distance knots | geometric (14) vs uniform every 8 / 4 / 2 / 1 ft (6-41 knots) | 0.6401-0.6404 | insensitive; geometric knots kept, P-splines not needed |
| L2 strength C | 0.001 to 100 | 0.6402-0.6404 for C ≥ 0.1; 0.652 at C = 0.001 | flat around C = 1, no tuning needed |
| Zone smoothing α (zone model) | 0 to 10,000 | 0.6610 for α ≤ 100; 0.6622 at α = 10,000 | flat around α = 20, no tuning needed |

The defaults were set before these checks and were not changed after them, so no test season was used to
select a hyperparameter. The window size (section 4) is a choice made on the same ten test seasons; the
hold-out season 2025-26 remains untouched.

## 7. Open points for Stage 4

- Compare an expanding and a short window (or season weights) also for LightGBM.
- Tune the LightGBM hyperparameters inside each training window, without the test season of the fold.
- Season 2020-21 is poorly predicted by every window.
- A season trend feature as an alternative to the short window has not been tested.

## 8. Reproduction

```bash
python scripts/00_data/06_fetch_pullup_tracking.py   # tracking totals, only for section 5
jupyter nbconvert --to notebook --execute --inplace notebooks/04_baselines.ipynb
```
The notebook needs `data/processed/shots_features.parquet` from Stage 2 and the `xpts` package
(`pip install -e .`). A full run takes about 5 minutes.
