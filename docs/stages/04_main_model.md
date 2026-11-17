# Stage 4: Main model

This document describes the main model for the probability that a shot is made, the experiments that led to it
and its limits. All numbers come from the notebooks `05_lightgbm`, `06_main_model`, `07_mlp` and `08_kaggle_defence`;
data files are not committed. The hold-out evaluation is in `09_holdout`.

## 1. Purpose, inputs and outputs

Purpose: answer RQ1 (how well can a shot be predicted from its location and context, are the probabilities
calibrated, does defence information help) and deliver out-of-sample probabilities for Stage 5 (player xPTS).

| | File | Content |
|---|---|---|
| Input | `data/processed/shots_features.parquet` | shot features (Stage 2), development seasons 2014-15 to 2024-25 |
| Input | `data/raw/pbp/pbp_{season}.parquet` | play-by-play events, `scripts/00_data/07_fetch_pbp.py` |
| Input | `data/processed/kaggle_features.parquet` | Kaggle NBA Shot Logs 2014-15 matched to nba_api (Stage 2) |
| Code | `scripts/02_features/06_build_pbp_features.py` | context features -> `data/processed/shots_context.parquet` |
| Code | `src/xpts/gbm.py` | `GBMModel`, `tune_gbm`, `GBM_FEATURES`, `CONTEXT_FEATURES` |
| Code | `src/xpts/baselines.py` | `ContextFeatures`, `make_logit(context=True)` |
| Code | `src/xpts/mlp.py` | `MLPModel`, `tune_mlp` (PyTorch) |
| Code | `src/xpts/metrics.py` | `paired_bootstrap` |
| Output | `data/processed/holdout_predictions.parquet` | predictions of all models for the hold-out season 2025-26 |
| Output | `data/processed/oos_predictions.parquet` | out-of-sample predictions of all models per shot (test seasons 2015-16 to 2024-25) |
| Output | `reports/figures/04_*.pdf` | calibration, SHAP, MLP learning curves, Kaggle tracking |

## 2. Validation and statistical test

- Walk-forward by season as in Stage 3 (ten test seasons 2015-16 to 2024-25), hold-out 2025-26 never loaded.
- Every choice (training window aside, see section 4) is made inside the training window: for test season S the
  hyperparameters, number of trees and number of epochs are chosen on S-1 after training on S-2, then the model is
  refitted on both seasons.
- **Paired bootstrap** (`paired_bootstrap`): per-shot difference of log losses of two models on the same test shots;
  95% CI from 1000 resamples of whole **games** (shots within a game are not independent). Negative = first model better.
- "2018-19+" denotes the seasons after the `ACTION_TYPE` label change (Stage 3); it is a descriptive split.

## 3. Play-by-play context features

Source: stats.nba.com play-by-play from the `shufinskiy/nba_data` archive (Apache-2.0), 6.13 M events, seasons 2014-15
to 2024-25. Joined to shots on `GAME_ID` + `EVENTNUM` (= `GAME_EVENT_ID`): 99.97% of shots matched (lowest season
2019-20, 99.71%, three games missing), shot result agrees in 100% (99.99% in 2017-18).

| Feature | Definition |
|---|---|
| `prev_event` | last event before the shot in the same period, substitutions and replays skipped: made_shot, def_rebound, off_rebound, steal, dead_turnover, free_throw, foul, violation, timeout, jump_ball, period_start, other |
| `secs_since_prev` | game seconds from that event to the shot |
| `score_margin` | shooter's team score minus opponent score before the shot |
| `is_home` | shooter plays at home (agrees with `HTM` in 100%) |

Leakage: only rows before the shot are used; the score margin is shifted by one event because the score on a made
shot's row already contains it. Timestamps are entered by hand (about 1-2 s precision). Unmatched shots have NaN.
The archive has 2025-26 only in the `nbastatsv3` format, so the hold-out features need an adapter (open point).

Raw FG% by `prev_event` (all development seasons): steal 56.8%, off_rebound 49.9%, def_rebound 46.4%, made_shot 44.5%,
timeout 42.7%.

## 4. Experiments and results

Mean over the ten test seasons.

| Model | Window | Log loss | Brier | AUC |
|---|---|---|---|---|
| Logit (Stage 3 baseline) | 1 | 0.6402 | 0.2255 | 0.656 |
| LightGBM, default | 1 | 0.6413 | 0.2260 | 0.654 |
| LightGBM, default | 2 | 0.6412 | 0.2260 | 0.654 |
| LightGBM, default | 3 | 0.6407 | 0.2257 | 0.656 |
| LightGBM, default | expanding | 0.6418 | 0.2263 | 0.654 |
| LightGBM, tuned | 2 | 0.6410 | 0.2259 | 0.655 |
| Logit + context | 1 | 0.6390 | 0.2250 | 0.658 |
| **LightGBM + context (main model)** | 2 | **0.6390** | 0.2251 | 0.658 |
| MLP + context, tuned | 2 | 0.6399 | 0.2254 | 0.658 |

Paired bootstrap, log loss difference [95% CI]:

| Comparison | All seasons | 2018-19+ |
|---|---|---|
| LightGBM w2 − logit | +0.0010 [0.0009; 0.0011] | −0.00005 [−0.00015; 0.00005] |
| LightGBM tuned − default (w2) | −0.0002 [−0.0003; −0.0002] | +0.00001 [−0.00005; 0.00007] |
| LightGBM + context − LightGBM | −0.0023 [−0.0024; −0.0022] | −0.0014 [−0.0015; −0.0013] |
| Logit + context − logit | −0.0012 [−0.0013; −0.0011] | −0.0009 [−0.0010; −0.0008] |
| LightGBM + context − logit + context | −0.00006 [−0.00017; 0.00006] | −0.0005 [−0.0006; −0.0004] |
| MLP + context − LightGBM + context | +0.0009 [0.0008; 0.0011] | +0.0016 [0.0014; 0.0017] |

Findings:
- **Model class and tuning change little.** Without context LightGBM does not beat the spline logit; tuning
  (`num_leaves` {7, 15, 31, 63} x `min_child_samples` {20, 100, 500}, early stopping) gains 0.0002 and the chosen
  parameters jump between folds (flat surface). Default parameters are kept.
- **Training window.** The expanding window is the worst from 2018-19 on (old labels hurt); 2 seasons is the best
  LightGBM window in recent seasons.
- **Context helps both models**; almost all of it is `prev_event` + `secs_since_prev` (home + margin alone: −0.00016).
- **Same information, same result overall**: LightGBM and the logit with context tie over ten seasons. The logit is
  much better in 2017-18 (label change, +0.0056 for LightGBM); LightGBM is better in each of 2021-22 to 2024-25
  (−0.0005 to −0.0010, CIs below zero).
- **MLP** (embeddings, 2-3 layers, grid of 16, tuned on S-1) is worse than both in 9 of 10 seasons; tuning always
  picks the small network.
- **Main model: LightGBM, 2-season window, default parameters, shot + context features.** Reference model: logit +
  context.

## 5. Calibration

- Calibration error per bin within ±1.5 p.p. for all models; ECE 0.005-0.007; mean bias below 0.3 p.p.
- Bias per season is larger: −1.2 p.p. (2015-16) to +1.6 p.p. (2020-21), the same for every model (league FG% drift).
- Subgroups (LightGBM + context, all seasons): layups +1.7 p.p., pull-up jumpers −2.2 p.p., mostly from 2017-18 and
  2018-19. In 2021-22 to 2024-25 all shot types within ±0.5 p.p. except layups (+1.9 p.p.).
- **Isotonic recalibration** (map fitted on the out-of-sample predictions of S-1) makes the log loss worse
  (+0.0005 [0.0005; 0.0006] LightGBM, +0.0010 logit) and does not reduce the season bias. Not used.

## 6. Interpretation (SHAP)

TreeSHAP (`pred_contrib=True`), model of the last fold, 100,000 shots of 2024-25. Mean |SHAP| (log-odds): `dist_ft`
0.35, `action_group` 0.18, `secs_since_prev` 0.057, `LOC_Y` 0.051, `prev_event` 0.046, `LOC_X` 0.041,
`secs_left_period` 0.038; score margin, home court and the flags below 0.02. A shot 2-4 s after a steal gets about
+0.4 log-odds; the bonus disappears after about 6 s. Shots right after an offensive rebound have a negative
contribution given their location and type. Dunks +1.4 log-odds on top of distance.

## 7. Defence and tracking features (Kaggle 2014-15)

LightGBM with default parameters, walk-forward by month (test December, January, February-March), 92,969 test shots.

| Feature set | Log loss | AUC |
|---|---|---|
| A shot | 0.6393 | 0.663 |
| B shot + context | 0.6360 | 0.667 |
| C shot + tracking | 0.6295 | 0.677 |
| D shot + context + tracking | 0.6270 | 0.681 |

D − B = −0.0090 [−0.0100; −0.0081]; B − A = −0.0033 [−0.0040; −0.0026]; D − C = −0.0026 [−0.0032; −0.0020].
One tracking feature added to B: defender distance −0.0048, touch time −0.0035, catch-and-shoot −0.0022, dribbles
−0.0018, shot clock −0.0016. Defender distance is the third most important feature of model D.

Answer to RQ1 (defence): tracking information clearly helps and is not available after 2016, so it is the main
limitation of the model. Part of what Stage 5 measures as shot making is unobserved shot difficulty.

## 8. Final evaluation on the hold-out season 2025-26

Run once in `09_holdout.ipynb` (2026-11-17), after all choices were fixed. Models trained on the seasons just before
2025-26 (as in the last walk-forward fold); context features from the v3 play-by-play format
(`scripts/00_data/08_fetch_pbp_v3.py`, checked on 2024-25 against v2: is_home 100%, score_margin 99.997%,
secs_since_prev 99.72%, prev_event 99.53% identical; v3 contains rebounds missing in the v2 archive).

| Model | Log loss | Brier | AUC | Bias (p.p.) |
|---|---|---|---|---|
| **LightGBM + context (main)** | **0.6387** | **0.2248** | **0.660** | −0.04 |
| Logit + context | 0.6396 | 0.2252 | 0.659 | +0.07 |
| MLP + context | 0.6405 | 0.2255 | 0.659 | −1.74 |
| LightGBM | 0.6399 | 0.2253 | 0.658 | −0.04 |
| Logit (Stage 3) | 0.6405 | 0.2255 | 0.657 | +0.05 |
| Zone | 0.6570 | 0.2322 | 0.637 | +0.13 |
| Null | 0.6915 | 0.2492 | 0.500 | +0.38 |

Main model minus: logit + context −0.0009 [−0.0011; −0.0006], MLP −0.0018 [−0.0021; −0.0014], LightGBM without context
−0.0012 [−0.0014; −0.0010], Stage 3 logit −0.0017 [−0.0020; −0.0014], zone −0.0182 [−0.0191; −0.0173].
ECE 0.0055; every shot type within ±1.2 p.p.; league expected 1.0947 vs actual 1.0913 points per shot.
Predictions: `data/processed/holdout_predictions.parquet`. Figure: `reports/figures/04_calibration_holdout.pdf`.

## 9. Open points

- Stage 5: centre shot making per season (season bias up to 1.6 p.p. in development); use out-of-sample predictions
  only (`oos_predictions.parquet` for 2015-16 to 2024-25, `holdout_predictions.parquet` for 2025-26).
- Optional: hot-hand feature, opponent defence strength, XGBoost as a robustness check.
- Layup bias in 2021-22 to 2024-25 (+1.9 p.p.) not explained (+0.5 p.p. in 2025-26).

## 10. Reproduction

```
python scripts/00_data/07_fetch_pbp.py
python scripts/00_data/08_fetch_pbp_v3.py --year 2025   # hold-out season, v3 format
python scripts/02_features/06_build_pbp_features.py
# notebooks in order: 05_lightgbm, 06_main_model, 07_mlp (GPU optional), 08_kaggle_defence, 09_holdout (run once)
```
`06_main_model` writes `oos_predictions.parquet`, `07_mlp` adds the MLP column to it.
