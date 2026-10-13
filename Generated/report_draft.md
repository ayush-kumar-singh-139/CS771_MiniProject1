# CS771 Mini-Project 1 — Next-day PM2.5 forecasting and PM2.5 maps for Bihar

*Group: ‹names / roll numbers›. Code: `part0_kanpur_cleaning.ipynb` (Part 0), `mini_project_1.ipynb` (Parts 1–5), shared modules in `src/`. Figures are in `figures/`, and every table is also saved as a CSV in `results/`.*

**Protocol.**
* **Task:** forecasts are issued at 12:00 IST on day *d* for 12:00 on *d*+1, using only readings timestamped ≤ the issue time.
* **Split by target date:** train = 12:00 targets 1 Jan–31 Aug (capped at **31 Aug 12:00**, the first validation issue time); validation = all 91 targets 1 Sep–30 Nov × 468 stations = 42,588 forecasts, nothing dropped.
* **Fitted statistics:** scalers, station means, PCA and K-means are fitted on training rows only. All choices were made on validation.
* **Leakage checks:** every lag is asserted to be ≥ 0, and a "future-poisoning" test overwrites all readings after the issue time with 10⁶ and verifies that no feature changes. The two persistence computations agree exactly, and the validation count is asserted.

## Part 0 — Cleaning the raw Kanpur data

**Format.**
* **Sensors:** 76 — 44 with 10-digit numeric IDs, 31 with 8-character IDs, and one 12-character hex ID (`AC0BFBE34CC7`).
* **Reporting:** the file is on a 15-minute UTC grid, but the 10-digit and 8-char sensors report **once per hour, always at minute :00**; only the hex sensor uses all four slots. Median share of hours reported: 0.80 (10-digit), 0.68 (8-char), 0.82 (hex).
* **Movement:** none — every sensor has one fixed coordinate for both months.
* **ID styles:** 10-digit sensors report integers; 8-char sensors report 4-decimal values (device-side averages). Compared hour by hour, the two families' medians differ by only ≈ 2 µg/m³ (ratio of means 1.03). The hex sensor reads lower (median 25 vs 33–37 µg/m³).

**Hourly reshaping.** We built a long table `sensor_id, timestamp_utc, latitude, longitude, pm25`, averaging sub-hourly readings and labelling each hour by its start. Error codes and out-of-range values are detected on the *raw* readings and left out of the average, so a 999 is never mixed into a real value. Hour-mean vs the :00 reading matters only for the hex sensor: RMS difference 3.5 µg/m³ (≈ 8.5 % of the level), no bias, and 4 hours that the :00 rule would lose entirely.

**Problems found** (each figure is `figures/part0_*.png`):

| check | threshold and justification | readings caught | sensors |
|---|---|---|---|
| error code | value = 999; an isolated spike in the histogram | 136 | 5 (120 from `1204260025`) |
| out of range | > 500 µg/m³; above the sensor range, network 99th pct ≈ 108 | 28 | 11 |
| implausible zero | exactly 0; mostly long zero runs of dead sensors | 813 h | 21 (`1203260082`: 414, `1203260068`: 350) |
| stuck | ≥ 24 identical consecutive reported hours; natural repeats max 22 h, faulty runs 74–186 h (clear gap) | 929 h | 7 |
| mostly unusable | > 50 % of hours flagged; next-worst sensor is at 15 % | 600 ok-looking hours | 2 (`1203260082`, `1203260068`) |
| network outage | < 10 % of 76 sensors reporting | 117 hours (5 Aug 08:00 – 10 Aug 04:00 UTC), zero reporters | all |
| barely follows network | r < 0.4 with the leave-one-out network median (median r = 0.71; r² < 0.16) | 5,801 h | 5 |

The flag distribution also exposes 65 "partial" hours in which the entire 10-digit family is offline. They are above the 10 % line, so they are not outages, but network averages in those hours represent only one device type.

`kanpur_hourly_clean.csv` is the full 76 × 1,488-hour grid. It has one `flag` per row (`error_code > out_of_range > zero > stuck > dropped_sensor > low_corr > ok`; rows without data are `outage` or `missing`) plus a boolean column per check. Flagged values are kept as they were and gaps are left as NaN.

**Calibration.** We paired each 10-digit sensor with its nearest 8-char sensor within 3 km (20 pairs) and fitted a pooled linear correction on July: 8-char ≈ 22.9 + 0.40 × 10-digit. Tested on August, **it makes disagreement worse** (RMSE 20.5 → 21.5, MAE 11.9 → 13.8; only 4 of 20 pairs improve), and an offset-only correction barely helps (20.53 → 20.45). Three reasons:
1. There is almost no systematic family offset (mean pair difference −1.4 / −1.8 µg/m³).
2. Pair members correlate weakly hour by hour (r = 0.33 in July), so least squares dilutes the slope to 0.4 and mostly pulls readings toward the July mean.
3. Differences are pair-specific (mean differences from −18 to +11 µg/m³ within the same family), reflecting micro-environments up to 3 km apart rather than device type.

A real calibration needs co-located sensors or a reference monitor.

**Summary.**

| stage | sensors | readings | share missing |
|---|---|---|---|
| raw 15-min file | 76 | 85,762 | 0.810 |
| hourly table (all values) | 76 | 82,131 | 0.274 |
| clean, `ok` only | 69 | 73,824 | 0.347 |

The network's daily cycle in IST peaks at ≈ 06:30 (41 µg/m³) and is lowest at ≈ 16:30 (34 µg/m³): night-time trapping in a shallow boundary layer versus daytime mixing (`part0_daily_cycle_ist.png`).

## Part 1 — Baselines

* **Persistence** = the reading at the issue time.
* **Ridge:** `Ridge(alpha=1.0)` on the 29 raw features (24 lags, 10-nearest-neighbour mean now and over 24 h, network mean, sin/cos of *d* = issue day since 1 Jan), with negatives clipped. It is trained on 242 issue days × 468 = **113,256 rows** (2 Jan–30 Aug issues, all with 24 h of history).

| | overall RMSE | MAE | Sep RMSE | MAE | Oct RMSE | MAE | Nov RMSE | MAE |
|---|---|---|---|---|---|---|---|---|
| Persistence | 32.75 | **17.47** | 23.33 | 8.91 | 20.22 | 11.97 | 47.81 | 31.70 |
| Ridge baseline | **29.58** | 18.16 | 20.34 | 10.35 | 19.36 | 13.46 | 43.06 | 30.81 |

**No baseline wins on both metrics.** Ridge wins RMSE; persistence wins MAE (clearly in Sep and Oct). Minimising squared error corresponds to a Gaussian likelihood and targets the conditional *mean*, while MAE rewards the *median* (Laplace likelihood, Lecture 13). To limit rare large errors, ridge shrinks toward typical levels:
* it keeps only 0.23 of the current reading plus the network mean (0.36);
* this wins in the tail (99th-percentile |error| 116 vs 137), which dominates RMSE;
* but it is closer than persistence on only 43 % of rows (median |error| 11.3 vs 8.8);
* its sin/cos terms, fitted on Jan–Aug, add a +5 µg/m³ mean bias in validation, which costs MAE on calm days (`part1_rmse_vs_mae.png`).

## Part 2 — Improving the forecast

**Features: 8 groups, 63 features,** all computed from rows ≤ issue time; 7 days of history needed, giving **110,448 training rows** (236 issue days × 468):

| group | # | content |
|---|---|---|
| recent24 | 24 | last 24 hourly readings |
| daily_hist | 6 | daily means 2–7 days back |
| same_hour | 7 | reading at the target hour (12:00) on each of the 7 previous days |
| trend | 6 | today − yesterday, 3-h change, 24-h slope/std/min/max |
| neighbourhood | 8 | 10-NN mean and means within 10/25/50 km, now and over 24 h |
| network | 4 | network mean now, 24 h, 7 days, day-on-day change |
| station | 4 | station training mean/std (rows ≤ 31 Aug 12:00), x/y km |
| calendar | 4 | sin/cos of target day-of-year and day-of-week |

**Experiment log** (validation; all models except E3 predict the change *y* − *x*ₙₒw):

| exp | model | train RMSE | val RMSE | val MAE |
|---|---|---|---|---|
| E1 | ridge baseline (29 feats) | 28.00 | 29.58 | 18.16 |
| E3 | scaled ridge, 63 feats, level target | 26.29 | 29.41 | 17.88 |
| E4 | scaled ridge, α = 3000 (best on α path) | 26.53 | 29.30 | 17.59 |
| E5 | lasso α = 0.1 (27 of 63 weights → 0) | 26.39 | 29.27 | 17.58 |
| E6 | Huber loss | 26.75 | 29.66 | 16.64 |
| E7 | log-log ridge (smearing) | 26.87 | 30.13 | 16.69 |
| E8 | KNN k = 200, distance-weighted | – | 30.69 | 17.13 |
| E9 | RBF random features (500) + ridge | 25.26 | 31.26 | 19.08 |
| E10/E11 | gradient boosting, level / change | 17.09 / 17.33 | 34.90 / 31.93 | 19.65 / 18.34 |
| E12 | gradient boosting, shallow, regularised | 23.99 | 30.99 | 17.14 |
| E13 | random forest | 20.68 | 29.94 | 16.90 |
| E14 / E15 | ridge with PCA(5) lags / with K-means cluster means | 26.75 / 26.52 | 29.41 / 29.27 | 17.60 / 17.55 |
| E16 | ridge, issue hours 11–13 (330,876 rows) | 26.82 | 29.33 | 17.68 |
| E18 | extra trees (leaf 50, 30 % features) | 22.72 | 29.05 | 16.23 |
| E17 | **ridge + extra trees average, 8 groups** | 23.93 | 28.56 | 16.42 |
| E19 | **final: same, 7 groups (trend dropped)** | 23.93 | **28.55** | **16.35** |

The ensemble's hyperparameters (ridge α ∈ {1…3000}, extra-trees leaf size ∈ {20, 50, 100}, max_features ∈ {0.3, 0.5}) were chosen jointly on validation. α ≤ 100 is a plateau (differences < 0.01), so we took α = 1. Extra training issue hours did not help, so the final model uses only the 12:00 issues (110,448 rows).

**Ablation** (remove one group from the 8-group candidate, same hyperparameters; positive ΔRMSE = the group helps):

| removed | calendar | network | recent24 | same_hour | station | daily_hist | neighbourhood | trend |
|---|---|---|---|---|---|---|---|---|
| # features | 4 | 4 | 24 | 7 | 4 | 6 | 8 | 6 |
| ΔRMSE | +0.48 | +0.27 | +0.14 | +0.06 | +0.05 | +0.04 | +0.03 | **−0.001** |

*Trend* does not earn its place and was dropped: the **final model has 7 groups, 57 features**. Groups overlap heavily: group summaries correlate 0.8–0.96 (recent24, same_hour and daily_hist all hold recent station levels; neighbourhood and network both measure regional pollution, r = 0.93). That is why each group's unique contribution is small.

**Final comparison:**

| | overall RMSE | MAE | Sep | | Oct | | Nov | |
|---|---|---|---|---|---|---|---|---|
| Persistence | 32.75 | 17.47 | 23.33 | 8.91 | 20.22 | 11.97 | 47.81 | 31.70 |
| Ridge baseline | 29.58 | 18.16 | 20.34 | 10.35 | 19.36 | 13.46 | 43.06 | 30.81 |
| **Our model** | **28.55** | **16.35** | **19.60** | **8.74** | **17.43** | **11.20** | **42.13** | **29.30** |

Our model beats both baselines on both metrics in every month (−3.5 % RMSE vs ridge). Refitting with four forest seeds moves RMSE by only 0.09.

**Questions.**
* **What did each group add?** Calendar (seasonal position; Nov/Dec sit next to Jan on the circle) and the network level helped most, then the recent readings. Trend summaries, the log transform, PCA, K-means features, kernel features and extra issue hours did not help: they re-describe information already present, or fit training noise.
* **Did a flexible model beat ridge?** Not alone. Gradient boosting reached train RMSE 17 but validation 32–35, against ridge's 26.5 / 29.3. It learned Jan–Aug patterns (winter haze, monsoon washout) that do not hold in the post-monsoon build-up, and tree leaves cannot output values beyond the training range. This is overfitting plus a seasonal shift. Predicting the change and averaging extra trees with ridge fixed most of it: the two models err differently.
* **What missing information would help most?** Target-day weather (boundary-layer height, wind, rain, inversions), crop-fire counts upwind, festival dates, and which values were gap-filled. The worst errors are weather-driven statewide jumps, e.g. 29 Nov (network mean 159 → 77 µg/m³).

## Part 3 — Lead time

| lead (issue time) | Persistence | Ridge (refit) | Our model (refit) |
|---|---|---|---|
| **1 h (11:00, d+1) — nowcast** | 23.95 | 18.95 | **18.86** |
| 3 h (09:00) | 51.43 | 28.51 | 27.94 |
| 6 h (06:00) | 66.82 | 34.06 | 31.93 |
| 12 h (00:00) | 72.74 | 31.97 | 31.04 |
| **24 h (12:00, d) — day-ahead** | 32.75 | 29.58 | **28.55** |

(RMSE µg/m³; MAE in `results/part3_lead_time.csv`; plot in `part3_rmse_vs_lead.png`.)

* **Persistence:** its error rises sharply from 1 h to 12 h and then **dips at 24 h**. Bihar PM2.5 has a strong daily cycle (`part3_hour_of_day.png`): validation means peak in the evening (≈ 106 µg/m³ at 20:00), stay high overnight (93 at 00:00, 88 at 06:00) and fall to ≈ 53 at noon, because the shallow night-time boundary layer traps emissions and daytime mixing dilutes them. Persistence from a night or morning hour copies a night level onto a noon target, giving a bias of +40 µg/m³ at 12 h. At 24 h it compares noon with noon (bias −0.5).
* **Learned models** remove most of this (they learn the night-to-noon drop) but show a mild dip too: 6–12 h are worse than 24 h even though their training RMSE is lower. In November they over-forecast by +9 to +15 µg/m³, because the required drop (≈ −70 µg/m³) is larger than training drops and cannot be extrapolated.
* **Public warnings:** we would use **24 h**, so people can plan. The cost is ≈ 9.7 µg/m³ RMSE vs a 1-h nowcast (28.55 vs 18.86), and an issue at 00:00–06:00 would not even be more accurate.

## Part 4 — Maps

IDW uses power 2 on km coordinates, all 468 stations, and one precomputed 3,396 × 468 weight matrix applied to observed, persistence and our forecast alike; zero distance copies the station value. Figure 1 is reproduced in `part4_figure1_check.png`.

Four-panel maps (observed | ours | persistence on one shared scale | ours − observed on a diverging scale):
* **clean**, 15 Sep (network mean 7.9; station RMSE ours 19.3, persistence 27.3);
* **polluted**, 27 Nov (175; 64.4 vs 71.7);
* **our worst day**, 29 Nov (77; 90.5 vs 112.0) — a statewide drop forecast nobody could see coming, red over the whole difference panel (`part4_maps_*.png`).

| (mean over 91 days) | map RMSE | station RMSE (daily mean) | station RMSE (pooled) | pattern correlation |
|---|---|---|---|---|
| Persistence | 15.37 | 28.57 | 32.75 | 0.580 |
| Our model | **14.17** | **24.88** | **28.55** | **0.618** |

* **Which RMSE is better, and which should be quoted?** Map RMSE looks much better (14 vs 25). IDW smooths both maps: the spatial spread drops from 30.4 across stations to 14.0 across grid cells, so station noise and extremes cancel in the difference. The **station RMSE** should be quoted as forecast accuracy, because it compares forecasts with measurements.
* **In what sense is the observed map not ground truth?** Between stations it is a weighted average, not a measurement. It cannot show hotspots, never exceeds the station extremes, makes bull's-eyes around stations, extrapolates near the border, and inherits sensor errors and the unmarked gap-filling.
* **Good pattern correlation but poor RMSE:** a forecast with the right spatial shape but the wrong overall level, e.g. 29 Nov (high everywhere). For public warnings, **RMSE/level matters more**: warnings depend on crossing health thresholds.

## Part 5 — Hidden December test

* **Refit:** the final model (same 7 groups and hyperparameters) on all 12:00 targets up to 30 Nov — 327 issue days, **153,036 rows**; station statistics recomputed on readings ≤ 30 Nov 12:00.
* **Submission files:** `submission/` holds `forecast.py`, `model.pkl` (11.7 MB, only sklearn/numpy objects), `bihar_stations.csv` and `features.py`, a **byte-identical copy** (SHA-256 checked) of the training feature code.
* **Consistency:** `forecast()` reproduces the notebook's predictions to 1.2 × 10⁻¹² µg/m³ and takes ≈ 0.06 s per call.
* **Checker:** `check_submission.py` prints *"RESULT: OK - format and timing look right"*. Its November RMSE (38.6 vs persistence 47.8) is in-sample after the refit, so it is not a score.

## Limitations

* The Bihar data were gap-filled upstream without marking, so some targets and features are interpolations, and 49 readings sit at a 500 µg/m³ ceiling.
* Validation covers only Sep–Nov. December is peak smog season, and errors are likely to be closer to November's (≈ 42) than to the overall 28.6.
* The gains over ridge are modest (3.5 %); most remaining error is weather-driven.
* The station training mean uses all of Jan–Aug, which can make training scores slightly optimistic; this is allowed and does not affect validation.
* The hyperparameter and group choices were made on a single validation period, so some selection optimism remains.
* Part 0 thresholds for out-of-range (500) and low correlation (0.4) are judgement calls supported by the plots, not sharp gaps.

## Use of AI tools

We used **Claude Code (Anthropic)** as a coding assistant for this project:
* writing and debugging the code (data loading, vectorised feature builders, model experiments, IDW interpolation, `forecast.py`);
* running the analysis;
* making the plots;
* drafting this report and our learning notes.

We reviewed the outputs, checked the leakage protections (asserts, the future-poisoning test, train-only fitting) and the numbers against the executed notebooks, and every group member can explain each step.
