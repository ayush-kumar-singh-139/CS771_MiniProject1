# Learning guide — CS771 Mini-Project 1

This guide is for **us**: how to understand, explain and rebuild every part of the project.
Read it alongside the two notebooks. Numbers quoted here come from the executed notebooks.

---

## 1. The project in plain English

Bihar has 468 cheap PM2.5 sensors that report every hour. PM2.5 is the mass of tiny particles (≤ 2.5 µm) per cubic metre of air, in µg/m³; above ~60 µg/m³ it is unhealthy.

Every day at **noon** we want to tell the public what PM2.5 will be **at noon tomorrow** at every sensor, and draw a **map** of the whole state. We are only allowed to use readings we already have at noon today: no weather forecasts, no satellites, and no peeking at the future.

The project has six pieces:

| Part | In one sentence |
|---|---|
| 0 | Clean a messy raw file from Kanpur: find broken sensors, error codes, stuck values and outages, flag them (don't delete), and check whether two sensor brands agree. |
| 1 | Build two reference forecasts: "tomorrow = today" (*persistence*) and a fixed 29-feature ridge regression. |
| 2 | Try to beat both with better features and models, choosing everything on the Sep–Nov validation months, and show which features helped (ablation). |
| 3 | See how accuracy changes when the forecast is made 1, 3, 6, 12 or 24 hours before the target. |
| 4 | Turn station values into state-wide maps by inverse-distance weighting and compare forecast maps with observed maps. |
| 5 | Retrain on all data and package the model as `forecast.py` so the course team can test it on the hidden December data. |

**Our main results** (validation = 91 days × 468 stations, Sep–Nov 2024):

| | RMSE (µg/m³) | MAE (µg/m³) |
|---|---|---|
| Persistence | 32.75 | 17.47 |
| Ridge baseline (Part 1) | 29.58 | 18.16 |
| **Our model** (ridge + extra trees on 57 features, 7 groups) | **28.55** | **16.35** |

(The exact numbers are in `results/part2_final_table.csv`; if they differ by a hundredth after a re-run, trust the CSV.)

---

## 2. Every concept we used, from zero

### 2.1 Features and feature vectors (Lecture 2)
A *feature vector* is a list of numbers describing one example. Here one example = **one station on one issue day**. Its features are things like "PM2.5 at this station right now", "PM2.5 one hour ago", "average of the 10 nearest stations". The *target* (label) is PM2.5 at that station 24 h later.

### 2.2 Distances and why we convert to km (Lecture 2)
Euclidean distance $\sqrt{(x_1-x_2)^2 + (y_1-y_2)^2}$ only makes sense if both axes are in the same unit. One degree of latitude ≈ 110.6 km, but one degree of longitude at 25.7°N ≈ 100.3 km. So we first convert: $x = (\text{lon}-85.5)\times100.3$, $y = (\text{lat}-25.7)\times110.6$. This is the "feature scaling changes neighbourhoods" lesson from Lecture 2 Part C.

### 2.3 Cyclic encoding (Lecture 2, C2)
Day-of-year 364 and day 0 are neighbours, but as plain numbers they are far apart. Mapping a day $d$ to $(\sin 2\pi d/365, \cos 2\pi d/365)$ places days on a circle, so 31 Dec sits next to 1 Jan. We do the same for day-of-week with period 7.

### 2.4 Prototypes / means (Lecture 2)
The mean of a group is its *prototype*. Our "station training mean" feature is a prototype of each station's typical level, and the "network mean" is a prototype of the whole state at a moment.

### 2.5 Validation and why the split must be by time (Lecture 3)
Training error is optimistic because the model has seen those examples. A **validation set** estimates performance on *new* data. For forecasting, "new" means **later in time**: we train on targets in Jan–Aug and validate on Sep–Nov. A random split would put, say, 14 Oct in training and 15 Oct in validation. Neighbouring days are very similar, so the model would effectively be tested on days it had almost seen, and the score would be too good. Lecture 3's "no peeking at the test set" rule is why we never use December for any choice.

### 2.6 K-nearest-neighbours regression (Lecture 3)
To predict for a new row, find the $K$ most similar training rows (smallest distance in feature space) and average their targets (optionally weighting closer ones more). Features must be **scaled** first, or the feature with the biggest numbers dominates the distance. KNN did poorly here (validation RMSE ≈ 31). With 63 features most rows are roughly equally far apart (the curse of dimensionality), and the closest rows come from other seasons.

### 2.7 Linear models (Lecture 5)
$\hat y = \mathbf w^\top\mathbf x + b$. Each weight says how much the prediction moves per unit of a feature. "Linear" means linear in the **weights**: we can feed in nonlinear features (lags, max, slope, sin/cos) and still use a linear model.

### 2.8 Squared loss, ridge and lasso (Lecture 7)
* **Least squares:** choose $\mathbf w$ to minimise $\sum_n (y_n - \mathbf w^\top\mathbf x_n)^2$.
* **Ridge:** add $\alpha\lVert\mathbf w\rVert_2^2$, which pulls all weights toward 0. This makes the fit stable when features are correlated (our 24 lags are highly correlated). Bigger $\alpha$ means smaller weights, more bias and less variance.
* **Lasso:** add $\alpha\lVert\mathbf w\rVert_1$ instead, which sets some weights *exactly* to 0 (feature selection).
* **Regularisation path:** the plot of validation error against $\alpha$; we choose $\alpha$ where validation error is lowest (`figures/part2_ridge_alpha_path.png`).
* **Why scaling matters for ridge:** the penalty treats every weight the same. If one feature is in thousands and another in fractions, the penalty hits them very differently. A `StandardScaler` (subtract mean, divide by std) **fitted on training rows only** puts all features on the same footing. The Part 1 baseline is deliberately left unscaled because the PDF says so.

### 2.9 Huber loss (Lecture 7)
Quadratic for small residuals and linear for large ones, so a few extreme smog days pull the fit less. It improved MAE slightly but not RMSE.

### 2.10 Gaussian vs Laplace likelihood — why RMSE and MAE disagree (Lecture 13)
If we assume $y = \mathbf w^\top\mathbf x + \text{noise}$:
* Gaussian noise → maximising likelihood = minimising **squared** error, and the best prediction is the **conditional mean**.
* Laplace noise → minimising **absolute** error, and the best prediction is the **conditional median**.

Day-to-day PM2.5 changes are skewed, with occasional big jumps. A squared-loss model (ridge) moves its forecast toward the mean to protect against those jumps. That lowers RMSE but adds a small error on the many ordinary days, which raises MAE. This is exactly why the ridge baseline beats persistence on RMSE but loses on MAE.

### 2.11 Decision trees and ensembles (Lecture 4)
A regression tree splits the feature space with yes/no questions and predicts the mean target in each leaf.
* **Deep trees overfit** (high variance).
* **Bagging / random forest:** many trees on bootstrap samples with random feature subsets, averaged. Averaging reduces variance.
* **Extra trees:** like a forest, but split thresholds are also random. That gives even more averaging and smoother predictions; it overfit least of our tree models.
* **Gradient boosting:** trees added one after another, each fitting the errors of the previous ones. Powerful, but it overfit badly here (train RMSE ≈ 17 vs validation ≈ 32–35).
* **Trees cannot extrapolate:** a leaf predicts an average of *training* targets. November levels (~100–175) are higher than most training days, so we let trees predict the **change** from the current reading instead of the level.

### 2.12 Averaging different models (Lecture 4 idea)
Our final model averages a ridge and an extra-trees forest. They make different kinds of errors (global linear vs local nonlinear), and averaging partly cancels them. It is the same reason a forest beats one tree.

### 2.13 Kernels and random features (Lecture 10)
A kernel (e.g. RBF) is a similarity function that corresponds to an implicit, very high-dimensional feature map. Exact kernel methods need an $N\times N$ matrix, and $N = 110{,}448$ is too large. **Random Fourier features** (`RBFSampler`) build an explicit 500-dimensional map whose dot products approximate the RBF kernel, followed by ridge. It fitted training better but validated worse than plain ridge (31.3 vs 29.3): the extra flexibility only fitted noise.

### 2.14 PCA (Lecture 2/3) and K-means (Lecture 14)
* **PCA** finds the few directions that capture most of the variation. The 24 hourly lags are highly redundant: 5 components keep about 95 % of their variance. Replacing them with 5 PCs changed little.
* **K-means** groups stations by their typical daily cycle and location. We tried "mean of my cluster" features; they did not beat the fixed-radius neighbourhoods.

### 2.15 Inverse-distance weighting (Part 4; Lecture 3 distance-weighted KNN)
The value at a grid point is a weighted average of **all** stations with weights $1/d^2$, so near stations matter most. It is distance-weighted KNN with $K$ = all stations and $1/d^2$ weights. Its properties:
* it reproduces station values exactly at the stations;
* it never goes above the largest or below the smallest station value;
* it smooths everything in between.

### 2.16 Metrics
* **RMSE** $=\sqrt{\text{mean}(e^2)}$ punishes big errors heavily (squares them).
* **MAE** $=\text{mean}|e|$ treats all errors linearly.
* **Pattern correlation** = Pearson correlation between two maps across grid cells. It asks "are the high and low areas in the right places?" and ignores the overall level.
* **Bias** = mean(forecast − observed): a systematic over- or under-forecast.

### 2.17 Data cleaning concepts (Part 0)
* **Sentinel / error code:** a fake value (999) the device writes when it fails.
* **Stuck sensor:** the same value repeated for a long time (a frozen sensor).
* **Flag vs delete:** flagging keeps the information and lets the next analyst decide; deleting hides problems.
* **Calibration:** two device types may read differently for the same air. A correction $y_8 \approx a + b\,x_{10}$ fitted on July and tested on August checks whether a fixed linear offset explains the disagreement.

---

## 3. Pipeline walkthrough: file by file, function by function

```
mp1-package/data/*.csv
        │
        ▼
src/config.py      constants: paths, split dates (train targets ≤ 31 Aug 12:00), RANDOM_STATE=0, km constants
src/data.py        load_bihar()            wide table, string columns
                   load_stations()         sensor_id → lat/lon
                   to_km(lon, lat)         PDF km formula
                   distance_matrix_km()    all pairwise distances
                   neighbour_index(xy,10)  10 nearest OTHER stations (self excluded)
                   noon_targets(start,end) all 12:00 targets in a window
                   split_issue_positions() issue rows = target − lead, keeps rows with enough history
        │
        ▼
src/features.py  (SHARED with submission/ — byte-identical copy)
                   _lag_block(V, pos, lags)    V[pos − lag]; asserts lag ≥ 0 → no future
                   _window_mean(csum, …)       fast window means from a cumulative sum
                   baseline_features()         the 29 Part-1 features
                   make_context(xy, nbr, V_train)  fitted stats (station mean/std from TRAIN rows) + radius matrices
                   rich_features(V, times, pos, ctx, lead, groups)  63 features in 8 named groups (final model uses 7 groups / 57)
                   clean_history(V)            forward-fill safety net (no-op on our data)
                   predict_residual_ensemble() persistence + mean of the models' predicted changes, clipped at 0
                   targets(V, pos, lead)       observed value at pos + lead
        │
        ▼
src/model.py       make_final_models() / fit_final_models() / predict_final()
                   ridge = StandardScaler → Ridge(alpha)   (scaler fitted inside .fit = train rows only)
                   forest = ExtraTreesRegressor(100 trees, min_samples_leaf 50, max_features 0.3, random_state 0)
src/metrics.py     rmse, mae, results_table (overall + Sep/Oct/Nov)
src/idw.py         idw_weight_matrix(grid_xy, station_xy, p=2)  (zero distance → copy station value)
src/plotting.py    draw_map(), four_panel_figure()   (Figure-1 style maps)
        │
        ▼
mini_project_1.ipynb   §0 load + checks + split + future-poisoning test
                       §1 persistence (two ways) + ridge baseline
                       §2 experiments → final model → ablation → final table
                       §3 lead times · §4 maps · §5 refit + model.pkl + forecast.py check + check_submission
        │
        ▼
submission/forecast.py   imports features.py (same code), loads model.pkl once, builds features from `history`
                         for the single issue time, predicts, returns a Series of 468 values
```

**Row ordering convention.** Every feature matrix has row `i*468 + s` = issue day `i`, station `s`. Targets and persistence use the same order, so `y.reshape(91, 468)` gives a day × station table.

**Why the features are fast.** Instead of looping over stations and days, `V[pos[:,None] - lags[None,:]]` pulls an `(n_days, n_lags, 468)` block in one numpy operation (fancy indexing). Window means use a cumulative sum: mean of rows a..b = `(csum[b+1] − csum[a]) / (b − a + 1)`.

---

## 4. Every place leakage could sneak in — and what we did

| Where | The risk | Our protection |
|---|---|---|
| Feature windows | an off-by-one (`pos + 1`) reads the hour after the issue time | `_lag_block` asserts every lag ≥ 0; the **future-poisoning test** sets all later readings to 10⁶ and checks the features don't change |
| Train/val split | random split, or training targets after validation issue times | split by target date; training targets capped at **31 Aug 12:00** = first validation issue time; asserted |
| Scaling | `StandardScaler().fit(all rows)` uses validation means/stds | scalers live inside sklearn pipelines and are only ever `.fit` on training rows |
| Station long-run mean | computing it over Jan–Nov would put Sep–Nov levels into the features | `make_context` receives only `V[: row of 31 Aug 12:00]`; asserted against a pandas recomputation |
| PCA / K-means | fitting them on all rows | fitted on training rows / training-period profiles only |
| Hyperparameter tuning | tuning on the test set | tuned on validation only; December is never seen |
| Extra training issue hours | a target at 31 Aug 23:00 is *after* the first validation issue time | all training targets capped at 31 Aug 12:00 |
| Part 5 refit | — | a refit on more data is allowed by the PDF; hyperparameters stay as chosen on validation |
| forecast.py | using rows after `issue_time`, or refitting | `history.loc[:issue_time]`; the model is loaded once and only `.predict` is called |
| Train/serve skew | the submission computing features differently from training | one shared `features.py` (hash-checked) and a test showing forecast.py = notebook predictions to < 1e-6 |
| Mild caveat (allowed by the PDF) | the station training mean uses all of Jan–Aug, so for a February training row it contains "future" summer readings | it is a fitted statistic from training rows, which the PDF explicitly permits; it can make *training* scores slightly optimistic, but validation is clean |

---

## 5. Suggested order to rebuild it yourselves

1. **Load and look** (30 min): load W and stations, plot a few stations, plot the network mean over the year, check for NaNs.
2. **Split** (30 min): write `noon_targets` and the issue positions, then print the first/last issue times and the counts (242 train days, 91 validation days).
3. **Persistence** (15 min): two ways, assert equal, compute RMSE/MAE → 32.75 / 17.47.
4. **Baseline features** (1–2 h): write a slow loop version for one day first, then the vectorised `_lag_block` version, and assert they match.
5. **Ridge baseline** (15 min): `Ridge(alpha=1)` → 29.58 / 18.16. Per-month table.
6. **Rich features** one group at a time; after each group, fit scaled ridge and log the validation RMSE.
7. **Models**: ridge α path → lasso → KNN → trees. Always record train *and* validation RMSE.
8. **Ensemble + ablation.**
9. **Lead time**: parametrise everything by `lead`.
10. **IDW**: write the weight matrix, check rows sum to 1 and that a grid point on a station copies it; reproduce Figure 1 first.
11. **Submission**: write `forecast.py`, compare with the notebook, run `check_submission.py`.
12. **Part 0** can be done at any time; it is independent.

---

## 6. Likely viva questions (with short answers)

**Protocol and leakage**
1. *Why split by time and not randomly?* Forecasts are used on future days. Consecutive days are highly correlated, so a random split tests on days almost identical to training days and gives optimistic scores (Lecture 3).
2. *Why fit the scaler only on training rows?* The scaler's mean/std are statistics of the data. Fitting on validation rows passes information about the validation period into the features (leakage) and makes the validation score dishonest.
3. *What exactly is the information cutoff?* Features for a forecast issued at day d, 12:00 may use readings with timestamp ≤ day d 12:00 from any station, and nothing later — not even for scaling or fitting.
4. *How do you know your features don't use the future?* (a) All lags are ≥ 0 by construction (asserted). (b) The future-poisoning test: we set every reading after the issue time to 10⁶, and the features are unchanged.
5. *Why cap training targets at 31 Aug 12:00 and not 23:00?* The first validation forecast is issued at 31 Aug 12:00; a model trained on a 23:00 target would have seen data from after that issue time.
6. *How many training rows did you use, and why?* Part 1: 242 issue days × 468 = 113,256 (every 12:00 target with 24 h of history). Part 2: 236 × 468 = 110,448, because the rich features need 7 days of history. Adding issue hours 11–13 tripled the rows but didn't improve validation, so we kept noon only. Part 5 refit: 327 × 468 = 153,036.
7. *Why does validation include every station and day?* The PDF requires it. Dropping hard days would flatter the model.

**Part 1**
8. *Which d did you use in sin/cos?* The issue day (d = 0 on 1 Jan 2024), as the PDF defines t as "day d, 12:00".
9. *Why does ridge beat persistence on RMSE but lose on MAE?* Squared loss gives the conditional mean (Gaussian likelihood); MAE rewards the median (Laplace). Ridge shrinks toward the mean to guard against big jumps: it wins on the rare large errors (RMSE) but adds small errors on many ordinary days (MAE). In September, persistence MAE 8.9 vs ridge 10.4.
10. *What happens if you change alpha in the baseline?* With raw features and 113k rows, α = 1 is almost no regularisation. Larger α shrinks the weights; because features are unscaled, the shrinkage would hit small-valued features (sin/cos) relatively harder. The PDF fixes α = 1 so all groups' numbers are comparable.
11. *Why clip negatives to 0?* PM2.5 cannot be negative; a linear model can output negatives for very clean days.

**Part 2**
12. *Why predict the change instead of the level?* Trees can only output averages of training targets, so they cannot go beyond the training range. The change (target − now) has a similar distribution in every season, so the forest works in November too.
13. *Did a flexible model beat ridge?* Not alone. Gradient boosting and random forest had much lower train RMSE (≈ 17–21) but higher validation RMSE (≈ 30–35): they memorised Jan–Aug patterns that don't hold in Sep–Nov (overfitting plus a seasonal shift). Extra trees overfit least, and *averaging* it with ridge beat both.
14. *What is your final model?* The average of (i) StandardScaler + Ridge and (ii) ExtraTrees (100 trees, min_samples_leaf 50, max_features 0.3), each predicting the 24 h change on 57 features in 7 groups (trend group dropped after the ablation); persistence is added back and negatives clipped.
15. *What did the ablation show?* Calendar (+0.48) and network level (+0.27) mattered most. The trend group did not earn its place (removing it changed RMSE by −0.001 and improved MAE), so we dropped it; every other group made RMSE slightly worse when removed. Groups overlap (recent24 and same_hour both contain today's noon reading; neighbourhood and network both describe regional pollution), so each one's unique contribution is small.
16. *Why does the calendar group help the forest but not ridge?* For ridge, sin/cos of day-of-year adds a smooth seasonal offset fitted on Jan–Aug that doesn't fit Sep–Nov. The forest can use it in combination with levels; and because the encoding is cyclic, Nov/Dec sit close to Jan.
17. *What information would help most?* Weather forecasts (wind, boundary-layer height, rain, temperature inversions), crop-burning fire counts, emissions/traffic data. The worst errors are sudden regional jumps and drops (e.g. 29 Nov) that past PM2.5 alone cannot foresee.
18. *Why are improvements so small?* Most of the 24 h error is weather-driven and unpredictable from PM2.5 history. The remainder is station-level noise (low-cost sensors, upstream gap filling).

**Part 3**
19. *Why does persistence get worse from 1 h to 12 h and then better at 24 h?* PM2.5 has a strong daily cycle (rising after sunset to an evening peak around 20:00, high through the night and morning, lowest in the early afternoon). Persistence from 00:00 or 06:00 copies a night/morning level onto a noon target, a systematic bias. At 24 h it compares noon with noon, so the daily cycle cancels.
20. *Do learned models show the same dip?* Much less. A refitted model learns the average night-to-noon drop, so the huge persistence bias disappears (31–34 instead of 67–73 at 6–12 h). A mild dip remains: 6 h and 12 h (31.9, 31.0) are worse than 24 h (28.55) although their training RMSE is lower. In November they over-forecast by +9 to +15 µg/m³, because the drop they must predict (≈ −70 µg/m³) is larger than drops seen in training, and trees cannot extrapolate.
21. *Which lead would you pick for public warnings?* 24 h (issued at noon the day before), so people can plan. It costs ≈ 9.7 µg/m³ RMSE compared with a 1 h nowcast (28.55 vs 18.86), but a 1 h warning is too late to act on.

**Part 4**
22. *Why is the map RMSE lower than the station RMSE?* IDW averages many stations, which smooths out station-level noise and extremes. Both maps are smoothed the same way, so their differences are smaller than the station differences. It is not a better forecast.
23. *Which should you quote as forecast accuracy?* Station RMSE: it compares forecasts with actual measurements. Map RMSE compares two interpolations.
24. *In what sense is the observed map not ground truth?* It is an interpolation of (gap-filled, low-cost) sensor readings: between stations it is a weighted average, not a measurement, so it can't show local hotspots and it inherits sensor errors.
25. *Good pattern correlation but poor RMSE — example?* A forecast with the right spatial shape but shifted up or down by 50 µg/m³ everywhere (e.g. it misses a statewide rise). The correlation is ~1 but the RMSE is huge. For a public warning the *level* matters most (whether it crosses a threshold), so RMSE/bias matter more.
26. *How do you handle a grid point exactly on a station?* Distance 0 would give division by zero; that grid point takes the station's value (weight 1).
27. *Why use the same interpolation for all maps?* So differences between maps come only from the forecasts, not from the method.

**Part 0**
28. *How did you choose the stuck threshold?* The PDF fixes 24 h. The run-length plot shows natural repeats of a working sensor stop at about 22 h while faulty runs last 74–414 h, so 24 h sits in the gap.
29. *Why average sub-hourly readings and not keep :00?* The average uses all information and is less noisy; the :00 reading is a single sample. For the only sub-hourly sensor the two differ by about 3.5 µg/m³ RMS, and the :00 rule also drops hours with no :00 reading.
30. *Why flag instead of deleting?* So nothing is lost and decisions are reversible and auditable; the next user can choose which flags to accept.
31. *Did calibration help?* No. The pooled July fit (8-char ≈ 22.9 + 0.40 × 10-digit) made August disagreement worse (RMSE 20.5 → 21.5), and an offset-only correction barely helped (20.53 → 20.45). The families have almost no systematic offset (≈ 1.5 µg/m³). Pair members correlate weakly (r = 0.33 in July), so least squares dilutes the slope and mostly pulls values toward the mean, and the differences are pair-specific (micro-environments up to 3 km apart).

**Submission**
32. *How do you guarantee forecast.py computes the same features as training?* It imports a byte-identical copy of `src/features.py` (SHA-256 checked), and we tested that forecast.py reproduces the notebook's predictions to < 1e-6 µg/m³.
33. *What if December has a missing hour?* `clean_history` forward-fills from earlier readings only (never the future), so features stay finite; any non-finite prediction falls back to persistence.
34. *Why is it fast?* Features are built for a single issue time from the last 216 hours only; the forest predicts 468 rows.
