# How to reproduce everything

All commands are run from the `MP1/` folder unless stated otherwise. The data is read from the
course package next to this folder (`../mp1-package/data/`); nothing is downloaded.

```
Mini Project-1/
├── mp1-package/            (course package: data, starter kit, PDF — unchanged)
└── MP1/                    (our work)
    ├── part0_kanpur_cleaning.ipynb
    ├── mini_project_1.ipynb
    ├── src/                config, data, features (shared with the submission), metrics, model, idw, plotting
    ├── submission/         forecast.py, features.py, model.pkl, bihar_stations.csv
    ├── kanpur_hourly_clean.csv
    ├── figures/            every plot as PNG
    ├── results/            every table as CSV + check_submission output
    ├── report_draft.md, LEARNING_GUIDE.md, README_RUN.md
```

## 1. Environment

Python 3.10+ with the packages in `../mp1-package/requirements.txt`
(tested with Python 3.13, numpy 2.1.3, pandas 2.2.3, scikit-learn 1.9.1, scipy 1.18, matplotlib 3.11).

```bash
pip install -r ../mp1-package/requirements.txt
```

## 2. Part 0 — Kanpur cleaning (≈ 1 min)

```bash
jupyter nbconvert --to notebook --execute --inplace part0_kanpur_cleaning.ipynb
```
Writes `kanpur_hourly_clean.csv` and `figures/part0_*.png`.

## 3. Parts 1–5 — main notebook (≈ 10–15 min on a 16-core laptop)

```bash
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 mini_project_1.ipynb
```
Writes `figures/part1_* … part4_*`, `results/*.csv`, and the submission
(`submission/model.pkl`, `submission/features.py`, `submission/bihar_stations.csv`).
At the end it runs the course checker itself and saves its output to `results/check_submission_output.txt`.

The notebook must be run with `MP1/` as the working directory (nbconvert does this automatically),
because it imports `src/`.

## 4. Check the submission on its own

From the `Mini Project-1/` folder:
```bash
python mp1-package/starter/check_submission.py MP1/submission
```
It must end with `RESULT: OK - format and timing look right.`

## Reproducibility notes
* Every random component uses `random_state = 0` (`src/config.py: RANDOM_STATE`).
* `submission/features.py` is a byte-for-byte copy of `src/features.py`; the notebook asserts the
  SHA-256 hashes match and that `forecast()` reproduces the notebook's own predictions (max difference < 1e-6).
* If you change anything in `src/features.py`, re-run the main notebook so the copy and `model.pkl` are regenerated.
