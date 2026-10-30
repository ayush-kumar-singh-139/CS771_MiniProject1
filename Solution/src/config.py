"""Project-wide constants: paths, split dates, seeds.

Every notebook imports from here so that a date or a path is written in exactly one place.
"""
from pathlib import Path

import pandas as pd

# ---- paths -------------------------------------------------------------------------------
MP1_DIR = Path(__file__).resolve().parent.parent          # .../Mini Project-1/MP1
PACKAGE_DIR = MP1_DIR.parent / "Task"             # course package (data + starter kit)
DATA_DIR = PACKAGE_DIR / "data"
FIG_DIR = MP1_DIR / "figures"
SUBMISSION_DIR = MP1_DIR / "submission"

BIHAR_HOURLY = DATA_DIR / "bihar_pm25_hourly_jan_nov_2024.csv"
BIHAR_STATIONS = DATA_DIR / "bihar_stations.csv"
BIHAR_GRID = DATA_DIR / "bihar_grid_005deg.csv"
BIHAR_BOUNDARY = DATA_DIR / "bihar_boundary.csv"
KANPUR_RAW = DATA_DIR / "kanpur_pm25_raw_jul_aug_2026.csv"

# ---- reproducibility ---------------------------------------------------------------------
RANDOM_STATE = 0

# ---- the forecasting task ----------------------------------------------------------------
ISSUE_HOUR = 12            # forecasts are issued at 12:00 IST ...
LEAD_HOURS = 24            # ... for 12:00 IST the next day

# Split by TARGET time (PDF section 3).
TRAIN_TARGET_START = pd.Timestamp("2024-01-01 00:00")
# Last training target. 31 Aug 12:00 is also the first validation issue time, so capping training
# targets here means no model ever sees a reading from after the issue time of any validation forecast.
TRAIN_TARGET_END = pd.Timestamp("2024-08-31 12:00")
VAL_TARGET_START = pd.Timestamp("2024-09-01 12:00")
VAL_TARGET_END = pd.Timestamp("2024-11-30 12:00")
N_VAL_DAYS = 91
N_STATIONS = 468

# ---- geometry ----------------------------------------------------------------------------
# Approximate km coordinates (PDF section 2): x = (lon - 85.5) * 100.3, y = (lat - 25.7) * 110.6
LON0, LAT0 = 85.5, 25.7
KM_PER_DEG_LON, KM_PER_DEG_LAT = 100.3, 110.6
N_NEIGHBOURS = 10          # Part 1: mean of the 10 nearest other stations

# Part 4: inverse-distance weighting power
IDW_POWER = 2
