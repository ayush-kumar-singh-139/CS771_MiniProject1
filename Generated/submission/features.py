"""Feature building for next-day PM2.5 forecasting.

THIS FILE IS SHARED BY TRAINING AND BY submission/forecast.py.
The notebook copies it byte-for-byte into submission/features.py and asserts that the two copies are
identical, so the features the model was trained on are exactly the features it sees at forecast time.
It therefore only imports numpy and pandas (nothing from the rest of src/).

Conventions
-----------
V          : (T, S) float array of hourly PM2.5, rows = hours (IST), columns = stations.
times      : DatetimeIndex of length T (the row labels of V), hourly and gap-free.
issue_pos  : (n,) integer row positions in V of the issue times.
lead       : hours between issue time and target time (24 for the main task).

Every feature for an issue time at row p is computed only from rows <= p (asserted below).
Feature matrices have one row per (issue time, station), ordered issue-major then station:
row = i * S + s.
"""
import numpy as np
import pandas as pd

YEAR_START = pd.Timestamp("2024-01-01")     # d = days since 1 Jan 2024 (0 for 1 Jan)


# =====================================================================================
# small helpers
# =====================================================================================
def _check_cutoff(issue_pos, oldest_lag):
    """Leakage / range guard: the oldest row we read is issue_pos - oldest_lag and must exist."""
    issue_pos = np.asarray(issue_pos)
    assert np.all(issue_pos - oldest_lag >= 0), "not enough history before some issue time"


def _lag_block(V, issue_pos, lags):
    """Readings at issue_pos - lag for every lag. Returns (n, len(lags), S).

    lags are >= 0, so every row read is at or before the issue time (no leakage by construction).
    """
    lags = np.asarray(lags)
    assert np.all(lags >= 0), "a negative lag would look into the future"
    _check_cutoff(issue_pos, lags.max())
    return V[np.asarray(issue_pos)[:, None] - lags[None, :]]


def _window_mean(csum, issue_pos, end_lag, length):
    """Mean of rows [p - end_lag - length + 1, p - end_lag] using a cumulative sum (fast).

    csum is the cumulative sum of V with a zero row on top, shape (T + 1, S).
    """
    assert end_lag >= 0
    end = np.asarray(issue_pos) - end_lag            # last row included (<= issue time)
    start = end - length + 1
    assert np.all(start >= 0), "not enough history before some issue time"
    return (csum[end + 1] - csum[start]) / length


def _cumsum0(V):
    return np.vstack([np.zeros((1, V.shape[1])), np.cumsum(V, axis=0)])


def _to_rows(block):
    """(n, S) -> (n*S,) or (n, S, k) -> (n*S, k), issue-major ordering."""
    if block.ndim == 2:
        return block.reshape(-1)
    n, S, k = block.shape
    return block.reshape(n * S, k)


def day_index(timestamps):
    """d = whole days since 1 Jan 2024 (0 for 1 Jan)."""
    ts = pd.DatetimeIndex(timestamps)
    return np.asarray((ts.normalize() - YEAR_START).days, dtype=float)


# =====================================================================================
# Part 1: the fixed 29-feature ridge baseline
# =====================================================================================
BASELINE_FEATURE_NAMES = (
    [f"lag_{k}h" for k in range(24)]
    + ["nbr10_now", "nbr10_mean24h", "network_now", "sin_doy", "cos_doy"]
)


def baseline_features(V, times, issue_pos, nbr_idx):
    """The 29 Part-1 features for every (issue time, station) pair.

    nbr_idx: (S, 10) indices of each station's 10 nearest OTHER stations (km distance).
    Returns X with shape (n * S, 29).
    """
    V = np.asarray(V, dtype=float)
    issue_pos = np.asarray(issue_pos)
    n, S = len(issue_pos), V.shape[1]

    lag = _lag_block(V, issue_pos, np.arange(24))            # (n, 24, S): t, t-1h, ..., t-23h
    now = lag[:, 0, :]                                       # (n, S) readings at t
    mean24 = lag.mean(axis=1)                                # (n, S) each station's 24 h mean

    nbr_now = now[:, nbr_idx].mean(axis=2)                   # (n, S) mean of the 10 neighbours at t
    nbr_24 = mean24[:, nbr_idx].mean(axis=2)                 # (n, S) mean of the 10 neighbours over 24 h
    network_now = np.repeat(now.mean(axis=1, keepdims=True), S, axis=1)   # all 468 stations at t

    d = day_index(pd.DatetimeIndex(times)[issue_pos])        # d of the ISSUE day
    sin_d = np.repeat((np.sin(2 * np.pi * d / 365))[:, None], S, axis=1)
    cos_d = np.repeat((np.cos(2 * np.pi * d / 365))[:, None], S, axis=1)

    X = np.column_stack([
        _to_rows(np.transpose(lag, (0, 2, 1))),              # (n*S, 24)
        _to_rows(nbr_now), _to_rows(nbr_24), _to_rows(network_now),
        _to_rows(sin_d), _to_rows(cos_d),
    ])
    assert X.shape == (n * S, 29)
    return X


# =====================================================================================
# Part 2: richer features, organised in named groups (for the ablation)
# =====================================================================================
HISTORY_HOURS = 7 * 24        # the rich features look back at most 7 days (168 h)


def make_context(xy_km, nbr_idx, V_train, radii_km=(10.0, 25.0, 50.0)):
    """Everything the rich features need besides the readings themselves.

    xy_km   : (S, 2) station coordinates in km.
    nbr_idx : (S, 10) nearest-other-station indices.
    V_train : readings from the TRAINING period only; used for the station long-run mean / std.
              (This is a fitted statistic, so it must never see validation readings.)
    Returns a plain dict of numpy arrays (pickles cleanly into model.pkl).
    """
    xy_km = np.asarray(xy_km, dtype=float)
    S = len(xy_km)
    D = np.sqrt(((xy_km[:, None, :] - xy_km[None, :, :]) ** 2).sum(axis=2))
    radius_W = {}
    for r in radii_km:
        M = ((D <= r) & ~np.eye(S, dtype=bool)).astype(float)          # other stations within r km
        empty = M.sum(axis=1) == 0
        # a station with nobody inside the radius falls back to its 10 nearest neighbours
        for s in np.where(empty)[0]:
            M[s, nbr_idx[s]] = 1.0
        radius_W[r] = M / M.sum(axis=1, keepdims=True)                  # row-normalised averaging matrix
    V_train = np.asarray(V_train, dtype=float)
    return {
        "xy_km": xy_km,
        "nbr_idx": np.asarray(nbr_idx),
        "radii_km": tuple(float(r) for r in radii_km),
        "radius_W": radius_W,
        "station_mean": V_train.mean(axis=0),
        "station_std": V_train.std(axis=0),
    }


def _group_recent24(lag24):
    return np.transpose(lag24, (0, 2, 1)), [f"lag_{k}h" for k in range(24)]


def _group_daily_hist(csum, issue_pos):
    
    # Lists to collect our calculated averages and their column names
    blocks = []
    names = []
    
    # Loop for the past 6 days (k goes from 1 to 6)
    for k in range(1, 7):
        
        # STEP 1: Calculate where the window ends (end_lag)
        # If k=1, end_lag = 24 hours ago (yesterday)
        # If k=2, end_lag = 48 hours ago (day before yesterday), etc.
        end_lag = 24 * k 
        
        # STEP 2: Calculate the mean of that 24-hour window
        # length is always 24 because we want a full daily average
        daily_average = _window_mean(csum, issue_pos, end_lag, length=24)
        
        # STEP 3: Store the calculated average in our list
        blocks.append(daily_average)
        
        # STEP 4: Create the column name for this feature
        # If k=1, k+1 = 2 (meaning 2 days ago)
        column_name = "daymean_" + str(k + 1) + "d_ago"
        names.append(column_name)
        
    # STEP 5: Stack all the blocks together
    # Currently, 'blocks' contains 6 separate 2D tables (one for each day).
    # np.stack joins them along the 3rd dimension (axis=2) to make a 3D matrix.
    final_3d_matrix = np.stack(blocks, axis=2)
    
    # Return the final 3D matrix and the list of column names
    return final_3d_matrix, names

def _group_same_hour(V, issue_pos, lead):
    
    # List to collect our calculated lags (hours looking backward from issue time)
    lags_list = []
    
    # Loop for the past 7 days (k goes from 1 to 7)
    for k in range(1, 8):
        # STEP 1: Calculate the exact lag for the target hour
        # Target time is (issue_time + lead). 
        # We want data from (Target time - 24*k hours).
        # Relative to issue_time, the formula becomes: (24 * k) - lead
        exact_lag = (24 * k) - lead
        lags_list.append(exact_lag)
        
    # Convert the python list to a numpy array so it works with other functions
    lags_array = np.array(lags_list)
    
    # STEP 2: Fetch the PM2.5 data for these exact lags using our helper function
    # This returns a 3D block of shape: (Issue Times, 7 Lags, Stations)
    data_block = _lag_block(V, issue_pos, lags_array)
    
    # STEP 3: Rearrange the shape (Transpose / Swapaxes)
    # We need Stations in the middle and our 7 Lags at the end.
    # New shape will be: (Issue Times, Stations, 7 Lags)
    arranged_data = np.swapaxes(data_block, 1, 2)
    
    # STEP 4: Create column names for the 7 features
    names = []
    for k in range(1, 8):
        column_name = "targethour_" + str(k) + "d_ago"
        names.append(column_name)
        
    # Return the formatted data and the column names
    return arranged_data, names

def _group_neighbourhood(now, mean24, ctx):
    feats, names = [], []
    nbr = ctx["nbr_idx"]
    feats += [now[:, nbr].mean(axis=2), mean24[:, nbr].mean(axis=2)]
    names += ["nbr10_now", "nbr10_mean24h"]
    for r in ctx["radii_km"]:
        Wr = ctx["radius_W"][r]
        feats += [now @ Wr.T, mean24 @ Wr.T]
        names += [f"r{int(r)}km_now", f"r{int(r)}km_mean24h"]
    return np.stack(feats, axis=2), names

def _group_trend(lag24, csum, issue_pos):
    
    # We will store our 6 calculated features in this list
    all_features = []
    
    # FEATURE 1 & 2: Averages (Today vs Yesterday)
    # Average of the last 24 hours (Today)
    mean_today = np.mean(lag24, axis=1) 
    # Average of the 24 hours before that (Yesterday)
    mean_yday = _window_mean(csum, issue_pos, end_lag=24, length=24)
    
    # Feature 1: How much did the average change compared to yesterday?
    day_over_day_change = mean_today - mean_yday
    all_features.append(day_over_day_change)
    
    # Feature 2: Short-term change (Right now vs 3 hours ago)
    # lag 0 is 'now', lag 3 is '3 hours ago'
    change_3h = lag24[:, 0, :] - lag24[:, 3, :]
    all_features.append(change_3h)
    
    # FEATURE 3: The Linear Trend (Slope)
    # Create a centered time axis (X-axis) from oldest (-11.5) to newest (+11.5)
    t = np.array([11.5, 10.5, 9.5, 8.5, 7.5, 6.5, 5.5, 4.5, 3.5, 2.5, 1.5, 0.5, 
                 -0.5, -1.5, -2.5, -3.5, -4.5, -5.5, -6.5, -7.5, -8.5, -9.5, -10.5, -11.5])
    
    # Calculate the Least Squares Slope (Y = mX + C, we are finding 'm')
    # Formula: Sum of (Y * X) / Sum of (X^2)
    sum_t_squared = np.sum(t ** 2)
    
    # Multiply each hour's pollution by its corresponding time 't', then sum them up
    # (Using a small numpy trick here just to avoid a massive 3-page for-loop)
    sum_y_times_t = np.sum(lag24 * t[None, :, None], axis=1)
    
    slope_24h = sum_y_times_t / sum_t_squared
    all_features.append(slope_24h)
    
    # FEATURES 4, 5, 6: Volatility and Extremes
    std_24h = np.std(lag24, axis=1)  # How much did pollution fluctuate?
    min_24h = np.min(lag24, axis=1)  # Lowest pollution in 24h
    max_24h = np.max(lag24, axis=1)  # Highest pollution in 24h
    
    all_features.append(std_24h)
    all_features.append(min_24h)
    all_features.append(max_24h)
    
    # Stack all 6 features together into a 3D matrix (Issue Times, Stations, 6 Features)
    final_3d_matrix = np.stack(all_features, axis=2)
    
    # Column names for these 6 features
    names = ["today_minus_yday", "change_3h", "slope_24h", "std_24h", "min_24h", "max_24h"]
    
    return final_3d_matrix, names
def _group_network(now, mean24, csum, issue_pos, S):
    net_now = now.mean(axis=1)
    net_24 = mean24.mean(axis=1)
    net_7d = _window_mean(csum, issue_pos, 0, HISTORY_HOURS).mean(axis=1)
    net_yday = _window_mean(csum, issue_pos, 24, 24).mean(axis=1)
    F = np.stack([net_now, net_24, net_7d, net_24 - net_yday], axis=1)    # (n, 4)
    return np.repeat(F[:, None, :], S, axis=1), ["net_now", "net_mean24h", "net_mean7d", "net_today_minus_yday"]


def _group_station(ctx, n):
    F = np.column_stack([ctx["station_mean"], ctx["station_std"], ctx["xy_km"]])   # (S, 4)
    return np.repeat(F[None, :, :], n, axis=0), ["station_train_mean", "station_train_std", "x_km", "y_km"]


def _group_calendar(times, issue_pos, lead, S):
    target_t = pd.DatetimeIndex(times)[issue_pos] + pd.Timedelta(hours=lead)
    d = day_index(target_t)                                   # day of year of the TARGET
    dow = np.asarray(target_t.dayofweek, dtype=float)
    F = np.column_stack([np.sin(2 * np.pi * d / 365), np.cos(2 * np.pi * d / 365),
                         np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7)])
    return np.repeat(F[:, None, :], S, axis=1), ["sin_doy", "cos_doy", "sin_dow", "cos_dow"]


ALL_GROUPS = ("recent24", "daily_hist", "same_hour", "trend", "neighbourhood", "network", "station", "calendar")


def rich_features(V, times, issue_pos, ctx, lead=24, groups=ALL_GROUPS):
    """Part-2 feature matrix. Returns (X of shape (n*S, F), list of feature names, dict group -> column slice).

    Needs HISTORY_HOURS of history before every issue time.
    """
    V = np.asarray(V, dtype=float)
    issue_pos = np.asarray(issue_pos)
    _check_cutoff(issue_pos, HISTORY_HOURS - 1)
    n, S = len(issue_pos), V.shape[1]
    csum = _cumsum0(V[: issue_pos.max() + 1])                 # only rows up to the latest issue time

    lag24 = _lag_block(V, issue_pos, np.arange(24))           # (n, 24, S)
    now, mean24 = lag24[:, 0, :], lag24.mean(axis=1)

    builders = {
        "recent24": lambda: _group_recent24(lag24),
        "daily_hist": lambda: _group_daily_hist(csum, issue_pos),
        "same_hour": lambda: _group_same_hour(V, issue_pos, lead),
        "trend": lambda: _group_trend(lag24, csum, issue_pos),
        "neighbourhood": lambda: _group_neighbourhood(now, mean24, ctx),
        "network": lambda: _group_network(now, mean24, csum, issue_pos, S),
        "station": lambda: _group_station(ctx, n),
        "calendar": lambda: _group_calendar(times, issue_pos, lead, S),
    }
    blocks, names, slices, col = [], [], {}, 0
    for g in groups:
        blk, nm = builders[g]()                               # blk: (n, S, k)
        assert blk.shape == (n, S, len(nm)), g
        blocks.append(_to_rows(blk))
        slices[g] = slice(col, col + len(nm))
        names += nm
        col += len(nm)
    return np.column_stack(blocks), names, slices


# =====================================================================================
# robustness + the prediction rule shared by notebook and forecast.py
# =====================================================================================
def clean_history(V):
    """Make the reading matrix finite WITHOUT looking into the future.

    The Bihar file has no gaps, so on our data this changes nothing (asserted in the notebook).
    It is a safety net for forecast.py: a missing value is replaced by the station's last earlier
    reading (forward fill only), and anything still missing by that hour's network mean (or 0).
    """
    V = np.array(V, dtype=float)
    if np.isfinite(V).all():
        return V
    V[~np.isfinite(V)] = np.nan
    V = pd.DataFrame(V).ffill().to_numpy()
    row_mean = np.nanmean(np.where(np.isfinite(V), V, np.nan), axis=1, keepdims=True)
    row_mean = np.where(np.isfinite(row_mean), row_mean, 0.0)
    return np.where(np.isfinite(V), V, row_mean)


def predict_residual_ensemble(models, X, persistence):
    """Our Part-2 model: each model predicts the CHANGE (target - reading at issue time);
    we average the changes, add back the persistence value, and clip negatives to 0."""
    change = np.mean([m.predict(X) for m in models], axis=0)
    return np.clip(persistence + change, 0, None)


# =====================================================================================
# targets and issue-time bookkeeping
# =====================================================================================
def positions_of(times, stamps):
    """Row positions of the given timestamps in the hourly index (all must exist)."""
    pos = pd.DatetimeIndex(times).get_indexer(pd.DatetimeIndex(stamps))
    assert np.all(pos >= 0), "some timestamps are not in the hourly index"
    return pos


def targets(V, issue_pos, lead=24):
    """Observed PM2.5 at issue time + lead, as a (n*S,) vector in the same row order as the features."""
    V = np.asarray(V, dtype=float)
    return _to_rows(V[np.asarray(issue_pos) + lead])
