"""Loading the Bihar data and simple geometry (km coordinates, nearest neighbours)."""
import numpy as np
import pandas as pd

from . import config as C


def load_bihar():
    """Wide hourly table: DatetimeIndex 'timestamp' (IST), one string column per sensor_id."""
    W = pd.read_csv(C.BIHAR_HOURLY, index_col="timestamp", parse_dates=True)
    W.columns = W.columns.astype(str)
    return W


def load_stations():
    """Station table indexed by sensor_id (string), same order as the hourly columns."""
    return pd.read_csv(C.BIHAR_STATIONS, dtype={"sensor_id": str}).set_index("sensor_id")


def to_km(lon, lat):
    """Convert lon/lat (degrees) to approximate km, as required by the PDF."""
    x = (np.asarray(lon, dtype=float) - C.LON0) * C.KM_PER_DEG_LON
    y = (np.asarray(lat, dtype=float) - C.LAT0) * C.KM_PER_DEG_LAT
    return np.column_stack([x, y])


def distance_matrix_km(xy_a, xy_b=None):
    """Euclidean distances (km) between every row of xy_a and every row of xy_b."""
    xy_b = xy_a if xy_b is None else xy_b
    diff = xy_a[:, None, :] - xy_b[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=2))


def noon_targets(start, end, target_hour=C.ISSUE_HOUR):
    """All target timestamps at target_hour between start and end (inclusive)."""
    days = pd.date_range(pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize(), freq="D")
    t = days + pd.Timedelta(hours=target_hour)
    return t[(t >= start) & (t <= end)]


def split_issue_positions(times, target_times, lead=C.LEAD_HOURS, min_history=24):
    """Row positions of issue times (= target - lead) that have at least `min_history` hours of history.

    Returns (issue_pos, target_times_kept). Rows without enough history are dropped (training only:
    validation targets always have plenty of history).
    """
    times = pd.DatetimeIndex(times)
    issue_times = pd.DatetimeIndex(target_times) - pd.Timedelta(hours=lead)
    pos = times.get_indexer(issue_times)
    keep = pos >= (min_history - 1)
    return pos[keep], pd.DatetimeIndex(target_times)[keep]


def neighbour_index(xy, k=C.N_NEIGHBOURS):
    """Indices of the k nearest OTHER stations for every station, shape (n_stations, k).

    The station itself (distance 0) is excluded. A stable sort breaks ties by column order.
    """
    D = distance_matrix_km(xy)
    np.fill_diagonal(D, np.inf)                      # never pick yourself
    return np.argsort(D, axis=1, kind="stable")[:, :k]
