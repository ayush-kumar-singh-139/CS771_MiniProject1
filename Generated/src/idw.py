"""Inverse-distance-weighting (IDW) interpolation from stations to the 0.05 degree grid (Part 4).

IDW estimate at grid point g:   z(g) = sum_j w_gj z_j / sum_j w_gj,   w_gj = 1 / d(g, j)^p
Distances are in km (PDF rule). Because the weights depend only on geometry, we build the
(n_grid x n_stations) weight matrix ONCE and then a whole day's map is a matrix-vector product.
The SAME matrix is used for observed values and for every forecast, so map differences come only
from the station values.
"""
import numpy as np

from .data import distance_matrix_km


def idw_weight_matrix(grid_xy_km, station_xy_km, power=2, eps_km=1e-6):
    """Row-normalised IDW weights, shape (n_grid, n_stations).

    Zero distance is handled safely: if a grid point sits on a station (d < eps_km) it simply takes
    that station's value (weight 1 on that station, 0 elsewhere) instead of dividing by zero.
    """
    D = distance_matrix_km(grid_xy_km, station_xy_km)
    on_station = D < eps_km
    Wt = 1.0 / np.maximum(D, eps_km) ** power
    hit = on_station.any(axis=1)
    Wt[hit] = on_station[hit].astype(float)
    return Wt / Wt.sum(axis=1, keepdims=True)


def interpolate(weights, station_values):
    """station_values: (n_stations,) or (n_days, n_stations) -> grid values (n_grid,) or (n_days, n_grid)."""
    return np.asarray(station_values, float) @ weights.T
