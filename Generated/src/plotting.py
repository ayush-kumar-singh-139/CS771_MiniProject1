"""Plot helpers: PM2.5 maps in the style of the PDF's Figure 1, and a fixed colour per method."""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

# One fixed colour per method, used in every line plot so a method never changes colour.
METHOD_COLORS = {
    "Persistence": "#2a78d6",          # blue
    "Ridge baseline": "#eb6834",       # orange
    "Our model": "#1baf7a",            # aqua
    "Extra": "#4a3aa7",                # violet
}
PM_UNITS = "PM2.5 (µg/m³)"


def grid_to_image(grid_df, values, step=0.05):
    """Put grid-point values on a regular lon/lat raster (NaN outside Bihar) for pcolormesh."""
    lon, lat = grid_df["longitude"].to_numpy(), grid_df["latitude"].to_numpy()
    ix = np.round((lon - lon.min()) / step).astype(int)
    iy = np.round((lat - lat.min()) / step).astype(int)
    img = np.full((iy.max() + 1, ix.max() + 1), np.nan)
    img[iy, ix] = values
    # cell edges so every square is centred on its grid point
    xe = lon.min() - step / 2 + step * np.arange(ix.max() + 2)
    ye = lat.min() - step / 2 + step * np.arange(iy.max() + 2)
    return xe, ye, img


def draw_map(ax, grid_df, values, boundary, stations, cmap="Reds", norm=None, vmin=None, vmax=None,
             title="", show_station_label=True):
    """One map panel: interpolated grid, state outline, stations. Returns the mappable for a colourbar."""
    xe, ye, img = grid_to_image(grid_df, values)
    m = ax.pcolormesh(xe, ye, img, cmap=cmap, norm=norm, vmin=None if norm else vmin,
                      vmax=None if norm else vmax, shading="flat")
    ax.plot(boundary["longitude"], boundary["latitude"], color="black", lw=0.8)
    ax.scatter(stations["longitude"], stations["latitude"], s=2, c="black",
               label="monitoring station" if show_station_label else None)
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_title(title, fontsize=10)
    ax.set_aspect(1 / np.cos(np.deg2rad(25.7)))      # so 1 km looks the same in x and y
    ax.grid(False)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    return m


def four_panel_figure(grid_df, boundary, stations, obs_map, our_map, pers_map, when_label, day_info=""):
    """Observed | our forecast | persistence (one shared Reds scale) | our forecast - observed (diverging)."""
    fig, axes = plt.subplots(1, 4, figsize=(22, 5.2), constrained_layout=True)
    # one shared, robust colour range (1st-99th percentile of the three maps) so a single extreme
    # station does not wash out the whole map; values outside it show as the end colours (arrows on the bar)
    allv = np.concatenate([obs_map, our_map, pers_map])
    vmin, vmax = (float(v) for v in np.percentile(allv, [1, 99]))
    panels = [(obs_map, "Observed"), (our_map, "Our forecast"), (pers_map, "Persistence forecast")]
    for k, (vals, name) in enumerate(panels):
        m = draw_map(axes[k], grid_df, vals, boundary, stations, vmin=vmin, vmax=vmax,
                     title=f"{name}, {when_label}", show_station_label=(k == 0))
    fig.colorbar(m, ax=axes[:3], shrink=0.85, label=PM_UNITS, extend="both")
    diff = our_map - obs_map
    lim = float(np.percentile(np.abs(diff), 99))
    md = draw_map(axes[3], grid_df, diff, boundary, stations, cmap="RdBu_r",
                  norm=TwoSlopeNorm(vcenter=0, vmin=-lim, vmax=lim),
                  title=f"Our forecast − observed, {when_label}", show_station_label=False)
    fig.colorbar(md, ax=axes[3], shrink=0.85, label="Forecast − observed (µg/m³)", extend="both")
    axes[0].legend(loc="lower right", fontsize=8, frameon=False)
    if day_info:
        fig.suptitle(day_info, fontsize=12)
    return fig
