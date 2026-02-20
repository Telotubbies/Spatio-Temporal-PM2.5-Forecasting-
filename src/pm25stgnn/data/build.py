from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree

from pm25stgnn.config import ProjectConfig
from pm25stgnn.utils import ensure_dir, latlon_to_xy_km, nanmean_std


@dataclass(frozen=True)
class Grid:
    lat: np.ndarray
    lon: np.ndarray
    lat2d: np.ndarray
    lon2d: np.ndarray


def make_era5_grid(region: Dict[str, float], res_deg: float) -> Grid:
    lat_min = region["lat_min"]
    lat_max = region["lat_max"]
    lon_min = region["lon_min"]
    lon_max = region["lon_max"]

    lats = np.arange(lat_min, lat_max + 1e-6, res_deg)
    lons = np.arange(lon_min, lon_max + 1e-6, res_deg)
    lat2d, lon2d = np.meshgrid(lats, lons, indexing="ij")

    return Grid(lat=lats, lon=lons, lat2d=lat2d, lon2d=lon2d)


def grid_centers_flat(grid: Grid) -> Tuple[np.ndarray, np.ndarray]:
    return grid.lat2d.reshape(-1), grid.lon2d.reshape(-1)


def load_era5_features(single_levels_nc: str | Path, pressure_levels_nc: str | Path) -> xr.Dataset:
    sfc_path = Path(single_levels_nc)
    pl_path = Path(pressure_levels_nc)

    if sfc_path.is_dir():
        sfc_files = sorted([str(p) for p in sfc_path.glob("*.nc")])
        ds_sfc = xr.open_mfdataset(sfc_files, combine="by_coords")
    else:
        ds_sfc = xr.open_dataset(sfc_path)

    if pl_path.is_dir():
        pl_files = sorted([str(p) for p in pl_path.glob("*.nc")])
        ds_pl = xr.open_mfdataset(pl_files, combine="by_coords")
    else:
        ds_pl = xr.open_dataset(pl_path)

    rename_sfc = {
        "u10": "u10",
        "v10": "v10",
        "10u": "u10",
        "10v": "v10",
        "blh": "blh",
        "boundary_layer_height": "blh",
        "r": "r",
        "relative_humidity": "r",
        "t": "t",
        "t2m": "t",
        "2t": "t",
        "2m_temperature": "t",
        "d2m": "td",
        "2d": "td",
        "2m_dewpoint_temperature": "td",
    }
    rename_pl = {
        "u": "u",
        "v": "v",
        "u_component_of_wind": "u",
        "v_component_of_wind": "v",
    }

    ds_sfc = ds_sfc.rename({k: v for k, v in rename_sfc.items() if k in ds_sfc})
    ds_pl = ds_pl.rename({k: v for k, v in rename_pl.items() if k in ds_pl})

    ds = xr.merge([ds_sfc, ds_pl], compat="override")

    if "r" not in ds and ("t" in ds) and ("td" in ds):
        t_c = ds["t"] - 273.15
        td_c = ds["td"] - 273.15

        a = 17.625
        b = 243.04
        es_td = np.exp(a * td_c / (b + td_c))
        es_t = np.exp(a * t_c / (b + t_c))
        rh = 100.0 * (es_td / (es_t + 1e-12))
        ds["r"] = rh.clip(0.0, 100.0)

    if "time" in ds.coords:
        t = pd.to_datetime(ds["time"].values, utc=True).tz_convert(None)
        ds["time"] = t.values

    return ds


def qc_pm25(df: pd.DataFrame, min_value: float, max_value: float) -> pd.DataFrame:
    df = df.copy()
    df = df.dropna(subset=["timestamp", "value", "lat", "lon"])
    df = df[(df["value"] >= float(min_value)) & (df["value"] <= float(max_value))]
    return df


def map_points_to_grid(lat: np.ndarray, lon: np.ndarray, grid_lat: np.ndarray, grid_lon: np.ndarray) -> np.ndarray:
    lat0 = float(np.mean(grid_lat))
    lon0 = float(np.mean(grid_lon))

    gx, gy = latlon_to_xy_km(grid_lat, grid_lon, lat0, lon0)
    px, py = latlon_to_xy_km(lat, lon, lat0, lon0)

    tree = cKDTree(np.c_[gx, gy])
    _, idx = tree.query(np.c_[px, py], k=1)
    return idx.astype(np.int64)


def aggregate_pm25_to_grid_hourly(
    df_pm25: pd.DataFrame,
    times: pd.DatetimeIndex,
    grid_lat_flat: np.ndarray,
    grid_lon_flat: np.ndarray,
) -> np.ndarray:
    df = df_pm25.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["hour"] = df["timestamp"].dt.floor("h")

    grid_idx = map_points_to_grid(df["lat"].to_numpy(), df["lon"].to_numpy(), grid_lat_flat, grid_lon_flat)
    df["grid_idx"] = grid_idx

    n_time = len(times)
    n_nodes = grid_lat_flat.shape[0]
    out = np.full((n_time, n_nodes), np.nan, dtype=np.float32)

    grouped = df.groupby(["hour", "grid_idx"], as_index=False)["value"].mean()

    time_to_i = {t: i for i, t in enumerate(times)}
    for _, r in grouped.iterrows():
        h = r["hour"]
        if h not in time_to_i:
            continue
        out[time_to_i[h], int(r["grid_idx"])] = float(r["value"])

    return out


def aggregate_fires_to_grid_hourly(
    df_fire: pd.DataFrame,
    times: pd.DatetimeIndex,
    grid_lat_flat: np.ndarray,
    grid_lon_flat: np.ndarray,
    radius_km: float,
) -> Tuple[np.ndarray, np.ndarray]:
    df = df_fire.copy()
    if "timestamp" not in df.columns:
        raise ValueError("FIRMS dataframe must contain a 'timestamp' column")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.dropna(subset=["timestamp", "latitude", "longitude"])
    df["hour"] = df["timestamp"].dt.floor("h")

    lat0 = float(np.mean(grid_lat_flat))
    lon0 = float(np.mean(grid_lon_flat))

    gx, gy = latlon_to_xy_km(grid_lat_flat, grid_lon_flat, lat0, lon0)
    tree = cKDTree(np.c_[gx, gy])

    n_time = len(times)
    n_nodes = grid_lat_flat.shape[0]
    fire_count = np.zeros((n_time, n_nodes), dtype=np.float32)
    frp_sum = np.zeros((n_time, n_nodes), dtype=np.float32)

    time_to_i = {t: i for i, t in enumerate(times)}

    for _, r in df.iterrows():
        h = r["hour"]
        ti = time_to_i.get(h)
        if ti is None:
            continue

        fx, fy = latlon_to_xy_km(np.array([r["latitude"]]), np.array([r["longitude"]]), lat0, lon0)
        idxs = tree.query_ball_point([float(fx[0]), float(fy[0])], r=float(radius_km))
        if not idxs:
            continue

        fire_count[ti, idxs] += 1.0
        frp = float(r.get("frp", 0.0))
        frp_sum[ti, idxs] += frp

    return fire_count, frp_sum


def build_and_save_tensor_dataset(
    cfg: ProjectConfig,
    ds_era5: xr.Dataset,
    pm25_grid: np.ndarray,
    fire_count: np.ndarray,
    frp_sum: np.ndarray,
    out_path: str | Path,
) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    times = pd.DatetimeIndex(pd.to_datetime(ds_era5["time"].values, utc=True))

    def _pick(name: str) -> np.ndarray:
        if name not in ds_era5:
            raise KeyError(f"ERA5 variable '{name}' not found in dataset")
        x = ds_era5[name].values
        if x.ndim == 4:
            x = x[:, 0]
        x = np.asarray(x, dtype=np.float32)
        x = x.reshape(x.shape[0], -1)
        return x

    u10 = _pick("u10")
    v10 = _pick("v10")
    blh = _pick("blh")
    rh = _pick("r")
    t = _pick("t")
    u850 = _pick("u")
    v850 = _pick("v")

    feature_names = [
        "pm25",
        "fire_count",
        "frp_sum",
        "u10",
        "v10",
        "u850",
        "v850",
        "blh",
        "rh",
        "t",
    ]

    x = np.stack(
        [
            pm25_grid,
            fire_count,
            frp_sum,
            u10,
            v10,
            u850,
            v850,
            blh,
            rh,
            t,
        ],
        axis=-1,
    ).astype(np.float32)

    train_end = pd.to_datetime(cfg.split.train_end, utc=True)
    train_mask = times <= train_end

    means, stds = nanmean_std(x[train_mask], axis=0)

    x_norm = (x - means) / stds

    np.savez_compressed(
        out,
        times=times.astype("datetime64[ns]").values,
        features=x_norm.astype(np.float32),
        targets=pm25_grid.astype(np.float32),
        feature_names=np.array(feature_names, dtype=object),
        means=means.astype(np.float32),
        stds=stds.astype(np.float32),
    )

    return out
