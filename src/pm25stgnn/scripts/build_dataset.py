from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pm25stgnn.config import load_config
from pm25stgnn.data.build import (
    aggregate_fires_to_grid_hourly,
    aggregate_pm25_to_grid_hourly,
    build_and_save_tensor_dataset,
    grid_centers_flat,
    load_era5_features,
    make_era5_grid,
    qc_pm25,
)
from pm25stgnn.data.firms import read_firms_csv
from pm25stgnn.data.openaq import OpenAQQuery, fetch_openaq_measurements_pm25


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--era5_single", default="data/raw/era5_single")
    ap.add_argument("--era5_pl", default="data/raw/era5_pl_850")
    ap.add_argument("--firms_dir", default="data/raw/firms/VIIRS_SNPP_NRT")
    ap.add_argument("--out", default="data/processed/tensor_dataset.npz")
    args = ap.parse_args()

    cfg = load_config(args.config)

    grid = make_era5_grid(cfg.raw["region"], res_deg=float(cfg.raw["era5"]["grid_res_deg"]))
    lat_flat, lon_flat = grid_centers_flat(grid)

    ds = load_era5_features(args.era5_single, args.era5_pl)

    times = pd.DatetimeIndex(pd.to_datetime(ds["time"].values, utc=True))
    times = times[(times >= pd.to_datetime(args.start, utc=True)) & (times <= pd.to_datetime(args.end, utc=True))]

    ds = ds.sel(time=times)

    q = OpenAQQuery(
        start=args.start,
        end=args.end,
        west=float(cfg.raw["region"]["lon_min"]),
        south=float(cfg.raw["region"]["lat_min"]),
        east=float(cfg.raw["region"]["lon_max"]),
        north=float(cfg.raw["region"]["lat_max"]),
    )
    pm25_df = fetch_openaq_measurements_pm25(q)
    pm25_df = qc_pm25(pm25_df, cfg.raw["pm25"]["qc"]["min_value"], cfg.raw["pm25"]["qc"]["max_value"])
    pm25_grid = aggregate_pm25_to_grid_hourly(pm25_df, times, lat_flat, lon_flat)

    firms_dir = Path(args.firms_dir)
    firms_files = sorted(firms_dir.glob("*.csv"))
    if not firms_files:
        raise FileNotFoundError(f"No FIRMS CSV files found in: {firms_dir}")
    fire_df = pd.concat([read_firms_csv(p) for p in firms_files], ignore_index=True)
    fire_count, frp_sum = aggregate_fires_to_grid_hourly(
        fire_df,
        times,
        lat_flat,
        lon_flat,
        radius_km=float(cfg.raw["firms"]["radius_km"]),
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    build_and_save_tensor_dataset(cfg, ds, pm25_grid, fire_count, frp_sum, out_path)

    grid_meta_path = out_path.with_suffix(".grid.npz")
    np.savez_compressed(grid_meta_path, lat_flat=lat_flat.astype(np.float32), lon_flat=lon_flat.astype(np.float32))


if __name__ == "__main__":
    main()
