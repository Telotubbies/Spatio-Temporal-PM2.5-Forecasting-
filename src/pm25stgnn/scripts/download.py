from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from pm25stgnn.config import load_config
from pm25stgnn.data.era5 import (
    Era5Request,
    download_era5_pressure_levels_monthly,
    download_era5_single_levels_monthly,
)
from pm25stgnn.data.firms import download_firms_range


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--start", required=True, help="ISO date/time, e.g., 2021-01-01")
    ap.add_argument("--end", required=True, help="ISO date/time, e.g., 2023-12-31")
    args = ap.parse_args()

    load_dotenv()
    cfg = load_config(args.config)

    data_dir = Path(cfg.data_dir)
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    region = cfg.raw["region"]
    area = [region["lat_max"], region["lon_min"], region["lat_min"], region["lon_max"]]
    grid = [cfg.raw["era5"]["grid_res_deg"], cfg.raw["era5"]["grid_res_deg"]]

    req = Era5Request(start_date=args.start, end_date=args.end, area=area, grid=grid)

    single_vars = [
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "boundary_layer_height",
        "2m_temperature",
        "2m_dewpoint_temperature",
    ]
    pl_vars = ["u_component_of_wind", "v_component_of_wind"]

    download_era5_single_levels_monthly(raw_dir / "era5_single", req, single_vars)
    download_era5_pressure_levels_monthly(
        raw_dir / "era5_pl_850",
        req,
        pl_vars,
        pressure_level_hpa=int(cfg.raw["era5"]["pressure_level_hpa"]),
    )

    api_key = os.environ.get("NASA_FIRMS_API_KEY", "")
    if not api_key:
        raise RuntimeError("NASA_FIRMS_API_KEY is not set. Put it in .env or environment variables.")

    firms_sources = cfg.raw["firms"]["sources"]
    for src in firms_sources:
        download_firms_range(
            api_key=api_key,
            source=str(src),
            west=float(region["lon_min"]),
            south=float(region["lat_min"]),
            east=float(region["lon_max"]),
            north=float(region["lat_max"]),
            start_date=args.start,
            end_date=args.end,
            out_dir=raw_dir / "firms" / str(src),
            chunk_days=5,
        )


if __name__ == "__main__":
    main()
