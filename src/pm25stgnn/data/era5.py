from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import cdsapi


@dataclass(frozen=True)
class Era5Request:
    start_date: str
    end_date: str
    area: List[float]
    grid: List[float]


def _date_range_ymd(start_date: str, end_date: str) -> List[str]:
    import pandas as pd

    dt0 = pd.to_datetime(start_date)
    dt1 = pd.to_datetime(end_date)
    days = pd.date_range(dt0.floor("D"), dt1.floor("D"), freq="D")
    return [d.strftime("%Y-%m-%d") for d in days]


def download_era5_single_levels(
    out_path: str | Path,
    req: Era5Request,
    variables: Iterable[str],
) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    c = cdsapi.Client()

    dates = _date_range_ymd(req.start_date, req.end_date)
    request = {
        "product_type": "reanalysis",
        "format": "netcdf",
        "variable": list(variables),
        "date": "/".join(dates),
        "time": [f"{h:02d}:00" for h in range(24)],
        "area": req.area,
        "grid": req.grid,
    }

    c.retrieve("reanalysis-era5-single-levels", request, str(out))
    return out


def download_era5_pressure_levels(
    out_path: str | Path,
    req: Era5Request,
    variables: Iterable[str],
    pressure_level_hpa: int,
) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    c = cdsapi.Client()

    dates = _date_range_ymd(req.start_date, req.end_date)
    request = {
        "product_type": "reanalysis",
        "format": "netcdf",
        "variable": list(variables),
        "pressure_level": [str(int(pressure_level_hpa))],
        "date": "/".join(dates),
        "time": [f"{h:02d}:00" for h in range(24)],
        "area": req.area,
        "grid": req.grid,
    }

    c.retrieve("reanalysis-era5-pressure-levels", request, str(out))
    return out


def _month_starts(start_date: str, end_date: str) -> List[str]:
    import pandas as pd

    dt0 = pd.to_datetime(start_date).floor("D")
    dt1 = pd.to_datetime(end_date).floor("D")
    ms = pd.date_range(dt0.replace(day=1), dt1.replace(day=1), freq="MS")
    return [d.strftime("%Y-%m-%d") for d in ms]


def download_era5_single_levels_monthly(
    out_dir: str | Path,
    req: Era5Request,
    variables: Iterable[str],
) -> List[Path]:
    import pandas as pd

    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    dt0 = pd.to_datetime(req.start_date).floor("D")
    dt1 = pd.to_datetime(req.end_date).floor("D")
    month_starts = pd.date_range(dt0.replace(day=1), dt1.replace(day=1), freq="MS")

    paths: List[Path] = []
    for ms in month_starts:
        me = (ms + pd.offsets.MonthEnd(1)).floor("D")
        s = max(dt0, ms)
        e = min(dt1, me)
        out = out_root / f"era5_single_levels_{s.strftime('%Y%m')}.nc"
        subreq = Era5Request(start_date=s.strftime("%Y-%m-%d"), end_date=e.strftime("%Y-%m-%d"), area=req.area, grid=req.grid)
        paths.append(download_era5_single_levels(out, subreq, variables))
    return paths


def download_era5_pressure_levels_monthly(
    out_dir: str | Path,
    req: Era5Request,
    variables: Iterable[str],
    pressure_level_hpa: int,
) -> List[Path]:
    import pandas as pd

    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    dt0 = pd.to_datetime(req.start_date).floor("D")
    dt1 = pd.to_datetime(req.end_date).floor("D")
    month_starts = pd.date_range(dt0.replace(day=1), dt1.replace(day=1), freq="MS")

    paths: List[Path] = []
    for ms in month_starts:
        me = (ms + pd.offsets.MonthEnd(1)).floor("D")
        s = max(dt0, ms)
        e = min(dt1, me)
        out = out_root / f"era5_pressure_levels_{pressure_level_hpa}hPa_{s.strftime('%Y%m')}.nc"
        subreq = Era5Request(start_date=s.strftime("%Y-%m-%d"), end_date=e.strftime("%Y-%m-%d"), area=req.area, grid=req.grid)
        paths.append(download_era5_pressure_levels(out, subreq, variables, pressure_level_hpa=pressure_level_hpa))
    return paths
