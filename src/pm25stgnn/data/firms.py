from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import requests


@dataclass(frozen=True)
class FirmsQuery:
    source: str
    west: float
    south: float
    east: float
    north: float
    days: int


def download_firms_csv(
    api_key: str,
    query: FirmsQuery,
    out_path: str | Path,
    date_ymd: Optional[str] = None,
    timeout_s: int = 60,
) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    bbox = f"{query.west},{query.south},{query.east},{query.north}"
    url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{api_key}/{query.source}/{bbox}/{int(query.days)}"
    if date_ymd is not None:
        url = f"{url}/{date_ymd}"

    r = requests.get(url, timeout=timeout_s)
    r.raise_for_status()
    out.write_bytes(r.content)
    return out


def download_firms_range(
    api_key: str,
    source: str,
    west: float,
    south: float,
    east: float,
    north: float,
    start_date: str,
    end_date: str,
    out_dir: str | Path,
    chunk_days: int = 5,
) -> List[Path]:
    import pandas as pd

    if chunk_days < 1 or chunk_days > 5:
        raise ValueError("FIRMS Area API supports chunk_days in [1, 5]")

    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    dt0 = pd.to_datetime(start_date).tz_localize("UTC") if pd.to_datetime(start_date).tzinfo is None else pd.to_datetime(start_date)
    dt1 = pd.to_datetime(end_date).tz_localize("UTC") if pd.to_datetime(end_date).tzinfo is None else pd.to_datetime(end_date)
    d0 = dt0.floor("D")
    d1 = dt1.floor("D")

    cur = d0
    paths: List[Path] = []
    while cur <= d1:
        chunk_end = min(cur + pd.Timedelta(days=chunk_days - 1), d1)
        days = int((chunk_end - cur).days) + 1

        q = FirmsQuery(source=source, west=west, south=south, east=east, north=north, days=days)
        date_ymd = chunk_end.strftime("%Y-%m-%d")
        out_path = out_root / f"firms_{source}_{cur.strftime('%Y%m%d')}_{chunk_end.strftime('%Y%m%d')}.csv"
        paths.append(download_firms_csv(api_key, q, out_path, date_ymd=date_ymd))

        cur = chunk_end + pd.Timedelta(days=1)

    return paths


def read_firms_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "acq_date" in df.columns and "acq_time" in df.columns:
        df["acq_time"] = df["acq_time"].astype(str).str.zfill(4)
        ts = df["acq_date"].astype(str) + " " + df["acq_time"].str.slice(0, 2) + ":" + df["acq_time"].str.slice(2, 4)
        df["timestamp"] = pd.to_datetime(ts, utc=True, errors="coerce")
    return df
