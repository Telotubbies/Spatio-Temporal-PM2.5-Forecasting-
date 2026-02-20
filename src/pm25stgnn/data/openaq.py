from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests


@dataclass(frozen=True)
class OpenAQQuery:
    start: str
    end: str
    west: float
    south: float
    east: float
    north: float
    limit: int = 1000


def fetch_openaq_measurements_pm25(query: OpenAQQuery, timeout_s: int = 60) -> pd.DataFrame:
    base = "https://api.openaq.org/v2/measurements"

    params = {
        "date_from": query.start,
        "date_to": query.end,
        "parameter": "pm25",
        "bbox": f"{query.west},{query.south},{query.east},{query.north}",
        "limit": int(query.limit),
        "page": 1,
        "sort": "asc",
        "order_by": "datetime",
    }

    rows: List[Dict] = []
    while True:
        r = requests.get(base, params=params, timeout=timeout_s)
        r.raise_for_status()
        j = r.json()
        results = j.get("results", [])
        if not results:
            break
        rows.extend(results)

        meta = j.get("meta", {})
        found = int(meta.get("found", 0))
        page = int(meta.get("page", params["page"]))
        limit = int(meta.get("limit", params["limit"]))
        if page * limit >= found:
            break
        params["page"] = page + 1

    if not rows:
        return pd.DataFrame(columns=["timestamp", "value", "lat", "lon", "location", "sensorType", "unit"])

    out_rows = []
    for r0 in rows:
        coords = ((r0.get("coordinates") or {}) if isinstance(r0.get("coordinates"), dict) else {})
        dt = (r0.get("date") or {}).get("utc")
        out_rows.append(
            {
                "timestamp": pd.to_datetime(dt, utc=True, errors="coerce"),
                "value": r0.get("value"),
                "unit": r0.get("unit"),
                "lat": coords.get("latitude"),
                "lon": coords.get("longitude"),
                "location": r0.get("location"),
                "sensorType": r0.get("sensorType"),
            }
        )

    df = pd.DataFrame(out_rows)
    return df
