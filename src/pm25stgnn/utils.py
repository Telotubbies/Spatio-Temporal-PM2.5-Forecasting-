from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(p: str | Path) -> Path:
    path = Path(p)
    path.mkdir(parents=True, exist_ok=True)
    return path


def haversine_km(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    r = 6371.0
    lat1r = np.deg2rad(lat1)
    lon1r = np.deg2rad(lon1)
    lat2r = np.deg2rad(lat2)
    lon2r = np.deg2rad(lon2)
    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1r) * np.cos(lat2r) * (np.sin(dlon / 2.0) ** 2)
    c = 2.0 * np.arcsin(np.sqrt(a))
    return r * c


def latlon_to_xy_km(lat: np.ndarray, lon: np.ndarray, lat0: float, lon0: float) -> Tuple[np.ndarray, np.ndarray]:
    r = 6371.0
    latr = np.deg2rad(lat)
    lonr = np.deg2rad(lon)
    lat0r = math.radians(lat0)
    lon0r = math.radians(lon0)
    x = r * (lonr - lon0r) * math.cos(lat0r)
    y = r * (latr - lat0r)
    return x, y


def nanmean_std(x: np.ndarray, axis: int) -> Tuple[np.ndarray, np.ndarray]:
    mean = np.nanmean(x, axis=axis)
    std = np.nanstd(x, axis=axis)
    std = np.where(std < 1e-6, 1.0, std)
    return mean, std


def masked_mae(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    diff = torch.abs(pred - target)
    diff = diff * mask
    return diff.sum() / (mask.sum() + 1e-8)


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    diff2 = (pred - target) ** 2
    diff2 = diff2 * mask
    return diff2.sum() / (mask.sum() + 1e-8)


def masked_r2(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    target_mean = (target * mask).sum() / (mask.sum() + 1e-8)
    ss_res = ((pred - target) ** 2 * mask).sum()
    ss_tot = (((target - target_mean) ** 2) * mask).sum()
    return 1.0 - ss_res / (ss_tot + 1e-8)
