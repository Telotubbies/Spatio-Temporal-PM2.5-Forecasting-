from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class TensorDatasetConfig:
    window_hours: int
    horizon_hours: int


class PM25TensorWindowDataset(Dataset):
    def __init__(
        self,
        npz_path: str | Path,
        cfg: TensorDatasetConfig,
        split: str,
        train_end: str,
        val_end: str,
        test_end: str,
    ):
        self.data = np.load(npz_path, allow_pickle=True)
        self.times = pd.DatetimeIndex(pd.to_datetime(self.data["times"], utc=True))
        self.features = self.data["features"].astype(np.float32)
        self.targets = self.data["targets"].astype(np.float32)
        self.feature_names = [str(x) for x in self.data["feature_names"].tolist()]

        self.window = int(cfg.window_hours)
        self.horizon = int(cfg.horizon_hours)

        train_end_ts = pd.to_datetime(train_end, utc=True)
        val_end_ts = pd.to_datetime(val_end, utc=True)
        test_end_ts = pd.to_datetime(test_end, utc=True)

        if split == "train":
            idx_mask = self.times <= train_end_ts
        elif split == "val":
            idx_mask = (self.times > train_end_ts) & (self.times <= val_end_ts)
        elif split == "test":
            idx_mask = (self.times > val_end_ts) & (self.times <= test_end_ts)
        else:
            raise ValueError("split must be one of: train, val, test")

        self.valid_time_idx = np.where(idx_mask)[0]
        self.valid_time_idx = self.valid_time_idx[self.valid_time_idx >= (self.window - 1)]
        self.valid_time_idx = self.valid_time_idx[self.valid_time_idx + self.horizon < len(self.times)]

    def __len__(self) -> int:
        return int(self.valid_time_idx.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        t_end = int(self.valid_time_idx[idx])
        t0 = t_end - (self.window - 1)
        t_y = t_end + self.horizon

        x = self.features[t0 : t_end + 1]
        y = self.targets[t_y]

        y_mask = np.isfinite(y).astype(np.float32)
        y = np.where(np.isfinite(y), y, 0.0).astype(np.float32)

        return {
            "x": torch.from_numpy(x),
            "y": torch.from_numpy(y),
            "y_mask": torch.from_numpy(y_mask),
        }

    def feature_index(self, name: str) -> int:
        return int(self.feature_names.index(name))
