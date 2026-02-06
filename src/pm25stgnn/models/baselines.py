from __future__ import annotations

import torch
from torch import nn


class GRUBaseline(nn.Module):
    def __init__(self, n_features: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.gru = nn.GRU(n_features, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, n, f = x.shape
        x_bn = x.permute(0, 2, 1, 3).reshape(b * n, t, f)
        _, h_last = self.gru(x_bn)
        h_last = self.dropout(h_last[-1])
        y = self.head(h_last).reshape(b, n)
        return y


def persistence_baseline(x: torch.Tensor, pm25_feature_index: int) -> torch.Tensor:
    return x[:, -1, :, pm25_feature_index]
