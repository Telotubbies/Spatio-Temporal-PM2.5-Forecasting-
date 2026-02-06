from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import nn

from pm25stgnn.graph import WindDirectedGraphBuilder
from pm25stgnn.models.layers import EdgeAttnConv


@dataclass(frozen=True)
class FeatureIndex:
    pm25: int
    u10: int
    v10: int


class SpatialEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int, edge_dim: int, dropout: float):
        super().__init__()
        layers = []
        d0 = in_dim
        for _ in range(int(num_layers)):
            layers.append(EdgeAttnConv(d0, hidden_dim, edge_dim=edge_dim, dropout=dropout))
            layers.append(nn.ELU())
            layers.append(nn.Dropout(dropout))
            d0 = hidden_dim
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.net:
            if isinstance(layer, EdgeAttnConv):
                h = layer(h, edge_index, edge_attr)
            else:
                h = layer(h)
        return h


class STGNN(nn.Module):
    def __init__(
        self,
        n_features: int,
        hidden_dim: int,
        spatial_layers: int,
        dropout: float,
        feat_idx: FeatureIndex,
        graph_builder: WindDirectedGraphBuilder,
    ):
        super().__init__()
        self.feat_idx = feat_idx
        self.graph_builder = graph_builder

        self.spatial = SpatialEncoder(
            in_dim=n_features,
            hidden_dim=hidden_dim,
            num_layers=spatial_layers,
            edge_dim=3,
            dropout=dropout,
        )
        self.temporal = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.readout = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, n, f = x.shape

        spatial_out = []
        for ti in range(t):
            xt = x[:, ti]
            u10 = xt[:, :, self.feat_idx.u10]
            v10 = xt[:, :, self.feat_idx.v10]
            edge_index, edge_attr = self.graph_builder.build_edges_batched(u10=u10, v10=v10)

            xt_flat = xt.reshape(b * n, f)
            ht = self.spatial(xt_flat, edge_index=edge_index, edge_attr=edge_attr)
            spatial_out.append(ht.reshape(b, n, -1))

        h = torch.stack(spatial_out, dim=1)

        h_bn = h.permute(0, 2, 1, 3).reshape(b * n, t, -1)
        out_seq, h_last = self.temporal(h_bn)
        h_last = h_last[-1]

        y = self.readout(h_last).reshape(b, n)
        return y
