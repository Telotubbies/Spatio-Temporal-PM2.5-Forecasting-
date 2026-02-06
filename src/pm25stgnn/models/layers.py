from __future__ import annotations

import torch
from torch import nn


def scatter_softmax(src: torch.Tensor, index: torch.Tensor, n_nodes: int) -> torch.Tensor:
    if hasattr(torch, "scatter_reduce"):
        max_per = torch.full((n_nodes,), -1e9, device=src.device, dtype=src.dtype)
        max_per = max_per.scatter_reduce(0, index, src, reduce="amax", include_self=True)
    else:
        max_per = torch.full((n_nodes,), -1e9, device=src.device, dtype=src.dtype)
        max_per.index_put_((index,), torch.maximum(max_per[index], src), accumulate=False)

    src_exp = torch.exp(src - max_per[index])
    denom = torch.zeros((n_nodes,), device=src.device, dtype=src.dtype)
    denom = denom.index_add(0, index, src_exp)
    return src_exp / (denom[index] + 1e-8)


class EdgeAttnConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, edge_dim: int, dropout: float):
        super().__init__()
        self.lin_node = nn.Linear(in_dim, out_dim, bias=False)
        self.lin_edge = nn.Linear(edge_dim, out_dim, bias=False)
        self.att_src = nn.Parameter(torch.empty(out_dim))
        self.att_dst = nn.Parameter(torch.empty(out_dim))
        self.att_edge = nn.Parameter(torch.empty(out_dim))
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)
        self.res = nn.Linear(in_dim, out_dim, bias=False) if in_dim != out_dim else nn.Identity()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.lin_node.weight)
        nn.init.xavier_uniform_(self.lin_edge.weight)
        nn.init.normal_(self.att_src, std=0.02)
        nn.init.normal_(self.att_dst, std=0.02)
        nn.init.normal_(self.att_edge, std=0.02)
        if isinstance(self.res, nn.Linear):
            nn.init.xavier_uniform_(self.res.weight)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]
        n = x.shape[0]

        z = self.lin_node(x)
        e = self.lin_edge(edge_attr)

        s_src = (z[src] * self.att_src).sum(dim=-1)
        s_dst = (z[dst] * self.att_dst).sum(dim=-1)
        s_edge = (e * self.att_edge).sum(dim=-1)

        score = self.leaky_relu(s_src + s_dst + s_edge)
        alpha = scatter_softmax(score, dst, n)
        alpha = self.dropout(alpha)

        msg = (z[src] + e) * alpha.unsqueeze(-1)

        out = torch.zeros((n, z.shape[1]), device=x.device, dtype=z.dtype)
        out = out.index_add(0, dst, msg)

        out = out + self.res(x)
        return out
