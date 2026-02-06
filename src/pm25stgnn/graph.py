from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from scipy.spatial import cKDTree

from pm25stgnn.utils import latlon_to_xy_km


@dataclass(frozen=True)
class CandidateNeighbors:
    idx: torch.LongTensor
    dx_hat: torch.FloatTensor
    dy_hat: torch.FloatTensor
    dist_km: torch.FloatTensor
    valid: torch.BoolTensor


def precompute_candidate_neighbors(
    grid_lat_flat: np.ndarray,
    grid_lon_flat: np.ndarray,
    radius_km: float,
    max_candidates: int = 48,
) -> CandidateNeighbors:
    n = int(grid_lat_flat.shape[0])
    lat0 = float(np.mean(grid_lat_flat))
    lon0 = float(np.mean(grid_lon_flat))

    gx, gy = latlon_to_xy_km(grid_lat_flat, grid_lon_flat, lat0, lon0)
    pts = np.c_[gx, gy]
    tree = cKDTree(pts)

    idx = np.full((n, max_candidates), -1, dtype=np.int64)
    dx_hat = np.zeros((n, max_candidates), dtype=np.float32)
    dy_hat = np.zeros((n, max_candidates), dtype=np.float32)
    dist_km = np.zeros((n, max_candidates), dtype=np.float32)
    valid = np.zeros((n, max_candidates), dtype=bool)

    for i in range(n):
        neigh = tree.query_ball_point(pts[i], r=float(radius_km))
        neigh = [j for j in neigh if j != i]
        if not neigh:
            continue

        dxy = pts[np.array(neigh)] - pts[i : i + 1]
        dist = np.sqrt((dxy**2).sum(axis=1))
        order = np.argsort(dist)[:max_candidates]
        neigh = np.array(neigh, dtype=np.int64)[order]
        dxy = dxy[order]
        dist = dist[order]

        k = int(neigh.shape[0])
        idx[i, :k] = neigh
        dist_km[i, :k] = dist.astype(np.float32)
        dx_hat[i, :k] = (dxy[:, 0] / (dist + 1e-8)).astype(np.float32)
        dy_hat[i, :k] = (dxy[:, 1] / (dist + 1e-8)).astype(np.float32)
        valid[i, :k] = True

    return CandidateNeighbors(
        idx=torch.from_numpy(idx),
        dx_hat=torch.from_numpy(dx_hat),
        dy_hat=torch.from_numpy(dy_hat),
        dist_km=torch.from_numpy(dist_km),
        valid=torch.from_numpy(valid),
    )


@dataclass
class WindDirectedGraphBuilder:
    candidates: CandidateNeighbors
    max_neighbors: int
    downwind_cos_threshold: float

    def build_edges_batched(
        self, u10: torch.Tensor, v10: torch.Tensor
    ) -> Tuple[torch.LongTensor, torch.FloatTensor]:
        device = u10.device
        cand = self.candidates

        idx = cand.idx.to(device)
        dx_hat = cand.dx_hat.to(device)
        dy_hat = cand.dy_hat.to(device)
        dist_km = cand.dist_km.to(device)
        valid = cand.valid.to(device)

        b, n = u10.shape
        m = idx.shape[1]
        k = int(self.max_neighbors)

        u = u10[:, :, None]
        v = v10[:, :, None]
        speed = torch.sqrt(u * u + v * v + 1e-8)
        ux = u / speed
        vy = v / speed

        cos = ux * dx_hat[None, :, :] + vy * dy_hat[None, :, :]

        score = cos * speed.squeeze(-1)[:, :, None] / (dist_km[None, :, :] + 1e-6)

        mask = valid[None, :, :] & (cos >= float(self.downwind_cos_threshold))
        score = torch.where(mask, score, torch.full_like(score, -1e9))

        topk_score, topk_pos = torch.topk(score, k=min(k, m), dim=-1)
        topk_idx = idx[None, :, :].expand(b, -1, -1).gather(-1, topk_pos)

        batch_offsets = (torch.arange(b, device=device).view(b, 1, 1) * n)
        src = (
            torch.arange(n, device=device).view(1, n, 1).expand(b, -1, topk_idx.shape[-1])
            + batch_offsets
        )
        dst = topk_idx + batch_offsets

        src = src.reshape(-1)
        dst = dst.reshape(-1)
        keep = topk_score.reshape(-1) > -1e8
        src = src[keep]
        dst = dst[keep]

        edge_index = torch.stack([src, dst], dim=0).long()

        wind_speed = speed.squeeze(-1).reshape(-1, 1).expand(-1, topk_idx.shape[-1]).reshape(-1)[keep]
        cos_kept = cos.gather(-1, topk_pos).reshape(-1)[keep]
        dist_kept = dist_km[None, :, :].expand(b, -1, -1).gather(-1, topk_pos).reshape(-1)[keep]

        edge_attr = torch.stack([wind_speed, cos_kept, dist_kept], dim=-1).float()
        return edge_index, edge_attr
