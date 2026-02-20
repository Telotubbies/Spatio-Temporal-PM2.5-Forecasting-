from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from pm25stgnn.config import load_config, resolve_device
from pm25stgnn.dataset import PM25TensorWindowDataset, TensorDatasetConfig
from pm25stgnn.graph import WindDirectedGraphBuilder, precompute_candidate_neighbors
from pm25stgnn.models.baselines import GRUBaseline, persistence_baseline
from pm25stgnn.models.stgnn import FeatureIndex, STGNN
from pm25stgnn.utils import masked_mae, masked_mse, masked_r2


@torch.no_grad()
def eval_model(model, loader, device):
    model.eval()
    maes = []
    mses = []
    r2s = []
    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        m = batch["y_mask"].to(device)
        pred = model(x)
        maes.append(masked_mae(pred, y, m).item())
        mses.append(masked_mse(pred, y, m).item())
        r2s.append(masked_r2(pred, y, m).item())
    mae = float(sum(maes) / max(len(maes), 1))
    rmse = float((sum(mses) / max(len(mses), 1)) ** 0.5)
    r2 = float(sum(r2s) / max(len(r2s), 1))
    return mae, rmse, r2


@torch.no_grad()
def eval_persistence(loader, device, pm25_idx: int):
    maes = []
    mses = []
    r2s = []
    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        m = batch["y_mask"].to(device)
        pred = persistence_baseline(x, pm25_feature_index=pm25_idx)
        maes.append(masked_mae(pred, y, m).item())
        mses.append(masked_mse(pred, y, m).item())
        r2s.append(masked_r2(pred, y, m).item())
    mae = float(sum(maes) / max(len(maes), 1))
    rmse = float((sum(mses) / max(len(mses), 1)) ** 0.5)
    r2 = float(sum(r2s) / max(len(r2s), 1))
    return mae, rmse, r2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=False, default="")
    ap.add_argument("--horizon", type=int, required=True, choices=[24, 48])
    ap.add_argument("--model", type=str, default="stgnn", choices=["stgnn", "gru"])
    ap.add_argument("--dataset", type=str, default="data/processed/tensor_dataset.npz")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = torch.device(resolve_device(cfg.train.device))

    ds_cfg = TensorDatasetConfig(window_hours=cfg.features.window_hours, horizon_hours=int(args.horizon))
    test_ds = PM25TensorWindowDataset(
        args.dataset,
        ds_cfg,
        split="test",
        train_end=cfg.split.train_end,
        val_end=cfg.split.val_end,
        test_end=cfg.split.test_end,
    )
    loader = DataLoader(test_ds, batch_size=cfg.train.batch_size, shuffle=False, num_workers=0)

    pm25_idx = test_ds.feature_index("pm25")
    p_mae, p_rmse, p_r2 = eval_persistence(loader, device, pm25_idx)

    n_features = test_ds.features.shape[-1]

    if args.model == "gru":
        model = GRUBaseline(n_features=n_features, hidden_dim=cfg.model.hidden_dim, dropout=cfg.model.dropout).to(device)
    else:
        grid_meta_path = Path(args.dataset).with_suffix(".grid.npz")
        grid_meta = np.load(grid_meta_path, allow_pickle=True)
        lat_flat = grid_meta["lat_flat"].astype(np.float64)
        lon_flat = grid_meta["lon_flat"].astype(np.float64)

        candidates = precompute_candidate_neighbors(
            lat_flat,
            lon_flat,
            radius_km=cfg.graph.radius_km,
            max_candidates=48,
        )
        builder = WindDirectedGraphBuilder(
            candidates=candidates,
            max_neighbors=cfg.graph.max_neighbors,
            downwind_cos_threshold=cfg.graph.downwind_cos_threshold,
        )

        feat_idx = FeatureIndex(
            pm25=test_ds.feature_index("pm25"),
            u10=test_ds.feature_index("u10"),
            v10=test_ds.feature_index("v10"),
        )

        model = STGNN(
            n_features=n_features,
            hidden_dim=cfg.model.hidden_dim,
            spatial_layers=cfg.model.spatial_layers,
            dropout=cfg.model.dropout,
            feat_idx=feat_idx,
            graph_builder=builder,
        ).to(device)

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state"], strict=True)

    mae, rmse, r2 = eval_model(model, loader, device)

    print(f"Persistence | MAE={p_mae:.4f} RMSE={p_rmse:.4f} R2={p_r2:.4f}")
    print(f"{args.model.upper()} | MAE={mae:.4f} RMSE={rmse:.4f} R2={r2:.4f}")


if __name__ == "__main__":
    main()
