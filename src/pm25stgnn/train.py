from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from pm25stgnn.config import load_config, resolve_device
from pm25stgnn.dataset import PM25TensorWindowDataset, TensorDatasetConfig
from pm25stgnn.graph import WindDirectedGraphBuilder, precompute_candidate_neighbors
from pm25stgnn.models.baselines import GRUBaseline
from pm25stgnn.models.stgnn import FeatureIndex, STGNN
from pm25stgnn.utils import masked_mae, masked_mse, masked_r2, set_seed


def make_loss(name: str, huber_delta: float) -> nn.Module:
    name = name.lower()
    if name == "mae":
        return nn.L1Loss(reduction="none")
    if name == "huber":
        return nn.HuberLoss(delta=float(huber_delta), reduction="none")
    raise ValueError("loss must be mae or huber")


def train_one_epoch(model, loader, opt, loss_fn, device, grad_clip_norm: float):
    model.train()
    total = 0.0
    count = 0
    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        m = batch["y_mask"].to(device)

        pred = model(x)
        per_node = loss_fn(pred, y)
        loss = (per_node * m).sum() / (m.sum() + 1e-8)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip_norm))
        opt.step()

        total += float(loss.item())
        count += 1
    return total / max(count, 1)


def _setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(str(log_path))
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    logger.propagate = False
    return logger


def _append_csv(path: Path, header: str, row: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(header + "\n", encoding="utf-8")
    with path.open("a", encoding="utf-8") as f:
        f.write(row + "\n")


def _save_checkpoint(path: Path, ckpt: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, path)


def _cleanup_step_checkpoints(out_dir: Path, keep_last_k: int) -> None:
    if keep_last_k <= 0:
        return
    files = sorted(out_dir.glob("step_*.pt"), key=lambda p: int(p.stem.split("_")[-1]))
    if len(files) <= keep_last_k:
        return
    for p in files[: -keep_last_k]:
        try:
            p.unlink(missing_ok=True)
        except Exception:
            pass


@torch.no_grad()
def eval_epoch(model, loader, device):
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
    mae = sum(maes) / max(len(maes), 1)
    rmse = (sum(mses) / max(len(mses), 1)) ** 0.5
    r2 = sum(r2s) / max(len(r2s), 1)
    return mae, rmse, r2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--horizon", type=int, required=True, choices=[24, 48])
    ap.add_argument("--model", type=str, default="stgnn", choices=["stgnn", "gru"])
    ap.add_argument("--dataset", type=str, default="data/processed/tensor_dataset.npz")
    ap.add_argument("--run_name", type=str, default="run")
    ap.add_argument("--no_resume", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)

    device = torch.device(resolve_device(cfg.train.device))

    ds_cfg = TensorDatasetConfig(window_hours=cfg.features.window_hours, horizon_hours=int(args.horizon))
    train_ds = PM25TensorWindowDataset(
        args.dataset,
        ds_cfg,
        split="train",
        train_end=cfg.split.train_end,
        val_end=cfg.split.val_end,
        test_end=cfg.split.test_end,
    )
    val_ds = PM25TensorWindowDataset(
        args.dataset,
        ds_cfg,
        split="val",
        train_end=cfg.split.train_end,
        val_end=cfg.split.val_end,
        test_end=cfg.split.test_end,
    )

    train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=cfg.train.batch_size, shuffle=False, num_workers=0)

    n_features = train_ds.features.shape[-1]

    if args.model == "gru":
        model = GRUBaseline(n_features=n_features, hidden_dim=cfg.model.hidden_dim, dropout=cfg.model.dropout).to(device)
    else:
        data = train_ds.data
        times = train_ds.times
        n_nodes = int(train_ds.features.shape[1])

        grid_meta_path = Path(args.dataset).with_suffix(".grid.npz")
        if not grid_meta_path.exists():
            raise FileNotFoundError(
                f"Missing grid metadata file: {grid_meta_path}. Build dataset with scripts/build_dataset first."
            )
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
            pm25=train_ds.feature_index("pm25"),
            u10=train_ds.feature_index("u10"),
            v10=train_ds.feature_index("v10"),
        )

        model = STGNN(
            n_features=n_features,
            hidden_dim=cfg.model.hidden_dim,
            spatial_layers=cfg.model.spatial_layers,
            dropout=cfg.model.dropout,
            feat_idx=feat_idx,
            graph_builder=builder,
        ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    loss_fn = make_loss(cfg.train.loss, cfg.train.huber_delta)

    out_dir = Path(cfg.outputs_dir) / args.run_name / f"h{args.horizon}_{args.model}"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = _setup_logger(out_dir / "train.log")
    logger.info(f"device={device}")
    logger.info(f"dataset={args.dataset}")
    logger.info(f"n_train={len(train_ds)} n_val={len(val_ds)} window={cfg.features.window_hours} horizon={args.horizon}")

    metrics_csv = out_dir / "metrics.csv"
    metrics_jsonl = out_dir / "metrics.jsonl"

    resume_enabled = bool(cfg.train.resume) and (not args.no_resume)
    last_ckpt_path = out_dir / "last.pt"
    start_epoch = 1
    global_step = 0
    best_mae = 1e9

    if resume_enabled and last_ckpt_path.exists():
        ckpt = torch.load(last_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"], strict=True)
        opt.load_state_dict(ckpt["opt_state"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        global_step = int(ckpt.get("global_step", 0))
        best_mae = float(ckpt.get("best_mae", 1e9))
        logger.info(f"resume_from={last_ckpt_path} start_epoch={start_epoch} global_step={global_step} best_mae={best_mae:.6f}")

    header = "epoch,global_step,train_loss,val_mae,val_rmse,val_r2,best_val_mae,elapsed_sec"

    for epoch in range(start_epoch, cfg.train.epochs + 1):
        t0 = time.time()

        model.train()
        total = 0.0
        count = 0
        for batch in train_loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            m = batch["y_mask"].to(device)

            pred = model(x)
            per_node = loss_fn(pred, y)
            loss = (per_node * m).sum() / (m.sum() + 1e-8)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(cfg.train.grad_clip_norm))
            opt.step()

            total += float(loss.item())
            count += 1
            global_step += 1

            if cfg.train.checkpoint_every_steps and (global_step % int(cfg.train.checkpoint_every_steps) == 0):
                step_ckpt = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "model_state": model.state_dict(),
                    "opt_state": opt.state_dict(),
                    "best_mae": best_mae,
                }
                _save_checkpoint(out_dir / f"step_{global_step}.pt", step_ckpt)
                _cleanup_step_checkpoints(out_dir, keep_last_k=int(cfg.train.keep_last_k_checkpoints))

        tr_loss = total / max(count, 1)
        val_mae, val_rmse, val_r2 = eval_epoch(model, val_loader, device)
        elapsed = time.time() - t0

        ckpt = {
            "epoch": epoch,
            "global_step": global_step,
            "model_state": model.state_dict(),
            "opt_state": opt.state_dict(),
            "val_mae": val_mae,
            "best_mae": best_mae,
        }

        if val_mae < best_mae:
            best_mae = val_mae
            ckpt["best_mae"] = best_mae
            _save_checkpoint(out_dir / "best.pt", ckpt)

        ckpt["best_mae"] = best_mae
        _save_checkpoint(out_dir / "last.pt", ckpt)

        row = f"{epoch},{global_step},{tr_loss:.6f},{val_mae:.6f},{val_rmse:.6f},{val_r2:.6f},{best_mae:.6f},{elapsed:.3f}"
        _append_csv(metrics_csv, header=header, row=row)
        with metrics_jsonl.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "epoch": epoch,
                        "global_step": global_step,
                        "train_loss": float(tr_loss),
                        "val_mae": float(val_mae),
                        "val_rmse": float(val_rmse),
                        "val_r2": float(val_r2),
                        "best_val_mae": float(best_mae),
                        "elapsed_sec": float(elapsed),
                    }
                )
                + "\n"
            )

        logger.info(
            f"Epoch {epoch:03d} | step={global_step} | train_loss={tr_loss:.4f} | val_mae={val_mae:.4f} | val_rmse={val_rmse:.4f} | val_r2={val_r2:.4f} | best_mae={best_mae:.4f} | sec={elapsed:.1f}"
        )


if __name__ == "__main__":
    main()
