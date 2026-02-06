from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


def load_yaml(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_in(d: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


@dataclass(frozen=True)
class RegionConfig:
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float


@dataclass(frozen=True)
class GraphConfig:
    radius_km: float
    max_neighbors: int
    downwind_cos_threshold: float


@dataclass(frozen=True)
class FeatureConfig:
    window_hours: int


@dataclass(frozen=True)
class TrainConfig:
    device: str
    batch_size: int
    epochs: int
    lr: float
    weight_decay: float
    loss: str
    huber_delta: float
    use_amp: bool
    grad_clip_norm: float
    checkpoint_every_steps: int
    keep_last_k_checkpoints: int
    resume: bool


@dataclass(frozen=True)
class ModelConfig:
    hidden_dim: int
    spatial_layers: int
    temporal: str
    dropout: float


@dataclass(frozen=True)
class SplitConfig:
    train_end: str
    val_end: str
    test_end: str


@dataclass(frozen=True)
class ProjectConfig:
    seed: int
    data_dir: Path
    outputs_dir: Path
    region: RegionConfig
    graph: GraphConfig
    features: FeatureConfig
    model: ModelConfig
    train: TrainConfig
    split: SplitConfig
    raw: Dict[str, Any]


def parse_config(cfg: Dict[str, Any]) -> ProjectConfig:
    project = cfg["project"]
    data_dir = Path(project["data_dir"])
    outputs_dir = Path(project["outputs_dir"])

    region = cfg["region"]
    region_cfg = RegionConfig(
        lat_min=float(region["lat_min"]),
        lat_max=float(region["lat_max"]),
        lon_min=float(region["lon_min"]),
        lon_max=float(region["lon_max"]),
    )

    graph = cfg["graph"]
    graph_cfg = GraphConfig(
        radius_km=float(graph["radius_km"]),
        max_neighbors=int(graph["max_neighbors"]),
        downwind_cos_threshold=float(graph["downwind_cos_threshold"]),
    )

    features = cfg["features"]
    feat_cfg = FeatureConfig(window_hours=int(features["window_hours"]))

    model = cfg["model"]
    model_cfg = ModelConfig(
        hidden_dim=int(model["hidden_dim"]),
        spatial_layers=int(model["spatial_layers"]),
        temporal=str(model["temporal"]),
        dropout=float(model["dropout"]),
    )

    train = cfg["train"]
    train_cfg = TrainConfig(
        device=str(train["device"]),
        batch_size=int(train["batch_size"]),
        epochs=int(train["epochs"]),
        lr=float(train["lr"]),
        weight_decay=float(train["weight_decay"]),
        loss=str(train["loss"]),
        huber_delta=float(train["huber_delta"]),
        use_amp=bool(train["use_amp"]),
        grad_clip_norm=float(train["grad_clip_norm"]),
        checkpoint_every_steps=int(train.get("checkpoint_every_steps", 0)),
        keep_last_k_checkpoints=int(train.get("keep_last_k_checkpoints", 5)),
        resume=bool(train.get("resume", True)),
    )

    split = cfg["split"]
    split_cfg = SplitConfig(
        train_end=str(split["train_end"]),
        val_end=str(split["val_end"]),
        test_end=str(split["test_end"]),
    )

    return ProjectConfig(
        seed=int(project["seed"]),
        data_dir=data_dir,
        outputs_dir=outputs_dir,
        region=region_cfg,
        graph=graph_cfg,
        features=feat_cfg,
        model=model_cfg,
        train=train_cfg,
        split=split_cfg,
        raw=cfg,
    )


def load_config(path: str | Path) -> ProjectConfig:
    cfg = load_yaml(path)
    return parse_config(cfg)


def resolve_device(device_cfg: str) -> str:
    import torch

    if device_cfg.lower() == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_cfg
