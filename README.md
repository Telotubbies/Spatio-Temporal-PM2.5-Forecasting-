# Spatio-Temporal PM2.5 Forecasting (Bangkok)

This repository implements an end-to-end **research-grade** PM2.5 forecasting system using a **wind-driven directed graph** over the **ERA5 0.25° grid** and a lightweight **Spatio-Temporal GNN**.

## What you get

- Data pipeline (API → caching → feature tensor)
- Wind-driven directed graph construction with edge features
- ST-GNN model (custom message passing; no CUDA-only dependencies)
- Training/evaluation for 24h and 48h horizons
- Baselines: persistence, non-graph GRU

## Setup

1. Create a Python environment.
2. Install deps:

```bash
pip install -r requirements.txt
pip install -e .
```

## Credentials / API setup

- ERA5 (Copernicus CDS): configure `~/.cdsapirc` (see CDS documentation).
- NASA FIRMS: set `NASA_FIRMS_API_KEY` in `.env`.
- OpenAQ: no key required for basic usage (rate limits apply).

Create `.env`:

```text
NASA_FIRMS_API_KEY=YOUR_KEY
```

## Quickstart (high level)

1. Download raw data:

```bash
python -m pm25stgnn.scripts.download --config configs/default.yaml --start 2021-01-01 --end 2023-12-31
```

2. Build features + graph:

```bash
python -m pm25stgnn.scripts.build_dataset --config configs/default.yaml --start 2021-01-01 --end 2023-12-31
```

3. Train:

```bash
python -m pm25stgnn.train --config configs/default.yaml --horizon 24
python -m pm25stgnn.train --config configs/default.yaml --horizon 48
```

4. Evaluate:

```bash
python -m pm25stgnn.eval --config configs/default.yaml --horizon 24 --model stgnn --checkpoint outputs/<run_name>/h24_stgnn/best.pt
```

## One-command pipeline (collect → build → train → eval)

```bash
python -m pm25stgnn.scripts.run_all --config configs/default.yaml --start 2021-01-01 --end 2023-12-31
```

This creates a timestamped run folder under `outputs/` and writes `outputs/<run_name>/pipeline.log`.

## Auto-checkpoint + auto-resume

Training writes:

- `best.pt` (best validation MAE)
- `last.pt` (latest)
- `step_*.pt` (periodic checkpoints, retained up to `train.keep_last_k_checkpoints`)
- `train.log`, `metrics.csv`, `metrics.jsonl`

If `train.resume: true`, rerunning the same command will auto-resume from `last.pt`. To disable resume:

```bash
python -m pm25stgnn.train --config configs/default.yaml --horizon 24 --model stgnn --no_resume
```

## Methodology

See `docs/methodology.md` for a paper-style explanation.

## AMD GPU note (ROCm)

This code uses standard PyTorch device selection. If you have ROCm-enabled PyTorch installed, `torch.cuda.is_available()` will be true and training will run on the AMD GPU. The `pip install torch` wheel on Windows may not provide ROCm; in that case it will fall back to CPU.
