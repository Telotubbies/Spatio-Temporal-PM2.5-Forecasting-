# Methodology (Paper-Style)

## Problem formulation

We forecast hourly PM2.5 concentration (μg/m³) over a Bangkok-centered ~300×300 km domain discretized on the ERA5 0.25° grid.

Let:

- `N` be the number of grid cells (graph nodes)
- `F` be the number of node features
- `T` be the historical window length (hours)
- `H ∈ {24, 48}` be the forecast horizon (hours)

For each reference time `t`, we construct an input tensor:

- `X_t ∈ R^{T×N×F}` containing features for times `[t-T+1, …, t]`

and predict:

- `ŷ_{t+H} ∈ R^{N}` (or a subset of nodes) representing PM2.5 at time `t+H`.

## Node features

Per grid cell and hour:

- `pm25` (station-mapped; NaN where unavailable)
- wildfire: `fire_count`, `frp_sum` aggregated within 150 km and mapped to grid
- meteorology from ERA5: `u10`, `v10`, `u850`, `v850`, `blh`, `rh`, `t`

All features are normalized **per grid cell** using train-split statistics with NaN-safe estimation.

## Wind-driven directed graph (transport prior)

We encode an advective transport prior with a **directed**, time-varying neighborhood.

For node `i` at time `t`, denote its near-surface wind vector:

- `w_i(t) = (u10_i(t), v10_i(t))` in (east, north) components.

For candidate neighbor `j` within radius `R=150 km`, let `d_ij` be the displacement vector from `i` to `j` in the same coordinate basis (east, north), and `d̂_ij = d_ij / ||d_ij||`.

Define the downwind alignment:

- `cosθ_ij(t) = (w_i(t) · d̂_ij) / (||w_i(t)|| + ε)`.

We create a directed edge `i → j` if:

- `||d_ij|| ≤ R`
- `cosθ_ij(t) ≥ τ` (default `τ=0`, i.e., at most 90° from wind direction)

We cap outgoing neighbors to `K` by selecting the top `K` candidates by a transport score (e.g., `cosθ_ij(t) * ||w_i(t)|| / (||d_ij|| + ε)`), enforcing a sparse graph.

### Edge features

For each edge `i → j` at time `t`:

- wind speed: `||w_i(t)||`
- alignment: `cosθ_ij(t)`
- distance: `||d_ij||`

These features allow the spatial message passing to modulate strength by transport plausibility.

## ST-GNN architecture (CPU/ROCm-friendly)

Input: `X ∈ R^{B×T×N×F}`.

1. **Spatial encoder** applied per time step:

   - reshape to `X_s ∈ R^{(B·T)×N×F}`
   - apply `L` layers of directed edge-aware attention/message passing
   - output `H_s ∈ R^{(B·T)×N×D}`

2. **Temporal model** per node:

   - reshape to `H_t ∈ R^{(B·N)×T×D}`
   - apply GRU (or TCN)
   - take last hidden state `h_last ∈ R^{(B·N)×D}`
   - reshape to `R^{B×N×D}`

3. **Readout**:

   - linear layer to predict `ŷ ∈ R^{B×N}`.

Loss is computed with a mask for nodes where PM2.5 is observed at the target time.

## Training protocol

- Sliding window sampling with chronological split (train/val/test)
- No shuffling across time order when forming splits
- Loss: MAE or Huber
- Metrics: MAE, RMSE, R²
- Baselines:
  - Persistence: `ŷ_{t+H} = y_t`
  - Non-graph GRU using the same features but without message passing

## Ablations (recommended)

- Remove wildfire features (`fire_count`, `frp_sum`)
- Use static graph (climatological wind) vs dynamic wind-directed edges
- Use u10/v10 vs u850/v850 for direction
- Replace attention with simple mean aggregation
- Vary radius `R` and neighbor cap `K`
- Vary history window `T`
