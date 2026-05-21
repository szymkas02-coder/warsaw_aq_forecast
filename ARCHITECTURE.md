# Warsaw PM2.5 Forecasting — Python Architecture Guide

This document explains every module, class, and major function in the
`warsaw_aq_forecast/` project to someone who has not previously seen the code.
It covers the data flow from raw CSV to final comparison table, the design
rationale behind each architecture decision, and the anti-leakage rules that
govern every modelling step.

---

## Table of Contents

1. [Project Layout](#1-project-layout)
2. [Configuration — `src/config.py`](#2-configuration)
3. [Utilities — `src/utils.py`](#3-utilities)
4. [Data Loader — `src/data_loader.py`](#4-data-loader)
5. [Feature Engineering — `src/feature_engineering.py`](#5-feature-engineering)
6. [Evaluation — `src/evaluation.py`](#6-evaluation)
7. [Architecture 1: Direct Multi-Step GBM — `src/models/baseline_gbm.py`](#7-architecture-1-direct-multi-step-gbm)
8. [Architecture 2: CNN-LSTM — `src/models/hybrid_lstm.py`](#8-architecture-2-cnn-lstm)
9. [Architecture 3: GNN-LSTM + Stacking — `src/models/gnn_stacking.py`](#9-architecture-3-gnn-lstm--stacking)
10. [Notebooks — Execution Order](#10-notebooks)
11. [Critical Anti-Leakage Rules](#11-critical-anti-leakage-rules)
12. [Tensor Shape Cheat-Sheet](#12-tensor-shape-cheat-sheet)

---

## 1. Project Layout

```
warsaw_aq_forecast/
├── data/
│   ├── raw/FINAL_merged_PM25_1g_all_seasons.csv   ← 52 608 hourly rows, 27 cols
│   └── processed/                                  ← reserved for derived artefacts
├── notebooks/
│   ├── MS_00_EDA.ipynb                    ← per-station EDA, naive baseline
│   ├── MS_00b_cross_station_EDA.ipynb     ← cross-station spatial correlation analysis
│   ├── MS_C1_xgboost.ipynb                ← Architecture C1: XGBoost/HGB (final)
│   ├── MS_C2_cnn_lstm.ipynb               ← Architecture C2: CNN-LSTM (final)
│   ├── MS_C3_gnn_stacking.ipynb           ← Architecture C3: GNN-LSTM + Stacking (final)
│   ├── MS_04_comparison.ipynb             ← cross-model comparison tables & plots
│   ├── MS_05_shap.ipynb                   ← SHAP feature importance analysis
│   └── MS_06_smog_episodes.ipynb          ← Top-5 smog episode case study
├── src/
│   ├── config.py                    ← all constants and paths (single source of truth)
│   ├── utils.py                     ← logging, seeding, directory helpers
│   ├── data_loader.py               ← CSV load, chronological split, leak validation
│   ├── feature_engineering.py       ← all feature construction; builds X and y
│   ├── evaluation.py                ← metrics (MAE/RMSE/R²/MBE/IA), plots, SHAP
│   └── models/
│       ├── baseline_gbm.py          ← Optuna-tuned XGBoost / HGB, direct multi-step
│       ├── hybrid_lstm.py           ← Bidirectional CNN-LSTM, MIMO
│       └── gnn_stacking.py          ← Graph construction, GNN-LSTM, stacking ensemble
├── outputs/
│   ├── models/                      ← serialised model artefacts (.pkl, .pt)
│   ├── figures/                     ← all saved plots (.png, .html)
│   └── results/                     ← metrics CSVs, comparison_table_all.csv
├── ARCHITECTURE.md                  ← this file
├── README.md                        ← concise GitHub-facing README
├── docs/
│   └── README_long.md               ← full project overview and literature review
├── requirements.txt
└── environment.yml
```

### Notebook naming conventions

The project went through three iterative rounds of experiments. The `MS_` series (main study) represents the final polished notebooks using `weather_mode="perfect_forecast"` and the complete 7-station dataset — these are the canonical results.

| Series prefix | Weather mode | Notes |
|---|---|---|
| `00`/`01`/`02`/`03` | `"current"` | Initial exploration |
| `B1`/`B2`/`B3` | `"perfect_forecast"` | Intermediate ablation |
| `C1`/`C2`/`C3` | `"perfect_forecast"` | Full dataset, superseded by MS_ |
| `MS_C1`/`MS_C2`/`MS_C3` | `"perfect_forecast"` | **Final results** |

All paths are resolved relative to the repository root using `pathlib.Path`
defined in `config.py`. Nothing is hardcoded.

---

## 2. Configuration

**File:** `src/config.py`

Everything that might need tuning lives here. Notebooks and modules import
constants from this file — never define magic numbers inline.

| Constant | Value | Purpose |
|---|---|---|
| `DATA_PATH` | `data/raw/FINAL_...csv` | Input CSV path |
| `TARGET` | `"MzWarChrosci"` | Primary PM2.5 station (Targówek) |
| `SPLIT_DATE` | `"2024-01-01"` | Chronological train/test boundary |
| `HORIZONS` | `[1, 2, …, 24]` | All forecast horizons in hours |
| `NEIGHBOR_STATIONS` | 6 station codes | Spatial feature sources |
| `STATION_COORDS` | lat/lon dict | Used for graph edge construction |
| `MET_COLS` | 8 weather column names | Okęcie synoptic features |
| `HYSPLIT_NUMERIC` | 10 trajectory columns | Numeric HYSPLIT features |
| `HYSPLIT_CATEGORICAL` | `["dir_48", "dir_24"]` | Encoded as one-hot |
| `WHO_24H_PM25` | `15` µg/m³ | Plotted as reference line in all figures |
| `RANDOM_SEED` | `42` | Passed to every stochastic operation |
| `SEQ_LEN_LSTM` | `48` | CNN-LSTM lookback window |
| `N_OPTUNA_TRIALS` | `50` | Trials per Optuna study |
| `CV_SPLITS` | `5` | `TimeSeriesSplit` folds everywhere |

---

## 3. Utilities

**File:** `src/utils.py`

Three functions, no dependencies beyond stdlib and numpy:

- **`setup_logging(name)`** — returns a `logging.Logger` that prints
  `HH:MM:SS | module | LEVEL | message` to stdout. Call once at module level:
  `log = setup_logging(__name__)`.
- **`set_seed(seed=42)`** — sets `random`, `numpy`, and `torch` seeds in one call.
- **`ensure_dirs(*paths)`** — `mkdir -p` for any number of `pathlib.Path` objects.

---

## 4. Data Loader

**File:** `src/data_loader.py`

### `load_data(path) → pd.DataFrame`

Reads the CSV, parses the first column as a `DatetimeIndex`, sorts chronologically,
validates that `TARGET` is present, and logs a per-column missing-value summary.

### `train_test_split(df) → (train_df, test_df)`

Slices at `SPLIT_DATE = "2024-01-01"`. No shuffling. Training is 2019–2023;
test is 2024.

```
train: 2019-01-01 → 2023-12-31  (~43 800 hours)
test:  2024-01-01 → 2024-12-31  (~8 760 hours)
```

### `validate_no_future_leak(df, feature_cols, horizon)`

Assert-based guard: raises if any raw (unlagged) station column name appears
in `feature_cols`. Call this after building any feature matrix to double-check
that lag functions were applied correctly.

---

## 5. Feature Engineering

**File:** `src/feature_engineering.py`

This is the most important module for correctness. Every feature is constructed
**at time t only using information available at or before t**.

### Building blocks

| Function | Adds columns | Key leakage guard |
|---|---|---|
| `add_cyclical_time(df)` | hour/month/dow sin+cos, is_weekend | Derived from the index only — no data |
| `add_pm25_lags(df)` | pm25_now, lag_24h, lag_48h, rolling_mean_24h, rolling_max_24h, trend_24h | Rolling window uses `.shift(1)` before `.rolling()` so t=0 is excluded |
| `add_neighbor_lags(df, lag=1)` | `{station}_lag{lag}h` | Default lag=1 ensures no contemporaneous neighbor reading leaks |
| `add_weather_features(df, horizon, mode)` | modifies MET_COLS in-place | See Weather Mode section below |
| `encode_hysplit(df)` | one-hot dir_48/dir_24 dummies + 10 numeric trajectory columns | HYSPLIT back-trajectories are computed from t backward, not forward |

### Weather Mode

All meteorological columns (`MET_COLS`: TEMP, WIND_SPEED, PRESS_SEA, HUMIDITY,
DEW_POINT, RAIN_6H, SUNSHINE, blh) can be used in two ways, controlled by
`weather_mode` in `build_feature_matrix` and `build_sequence_dataset`.

| `weather_mode` | What MET_COLS contain at row t | Represents |
|---|---|---|
| `"current"` (default) | Weather measured at t | Weather when forecast is issued |
| `"perfect_forecast"` | Weather measured at t+h | Perfect NWP forecast for the predicted moment |
| `"perfect_forecast_full"` | Met + HYSPLIT both shifted to t+h | Used in final MS_ notebooks; maximum information upper bound |

```python
# Default — weather at t
X, y = build_feature_matrix(train_df, horizon=24)

# Perfect-forecast — weather at t+24
X, y = build_feature_matrix(train_df, horizon=24, weather_mode="perfect_forecast")
```

**Why `"perfect_forecast"` is not leakage:**
In an operational system the values come from a numerical weather prediction
(NWP) model (ECMWF, GFS), not from future observations.  At h=24 these
forecasts are near-perfect for synoptic-scale variables (BLH, temperature,
pressure).  This is the standard setup in the academic AQ-forecasting
literature (Liao et al., 2023; Qi et al., 2019).  The flag exists to
enable a clean ablation between the two setups.

**Implementation** (`add_weather_features`):
```python
# perfect_forecast: row t gets the value recorded at t+horizon
df[col] = df[col].shift(-horizon)
```
Because the target is also `TARGET.shift(-horizon)`, both the weather features
and the label correspond to the same future moment.  The `dropna()` call in
`build_feature_matrix` removes the last `h` rows that lose their values from
both the shifted met columns and the shifted target — no row-count mismatch.

### `build_feature_matrix(df, horizon, neighbor_lag=1, weather_mode="current") → (X, y)`

The main entry point for tree-based models.

```
1. add_cyclical_time(df)
2. add_pm25_lags(df)                      — always at t
3. add_neighbor_lags(df, neighbor_lag)    — always at t-lag
4. add_weather_features(df, horizon, mode) — t or t+h depending on mode
5. encode_hysplit(df)
6. y = TARGET.shift(-horizon)              ← HONEST TARGET
7. concat(X, y).dropna()
8. assert X has no NaN
```

### `build_sequence_dataset(df, seq_len=48, horizons, scaler_X, fit_scaler, weather_mode="current") → (X_seq, y_seq, feature_names, scaler)`

Builds 3-D sliding-window arrays for the CNN-LSTM.  For MIMO with
`weather_mode="perfect_forecast"`, met columns are shifted by `max(horizons)=24`
so the entire window sees the weather at the furthest predicted moment.

```
Input rows:  N (after dropna)
Window size: seq_len = 48

Sample i:
  X[i] = feature_matrix[i : i+seq_len]          shape (48, n_features)
  y[i] = log1p( target_matrix[i+seq_len-1] )    shape (24,)  — all 24 horizons
```

The target is aligned to the **last row of the window** — row `i+seq_len-1`.
That row's timestamp is `t`. Its target horizon values are `PM25[t+1], …, PM25[t+24]`.

`scaler_X` is always fitted on train data only, then `.transform()`-ed on test.

---

## 6. Evaluation

**File:** `src/evaluation.py`

### `compute_metrics(y_true, y_pred) → dict`

Returns 5 metrics:

| Metric | Formula | Notes |
|---|---|---|
| **MAE** | mean(|y − ŷ|) | Primary metric |
| **RMSE** | sqrt(mean((y − ŷ)²)) | Penalises large errors more |
| **R²** | 1 − SS_res/SS_tot | Fraction of variance explained |
| **MBE** | mean(ŷ − y) | Sign matters: + = over-prediction |
| **IA** | 1 − SS_res / Σ(|ŷ−ȳ| + |y−ȳ|)² | Willmott (1981); bounded 0–1 |

### `plot_horizon_metrics(results_dict, metric)`

`results_dict` has the shape `{model_name: {horizon_int: metrics_dict}}`.
Plots all models on the same axes with different markers.

### `plot_actual_vs_predicted(y_true, y_pred, title, zoom_period)`

Time-series overlay. Draws a horizontal dashed line at `WHO_24H_PM25 = 15 µg/m³`.
Pass `zoom_period=("2024-01-01", "2024-03-31")` to zoom into winter 2024.

### `plot_smog_episodes(y_true, predictions_dict, n_episodes, window_hours)`

Finds the top-N worst PM2.5 episodes using a greedy peak-finding algorithm
(non-overlapping windows), then plots each episode with all model overlaid.

### `shap_analysis(model, X_test, feature_names)`

Wraps `shap.TreeExplainer` — only works with tree-based models (XGBoost, HGB, LightGBM).
For the `TransformedTargetRegressor` wrapper, pass `.regressor_` as `model`.

---

## 7. Architecture 1: Direct Multi-Step GBM

**File:** `src/models/baseline_gbm.py`

### Strategy

**Direct multi-step**: train one completely independent model per horizon `h ∈ {1..24}`.
Each model receives the same feature vector (available at t=0) but predicts a
different future time. No recursive error accumulation.

### Log transform

All models are wrapped in sklearn's `TransformedTargetRegressor`:

```python
TransformedTargetRegressor(regressor=..., func=np.log1p, inverse_func=np.expm1)
```

PM2.5 is right-skewed. Predicting `log1p(PM2.5)` and inverting with `expm1`
produces better-calibrated predictions and protects against negative outputs.

### Optuna hyperparameter search

`build_optuna_objective(X_train, y_train, model_type)` returns a closure that:
1. Samples a hyperparameter set from the search space.
2. Evaluates it with `TimeSeriesSplit(n_splits=5)` — splits are always temporal,
   validation always follows training chronologically.
3. Returns mean CV MAE (lower = better).

The study runs for `N_OPTUNA_TRIALS = 50` trials using the TPE (Tree-structured
Parzen Estimator) sampler seeded at 42.

### `train_direct_multistep(df, model_type, …) → Dict[int, model]`

Iterates over all 24 horizons. After each horizon completes, it logs:
- Horizon number and total elapsed time.
- Estimated remaining time (rolling average per horizon).
- Optuna trial progress (every 10%).

### `predict_all_horizons(models, X) → np.ndarray`

Shape: `(n_test_rows, 24)`. Column `i` is the prediction for horizon `i+1`.

---

## 8. Architecture 2: CNN-LSTM

**File:** `src/models/hybrid_lstm.py`

### Strategy: MIMO (Multi-Input Multi-Output)

One model, one forward pass, 24 outputs. More stable gradient flow than
training 24 separate deep learning models.

### Model: `CnnLstmModel` (built inside `build_cnn_lstm_model`)

```
Input:  (batch, seq_len=48, n_features)

Step 1 — Local pattern extraction (CNN)
  permute → (batch, n_features, seq_len)    [channels-first for Conv1d]
  Conv1d(n_features → 64, k=3, pad=1) + BatchNorm1d + ReLU
  Conv1d(64 → 128, k=3, pad=1) + BatchNorm1d + ReLU
  MaxPool1d(2) + Dropout(0.2)
  permute → (batch, seq_len//2, 128)        [back to time-major]

Step 2 — Temporal modelling (BiLSTM)
  BiLSTM(128 → 256 output) + Dropout(0.3)
  BiLSTM(256 → 128 output)
  x[:, -1, :]    ← take the last timestep's hidden state

Step 3 — Prediction head
  Linear(128 → 64) + ReLU
  Linear(64 → 24)  ← one logit per horizon

Output: (batch, 24)  — log1p-transformed PM2.5
```

**Why CNN before LSTM?**
CNNs efficiently detect short-range patterns (e.g. the typical 6–8 hour
diurnal pollution ramp-up). By halving the sequence length via MaxPool,
the LSTM handles a cleaner, more compressed temporal representation.

**Why Bidirectional LSTM?**
The lookback window is fixed and fully available at prediction time — there
is no causal constraint within the window itself. BiLSTM can use context
from both directions within the 48-hour history.

### Training: `train_cnn_lstm`

- **Loss**: `HuberLoss(delta=1.0)` — behaves like MAE for errors >1 log-unit
  and like MSE for smaller errors. Robust to the rare extreme PM2.5 spikes.
- **Optimiser**: Adam, `lr=1e-3`, no weight decay.
- **LR schedule**: `ReduceLROnPlateau(patience=5, factor=0.5)` — halves the
  learning rate if validation loss stagnates for 5 epochs.
- **Early stopping**: restores the best weights when val loss stops improving
  for `patience=15` consecutive epochs.
- **Progress**: every epoch prints `train_loss  val_loss  best_loss` with a
  `*` marker when a new best is reached.

### Sequence alignment

```
Train window i: features[i : i+48]  →  target[i+47]
                 ─────────────────       ──────────────
                 48h of history         all 24 horizons
                 up to and incl. t      t+1 through t+24
```

---

## 9. Architecture 3: GNN-LSTM + Stacking

**File:** `src/models/gnn_stacking.py`

### 9a. Graph Construction

Warsaw has 7 PM2.5 monitoring stations. We model them as nodes in a graph.

#### Static adjacency matrix: `build_adjacency_matrix(sigma_km=50)`

```python
W[i, j] = exp( -d(i,j)^2 / (2 * sigma_km^2) )
```

where `d(i, j)` is the Haversine (great-circle) distance in km.
Diagonal is zero (no self-loops). The matrix is **row-normalised** so each
node's incoming edge weights sum to 1 — equivalent to a mean neighbourhood
aggregation.

#### Dynamic adjacency: `dynamic_adjacency(static_adj, wind_dir_val, alpha=0.5)`

HYSPLIT back-trajectories tell us where air came from. If the wind is from
the North, stations north of the target are "upwind" and should receive higher
edge weight:

```python
bearing_to_target = bearing(neighbour_coord, target_coord)
angle_diff = |bearing_to_target - wind_deg|
dynamic_weight[i, j] = exp( -angle_diff^2 / (2 * sigma_angle^2) )

final = alpha * static + (1 - alpha) * dynamic   [then row-normalise]
```

`alpha=0.5` blends both components. The GNN-LSTM currently uses the static
matrix; dynamic adjacency can be applied per-timestep in advanced extensions.

### 9b. GNN Sequence Dataset: `build_gnn_sequence_dataset`

Each node (station) has 10 features per timestep:

```
[pm25_lag1, pm25_lag24, TEMP, WIND_SPEED, blh, HUMIDITY,
 hour_sin, hour_cos, month_sin, month_cos]
```

Shared meteorological features (TEMP, WIND_SPEED, blh, HUMIDITY) are the same
for all nodes — only the PM2.5 lags are station-specific.

```
X shape:  (n_samples, seq_len=24, n_nodes=7, n_node_features=10)
y shape:  (n_samples, 24)   — log1p PM2.5 at the target node, h=1..24
```

### 9c. GNN-LSTM Model: `GnnLstmModel` (built inside `build_gnn_lstm_model`)

```
Input: (batch, seq_len=24, n_nodes=7, n_node_features=10)

For each timestep t ∈ {0..23}:
  x_t: (batch, 7, 10)

  Graph convolution:
    h_i = ReLU( W_self @ x_i  +  A @ (W_neigh @ X) )
    BatchNorm over (batch × n_nodes, gnn_hidden=64)

  Select target node (node 0):
    h_target_t: (batch, 64)

Stack over time → target_seq: (batch, 24, 64)

BiLSTM(64 → 256 output):
  lstm_out: (batch, 24, 256)
  Dropout(0.3)

Additive attention:
  scores = tanh( W_attn @ lstm_out )        shape (batch, 24, 1)
  weights = softmax(scores, dim=time)
  context = sum(weights * lstm_out, dim=1)  shape (batch, 256)

Prediction head:
  Linear(256 → 64) + ReLU
  Linear(64 → 24)

Output: (batch, 24)  — log1p PM2.5
```

**Why graph convolution?**
Pollution at Targówek doesn't depend only on local emissions — it depends on
what surrounding stations are measuring right now, weighted by distance and
wind direction. One graph conv layer aggregates all 7 stations into a spatial
embedding before the temporal LSTM processes the sequence.

**Why attention over LSTM outputs?**
For a 24-hour lookback, the most predictive timestep is not always the last one.
Attention lets the model learn that 6–12 hours ago might carry more signal for
certain horizons than the most recent observation.

### 9d. Stacking Ensemble: `train_stacking_ensemble`

**Level 0 — Base models (one per horizon):**

| Model | Wrapper | Notes |
|---|---|---|
| XGBoost | `TransformedTargetRegressor(log1p)` | Histogram trees |
| LightGBM | `TransformedTargetRegressor(log1p)` | Leaf-wise growth |
| CatBoost | Native (MAE loss) | No log-wrap needed — robust natively |
| HistGradientBoosting | `TransformedTargetRegressor(log1p)` | sklearn's native GBM |
| GNN-LSTM | Output fed directly | Pre-trained, single inference pass |

**Level 1 — Meta-learner:**

LightGBM trained on `[oof_preds_1..5, hour_sin, hour_cos, month_sin, month_cos, blh, pm25_now]`.

**Out-of-fold (OOF) generation with `TimeSeriesSplit`:**

```
Fold 1: train=[0..K]      val=[K+1..2K]
Fold 2: train=[0..2K]     val=[2K+1..3K]
...
Fold 5: train=[0..4K]     val=[4K+1..N]
```

Each base model is **re-instantiated** for each fold so no weights or state are
shared between folds. This is the critical correctness property — a single
shared instance fitted on fold 1 would carry stale gradients into fold 2.

The meta-learner sees only the OOF predictions — values the base models
produced on data they were **not** trained on — which prevents the meta-learner
from simply memorising the strongest base model.

---

## 10. Notebooks

Run the `MS_` series in order. Each notebook imports from `src/` — add the repo
root to `sys.path` with `sys.path.insert(0, '..')` at the top of each notebook.

| Notebook | Reads | Writes |
|---|---|---|
| `MS_00_EDA.ipynb` | raw CSV | `outputs/{station}/figures/` (gitignored), `outputs/results/naive_baseline.csv` |
| `MS_00b_cross_station_EDA.ipynb` | raw CSV | `outputs/{station}/figures/` (gitignored) |
| `MS_C1_xgboost.ipynb` | raw CSV | `outputs/{station}/models/xgb_pfxf_h*.pkl`, `hgb_pfxf_h*.pkl` (gitignored), `C1_metrics.csv` |
| `MS_C2_cnn_lstm.ipynb` | raw CSV | `outputs/{station}/models/cnn_lstm_pfxf_model.pt` (gitignored), `C2_metrics.csv` |
| `MS_C3_gnn_stacking.ipynb` | raw CSV + C1 models | `gnn_lstm_pfxf_model.pt`, `meta_learner_pfxf_h*.pkl` (gitignored), `C3_metrics.csv`, `comparison_table_all.csv` |
| `MS_04_comparison.ipynb` | `C1/C2/C3_metrics.csv` per station | cross-station comparison figures (gitignored) |
| `MS_05_shap.ipynb` | C1 models | SHAP summary and dependence plots (gitignored) |
| `MS_06_smog_episodes.ipynb` | C1 + C3 models, raw CSV | episode overlay figures (gitignored) |

> **Note:** `outputs/**/figures/` and `outputs/**/models/` are gitignored — figures and model weights are not tracked in the repository. Only results CSVs under `outputs/**/results/` are committed.

---

## 11. Critical Anti-Leakage Rules

These rules are invariants of the entire codebase. Violating any of them
invalidates the reported metrics.

### Rule 1: Target construction — always shift the target, never shift features

```python
# CORRECT
y = df[TARGET].shift(-h)   # row t predicts TARGET at t+h

# WRONG — shifts the entire feature matrix forward, leaking future data into X
X_future = df.shift(-h)
```

### Rule 2: Neighbor station lag must be ≥ 1

At prediction time `t`, the value of a neighboring station at time `t` is
unknown in an operational setting (the network may not have reported yet).
Using `lag=1` means we use the neighbour's reading from `t-1`, which is always
available.

```python
# CORRECT
neighbor_lag1 = df["MzOtwoBrzozo"].shift(1)

# WRONG — contemporaneous neighbor value leaks into features
neighbor_now = df["MzOtwoBrzozo"]
```

### Rule 3: Rolling statistics exclude the current observation

```python
# CORRECT — shift by 1 before rolling so t=0 is excluded from the window
rolled = series.shift(1).rolling(window=24, min_periods=12).mean()

# WRONG — rolling window includes t=0 (the current value)
rolled = series.rolling(window=24).mean()
```

### Rule 4: Cross-validation is always TimeSeriesSplit

```python
# CORRECT
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)

# WRONG — KFold shuffles, allowing validation folds earlier than training folds
from sklearn.model_selection import KFold
```

### Rule 5: Scalers are fitted on training data only

```python
# CORRECT
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)   # no fit — uses train statistics

# WRONG — leaks test distribution into the scaler
scaler.fit(X_all)
```

### Rule 6: Test data is never seen during hyperparameter tuning

`train_single_horizon` receives only `train_df`. The test split is created in
the notebook and passed only to the evaluation step.

### Rule 7: Stacking OOF — fresh model per fold

```python
# CORRECT — re-instantiate so folds are independent
for fold, (tr, val) in enumerate(tscv.split(X)):
    model = factory()        # fresh instance
    model.fit(X[tr], y[tr])
    oof[val] = model.predict(X[val])

# WRONG — same object accumulates state across folds
model = XGBRegressor(...)
for fold, (tr, val) in enumerate(tscv.split(X)):
    model.fit(X[tr], y[tr])  # overwrites previous fit
    oof[val] = model.predict(X[val])
```

### Rule 8: `"perfect_forecast"` weather mode is not leakage

Shifting MET_COLS by `-h` places the **future measured weather** at row `t`.
This is valid because in an operational system those values come from an NWP
model, not from future observations.  The rule is: never use future **PM2.5**
as a feature (that is what you are predicting).  Future weather from a forecast
model is a legitimate input.

```python
# CORRECT — simulates NWP output; future PM2.5 is never used as a feature
X, y = build_feature_matrix(df, horizon=24, weather_mode="perfect_forecast")

# WRONG — would use future PM2.5 directly (TARGET at t+24 as a feature)
df["pm25_future"] = df[TARGET].shift(-24)   # never do this
```

---

## 12. Tensor Shape Cheat-Sheet

| Variable | Shape | Where created |
|---|---|---|
| `X` (tabular) | `(n_rows, n_features)` | `build_feature_matrix` |
| `y` (tabular) | `(n_rows,)` | `build_feature_matrix` |
| `X_seq` (CNN-LSTM) | `(n_samples, 48, n_features)` | `build_sequence_dataset` |
| `y_seq` (CNN-LSTM) | `(n_samples, 24)` — log1p | `build_sequence_dataset` |
| `X_gnn` | `(n_samples, 24, 7, 10)` | `build_gnn_sequence_dataset` |
| `y_gnn` | `(n_samples, 24)` — log1p | `build_gnn_sequence_dataset` |
| `A_static` | `(7, 7)` — row-normalised | `build_adjacency_matrix` |
| CNN-LSTM output | `(batch, 24)` — log1p | `CnnLstmModel.forward` |
| GNN-LSTM output | `(batch, 24)` — log1p | `GnnLstmModel.forward` |
| `predict_cnn_lstm` return | `(n_samples, 24)` — µg/m³ | expm1 applied inside |
| `predict_gnn_lstm` return | `(n_samples, 24)` — µg/m³ | expm1 applied inside |
| `oof_matrix` | `(n_train, n_base_models)` | `train_stacking_ensemble` |
| `meta_X` | `(n_train, n_base + n_meta_feats)` | `train_stacking_ensemble` |
| `predict_all_horizons` | `(n_test, 24)` — µg/m³ | `baseline_gbm.py` |

**n_features** (tabular) ≈ 50, depending on HYSPLIT dummy columns present in the data.
**n_node_features** (GNN) = 10 (fixed: 2 PM2.5 lags + 8 shared met features).
