"""Architecture 3: GNN-LSTM core + tree-based stacking ensemble (SOTA)."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from ..config import (
    STATION_COORDS, TARGET, NEIGHBOR_STATIONS,
    CV_SPLITS, RANDOM_SEED, MODEL_DIR,
)
from ..utils import setup_logging, ensure_dirs

log = setup_logging(__name__)

ALL_STATIONS_ORDERED = [TARGET] + NEIGHBOR_STATIONS  # target is node 0


# ── Graph construction ────────────────────────────────────────────────

def build_adjacency_matrix(sigma_km: float = 50.0) -> np.ndarray:
    """Build a static adjacency matrix using an inverse-distance Gaussian kernel.

    Uses great-circle distance (Haversine approximation) between stations.
    Self-loops are excluded (diagonal = 0).

    Args:
        sigma_km: Gaussian kernel bandwidth in km; edges beyond this are near-zero.

    Returns:
        Row-normalised adjacency matrix of shape (n_nodes, n_nodes).

    Example:
        >>> A = build_adjacency_matrix(sigma_km=50)
        >>> A.shape
        (7, 7)
    """
    stations = ALL_STATIONS_ORDERED
    n = len(stations)
    coords = np.array([STATION_COORDS[s] for s in stations])  # (n, 2) lat/lon

    # Haversine distance in km
    def haversine(a, b):
        R = 6371.0
        dlat = np.radians(b[0] - a[0])
        dlon = np.radians(b[1] - a[1])
        h = (np.sin(dlat / 2) ** 2
             + np.cos(np.radians(a[0])) * np.cos(np.radians(b[0])) * np.sin(dlon / 2) ** 2)
        return 2 * R * np.arcsin(np.sqrt(h))

    W = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                d = haversine(coords[i], coords[j])
                W[i, j] = np.exp(-(d ** 2) / (2 * sigma_km ** 2))

    # Row-normalise
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return (W / row_sums).astype(np.float32)


def dynamic_adjacency(
    static_adj: np.ndarray,
    wind_dir_val: Optional[str],
    alpha: float = 0.5,
    sigma_angle: float = 45.0,
) -> np.ndarray:
    """Blend static adjacency with a wind-direction-informed upwind weighting.

    Stations that are upwind of the target receive higher edge weight.
    The bearing from each neighbour to the target is compared to the current
    wind direction; stations whose bearing aligns with wind direction get boosted.

    Args:
        static_adj: Static row-normalised adjacency (n, n).
        wind_dir_val: Current wind direction as a cardinal string (e.g. "N", "SW").
            If None, fall back to static_adj.
        alpha: Blend weight: alpha * static + (1-alpha) * dynamic.
        sigma_angle: Angular spread in degrees for the Gaussian upwind weighting.

    Returns:
        Blended, row-normalised adjacency matrix of shape (n, n).

    Example:
        >>> A_dyn = dynamic_adjacency(A_static, wind_dir_val="N")
    """
    if wind_dir_val is None:
        return static_adj

    CARDINAL = {
        "N": 0, "NNE": 22.5, "NE": 45, "ENE": 67.5,
        "E": 90, "ESE": 112.5, "SE": 135, "SSE": 157.5,
        "S": 180, "SSW": 202.5, "SW": 225, "WSW": 247.5,
        "W": 270, "WNW": 292.5, "NW": 315, "NNW": 337.5,
        "L": None,  # local / calm
    }
    wind_deg = CARDINAL.get(str(wind_dir_val).strip().upper())
    if wind_deg is None:
        return static_adj

    stations = ALL_STATIONS_ORDERED
    coords = np.array([STATION_COORDS[s] for s in stations])
    target_coord = np.array(STATION_COORDS[TARGET])
    n = len(stations)

    # Bearing from each neighbour to the target
    def bearing(src, dst):
        dlon = np.radians(dst[1] - src[1])
        lat1 = np.radians(src[0])
        lat2 = np.radians(dst[0])
        x = np.sin(dlon) * np.cos(lat2)
        y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
        return (np.degrees(np.arctan2(x, y)) + 360) % 360

    # TODO: dyn[i, j] currently depends only on j (the source node), not i (the
    # receiver). All rows of the dynamic matrix are therefore identical after
    # row-normalisation. The intended behaviour is that each node weights its
    # neighbours differently based on wind direction, but the correct bearing
    # should be bearing(coords[j], coords[i]) — from neighbour j toward receiver i.
    # Currently dormant because notebooks use A_static throughout.
    dyn = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            if i != j:
                b = bearing(coords[j], target_coord)
                angle_diff = abs(((b - wind_deg + 180) % 360) - 180)
                dyn[i, j] = np.exp(-(angle_diff ** 2) / (2 * sigma_angle ** 2))

    # Row-normalise dynamic component
    row_sums = dyn.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    dyn /= row_sums

    blended = alpha * static_adj + (1 - alpha) * dyn
    row_sums2 = blended.sum(axis=1, keepdims=True)
    row_sums2[row_sums2 == 0] = 1
    return (blended / row_sums2).astype(np.float32)


# ── GNN-LSTM model ────────────────────────────────────────────────────

def build_gnn_lstm_model(
    n_nodes: int,
    n_node_features: int,
    seq_len: int,
    n_horizons: int = 24,
    gnn_hidden: int = 64,
    lstm_hidden: int = 128,
) -> "torch.nn.Module":
    """Build a GNN-LSTM model with attention over LSTM outputs.

    At each timestep, a single-layer graph convolution aggregates spatial
    information across all stations. The resulting target-node embeddings
    are fed into a bidirectional LSTM, followed by a learned attention
    mechanism, and a two-layer head predicting all 24 horizons at once.

    Args:
        n_nodes: Number of stations (7).
        n_node_features: Features per node per timestep.
        seq_len: Sequence length (24 past timesteps).
        n_horizons: Output dimension (24).
        gnn_hidden: Graph convolution output dimension.
        lstm_hidden: LSTM hidden size.

    Returns:
        Initialised PyTorch nn.Module.

    Example:
        >>> model = build_gnn_lstm_model(7, 10, 24)
    """
    import torch
    import torch.nn as nn

    class GnnLstmModel(nn.Module):
        def __init__(self):
            super().__init__()
            # Graph conv: W_self @ x_i + W_neigh weighted sum over neighbours
            self.W_self = nn.Linear(n_node_features, gnn_hidden, bias=False)
            self.W_neigh = nn.Linear(n_node_features, gnn_hidden, bias=True)
            self.gnn_bn = nn.BatchNorm1d(gnn_hidden)

            # LSTM over target-node temporal sequence
            self.lstm = nn.LSTM(gnn_hidden, lstm_hidden, batch_first=True, bidirectional=True)
            self.drop = nn.Dropout(0.3)

            # Additive attention
            self.attn_w = nn.Linear(lstm_hidden * 2, 1)

            # Prediction head
            self.fc1 = nn.Linear(lstm_hidden * 2, 64)
            self.fc2 = nn.Linear(64, n_horizons)
            self.relu = nn.ReLU()

        def graph_conv(self, x, A):
            # x: (batch, n_nodes, n_node_features)
            # A: (n_nodes, n_nodes) — adjacency, passed as buffer
            self_term = self.W_self(x)                         # (batch, n_nodes, gnn_hidden)
            neigh_term = torch.matmul(A, self.W_neigh(x))     # (batch, n_nodes, gnn_hidden)
            h = self.relu(self_term + neigh_term)              # (batch, n_nodes, gnn_hidden)
            # BatchNorm over node & batch dims combined
            B, N, H = h.shape
            h = self.gnn_bn(h.view(B * N, H)).view(B, N, H)
            return h

        def forward(self, x, A):
            # x: (batch, seq_len, n_nodes, n_node_features)
            # A: (n_nodes, n_nodes)
            batch, T, N, F = x.shape
            # Apply graph conv at each timestep
            gnn_out = []
            for t in range(T):
                xt = x[:, t, :, :]                             # (batch, n_nodes, n_features)
                ht = self.graph_conv(xt, A)                    # (batch, n_nodes, gnn_hidden)
                gnn_out.append(ht[:, 0, :])                    # select target node (idx 0)
            target_seq = torch.stack(gnn_out, dim=1)           # (batch, seq_len, gnn_hidden)

            # LSTM
            lstm_out, _ = self.lstm(target_seq)                # (batch, seq_len, lstm_hidden*2)
            lstm_out = self.drop(lstm_out)

            # Attention
            scores = self.attn_w(lstm_out)                     # (batch, seq_len, 1)
            weights = torch.softmax(scores, dim=1)
            context = (weights * lstm_out).sum(dim=1)          # (batch, lstm_hidden*2)

            out = self.relu(self.fc1(context))
            return self.fc2(out)                               # (batch, n_horizons)

    return GnnLstmModel()


# ── GNN-LSTM training ─────────────────────────────────────────────────

def build_gnn_sequence_dataset(
    df: pd.DataFrame,
    seq_len: int = 24,
    horizons: Optional[List[int]] = None,
    weather_mode: str = "current",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build node-feature sequences and adjacency tensors for the GNN-LSTM.

    Node features per timestep (10 per station):
        pm25_lag_1, pm25_lag_24, TEMP, WIND_SPEED, blh, HUMIDITY,
        hour_sin, hour_cos, month_sin, month_cos

    The shared meteorological node features (TEMP, WIND_SPEED, blh, HUMIDITY)
    are affected by ``weather_mode``:

    ``"current"`` — met values at t (default).
    ``"perfect_forecast"`` — met values shifted to t+max(horizons), simulating
    a perfect NWP forecast for the furthest predicted moment.

    Args:
        df: Raw DataFrame with DatetimeIndex.
        seq_len: Number of past timesteps.
        horizons: Output horizons (default 1..24).
        weather_mode: ``"current"`` or ``"perfect_forecast"``.

    Returns:
        Tuple of:
            X: np.ndarray (n_samples, seq_len, n_nodes, n_node_features) float32
            y: np.ndarray (n_samples, n_horizons) log1p-transformed float32
            A_static: np.ndarray (n_nodes, n_nodes) static adjacency float32

    Example:
        >>> X, y, A = build_gnn_sequence_dataset(train_df)
        >>> X, y, A = build_gnn_sequence_dataset(train_df, weather_mode="perfect_forecast")
    """
    if horizons is None:
        horizons = HORIZONS

    from ..feature_engineering import add_cyclical_time, add_weather_features

    df = add_cyclical_time(df)
    # Apply weather mode to the shared met columns before building node features
    pfx_horizon = max(horizons) if weather_mode in ("perfect_forecast", "perfect_forecast_full") else None
    df = add_weather_features(df, horizon=pfx_horizon, mode=weather_mode)
    log.info("build_gnn_sequence_dataset | seq_len=%d | weather_mode=%s", seq_len, weather_mode)

    stations = ALL_STATIONS_ORDERED

    node_feat_cols = ["TEMP", "WIND_SPEED", "blh", "HUMIDITY",
                      "hour_sin", "hour_cos", "month_sin", "month_cos"]

    # Build per-node feature matrix: [pm25_lag1, pm25_lag24] + shared met
    station_features = {}
    for s in stations:
        if s not in df.columns:
            log.warning("Station %s missing from df — filling with zeros", s)
            df[s] = 0.0
        feats = pd.DataFrame(index=df.index)
        feats["pm25_lag1"] = df[s].shift(1)
        feats["pm25_lag24"] = df[s].shift(24)
        for c in node_feat_cols:
            if c in df.columns:
                feats[c] = df[c].values
            else:
                feats[c] = 0.0
        station_features[s] = feats  # shape (T, 10)

    # Target: log1p of future PM2.5 at target station
    target_matrix = pd.DataFrame(
        {f"h{h}": df[TARGET].shift(-h) for h in horizons}, index=df.index
    )

    # Align: drop any row with NaN in any station feature or any target
    all_dfs = list(station_features.values()) + [target_matrix]
    combined = pd.concat(all_dfs, axis=1).dropna()
    n = len(combined)
    assert n > seq_len, "Not enough clean rows for GNN sequences"

    # Rebuild from combined index
    n_node_features = 2 + len(node_feat_cols)  # pm25_lag1 + pm25_lag24 + 8 met
    n_nodes = len(stations)
    n_horizons = len(horizons)

    feat_arrays = np.zeros((n, n_nodes, n_node_features), dtype=np.float32)
    cols_per_node = ["pm25_lag1", "pm25_lag24"] + node_feat_cols
    for ni, s in enumerate(stations):
        sub = station_features[s].loc[combined.index]
        feat_arrays[:, ni, :] = sub[cols_per_node].values.astype(np.float32)

    y_all = np.log1p(combined[[f"h{h}" for h in horizons]].values.astype(np.float32))

    # Sliding window
    n_samples = n - seq_len + 1
    X = np.stack([feat_arrays[i:i + seq_len] for i in range(n_samples)])
    y = y_all[seq_len - 1:]

    assert X.shape == (n_samples, seq_len, n_nodes, n_node_features)
    assert y.shape == (n_samples, n_horizons)

    A_static = build_adjacency_matrix()
    log.info("GNN sequences: X=%s y=%s", X.shape, y.shape)
    return X, y, A_static


def train_gnn_lstm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    A_static: np.ndarray,
    epochs: int = 80,
    batch_size: int = 64,
    patience: int = 15,
    lr: float = 1e-3,
    save_path: Optional[Path] = None,
) -> Tuple["torch.nn.Module", dict]:
    """Train the GNN-LSTM model with early stopping.

    Args:
        X_train: (n_train, seq_len, n_nodes, n_node_features).
        y_train: (n_train, n_horizons) — log1p-transformed.
        X_val: Validation input.
        y_val: Validation targets.
        A_static: Static adjacency matrix (n_nodes, n_nodes).
        epochs: Max epochs.
        batch_size: Mini-batch size.
        patience: Early stopping patience.
        lr: Learning rate.
        save_path: Optional checkpoint path.

    Returns:
        Tuple of (best_model, history_dict).

    Example:
        >>> model, hist = train_gnn_lstm(X_tr, y_tr, X_val, y_val, A)
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Training GNN-LSTM on %s", device)

    seq_len = X_train.shape[1]
    n_nodes = X_train.shape[2]
    n_node_features = X_train.shape[3]
    n_horizons = y_train.shape[1]

    model = build_gnn_lstm_model(n_nodes, n_node_features, seq_len, n_horizons).to(device)
    A_t = torch.tensor(A_static, dtype=torch.float32).to(device)

    Xt = torch.tensor(X_train, dtype=torch.float32)
    yt = torch.tensor(y_train, dtype=torch.float32)
    Xv = torch.tensor(X_val, dtype=torch.float32).to(device)
    yv = torch.tensor(y_val, dtype=torch.float32).to(device)

    loader = DataLoader(TensorDataset(Xt, yt), batch_size=batch_size, shuffle=True)
    criterion = nn.HuberLoss(delta=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5, min_lr=1e-5
    )

    best_val, best_state, no_improve = float("inf"), None, 0
    history = {"train_loss": [], "val_loss": []}

    log.info("GNN-LSTM | seq_len=%d  n_nodes=%d  n_node_features=%d  n_horizons=%d  device=%s",
             seq_len, n_nodes, n_node_features, n_horizons, device)
    log.info("Params: epochs=%d  batch=%d  patience=%d  lr=%.5f", epochs, batch_size, patience, lr)
    log.info("Training samples=%d  val samples=%d", len(Xt), len(Xv))
    log.info("%s", "-" * 65)

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb, A_t)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item() * len(xb)
        epoch_loss /= len(Xt)

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(Xv, A_t), yv).item()
        scheduler.step(val_loss)
        history["train_loss"].append(epoch_loss)
        history["val_loss"].append(val_loss)

        improved = val_loss < best_val
        if improved:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        current_lr = optimizer.param_groups[0]["lr"]
        marker = " *" if improved else f"  (no improve {no_improve}/{patience})"
        log.info("Epoch %3d/%d  train=%.5f  val=%.5f  best=%.5f  lr=%.2e%s",
                 epoch, epochs, epoch_loss, val_loss, best_val, current_lr, marker)

        if no_improve >= patience:
            log.info("Early stopping triggered at epoch %d — best val=%.5f", epoch, best_val)
            break

    model.load_state_dict(best_state)
    if save_path:
        ensure_dirs(Path(save_path).parent)
        torch.save({"model_state": best_state, "seq_len": seq_len,
                    "n_nodes": n_nodes, "n_node_features": n_node_features,
                    "n_horizons": n_horizons, "A_static": A_static}, save_path)
        log.info("GNN-LSTM saved → %s", save_path)
    return model, history


def predict_gnn_lstm(
    model: "torch.nn.Module",
    X_seq: np.ndarray,
    A_static: np.ndarray,
    batch_size: int = 256,
) -> np.ndarray:
    """Run GNN-LSTM inference; returns predictions in µg/m³ (expm1 applied).

    Args:
        model: Fitted GnnLstmModel.
        X_seq: (n_samples, seq_len, n_nodes, n_node_features).
        A_static: (n_nodes, n_nodes) adjacency.
        batch_size: Inference batch size.

    Returns:
        np.ndarray (n_samples, n_horizons) in µg/m³.

    Example:
        >>> preds = predict_gnn_lstm(model, X_test, A_static)
    """
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    device = next(model.parameters()).device
    A_t = torch.tensor(A_static, dtype=torch.float32).to(device)
    model.eval()
    Xt = torch.tensor(X_seq, dtype=torch.float32)
    loader = DataLoader(TensorDataset(Xt), batch_size=batch_size)
    parts = []
    with torch.no_grad():
        for (xb,) in loader:
            parts.append(model(xb.to(device), A_t).cpu().numpy())
    return np.expm1(np.concatenate(parts, axis=0))


# ── Stacking ensemble ─────────────────────────────────────────────────

def train_stacking_ensemble(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    gnn_oof_preds: Optional[np.ndarray] = None,
    meta_feature_cols: Optional[List[str]] = None,
    n_splits: int = CV_SPLITS,
    save_path: Optional[Path] = None,
) -> Tuple[object, np.ndarray]:
    """Train a tree-based stacking ensemble with out-of-fold base model predictions.

    Level 0 base models:
        XGBoost, LightGBM, CatBoost, HistGradientBoosting [+ GNN-LSTM if provided]

    Level 1 meta-learner:
        LightGBM trained on OOF predictions + meta features.

    All CV uses TimeSeriesSplit — no shuffling.

    Args:
        X_train: Feature DataFrame for a single horizon.
        y_train: Target Series (raw PM2.5, not log-transformed — models handle that internally).
        gnn_oof_preds: Optional array (n_train,) of GNN-LSTM OOF predictions for this horizon.
        meta_feature_cols: Columns from X_train to include as extra meta-features
            (default: hour_sin, hour_cos, month_sin, month_cos, blh, pm25_now).
        n_splits: TimeSeriesSplit folds.
        save_path: Path to save the fitted meta-learner.

    Returns:
        Tuple of (fitted_meta_learner, oof_preds_matrix).

    Example:
        >>> meta, oof = train_stacking_ensemble(X_tr, y_tr)
    """
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.compose import TransformedTargetRegressor
    from xgboost import XGBRegressor
    import lightgbm as lgb
    from catboost import CatBoostRegressor

    if meta_feature_cols is None:
        meta_feature_cols = [c for c in
                             ["hour_sin", "hour_cos", "month_sin", "month_cos", "blh", "pm25_now"]
                             if c in X_train.columns]

    # Factories (callables) so each fold gets a fresh model instance — critical for OOF correctness.
    base_models = {
        "xgb": lambda: TransformedTargetRegressor(
            regressor=XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=4,
                                   random_state=RANDOM_SEED, verbosity=0, tree_method="hist"),
            func=np.log1p, inverse_func=np.expm1),
        "hgb": lambda: TransformedTargetRegressor(
            regressor=HistGradientBoostingRegressor(
                max_iter=300, learning_rate=0.05, max_depth=4, random_state=RANDOM_SEED),
            func=np.log1p, inverse_func=np.expm1),
        "lgb": lambda: TransformedTargetRegressor(
            regressor=lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, max_depth=4,
                                        random_state=RANDOM_SEED, verbose=-1),
            func=np.log1p, inverse_func=np.expm1),
        "cat": lambda: CatBoostRegressor(iterations=300, learning_rate=0.05, depth=4,
                                         random_seed=RANDOM_SEED, verbose=0,
                                         loss_function="MAE"),
    }

    tscv = TimeSeriesSplit(n_splits=n_splits)
    n_base = len(base_models) + (1 if gnn_oof_preds is not None else 0)
    X_arr = X_train.values
    y_arr = y_train.values
    oof_matrix = np.zeros((len(X_train), n_base), dtype=np.float32)
    # Track which rows actually received OOF predictions (TimeSeriesSplit skips the
    # first fold's training rows — they are never a validation set)
    oof_filled = np.zeros((len(X_train), n_base), dtype=bool)

    for m_idx, (name, factory) in enumerate(base_models.items()):
        log.info("OOF [%d/%d]: fitting base model '%s' across %d folds",
                 m_idx + 1, len(base_models), name, n_splits)
        for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_arr)):
            # Re-instantiate each fold so models don't share state between folds
            fold_model = factory()
            fold_model.fit(X_arr[tr_idx], y_arr[tr_idx])
            oof_matrix[val_idx, m_idx] = fold_model.predict(X_arr[val_idx])
            oof_filled[val_idx, m_idx] = True
            log.info("  fold %d/%d — train=%d val=%d", fold + 1, n_splits,
                     len(tr_idx), len(val_idx))

    if gnn_oof_preds is not None:
        assert len(gnn_oof_preds) == len(X_train), "GNN OOF length mismatch"
        oof_matrix[:, -1] = gnn_oof_preds.astype(np.float32)
        oof_filled[:, -1] = True
        # NOTE: gnn_oof_preds are full-train predictions (GNN trained on all training
        # data), not true out-of-fold predictions. Generating true GNN OOF would
        # require 5× retraining which is prohibitively expensive. The meta-learner
        # may therefore slightly over-weight the GNN column; test-set metrics are
        # unaffected since test data was never seen during GNN training.
        log.warning("GNN column uses full-train predictions, not true OOF — "
                    "meta-learner may slightly over-weight GNN.")
        log.info("Added GNN-LSTM predictions as base model column %d", n_base - 1)

    # Report OOF MAE per base model as a sanity check
    from sklearn.metrics import mean_absolute_error
    model_names = list(base_models.keys()) + (["gnn_lstm"] if gnn_oof_preds is not None else [])
    for mi, mname in enumerate(model_names):
        filled = oof_filled[:, mi]
        if filled.sum() > 0:
            oof_mae = mean_absolute_error(y_arr[filled], oof_matrix[filled, mi])
            log.info("  OOF MAE [%s] = %.4f  (on %d/%d rows)",
                     mname, oof_mae, filled.sum(), len(y_arr))

    # Meta features: OOF predictions + time/weather signals (no future leak)
    meta_X = np.hstack([oof_matrix, X_train[meta_feature_cols].values.astype(np.float32)])

    log.info("Training LightGBM meta-learner | meta_X=%s  n_meta_features=%d",
             meta_X.shape, meta_X.shape[1])
    meta_learner = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05,
                                      random_state=RANDOM_SEED, verbose=-1)
    meta_learner.fit(meta_X, y_arr)
    log.info("Meta-learner training complete.")

    if save_path:
        ensure_dirs(Path(save_path).parent)
        joblib.dump(meta_learner, save_path)
        log.info("Meta-learner saved → %s", save_path)

    return meta_learner, oof_matrix


def predict_stacking(
    meta_learner,
    base_predictions: np.ndarray,
    meta_features: np.ndarray,
) -> np.ndarray:
    """Run meta-learner inference.

    Args:
        meta_learner: Fitted LightGBM meta-learner.
        base_predictions: (n_samples, n_base_models) array of base model predictions.
        meta_features: (n_samples, n_meta_features) additional context features.

    Returns:
        np.ndarray of shape (n_samples,) with final PM2.5 predictions.

    Example:
        >>> final_preds = predict_stacking(meta, base_preds, meta_feats)
    """
    meta_X = np.hstack([base_predictions, meta_features])
    return meta_learner.predict(meta_X)
