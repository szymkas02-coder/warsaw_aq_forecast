"""Architecture 2: Bidirectional CNN-LSTM hybrid (MIMO, 24-horizon forecast)."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from ..config import MODEL_DIR, RANDOM_SEED, HORIZONS
from ..utils import setup_logging, ensure_dirs

log = setup_logging(__name__)


# ── Model definition ──────────────────────────────────────────────────

def build_cnn_lstm_model(
    seq_len: int,
    n_features: int,
    n_horizons: int = 24,
) -> "torch.nn.Module":
    """Build the CNN-LSTM MIMO architecture.

    Architecture:
        Conv1D(64, k=3) + BatchNorm + ReLU
        Conv1D(128, k=3) + BatchNorm + ReLU
        MaxPool1D(2) + Dropout(0.2)
        BiLSTM(128) + Dropout(0.3)
        BiLSTM(64)
        Dense(64, ReLU) → Dense(n_horizons)

    Args:
        seq_len: Input sequence length (number of past timesteps).
        n_features: Number of input features per timestep.
        n_horizons: Number of output horizons (24 for full direct MIMO).

    Returns:
        Initialised PyTorch nn.Module.

    Example:
        >>> model = build_cnn_lstm_model(seq_len=48, n_features=50, n_horizons=24)
    """
    import torch
    import torch.nn as nn

    class CnnLstmModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv1d(n_features, 64, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm1d(64)
            self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm1d(128)
            self.pool = nn.MaxPool1d(kernel_size=2)
            self.drop1 = nn.Dropout(0.2)
            # After MaxPool: seq_len // 2
            self.lstm1 = nn.LSTM(128, 128, batch_first=True, bidirectional=True)
            self.drop2 = nn.Dropout(0.3)
            self.lstm2 = nn.LSTM(256, 64, batch_first=True, bidirectional=True)
            self.fc1 = nn.Linear(128, 64)
            self.fc2 = nn.Linear(64, n_horizons)
            self.relu = nn.ReLU()

        def forward(self, x):
            # x: (batch, seq_len, n_features)
            x = x.permute(0, 2, 1)          # → (batch, n_features, seq_len)
            x = self.relu(self.bn1(self.conv1(x)))
            x = self.relu(self.bn2(self.conv2(x)))
            x = self.drop1(self.pool(x))     # → (batch, 128, seq_len//2)
            x = x.permute(0, 2, 1)          # → (batch, seq_len//2, 128)
            x, _ = self.lstm1(x)
            x = self.drop2(x)
            x, _ = self.lstm2(x)
            x = x[:, -1, :]                  # last timestep → (batch, 128)
            x = self.relu(self.fc1(x))
            return self.fc2(x)               # (batch, n_horizons)

    return CnnLstmModel()


# ── Training ──────────────────────────────────────────────────────────

def train_cnn_lstm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 100,
    batch_size: int = 64,
    patience: int = 15,
    lr: float = 1e-3,
    save_path: Optional[Path] = None,
) -> Tuple["torch.nn.Module", dict]:
    """Train the CNN-LSTM model with early stopping.

    Targets are expected to already be log1p-transformed (use build_sequence_dataset).
    Loss function: Huber (delta=1.0) — robust to PM2.5 outlier spikes.

    Args:
        X_train: shape (n_train, seq_len, n_features).
        y_train: shape (n_train, n_horizons) — log1p-transformed.
        X_val: shape (n_val, seq_len, n_features).
        y_val: shape (n_val, n_horizons) — log1p-transformed.
        epochs: Maximum training epochs.
        batch_size: Mini-batch size.
        patience: Early-stopping patience.
        lr: Adam learning rate.
        save_path: If given, save best model weights here.

    Returns:
        Tuple of (best_model, history_dict).

    Example:
        >>> model, history = train_cnn_lstm(X_tr, y_tr, X_val, y_val)
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Training CNN-LSTM on %s", device)

    seq_len = X_train.shape[1]
    n_features = X_train.shape[2]
    n_horizons = y_train.shape[1]

    model = build_cnn_lstm_model(seq_len, n_features, n_horizons).to(device)

    Xt = torch.tensor(X_train, dtype=torch.float32)
    yt = torch.tensor(y_train, dtype=torch.float32)
    Xv = torch.tensor(X_val, dtype=torch.float32).to(device)
    yv = torch.tensor(y_val, dtype=torch.float32).to(device)

    train_ds = TensorDataset(Xt, yt)
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    criterion = nn.HuberLoss(delta=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5, min_lr=1e-5
    )

    best_val_loss = float("inf")
    best_state = None
    no_improve = 0
    history = {"train_loss": [], "val_loss": []}

    log.info("CNN-LSTM | seq_len=%d  n_features=%d  n_horizons=%d  device=%s",
             seq_len, n_features, n_horizons, device)
    log.info("Params: epochs=%d  batch=%d  patience=%d  lr=%.5f", epochs, batch_size, patience, lr)
    log.info("Training samples=%d  val samples=%d", len(Xt), len(Xv))
    log.info("%s", "-" * 65)

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item() * len(xb)
        epoch_loss /= len(Xt)

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(Xv), yv).item()

        scheduler.step(val_loss)
        history["train_loss"].append(epoch_loss)
        history["val_loss"].append(val_loss)

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        current_lr = optimizer.param_groups[0]["lr"]
        marker = " *" if improved else f"  (no improve {no_improve}/{patience})"
        log.info("Epoch %3d/%d  train=%.5f  val=%.5f  best=%.5f  lr=%.2e%s",
                 epoch, epochs, epoch_loss, val_loss, best_val_loss, current_lr, marker)

        if no_improve >= patience:
            log.info("Early stopping triggered at epoch %d — best val=%.5f", epoch, best_val_loss)
            break

    model.load_state_dict(best_state)
    log.info("Training complete — best val loss=%.5f", best_val_loss)

    if save_path:
        ensure_dirs(Path(save_path).parent)
        torch.save({"model_state": best_state,
                    "seq_len": seq_len,
                    "n_features": n_features,
                    "n_horizons": n_horizons}, save_path)
        log.info("Model saved → %s", save_path)

    return model, history


def predict_cnn_lstm(
    model: "torch.nn.Module",
    X_seq: np.ndarray,
    batch_size: int = 256,
) -> np.ndarray:
    """Run inference with the CNN-LSTM model.

    Returns predictions in the original (expm1) PM2.5 scale.

    Args:
        model: Fitted CnnLstmModel.
        X_seq: shape (n_samples, seq_len, n_features).
        batch_size: Inference batch size.

    Returns:
        np.ndarray shape (n_samples, n_horizons) in µg/m³.

    Example:
        >>> preds = predict_cnn_lstm(model, X_seq_test)
    """
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    device = next(model.parameters()).device
    model.eval()
    Xt = torch.tensor(X_seq, dtype=torch.float32)
    ds = TensorDataset(Xt)
    loader = DataLoader(ds, batch_size=batch_size)
    parts = []
    with torch.no_grad():
        for (xb,) in loader:
            parts.append(model(xb.to(device)).cpu().numpy())
    log_preds = np.concatenate(parts, axis=0)
    return np.expm1(log_preds)


def load_cnn_lstm(path: Path) -> "torch.nn.Module":
    """Load a saved CNN-LSTM model from disk.

    Args:
        path: Path to .pt checkpoint saved by train_cnn_lstm.

    Returns:
        Model loaded to CPU, in eval mode.

    Example:
        >>> model = load_cnn_lstm(MODEL_DIR / "cnn_lstm_model.pt")
    """
    import torch
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = build_cnn_lstm_model(ckpt["seq_len"], ckpt["n_features"], ckpt["n_horizons"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model
