"""All feature construction functions for the Warsaw PM2.5 pipeline."""
import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .config import (
    TARGET, NEIGHBOR_STATIONS, MET_COLS,
    HYSPLIT_NUMERIC, HYSPLIT_CATEGORICAL, RANDOM_SEED,
)
from .utils import setup_logging

log = setup_logging(__name__)


# ── Temporal features ─────────────────────────────────────────────────

def add_cyclical_time(df: pd.DataFrame) -> pd.DataFrame:
    """Add sin/cos cyclical encodings for hour, month, day-of-week, plus is_weekend flag.

    Args:
        df: DataFrame with DatetimeIndex.

    Returns:
        DataFrame with new columns: hour_sin, hour_cos, month_sin, month_cos,
        day_sin, day_cos, is_weekend.

    Example:
        >>> df = add_cyclical_time(df)
        >>> "hour_sin" in df.columns
        True
    """
    df = df.copy()
    df["hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24)
    df["month_sin"] = np.sin(2 * np.pi * df.index.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * df.index.month / 12)
    df["day_sin"] = np.sin(2 * np.pi * df.index.dayofweek / 7)
    df["day_cos"] = np.cos(2 * np.pi * df.index.dayofweek / 7)
    df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)
    return df


# ── PM2.5 autoregressive features ────────────────────────────────────

def add_pm25_lags(df: pd.DataFrame) -> pd.DataFrame:
    """Add lagged PM2.5 values and rolling statistics, all computed on past data only.

    Features added:
        pm25_now       — current observation (t=0)
        pm25_lag_24h   — value 24 h ago
        pm25_lag_48h   — value 48 h ago
        rolling_mean_24h — 24-h rolling mean (closed on left, excludes t=0)
        rolling_max_24h  — 24-h rolling max (same)
        trend_24h        — pm25_now - pm25_lag_24h

    Args:
        df: DataFrame with DatetimeIndex containing TARGET column.

    Returns:
        DataFrame with new PM2.5 feature columns.

    Example:
        >>> df = add_pm25_lags(df)
    """
    df = df.copy()
    s = df[TARGET]
    df["pm25_now"] = s
    df["pm25_lag_24h"] = s.shift(24)
    df["pm25_lag_48h"] = s.shift(48)
    # Rolling window on past 24h (shift by 1 so t=0 is excluded)
    rolled = s.shift(1).rolling(window=24, min_periods=12)
    df["rolling_mean_24h"] = rolled.mean()
    df["rolling_max_24h"] = rolled.max()
    df["trend_24h"] = df["pm25_now"] - df["pm25_lag_24h"]
    return df


# ── Spatial / neighbor features ───────────────────────────────────────

def add_neighbor_lags(df: pd.DataFrame, lag: int = 1) -> pd.DataFrame:
    """Add lagged neighbor station PM2.5 readings.

    Args:
        df: DataFrame containing NEIGHBOR_STATIONS columns.
        lag: Number of hours to lag (default 1 for general use; use 24 for +24h-only models).

    Returns:
        DataFrame with columns {station}_lag{lag}h for each neighbor.

    Example:
        >>> df = add_neighbor_lags(df, lag=1)
    """
    df = df.copy()
    for col in NEIGHBOR_STATIONS:
        if col in df.columns:
            df[f"{col}_lag{lag}h"] = df[col].shift(lag)
        else:
            log.warning("Neighbor column '%s' not found, skipping", col)
    return df


# ── Weather features (mode-aware) ────────────────────────────────────

WEATHER_MODE_CURRENT       = "current"               # MET_COLS at t   (default)
WEATHER_MODE_PERFECT       = "perfect_forecast"       # MET_COLS at t+h
WEATHER_MODE_PERFECT_FULL  = "perfect_forecast_full"  # MET_COLS + HYSPLIT at t+h


def add_weather_features(
    df: pd.DataFrame,
    horizon: Optional[int] = None,
    mode: str = WEATHER_MODE_CURRENT,
) -> pd.DataFrame:
    """Apply the selected forecast mode to meteorological (and optionally HYSPLIT) columns.

    Three modes:

    ``"current"`` (default)
        All columns stay at time t.  No transformation.

    ``"perfect_forecast"``
        MET_COLS shifted to t+horizon.  Simulates a perfect NWP forecast for
        the predicted moment (standard academic setup, e.g. Liao et al., 2023).
        HYSPLIT features remain at t.

    ``"perfect_forecast_full"``
        MET_COLS **and** HYSPLIT_NUMERIC **and** HYSPLIT_CATEGORICAL all shifted
        to t+horizon.  The categorical direction columns are shifted as raw
        strings before ``encode_hysplit`` runs, so the resulting dummies reflect
        the air-mass origin at the predicted moment.  Maximum information scenario.

    None of these modes use future PM2.5 — only weather/trajectory proxies that
    would come from an NWP or trajectory forecast model.

    Args:
        df: DataFrame with DatetimeIndex.
        horizon: Forecast horizon in hours.  Required for any non-``"current"`` mode.
        mode: One of ``"current"``, ``"perfect_forecast"``,
            ``"perfect_forecast_full"``.

    Returns:
        DataFrame with the relevant columns replaced in-place (copy made internally).

    Raises:
        ValueError: Unknown mode, or horizon is None for a non-current mode.

    Example:
        >>> df = add_weather_features(df, horizon=24, mode="perfect_forecast")
        >>> df = add_weather_features(df, horizon=24, mode="perfect_forecast_full")
    """
    if mode == WEATHER_MODE_CURRENT:
        return df

    if mode in (WEATHER_MODE_PERFECT, WEATHER_MODE_PERFECT_FULL):
        if horizon is None:
            raise ValueError(f"horizon must be provided when mode='{mode}'")
        df = df.copy()

        # Always shift MET_COLS
        n_met = 0
        for col in MET_COLS:
            if col in df.columns:
                df[col] = df[col].shift(-horizon)
                n_met += 1

        if mode == WEATHER_MODE_PERFECT_FULL:
            # Also shift HYSPLIT numeric columns
            n_num = 0
            for col in HYSPLIT_NUMERIC:
                if col in df.columns:
                    df[col] = df[col].shift(-horizon)
                    n_num += 1
            # Shift HYSPLIT categorical columns BEFORE encode_hysplit one-hot encodes them
            n_cat = 0
            for col in HYSPLIT_CATEGORICAL:
                if col in df.columns:
                    df[col] = df[col].shift(-horizon)
                    n_cat += 1
            log.info("perfect_forecast_full: shifted %d MET + %d HYSPLIT_NUM + %d HYSPLIT_CAT by -%d h",
                     n_met, n_num, n_cat, horizon)
        else:
            log.info("perfect_forecast: shifted %d MET_COLS by -%d hours", n_met, horizon)

        return df

    raise ValueError(
        f"Unknown weather_mode '{mode}'. "
        "Use 'current', 'perfect_forecast', or 'perfect_forecast_full'."
    )


# ── HYSPLIT features ──────────────────────────────────────────────────

def encode_hysplit(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode categorical HYSPLIT direction columns and keep all numeric HYSPLIT features.

    Categorical columns (dir_48, dir_24) are one-hot encoded with drop_first=True.
    Numeric columns (lon_*, lat_*, dist_*, ratio_*) are passed through unchanged.

    Args:
        df: DataFrame that may contain HYSPLIT_CATEGORICAL and HYSPLIT_NUMERIC columns.

    Returns:
        DataFrame with categorical HYSPLIT columns replaced by dummies and numeric
        HYSPLIT columns retained as-is.

    Example:
        >>> df = encode_hysplit(df)
    """
    df = df.copy()
    present_cat = [c for c in HYSPLIT_CATEGORICAL if c in df.columns]
    if present_cat:
        dummies = pd.get_dummies(df[present_cat], drop_first=True, dtype=float)
        df = pd.concat([df.drop(columns=present_cat), dummies], axis=1)
        log.info("One-hot encoded HYSPLIT categoricals: %s → %d dummy cols",
                 present_cat, len(dummies.columns))
    present_num = [c for c in HYSPLIT_NUMERIC if c in df.columns]
    if present_num:
        log.info("Retaining %d HYSPLIT numeric columns", len(present_num))
    return df


# ── Full feature matrix builder ───────────────────────────────────────

def _get_feature_cols(df: pd.DataFrame) -> List[str]:
    """Return ordered list of all feature columns (excludes raw station columns and target)."""
    exclude = set([TARGET] + NEIGHBOR_STATIONS + HYSPLIT_CATEGORICAL)
    return [c for c in df.columns if c not in exclude]


def build_feature_matrix(
    df: pd.DataFrame,
    horizon: int,
    neighbor_lag: int = 1,
    weather_mode: str = WEATHER_MODE_CURRENT,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Assemble the full feature matrix X and honest future target y for a given horizon.

    Pipeline:
        1. Add cyclical time features
        2. Add PM2.5 lags and rolling statistics  (always at t)
        3. Add neighbor station lags              (always at t-neighbor_lag)
        4. Apply weather mode                     (t or t+horizon depending on mode)
        5. Encode HYSPLIT
        6. Construct target y = TARGET.shift(-horizon)
        7. Drop rows with NaN; assert X clean

    Weather modes
    -------------
    ``"current"`` (default)
        MET_COLS are at time t — the weather when the forecast is issued.
    ``"perfect_forecast"``
        MET_COLS are shifted to t+horizon — simulates a perfect NWP forecast
        for the predicted moment.  Standard in academic literature.

    Args:
        df: Raw DataFrame with DatetimeIndex.
        horizon: Forecast horizon in hours (1–24).
        neighbor_lag: Lag applied to neighbor stations (default 1).
        weather_mode: ``"current"`` or ``"perfect_forecast"``.

    Returns:
        Tuple of (X: pd.DataFrame, y: pd.Series) with aligned indices.

    Example:
        >>> X, y = build_feature_matrix(df, horizon=24)
        >>> X_pfx, y = build_feature_matrix(df, horizon=24, weather_mode="perfect_forecast")
    """
    log.info("build_feature_matrix | horizon=%d | weather_mode=%s", horizon, weather_mode)

    df = add_cyclical_time(df)
    df = add_pm25_lags(df)
    df = add_neighbor_lags(df, lag=neighbor_lag)
    df = add_weather_features(df, horizon=horizon, mode=weather_mode)
    df = encode_hysplit(df)

    # Honest target: row t predicts TARGET at t+horizon
    y = df[TARGET].shift(-horizon).rename(f"pm25_h{horizon}")

    # Feature columns = everything except raw station cols and categoricals
    feature_cols = _get_feature_cols(df)
    X = df[feature_cols]

    # Align and drop NaN rows (covers lags at the start AND horizon shift at the end)
    combined = pd.concat([X, y], axis=1).dropna()
    X = combined[feature_cols]
    y = combined[y.name]

    assert X.isnull().sum().sum() == 0, "NaN values remain in feature matrix"
    assert len(X) == len(y), "X and y length mismatch"

    log.info("  X: %s | y mean=%.2f, y std=%.2f", X.shape, y.mean(), y.std())
    return X, y


# ── Sequence dataset builder (CNN-LSTM / GNN) ─────────────────────────

def build_sequence_dataset(
    df: pd.DataFrame,
    seq_len: int = 48,
    horizons: Optional[List[int]] = None,
    neighbor_lag: int = 1,
    scaler_X: Optional[StandardScaler] = None,
    fit_scaler: bool = False,
    weather_mode: str = WEATHER_MODE_CURRENT,
) -> Tuple[np.ndarray, np.ndarray, List[str], Optional[StandardScaler]]:
    """Build sliding-window 3-D arrays for sequence models (CNN-LSTM, GNN).

    Each sample contains seq_len timesteps of features and a target vector of
    len(horizons) future PM2.5 values. The target is log1p-transformed.

    Weather modes
    -------------
    ``"current"`` (default)
        MET_COLS at time t throughout the sequence.
    ``"perfect_forecast"``
        MET_COLS shifted by ``-max(horizons)`` hours, so the last timestep of
        every window carries the weather at the furthest predicted moment.
        This is a conservative approximation for MIMO: the same shifted weather
        is used across all 24 output horizons.  Consistent with how NWP output
        would be used in practice (one forecast run covers the whole window).

    Args:
        df: Raw DataFrame with DatetimeIndex (should be train or test only).
        seq_len: Number of past timesteps per sample.
        horizons: List of forecast horizons to predict simultaneously (MIMO).
            Defaults to [1, ..., 24].
        neighbor_lag: Lag applied to neighbor stations.
        scaler_X: Pre-fitted StandardScaler. If None and fit_scaler=True, one is fitted.
        fit_scaler: Whether to fit a new StandardScaler on this data.
        weather_mode: ``"current"`` or ``"perfect_forecast"``.

    Returns:
        Tuple of:
            X_seq: np.ndarray shape (n_samples, seq_len, n_features)
            y_seq: np.ndarray shape (n_samples, len(horizons)) — log1p-transformed
            feature_names: list of feature column names
            scaler_X: fitted scaler (or None if fit_scaler=False and none provided)

    Example:
        >>> X_seq, y_seq, feat_names, scaler = build_sequence_dataset(train_df, fit_scaler=True)
        >>> X_pfx, y_seq, _, _ = build_sequence_dataset(train_df, fit_scaler=True,
        ...                                              weather_mode="perfect_forecast")
    """
    if horizons is None:
        horizons = list(range(1, 25))

    # For MIMO perfect_forecast, shift met cols by the furthest horizon
    _pfx_horizon = max(horizons) if weather_mode in (WEATHER_MODE_PERFECT, WEATHER_MODE_PERFECT_FULL) else None
    log.info("build_sequence_dataset | seq_len=%d | weather_mode=%s", seq_len, weather_mode)

    df = add_cyclical_time(df)
    df = add_pm25_lags(df)
    df = add_neighbor_lags(df, lag=neighbor_lag)
    df = add_weather_features(df, horizon=_pfx_horizon, mode=weather_mode)
    df = encode_hysplit(df)

    feature_cols = _get_feature_cols(df)
    X_df = df[feature_cols].copy()

    # Build target matrix: each column is one horizon's future value
    target_matrix = pd.DataFrame(
        {f"h{h}": df[TARGET].shift(-h) for h in horizons},
        index=df.index,
    )

    # Drop rows where any feature or any target is NaN
    combined = pd.concat([X_df, target_matrix], axis=1).dropna()
    X_df = combined[feature_cols]
    target_matrix = combined[[f"h{h}" for h in horizons]]

    # Scale features
    X_vals = X_df.values.astype(np.float32)
    if fit_scaler:
        scaler_X = StandardScaler()
        X_vals = scaler_X.fit_transform(X_vals)
    elif scaler_X is not None:
        X_vals = scaler_X.transform(X_vals)

    # Log-transform targets
    y_vals = np.log1p(target_matrix.values.astype(np.float32))

    # Sliding window: sample i uses rows i..i+seq_len-1 as input,
    # and target at row i+seq_len-1 (last step in window)
    n = len(X_vals)
    n_samples = n - seq_len + 1
    assert n_samples > 0, f"Not enough rows ({n}) for seq_len={seq_len}"

    X_seq = np.stack([X_vals[i:i + seq_len] for i in range(n_samples)])
    y_seq = y_vals[seq_len - 1:]  # target aligned to last timestep of each window

    assert X_seq.shape == (n_samples, seq_len, len(feature_cols))
    assert y_seq.shape == (n_samples, len(horizons))

    log.info("Sequence dataset: X=%s, y=%s, features=%d",
             X_seq.shape, y_seq.shape, len(feature_cols))
    return X_seq, y_seq, feature_cols, scaler_X
