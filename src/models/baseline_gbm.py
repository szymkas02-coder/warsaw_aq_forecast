"""Architecture 1: Direct multi-step XGBoost / HistGradientBoosting forecasting."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import TimeSeriesSplit

from ..config import (
    HORIZONS, CV_SPLITS, N_OPTUNA_TRIALS, RANDOM_SEED,
    MODEL_DIR,
)
from ..feature_engineering import build_feature_matrix
from ..utils import setup_logging, ensure_dirs

log = setup_logging(__name__)


def _make_hgb(params: dict):
    """Instantiate a log-transformed HistGradientBoostingRegressor."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    base = HistGradientBoostingRegressor(random_state=RANDOM_SEED, **params)
    return TransformedTargetRegressor(
        regressor=base, func=np.log1p, inverse_func=np.expm1
    )


def _make_xgb(params: dict):
    """Instantiate a log-transformed XGBRegressor."""
    from xgboost import XGBRegressor
    base = XGBRegressor(random_state=RANDOM_SEED, tree_method="hist",
                        verbosity=0, **params)
    return TransformedTargetRegressor(
        regressor=base, func=np.log1p, inverse_func=np.expm1
    )


# ── Optuna objectives ─────────────────────────────────────────────────

def build_optuna_objective(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_type: str = "hgb",
):
    """Return an Optuna objective function for hyperparameter search.

    Uses TimeSeriesSplit cross-validation to prevent future data leakage.

    Args:
        X_train: Feature matrix.
        y_train: Target series.
        model_type: "hgb" or "xgb".

    Returns:
        Callable objective(trial) -> float (mean CV MAE).

    Example:
        >>> study = optuna.create_study(direction="minimize")
        >>> study.optimize(build_optuna_objective(X_tr, y_tr, "hgb"), n_trials=50)
    """
    from sklearn.metrics import mean_absolute_error

    tscv = TimeSeriesSplit(n_splits=CV_SPLITS)

    def objective(trial):
        if model_type == "hgb":
            params = {
                "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
                "max_iter": trial.suggest_int("max_iter", 100, 1000),
                "max_depth": trial.suggest_int("max_depth", 2, 8),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 50),
                "l2_regularization": trial.suggest_float("l2_regularization", 0.0, 1.0),
                "max_bins": trial.suggest_int("max_bins", 64, 255),
            }
            model = _make_hgb(params)
        else:
            params = {
                "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "max_depth": trial.suggest_int("max_depth", 2, 8),
                "min_child_weight": trial.suggest_int("min_child_weight", 5, 50),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            }
            model = _make_xgb(params)

        maes = []
        X_arr = X_train.values
        y_arr = y_train.values
        for tr_idx, val_idx in tscv.split(X_arr):
            model.fit(X_arr[tr_idx], y_arr[tr_idx])
            preds = model.predict(X_arr[val_idx])
            maes.append(mean_absolute_error(y_arr[val_idx], preds))
        return float(np.mean(maes))

    return objective


# ── Per-horizon training ──────────────────────────────────────────────

def train_single_horizon(
    df: pd.DataFrame,
    horizon: int,
    model_type: str = "hgb",
    n_trials: int = N_OPTUNA_TRIALS,
    neighbor_lag: int = 1,
    weather_mode: str = "current",
) -> Tuple[object, pd.DataFrame, pd.Series]:
    """Train one Optuna-tuned model for a single forecast horizon.

    Args:
        df: Full DataFrame (train slice only — do NOT pass test data).
        horizon: Forecast horizon in hours.
        model_type: "hgb" or "xgb".
        n_trials: Number of Optuna trials.
        neighbor_lag: Lag for neighbor station features.
        weather_mode: ``"current"`` or ``"perfect_forecast"``.

    Returns:
        Tuple of (fitted_model, X_train, y_train).

    Example:
        >>> model, X_tr, y_tr = train_single_horizon(train_df, horizon=24, model_type="hgb")
    """
    import optuna
    import time
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    X, y = build_feature_matrix(df, horizon=horizon, neighbor_lag=neighbor_lag,
                                 weather_mode=weather_mode)
    log.info("  X=%s  y_mean=%.2f  y_std=%.2f", X.shape, y.mean(), y.std())

    log.info("  Optuna search: %s | horizon=+%dh | %d trials | %d-fold TimeSeriesSplit",
             model_type.upper(), horizon, n_trials, CV_SPLITS)
    t0 = time.time()
    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))

    completed = [0]

    def _callback(study, trial):
        completed[0] += 1
        if completed[0] % max(1, n_trials // 10) == 0 or completed[0] == n_trials:
            log.info("  trial %d/%d — best CV MAE so far: %.4f",
                     completed[0], n_trials, study.best_value)

    study.optimize(build_optuna_objective(X, y, model_type),
                   n_trials=n_trials, show_progress_bar=False, callbacks=[_callback])

    best_params = study.best_params
    log.info("  Best h=%d: CV MAE=%.4f in %.0fs | params=%s",
             horizon, study.best_value, time.time() - t0, best_params)

    if model_type == "hgb":
        model = _make_hgb(best_params)
    else:
        model = _make_xgb(best_params)

    model.fit(X.values, y.values)
    return model, X, y


def train_direct_multistep(
    df: pd.DataFrame,
    model_type: str = "hgb",
    horizons: Optional[list] = None,
    n_trials: int = N_OPTUNA_TRIALS,
    neighbor_lag: int = 1,
    save_models: bool = True,
    weather_mode: str = "current",
) -> Dict[int, object]:
    """Train separate models for each forecast horizon (direct multi-step strategy).

    Args:
        df: Training DataFrame.
        model_type: "hgb" or "xgb".
        horizons: List of horizons to train (default HORIZONS = 1..24).
        n_trials: Optuna trials per horizon.
        neighbor_lag: Lag for neighbor features.
        save_models: If True, persist each model to MODEL_DIR.
        weather_mode: ``"current"`` or ``"perfect_forecast"``.

    Returns:
        Dict mapping horizon (int) → fitted model.

    Example:
        >>> models = train_direct_multistep(train_df, model_type="xgb")
        >>> models = train_direct_multistep(train_df, model_type="hgb",
        ...                                 weather_mode="perfect_forecast")
    """
    import time

    if horizons is None:
        horizons = HORIZONS

    total = len(horizons)
    log.info("=" * 60)
    log.info("Direct multi-step training: %s | %d horizons | %d Optuna trials each | weather=%s",
             model_type.upper(), total, n_trials, weather_mode)
    log.info("=" * 60)

    models: Dict[int, object] = {}
    wall_start = time.time()

    for idx, h in enumerate(horizons, start=1):
        t0 = time.time()
        log.info("[%d/%d] Starting horizon h=%d ...", idx, total, h)
        model, _, _ = train_single_horizon(df, h, model_type, n_trials, neighbor_lag,
                                           weather_mode=weather_mode)
        elapsed = time.time() - t0
        total_elapsed = time.time() - wall_start
        avg_per_h = total_elapsed / idx
        remaining = avg_per_h * (total - idx)
        log.info("[%d/%d] h=%d done in %.0fs | elapsed=%.0fs | ETA ~%.0fs",
                 idx, total, h, elapsed, total_elapsed, remaining)
        models[h] = model
        if save_models:
            ensure_dirs(MODEL_DIR)
            suffix = ("_pfx_full" if weather_mode == "perfect_forecast_full" else
                      "_pfx"      if weather_mode == "perfect_forecast" else "")
            path = MODEL_DIR / f"{model_type}{suffix}_h{h}.pkl"
            joblib.dump(model, path)
            log.info("  Saved → %s", path)

    log.info("All %d horizons trained in %.0fs total.", total, time.time() - wall_start)
    return models


def load_models(model_type: str = "hgb", horizons: Optional[list] = None) -> Dict[int, object]:
    """Load previously saved horizon models from disk.

    Args:
        model_type: "hgb" or "xgb".
        horizons: List of horizons to load (default 1..24).

    Returns:
        Dict {horizon: model}.

    Example:
        >>> models = load_models("hgb")
    """
    if horizons is None:
        horizons = HORIZONS
    models = {}
    for h in horizons:
        path = MODEL_DIR / f"{model_type}_h{h}.pkl"
        assert path.exists(), f"Model not found: {path}"
        models[h] = joblib.load(path)
    return models


def predict_all_horizons(
    models: Dict[int, object],
    X: pd.DataFrame,
) -> np.ndarray:
    """Run predictions for all horizons stored in models dict.

    Args:
        models: Dict {horizon: fitted_model}.
        X: Feature DataFrame (same feature set used during training).

    Returns:
        np.ndarray of shape (n_samples, len(models)), columns ordered by ascending horizon.

    Example:
        >>> preds = predict_all_horizons(models, X_test)
        >>> preds.shape
        (8760, 24)
    """
    horizons = sorted(models.keys())
    out = np.column_stack([models[h].predict(X.values) for h in horizons])
    assert out.shape == (len(X), len(horizons))
    return out
