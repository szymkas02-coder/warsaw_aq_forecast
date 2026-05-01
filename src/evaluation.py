"""Metrics computation and visualisation utilities used across all notebooks."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from .config import WHO_24H_PM25, FIGURES_DIR
from .utils import setup_logging, ensure_dirs

log = setup_logging(__name__)

# ── Matplotlib style ──────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 120,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
})


# ── Core metrics ──────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute MAE, RMSE, R², MBE, and Index of Agreement (Willmott 1981).

    Args:
        y_true: Array of true PM2.5 values (µg/m³).
        y_pred: Array of predicted PM2.5 values (µg/m³).

    Returns:
        Dictionary with keys: MAE, RMSE, R2, MBE, IA.

    Example:
        >>> metrics = compute_metrics(y_true, y_pred)
        >>> metrics["MAE"]
        4.26
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = y_true[mask], y_pred[mask]

    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    mbe = np.mean(y_pred - y_true)
    # Willmott (1981) Index of Agreement
    ymean = y_true.mean()
    denom = np.sum((np.abs(y_pred - ymean) + np.abs(y_true - ymean)) ** 2)
    ia = 1 - ss_res / denom if denom > 0 else float("nan")

    return {"MAE": round(mae, 4), "RMSE": round(rmse, 4),
            "R2": round(r2, 4), "MBE": round(mbe, 4), "IA": round(ia, 4)}


# ── Per-horizon plots ─────────────────────────────────────────────────

def plot_horizon_metrics(
    results_dict: Dict[str, Dict[int, Dict[str, float]]],
    metric: str = "MAE",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot a chosen metric vs. forecast horizon for multiple models.

    Args:
        results_dict: Nested dict {model_name: {horizon: metrics_dict}}.
        metric: Metric key to plot (MAE, RMSE, R2, IA).
        save_path: Optional path to save the figure.

    Returns:
        Matplotlib Figure.

    Example:
        >>> fig = plot_horizon_metrics({"XGB": xgb_results, "HGB": hgb_results})
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    markers = ["o", "s", "^", "D", "v"]
    for i, (name, horizon_metrics) in enumerate(results_dict.items()):
        horizons = sorted(horizon_metrics.keys())
        values = [horizon_metrics[h][metric] for h in horizons]
        ax.plot(horizons, values, marker=markers[i % len(markers)], label=name, linewidth=2)

    ax.set_xlabel("Forecast Horizon (h)")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} vs. Forecast Horizon")
    ax.set_xticks(range(1, 25, 2))
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if save_path:
        ensure_dirs(Path(save_path).parent)
        fig.savefig(save_path, bbox_inches="tight")
        log.info("Saved figure: %s", save_path)
    return fig


# ── Actual vs predicted time series ───────────────────────────────────

def plot_actual_vs_predicted(
    y_true: pd.Series,
    y_pred: np.ndarray,
    title: str = "Actual vs. Predicted",
    zoom_period: Optional[Tuple[str, str]] = None,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Time-series plot of actual and predicted PM2.5 with WHO guideline line.

    Args:
        y_true: Series with DatetimeIndex.
        y_pred: Array aligned to y_true.
        title: Plot title.
        zoom_period: Optional (start, end) ISO date strings for zoomed view.
        save_path: Optional save path.

    Returns:
        Matplotlib Figure.

    Example:
        >>> fig = plot_actual_vs_predicted(y_test, preds, "XGBoost +24h", ("2024-01", "2024-03"))
    """
    pred_series = pd.Series(y_pred, index=y_true.index, name="Predicted")
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(y_true.index, y_true.values, label="Actual", color="#2c7bb6", linewidth=1, alpha=0.85)
    ax.plot(pred_series.index, pred_series.values, label="Predicted",
            color="#d7191c", linewidth=1, alpha=0.75)
    ax.axhline(WHO_24H_PM25, color="green", linestyle="--", linewidth=1.2, label=f"WHO 24h ({WHO_24H_PM25} µg/m³)")

    if zoom_period:
        ax.set_xlim(pd.Timestamp(zoom_period[0]), pd.Timestamp(zoom_period[1]))

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
    ax.set_ylabel("PM2.5 (µg/m³)")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    if save_path:
        ensure_dirs(Path(save_path).parent)
        fig.savefig(save_path, bbox_inches="tight")
    return fig


# ── Smog episode analysis ─────────────────────────────────────────────

def plot_smog_episodes(
    y_true: pd.Series,
    predictions_dict: Dict[str, np.ndarray],
    n_episodes: int = 5,
    window_hours: int = 72,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Find the top-N worst PM2.5 episodes and plot all model forecasts against actual.

    An episode is centred on the peak PM2.5 hour, extended ±window_hours/2.

    Args:
        y_true: Series of actual PM2.5 with DatetimeIndex.
        predictions_dict: {model_name: predictions_array} aligned to y_true.
        n_episodes: Number of worst episodes to show.
        window_hours: Width of each episode window in hours.
        save_path: Optional save path prefix (episode index appended).

    Returns:
        Matplotlib Figure (last episode).

    Example:
        >>> fig = plot_smog_episodes(y_test, {"XGB": xgb_preds, "HGB": hgb_preds})
    """
    # Find top-N peak hours (non-overlapping, greedy)
    remaining = y_true.copy()
    episode_peaks = []
    half = window_hours // 2
    for _ in range(n_episodes):
        if remaining.empty or remaining.max() < 0:
            break
        peak_idx = remaining.idxmax()
        episode_peaks.append(peak_idx)
        # Mask out this window
        mask = (remaining.index >= peak_idx - pd.Timedelta(hours=half)) & \
               (remaining.index <= peak_idx + pd.Timedelta(hours=half))
        remaining[mask] = -np.inf

    colors = ["#2c7bb6", "#d7191c", "#1a9641", "#fdae61", "#762a83"]
    fig, axes = plt.subplots(n_episodes, 1, figsize=(14, 4 * n_episodes), sharex=False)
    if n_episodes == 1:
        axes = [axes]

    for ep_i, (peak, ax) in enumerate(zip(episode_peaks, axes)):
        start = peak - pd.Timedelta(hours=half)
        end = peak + pd.Timedelta(hours=half)
        mask = (y_true.index >= start) & (y_true.index <= end)
        t_ep = y_true[mask]

        ax.plot(t_ep.index, t_ep.values, label="Actual", color=colors[0], linewidth=2)
        for c_i, (name, preds) in enumerate(predictions_dict.items()):
            p_ep = pd.Series(preds, index=y_true.index)[mask]
            ax.plot(p_ep.index, p_ep.values, label=name,
                    color=colors[(c_i + 1) % len(colors)], linewidth=1.5, linestyle="--")

        ax.axhline(WHO_24H_PM25, color="green", linestyle=":", linewidth=1)
        ax.set_title(f"Episode {ep_i + 1} — peak {peak.strftime('%Y-%m-%d %H:%M')}"
                     f" ({t_ep.max():.1f} µg/m³)")
        ax.set_ylabel("PM2.5 (µg/m³)")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.25)

    fig.tight_layout()
    if save_path:
        ensure_dirs(Path(save_path).parent)
        fig.savefig(save_path, bbox_inches="tight")
    return fig


# ── Summary comparison table ──────────────────────────────────────────

def comparison_table(
    all_results: Dict[str, Dict[int, Dict[str, float]]],
    key_horizons: Optional[List[int]] = None,
) -> pd.DataFrame:
    """Build a LaTeX-ready comparison table for key horizons.

    Args:
        all_results: Nested dict {model_name: {horizon: metrics_dict}}.
        key_horizons: Horizons to include (default [1, 6, 12, 24]).

    Returns:
        Multi-index DataFrame with (model, horizon) rows and metric columns.

    Example:
        >>> df = comparison_table(all_results)
        >>> print(df.to_latex(float_format="%.2f"))
    """
    if key_horizons is None:
        key_horizons = [1, 6, 12, 24]

    rows = []
    for model_name, horizon_metrics in all_results.items():
        for h in key_horizons:
            if h in horizon_metrics:
                row = {"Model": model_name, "Horizon": h}
                row.update(horizon_metrics[h])
                rows.append(row)

    df = pd.DataFrame(rows).set_index(["Model", "Horizon"]).sort_index()
    return df


# ── SHAP analysis ─────────────────────────────────────────────────────

def shap_analysis(
    model,
    X_test: pd.DataFrame,
    feature_names: List[str],
    save_dir: Optional[Path] = None,
    title_suffix: str = "",
) -> None:
    """Produce SHAP summary and waterfall plots for the worst prediction error.

    Args:
        model: Fitted sklearn-compatible model (tree-based).
        X_test: Test feature DataFrame.
        feature_names: List of feature names aligned to X_test columns.
        save_dir: Directory to save figures (uses FIGURES_DIR if None).
        title_suffix: Appended to figure titles (e.g., "h=24").

    Example:
        >>> shap_analysis(xgb_model, X_test, feature_names, title_suffix="h=24")
    """
    try:
        import shap
    except ImportError:
        log.warning("shap not installed — skipping SHAP analysis")
        return

    save_dir = Path(save_dir) if save_dir else FIGURES_DIR
    ensure_dirs(save_dir)

    explainer = shap.TreeExplainer(model)
    # Use a sample to keep it fast (max 2000 rows)
    sample = X_test.iloc[:2000] if len(X_test) > 2000 else X_test
    shap_values = explainer.shap_values(sample)

    # Summary plot — shap.summary_plot creates its own figure internally,
    # so we capture it via plt.gcf() after the call rather than using a pre-created figure.
    shap.summary_plot(shap_values, sample, feature_names=feature_names,
                      show=False, plot_size=(10, 6))
    plt.title(f"SHAP Feature Importance {title_suffix}")
    plt.tight_layout()
    fig_summary = plt.gcf()
    fig_summary.savefig(save_dir / f"shap_summary_{title_suffix}.png", bbox_inches="tight")
    plt.close(fig_summary)

    log.info("SHAP plots saved to %s", save_dir)
