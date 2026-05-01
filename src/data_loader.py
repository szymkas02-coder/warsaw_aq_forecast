"""Data ingestion, validation, and train/test splitting."""
import logging
from pathlib import Path
from typing import Tuple

import pandas as pd

from .config import (
    DATA_PATH, SPLIT_DATE, TARGET, NEIGHBOR_STATIONS,
    MET_COLS, HYSPLIT_NUMERIC, HYSPLIT_CATEGORICAL,
)
from .utils import setup_logging

log = setup_logging(__name__)


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the merged PM2.5 CSV, parse the DatetimeIndex, and validate columns.

    Args:
        path: Path to FINAL_merged_PM25_1g_all_seasons.csv.

    Returns:
        DataFrame with a sorted DatetimeIndex (hourly frequency).

    Example:
        >>> df = load_data()
        >>> df.shape
        (52608, 27)
    """
    path = Path(path)
    log.info("Loading data from %s", path)
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # Basic validation
    assert TARGET in df.columns, f"Target column '{TARGET}' not found"
    missing_met = [c for c in MET_COLS if c not in df.columns]
    if missing_met:
        log.warning("Missing meteorological columns: %s", missing_met)

    # Missing value report
    n_missing = df.isnull().sum()
    n_missing = n_missing[n_missing > 0]
    if len(n_missing):
        log.info("Missing values per column:\n%s", n_missing.to_string())
    else:
        log.info("No missing values detected")

    log.info("Loaded %d rows, %d columns, range %s → %s",
             len(df), df.shape[1], df.index.min(), df.index.max())
    return df


def train_test_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split chronologically at SPLIT_DATE (no shuffling).

    Args:
        df: Full DataFrame with DatetimeIndex.

    Returns:
        Tuple of (train_df, test_df).

    Example:
        >>> train, test = train_test_split(df)
        >>> test.index.min()
        Timestamp('2024-01-01 00:00:00')
    """
    train = df[df.index < SPLIT_DATE].copy()
    test = df[df.index >= SPLIT_DATE].copy()
    log.info("Train: %d rows (%s → %s)", len(train), train.index.min(), train.index.max())
    log.info("Test : %d rows (%s → %s)", len(test), test.index.min(), test.index.max())
    return train, test


def validate_no_future_leak(df: pd.DataFrame, feature_cols: list, horizon: int) -> None:
    """Assert that no feature column contains observations shifted into the future.

    Checks that raw (unlagged) target and neighbor values at time t are not
    included as features when the target horizon is t+horizon.

    Args:
        df: DataFrame with features and target.
        feature_cols: List of feature column names to check.
        horizon: Forecast horizon in hours.

    Raises:
        AssertionError: If raw (unlagged) station columns are found in feature_cols.

    Example:
        >>> validate_no_future_leak(df, X.columns.tolist(), horizon=24)
    """
    forbidden = [TARGET] + NEIGHBOR_STATIONS
    # Raw station cols should never appear unlagged in features
    leaking = [c for c in feature_cols if c in forbidden]
    assert not leaking, (
        f"Potential data leakage at horizon={horizon}: "
        f"raw station columns found in features: {leaking}"
    )
    log.info("Leakage check passed for horizon=%d", horizon)
