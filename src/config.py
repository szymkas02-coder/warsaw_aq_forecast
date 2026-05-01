"""Central configuration: paths, constants, hyperparameter defaults."""
from pathlib import Path

# ── Repository root ───────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]

# ── Data ──────────────────────────────────────────────────────────────
DATA_PATH = ROOT / "data" / "raw" / "FINAL_merged_PM25_1g_all_seasons.csv"
PROCESSED_DIR = ROOT / "data" / "processed"

TARGET = "MzWarChrosci"
SPLIT_DATE = "2024-01-01"
HORIZONS = list(range(1, 25))  # 1h to 24h

# ── Stations ──────────────────────────────────────────────────────────
NEIGHBOR_STATIONS = [
    "MzOtwoBrzozo",
    "MzWarWokalna",
    "MzWarAlNiepo",
    "MzLegZegrzyn",
    "MzPiasPulask",
    "MzWarBajkowa",
]

STATION_COORDS = {
    "MzWarAlNiepo": (52.219298, 21.004724),
    "MzWarChrosci": (52.207742, 20.906073),
    "MzPiasPulask": (52.191728, 20.837489),
    "MzOtwoBrzozo": (52.115725, 21.237297),
    "MzWarBajkowa": (52.188474, 21.176233),
    "MzWarWokalna": (52.160772, 21.033819),
    "MzLegZegrzyn": (52.407578, 20.955928),
}

ALL_STATIONS = [
    "MzWarChrosci",
    "MzOtwoBrzozo",
    "MzWarWokalna",
    "MzWarAlNiepo",
    "MzLegZegrzyn",
    "MzPiasPulask",
    "MzWarBajkowa",
]

# ── Meteorological columns ─────────────────────────────────────────────
MET_COLS = ["TEMP", "WIND_SPEED", "PRESS_SEA", "HUMIDITY", "DEW_POINT", "RAIN_6H", "SUNSHINE", "blh"]

# ── HYSPLIT numeric columns ────────────────────────────────────────────
HYSPLIT_NUMERIC = [
    "lon_48", "lat_48", "lon_24", "lat_24",
    "dist_48_straight", "dist_48_total", "ratio_48",
    "dist_24_straight", "dist_24_total", "ratio_24",
]
HYSPLIT_CATEGORICAL = ["dir_48", "dir_24"]

# ── WHO / Polish thresholds ────────────────────────────────────────────
WHO_24H_PM25 = 15   # µg/m³ — WHO 2021 guideline
ALERT_THRESHOLD = 50  # µg/m³ — Polish alert level

# ── Model / training ──────────────────────────────────────────────────
RANDOM_SEED = 42
SEQ_LEN_GBM = 48    # lookback window for lag features
SEQ_LEN_LSTM = 48   # sequence length for CNN-LSTM / GNN
N_OPTUNA_TRIALS = 50
CV_SPLITS = 5

# ── Output paths ──────────────────────────────────────────────────────
OUTPUTS_DIR = ROOT / "outputs"


def get_station_paths(station: str) -> dict:
    """Return output paths parameterized by station name."""
    base = OUTPUTS_DIR / station
    return {
        "models":   base / "models",
        "figures":  base / "figures",
        "results":  base / "results",
    }


# Backward-compat: existing notebooks import these and expect outputs/MzWarChrosci/...
MODEL_DIR   = OUTPUTS_DIR / TARGET / "models"
FIGURES_DIR = OUTPUTS_DIR / TARGET / "figures"
RESULTS_DIR = OUTPUTS_DIR / TARGET / "results"
