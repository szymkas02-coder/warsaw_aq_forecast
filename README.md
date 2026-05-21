# Warsaw PM2.5 Forecasting

**Multi-architecture ML system for 1–24 hour PM2.5 forecasting in Warsaw, Poland**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## What this is

A complete forecasting pipeline that compares three increasingly complex ML architectures for predicting fine particulate matter (PM2.5) at the MzWarChrosci monitoring station (Targówek, Warsaw) on all horizons from 1 to 24 hours ahead.

The models are trained on 2019–2023 hourly data (Okęcie synoptic observations, ERA5 Boundary Layer Height, NOAA HYSPLIT back-trajectories, 7 PM2.5 monitoring stations) and evaluated on a fully held-out 2024 test year.

---

## Architectures

| ID | Architecture | Strategy | Key technique |
|---|---|---|---|
| **C1** | XGBoost / HistGradientBoosting | Direct multi-step (24 models) | Optuna + TimeSeriesSplit CV |
| **C2** | Bidirectional CNN-LSTM | MIMO (one model, 24 outputs) | Huber loss, early stopping |
| **C3** | GNN-LSTM + Stacking ensemble | OOF stacking meta-learner | Spatial graph over 7 stations |

All models use `weather_mode="perfect_forecast_full"` — both meteorological inputs and HYSPLIT trajectory features aligned to the predicted moment, simulating NWP output. This is the standard setup in the academic literature.

---

## Results (2024 test set)

| Model | h=1 MAE | h=6 MAE | h=12 MAE | h=24 MAE | h=24 R² |
|---|---|---|---|---|---|
| Persistence baseline | — | — | — | 5.41 µg/m³ | 0.10 |
| C1: HGB | **1.13** | **3.07** | **3.90** | 4.33 µg/m³ | 0.47 |
| C1: XGBoost | 1.14 | 3.17 | 3.97 | 4.39 µg/m³ | 0.45 |
| C2: CNN-LSTM | 2.67 | 4.28 | 4.66 | 4.38 µg/m³ | 0.37 |
| C3: GNN-LSTM (base) | 2.68 | 3.74 | 4.35 | **3.86 µg/m³** | **0.56** |
| C3: Stacking | 1.16 | 3.12 | 3.97 | 4.33 µg/m³ | 0.47 |

MAE in µg/m³. The stacking ensemble combines the short-horizon strength of tree models with the spatial-temporal skill of the GNN-LSTM. At h=24, the GNN-LSTM base model achieves the best R²=0.56, a **20% MAE reduction** vs. persistence.

---

## Data sources

| Source | Variables |
|---|---|
| GIOŚ monitoring network | PM2.5 at 7 Warsaw stations (1h, 2019–2024) |
| IMGW Okęcie | TEMP, WIND_SPEED, PRESSURE, HUMIDITY, DEW_POINT, RAIN, SUNSHINE (1h) |
| ERA5 (ECMWF) | Boundary Layer Height (1h) |
| NOAA HYSPLIT | 24h & 48h back-trajectory cluster directions (1h) |

---

## Project structure

```
warsaw_aq_forecast/
├── notebooks/
│   ├── MS_00_EDA.ipynb              ← EDA & naive baseline
│   ├── MS_00b_cross_station_EDA.ipynb
│   ├── MS_C1_xgboost.ipynb          ← Architecture C1 (train + evaluate)
│   ├── MS_C2_cnn_lstm.ipynb         ← Architecture C2
│   ├── MS_C3_gnn_stacking.ipynb     ← Architecture C3
│   ├── MS_04_comparison.ipynb       ← Cross-model comparison
│   ├── MS_05_shap.ipynb             ← SHAP analysis
│   └── MS_06_smog_episodes.ipynb    ← Top-5 smog episode case study
├── src/
│   ├── config.py                    ← constants & paths
│   ├── data_loader.py
│   ├── feature_engineering.py       ← all feature construction (leak-free)
│   ├── evaluation.py
│   └── models/
│       ├── baseline_gbm.py          ← C1
│       ├── hybrid_lstm.py           ← C2
│       └── gnn_stacking.py          ← C3
└── outputs/
    ├── models/                      ← .pkl and .pt artifacts
    ├── figures/
    └── results/                     ← comparison_table_all.csv
```

---

## Quickstart

```bash
git clone https://github.com/youruser/warsaw_aq_forecast.git
cd warsaw_aq_forecast

conda env create -f environment.yml
conda activate warsaw_aq

# Place FINAL_merged_PM25_1g_all_seasons.csv in data/raw/
jupyter notebook notebooks/
```

Run notebooks in order: `MS_00_EDA` → `MS_C1_xgboost` → `MS_C2_cnn_lstm` → `MS_C3_gnn_stacking` → `MS_04_comparison`.

---

## Anti-leakage guarantees

- Target: `y = PM2.5.shift(-h)` — inputs always at t, target always at t+h
- Neighbor PM2.5 lagged by ≥1h; rolling statistics use `.shift(1)` before `.rolling()`
- All CV uses `TimeSeriesSplit` — no shuffled k-fold
- Scalers fitted on train data only; stacking OOF uses per-fold fresh model instances

See [ARCHITECTURE.md](ARCHITECTURE.md) for a complete module-level code guide.

---

## References

Key references: Czernecki et al. (2021) — Polish AQ benchmarks; Liao et al. (2023) DM-STGNN — HYSPLIT-aware GNN; Tian et al. (2024) — stacking for PM2.5; Qi et al. (2019) GC-LSTM.

Full bibliography in [docs/README_long.md](docs/README_long.md#11-references).

---

## Acknowledgements

PM2.5 data: GIOŚ Poland | Meteorology: IMGW-PIB | ERA5: Copernicus/ECMWF | HYSPLIT: NOAA ARL

*Research project on ML applications for urban air quality management in Warsaw.*

---

## License

Code: [MIT License](LICENSE) | Documentation, results, and figures: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
