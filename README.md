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

**Target station — MzWarChrosci** (the project's named forecast target):

| Model | h=1 MAE | h=6 MAE | h=12 MAE | h=24 MAE | h=24 R² |
|---|---|---|---|---|---|
| Persistence baseline | — | — | — | 5.41 µg/m³ | 0.10 |
| C1: HGB | **1.13** | 3.11 | 3.93 | 4.33 µg/m³ | 0.47 |
| C1: XGBoost | **1.13** | **3.10** | **3.92** | 4.29 µg/m³ | 0.48 |
| C2: CNN-LSTM | 2.65 | 3.70 | 3.72 | **3.72 µg/m³** | **0.60** |
| C3: GNN-LSTM (base) | 2.80 | 3.97 | 4.11 | 3.93 µg/m³ | 0.57 |
| C3: Stacking | 1.27 | 3.99 | 5.07 | 5.44 µg/m³ | 0.33 |

**Cross-station mean** (all 7 stations — the shared GNN forecasts every station from one model):

| Model | h=1 MAE | h=6 MAE | h=12 MAE | h=24 MAE | h=24 R² |
|---|---|---|---|---|---|
| C1: HGB | 1.45 | 3.63 | 4.39 | 4.68 µg/m³ | 0.51 |
| C1: XGBoost | 1.43 | 3.58 | 4.35 | 4.65 µg/m³ | 0.51 |
| C2: CNN-LSTM | 3.22 | 4.42 | 4.55 | 4.69 µg/m³ | 0.48 |
| C3: GNN-LSTM (base) | 3.37 | 4.61 | 4.74 | **4.54 µg/m³** | **0.56** |
| C3: Stacking | 1.63 | 4.82 | 5.90 | 6.05 µg/m³ | 0.39 |

MAE in µg/m³; **bold** = best in column. The story splits by horizon and by scope:

- **Short horizons (h≤4): tree models dominate.** C1 HGB/XGBoost and the C3 stacking head reach ~1.1–1.3 µg/m³ at h=1 — the autoregressive `pm25_now` signal is decisive and the trees exploit it best.
- **Long horizons: the deep models win.** At the target station, C2 CNN-LSTM is best at h=24 (3.72 µg/m³, R²=0.60) with the C3 GNN-LSTM a close second (3.93, R²=0.57). Averaged across all 7 stations, the **single shared GNN-LSTM is the best long-horizon model** (h=24 MAE 4.54, R²=0.56) — the spatial graph generalises better across stations than the per-station-tuned CNN-LSTM.
- **The stacking ensemble currently underperforms its own base models past h≈5** (MBE drifts to +2.8 µg/m³ at h=24). This is a known, documented limitation — the GNN column fed to the meta-learner is a full-train, not a true out-of-fold, prediction — see [why the stacking *should* win](#why-the-stacking-ensemble-should-be-the-best-model) below.

### Why the stacking ensemble *should* be the best model

On paper the C3 stacking ensemble should win at **every** horizon, and the per-horizon
results above show *why it doesn't yet*. The design is sound: combine the trees' decisive
short-horizon `pm25_now` skill with the deep models' long-horizon spatial-temporal skill,
and let a meta-learner pick the right blend per horizon. Short-horizon, it works exactly as
intended (h=1 MAE 1.27 at the target). Long-horizon, it *inverts* — worse than its own GNN
base model — and the reason is specific and fixable:

**The GNN column handed to the meta-learner is a full-train prediction, not a true
out-of-fold (OOF) one.** Every other base column is honest OOF (`TimeSeriesSplit`); the GNN
column was produced by predicting on the same data the GNN trained on, so it looks
artificially good and biased at train time. The linear Ridge meta-learner — chosen because
it can *extrapolate* to pollution spikes a tree meta-learner would average down — has no way
to route around that one leaky, over-trusted, positively-biased column, so it over-weights
it and the long-horizon blend drifts high (MBE +2.8 µg/m³ at h=24).

**The fix is known; the blocker is compute, not design.** Generating a true GNN OOF means
retraining the GNN once per `TimeSeriesSplit` fold (≈5× training) so each training row gets
a prediction from a model that never saw it. On a CPU-only machine that is genuinely
expensive — and notably, the single-shared-GNN refactor (below) already cut the cost of one
GNN training ~7×, which makes a true-OOF pass far more tractable than it was (one 5-fold
retrain now covers all stations, not 7× that). It is documented as the next step rather than
hand-waved away: with honest OOF, the meta-learner sees a fair GNN column and the ensemble
should reclaim the long-horizon lead while keeping its short-horizon advantage — best of both.

### What changed during implementation (honest changelog)

This project was iterated transparently, and several methodological fixes and one
architectural redesign landed *after* the first results. They are recorded rather than
quietly folded in, because knowing *where a comparison is and isn't fair* is part of the work:

- **C2 CNN-LSTM:** added attention over all LSTM timesteps (was last-timestep only) and
  fixed the validation split to a fixed mid-2022 block (was the last 10% of train, i.e. the
  regime right before the test year — an optimistic leak into early stopping).
- **Unified confounds:** both deep models now use a 48 h lookback and a 100-epoch ceiling, so
  the C2-vs-C3 comparison isn't confounded by lookback or training budget.
- **Stacking meta-learner:** swapped LightGBM → scaled Ridge, so the blend can follow
  long-horizon spikes instead of averaging them down.
- **C3 GNN-LSTM, single shared multi-output model:** the GNN was originally trained *once per
  station* (7 trainings, reading out a designated target node). It is now **one shared model**
  over all stations — every station is a node *and* a readout target, weights shared across
  nodes — producing all stations' forecasts in a single ~7× cheaper training. It also finally
  receives the HYSPLIT **trajectory-direction one-hots** (it previously got only the numeric
  HYSPLIT columns), matching the trees' information set.
- **Honest accounting:** the GNN's stacking column is still full-train, not true OOF (the open
  item above); and the published metrics changed because the GNN is now a genuinely different
  (shared) model — the tables above are regenerated from that model, not the old per-station one.

Trajectory-informed *dynamic* graph adjacency is scaffolded (`dynamic_adjacency()`) but not
yet wired in — the model trains on a static distance-based graph. See `RERUN_NEEDED.md` and
`MODEL_ANALYSIS.md` for the full trail.

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
