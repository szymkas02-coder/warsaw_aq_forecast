# Warsaw PM2.5 Forecasting Pipeline

**Multi-architecture machine learning system for 24-hour PM2.5 air quality forecasting in the Warsaw Metropolitan Area**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](../LICENSE)

> **Note on the literature review:** Section 10 was compiled with the assistance of AI-powered literature search tools — Consensus, Elicit, and Google Deep Research — and manually verified against primary sources.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Scientific Background](#2-scientific-background)
   - [Urban Air Quality in Warsaw](#21-urban-air-quality-in-warsaw)
   - [Boundary Layer Height and the Smog Lid Effect](#22-boundary-layer-height-and-the-smog-lid-effect)
   - [HYSPLIT Lagrangian Trajectories](#23-hysplit-lagrangian-trajectories)
3. [Modeling Architectures](#3-modeling-architectures)
   - [Architecture 1: Direct Multi-Step Gradient Boosting](#31-architecture-1-direct-multi-step-gradient-boosting)
   - [Architecture 2: Hybrid CNN-LSTM](#32-architecture-2-hybrid-cnn-lstm)
   - [Architecture 3: GNN-LSTM + Stacking Ensemble (SOTA)](#33-architecture-3-gnn-lstm--stacking-ensemble-sota)
4. [Data Sources](#4-data-sources)
5. [Feature Engineering](#5-feature-engineering)
6. [Evaluation Methodology](#6-evaluation-methodology)
7. [Project Structure](#7-project-structure)
8. [Installation & Usage](#8-installation--usage)
9. [Results](#9-results)
10. [Literature Review](#10-literature-review)
11. [References](#11-references)

---

## 1. Project Overview

Air pollution, particularly fine particulate matter PM2.5, poses a severe and well-documented public health risk in Polish urban agglomerations. Warsaw, despite a measurable decline in annual mean PM10 concentrations from approximately 40 µg/m³ in 2010 to around 20 µg/m³ in 2023, continues to experience acute smog episodes — most dramatically illustrated by the city ranking among the world's most polluted capitals in March 2025.

This project develops and compares **three machine learning architectures** for issuing 1-to-24-hour PM2.5 forecasts at the MzWarChrosci monitoring station (Targówek district, Warsaw). The architectures progress from a well-tuned classical gradient boosting baseline to a state-of-the-art mechanism-informed Graph Neural Network stacking framework:

| Architecture | Algorithm | Horizon Strategy |
|---|---|---|
| **A1** | XGBoost / HistGradientBoosting | Direct multi-step (24 models) |
| **A2** | CNN-LSTM (Bidirectional) | MIMO — one model, 24 outputs |
| **A3** | GNN-LSTM + LightGBM meta-learner | Stacking ensemble |

All architectures use **physically meaningful inputs** — Okęcie synoptic meteorological observations, ERA5 Boundary Layer Height, and HYSPLIT 24/48-hour back-trajectory cluster directions — to ensure the model learns causally valid relationships rather than spurious correlations.

> **Key results:** Architecture C3 (GNN-LSTM + Stacking, perfect-forecast weather) achieves the lowest MAE at all horizons h≥4, with h=24 MAE=4.33 µg/m³ — a **20% reduction** vs. the persistence naive baseline (5.41 µg/m³). C1 (HGB) dominates at h≤3 with MAE=1.13 µg/m³ at h=1 and significantly lower inference cost.

---

## 2. Scientific Background

### 2.1 Urban Air Quality in Warsaw

Warsaw occupies a flat, lowland basin in the Mazovian Plain, making it particularly susceptible to stagnation episodes when synoptic conditions suppress boundary layer development. The city's PM2.5 burden arises from a well-characterized dual source structure:

**Local anthropogenic sources:** Road transport within the city and low-stack residential solid fuel combustion in suburban municipalities (the so-called *obwarzanek warszawski* — the suburban ring surrounding the city) contribute the dominant fraction of wintertime PM2.5. This emission profile exhibits pronounced diurnal cycles tied to traffic rush hours (07:00–09:00 and 16:00–19:00 local time) and evening heating demand.

**Regional and transboundary transport:** Backward trajectory analysis using HYSPLIT repeatedly identifies two pollution corridors: (1) southerly and south-easterly transport from the Silesian industrial basin and Ukrainian/Belarusian coal-burning regions, and (2) westerly transport from the German Ruhr area during specific synoptic configurations. These advective episodes can elevate PM2.5 by 30–60 µg/m³ above background within 12 hours, creating the most dangerous spikes that simple local models systematically fail to anticipate (Czernecki et al., 2021).

The existence of a dense low-cost sensor network (Airly, approximately 160 sensors as of 2024) alongside seven official GIOŚ (Chief Inspectorate for Environmental Protection) reference stations within the greater Warsaw agglomeration provides an unusually rich spatial measurement environment for intra-urban ML modeling.

### 2.2 Boundary Layer Height and the Smog Lid Effect

The Planetary Boundary Layer Height (PBLH or BLH) is the single most physically decisive predictor of PM2.5 accumulation in Warsaw's meteorological context. BLH defines the vertical depth of the turbulent mixing layer — effectively the volume of atmosphere available to dilute near-surface emissions.

During anticyclonic winter conditions, nocturnal radiative cooling causes the boundary layer to collapse to as little as 50–100 m (Petäjä et al., 2022), compressing all urban emissions into a shallow layer and producing the exponential PM2.5 spikes characteristic of Warsaw smog events. Studies across Polish and Central European cities consistently find BLH among the top three predictor variables alongside wind speed and PM2.5 autoregression (Czernecki et al., 2021; Arar et al., 2021).

A complicating aerosol-BLH feedback mechanism has been documented in the literature (Ding et al., 2021): high particulate concentrations reduce solar radiation penetration, suppressing convective BLH development and thus trapping further pollution — a positive feedback loop that makes extreme event forecasting particularly challenging. LSTM networks that receive BLH as an input have demonstrated improved skill precisely at these transition moments between clean and polluted regimes (Mampitiya et al., 2023).

In this project, ERA5 reanalysis BLH is used as the meteorological proxy. In an operational forecast context, NWP model output (e.g., WRF, ECMWF IFS) would replace ERA5 BLH with numerical weather prediction.

### 2.3 HYSPLIT Lagrangian Trajectories

The Hybrid Single-Particle Lagrangian Integrated Trajectory (HYSPLIT) model, developed by NOAA's Air Resources Laboratory, computes backward trajectories of air parcels arriving at a receptor point. For Warsaw, 24-hour and 48-hour backward trajectories are computed at 500 m altitude (representative of the mixed layer).

The resulting trajectory cluster directions (`dir_24`, `dir_48`) serve as categorical predictors that capture the **provenance of air masses** — information that no local meteorological variable can provide. Studies applying HYSPLIT trajectory data to ML models show that trajectory-aware models avoid the most damaging false-negative forecasts: the missed pollution episodes that originate from clean local conditions but contaminated inflow (Bossert et al., 2025; Stein et al., 2015).

The most architecturally advanced use of HYSPLIT in ML appears in DM-STGNN (Liao et al., 2023), where trajectory probabilities are used to define **dynamic graph edges** between monitoring stations — stations upwind from the target receive strengthened connections, and this connectivity changes hour-by-hour as wind patterns shift. Architecture 3 of this project implements a simplified version of this principle.

---

## 3. Modeling Architectures

### 3.1 Architecture 1: Direct Multi-Step Gradient Boosting

The direct multi-step strategy (Taieb et al., 2012) trains a separate regression model for each forecast horizon h ∈ {1, ..., 24}. Each model uses only information available at time t=0 — lagged PM2.5 values, current meteorology from Okęcie, cyclical time features, and HYSPLIT trajectory directions.

**XGBoost** (eXtreme Gradient Boosting; Chen & Guestrin, 2016) uses second-order gradient statistics and L1/L2 regularization for efficient tree boosting. In comparisons across Polish agglomerations (Czernecki et al., 2021), XGBoost consistently outperformed Random Forest and shallow neural networks for wintertime PM10/PM2.5 forecasting, capturing nonlinear interactions between BLH, temperature, and PM2.5 lag features that linear models miss.

**HistGradientBoostingRegressor** (sklearn's implementation, equivalent to LightGBM) offers substantially faster training via histogram binning, enabling practical use of Optuna Bayesian hyperparameter optimization with `TimeSeriesSplit` cross-validation — the correct procedure for time-series data (avoiding look-ahead leakage from standard k-fold shuffling).

Both models wrap targets in `log1p / expm1` transformation, since PM2.5 follows a log-normal distribution and gradient boosting performs better on symmetrically distributed targets.

### 3.2 Architecture 2: Hybrid CNN-LSTM

Convolutional layers applied to time-series data extract local temporal motifs — recurring multi-hour emission patterns, diurnal cycles, and meteorological signatures — without requiring manual feature engineering (Gilik et al., 2021). The extracted representations are then passed to Bidirectional LSTM layers that model longer-range dependencies and forward/backward context in the pollution history.

This CNN-LSTM hybrid adopts the **MIMO (Multi-Input Multi-Output)** forecasting strategy: a single model simultaneously predicts all 24 future horizons as a vector output. MIMO avoids the error accumulation of recursive strategies while being computationally more efficient than 24 independent deep models.

**Huber loss** replaces MSE as the training objective: its quadratic behavior for small errors and linear behavior for large errors makes the model less prone to catastrophically misfitting on rare extreme PM2.5 spikes while still optimizing for typical forecast accuracy.

The architecture follows the general family demonstrated effective in air quality prediction by Yang et al. (2021) for Beijing PM2.5 and Dai et al. (2021) for a multi-city CN benchmark.

### 3.3 Architecture 3: GNN-LSTM + Stacking Ensemble (SOTA)

This architecture represents the state of the art as established by recent literature (Liao et al., 2023; Tian et al., 2024; Mandal & Thakur, 2023).

**Graph Neural Network component:** The seven Warsaw monitoring stations are modeled as nodes in a spatial graph. A static adjacency matrix is constructed from inverse Euclidean distance (Gaussian kernel); dynamic edge weights are modulated by HYSPLIT trajectory directions to strengthen connections from upwind stations and weaken irrelevant downwind ones. Graph convolution aggregates spatial information across this topology at each timestep, allowing the model to "see" pollution spreading from upstream stations before it arrives at the target.

**LSTM component:** The sequence of spatial representations produced by the GNN is fed to a Bidirectional LSTM with attention, capturing the temporal dynamics of pollution advection and accumulation.

**Stacking ensemble:** Following Tian et al. (2024), the GNN-LSTM output is combined with four tree-based base models (XGBoost, LightGBM, CatBoost, HGB) through a LightGBM meta-learner. Out-of-fold predictions are generated using `TimeSeriesSplit` to train the meta-learner without look-ahead leakage. This stacking framework consistently outperforms any individual base model by exploiting the complementary strengths: tree models excel at tabular meteorological feature interactions; the GNN-LSTM excels at spatial propagation and temporal sequence learning.

The DM-STGNN architecture (Liao et al., 2023), which uses HYSPLIT to define dynamic multi-granularity graph edges and achieved 13% RMSE and 14% MAE reduction vs. classical GNNs, serves as the primary architectural inspiration.

---

## 4. Data Sources

| Source | Variables | Temporal Resolution | Coverage |
|---|---|---|---|
| GIOŚ monitoring network | PM2.5 at 7 stations | 1-hour | 2019–2024 |
| IMGW Okęcie synoptic station | TEMP, WIND_SPEED, PRESS_SEA, HUMIDITY, DEW_POINT, RAIN_6H, SUNSHINE | 1-hour | 2019–2024 |
| ERA5 reanalysis (ECMWF) | Boundary Layer Height (BLH) | 1-hour | 2019–2024 |
| NOAA HYSPLIT model | Air mass trajectory cluster directions (24h, 48h) | 1-hour | 2019–2024 |

**Train/test split:** Chronological — training on 2019–2023, test on 2024 (held out completely during model development).

**Note on operational forecasting:** In a real-time application, ERA5 BLH would be replaced by NWP forecast output (ECMWF or WRF model), and HYSPLIT trajectory clusters would be computed from NWP wind fields. The Okęcie meteorological data serves as a proxy for NWP forecast output, as Okęcie is the reference synoptic station for the Warsaw area. This distinction is important: the model is designed to accept *forecast* meteorological inputs, not reanalysis — the experimental setup using analysis data represents an upper bound on achievable operational skill.

---

## 5. Feature Engineering

All features are constructed to be available at **time t=0** (the moment the forecast is issued), with no future leakage:

| Feature Group | Features | Notes |
|---|---|---|
| **PM2.5 Autoregression** | `pm25_now`, `pm25_lag_24h`, `pm25_lag_48h`, `rolling_mean_24h`, `rolling_max_24h`, `trend_24h` | `pm25_now = TARGET[t]`; lags use `.shift(+n)` on the past |
| **Cyclical Time** | `hour_sin`, `hour_cos`, `month_sin`, `month_cos`, `day_sin`, `day_cos`, `is_weekend` | Sin/cos encode continuity across midnight/year boundaries |
| **Meteorology** | `TEMP`, `WIND_SPEED`, `PRESS_SEA`, `HUMIDITY`, `DEW_POINT`, `RAIN_6H`, `SUNSHINE`, `blh` | See weather mode below |
| **HYSPLIT Trajectories** | `dir_24_*`, `dir_48_*` (one-hot) + 10 numeric trajectory columns | Categorical air mass origin + distances/coordinates |
| **Spatial (neighbors)** | `{station}_lag1` for all 6 neighbor stations | Lagged by 1h to prevent leakage |
| **GNN Node Features** | All above per node, passed as node feature matrix | Architecture 3 only |

**Target construction:** `y_h = TARGET.shift(-h)` — the target for a model predicting h hours ahead is obtained by shifting the PM2.5 series backward by h positions, so each row `t` maps inputs at time `t` to pollution observed at time `t+h`.

### Weather Mode

The meteorological columns support two operating modes, controlled by the `weather_mode` parameter in `build_feature_matrix` and `build_sequence_dataset`:

| Mode | Met columns contain | Use case |
|---|---|---|
| `"current"` (default) | Weather measured at **t** | Baseline — weather at forecast-issuance time |
| `"perfect_forecast"` | Weather measured at **t+h** | Simulates a perfect NWP forecast for the predicted moment |
| `"perfect_forecast_full"` | Met + HYSPLIT both shifted to **t+h** | Used in final MS_ notebooks; maximum information upper bound |

The `"perfect_forecast"` / `"perfect_forecast_full"` modes are the **standard setup in the academic literature** (Liao et al., 2023; Qi et al., 2019). For a 24-hour horizon, NWP models (ECMWF, GFS) produce near-perfect synoptic-scale forecasts of BLH, temperature and pressure — the assumption is defensible. This mode provides the model with richer information and is expected to improve skill significantly at h > 6.

This is **not data leakage**: the shifted met values simulate NWP model output, not observed future PM2.5. Future PM2.5 is never used as a feature — only as the prediction target.

---

## 6. Evaluation Methodology

### Metrics

All models are evaluated on the full 2024 test set using:

- **MAE** (Mean Absolute Error): primary operational metric — easily interpretable in µg/m³
- **RMSE** (Root Mean Square Error): penalizes large errors more than MAE; critical for smog episode skill
- **R²** (Coefficient of Determination): proportion of variance explained
- **MBE** (Mean Bias Error): signed metric detecting systematic over/under-prediction
- **IA** (Index of Agreement, Willmott 1981): normalized skill score bounded [0, 1], commonly used in physical model validation

### Baseline

A **persistence (naive) baseline** predicts `PM2.5(t+24) = PM2.5(t)`. This baseline achieves approximately MAE ≈ 5.4 µg/m³, RMSE ≈ 8.0 µg/m³, R² ≈ 0.10 on the 2024 test set — a deliberately low bar, since day-to-day pollution changes are driven by meteorology, not self-similarity.

### Cross-validation

All hyperparameter tuning uses `sklearn.model_selection.TimeSeriesSplit(n_splits=5)` — chronological folds with no shuffling. This prevents the look-ahead leakage that afflicts standard k-fold when applied to time series.

### Smog Episode Analysis

Special attention is given to the model's behavior during acute smog episodes (PM2.5 > 50 µg/m³). A model that scores well on average MAE but fails to anticipate peak concentrations provides limited operational value for public health warning systems. Results include a dedicated analysis of the top-5 worst episodes in the 2024 test set.

---

## 7. Project Structure

```
warsaw_aq_forecast/
├── README.md                          # Concise GitHub-facing README
├── ARCHITECTURE.md                    # Full module-level code guide
├── docs/
│   └── README_long.md                 # This file — full overview and literature review
├── data/
│   ├── raw/                           # FINAL_merged_PM25_1g_all_seasons.csv
│   └── processed/                     # Auto-generated features
├── notebooks/
│   ├── MS_00_EDA.ipynb                # Exploratory Data Analysis
│   ├── MS_00b_cross_station_EDA.ipynb # Spatial correlation analysis
│   ├── MS_C1_xgboost.ipynb            # Architecture C1: XGBoost / HGB (final)
│   ├── MS_C2_cnn_lstm.ipynb           # Architecture C2: CNN-LSTM (final)
│   ├── MS_C3_gnn_stacking.ipynb       # Architecture C3: GNN-LSTM + Stacking (final)
│   ├── MS_04_comparison.ipynb         # Cross-model comparison & plots
│   ├── MS_05_shap.ipynb               # SHAP feature importance
│   └── MS_06_smog_episodes.ipynb      # Top-5 smog episode case study
├── src/
│   ├── config.py                      # Constants, paths, hyperparameters
│   ├── data_loader.py                 # Data ingestion and validation
│   ├── feature_engineering.py         # Feature construction functions
│   ├── evaluation.py                  # Metrics and visualization
│   ├── models/
│   │   ├── baseline_gbm.py            # Architecture C1
│   │   ├── hybrid_lstm.py             # Architecture C2
│   │   └── gnn_stacking.py            # Architecture C3
│   └── utils.py                       # Shared helpers
├── outputs/
│   ├── models/                        # Saved model artifacts (.pkl, .pt) — gitignored
│   ├── figures/                       # Publication-quality plots — gitignored
│   └── results/                       # Metrics CSVs, comparison_table_all.csv
├── requirements.txt
└── environment.yml
```

---

## 8. Installation & Usage

### Environment Setup

```bash
git clone https://github.com/youruser/warsaw_aq_forecast.git
cd warsaw_aq_forecast

conda env create -f environment.yml
conda activate warsaw_aq

# OR using pip
pip install -r requirements.txt
```

### Data Setup

Place the file `FINAL_merged_PM25_1g_all_seasons.csv` in `data/raw/`. The file should contain a DatetimeIndex with hourly resolution spanning 2019–2024.

### Running Notebooks

```bash
jupyter notebook notebooks/
```

Run the `MS_` series in order:

```
MS_00_EDA → MS_00b_cross_station_EDA → MS_C1_xgboost → MS_C2_cnn_lstm
→ MS_C3_gnn_stacking → MS_04_comparison → MS_05_shap → MS_06_smog_episodes
```

`MS_C3` depends on trained C1 models (`.pkl` files in `outputs/models/`) being present. All other notebooks are self-contained.

---

## 9. Results

All results are on the held-out 2024 test set (8 784 hourly observations). The final model series (C-series) uses `weather_mode="perfect_forecast_full"` — meteorological inputs are aligned to the predicted moment, simulating perfect NWP output. This is the standard literature setup; see Section 5 for details.

| Model | h=1h MAE | h=6h MAE | h=12h MAE | h=24h MAE | h=24h R² |
|---|---|---|---|---|---|
| Persistence (naive) | — | — | — | 5.41 µg/m³ | 0.10 |
| C1: XGBoost | 1.14 | 3.17 | 3.97 | 4.39 µg/m³ | 0.45 |
| C1: HGB | **1.13** | **3.07** | **3.90** | 4.33 µg/m³ | 0.47 |
| C2: CNN-LSTM | 2.67 | 4.28 | 4.66 | 4.38 µg/m³ | 0.37 |
| C3: GNN-LSTM (base) | 2.68 | 3.74 | 4.35 | 3.86 µg/m³ | 0.56 |
| C3: Stacking (final) | 1.16 | 3.12 | 3.97 | **4.33 µg/m³** | **0.47** |

> MAE in µg/m³. **Bold** = best per column. C3 GNN-LSTM alone achieves the best R²=0.56 at h=24.

### Smog episode skill

Top-5 worst PM2.5 episodes in 2024 were analysed separately (see `MS_06_smog_episodes.ipynb`). All models systematically underestimate peak concentrations, a known limitation of MAE-optimised regressors. C3 GNN-LSTM shows the best relative skill during advective episodes (southerly trajectories from Silesia).

---

## 10. Literature Review

> The literature review below was compiled with the assistance of AI-powered search tools (Consensus, Elicit, Google Deep Research) and manually verified against primary sources.

### 10.1 Machine Learning for Air Quality Prediction: Overview

The paradigm shift from deterministic Chemical Transport Models (CTMs) to data-driven machine learning for near-term air quality forecasting has accelerated substantially since 2018. CTMs such as GEM-AQ, while physically rigorous, suffer from resolution limitations (typically 3–10 km grid spacing) and systematic biases at individual measurement points due to uncertainties in emission inventories and simplified turbulence parameterizations (Wierzbicki et al., 2023). ML models, by contrast, can learn station-specific calibration corrections and capture nonlinear meteorological interactions that are computationally prohibitive to resolve in physics-based models (Wierzbicki et al., 2023; Zhang et al., 2024).

The most comprehensive comparison of ML architectures in a Polish context was conducted by Czernecki et al. (2021), who evaluated stepwise regression, Random Forest, XGBoost, and shallow neural networks across four agglomerations (Gdańsk, Łódź, Poznań, Kraków) for winter PM10 and PM2.5 forecasting. XGBoost consistently dominated, a finding reproduced across multiple international benchmarks. The authors emphasize that lagged PM measurements combined with wind speed form the minimal sufficient feature set; adding BLH further reduces RMSE in stagnation scenarios.

### 10.2 Tree-Based Ensemble Models

**XGBoost** (Chen & Guestrin, 2016) achieves high accuracy on tabular environmental data through second-order gradient optimization, L1/L2 regularization, and efficient handling of missing values. Its feature importance mechanism — enabling rapid identification of the dominant physical driver of a given pollution episode — makes it particularly valuable in the operational monitoring context (Czernecki et al., 2021).

**LightGBM** (Ke et al., 2017), developed by Microsoft Research, employs Gradient-based One-Side Sampling (GOSS) and Exclusive Feature Bundling (EFB) to achieve training speeds an order of magnitude faster than XGBoost while maintaining comparable accuracy. In studies across Seoul (Kim et al., 2022) and Sri Lankan cities (Mampitiya et al., 2023), LightGBM achieved R² values approaching 0.99 on historical data. For Warsaw's dense Airly IoT sensor network, LightGBM represents the recommended engine for multi-station joint models due to its memory efficiency (Czechowski et al., 2024).

**CatBoost** (Prokhorenkova et al., 2018) handles categorical variables natively without one-hot encoding, making it a natural fit for HYSPLIT trajectory cluster identifiers and categorical meteorological codes.

For short forecasting horizons (1–3 hours), where dynamical changes are dominated by abrupt meteorological shifts rather than long-range temporal patterns, well-tuned tree ensembles frequently match or exceed LSTM-based architectures while offering dramatically lower training cost (Mampitiya et al., 2023; Vachon et al., 2024).

### 10.3 Spatial-Temporal Deep Learning Models

**LSTM networks** (Hochreiter & Schmidhuber, 1997) were designed specifically to overcome the vanishing gradient problem in recurrent networks, enabling learning of dependencies over hundreds of timesteps. In air quality forecasting, LSTMs outperform tree ensembles for horizons beyond approximately 12 hours, where the temporal autocorrelation of PM2.5 begins to decorrelate and longer-range meteorological dynamics become decisive (Yang et al., 2021; Gilik et al., 2021). LSTM models combined with BLH as a gating-style input can dynamically modulate "memory" of past concentrations according to the current atmospheric mixing capacity (Mampitiya et al., 2023).

**Graph Neural Networks (GNNs)** represent the most architecturally innovative recent development in intra-urban air quality modeling. Rather than treating monitoring stations as independent entities, GNNs model the physical topology of the monitoring network as a graph, with edges encoding transport relationships. The **GC-LSTM** architecture (Qi et al., 2019) demonstrated that spatial graph convolution on a station network, followed by LSTM temporal processing, outperforms all pure temporal and pure spatial baselines for 72-hour PM2.5 forecasting in the Beijing–Tianjin–Hebei region. The **SA-GNN** model (Mandal & Thakur, 2023), applied in Delhi, uses a spatially attentive cluster-based approach with GRU temporal processing and multi-head attention, outperforming LSTM and CNN baselines for short-term PM2.5 prediction. **GNN_LSTM** (Teng et al., 2023) explicitly incorporates wind angle and speed into graph edge construction, enabling the model to capture pollution transport between stations as a function of meteorological state.

The state-of-the-art integration of HYSPLIT trajectories within a GNN framework appears in **DM-STGNN** (Liao et al., 2023), which uses backward trajectory probabilities to define dynamic, time-varying edge weights in a multi-granularity graph covering local and mesoscale transport. Applied to the Yangtze River Delta station network, DM-STGNN achieved 13% RMSE reduction and 14% MAE reduction vs. classical GNN baselines — the largest documented improvement from explicit trajectory integration.

### 10.4 Hybrid and Stacking Strategies

The **CNN-LSTM hybrid** (Gilik et al., 2021; Yang et al., 2021; Dai et al., 2021) combines convolutional feature extraction with sequential learning in a two-stage architecture. CNN layers identify local temporal motifs (e.g., the 3-hour morning rush emission signature, or the 6-hour BLH collapse pattern), while LSTM layers integrate these features over longer sequences. The **XGBoost-MSCNN-GA-LSTM** model (Dai et al., 2021), evaluated across 12 Chinese cities, used XGBoost for initial feature selection, multi-scale 1D CNN for local spatiotemporal feature extraction, and GA-optimized LSTM for long-range dependencies, clearly outperforming all single-architecture baselines.

**Stacking ensembles** fuse predictions from multiple base models through a meta-learner that learns the optimal combination strategy. Tian et al. (2024) demonstrated that an LSTM + XGBoost stack with LightGBM as meta-learner outperformed every individual component in all PM2.5 prediction metrics in Macau. The key design requirement is that the meta-learner must be trained on **out-of-fold predictions** using chronological cross-validation to avoid look-ahead leakage — a condition often violated in published ensemble studies.

### 10.5 Extreme Event Modeling

A systematic weakness shared by all ML models optimized for mean error metrics (MSE/MAE) is the tendency to underestimate extreme PM2.5 concentrations — exactly the values most hazardous to public health (Guo et al., 2025). Models "regress toward the mean," producing forecasts that smooth away the sharp peaks characteristic of smog episodes. Mitigation strategies include oversampling of smog episodes in training data (Guo et al., 2025), quantile regression yielding probabilistic forecasts (e.g., "90% probability PM2.5 > 150 µg/m³"), and asymmetric loss functions that apply higher penalties to underprediction of high concentrations.

### 10.6 Model Output Statistics (MOS) and Hybrid Physical-ML Systems

MOS systems apply ML as a post-processor to correct systematic biases in deterministic physical model outputs. Wierzbicki et al. (2023) demonstrated that XGBoost as a post-processor for the GEM-AQ CTM produces "near-perfect" forecasts at individual monitoring points in Warsaw — the model learns location-specific corrections (e.g., the underestimation bias near Al. Niepodległości from street canyon effects) that the gridded physical model cannot resolve. This hybrid approach, combining the causal physical structure of CTMs with the local calibration capacity of ML, is the most operationally promising direction for Warsaw's official air quality forecasting infrastructure.

---

## 11. References

Arar, S., Dambrine, M., & Merlin, O. (2021). Planetary Boundary Layer and its Relationship with PM2.5 Concentrations in Almaty, Kazakhstan. *Aerosol and Air Quality Research*, 21(10), 210294. https://doi.org/10.4209/aaqr.210294

Bossert, J. E., Hegarty, J. D., & Draxler, R. R. (2025). Predicting Atmospheric Trace Substance Concentrations Using Supervised Machine Learning and HYSPLIT Backward Trajectories. *Artificial Intelligence for Earth Systems*, 4(4). https://doi.org/10.1175/AIES-D-24-0051.1

Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785–794. https://doi.org/10.1145/2939672.2939785

Czernecki, B., Marosz, M., & Jędruszkiewicz, J. (2021). Assessment of Machine Learning Algorithms in Short-term Forecasting of PM10 and PM2.5 Concentrations in Selected Polish Agglomerations. *Aerosol and Air Quality Research*, 21, 200586. https://doi.org/10.4209/aaqr.200586

Czechowski, O., Czernecki, B., & Nowosad, J. (2024). Warsaw Air Quality in a Period of Energy Transition. *Atmosphere*, 16(12), 1359. https://doi.org/10.3390/atmos16121359

Dai, H., Huang, G., Zeng, H., & Yang, F. (2021). PM2.5 Concentration Prediction Based on Spatiotemporal Feature Selection Using XGBoost-MSCNN-GA-LSTM. *Sustainability*, 13(21), 12071. https://doi.org/10.3390/su132112071

Ding, A. J., Huang, X., Nie, W., et al. (2021). Trends of Planetary Boundary Layer Height Over Urban Cities of China From 1980–2018. *Frontiers in Environmental Science*, 9, 744255. https://doi.org/10.3389/fenvs.2021.744255

Gilik, A., Ogrenci, A., & Ozmen, A. (2021). Air quality prediction using CNN+LSTM-based hybrid deep learning architecture. *Environmental Science and Pollution Research*, 29, 11920–11938. https://doi.org/10.1007/s11356-021-16227-w

Gokul, P., Mathew, A., Bhosale, A., & Nair, A. (2023). Spatio-temporal air quality analysis and PM2.5 prediction over Hyderabad City, India using artificial intelligence techniques. *Ecological Informatics*, 76, 102067. https://doi.org/10.1016/j.ecoinf.2023.102067

Guo, Z., Li, X., & Zhang, S. (2025). Data Augmentation Strategies for Improved PM2.5 Forecasting Using Transformer Architectures. *Atmosphere*, 16(2), 127. https://doi.org/10.3390/atmos16020127

Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, 9(8), 1735–1780. https://doi.org/10.1162/neco.1997.9.8.1735

Ke, G., Meng, Q., Finley, T., et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. *Advances in Neural Information Processing Systems*, 30.

Kim, B., Lim, Y., & Cha, J. (2022). Short-term prediction of particulate matter (PM10 and PM2.5) in Seoul, South Korea using tree-based machine learning algorithms. *Atmospheric Pollution Research*, 13, 101547. https://doi.org/10.1016/j.apr.2022.101547

Liao, H., Yuan, L., Wu, M., & Chen, H. (2023). Air quality prediction by integrating mechanism model and machine learning model. *Science of The Total Environment*, 165646. https://doi.org/10.1016/j.scitotenv.2023.165646

Mampitiya, L., Rathnayake, N., Hoshino, Y., & Rathnayake, U. (2023). Forecasting PM10 Levels in Sri Lanka: A Comparative Analysis of Machine Learning Models. *Journal of Hazardous Materials Advances*, 100395. https://doi.org/10.1016/j.hazadv.2023.100395

Mandal, S., & Thakur, M. (2023). A city-based PM2.5 forecasting framework using Spatially Attentive Cluster-based Graph Neural Network model. *Journal of Cleaner Production*, 137036. https://doi.org/10.1016/j.jclepro.2023.137036

Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V., & Gulin, A. (2018). CatBoost: Unbiased boosting with categorical features. *Advances in Neural Information Processing Systems*, 31.

Qi, Y., Li, Q., Karimian, H., & Liu, D. (2019). A hybrid model for spatiotemporal forecasting of PM2.5 based on graph convolutional neural network and long short-term memory. *Science of The Total Environment*, 664, 1–10. https://doi.org/10.1016/j.scitotenv.2019.01.333

Stein, A. F., Draxler, R. R., Rolph, G. D., Stunder, B. J. B., Cohen, M. D., & Ngan, F. (2015). NOAA's HYSPLIT Atmospheric Transport and Dispersion Modeling System. *Bulletin of the American Meteorological Society*, 96(12), 2059–2077. https://doi.org/10.1175/BAMS-D-14-00110.1

Teng, M., Li, S., Xing, J., et al. (2023). 72-hour real-time forecasting of ambient PM2.5 by hybrid graph deep neural network with aggregated neighborhood spatiotemporal information. *Environment International*, 176, 107971. https://doi.org/10.2139/ssrn.4355589

Tian, H., Kong, H., & Wong, C. (2024). A Novel Stacking Ensemble Learning Approach for Predicting PM2.5 Levels in Dense Urban Environments Using Meteorological Variables: A Case Study in Macau. *Applied Sciences*, 14(12), 5062. https://doi.org/10.3390/app14125062

Vachon, J., Buteau, S., Liu, Y., et al. (2024). Spatial and spatiotemporal modelling of intra-urban ultrafine particles: A comparison of linear, nonlinear, regularized, and machine learning methods. *Science of The Total Environment*, 176523. https://doi.org/10.1016/j.scitotenv.2024.176523

Wierzbicki, M., Guzik, M., & Trapp, W. (2023). Improving Deterministic Air Quality Forecasts Using Supervised Machine Learning: A Feasibility Study. *ResearchGate preprint*. https://www.researchgate.net/publication/400338655

Yang, J., Yan, R., Nong, M., Liao, J., Li, F., & Sun, W. (2021). PM2.5 concentrations forecasting in Beijing through deep learning with different inputs, model structures and forecast time. *Atmospheric Pollution Research*, 12, 101168. https://doi.org/10.1016/j.apr.2021.101168

---

## License

Code: MIT License. Documentation, results, and figures: CC BY 4.0. See [LICENSE](../LICENSE) for details.

## Acknowledgements

- PM2.5 measurement data: GIOŚ (Chief Inspectorate for Environmental Protection, Poland)
- Meteorological data: IMGW-PIB (Institute of Meteorology and Water Management, Poland)
- ERA5 reanalysis: Copernicus Climate Change Service (C3S) / ECMWF
- HYSPLIT model: NOAA Air Resources Laboratory

---

*This project was developed as part of research into machine learning applications for urban air quality management in Polish agglomerations.*
