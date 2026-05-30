# Weather Forecasting Using PSO Feature Selection and Random Forest

**Kwara State University, Malete**
**Faculty of Information and Communication Technology**
**Department of Computer Science**

---

## Project Overview

This project implements a weather forecasting system that combines **Particle Swarm Optimisation (PSO)** for feature selection and hyperparameter tuning with a **Random Forest** regression model to predict next-day precipitation. The system is applied to historical meteorological data for all 36 Nigerian states and the Federal Capital Territory (FCT), downloaded from NASA's POWER API (2000–2025).

The project produces **24 figures** and **11 tables** covering data analysis, model training, evaluation, comparison with baseline models (ARIMA and plain Random Forest), and 14-day future forecasts for every Nigerian state.

---

## Version History

| Version | Date       | Author         | Changes |
|---------|------------|----------------|---------|
| 1.0.0   | 2024-01-01 | Project Team   | Initial release — Nigeria dataset, PSO-RF pipeline |
| 1.1.0   | 2024-03-01 | Project Team   | Added step6 future forecast figures (fig21–fig24) |
| 1.2.0   | 2024-05-01 | Project Team   | Restructured config.py — all variables in one place |
| 1.3.0   | 2025-01-01 | Project Team   | Added install_dependencies.py, extended to 2025 data |
| 1.4.0   | 2025-05-01 | Project Team   | Country-agnostic: STATES list moved to config.py |

**Current Version: 1.4.0**

---

## Quick Start

```bash
# Run everything with one command
python main.py
```

`main.py` handles everything automatically:
1. Installs all required Python libraries
2. Downloads Nigeria weather data from NASA POWER (skipped if already exists)
3. Runs the full 6-step pipeline

---

## Prerequisites

- Python 3.8 or higher
- Internet connection (for first-time data download)
- ~500 MB disk space (for full Nigeria dataset)

No manual library installation needed — `install_dependencies.py` handles it automatically.

---

## Project Structure

```
project/
|
|-- main.py                      <- Run this. Does everything.
|-- config.py                    <- All settings. Edit this only.
|-- install_dependencies.py      <- Installs all Python libraries
|-- download_weather_data.py     <- Downloads NASA POWER data
|-- run_all_scripts.py           <- Runs steps 1-6 in sequence
|-- README.md                    <- This file
|
|-- scripts/
|   |-- step1_load_preprocess.py     <- Load, clean, lag features
|   |-- step2_pso_feature_selection.py <- PSO selects best features
|   |-- step3_pso_hp_tuning_train.py <- PSO tunes RF, trains models
|   |-- step4_evaluation.py          <- MAE, RMSE, R2, ARIMA comparison
|   |-- step5_nigeria_forecast.py    <- Per-state RF forecast
|   `-- step6_future_forecast.py     <- 14-day forecast + hourly
|
|-- nigeria_weather_data/        <- Created by download script
|   |-- all_states_combined.csv
|   |-- training_2000_2022.csv
|   |-- testing_2023_2026.csv
|   |-- by_state/
|   `-- by_year/
|
|-- figures/                     <- Auto-generated (24 PNG files)
|-- tables/                      <- Auto-generated (11 CSV files)
|-- results/                     <- Auto-generated (predictions, masks)
`-- models/                      <- Auto-generated (trained .pkl files)
```

---

## Configuration (config.py)

**This is the only file you need to edit.**
All scripts read every setting from here automatically.

### Key Settings

| Variable | Default | Description |
|---|---|---|
| `COUNTRY_NAME` | `"Nigeria"` | Country name used in figure titles |
| `FORECAST_MODE` | `"all"` | `"all"` = all states, `"single"` = one state |
| `SINGLE_STATE` | `"Kwara"` | State to use when mode is `"single"` |
| `PRIMARY_STATE` | `"Kwara"` | State used to train the main PSO-RF model |
| `TARGET_VARIABLE` | `"precipitation"` | What to predict (`"temp_mean"`, `"temp_max"`) |
| `TRAIN_YEARS_END` | `2022` | Last year of training data |
| `TEST_YEARS_START` | `2023` | First year of test data |
| `RAIN_THRESHOLD_MM` | `1.0` | mm/day cutoff for "Rain Likely" |
| `PSO_N_PARTICLES_FS` | `10` | PSO particles for feature selection |
| `PSO_N_ITERATIONS_FS` | `15` | PSO iterations for feature selection |
| `PSO_N_PARTICLES_HP` | `10` | PSO particles for hyperparameter tuning |
| `PSO_N_ITERATIONS_HP` | `20` | PSO iterations for hyperparameter tuning |
| `FIG_DPI` | `150` | Figure resolution |

### Adding or Changing States

Edit the `STATES` list in `config.py`:

```python
STATES = [
    # (state_key, state_name, capital_city, latitude, longitude)
    ("kwara", "Kwara", "Ilorin", 8.5000, 4.5500),
    ("lagos", "Lagos", "Ikeja",  6.5244, 3.3792),
    # Add more states here ...
]
```

### Switching to a Different Country

1. Update `COUNTRY_NAME` in `config.py`
2. Update `DATA_FOLDER_NAME` in `config.py`
3. Replace the `STATES` list with your country's regions and coordinates
4. Delete the old `nigeria_weather_data/` folder
5. Run `python main.py` — data downloads automatically

---

## Pipeline Steps

| Step | Script | Output |
|---|---|---|
| 1 | `step1_load_preprocess.py` | fig1–fig5, table1–table4, preprocessed_data.csv |
| 2 | `step2_pso_feature_selection.py` | fig6–fig7, table5 |
| 3 | `step3_pso_hp_tuning_train.py` | fig8, table6, trained models (.pkl) |
| 4 | `step4_evaluation.py` | fig9–fig15, table7–table9, all_predictions.csv |
| 5 | `step5_nigeria_forecast.py` | fig16–fig20, table10 |
| 6 | `step6_future_forecast.py` | fig21–fig24, table11 |

---

## Figures Generated (24 total)

| Figure | Description |
|---|---|
| fig1 | Feature distributions histogram |
| fig2 | Feature correlation heatmap |
| fig3 | Average monthly precipitation |
| fig4 | Yearly precipitation trend (2000–2025) |
| fig5 | Seasonal precipitation boxplot |
| fig6 | PSO feature selection convergence curve |
| fig7 | PSO feature weights (selected vs dropped) |
| fig8 | PSO hyperparameter tuning convergence |
| fig9 | Actual vs PSO-RF predicted precipitation |
| fig10 | Three-model comparison (PSO-RF, Base-RF, ARIMA) |
| fig11 | Metrics bar chart (MAE, RMSE, R2, Accuracy, MAPE) |
| fig12 | Scatter plots — actual vs predicted (all models) |
| fig13 | Residual error analysis |
| fig14 | Learning curve |
| fig15 | Seasonal performance comparison |
| fig16 | Feature importance (Gini) |
| fig17 | Next-day precipitation forecast by state |
| fig18 | R2 score per Nigerian state |
| fig19 | Nigeria forecast map (geographic scatter) |
| fig20 | MAE per Nigerian state |
| fig21 | Tomorrow's forecast |
| fig22 | 7-day forecast (this week) |
| fig23 | 7-day forecast (next week, days 8–14) |
| fig24 | Estimated hourly precipitation today |

---

## Tables Generated (11 total)

| Table | Description |
|---|---|
| table1 | Summary statistics of all features |
| table2 | Missing values report |
| table3 | Feature correlation matrix |
| table4 | Seasonal statistics |
| table5 | PSO feature selection weights |
| table6 | PSO-tuned vs baseline hyperparameters |
| table7 | Model comparison (MAE, RMSE, R2, MSE, MAPE) |
| table8 | Literature comparison with published papers |
| table9 | Seasonal performance breakdown |
| table10 | Per-state R2, MAE, RMSE and next-day forecast |
| table11 | Full 14-day structured forecast (all states) |

---

## Dependencies

All installed automatically by `install_dependencies.py`:

| Library | Version | Purpose |
|---|---|---|
| pandas | >=1.3.0 | Data loading and manipulation |
| numpy | >=1.21.0 | Numerical computation |
| matplotlib | >=3.4.0 | All figures and plots |
| seaborn | >=0.11.0 | Heatmaps and statistical plots |
| scikit-learn | >=0.24.0 | Random Forest, metrics, scaling |
| pyswarms | >=1.3.0 | Particle Swarm Optimisation |
| statsmodels | >=0.12.0 | ARIMA baseline model |
| joblib | >=1.0.0 | Model saving and loading |
| requests | >=2.25.0 | NASA POWER API download |
| tqdm | >=4.60.0 | Progress bars during download |

---

## Data Source

**NASA POWER (Prediction of Worldwide Energy Resources)**
- URL: https://power.larc.nasa.gov
- API: Temporal Daily Point
- Model: MERRA-2 (Modern-Era Retrospective Analysis)
- Period: 2000–2025
- Coverage: All 36 Nigerian states + FCT (37 locations)

Variables downloaded per location:
- Temperature (mean, min, max) at 2m
- Relative humidity at 2m
- Precipitation (bias-corrected)
- Wind speed and gust at 2m
- Cloud cover
- Surface pressure
- Global solar radiation
- Photosynthetically active radiation (sunshine proxy)

---

## Methodology Summary

1. **Data Collection** — NASA POWER API, daily data, 37 Nigerian states
2. **Preprocessing** — Missing value imputation (column mean), lag-1 features
3. **PSO Feature Selection** — GlobalBestPSO selects best subset of 20 candidate features
4. **PSO Hyperparameter Tuning** — PSO optimises `n_estimators` and `max_depth` for Random Forest
5. **Model Training** — PSO-RF trained on 2000–2022, tested on 2023–2025
6. **Evaluation** — MAE, RMSE, MSE, R2, Accuracy, MAPE vs Baseline RF and ARIMA(2,1,2)
7. **Forecasting** — Recursive 14-day forecast per state, hourly distribution using West Africa climatological pattern

---

## Note on Hourly Estimates (fig24)

NASA POWER provides daily data only. Hourly estimates in fig24 are derived by distributing the predicted daily total using West Africa's climatological rainfall timing pattern (peak in the afternoon and early evening, low overnight). This is a standard approach in meteorological research and should be clearly stated in any academic write-up based on this project.

---

## .gitignore Recommendation

Add this `.gitignore` file to avoid committing auto-generated files:

```
figures/
tables/
results/
models/
nigeria_weather_data/
__pycache__/
*.pyc
*.pkl
*.npy
```

---

## Citation

If you use this project in your research, please cite:

> Kwara State University, Malete. *Weather Forecasting Using PSO Feature Selection and Random Forest*. Faculty of ICT, Department of Computer Science.

---

*README version 1.4.0*