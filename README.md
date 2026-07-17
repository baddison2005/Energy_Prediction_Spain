# Energy Prediction in Spain

Forecasting hourly Spanish solar generation, onshore wind generation, total electricity load, and electricity price with statistical time-series models and gradient-boosted trees.

This is a research and portfolio repository built around four years of Spanish electricity-system and weather data (2015–2018). It develops reproducible exploratory analyses, feature-engineering workflows, ARIMAX/SARIMAX baselines, LightGBM models, walk-forward validation, and rolling 24-hour forecasts. Model performance is compared with the day-ahead forecast fields supplied in the source data.

> **Project status:** experimental. The notebooks contain long-running training cells and saved outputs, and the limitations below should be addressed before interpreting the work as an operational forecasting system.

## Objectives

The project asks whether public electricity and weather data can support useful day-ahead forecasts for four national targets:

| Forecast target | Actual field | Day-ahead benchmark | Typical unit |
|---|---|---|---|
| Solar generation | `generation solar` | `forecast solar day ahead` | MW in the source data |
| Onshore wind generation | `generation wind onshore` | `forecast wind onshore day ahead` | MW in the source data |
| Total electricity load | `total load actual` | `total load forecast` | MW |
| Electricity price | `price actual` | `price day ahead` | EUR/MWh |

The main objectives are to:

- clean and align hourly electricity and city-level weather observations;
- engineer calendar, solar-position, weather, lag, and rolling features without using future target values;
- compare interpretable ARIMAX/SARIMAX models with nonlinear LightGBM models;
- tune and evaluate models with chronological rather than random splits;
- produce 24-hour forecast blocks, including mean and quantile estimates; and
- benchmark model errors against the Spanish system/market day-ahead reference fields.

## Data sources

### Spanish electricity and city weather

The base files are from the Kaggle dataset [Hourly energy demand, generation and weather](https://www.kaggle.com/datasets/nicholasjhana/energy-consumption-generation-prices-and-weather/data). The local copies contain:

- `data/energy_dataset.csv`: 35,064 hourly observations from 1 January 2015 through 31 December 2018, including generation by technology, load, prices, and day-ahead forecast fields;
- `data/weather_features.csv`: 178,396 city-hour weather observations for Barcelona, Bilbao, Madrid, Seville, and Valencia; and
- provider fields originating from ENTSO-E/Red Eléctrica de España and OpenWeather, as described on the dataset page.

The Kaggle dataset is marked CC0. Always consult the source page and the original data providers for current terms and attribution requirements.

### CAMS solar-radiation data

The solar experiments add point time series from the [Copernicus Atmosphere Monitoring Service (CAMS) solar-radiation dataset](https://ads.atmosphere.copernicus.eu/datasets/cams-solar-radiation-timeseries?tab=overview). CAMS provides clear-sky and all-sky solar irradiation, including global horizontal irradiation (GHI), together with cloud-related fields for selected coordinates.

`CAMS_solar_data.ipynb` and `CAMS_solar_data_final.ipynb` retrieve, clean, resample, and aggregate the point data. The repository includes selected site-level CSV files in `data/` and the national modelling table `output/agreggated_CAMS_solar_dataset.csv`.

### Optional ERA5 extension

The repository also contains an experimental workflow based on [ERA5 hourly single-level reanalysis](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=overview):

- `era5_download.py` downloads monthly 2015–2018 files for Spain;
- `era5_feature_engineering.py` extracts weighted features at representative solar-plant locations; and
- `era5_model_training.ipynb` trains and evaluates a solar LightGBM model with the engineered ERA5 fields.

ERA5 downloads are large and are not required for the core Kaggle/CAMS notebooks.

## Feature engineering

The notebooks build several groups of features:

- **Time alignment:** conversion of local timestamps to UTC, hourly resampling, sorting, and merging electricity and weather observations.
- **Missing-data handling:** rolling or fitted replacements for short gaps and LOESS-based smoothing experiments, with replacement flags retained for inspection.
- **Calendar features:** hour, day of week, day of year, month, weekend, Spanish public holiday, season, and peak-period indicators.
- **Cyclical encodings:** sine/cosine representations for hour, weekday, month, and annual position.
- **Solar geometry:** sunrise, sunset, daylight duration, solar elevation, clear-sky irradiance, and an estimated solar-flux feature.
- **City weather:** temperature, pressure, humidity, wind speed/direction, rain, snow, clouds, weather category/severity, and city-weighted aggregates.
- **Lag-safe history:** shifted target/weather lags and rolling means. The latest CAMS LightGBM experiment recursively rebuilds solar-generation lag and rolling features inside each 24-hour prediction block from historical actuals and earlier predictions.
- **CAMS radiation:** weighted GHI, clear-sky GHI, cloud optical depth, cloud coverage, snow probability, clear-sky ratios, interaction terms, and selected lag/rolling variants.
- **ERA5 radiation and weather:** GHI, clear-sky index, cloud layers, temperature/dew point, pressure, wind, precipitation, boundary-layer height, spatial dispersion, and plant-level features.

`Rolling_Feature_Leakage_Demo.ipynb` demonstrates why a rolling feature used to predict time *t* must be shifted so its window ends before *t*.

## Modelling approaches

### ARIMAX and SARIMAX

The statistical notebooks use `statsmodels` SARIMAX with exogenous weather, calendar, and solar-geometry variables. They explore:

- autoregressive, differencing, and moving-average orders;
- daily seasonal periods, with alternative periods for price;
- ADF and KPSS stationarity tests;
- parameter-grid experiments evaluated with AIC/BIC; and
- residual, autocorrelation, seasonality, and periodogram diagnostics.

These models provide an interpretable baseline but become computationally expensive as the feature set and seasonal order grow.

### LightGBM

The LightGBM notebooks fit nonlinear boosted trees for all four targets. They include:

- mean regression and P16/P50/P84 quantile objectives;
- Optuna TPE hyperparameter tuning;
- categorical and continuous feature handling;
- early stopping on a chronological validation tail;
- feature importance by gain and split count;
- residual and error analysis; and
- recursive target-lag construction for the latest CAMS solar experiment.

The quantile models supply empirical uncertainty bands; they should not be treated as calibrated prediction intervals without a separate coverage analysis.

## Validation and rolling 24-hour evaluation

The modelling notebooks use a chronological 70/30 train/test boundary. Random shuffling is avoided.

Within the training span, the LightGBM workflow creates 30-day walk-forward validation folds. A fixed-length training window slides forward, each validation window follows its training window in time, and Optuna aggregates the fold scores for model selection.

For the final evaluation, the code advances through the held-out test period one day at a time:

1. end the training history one hour before the next prediction block;
2. fit or update the model using only the available history;
3. predict the next 24 hourly values; and
4. store mean, median, lower-quantile, and upper-quantile predictions before advancing one day.

The latest solar lag/rolling experiment predicts sequentially within each 24-hour block so target-derived inputs for later hours come from prior predictions rather than unseen actual generation.

Reported metrics include MSE, RMSE, MAE, MAPE, sMAPE, R², and RMSE skill relative to the supplied day-ahead benchmark:

```text
RMSE skill = 1 - (model RMSE / benchmark RMSE)
```

A positive skill score means the model improved on the benchmark; zero is a tie; a negative value means the benchmark performed better.

## Comparison with the Spanish day-ahead benchmark

The saved rolling-test reports currently show the following overall results. The model variant listed is the lower-RMSE mean/median estimate in each report.

| Target / experiment | Model variant | Model RMSE | Benchmark RMSE | RMSE skill | Model R² | Benchmark R² |
|---|---:|---:|---:|---:|---:|---:|
| Solar, city weather | Mean | 629.883 | 216.105 | -1.915 | 0.853 | 0.983 |
| Solar, CAMS + recursive lag/roll | Mean | 553.862 | 216.105 | -1.563 | 0.886 | 0.983 |
| Onshore wind | Mean | 2,819.780 | 575.732 | -3.898 | 0.274 | 0.970 |
| Total load | Median | 2,310.340 | 399.650 | -4.781 | 0.753 | 0.993 |
| Electricity price | Median | 11.133 | 12.355 | **0.099** | 0.190 | 0.002 |

The results suggest that CAMS and recursive solar-history features improve the solar LightGBM experiment, but the supplied solar, wind, and load forecasts remain substantially stronger. The rolling price model is the only saved experiment in this table with positive RMSE skill, improving RMSE by about 9.9% relative to the day-ahead price field.

Solar MAPE is unstable because actual generation is zero or close to zero at night. RMSE, MAE, sMAPE, and a separate daylight-only evaluation are more informative for that target.

Detailed tables and interactive Bokeh plots are under `output/results/ARIMAX/` and `output/results/LightGBM/`.

## Repository guide

| Path | Purpose |
|---|---|
| `Energy_Prediction_Spain.ipynb` | Initial electricity/weather exploration, cleaning, visualisation, and merged feature preparation. |
| `Energy_Prediction_SARIMAX_solar_wind_Spain.ipynb` | ARIMAX/SARIMAX experiments for solar and onshore wind generation. |
| `Energy_prediction_SARIMAX_load_price_Spain.ipynb` | ARIMAX/SARIMAX experiments for total load and electricity price. |
| `Energy_prediction_LightGBM_Spain.ipynb` | Main LightGBM workflow for solar, wind, load, and price, including walk-forward and rolling 24-hour evaluation. |
| `Energy_prediction_LightGBM_Spain_error_analysis.ipynb` | Follow-up LightGBM residual and error analysis. |
| `CAMS_solar_data.ipynb` | Early CAMS retrieval and processing workflow. |
| `CAMS_solar_data_final.ipynb` | Refined CAMS site processing and aggregation workflow. |
| `Energy_prediction_LightGBM_Spain_CAMS.ipynb` | Solar LightGBM models using CAMS radiation, including the recursive lag/rolling experiment. |
| `Rolling_Feature_Leakage_Demo.ipynb` | Minimal demonstration of leakage-safe rolling features. |
| `era5_download.py` | Optional ERA5 downloader for 2015–2018 monthly files. |
| `era5_feature_engineering.py` | Optional ERA5 extraction, spatial aggregation, and merge pipeline. |
| `era5_model_training.ipynb` | Optional LightGBM solar experiment using ERA5 features. |
| `data/` | Source electricity, weather, and selected CAMS files. |
| `output/exploratory/` | Saved exploratory Bokeh plots and correlation reports. |
| `output/results/` | Model metrics, diagnostics, tuning logs, feature importance, and interactive plots. |

The notebooks are exploratory and partially self-contained rather than a single automated pipeline. Run them from the repository root because many paths are relative to the current working directory.

## Requirements

The supported setup is Conda on macOS, Linux, or Windows. `environment.yml` uses Python 3.11 for broad compatibility across the notebooks and their scientific dependencies.

The environment covers every third-party import in the current notebooks and Python scripts:

- data and statistics: pandas, NumPy, SciPy, scikit-learn, statsmodels;
- modelling and tuning: LightGBM, Optuna, joblib;
- plotting: Matplotlib, Seaborn, Bokeh;
- solar, calendar, and location utilities: pvlib, holidays, suntime, pgeocode, geopy;
- smoothing and persistence: loess, dill;
- CAMS/ERA5 access and processing: ecmwf-datastores-client, cdsapi, xarray, netCDF4, h5netcdf; and
- notebook/report utilities: JupyterLab, ipykernel, tqdm, and tabulate.

Selenium is not required by the current tracked code: the notebooks save interactive Bokeh HTML and do not call Bokeh's browser-based PNG/SVG export functions.

## Installation with Conda

Clone the repository and create the environment:

```bash
git clone git@github.com:baddison2005/Energy_Prediction_Spain.git
cd Energy_Prediction_Spain
conda env create -f environment.yml
conda activate Spain_energy
```

Register the environment as a notebook kernel:

```bash
python -m ipykernel install --user --name Spain_energy --display-name "Python (Spain_energy)"
jupyter lab
```

To update an existing environment after `environment.yml` changes:

```bash
conda env update -n Spain_energy -f environment.yml --prune
```

## Installation with pip

Conda is recommended because it manages the compiled scientific and NetCDF dependencies. A pip-compatible package list is also provided for existing Python 3.11 environments:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m ipykernel install --user --name Spain_energy --display-name "Python (Spain_energy)"
```

`requirements.txt` mirrors the direct packages in `environment.yml`; transitive dependency versions are resolved by pip for the current platform.

## Usage

### Core Kaggle workflow

1. Confirm `data/energy_dataset.csv` and `data/weather_features.csv` are present.
2. Open `Energy_Prediction_Spain.ipynb` for exploration and preprocessing.
3. Run the relevant SARIMAX or LightGBM notebook for the target of interest.
4. Review saved metrics and interactive HTML plots under `output/results/`.

### CAMS solar workflow

1. Configure access to the CAMS Atmosphere Data Store according to the [ECMWF data-stores client documentation](https://ecmwf.github.io/ecmwf-datastores-client/). Never commit API credentials.
2. Run `CAMS_solar_data_final.ipynb` to retrieve/process selected locations, or use the CAMS CSV files already present in `data/`.
3. Run `Energy_prediction_LightGBM_Spain_CAMS.ipynb` for the CAMS-enhanced solar experiments.

### Optional ERA5 workflow

Configure a Copernicus Climate Data Store API key, then run:

```bash
python era5_download.py
python era5_feature_engineering.py
jupyter lab era5_model_training.ipynb
```

The downloader requests 48 monthly files for 2015–2018. Expect substantial download time and disk usage. Local ERA5 files under `data/era5/` are ignored by Git.

## Limitations

- **Historical scope:** the source data ends in 2018. Spain's generation fleet, demand patterns, market behaviour, and forecasting systems have changed since then.
- **Forecast-time availability:** city weather, CAMS all-sky radiation, and ERA5 reanalysis are historical observations/reanalysis products, not automatically day-ahead forecasts. Any same-hour feature used inside a future 24-hour block must be replaced with a value genuinely available at forecast issue time.
- **Leakage audit:** the recursive solar loop prevents direct use of unseen target lags within each test block, but the same recursive construction should also be applied inside model-selection folds. Imputation, aggregation, and all rolling features should be audited against explicit release timestamps.
- **Spatial approximation:** five-city weather and selected CAMS/ERA5 sites are proxies for national conditions. Plant locations and aggregation weights are approximate and do not fully represent the generation fleet.
- **Benchmark asymmetry:** the Spanish TSO/system forecasts may use richer operational, meteorological, outage, and market information than is available in this repository.
- **Single historical holdout:** one chronological test segment does not establish robustness across years, structural changes, or extreme events.
- **Notebook reproducibility:** several notebooks are stateful, computationally expensive, and contain saved outputs. They are not yet covered by automated end-to-end tests.
- **Uncertainty:** the P16/P50/P84 LightGBM models are fitted independently, so quantiles may cross and empirical interval coverage is not guaranteed.

## Conclusions and future work

The project demonstrates an end-to-end comparative forecasting workflow across renewable generation, demand, and price. The strongest current finding is that richer radiation and target-history features improve the solar LightGBM model, while the professional day-ahead benchmarks remain difficult to beat for solar, wind, and load. The rolling price experiment achieves modest positive RMSE skill and merits further investigation.

High-priority next steps are to:

- replace retrospective weather/radiation fields with archived day-ahead forecast vintages;
- apply recursive lag construction inside every tuning and validation fold;
- create daylight-only solar metrics and calibrated probabilistic-forecast diagnostics;
- automate preprocessing, training, and evaluation as scripts or a reproducible pipeline;
- expand backtesting across multiple rolling origins and newer Spanish electricity data;
- compare persistence, seasonal-naive, and ensemble baselines alongside the TSO fields;
- add SHAP or permutation-based interpretation and systematic error slicing; and
- add smoke tests, notebook validation, and continuous integration.

## License

Repository code is released under the [MIT License](LICENSE). Data and third-party services retain their own licences and terms; consult the linked source pages before redistributing or using them commercially.
