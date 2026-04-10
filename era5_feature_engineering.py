"""
ERA5 Feature Engineering Pipeline for Solar Power Prediction in Spain
======================================================================
Reads ERA5 zip files (containing NetCDF), extracts weather data at solar
plant locations, engineers derived features, and merges with the energy
generation dataset.

Prerequisites:
  - ERA5 zip/NetCDF files in data/era5/ (from era5_download.py)
  - energy_dataset.csv in data/
  - pip install xarray netCDF4 pvlib pandas numpy

Usage:
  python era5_feature_engineering.py
"""

import io
import os
import glob
import zipfile
import numpy as np
import pandas as pd
import xarray as xr
import pvlib
from pvlib.location import Location

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ERA5_DIR = os.path.join(SCRIPT_DIR, "data", "era5")
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
OUTPUT_FILE = os.path.join(DATA_DIR, "era5_features_merged.csv")

# Solar plant locations: name -> (latitude, longitude)
# Approximate installed capacity weights (relative)
SOLAR_PLANTS = {
    "Andasol":     {"lat": 37.23, "lon": -3.07, "weight": 0.20},
    "Gemasolar":   {"lat": 37.56, "lon": -5.33, "weight": 0.15},
    "Olmedilla":   {"lat": 39.57, "lon": -2.43, "weight": 0.25},
    "Puertollano": {"lat": 38.68, "lon": -4.10, "weight": 0.20},
    "Extremadura": {"lat": 38.90, "lon": -6.30, "weight": 0.20},
}

def _extract_point_from_dataset(ds, plant_info):
    """Extract data at a single plant location from an xarray Dataset."""
    point = ds.sel(
        latitude=plant_info["lat"],
        longitude=plant_info["lon"],
        method="nearest",
    )
    df = point.to_dataframe().reset_index()

    # Rename 'valid_time' or 'time' to 'utc_time'
    time_col = "valid_time" if "valid_time" in df.columns else "time"
    df = df.rename(columns={time_col: "utc_time"})

    # Drop spatial coordinate columns
    for col in ["latitude", "longitude", "number", "expver"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Handle duplicate times (e.g. from multiple expver)
    df = df.groupby("utc_time").first().reset_index()
    return df


def _open_era5_file(filepath):
    """Open an ERA5 file, handling both zip and raw NetCDF formats.
    
    For zips, reads NetCDF members directly from the zip into memory
    using BytesIO + h5netcdf engine — no disk I/O needed beyond reading the zip.
    """
    try:
        if zipfile.is_zipfile(filepath):
            zf = zipfile.ZipFile(filepath)
            nc_members = [m for m in zf.namelist() if m.endswith('.nc')]
            datasets = []
            for member in nc_members:
                data = zf.read(member)
                buf = io.BytesIO(data)
                ds = xr.open_dataset(buf, engine="h5netcdf")
                ds = ds.load()  # Load into memory so buffer can be freed
                datasets.append(ds)
                del data, buf
            zf.close()
            if len(datasets) == 1:
                merged = datasets[0]
            else:
                merged = xr.merge(datasets, compat="override")
            return merged
        else:
            ds = xr.load_dataset(filepath, engine="netcdf4")
            return ds
    except Exception:
        ds = xr.load_dataset(filepath, engine="netcdf4")
        return ds


def extract_at_plants_streaming():
    """
    Process ERA5 files one at a time.
    Each zip file is opened, point data extracted, then closed.
    Returns a dict of DataFrames keyed by plant name.
    """
    era5_files = sorted(glob.glob(os.path.join(ERA5_DIR, "era5_spain_*.nc")))
    if not era5_files:
        raise FileNotFoundError(f"No ERA5 files found in {ERA5_DIR}")
    print(f"Processing {len(era5_files)} ERA5 files ...")

    # Initialize storage for each plant
    plant_chunks = {name: [] for name in SOLAR_PLANTS}

    for i, filepath in enumerate(era5_files):
        basename = os.path.basename(filepath)
        print(f"  [{i+1}/{len(era5_files)}] {basename} ...", end=" ")
        try:
            ds = _open_era5_file(filepath)

            for name, info in SOLAR_PLANTS.items():
                df_chunk = _extract_point_from_dataset(ds, info)
                plant_chunks[name].append(df_chunk)

            # Free memory
            ds.close()
            del ds
            print("OK")
        except Exception as e:
            print(f"ERROR: {e}")
            continue

    # Combine chunks for each plant
    plant_dfs = {}
    for name in SOLAR_PLANTS:
        if plant_chunks[name]:
            df = pd.concat(plant_chunks[name], ignore_index=True)
            df = df.groupby("utc_time").first().reset_index()
            df = df.sort_values("utc_time").reset_index(drop=True)
            df["plant_name"] = name
            plant_dfs[name] = df
            print(f"  {name}: {len(df)} timesteps")
        else:
            print(f"  WARNING: No data for {name}")

    return plant_dfs


def compute_clear_sky_ghi(times, lat, lon):
    """Compute clear-sky GHI using pvlib Ineichen model for given times and location."""
    location = Location(lat, lon)
    # Get solar position
    solpos = location.get_solarposition(times)
    # Ineichen clear-sky model
    cs = location.get_clearsky(times, model="ineichen", solar_position=solpos)
    return cs["ghi"].values


def process_plant_features(df, plant_info):
    """Process a single plant's ERA5 data into features."""
    df = df.copy()
    df = df.set_index("utc_time").sort_index()

    # 1. Convert ssrd and tp to usable units
    # NOTE: ERA5 hourly data (CDS API 2024+) provides ssrd/tp as hourly accumulated
    # values, not running accumulations — no de-accumulation needed.
    if "ssrd" in df.columns:
        # Convert J/m² to W/m² (divide by 3600 seconds)
        df["ghi_wm2"] = (df["ssrd"] / 3600.0).clip(lower=0)
    if "tp" in df.columns:
        # Convert m to mm
        df["precip_mm"] = (df["tp"] * 1000.0).clip(lower=0)

    # 2. Compute clear-sky GHI
    times = df.index
    cs_ghi = compute_clear_sky_ghi(
        times, plant_info["lat"], plant_info["lon"]
    )
    df["clearsky_ghi_wm2"] = cs_ghi

    # 3. Derived features
    # Clear-sky index (kt): ratio of actual to clear-sky GHI
    # when ghi < 10, set to 0 to indicate very low irradiance
    df["clearsky_index"] = np.where(
        df["clearsky_ghi_wm2"] > 10,
        df["ghi_wm2"] / df["clearsky_ghi_wm2"],
        0.0,
    )
    df["clearsky_index"] = df["clearsky_index"].clip(0, 2.0)

    # Cloud impact ratio: 1 - tcc (higher = clearer sky)
    # tcc = Total cloud cover
    # - Fraction ranging from 0 to 1
    # - 0: completely clear sky, 
    # - 1: completely overcast
    # TODO: We can omit because model can learn this from tcc directly
    if "tcc" in df.columns:
        df["cloud_clearness"] = 1.0 - df["tcc"]

    # Wind speed from u10, v10
    # u10 = eastward wind component at 10m above ground level (m/s)
    # v10 = northward wind component at 10m above ground level (m/s)
    if "u10" in df.columns and "v10" in df.columns:
        df["wind_speed_10m"] = np.sqrt(df["u10"] ** 2 + df["v10"] ** 2)
        
        # Note: LightGBM should be able to capture circularity problem with 
        # degrees (359° and 1° are close but numerically far apart).
        # Might be an issue if we train NN models.
        df["wind_dir_10m"] = np.degrees(np.arctan2(-df["u10"], -df["v10"])) % 360

    # Dewpoint depression (indicates moisture/fog potential)
    # t2m = air temperature at 2m above ground level (K)
    # d2m = dewpoint temperature at 2m above ground level (K)
    # Small values indicates near saturation, which can lead to fog and reduced solar output
    if "t2m" in df.columns and "d2m" in df.columns:
        df["dewpoint_depression"] = df["t2m"] - df["d2m"]

    # Temperature-irradiance interaction
    # High GHI + High temp -> panels can overheat and reduce efficiency
    # High GHI + Low temp -> panels can operate more efficiently
    # Low GHI -> low output regardless of temp
    if "t2m" in df.columns and "ghi_wm2" in df.columns:
        df["temp_irradiance_interaction"] = df["t2m"] * df["ghi_wm2"] / 1000.0

    # Surface pressure in hPa
    if "sp" in df.columns:
        df["pressure_hpa"] = df["sp"] / 100.0

    return df.reset_index()


def aggregate_across_plants(plant_dfs):
    """
    Returns a single DataFrame with one row per UTC hour.
    """
    # Columns to aggregate (weighted average)
    agg_cols = [
        "ghi_wm2", "clearsky_ghi_wm2", "clearsky_index",
        "tcc", "lcc", "mcc", "hcc", "cloud_clearness",
        "t2m", "d2m", "dewpoint_depression",
        "pressure_hpa", "wind_speed_10m", "wind_dir_10m",
        "precip_mm", "blh",
        "temp_irradiance_interaction",
    ]

    # Key per-plant features to expose individually
    per_plant_cols = ["ghi_wm2", "clearsky_index", "tcc", "t2m"]

    all_dfs = []
    for name, df in plant_dfs.items():
        weight = SOLAR_PLANTS[name]["weight"]
        sub = df[["utc_time"] + [c for c in agg_cols if c in df.columns]].copy()
        for col in [c for c in agg_cols if c in sub.columns]:
            sub[col] = sub[col] * weight
        all_dfs.append(sub)

    # Sum the weighted values (weights sum to 1.0)
    combined = pd.concat(all_dfs, ignore_index=True)
    agg = combined.groupby("utc_time").sum().reset_index()
    agg = agg.sort_values("utc_time").reset_index(drop=True)

    # Add per-plant features (unweighted, so model can learn spatial importance)
    for name, df in plant_dfs.items():
        plant_key = name.lower()
        for col in per_plant_cols:
            if col in df.columns:
                plant_sub = df[["utc_time", col]].copy()
                plant_sub = plant_sub.rename(columns={col: f"{plant_key}_{col}"})
                agg = agg.merge(plant_sub, on="utc_time", how="left")

    # Add spatial dispersion features (std across plants)
    # Low std means all plants see similar conditions (e.g. large-scale weather pattern)
    for col in ["ghi_wm2", "tcc"]:
        plant_vals = []
        for name, df in plant_dfs.items():
            if col in df.columns:
                pv = df[["utc_time", col]].rename(columns={col: f"_pv_{name}"})
                plant_vals.append(pv)
        if plant_vals:
            pv_merged = plant_vals[0]
            for pv in plant_vals[1:]:
                pv_merged = pv_merged.merge(pv, on="utc_time", how="outer")
            val_cols = [c for c in pv_merged.columns if c.startswith("_pv_")]
            agg = agg.merge(
                pv_merged[["utc_time"]].assign(
                    **{f"{col}_spatial_std": pv_merged[val_cols].std(axis=1)}
                ),
                on="utc_time", how="left"
            )

    # Prefix ERA5 columns to distinguish from original weather features
    rename_map = {col: f"era5_{col}" for col in agg.columns if col != "utc_time"}
    agg = agg.rename(columns=rename_map)

    return agg


def add_rolling_features(df, columns, window=3):
    """Add lag-safe rolling mean features (shift(1) before rolling)."""
    for col in columns:
        if col in df.columns:
            # shift(1) ensures no data leakage; fillna handles first row
            df[f"{col}_rolling{window}h"] = (
                df[col].shift(1).rolling(window=window, min_periods=1).mean().fillna(df[col])
            )
    return df


def merge_with_energy_data(era5_agg):
    """Merge ERA5 features with the energy generation dataset."""
    energy_file = os.path.join(DATA_DIR, "energy_dataset.csv")
    if not os.path.exists(energy_file):
        raise FileNotFoundError(f"Energy dataset not found: {energy_file}")

    print("Loading energy_dataset.csv ...")
    energy = pd.read_csv(energy_file, parse_dates=["time"])
    energy = energy.rename(columns={"time": "utc_time"})
    # Convert to UTC and make timezone-naive to match ERA5
    energy["utc_time"] = pd.to_datetime(energy["utc_time"], utc=True).dt.tz_localize(None)
    energy = energy.sort_values("utc_time").reset_index(drop=True)

    print(f"  Energy dataset: {len(energy)} rows, {energy['utc_time'].min()} to {energy['utc_time'].max()}")
    print(f"  ERA5 features:  {len(era5_agg)} rows, {era5_agg['utc_time'].min()} to {era5_agg['utc_time'].max()}")

    # Ensure both are datetime and timezone-naive
    era5_agg["utc_time"] = pd.to_datetime(era5_agg["utc_time"]).dt.tz_localize(None)

    # Merge on UTC time
    merged = pd.merge(energy, era5_agg, on="utc_time", how="inner")
    print(f"  Merged dataset: {len(merged)} rows")

    return merged


def main():
    print("=" * 70)
    print("ERA5 Feature Engineering for Solar Power Prediction")
    print("=" * 70)

    # Step 1+2: Load ERA5 data and extract at solar plant locations (streaming)
    print("\nExtracting data at solar plant locations ...")
    plant_dfs = extract_at_plants_streaming()

    # Step 3: Process features for each plant
    print("\nComputing derived features per plant ...")
    processed_plants = {}
    for name, df in plant_dfs.items():
        print(f"  Processing {name} ...")
        processed_plants[name] = process_plant_features(df, SOLAR_PLANTS[name])

    # Step 4: Aggregate across plants
    print("\nAggregating across solar plant locations ...")
    era5_agg = aggregate_across_plants(processed_plants)
    print(f"  Aggregated shape: {era5_agg.shape}")

    # Step 5: Add rolling features
    print("\nAdding rolling features ...")
    rolling_cols = [
        "era5_ghi_wm2", "era5_clearsky_index",
        "era5_tcc", "era5_t2m", "era5_pressure_hpa",
        "era5_wind_speed_10m", "era5_precip_mm", "era5_blh",
    ]
    era5_agg = add_rolling_features(era5_agg, rolling_cols, window=3)

    # Step 6: Merge with energy dataset
    print("\nMerging with energy generation data ...")
    merged = merge_with_energy_data(era5_agg)

    # Step 7: Save
    print(f"\nSaving to {OUTPUT_FILE} ...")
    merged.to_csv(OUTPUT_FILE, index=False)
    print(f"  Saved {len(merged)} rows × {len(merged.columns)} columns")

    # Print summary of ERA5 columns added
    era5_cols = [c for c in merged.columns if c.startswith("era5_")]
    print(f"\n  ERA5 feature columns ({len(era5_cols)}):")
    for col in era5_cols:
        print(f"    - {col}")

    print("\nDone!")


if __name__ == "__main__":
    main()
