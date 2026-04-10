"""
ERA5 Reanalysis Data Download Script for Solar Power Prediction in Spain
=========================================================================
Downloads hourly ERA5 single-level reanalysis data from the Copernicus
Climate Data Store (CDS) for Spain (2015-2018).

Uses parallel workers to speed up downloads (default: 4 concurrent requests).

Prerequisites:
  - pip install cdsapi
  - CDS API key configured: either ~/.cdsapirc file or environment variables
    CDSAPI_URL and CDSAPI_KEY.  Register at https://cds.climate.copernicus.eu

Usage:
  python era5_download.py            # 4 parallel workers (default)
  python era5_download.py --workers 6
"""

import os
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import cdsapi

YEARS = [str(y) for y in range(2015, 2019)]     # 2015-2018
MONTHS = [f"{m:02d}" for m in range(1, 13)]
DAYS = [f"{d:02d}" for d in range(1, 32)]
HOURS = [f"{h:02d}:00" for h in range(24)]

# Spain bounding box [North, West, South, East]
AREA = [44, -10, 35, 5]

VARIABLES = [
    "surface_solar_radiation_downwards",        # ssrd – accumulated GHI (J/m²)
    "total_cloud_cover",                        # tcc  – fraction [0-1]
    "low_cloud_cover",                          # lcc
    "medium_cloud_cover",                       # mcc
    "high_cloud_cover",                         # hcc
    "2m_temperature",                           # t2m  – Kelvin
    "2m_dewpoint_temperature",                  # d2m  – Kelvin
    "surface_pressure",                         # sp   – Pa
    "10m_u_component_of_wind",                  # u10  – m/s
    "10m_v_component_of_wind",                  # v10  – m/s
    "total_precipitation",                      # tp   – accumulated m
    "boundary_layer_height",                    # blh  – metres
]

# Output directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "data", "era5")

PRODUCT = "reanalysis-era5-single-levels"

def download_one_month(year, month, max_retries=5):
    """Download a single year-month of ERA5 data. """
    out_file = os.path.join(OUTPUT_DIR, f"era5_spain_{year}_{month}.nc")

    if os.path.exists(out_file):
        return f"[skip] {year}-{month} already exists"

    for attempt in range(1, max_retries + 1):
        try:
            client = cdsapi.Client(quiet=True)
            client.retrieve(
                PRODUCT,
                {
                    "product_type": "reanalysis",
                    "variable": VARIABLES,
                    "year": year,
                    "month": month,
                    "day": DAYS,
                    "time": HOURS,
                    "area": AREA,
                    "format": "netcdf",
                },
                out_file,
            )
            size_mb = os.path.getsize(out_file) / (1024 * 1024)
            return f"[done] {year}-{month}  ({size_mb:.1f} MB)"
        except Exception as e:
            err_msg = str(e)
            if "temporarily limited" in err_msg or "queued" in err_msg:
                wait = 30 * attempt
                if attempt < max_retries:
                    time.sleep(wait)
                    continue
            if attempt == max_retries:
                return f"[ERROR] {year}-{month}: {e}"
            time.sleep(10 * attempt)


def download_era5(max_workers=4):
    """Download ERA5 data with parallel workers."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Build list of (year, month) jobs
    jobs = [(y, m) for y in YEARS for m in MONTHS]
    total = len(jobs)
    done = 0

    print(f"Submitting {total} download jobs with {max_workers} parallel workers ...\n")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(download_one_month, year, month): (year, month)
            for year, month in jobs
        }
        for future in as_completed(futures):
            done += 1
            year, month = futures[future]
            result = future.result()
            print(f"  [{done}/{total}] {result}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download ERA5 data for Spain")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel download workers (default: 4)")
    args = parser.parse_args()

    print("ERA5 Download for Spain Solar Power Prediction")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Years: {YEARS}")
    print(f"Variables: {len(VARIABLES)}")
    print(f"Workers: {args.workers}")
    print()
    download_era5(max_workers=args.workers)
    print("\nDownload complete.")
