# =============================================================
# download_weather_data.py
# Downloads historical weather data for all states defined
# in config.py from the NASA POWER API.
#
# OUTPUT STRUCTURE:
#   <DATA_FOLDER_NAME>/
#   |-- all_states_combined.csv
#   |-- training_YYYY_YYYY.csv
#   |-- testing_YYYY_YYYY.csv
#   |-- by_state/
#   |   |-- kwara_ilorin.csv
#   |   `-- ...
#   `-- by_year/
#       |-- 2000.csv
#       `-- ...
#
# HOW TO RUN:
#   python download_weather_data.py
# =============================================================

import os, sys, time
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    BASE_DIR, DATA_DIR, DATA_FOLDER_NAME,
    STATES, NASA_PARAMETERS,
    DOWNLOAD_START_YEAR, DOWNLOAD_END_YEAR,
    TRAIN_YEARS_END, TEST_YEARS_START,
    COUNTRY_NAME,
)

# -- Output folders -------------------------------------------
STATE_DIR = os.path.join(DATA_DIR, "by_state")
YEAR_DIR  = os.path.join(DATA_DIR, "by_year")
for d in [DATA_DIR, STATE_DIR, YEAR_DIR]:
    os.makedirs(d, exist_ok=True)

NASA_PARAMS = ",".join(NASA_PARAMETERS.keys())
API_BASE    = "https://power.larc.nasa.gov/api/temporal/daily/point"

# -- NASA POWER fetch function --------------------------------
def fetch_nasa_power(lat, lon, start_year, end_year):
    url = (
        f"{API_BASE}"
        f"?parameters={NASA_PARAMS}"
        f"&community=AG"
        f"&longitude={lon}"
        f"&latitude={lat}"
        f"&start={start_year}0101"
        f"&end={end_year}1231"
        f"&format=JSON"
    )
    try:
        resp = requests.get(url, timeout=60)
        if resp.status_code != 200:
            return None
        data  = resp.json()
        props = data.get("properties", {}).get("parameter", {})
        if not props:
            return None

        records = {}
        for nasa_key, col_name in NASA_PARAMETERS.items():
            if nasa_key in props:
                records[col_name] = props[nasa_key]
        if not records:
            return None

        df            = pd.DataFrame(records)
        first_col     = list(records.values())[0]
        df.index      = pd.to_datetime(list(first_col.keys()), format="%Y%m%d")
        df.index.name = "DATE"
        df            = df.sort_index()
        df            = df.replace(-999.0, np.nan)
        df            = df.replace(-99.0,  np.nan)
        df.insert(0, "YEAR",  df.index.year)
        df.insert(1, "MONTH", df.index.month)
        df.insert(2, "DAY",   df.index.day)
        return df
    except Exception:
        return None

# -- Main download loop ---------------------------------------
print("\n" + "="*65)
print(f"  {COUNTRY_NAME.upper()} WEATHER DATA DOWNLOADER")
print(f"  Source : NASA POWER API (MERRA-2)")
print(f"  Period : {DOWNLOAD_START_YEAR} - {DOWNLOAD_END_YEAR}")
print(f"  States : {len(STATES)}")
print("="*65 + "\n")

all_frames    = []
failed_states = []

for state_key, state_name, capital, lat, lon in tqdm(
        STATES, desc="Downloading", unit="state"):

    tqdm.write(f"\n  [{state_name}] {capital} (lat={lat}, lon={lon})")

    df         = None
    actual_end = DOWNLOAD_END_YEAR
    for try_end in [DOWNLOAD_END_YEAR, DOWNLOAD_END_YEAR - 1]:
        df = fetch_nasa_power(lat, lon, DOWNLOAD_START_YEAR, try_end)
        if df is not None and len(df) > 100:
            actual_end = try_end
            break
        time.sleep(1)

    if df is None or len(df) == 0:
        tqdm.write(f"  [FAILED] Could not download {state_name}")
        failed_states.append(state_name)
        continue

    df.insert(0, "STATE",   state_name)
    df.insert(1, "CAPITAL", capital)
    df.insert(2, "LAT",     lat)
    df.insert(3, "LON",     lon)

    # Rename columns: feature -> STATEKEY_feature
    code       = state_key.upper()
    rename_map = {col: f"{code}_{col}"
                  for col in list(NASA_PARAMETERS.values())
                  if col in df.columns}
    df = df.rename(columns=rename_map)

    state_file = os.path.join(
        STATE_DIR,
        f"{state_key}_{capital.lower().replace(' ', '_')}.csv")
    df.to_csv(state_file)
    tqdm.write(f"  [OK] {len(df)} days -> by_state/{os.path.basename(state_file)}"
               f" (up to {actual_end})")

    all_frames.append(df)
    time.sleep(0.8)

# -- Combine --------------------------------------------------
print("\n" + "-"*65)
print("  Combining all state data ...")

if not all_frames:
    print("[ERROR] No data downloaded. Check internet connection.")
    sys.exit(1)

df_all = pd.concat(all_frames, axis=0, sort=False)
df_all = df_all.sort_values(["STATE", "DATE"])

combined_path = os.path.join(DATA_DIR, "all_states_combined.csv")
df_all.to_csv(combined_path)
print(f"  [OK] all_states_combined.csv  ({len(df_all):,} rows)")
print(f"       States     : {df_all['STATE'].nunique()}")
print(f"       Date range : {df_all.index.min()} to {df_all.index.max()}")

# -- Train / test splits --------------------------------------
print("\n  Creating train/test splits ...")
df_train = df_all[df_all["YEAR"] <= TRAIN_YEARS_END]
df_test  = df_all[df_all["YEAR"] >= TEST_YEARS_START]

train_path = os.path.join(DATA_DIR,
    f"training_{DOWNLOAD_START_YEAR}_{TRAIN_YEARS_END}.csv")
test_path  = os.path.join(DATA_DIR,
    f"testing_{TEST_YEARS_START}_{DOWNLOAD_END_YEAR}.csv")

df_train.to_csv(train_path)
df_test.to_csv(test_path)
print(f"  [OK] training_{DOWNLOAD_START_YEAR}_{TRAIN_YEARS_END}.csv"
      f"  ({len(df_train):,} rows, {df_train['YEAR'].nunique()} years)")
print(f"  [OK] testing_{TEST_YEARS_START}_{DOWNLOAD_END_YEAR}.csv"
      f"  ({len(df_test):,} rows, {df_test['YEAR'].nunique()} years)")

# -- Per-year files -------------------------------------------
print("\n  Creating per-year files ...")
available_years = sorted(df_all["YEAR"].unique())
for yr in available_years:
    df_all[df_all["YEAR"] == yr].to_csv(
        os.path.join(YEAR_DIR, f"{yr}.csv"))
print(f"  [OK] {len(available_years)} yearly files -> by_year/")
print(f"       Years: {available_years[0]} to {available_years[-1]}")

# -- Summary --------------------------------------------------
print("\n" + "="*65)
print("  DOWNLOAD COMPLETE")
print("="*65)
print(f"\n  States downloaded : {len(all_frames)} / {len(STATES)}")
if failed_states:
    print(f"  Failed states     : {', '.join(failed_states)}")
print(f"\n  Output: {DATA_DIR}/")
print(f"    all_states_combined.csv  ({len(df_all):,} rows)")
print(f"    by_state/  ({len(all_frames)} files)")
print(f"    by_year/   ({len(available_years)} files)")
print(f"\n  Features downloaded:")
for nasa_key, col_name in NASA_PARAMETERS.items():
    print(f"    {col_name:<25} <- {nasa_key}")
