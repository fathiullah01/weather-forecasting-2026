# =============================================================
# config.py  -  PROJECT CONFIGURATION
# Kwara State University, Nigeria
# Faculty of ICT - Department of Computer Science
#
# THIS IS THE ONLY FILE YOU NEED TO EDIT.
# All scripts read every variable from here automatically.
# =============================================================
 
import os
 
# ==============================================================
# SECTION 1: PATHS  (do not change)
# ==============================================================
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, "country_weather_data")
FIG_DIR     = os.path.join(BASE_DIR, "figures")
TABLE_DIR   = os.path.join(BASE_DIR, "tables")
RESULT_DIR  = os.path.join(BASE_DIR, "results")
MODEL_DIR   = os.path.join(BASE_DIR, "models")
SCRIPTS_DIR = os.path.join(BASE_DIR, "scripts")
 
# ==============================================================
# SECTION 2: COUNTRY SETTINGS
# Change these when switching to a different country.
# ==============================================================
 
# Name of the country (used in figure titles and print output)
COUNTRY_NAME = "Nigeria"
 
# Folder where downloaded weather data is saved
# If you change country, update this path to match your downloader output
DATA_FOLDER_NAME = "country_weather_data"
DATA_DIR         = os.path.join(BASE_DIR, DATA_FOLDER_NAME)
 
# Dataset file the project reads from
# Change to "training_2000_2022.csv" if you want training set only
DATASET_FILE = os.path.join(DATA_DIR, "all_states_combined.csv")
 
# NASA POWER API parameters to download
# Keys = NASA parameter codes, Values = column names used in project
# Do not change values (column names) unless you update FEATURE_NAMES too
NASA_PARAMETERS = {
    "T2M"            : "temp_mean",        # Temperature at 2m (C)
    "T2M_MIN"        : "temp_min",         # Min temperature at 2m (C)
    "T2M_MAX"        : "temp_max",         # Max temperature at 2m (C)
    "RH2M"           : "humidity",         # Relative humidity at 2m (%)
    "PRECTOTCORR"    : "precipitation",    # Precipitation corrected (mm/day)
    "WS2M"           : "wind_speed",       # Wind speed at 2m (m/s)
    "WS2M_MAX"       : "wind_gust",        # Max wind speed at 2m (m/s)
    "CLOUD_AMT"      : "cloud_cover",      # Cloud amount (%)
    "PS"             : "pressure",         # Surface pressure (kPa)
    "ALLSKY_SFC_SW_DWN" : "global_radiation", # Solar radiation (MJ/m2/day)
    "ALLSKY_SFC_PAR_TOT": "sunshine",      # Photosynthetically active radiation
}
 
# ==============================================================
# SECTION 3: STATES / REGIONS
# Format: (state_key, state_name, capital_city, latitude, longitude)
#
# state_key   -> used as column prefix in dataset (e.g. "kwara")
# state_name  -> must match the STATE column in your dataset exactly
# capital_city-> city name shown in figures
# latitude    -> decimal degrees (positive = North, negative = South)
# longitude   -> decimal degrees (positive = East, negative = West)
#
# TO ADD A NEW COUNTRY: replace the list below with your regions.
# Example for Ghana:
#   ("greater_accra", "Greater Accra", "Accra",  5.6037,  -0.1870),
#   ("ashanti",       "Ashanti",       "Kumasi", 6.6885,  -1.6244),
# ==============================================================
STATES = [
    # (state_key,     state_name,    capital_city,    lat,      lon)
    ("abia",          "Abia",        "Umuahia",        5.5320,   7.4860),
    ("adamawa",       "Adamawa",     "Yola",           9.2035,  12.4954),
    ("akwa_ibom",     "Akwa Ibom",   "Uyo",            5.0377,   7.9128),
    ("anambra",       "Anambra",     "Awka",           6.2100,   7.0700),
    ("bauchi",        "Bauchi",      "Bauchi",        10.3158,   9.8442),
    ("bayelsa",       "Bayelsa",     "Yenagoa",        4.9267,   6.2676),
    ("benue",         "Benue",       "Makurdi",        7.7337,   8.5213),
    ("borno",         "Borno",       "Maiduguri",     11.8333,  13.1500),
    ("cross_river",   "Cross River", "Calabar",        4.9757,   8.3417),
    ("delta",         "Delta",       "Asaba",          6.1948,   6.7354),
    ("ebonyi",        "Ebonyi",      "Abakaliki",      6.3249,   8.1137),
    ("edo",           "Edo",         "Benin City",     6.3350,   5.6270),
    ("ekiti",         "Ekiti",       "Ado-Ekiti",      7.6217,   5.2216),
    ("enugu",         "Enugu",       "Enugu",          6.4584,   7.5464),
    ("fct",           "FCT",         "Abuja",          9.0579,   7.4951),
    ("gombe",         "Gombe",       "Gombe",         10.2897,  11.1673),
    ("imo",           "Imo",         "Owerri",         5.4836,   7.0333),
    ("jigawa",        "Jigawa",      "Dutse",         11.7904,   9.3417),
    ("kaduna",        "Kaduna",      "Kaduna",        10.5105,   7.4165),
    ("kano",          "Kano",        "Kano",          12.0022,   8.5920),
    ("katsina",       "Katsina",     "Katsina",       12.9816,   7.6174),
    ("kebbi",         "Kebbi",       "Birnin Kebbi",  12.4539,   4.1975),
    ("kogi",          "Kogi",        "Lokoja",         7.7975,   6.7392),
    ("kwara",         "Kwara",       "Ilorin",         8.5000,   4.5500),
    ("lagos",         "Lagos",       "Ikeja",          6.5244,   3.3792),
    ("nasarawa",      "Nasarawa",    "Lafia",          8.4926,   8.5140),
    ("niger",         "Niger",       "Minna",          9.6139,   6.5569),
    ("ogun",          "Ogun",        "Abeokuta",       7.1558,   3.3451),
    ("ondo",          "Ondo",        "Akure",          7.2526,   5.1931),
    ("osun",          "Osun",        "Osogbo",         7.7718,   4.5560),
    ("oyo",           "Oyo",         "Ibadan",         7.3775,   3.9470),
    ("plateau",       "Plateau",     "Jos",            9.8965,   8.8583),
    ("rivers",        "Rivers",      "Port Harcourt",  4.8156,   7.0498),
    ("sokoto",        "Sokoto",      "Sokoto",        13.0622,   5.2339),
    ("taraba",        "Taraba",      "Jalingo",        8.9013,  11.3734),
    ("yobe",          "Yobe",        "Damaturu",      11.7471,  11.9608),
    ("zamfara",       "Zamfara",     "Gusau",         12.1704,   6.6649),
]
 
# Convenience lookups built from STATES (do not change)
STATE_NAMES    = [s[1] for s in STATES]
STATE_KEYS     = [s[0] for s in STATES]
STATE_CAPITALS = {s[1]: s[2] for s in STATES}
STATE_COORDS   = {s[1]: (s[3], s[4]) for s in STATES}
 
# ==============================================================
# SECTION 4: FORECAST SETTINGS
# ==============================================================
 
# Forecast mode:
#   "all"    -> run and show forecasts for all states in STATES
#   "single" -> run for one state only (set SINGLE_STATE below)
FORECAST_MODE = "all"
 
# Forecast date mode (used by step6_future_forecast.py):
#   "auto"   -> automatically uses the last date available in the dataset
#   "manual" -> uses the exact date you specify in FORECAST_DATE below
FORECAST_DATE_MODE = "auto"
 
# Used only when FORECAST_DATE_MODE = "manual"
# Must be a date that exists in your dataset. Format: YYYY-MM-DD
FORECAST_DATE = "2024-08-01"
 
# Used only when FORECAST_MODE = "single"
# Must exactly match a state_name in STATES above
SINGLE_STATE = "Kwara"
SINGLE_CITY  = "Ilorin"   # label used in chart titles
 
# Primary state used to train the main PSO-RF model
# Other states are forecasted using the same tuned hyperparameters
# Change to whichever state is most relevant to your study
PRIMARY_STATE = "Kwara"
 
# ==============================================================
# SECTION 5: DATE SETTINGS
# ==============================================================
DOWNLOAD_START_YEAR = 2000   # first year to download
DOWNLOAD_END_YEAR   = 2026   # tries this year, falls back to previous if unavailable
TRAIN_YEARS_END     = 2022   # training:  START -> this year  (23 years)
TEST_YEARS_START    = 2023   # testing:   this year -> latest (3 years)
 
# ==============================================================
# SECTION 6: MODEL SETTINGS
# ==============================================================
 
# What the model predicts (next-day value)
# Options: "precipitation", "temp_mean", "temp_max"
TARGET_VARIABLE = "precipitation"
 
# Input features (must match NASA_PARAMETERS values above)
FEATURE_NAMES = [
    "temp_mean",
    "temp_min",
    "temp_max",
    "humidity",
    "wind_speed",
    "wind_gust",
    "cloud_cover",
    "pressure",
    "global_radiation",
    "sunshine",
]
 
# PSO - Feature Selection
PSO_N_PARTICLES_FS  = 10   # number of particles
PSO_N_ITERATIONS_FS = 15   # number of iterations
 
# PSO - Hyperparameter Tuning
PSO_N_PARTICLES_HP  = 10   # number of particles
PSO_N_ITERATIONS_HP = 20   # number of iterations
 
# PSO - Shared coefficients
PSO_C1 = 0.5   # cognitive coefficient
PSO_C2 = 0.3   # social coefficient
PSO_W  = 0.9   # inertia weight
 
# Random Forest fallback (used if PSO fails or times out)
RF_FALLBACK_N_ESTIMATORS = 100
RF_FALLBACK_MAX_DEPTH    = 8
 
# ==============================================================
# SECTION 7: THRESHOLD & OUTPUT SETTINGS
# ==============================================================
 
# Precipitation threshold for "Rain Likely" classification (mm/day)
RAIN_THRESHOLD_MM = 1.0
 
# Figure output settings
FIG_DPI    = 150
FIG_FORMAT = "png"
 
# Folders wiped and recreated on every run of run_all_scripts.py
CLEAN_DIRS = ["figures", "tables", "results", "models"]
 
# ==============================================================
# SECTION 8: LITERATURE COMPARISON TABLE
# Add or edit entries to update table8_literature_comparison.csv
# ==============================================================
LITERATURE_REFS = [
    {
        "Author"  : "Sen et al. (2023)",
        "Method"  : "PSO + GRU",
        "Dataset" : "Ontario, Canada (10 yr)",
        "MAE"     : "N/A",
        "RMSE"    : "N/A",
        "R2"      : "N/A",
        "MSE"     : "Reported",
        "MAPE"    : "1.15%",
    },
    {
        "Author"  : "Xu et al. (2022)",
        "Method"  : "PSO + LSTM",
        "Dataset" : "Rainfall-Runoff",
        "MAE"     : "N/A",
        "RMSE"    : "Reported",
        "R2"      : "Reported",
        "MSE"     : "N/A",
        "MAPE"    : "N/A",
    },
    {
        "Author"  : "This Study",
        "Method"  : "PSO + Random Forest",
        "Dataset" : f"{COUNTRY_NAME} ({len(STATES)} states, 25 yr)",
        "MAE"     : "TBD",
        "RMSE"    : "TBD",
        "R2"      : "TBD",
        "MSE"     : "TBD",
        "MAPE"    : "TBD",
    },
]
