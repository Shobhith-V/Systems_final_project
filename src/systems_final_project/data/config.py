from datetime import date, timedelta
from pathlib import Path

# Repo root = three levels up from this file (src/systems_final_project/data/config.py)
REPO_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = REPO_ROOT / "data" / "raw"
USGS_DIR = RAW_DIR / "usgs"
RAIN_DIR = RAW_DIR / "rain_observed"
RAIN_FCST_DIR = RAW_DIR / "rain_forecast"


GAUGES = {
    "point":       "03085152",  # Monongahela River at Point State Park (TARGET)
    "braddock":    "03085000",  # Monongahela River at Braddock
    "elizabeth":   "03075070",  # Monongahela River at Elizabeth
    "sutersville": "03083500",  # Youghiogheny River at Sutersville
    "natrona":     "03049500",  # Allegheny River at Natrona
    "kittanning":  "03036500",  # Allegheny River at Kittanning
}
PARAMETER_CODE = "00065" # this is the parameter code for gauge height, feet

# Water years run Oct 1 -> Sep 30. WY2016 = 2015-10-01 .. 2016-09-30.
START_DATE = date(2015, 10, 1)
END_DATE = date.today()

# ---------------------------------------------------------------------------
# Rain locations (lat, lon)
# ---------------------------------------------------------------------------
RAIN_LOCATIONS = {
    "pittsburgh":  (40.4406, -79.9959),
    "kittanning":  (40.8165, -79.5214),
    "elizabeth":   (40.2690, -79.8895),
    "sutersville": (40.2370, -79.8028),
}

# Open-Meteo archive (ERA5-based) lags real time by ~5 days.
RAIN_END_DATE = date.today() - timedelta(days=6)

# Archived forecasts (Previous Runs API) only exist from ~2024 for most models.
RAIN_FCST_START_DATE = date(2024, 1, 1)

# ---------------------------------------------------------------------------
# Modelling inputs and outputs (features.py, models.py, evaluation.py)
# ---------------------------------------------------------------------------
CLEAN_HOURLY = REPO_ROOT / "data" / "clean" / "hourly.parquet"  # from data_preprocessing.ipynb
FEATURES_DIR = REPO_ROOT / "data" / "features"                  # features.parquet, feature_groups.json
RESULTS_DIR = REPO_ROOT / "results"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"                   # <model>.parquet held-out forecasts
FIGURES_DIR = RESULTS_DIR / "figures"                           # fig1_accuracy, fig2_april_2024
