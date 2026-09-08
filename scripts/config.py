"""Shared configuration for the NYISO pipeline.

Everything that both ingest and preprocess (and later, training) need to agree
on lives here. The feature list in particular: the classic way to break a
forecasting pipeline is to let the training script and the serving path drift
apart on which columns exist and in what order.
"""

import os
from pathlib import Path

import pandas as pd

# --- paths ---------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"

RAW_DEMAND = RAW_DIR / "demand_hourly.parquet"
RAW_WEATHER = RAW_DIR / "weather_daily.parquet"

DAILY_FEATURES = DATA_DIR / "nyiso_daily_features.csv"
WEEKLY_DEMAND = DATA_DIR / "nyiso_weekly_demand.csv"

# --- sources -------------------------------------------------------------
# NYISO respondent code in EIA's RTO dataset.
EIA_RESPONDENT = "NYIS"
# LaGuardia Airport - good proxy for NYC-area temps, long unbroken history.
NOAA_STATION_ID = "GHCND:USW00014732"

START_DATE = "2019-01-01"

# EIA revises recently published demand figures, so an append-only ingest would
# freeze the first (wrong) value it ever saw. Re-pull this many days back from
# the local high-water mark on every incremental run and overwrite.
REVISION_OVERLAP_DAYS = 7

# --- features ------------------------------------------------------------
BALANCE_POINT_F = 65  # HDD/CDD split; see the U-curve in section 4 of the notebook

TARGET_COL = "daily_peak_mw"

FEATURE_COLS = [
    "temp_max_f", "temp_min_f", "temp_avg_f", "precip_in", "hdd", "cdd", "temp_lag_1",
    "day_of_week", "is_weekend", "month", "day_of_year", "is_holiday",
    "peak_lag_1", "peak_lag_7", "peak_lag_14", "peak_roll_mean_7", "peak_roll_mean_14",
]


# --- training ------------------------------------------------------------
EXPERIMENT_NAME = "nyiso-daily-peak-demand"

# Local tracking store. Point MLFLOW_TRACKING_URI at a real server when this
# leaves the laptop; artifacts are what grow (~750 KB per logged model), not
# the metadata.
DEFAULT_TRACKING_URI = f"sqlite:///{ROOT / 'mlflow.db'}"

VALIDATION_WEEKS = 8

XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
}


def tracking_uri():
    return os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)


def load_dotenv(path=None):
    """Minimal .env loader so we don't need python-dotenv."""
    p = Path(path) if path else ROOT / ".env"
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip("'\""))


def require_keys(need_noaa=True):
    """Fail fast with a useful message rather than a confusing 403 later."""
    load_dotenv()
    eia = os.environ.get("EIA_API_KEY", "")
    noaa = os.environ.get("NOAA_TOKEN", "")

    if not eia or eia.startswith("PASTE_"):
        raise SystemExit(
            "EIA_API_KEY is not set. Get a free key at "
            "https://www.eia.gov/opendata/register.php then add it to .env"
        )
    if need_noaa and (not noaa or noaa.startswith("PASTE_")):
        raise SystemExit(
            "NOAA_TOKEN is not set. Request one at "
            "https://www.ncdc.noaa.gov/cdo-web/token then add it to .env"
        )
    return eia, noaa


def today():
    return pd.Timestamp.today().normalize()
