#!/usr/bin/env python
"""Turn raw demand + weather into the modelling table.

Pure and deterministic - no network, no API keys. Reads data/raw/, writes
data/nyiso_daily_features.csv and data/nyiso_weekly_demand.csv. Safe to re-run
as often as you like while iterating on features.

    python scripts/preprocess.py
    python scripts/preprocess.py --check   # compare against existing CSV, write nothing
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config


def aggregate_targets(demand_hourly):
    """Hourly demand -> the two prediction targets.

    NOTE: EIA's `hourly` frequency is UTC and is not localised anywhere in this
    pipeline, so a "day" here runs 19:00-19:00 local. The evening peak sits near
    that boundary, so a minority of days book their peak against the neighbouring
    calendar date. Switching the ingest to `local-hourly` is the fix; it would
    re-score everything downstream.
    """
    daily = demand_hourly.resample("D").agg(
        daily_peak_mw=("demand_mwh", "max"),
        daily_total_mwh=("demand_mwh", "sum"),
    ).dropna()
    daily.index.name = "date"

    weekly = demand_hourly.resample("W-SUN").agg(
        weekly_total_mwh=("demand_mwh", "sum"),
    ).dropna()

    return daily, weekly


def add_features(df):
    """Calendar, autoregressive and weather-derived features.

    Every autoregressive term is shifted before any rolling window, so no
    feature carries information from the day being predicted. That is the
    single easiest mistake to make here and it produces spectacular validation
    scores that vanish in production.
    """
    df = df.copy()

    # Calendar
    df["day_of_week"] = df.index.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["month"] = df.index.month
    df["day_of_year"] = df.index.dayofyear

    try:
        import holidays as holidays_pkg
        us_holidays = holidays_pkg.US(
            years=range(df.index.year.min(), df.index.year.max() + 1))
        df["is_holiday"] = df.index.to_series().apply(lambda d: d in us_holidays).astype(int)
    except ImportError:
        df["is_holiday"] = 0

    # Autoregressive - yesterday, same weekday last week, two weeks back
    df["peak_lag_1"] = df["daily_peak_mw"].shift(1)
    df["peak_lag_7"] = df["daily_peak_mw"].shift(7)
    df["peak_lag_14"] = df["daily_peak_mw"].shift(14)
    df["peak_roll_mean_7"] = df["daily_peak_mw"].shift(1).rolling(7).mean()
    df["peak_roll_mean_14"] = df["daily_peak_mw"].shift(1).rolling(14).mean()

    # Weather - split the U-curve into two half-line variables so a model can
    # put a different slope on the heating and cooling arms.
    bp = config.BALANCE_POINT_F
    df["hdd"] = (bp - df["temp_avg_f"]).clip(lower=0)
    df["cdd"] = (df["temp_avg_f"] - bp).clip(lower=0)
    df["temp_lag_1"] = df["temp_avg_f"].shift(1)

    return df


def build():
    if not config.RAW_DEMAND.exists() or not config.RAW_WEATHER.exists():
        raise SystemExit(
            "Missing raw data. Run `python scripts/ingest.py` first "
            f"(expected {config.RAW_DEMAND.name} and {config.RAW_WEATHER.name} "
            f"in {config.RAW_DIR})."
        )

    demand_hourly = pd.read_parquet(config.RAW_DEMAND)
    weather_daily = pd.read_parquet(config.RAW_WEATHER)
    print(f"raw demand:  {len(demand_hourly):,} hourly rows "
          f"({demand_hourly.index.min():%Y-%m-%d} to {demand_hourly.index.max():%Y-%m-%d})")
    print(f"raw weather: {len(weather_daily):,} daily rows")

    daily, weekly = aggregate_targets(demand_hourly)

    weather = weather_daily.copy()
    weather["temp_avg_f"] = (weather["temp_max_f"] + weather["temp_min_f"]) / 2

    df = daily.join(weather, how="inner").sort_index()
    print(f"merged:      {df.shape[0]:,} days x {df.shape[1]} cols")

    df_feat = add_features(df).dropna()
    print(f"featured:    {df_feat.shape[0]:,} days x {df_feat.shape[1]} cols "
          f"({df_feat.index.min():%Y-%m-%d} to {df_feat.index.max():%Y-%m-%d})")

    missing = [c for c in config.FEATURE_COLS if c not in df_feat.columns]
    if missing:
        raise SystemExit(f"feature columns missing after build: {missing}")

    return df_feat, weekly


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="compare against the committed CSV instead of overwriting it")
    args = ap.parse_args()

    df_feat, weekly = build()

    if args.check:
        if not config.DAILY_FEATURES.exists():
            raise SystemExit(f"nothing to compare against: {config.DAILY_FEATURES} missing")
        ref = pd.read_csv(config.DAILY_FEATURES, index_col=0, parse_dates=True)
        common = ref.index.intersection(df_feat.index)
        cols = [c for c in ref.columns if c in df_feat.columns]
        diff = (ref.loc[common, cols] - df_feat.loc[common, cols]).abs().max()
        worst = diff.max()
        print(f"\n--check: {len(common):,} overlapping rows, {len(cols)} columns")
        print(f"largest absolute difference: {worst:.10g}")
        print("MATCH" if worst < 1e-6 else "MISMATCH - see per-column deltas:")
        if worst >= 1e-6:
            print(diff[diff > 1e-6].to_string())
        return

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    df_feat.to_csv(config.DAILY_FEATURES)
    weekly.to_csv(config.WEEKLY_DEMAND)
    print(f"\nwrote {config.DAILY_FEATURES.relative_to(config.ROOT)} "
          f"({len(df_feat):,} rows)")
    print(f"wrote {config.WEEKLY_DEMAND.relative_to(config.ROOT)} ({len(weekly):,} rows)")


if __name__ == "__main__":
    main()
