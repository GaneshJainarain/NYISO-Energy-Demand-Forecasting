#!/usr/bin/env python
"""Fetch raw NYISO demand (EIA) and NYC weather (NOAA) into data/raw/.

Network-bound and rate-limited, so it is deliberately separate from
preprocessing: you should be able to iterate on features all day without
touching either API.

Incremental by default. It reads the high-water mark out of the existing
parquet and asks only for what is missing, minus a re-pull window because EIA
revises recent figures after publication (see REVISION_OVERLAP_DAYS).

    python scripts/ingest.py              # incremental, both sources
    python scripts/ingest.py --full       # ignore local data, refetch from 2019
    python scripts/ingest.py --source eia # just one source
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

EIA_PAGE_SIZE = 5000   # EIA v2 max rows per request
NOAA_PAGE_SIZE = 1000  # CDO max records per request


# --- EIA -----------------------------------------------------------------

def fetch_eia_demand(api_key, respondent, start_date, end_date, chunk_days=200):
    """Hourly demand (MWh) for an EIA respondent, paginating by date window
    and by offset within each window."""
    base_url = "https://api.eia.gov/v2/electricity/rto/region-data/data/"
    all_rows = []

    edges = pd.date_range(start_date, end_date, freq=f"{chunk_days}D")
    if len(edges) == 0:
        edges = pd.DatetimeIndex([pd.Timestamp(start_date)])
    if edges[-1] < pd.Timestamp(end_date):
        edges = edges.append(pd.DatetimeIndex([pd.Timestamp(end_date)]))
    if len(edges) == 1:
        edges = edges.append(pd.DatetimeIndex([pd.Timestamp(end_date)]))

    for i in range(len(edges) - 1):
        window_start = edges[i]
        # Half-open windows: EIA treats both ends as inclusive, so every window
        # but the last stops a day short of the next one's start.
        window_end = edges[i + 1] if i == len(edges) - 2 else edges[i + 1] - pd.Timedelta(days=1)
        ws, we = window_start.strftime("%Y-%m-%d"), window_end.strftime("%Y-%m-%d")

        offset, window_rows = 0, 0
        while True:
            params = {
                "api_key": api_key,
                "frequency": "hourly",
                "data[0]": "value",
                "facets[respondent][0]": respondent,
                "facets[type][0]": "D",  # D = Demand
                "start": ws,
                "end": we,
                "sort[0][column]": "period",
                "sort[0][direction]": "asc",
                "offset": offset,
                "length": EIA_PAGE_SIZE,
            }
            resp = requests.get(base_url, params=params, timeout=30)
            if not resp.ok:
                raise SystemExit(
                    f"EIA request failed ({resp.status_code}) for {ws} -> {we}: "
                    f"{resp.text[:500]}"
                )
            rows = resp.json().get("response", {}).get("data", [])
            all_rows.extend(rows)
            window_rows += len(rows)

            if len(rows) < EIA_PAGE_SIZE:
                break
            offset += EIA_PAGE_SIZE
            time.sleep(0.3)

        print(f"  {ws} -> {we}: {window_rows} rows (total {len(all_rows)})")
        time.sleep(0.3)  # be polite

    if not all_rows:
        return pd.DataFrame(columns=["datetime", "demand_mwh"]).set_index("datetime")

    df = pd.DataFrame(all_rows)[["period", "value"]]
    df.columns = ["datetime", "demand_mwh"]
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["demand_mwh"] = pd.to_numeric(df["demand_mwh"], errors="coerce")
    return df.dropna().drop_duplicates(subset="datetime").sort_values("datetime").set_index("datetime")


# --- NOAA ----------------------------------------------------------------

def fetch_noaa_daily(token, station_id, start_date, end_date):
    """Daily TMAX/TMIN/PRCP for a station, looping year by year because the
    CDO API caps each request at a 1-year range."""
    base_url = "https://www.ncei.noaa.gov/cdo-web/api/v2/data"
    headers = {"token": token}
    all_rows = []

    years = pd.date_range(start_date, end_date, freq="YS")
    if len(years) == 0 or years[0] > pd.Timestamp(start_date):
        years = pd.DatetimeIndex([pd.Timestamp(start_date)]).append(years)

    for year_start in years:
        year_end = min(year_start + pd.DateOffset(years=1) - pd.Timedelta(days=1),
                       pd.Timestamp(end_date))
        if year_end < year_start:
            continue
        offset = 1
        while True:
            params = {
                "datasetid": "GHCND",
                "stationid": station_id,
                "startdate": year_start.strftime("%Y-%m-%d"),
                "enddate": year_end.strftime("%Y-%m-%d"),
                "datatypeid": ["TMAX", "TMIN", "PRCP"],
                "units": "standard",
                "limit": NOAA_PAGE_SIZE,
                "offset": offset,
            }
            resp = requests.get(base_url, headers=headers, params=params, timeout=30)
            resp.raise_for_status()
            rows = resp.json().get("results", [])
            if not rows:
                break
            all_rows.extend(rows)
            if len(rows) < NOAA_PAGE_SIZE:
                break
            offset += NOAA_PAGE_SIZE
            time.sleep(0.3)

        print(f"  {year_start.year}: total rows {len(all_rows)}")
        time.sleep(0.3)

    if not all_rows:
        return pd.DataFrame()

    raw = pd.DataFrame(all_rows)
    raw["date"] = pd.to_datetime(raw["date"]).dt.normalize()
    wide = raw.pivot_table(index="date", columns="datatype", values="value", aggfunc="first")
    wide.columns = [c.lower() for c in wide.columns]
    return wide.rename(columns={"tmax": "temp_max_f", "tmin": "temp_min_f", "prcp": "precip_in"})


# --- incremental plumbing ------------------------------------------------

def read_existing(path):
    if path.exists():
        return pd.read_parquet(path)
    return None


def resume_point(existing, full):
    """Where to start fetching: the local high-water mark minus the revision
    window, or START_DATE for a full refresh / cold start."""
    if full or existing is None or existing.empty:
        return pd.Timestamp(config.START_DATE), "full"
    high_water = existing.index.max().normalize()
    start = high_water - pd.Timedelta(days=config.REVISION_OVERLAP_DAYS)
    return max(start, pd.Timestamp(config.START_DATE)), "incremental"


def upsert(existing, fresh):
    """Combine old and new, letting freshly fetched rows win on overlap -
    that is what picks up EIA's revisions."""
    if existing is None or existing.empty:
        return fresh.sort_index()
    if fresh is None or fresh.empty:
        return existing.sort_index()
    combined = pd.concat([existing, fresh])
    combined = combined[~combined.index.duplicated(keep="last")]
    return combined.sort_index()


def run_source(name, path, fetch, full, end):
    existing = read_existing(path)
    start, mode = resume_point(existing, full)
    before = 0 if existing is None else len(existing)

    if start.normalize() > pd.Timestamp(end).normalize():
        print(f"{name}: already current through {existing.index.max():%Y-%m-%d}, nothing to do")
        return

    print(f"{name}: {mode} fetch from {start:%Y-%m-%d} to {end:%Y-%m-%d} "
          f"(have {before:,} rows)")
    fresh = fetch(start.strftime("%Y-%m-%d"), pd.Timestamp(end).strftime("%Y-%m-%d"))
    merged = upsert(existing, fresh)

    path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(path)
    print(f"{name}: {before:,} -> {len(merged):,} rows "
          f"(+{len(merged) - before:,}), {merged.index.min():%Y-%m-%d} to "
          f"{merged.index.max():%Y-%m-%d} -> {path.relative_to(config.ROOT)}\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--full", action="store_true",
                    help="ignore local data and refetch the whole history")
    ap.add_argument("--source", choices=["eia", "noaa", "both"], default="both")
    ap.add_argument("--end", default=None, help="end date (default: today)")
    args = ap.parse_args()

    end = pd.Timestamp(args.end) if args.end else config.today()
    need_noaa = args.source in ("noaa", "both")
    eia_key, noaa_token = config.require_keys(need_noaa=need_noaa)

    if args.source in ("eia", "both"):
        run_source("EIA demand", config.RAW_DEMAND,
                   lambda s, e: fetch_eia_demand(eia_key, config.EIA_RESPONDENT, s, e),
                   args.full, end)

    if need_noaa:
        run_source("NOAA weather", config.RAW_WEATHER,
                   lambda s, e: fetch_noaa_daily(noaa_token, config.NOAA_STATION_ID, s, e),
                   args.full, end)


if __name__ == "__main__":
    main()
