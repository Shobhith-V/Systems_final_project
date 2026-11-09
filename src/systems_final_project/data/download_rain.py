"""Download hourly rainfall from Open-Meteo for the 4 rain locations.

1. Observed rain  (Historical Weather / ERA5 archive): 2015-10-01 -> ~6 days ago
   Output: data/raw/rain_observed/<location>.parquet   columns: time, precip_mm
2. Archived forecast rain (Previous Runs API): forecast issued ~24 h before
   each hour. Only available from ~2024.
   Output: data/raw/rain_forecast/<location>.parquet   columns: time, precip_fcst_24h_mm

Usage:
    uv run python -m systems_final_project.data.download_rain
    uv run python -m systems_final_project.data.download_rain --skip-forecast

Open-Meteo is free for non-commercial use with attribution (credit it in the report).
"""

import argparse
import time
from datetime import date

import pandas as pd
import requests

from .config import (
    RAIN_DIR, RAIN_END_DATE, RAIN_FCST_DIR, RAIN_FCST_START_DATE,
    RAIN_LOCATIONS, START_DATE,
)

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
PREV_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"


def get(url: str, params: dict, retries: int = 5) -> dict:
    for attempt in range(retries):
        r = requests.get(url, params=params, timeout=120)
        if r.status_code == 429:
            time.sleep(60)
            continue
        if r.status_code >= 500:
            time.sleep(5 * (attempt + 1))
            continue
        if r.status_code >= 400:
            raise RuntimeError(f"{r.status_code}: {r.text[:300]}")
        return r.json()
    raise RuntimeError(f"Failed after {retries} retries")


def yearly_chunks(start: date, end: date):
    y = start.year
    while True:
        lo = max(start, date(y, 1, 1))
        hi = min(end, date(y, 12, 31))
        if lo > hi:
            break
        yield lo, hi
        y += 1


def fetch(url: str, lat: float, lon: float, start: date, end: date, var: str) -> pd.DataFrame:
    frames = []
    for lo, hi in yearly_chunks(start, end):
        data = get(url, {
            "latitude": lat,
            "longitude": lon,
            "start_date": lo.isoformat(),
            "end_date": hi.isoformat(),
            "hourly": var,
            "timezone": "UTC",
            "precipitation_unit": "mm",
        })
        h = data["hourly"]
        frames.append(pd.DataFrame({"time": h["time"], var: h[var]}))
        time.sleep(1)  # be polite
    df = pd.concat(frames, ignore_index=True)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df.drop_duplicates("time").sort_values("time").reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-forecast", action="store_true")
    args = ap.parse_args()

    RAIN_DIR.mkdir(parents=True, exist_ok=True)
    RAIN_FCST_DIR.mkdir(parents=True, exist_ok=True)

    for name, (lat, lon) in RAIN_LOCATIONS.items():
        print(f"\n== {name} ({lat}, {lon})")

        df = fetch(ARCHIVE_URL, lat, lon, START_DATE, RAIN_END_DATE, "precipitation")
        df = df.rename(columns={"precipitation": "precip_mm"})
        df.to_parquet(RAIN_DIR / f"{name}.parquet", index=False)
        print(f"  observed: {len(df)} rows, {df['precip_mm'].isna().mean():.1%} missing")

        if args.skip_forecast:
            continue
        var = "precipitation_previous_day1"  # value forecast ~24 h before that hour
        df = fetch(PREV_RUNS_URL, lat, lon, RAIN_FCST_START_DATE, date.today(), var)
        df = df.rename(columns={var: "precip_fcst_24h_mm"})
        first_valid = df.loc[df["precip_fcst_24h_mm"].notna(), "time"].min()
        df.to_parquet(RAIN_FCST_DIR / f"{name}.parquet", index=False)
        print(f"  forecast: {len(df)} rows, first non-null {first_valid}")


if __name__ == "__main__":
    main()