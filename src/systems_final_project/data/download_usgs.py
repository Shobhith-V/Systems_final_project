import argparse
import os
import sys
import time
from datetime import date, datetime, timezone

import pandas as pd
import requests

from .config import GAUGES, PARAMETER_CODE, START_DATE, END_DATE, USGS_DIR

BASE = "https://api.waterdata.usgs.gov/ogcapi/v1"
PAGE_LIMIT = 10_000
MISSING_SENTINELS = {-999999, -999999.0}


def session() -> requests.Session:
    s = requests.Session()
    key = os.environ.get("USGS_API_KEY")
    if key:
        s.headers["X-Api-Key"] = key
    else:
        print("WARNING: no USGS_API_KEY set; limited to 50 requests/hour.", file=sys.stderr)
    s.headers["User-Agent"] = "cmu-river-forecast-project"
    return s


def get_json(s: requests.Session, url: str, params: dict | None = None, retries: int = 5) -> dict:
    for attempt in range(retries):
        r = s.get(url, params=params, timeout=120)
        if r.status_code == 429:
            wait = int(r.headers.get("Retry-After", 60))
            print(f"  rate limited, sleeping {wait}s")
            time.sleep(wait)
            continue
        if r.status_code >= 500:
            time.sleep(5 * (attempt + 1))
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"Failed after {retries} retries: {url}")


def site_name(s: requests.Session, site: str) -> str:
    url = f"{BASE}/collections/monitoring-locations/items/USGS-{site}"
    try:
        props = get_json(s, url, {"f": "json"})["properties"]
        return props.get("monitoring_location_name", "?")
    except Exception as e:  # noqa: BLE001
        return f"(lookup failed: {e})"


def water_years(start: date, end: date):
    """Yield (wy, start_dt, end_dt) chunks clipped to [start, end]."""
    wy = start.year + 1 if start.month >= 10 else start.year
    while True:
        wy_start = datetime(wy - 1, 10, 1, tzinfo=timezone.utc)
        wy_end = datetime(wy, 10, 1, tzinfo=timezone.utc)
        lo = max(wy_start, datetime(start.year, start.month, start.day, tzinfo=timezone.utc))
        hi = min(wy_end, datetime(end.year, end.month, end.day, tzinfo=timezone.utc))
        if lo >= hi:
            break
        yield wy, lo, hi
        wy += 1


def fetch_range(s: requests.Session, site: str, lo: datetime, hi: datetime) -> pd.DataFrame:
    url = f"{BASE}/collections/continuous/items"
    params = {
        "f": "json",
        "monitoring_location_id": f"USGS-{site}",
        "parameter_code": PARAMETER_CODE,
        "time": f"{lo:%Y-%m-%dT%H:%M:%SZ}/{hi:%Y-%m-%dT%H:%M:%SZ}",
        "limit": PAGE_LIMIT,
        "skipGeometry": "true",
    }
    rows = []
    data = get_json(s, url, params)
    while True:
        for feat in data.get("features", []):
            p = feat["properties"]
            rows.append({
                "time": p.get("time"),
                "value": p.get("value"),
                "approval_status": p.get("approval_status"),
                "qualifier": p.get("qualifier"),
            })
        nxt = next((l["href"] for l in data.get("links", []) if l.get("rel") == "next"), None)
        if not nxt or not data.get("features"):
            break
        data = get_json(s, nxt)  # API key travels in the session header

    df = pd.DataFrame(rows, columns=["time", "value", "approval_status", "qualifier"])
    if df.empty:
        return df
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df.loc[df["value"].isin(MISSING_SENTINELS), "value"] = pd.NA
    df["qualifier"] = df["qualifier"].astype("string")
    return df.sort_values("time").drop_duplicates("time").reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gauges", nargs="*", default=list(GAUGES), choices=list(GAUGES))
    ap.add_argument("--overwrite", action="store_true", help="re-download existing files")
    args = ap.parse_args()

    s = session()
    for name in args.gauges:
        site = GAUGES[name]
        print(f"\n== {name} ({site}): {site_name(s, site)}")
        out_dir = USGS_DIR / name
        out_dir.mkdir(parents=True, exist_ok=True)

        for wy, lo, hi in water_years(START_DATE, END_DATE):
            out = out_dir / f"WY{wy}.parquet"
            # Always refresh the current (incomplete) water year.
            is_current = hi.date() >= END_DATE
            if out.exists() and not args.overwrite and not is_current:
                print(f"  WY{wy}: exists, skipping")
                continue
            df = fetch_range(s, site, lo, hi)
            if df.empty:
                print(f"  WY{wy}: NO DATA (check site / parameter code)")
                continue
            df.to_parquet(out, index=False)
            print(f"  WY{wy}: {len(df):>6} rows, {df['value'].isna().mean():.1%} missing")


if __name__ == "__main__":
    main()