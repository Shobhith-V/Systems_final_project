"""Turn the raw downloads into one clean hourly table: data/clean/hourly.parquet.

Script version of notebooks/data_preprocessing.ipynb (the notebook keeps the
exploration and plots; this file only does what the saved table needs).

Steps, in order:
  1. Load every gauge's water-year files; drop duplicate timestamps (each water-year
     file repeats Oct 1 00:00).
  2. Resample each gauge to hourly. The value at hour t is the mean of readings in
     (t-1h, t], so a row never contains readings from after t.
  3. Put every gauge on one complete hourly UTC grid (missing hours become NaN).
  4. Forward-fill gaps of <= MAX_FILL_HOURS and flag them in <gauge>_filled.
     Longer gaps stay NaN. Forward-fill (not interpolation) so a filled value never
     uses a reading from after it.
  5. post_dam3 flag (Locks and Dam 3 demolition began 2024-07-08). For analysis only:
     it coincides with "no high water since", so it is not a model input.
  6. Join observed and forecast rain; cap implausible forecast hours; basin mean.
  7. water_year (Oct 1 - Sep 30, named by the year it ends).
  8. Checks, then save.

No spike removal: the largest jumps are the first reading after an outage or real
rapid rises, and a "differs from both neighbours" rule would delete the 27.5 ft
reading at the 2018-02-17 flood peak (it sits between two outages).

Usage (from the repository root):
    uv run python -m systems_final_project.data.clean
"""

import argparse
from pathlib import Path

import pandas as pd

from .config import CLEAN_HOURLY, GAUGES, RAIN_DIR, RAIN_FCST_DIR, RAIN_LOCATIONS, USGS_DIR

MAX_FILL_HOURS = 3
DAM3_DATE = pd.Timestamp("2024-07-08", tz="UTC")
FCST_CAP_MM = 50.0          # largest observed hourly rain is ~21 mm; forecasts reach 132.8 mm
POINT_RANGE_FT = (10, 40)   # sanity bounds for the target


def load_gauge(name: str) -> pd.Series:
    """All readings of one gauge (native 5/15-min resolution), duplicates dropped."""
    files = sorted((USGS_DIR / name).glob("WY*.parquet"))
    if not files:
        raise FileNotFoundError(f"no files in {USGS_DIR / name}; run data.download_usgs first")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = df.sort_values("time").drop_duplicates("time", keep="last")
    return df.set_index("time")["value"]


def to_hourly(s: pd.Series) -> pd.Series:
    """Mean of readings in (t-1h, t], labelled t."""
    return s.resample("1h", label="right", closed="right").mean()


def fill_short_gaps(s: pd.Series, max_hours: int = MAX_FILL_HOURS) -> tuple[pd.Series, pd.Series]:
    """Forward-fill NaN runs of length <= max_hours. Returns (filled, was_filled)."""
    is_na = s.isna()
    run_id = (is_na != is_na.shift()).cumsum()
    run_len = is_na.groupby(run_id).transform("sum")
    short = is_na & (run_len <= max_hours)
    filled = s.where(~short, s.ffill())
    return filled, short & filled.notna()


def load_rain(folder: Path, column: str) -> pd.DataFrame:
    return pd.DataFrame({
        loc: pd.read_parquet(folder / f"{loc}.parquet").set_index("time")[column]
        for loc in RAIN_LOCATIONS
    })


def build() -> pd.DataFrame:
    # 1-3. gauges -> hourly -> complete grid
    hourly = pd.DataFrame({g: to_hourly(load_gauge(g)) for g in GAUGES})
    grid = pd.date_range(hourly.index.min(), hourly.index.max(), freq="1h")
    hourly = hourly.reindex(grid)
    hourly.index.name = "time"

    # 4. short gaps
    for g in GAUGES:
        hourly[g], hourly[f"{g}_filled"] = fill_short_gaps(hourly[g])

    # 5. regime flag
    hourly["post_dam3"] = hourly.index >= DAM3_DATE

    # 6. rain (both sources are hourly totals labelled at the end of the hour)
    obs = load_rain(RAIN_DIR, "precip_mm")
    fcst = load_rain(RAIN_FCST_DIR, "precip_fcst_24h_mm")
    for loc in RAIN_LOCATIONS:
        hourly[f"rain_{loc}"] = obs[loc].reindex(hourly.index)
        hourly[f"rain_fcst_{loc}"] = fcst[loc].reindex(hourly.index).clip(upper=FCST_CAP_MM)
    hourly["rain_basin_mean"] = hourly[[f"rain_{loc}" for loc in RAIN_LOCATIONS]].mean(axis=1)

    # 7. water year
    hourly["water_year"] = hourly.index.year + (hourly.index.month >= 10)
    return hourly


def check(hourly: pd.DataFrame) -> None:
    idx = hourly.index
    assert idx.is_unique, "duplicate hours"
    assert idx.is_monotonic_increasing, "not sorted"
    assert (idx.to_series().diff().dropna() == pd.Timedelta("1h")).all(), "missing hours in the grid"
    assert str(idx.tz) == "UTC", "timezone is not UTC"
    lo, hi = POINT_RANGE_FT
    assert hourly["point"].dropna().between(lo, hi).all(), "Point level out of plausible range"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=CLEAN_HOURLY)
    args = ap.parse_args()

    hourly = build()
    check(hourly)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    hourly.to_parquet(args.out)

    print(f"{len(hourly):,} hours  {hourly.index.min()}  ->  {hourly.index.max()}")
    filled = {g: int(hourly[f"{g}_filled"].sum()) for g in GAUGES}
    missing = {g: int(hourly[g].isna().sum()) for g in GAUGES}
    print("filled hours: ", filled)
    print("missing hours:", missing)
    print(f"saved {args.out}  ({args.out.stat().st_size / 1e6:.1f} MB, {hourly.shape[1]} columns)")


if __name__ == "__main__":
    main()
