import argparse
from pathlib import Path
import pandas as pd
from .config import CLEAN_HOURLY, GAUGES, RAIN_DIR, RAIN_FCST_DIR, RAIN_LOCATIONS, USGS_DIR

MAX_FILL_HOURS = 3
DAM3_DATE = pd.Timestamp("2024-07-08", tz="UTC")
FCST_CAP_MM = 50.0
POINT_MIN_FT = 10
POINT_MAX_FT = 40


# ---------------------------------------------------------------------------
# Gauges
# ---------------------------------------------------------------------------

def load_gauge(name):
    """Read every water-year file for one gauge and return its raw readings.

    Readings stay at their native resolution (5 or 15 minutes). If the same
    timestamp appears more than once, only the last copy is kept.
    """
    gauge_folder = USGS_DIR / name
    files = sorted(gauge_folder.glob("WY*.parquet"))

    if len(files) == 0:
        raise FileNotFoundError(f"no files in {gauge_folder}; run data.download_usgs first")

    # Read each file and stack them into one table.
    tables = []
    for file in files:
        table = pd.read_parquet(file)
        tables.append(table)
    readings = pd.concat(tables, ignore_index=True)

    # Sort by time and remove repeated timestamps.
    readings = readings.sort_values("time")
    readings = readings.drop_duplicates("time", keep="last")

    # We only need the time and the value.
    readings = readings.set_index("time")
    values = readings["value"]
    return values


def to_hourly(readings):
    """Average the readings into one value per hour.

    The value labelled 10:00 is the average of readings after 09:00 up to
    and including 10:00. So it never uses anything recorded after 10:00.
    """
    hourly = readings.resample("1h", label="right", closed="right").mean()
    return hourly


def fill_short_gaps(series, max_hours=MAX_FILL_HOURS):
    """Fill short runs of missing values with the last known value.

    A run of missing hours is filled only if it is max_hours long or less.
    Longer runs are left missing. A run at the very start of the data has no
    earlier value to copy, so it is also left missing.

    Returns two series:
      - the values after filling
      - True/False for each hour, True where we filled in a value
    """
    original_values = series.tolist()
    total_hours = len(original_values)

    # Start with a copy of the original values and nothing marked as filled.
    new_values = list(original_values)
    was_filled = []
    for i in range(total_hours):
        was_filled.append(False)

    i = 0
    while i < total_hours:
        # Skip over hours that already have a value.
        if not pd.isna(original_values[i]):
            i = i + 1
            continue

        # We are at the start of a gap. Walk forward to find where it ends.
        gap_start = i
        while i < total_hours and pd.isna(original_values[i]):
            i = i + 1
        gap_end = i  # first hour after the gap (or the end of the data)
        gap_length = gap_end - gap_start

        # Long gaps stay missing.
        if gap_length > max_hours:
            continue

        # A gap at the very beginning has no earlier value to copy.
        if gap_start == 0:
            continue

        # Copy the last value before the gap into every hour of the gap.
        last_known_value = original_values[gap_start - 1]
        for j in range(gap_start, gap_end):
            new_values[j] = last_known_value
            was_filled[j] = True

    filled_series = pd.Series(new_values, index=series.index, dtype="float64")
    was_filled_series = pd.Series(was_filled, index=series.index, dtype="bool")
    return filled_series, was_filled_series


# ---------------------------------------------------------------------------
# Rain
# ---------------------------------------------------------------------------

def load_rain(folder, column):
    """Load one rain column for every location into a single table.

    The result has one column per location, indexed by time.
    """
    rain = pd.DataFrame()
    for location in RAIN_LOCATIONS:
        file_path = folder / f"{location}.parquet"
        table = pd.read_parquet(file_path)
        table = table.set_index("time")
        rain[location] = table[column]
    return rain


# ---------------------------------------------------------------------------
# Build the table
# ---------------------------------------------------------------------------

def build():
    # Steps 1 and 2: load each gauge and turn it into hourly values.
    hourly_by_gauge = {}
    for gauge in GAUGES:
        raw_readings = load_gauge(gauge)
        hourly_by_gauge[gauge] = to_hourly(raw_readings)
    hourly = pd.DataFrame(hourly_by_gauge)

    # Step 3: make sure every hour between the first and last timestamp has a
    # row. Hours with no data become NaN.
    first_hour = hourly.index.min()
    last_hour = hourly.index.max()
    all_hours = pd.date_range(first_hour, last_hour, freq="1h")
    hourly = hourly.reindex(all_hours)
    hourly.index.name = "time"

    # Step 4: fill short gaps and record which hours were filled.
    for gauge in GAUGES:
        filled_values, was_filled = fill_short_gaps(hourly[gauge])
        hourly[gauge] = filled_values
        hourly[gauge + "_filled"] = was_filled

    # Step 5: mark the hours on or after the start of the dam demolition.
    hourly["post_dam3"] = hourly.index >= DAM3_DATE

    # Step 6: rain. Both sources are hourly totals labelled at the end of the
    # hour, which matches the way we labelled the gauge hours.
    observed_rain = load_rain(RAIN_DIR, "precip_mm")
    forecast_rain = load_rain(RAIN_FCST_DIR, "precip_fcst_24h_mm")

    observed_columns = []
    for location in RAIN_LOCATIONS:
        observed_name = "rain_" + location
        forecast_name = "rain_fcst_" + location

        # Line the rain up with our hourly rows.
        observed = observed_rain[location].reindex(hourly.index)
        forecast = forecast_rain[location].reindex(hourly.index)

        # Cap forecast values that are too large to be realistic.
        forecast = forecast.clip(upper=FCST_CAP_MM)

        hourly[observed_name] = observed
        hourly[forecast_name] = forecast
        observed_columns.append(observed_name)

    # Average observed rain across all locations for each hour.
    hourly["rain_basin_mean"] = hourly[observed_columns].mean(axis=1)

    # Step 7: water year. A water year runs from October 1 to September 30
    # and is named after the year it ends in. So October, November and
    # December belong to the next year's water year
    # (for example, October 2023 is in water year 2024).
    water_years = []
    for timestamp in hourly.index:
        if timestamp.month >= 10:
            water_years.append(timestamp.year + 1)
        else:
            water_years.append(timestamp.year)

    # Store as int32, which is the type pandas uses for years.
    hourly["water_year"] = pd.Series(water_years, index=hourly.index, dtype="int32")

    return hourly


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def check(hourly):
    """Basic sanity checks before saving."""
    times = hourly.index

    # Every hour should appear once.
    assert times.is_unique, "duplicate hours"

    # Hours should be in order.
    assert times.is_monotonic_increasing, "not sorted"

    # Each row should be exactly one hour after the one before it.
    gaps_between_rows = times.to_series().diff()
    gaps_between_rows = gaps_between_rows.dropna()
    one_hour = pd.Timedelta("1h")
    all_one_hour = (gaps_between_rows == one_hour).all()
    assert all_one_hour, "missing hours in the grid"

    # Times should be in UTC.
    assert str(times.tz) == "UTC", "timezone is not UTC"

    # River levels at the Point should be within a believable range.
    point_levels = hourly["point"].dropna()
    all_in_range = point_levels.between(POINT_MIN_FT, POINT_MAX_FT).all()
    assert all_in_range, "Point level out of plausible range"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=CLEAN_HOURLY)
    args = parser.parse_args()
    output_path = args.out

    hourly = build()
    check(hourly)

    # Make sure the output folder exists, then save.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    hourly.to_parquet(output_path)

    # Print a short summary of what was produced.
    number_of_hours = len(hourly)
    first_hour = hourly.index.min()
    last_hour = hourly.index.max()
    print(f"{number_of_hours:,} hours  {first_hour}  ->  {last_hour}")

    filled_counts = {}
    missing_counts = {}
    for gauge in GAUGES:
        filled_counts[gauge] = int(hourly[gauge + "_filled"].sum())
        missing_counts[gauge] = int(hourly[gauge].isna().sum())
    print("filled hours: ", filled_counts)
    print("missing hours:", missing_counts)

    size_in_mb = output_path.stat().st_size / 1e6
    number_of_columns = hourly.shape[1]
    print(f"saved {output_path}  ({size_in_mb:.1f} MB, {number_of_columns} columns)")


if __name__ == "__main__":
    main()