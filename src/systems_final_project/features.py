"""
Build the model feature table from data/clean/hourly.parquet.

Each row is one forecast issue time t (hourly, UTC). Every feature only uses
information that is available at time t. The targets are the Point level at
t+6, t+12 and t+24 hours.

Feature groups (these are listed in feature_groups.json):

  point          Point's own level now, 6 h ago and 24 h ago, its recent
                 change, and its maximum over the last 24 h.

  upstream       Natrona, Kittanning, Sutersville: level now, 6 h ago and
                 12 h ago, plus recent changes.
                 Braddock: changes only, because the Dam 3 removal shifted its
                 level (Jul 2024).

  upstream_eliz  Elizabeth: changes only (same Dam 3 reason; the gauge has
                 also been offline since Jun 2026). Only used in the ablation,
                 not in the main models.

  rain_obs       Observed rain totals over the past 6/24/48/72 h, for each
                 location and for the basin mean.

  season         Day of the year, as sin and cos.

  flags          Flags marking hours where a short gap was filled.

  info           post_dam3. Kept for analysis but NOT used as a model input:
                 there has been no high water at any gauge since Jul 2024
                 (including Allegheny gauges the dam can't affect), so the
                 flag would teach the model "post-dam = low water".

Targets, for each horizon h:
  y_{h}h     Point level at t+h (ft)
  dy_{h}h    y_{h}h - point(t). This change is what the models are trained
             to predict.

There is also a water_year column (the water year of the issue time t), which
is used to split the data into training and test sets.

Run from the repository root:
    uv run python -m systems_final_project.features [--flood-ft 18]

Other modules read the result with load_features().

Outputs:
    data/features/features.parquet
    data/features/feature_groups.json   (flood_ft, horizons, feature_groups, targets)
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from systems_final_project.data.config import CLEAN_HOURLY, FEATURES_DIR

# How many hours ahead we forecast.
HORIZONS = (6, 12, 24)

# NWS action stage for the Ohio River at Pittsburgh.
# (The flood stage has been 22 ft since Oct 2024.)
FLOOD_FT = 18.0

# Locations with rain data.
RAIN_LOCS = ("pittsburgh", "kittanning", "elizabeth", "sutersville")

# Rain totals are summed over these many past hours.
RAIN_WINDOWS = (6, 24, 48, 72)

# Gauges whose datum is stable, so we can use their actual levels.
LEVEL_GAUGES = ("natrona", "kittanning", "sutersville")

# Gauges whose datum was shifted by Dam 3, so we only use their changes.
DELTA_GAUGES = ("braddock",)

# All feature groups, in order.
GROUPS = ("point", "upstream", "upstream_eliz", "rain_obs", "season", "flags", "info")


def load_features(features_dir=FEATURES_DIR):
    """Load the feature table and its metadata.

    The metadata contains flood_ft, horizons, feature_groups and targets.
    """
    meta_text = (features_dir / "feature_groups.json").read_text()
    meta = json.loads(meta_text)

    features = pd.read_parquet(features_dir / "features.parquet")
    return features, meta


def load_hourly(path):
    """Load the cleaned hourly data and check it has no missing hours.

    This matters because the features use shift(n) to look n rows back,
    which only means "n hours back" if every hour has a row.
    """
    df = pd.read_parquet(path)
    times = df.index

    assert times.is_unique, "index must be unique and sorted"
    assert times.is_monotonic_increasing, "index must be unique and sorted"

    steps = times.to_series().diff()
    steps = steps.dropna()
    every_step_is_one_hour = (steps == pd.Timedelta("1h")).all()
    assert every_step_is_one_hour, "hourly grid has holes: shift(n) would not mean n hours"

    return df


def past_sum(s, hours):
    """Total over the last `hours` hours, i.e. (t-hours, t].

    The result is NaN if any hour in that window is missing.
    """
    return s.rolling(hours, min_periods=hours).sum()


def add_column(columns, groups, group, name, values):
    """Add one feature column and record which group it belongs to."""
    columns[name] = values
    groups[group].append(name)


def build(df, flood_ft):
    """Build the feature table from the cleaned hourly data.

    Returns the feature table and its metadata.
    """
    # All the output columns, in order: {column name: values}.
    columns = {}

    # Which columns belong to each feature group: {group: [column names]}.
    groups = {}
    for group in GROUPS:
        groups[group] = []

    # --- Point's own history ---------------------------------------------------
    point = df["point"]
    add_column(columns, groups, "point", "point", point)
    add_column(columns, groups, "point", "point_lag6", point.shift(6))
    add_column(columns, groups, "point", "point_lag24", point.shift(24))
    add_column(columns, groups, "point", "point_d1", point - point.shift(1))
    add_column(columns, groups, "point", "point_d6", point - point.shift(6))

    # Highest level in the last 24 hours (needs at least 18 of them to be known).
    point_max24 = point.rolling(24, min_periods=18).max()
    add_column(columns, groups, "point", "point_max24", point_max24)

    # --- Upstream gauges -------------------------------------------------------
    # Gauges with a stable datum: levels and changes.
    for gauge in LEVEL_GAUGES:
        level = df[gauge]
        add_column(columns, groups, "upstream", gauge, level)
        add_column(columns, groups, "upstream", f"{gauge}_lag6", level.shift(6))
        add_column(columns, groups, "upstream", f"{gauge}_lag12", level.shift(12))
        add_column(columns, groups, "upstream", f"{gauge}_d1", level - level.shift(1))
        add_column(columns, groups, "upstream", f"{gauge}_d6", level - level.shift(6))

    # Gauges whose datum shifted: changes only.
    for gauge in DELTA_GAUGES:
        level = df[gauge]
        add_column(columns, groups, "upstream", f"{gauge}_d1", level - level.shift(1))
        add_column(columns, groups, "upstream", f"{gauge}_d6", level - level.shift(6))
        add_column(columns, groups, "upstream", f"{gauge}_d12", level - level.shift(12))

    # Elizabeth (ablation only): changes over 1, 6 and 12 hours.
    elizabeth = df["elizabeth"]
    for hours_back in (1, 6, 12):
        change = elizabeth - elizabeth.shift(hours_back)
        add_column(columns, groups, "upstream_eliz", f"elizabeth_d{hours_back}", change)

    # --- Observed rain (past only) ---------------------------------------------
    rain_columns = list(RAIN_LOCS) + ["basin_mean"]
    for location in rain_columns:
        rain = df[f"rain_{location}"]
        for window in RAIN_WINDOWS:
            total = past_sum(rain, window)
            add_column(columns, groups, "rain_obs", f"rain_{location}_past{window}h", total)

    # --- Season ----------------------------------------------------------------
    # Day of the year as a point on a circle, so Dec 31 and Jan 1 end up close.
    day_of_year = df.index.day_of_year.to_numpy()
    angle = 2 * np.pi * day_of_year / 365.25
    add_column(columns, groups, "season", "doy_sin", np.sin(angle))
    add_column(columns, groups, "season", "doy_cos", np.cos(angle))

    # --- Info and flags --------------------------------------------------------
    add_column(columns, groups, "info", "post_dam3", df["post_dam3"].astype(int))

    for gauge in ("point", "braddock", "natrona", "kittanning", "sutersville"):
        was_filled = df[f"{gauge}_filled"].astype(int)
        add_column(columns, groups, "flags", f"{gauge}_filled", was_filled)

    # --- Targets ---------------------------------------------------------------
    targets = []
    for h in HORIZONS:
        # The level h hours in the future, and the change from now.
        future_level = point.shift(-h)
        columns[f"y_{h}h"] = future_level
        columns[f"dy_{h}h"] = future_level - point

        targets.append(f"y_{h}h")
        targets.append(f"dy_{h}h")

    # Split key: the water year of the issue time t (not of the target time).
    columns["water_year"] = df["water_year"]

    features = pd.DataFrame(columns, index=df.index)

    meta = {
        "flood_ft": flood_ft,
        "horizons": list(HORIZONS),
        "feature_groups": groups,
        "targets": targets,
    }
    return features, meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", type=Path, default=CLEAN_HOURLY)
    parser.add_argument("--out-dir", type=Path, default=FEATURES_DIR)
    parser.add_argument("--flood-ft", type=float, default=FLOOD_FT)
    args = parser.parse_args()

    hourly = load_hourly(args.clean)
    feats, meta = build(hourly, args.flood_ft)

    # Save the table and its metadata.
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(out_dir / "features.parquet")
    (out_dir / "feature_groups.json").write_text(json.dumps(meta, indent=2))

    # Count the model inputs ("info" columns are not model inputs).
    number_of_features = 0
    for group, group_columns in meta["feature_groups"].items():
        if group != "info":
            number_of_features = number_of_features + len(group_columns)

    number_of_targets = len(meta["targets"])
    print(f"rows: {len(feats):,}   features: {number_of_features}   targets: {number_of_targets}")

    # How much of each group is missing, on average.
    for group, group_columns in meta["feature_groups"].items():
        missing_percent = feats[group_columns].isna().mean().mean() * 100
        print(f"  {group:<10} {len(group_columns):>3} columns   mean missing {missing_percent:5.1f}%")

    print(f"saved {out_dir / 'features.parquet'}")


if __name__ == "__main__":
    main()