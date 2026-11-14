"""Build the model feature table from data/clean/hourly.parquet.

Each row is one forecast issue time t (hourly, UTC). Every feature uses only
information available at t; targets are the Point level at t+6, t+12, t+24.

Feature groups (listed in feature_groups.json):
  point      Point's own level now, 6 h and 24 h ago, recent change, 24 h max
  upstream   Natrona, Kittanning, Sutersville: level now / 6 h / 12 h ago + changes;
             Braddock: changes only (Dam 3 removal shifted its level, Jul 2024)
  upstream_eliz  Elizabeth: changes only (same Dam 3 reason; offline since Jun 2026).
             Ablation only -- not in the main models.
  rain_obs   observed rain totals over the past 6/24/48/72 h, per location + basin mean
  season     day-of-year sin/cos
  flags      short-gap fill flags
  info       post_dam3 -- kept for analysis, NOT a model input: no high water has
             occurred since Jul 2024 at any gauge (incl. Allegheny gauges the dam
             cannot affect), so the flag would teach the model "post-dam = low water"

Targets per horizon h:
  y_{h}h     Point level at t+h (ft)
  dy_{h}h    y_{h}h - point(t)   (what the models are trained to predict)
Plus water_year (of the issue time t), the train/test split key.

Usage (from the repository root):
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

HORIZONS = (6, 12, 24)
FLOOD_FT = 18.0  # NWS action stage, Ohio River at Pittsburgh (flood stage: 22 ft since Oct 2024)
RAIN_LOCS = ("pittsburgh", "kittanning", "elizabeth", "sutersville")
RAIN_WINDOWS = (6, 24, 48, 72)
LEVEL_GAUGES = ("natrona", "kittanning", "sutersville")  # stable datum: use levels
DELTA_GAUGES = ("braddock",)                             # Dam 3 shifted datum: changes only
GROUPS = ("point", "upstream", "upstream_eliz", "rain_obs", "season", "flags", "info")


def load_features(features_dir: Path = FEATURES_DIR) -> tuple[pd.DataFrame, dict]:
    """The feature table and its metadata (flood_ft, horizons, feature_groups, targets)."""
    meta = json.loads((features_dir / "feature_groups.json").read_text())
    return pd.read_parquet(features_dir / "features.parquet"), meta


def load_hourly(path: Path) -> pd.DataFrame:
    """Cleaned hourly data, checked to be a gap-free hourly grid."""
    df = pd.read_parquet(path)
    idx = df.index
    assert idx.is_unique and idx.is_monotonic_increasing, "index must be unique and sorted"
    assert (idx.to_series().diff().dropna() == pd.Timedelta("1h")).all(), \
        "hourly grid has holes: shift(n) would not mean n hours"
    return df


def past_sum(s: pd.Series, hours: int) -> pd.Series:
    """Total over (t-hours, t]. NaN if any hour in the window is missing."""
    return s.rolling(hours, min_periods=hours).sum()


def build(df: pd.DataFrame, flood_ft: float) -> tuple[pd.DataFrame, dict]:
    cols: dict[str, object] = {}
    groups: dict[str, list[str]] = {k: [] for k in GROUPS}

    def add(group: str, name: str, values) -> None:
        cols[name] = values
        groups[group].append(name)

    # --- Point's own history ---------------------------------------------------
    p = df["point"]
    add("point", "point", p)
    add("point", "point_lag6", p.shift(6))
    add("point", "point_lag24", p.shift(24))
    add("point", "point_d1", p - p.shift(1))
    add("point", "point_d6", p - p.shift(6))
    add("point", "point_max24", p.rolling(24, min_periods=18).max())

    # --- upstream gauges -------------------------------------------------------
    for g in LEVEL_GAUGES:
        s = df[g]
        add("upstream", g, s)
        add("upstream", f"{g}_lag6", s.shift(6))
        add("upstream", f"{g}_lag12", s.shift(12))
        add("upstream", f"{g}_d1", s - s.shift(1))
        add("upstream", f"{g}_d6", s - s.shift(6))
    for g in DELTA_GAUGES:
        s = df[g]
        add("upstream", f"{g}_d1", s - s.shift(1))
        add("upstream", f"{g}_d6", s - s.shift(6))
        add("upstream", f"{g}_d12", s - s.shift(12))
    e = df["elizabeth"]
    for k in (1, 6, 12):
        add("upstream_eliz", f"elizabeth_d{k}", e - e.shift(k))

    # --- observed rain (past only) ----------------------------------------------
    for loc in (*RAIN_LOCS, "basin_mean"):
        s = df[f"rain_{loc}"]
        for w in RAIN_WINDOWS:
            add("rain_obs", f"rain_{loc}_past{w}h", past_sum(s, w))

    # --- season and flags ------------------------------------------------------
    doy = df.index.day_of_year.to_numpy()
    add("season", "doy_sin", np.sin(2 * np.pi * doy / 365.25))
    add("season", "doy_cos", np.cos(2 * np.pi * doy / 365.25))
    add("info", "post_dam3", df["post_dam3"].astype(int))
    for g in ("point", "braddock", "natrona", "kittanning", "sutersville"):
        add("flags", f"{g}_filled", df[f"{g}_filled"].astype(int))

    # --- targets ---------------------------------------------------------------
    targets = []
    for h in HORIZONS:
        y = p.shift(-h)
        cols[f"y_{h}h"] = y
        cols[f"dy_{h}h"] = y - p
        targets += [f"y_{h}h", f"dy_{h}h"]

    # Split key: the water year of the issue time t (not of the target time).
    cols["water_year"] = df["water_year"]
    f = pd.DataFrame(cols, index=df.index)

    meta = {
        "flood_ft": flood_ft,
        "horizons": list(HORIZONS),
        "feature_groups": groups,
        "targets": targets,
    }
    return f, meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", type=Path, default=CLEAN_HOURLY)
    ap.add_argument("--out-dir", type=Path, default=FEATURES_DIR)
    ap.add_argument("--flood-ft", type=float, default=FLOOD_FT)
    args = ap.parse_args()

    feats, meta = build(load_hourly(args.clean), args.flood_ft)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(args.out_dir / "features.parquet")
    (args.out_dir / "feature_groups.json").write_text(json.dumps(meta, indent=2))

    n_feat = sum(len(v) for k, v in meta["feature_groups"].items() if k != "info")
    print(f"rows: {len(feats):,}   features: {n_feat}   targets: {len(meta['targets'])}")
    for g, cols in meta["feature_groups"].items():
        miss = feats[cols].isna().mean().mean() * 100
        print(f"  {g:<10} {len(cols):>3} columns   mean missing {miss:5.1f}%")
    print(f"saved {args.out_dir / 'features.parquet'}")


if __name__ == "__main__":
    main()
