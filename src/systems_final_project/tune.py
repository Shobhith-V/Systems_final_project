"""
Choose the model settings on the validation year (WY2018), never on a test year.

How it works:
  - Models are fitted on WY2016-2017. Only rows whose target time is also
    before WY2018 are used.
  - They are scored on the WY2018 hours where both the issue time AND the
    target time are in WY2018, so no score depends on the first test year.
  - WY2018 contains 6 of the 13 major floods, so high-water skill can
    actually be measured there.

What it chooses:
  - GBM hyperparameters: one set, shared by every GBM and every horizon
    (searched using gbm_A's inputs), so the different GBM variants stay
    comparable with each other.
  - persistence_trend: the trend window and damping, chosen separately for
    each horizon.

How settings are compared: mean absolute error on high-water hours (observed
level >= the flood level in feature_groups.json). If two settings tie, the one
with the lower RMSE over all hours wins.

Run from the repository root, after systems_final_project.features:
    uv run python -m systems_final_project.tune

Writes:
  - results/tuning.json (read by models.py)
  - results/tuning_gbm.csv and results/tuning_trend.csv (every setting that
    was tried, for the report)
"""

import json
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from systems_final_project import models
from systems_final_project.data.config import RESULTS_DIR
from systems_final_project.features import load_features

# GBM settings to try (every combination is tested).
GBM_GRID = {
    "max_iter": [200, 400],
    "max_leaf_nodes": [15, 31, 63],
    "min_samples_leaf": [20, 40, 80],
}

# Trend baseline settings to try (every combination, for each horizon).
TREND_GRID = {
    "window_h": [3, 6, 12, 24],
    "damp": [1.0, 0.5, 0.25],
}

# The validation year.
VAL = models.VALIDATION_YEAR


def split_rows(f, h):
    """Return two True/False masks for horizon h: training rows and validation rows."""
    issue_wy = f["water_year"]
    target_wy = issue_wy.shift(-h)

    # A row is usable if we know the level now and the change h hours later.
    has_level_now = f["point"].notna()
    has_target = f[f"dy_{h}h"].notna()
    usable = has_level_now & has_target

    # Training: years before the validation year (same rule as walk-forward).
    before_validation = models.train_mask(issue_wy, target_wy, VAL, "walk_forward")
    train = usable & before_validation

    # Validation: both the issue time and the target time in the validation year.
    issued_in_val = issue_wy == VAL
    target_in_val = target_wy == VAL
    val = usable & issued_in_val & target_in_val

    return train, val


def score(y, pred, thr):
    """Return (MAE on high-water hours, RMSE on all hours)."""
    error = pred - y
    is_high_water = y >= thr

    mae_high_water = float(error[is_high_water].abs().mean())
    rmse = float(np.sqrt((error ** 2).mean()))
    return mae_high_water, rmse


def best_row(table, mae_col, rmse_col):
    """Return the row with the lowest MAE (ties broken by the lowest RMSE)."""
    sorted_table = table.sort_values([mae_col, rmse_col], na_position="last")
    return sorted_table.iloc[0]


def tune_gbm(f, meta, thr):
    """Try every GBM setting in GBM_GRID and pick the best one.

    Returns the chosen setting and a table with the scores of every setting.
    """
    feats = models.features_for(meta, "gbm_A")
    horizons = meta["horizons"]

    # Work out the training and validation rows once for each horizon.
    masks = {}
    for h in horizons:
        masks[h] = split_rows(f, h)

    rows = []

    # Try every combination of the three settings.
    for max_iter in GBM_GRID["max_iter"]:
        for max_leaf_nodes in GBM_GRID["max_leaf_nodes"]:
            for min_samples_leaf in GBM_GRID["min_samples_leaf"]:
                setting = {
                    "max_iter": max_iter,
                    "max_leaf_nodes": max_leaf_nodes,
                    "min_samples_leaf": min_samples_leaf,
                }

                # Start from the default settings and overwrite the ones we're testing.
                params = models.DEFAULT_GBM.copy()
                for key, value in setting.items():
                    params[key] = value

                # This row of the results table starts with the setting itself.
                row = setting.copy()

                high_water_maes = []
                rmses = []
                for h in horizons:
                    train, val = masks[h]

                    # Train on the training rows. The model predicts the change in level.
                    model = HistGradientBoostingRegressor(**params)
                    model.fit(f.loc[train, feats], f.loc[train, f"dy_{h}h"])

                    # Predicted level = level now + predicted change.
                    predicted_change = model.predict(f.loc[val, feats])
                    pred = f.loc[val, "point"] + predicted_change

                    mae_high, rmse = score(f.loc[val, f"y_{h}h"], pred, thr)
                    row[f"mae_high_{h}h"] = mae_high
                    row[f"rmse_{h}h"] = rmse
                    high_water_maes.append(mae_high)
                    rmses.append(rmse)

                # Average over the horizons.
                row["mae_high_mean"] = np.mean(high_water_maes)
                row["rmse_mean"] = np.mean(rmses)
                rows.append(row)

                print(f"  gbm {setting}: high-water MAE {row['mae_high_mean']:.3f}, "
                      f"RMSE {row['rmse_mean']:.3f}", flush=True)

    # Sort the results table so the best setting is first.
    table = pd.DataFrame(rows)
    table = table.sort_values(["mae_high_mean", "rmse_mean"])
    table = table.reset_index(drop=True)

    best = best_row(table, "mae_high_mean", "rmse_mean")

    chosen = {}
    for key in GBM_GRID:
        chosen[key] = int(best[key])

    return chosen, table


def tune_trend(f, meta, thr):
    """Try every trend setting in TREND_GRID, separately for each horizon.

    Returns the chosen settings {horizon (as text): setting} and a table with
    the scores of every setting tried.
    """
    all_rows = []
    chosen = {}

    for h in meta["horizons"]:
        train, val = split_rows(f, h)  # only the validation rows are needed here

        # Results for this horizon only.
        horizon_rows = []

        for window_h in TREND_GRID["window_h"]:
            for damp in TREND_GRID["damp"]:
                forecast = models.trend_forecast(f["point"], h, window_h, damp)
                pred = forecast[val]

                mae_high, rmse = score(f.loc[val, f"y_{h}h"], pred, thr)

                row = {
                    "horizon_h": h,
                    "window_h": window_h,
                    "damp": damp,
                    "mae_high": mae_high,
                    "rmse": rmse,
                }
                all_rows.append(row)
                horizon_rows.append(row)

        # Pick the best setting for this horizon.
        best = best_row(pd.DataFrame(horizon_rows), "mae_high", "rmse")
        chosen[str(h)] = {
            "window_h": int(best["window_h"]),
            "damp": float(best["damp"]),
        }

    return chosen, pd.DataFrame(all_rows)


def main():
    f, meta = load_features()
    thr = meta["flood_ft"]

    # Show how many rows we train and validate on, and make sure the
    # validation year actually has some high water.
    for h in meta["horizons"]:
        train, val = split_rows(f, h)
        n_train = int(train.sum())
        n_val = int(val.sum())
        n_high = int((f.loc[val, f"y_{h}h"] >= thr).sum())

        print(f"{h:>2} h: train {n_train:,} rows (WY<={VAL - 1}), validate {n_val:,} rows "
              f"(WY{VAL}, {n_high} at >= {thr:.0f} ft)")
        assert n_high > 0, f"no high-water hours in WY{VAL}: cannot tune on high-water error"

    start_time = time.time()
    gbm, gbm_table = tune_gbm(f, meta, thr)
    trend, trend_table = tune_trend(f, meta, thr)

    # Save every setting tried, plus the chosen settings for models.py.
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    gbm_table.to_csv(RESULTS_DIR / "tuning_gbm.csv", index=False)
    trend_table.to_csv(RESULTS_DIR / "tuning_trend.csv", index=False)

    tuning = {
        "validation_year": VAL,
        "metric": f"MAE on hours observed >= {thr} ft (ties: RMSE)",
        "gbm": gbm,
        "trend": trend,
    }
    models.TUNING_FILE.write_text(json.dumps(tuning, indent=2))

    # Compare the chosen GBM settings with the old defaults (if the defaults
    # were one of the settings tried).
    default = {}
    for key in GBM_GRID:
        default[key] = models.DEFAULT_GBM[key]
    default_rows = gbm_table.merge(pd.DataFrame([default]), on=list(GBM_GRID))

    best_mae = gbm_table["mae_high_mean"].iloc[0]
    message = f"\nchosen GBM settings: {gbm}  (high-water MAE {best_mae:.3f} ft"
    if len(default_rows) > 0:
        default_mae = default_rows["mae_high_mean"].iloc[0]
        message = message + f"; old defaults {default}: {default_mae:.3f} ft)"
    else:
        message = message + ")"
    print(message)

    print("chosen trend rule per horizon:", trend)

    seconds = time.time() - start_time
    print(f"saved {models.TUNING_FILE}  ({seconds:.0f}s)")


if __name__ == "__main__":
    main()