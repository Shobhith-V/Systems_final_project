"""Choose model settings on the validation year (WY2018) -- never on a test year.

Fits on WY2016-2017 (rows whose target time is also before WY2018) and scores on the
WY2018 hours whose issue time AND target time are both in WY2018 (so no score depends
on the first test year). WY2018 holds 6 of the 13 major floods, so high-water skill can
actually be measured there.

Chooses:
  - GBM hyperparameters: one set shared by every GBM and horizon (searched with gbm_A's
    inputs), so the GBM variants stay comparable
  - persistence_trend: trend window and damping, separately for each horizon

Selection metric: mean absolute error on high-water hours (observed >= the flood level
in feature_groups.json), ties broken by RMSE over all hours.

Usage (from the repository root, after systems_final_project.features):
    uv run python -m systems_final_project.tune
Writes results/tuning.json (read by models.py) and results/tuning_gbm.csv,
results/tuning_trend.csv (every setting tried, for the report).
"""

import itertools
import json
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from systems_final_project import models
from systems_final_project.data.config import RESULTS_DIR
from systems_final_project.features import load_features

GBM_GRID = {"max_iter": [200, 400], "max_leaf_nodes": [15, 31, 63], "min_samples_leaf": [20, 40, 80]}
TREND_GRID = {"window_h": [3, 6, 12, 24], "damp": [1.0, 0.5, 0.25]}
VAL = models.VALIDATION_YEAR


def split_rows(f: pd.DataFrame, h: int) -> tuple[pd.Series, pd.Series]:
    """(train, validation) masks for horizon h."""
    issue_wy = f["water_year"]
    target_wy = issue_wy.shift(-h)
    usable = f["point"].notna() & f[f"dy_{h}h"].notna()
    train = usable & models.train_mask(issue_wy, target_wy, VAL, "walk_forward")
    val = usable & (issue_wy == VAL) & (target_wy == VAL)
    return train, val


def score(y: pd.Series, pred: pd.Series, thr: float) -> tuple[float, float]:
    """(MAE on high-water hours, RMSE on all hours)."""
    err = pred - y
    high = y >= thr
    return float(err[high].abs().mean()), float(np.sqrt((err ** 2).mean()))


def best_row(table: pd.DataFrame, mae_col: str, rmse_col: str) -> pd.Series:
    return table.sort_values([mae_col, rmse_col], na_position="last").iloc[0]


def tune_gbm(f, meta, thr) -> tuple[dict, pd.DataFrame]:
    feats = models.features_for(meta, "gbm_A")
    H = meta["horizons"]
    masks = {h: split_rows(f, h) for h in H}
    rows = []
    for combo in itertools.product(*GBM_GRID.values()):
        setting = dict(zip(GBM_GRID, combo))
        params = {**models.DEFAULT_GBM, **setting}
        row = dict(setting)
        for h in H:
            train, val = masks[h]
            m = HistGradientBoostingRegressor(**params).fit(f.loc[train, feats], f.loc[train, f"dy_{h}h"])
            pred = f.loc[val, "point"] + m.predict(f.loc[val, feats])
            row[f"mae_high_{h}h"], row[f"rmse_{h}h"] = score(f.loc[val, f"y_{h}h"], pred, thr)
        row["mae_high_mean"] = np.mean([row[f"mae_high_{h}h"] for h in H])
        row["rmse_mean"] = np.mean([row[f"rmse_{h}h"] for h in H])
        rows.append(row)
        print(f"  gbm {setting}: high-water MAE {row['mae_high_mean']:.3f}, RMSE {row['rmse_mean']:.3f}", flush=True)
    table = pd.DataFrame(rows).sort_values(["mae_high_mean", "rmse_mean"]).reset_index(drop=True)
    best = best_row(table, "mae_high_mean", "rmse_mean")
    return {k: int(best[k]) for k in GBM_GRID}, table


def tune_trend(f, meta, thr) -> tuple[dict, pd.DataFrame]:
    rows, chosen = [], {}
    for h in meta["horizons"]:
        _, val = split_rows(f, h)
        for w, d in itertools.product(*TREND_GRID.values()):
            pred = models.trend_forecast(f["point"], h, w, d)[val]
            mae_high, rmse = score(f.loc[val, f"y_{h}h"], pred, thr)
            rows.append({"horizon_h": h, "window_h": w, "damp": d, "mae_high": mae_high, "rmse": rmse})
        best = best_row(pd.DataFrame([r for r in rows if r["horizon_h"] == h]), "mae_high", "rmse")
        chosen[str(h)] = {"window_h": int(best["window_h"]), "damp": float(best["damp"])}
    return chosen, pd.DataFrame(rows)


def main() -> None:
    f, meta = load_features()
    thr = meta["flood_ft"]
    for h in meta["horizons"]:
        train, val = split_rows(f, h)
        n_high = int((f.loc[val, f"y_{h}h"] >= thr).sum())
        print(f"{h:>2} h: train {int(train.sum()):,} rows (WY<={VAL - 1}), validate {int(val.sum()):,} rows "
              f"(WY{VAL}, {n_high} at >= {thr:.0f} ft)")
        assert n_high > 0, f"no high-water hours in WY{VAL}: cannot tune on high-water error"

    t0 = time.time()
    gbm, gbm_table = tune_gbm(f, meta, thr)
    trend, trend_table = tune_trend(f, meta, thr)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    gbm_table.to_csv(RESULTS_DIR / "tuning_gbm.csv", index=False)
    trend_table.to_csv(RESULTS_DIR / "tuning_trend.csv", index=False)
    models.TUNING_FILE.write_text(json.dumps({
        "validation_year": VAL,
        "metric": f"MAE on hours observed >= {thr} ft (ties: RMSE)",
        "gbm": gbm,
        "trend": trend,
    }, indent=2))

    default = {k: models.DEFAULT_GBM[k] for k in GBM_GRID}
    d = gbm_table.merge(pd.DataFrame([default]), on=list(GBM_GRID))
    print(f"\nchosen GBM settings: {gbm}  (high-water MAE {gbm_table.mae_high_mean.iloc[0]:.3f} ft"
          + (f"; old defaults {default}: {d.mae_high_mean.iloc[0]:.3f} ft)" if len(d) else ")"))
    print("chosen trend rule per horizon:", trend)
    print(f"saved {models.TUNING_FILE}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
