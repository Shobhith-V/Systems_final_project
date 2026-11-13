"""Train the forecasting models leave-one-water-year-out and save their predictions.

Models (HistGradientBoostingRegressor, identical hyperparameters GBM):
  gbm_A        Point history + upstream gauges + observed rain + season + flags
               (deployable today: every input is still reported)
  gbm_AE       gbm_A + Elizabeth gauge (most accurate, but Elizabeth is offline
               since 2026-06, so it cannot be relied on for real-time use)
  persistence  baseline: level at t+h = level now (no training)

These two were kept from 15 configurations tried; the others were deleted.
They ranked best on mean RMSE at 6 h and 12 h over held-out water years
2016-2023, and 2024-2027 confirmed the choice.

Each GBM predicts the change dy = level(t+h) - level(t), one model per horizon.
Folds: for each water year, train on all other years and drop training rows
whose target time falls in the held-out year.

Usage (from the repository root, after systems_final_project.features):
    uv run python -m systems_final_project.models [--run NAME ...] [--force]
(default --run: gbm_A gbm_AE persistence). Writes the hourly held-out predictions
(columns pred_6h, pred_12h, pred_24h) to results/predictions/<name>.parquet;
a model that already has a file is skipped unless --force.
"""

import argparse
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from systems_final_project.data.config import PREDICTIONS_DIR as OUT
from systems_final_project.features import load_features

_A = ("point", "upstream", "rain_obs", "season", "flags")
INPUTS = {"A": _A, "AE": _A + ("elizabeth",)}  # feature groups, in column order
MODELS = ("gbm_A", "gbm_AE")
REFERENCE = "persistence"
GBM = dict(learning_rate=0.05, max_iter=400, max_leaf_nodes=31, min_samples_leaf=40,
           l2_regularization=1.0, early_stopping=False, random_state=0)


def features_for(name: str, meta: dict) -> list[str]:
    """Feature columns of a model ("gbm_A" -> groups INPUTS["A"], in order)."""
    return [c for g in INPUTS[name.split("_")[1]] for c in meta["feature_groups"][g]]


def loyo(f: pd.DataFrame, h: int, fit_predict) -> pd.Series:
    """Leave-one-water-year-out predicted level at t+h, indexed by issue time t."""
    target = f"dy_{h}h"
    pred = pd.Series(np.nan, index=f.index)
    issue_wy = f["water_year"]
    target_wy = issue_wy.shift(-h)
    has_now = f["point"].notna()
    for wy in sorted(issue_wy.unique()):
        test = has_now & (issue_wy == wy)
        train = f[target].notna() & has_now & (issue_wy != wy) & (target_wy != wy)
        pred[test] = f.loc[test, "point"].to_numpy() + fit_predict(train, test, target)
    return pred


def run(f: pd.DataFrame, meta: dict, name: str) -> dict[int, pd.Series]:
    if name == REFERENCE:
        return {h: f["point"].copy() for h in meta["horizons"]}
    assert name in MODELS, f"unknown model {name}; choose from {MODELS + (REFERENCE,)}"
    feats = features_for(name, meta)

    def fit_predict(train, test, target):
        m = HistGradientBoostingRegressor(**GBM).fit(f.loc[train, feats], f.loc[train, target])
        return m.predict(f.loc[test, feats])

    return {h: loyo(f, h, fit_predict) for h in meta["horizons"]}


def save(name: str, preds: dict[int, pd.Series]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({f"pred_{h}h": p for h, p in preds.items()}).to_parquet(OUT / f"{name}.parquet")


def load(name: str, horizons) -> dict[int, pd.Series]:
    """Cached predictions of a model: {h: predicted level at t+h}."""
    d = pd.read_parquet(OUT / f"{name}.parquet")
    return {h: d[f"pred_{h}h"] for h in horizons}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", nargs="*", default=[*MODELS, REFERENCE])
    ap.add_argument("--force", action="store_true", help="retrain even if cached")
    args = ap.parse_args()

    f, meta = load_features()
    for name in args.run:
        if (OUT / f"{name}.parquet").exists() and not args.force:
            print(f"{name}: cached")
            continue
        t0 = time.time()
        save(name, run(f, meta, name))
        print(f"{name}: done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
