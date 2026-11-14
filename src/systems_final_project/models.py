"""Train the forecasting models leave-one-water-year-out and save their predictions.

Models (one HistGradientBoostingRegressor per horizon, identical hyperparameters GBM):
  gbm_A        Point history + upstream gauges + observed rain + season + flags (52 inputs)
  gbm_AR_mono  physics-informed gbm_A:
                 - the fitted routing model's forecast (physics.py) as a 53rd input
                 - monotonic constraints: the forecast may only rise with upstream levels,
                   upstream rises, rain totals and the routing forecast
  persistence  baseline: level at t+h = level now (no training)

Each GBM predicts the change dy = level(t+h) - level(t).
Folds: for each water year, train on all other years and drop training rows whose
target time falls in the held-out year. The routing model is refitted in every fold
on that fold's training hours only.

Physical-consistency check: every model also predicts with all upstream gauges
PERTURB_FT higher at the issue time (level and recent rises). A physically consistent
model never lowers its forecast when that happens.

Usage (from the repository root, after systems_final_project.features):
    uv run python -m systems_final_project.models [--run NAME ...] [--force]
Writes the hourly held-out forecasts (pred_6h, pred_12h, pred_24h, and pert_* for the
perturbed forecasts) to results/predictions/<name>.parquet; a model that already has a
file is skipped unless --force.
"""

import argparse
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from systems_final_project.data.config import CLEAN_HOURLY, PREDICTIONS_DIR as OUT
from systems_final_project.features import load_features, load_hourly
from systems_final_project.physics import ROUTED_GAUGES, RoutingModel, routing_design

INPUTS = ("point", "upstream", "rain_obs", "season", "flags")  # feature groups, in column order
MODELS = ("gbm_A", "gbm_AR_mono")
REFERENCE = "persistence"
UPSTREAM_LEVELS = ("natrona", "kittanning", "sutersville")      # current upstream levels
GBM = dict(learning_rate=0.05, max_iter=400, max_leaf_nodes=31, min_samples_leaf=40,
           l2_regularization=1.0, early_stopping=False, random_state=0)
PERTURB_FT = 0.5


def features_for(meta: dict) -> list[str]:
    """The 52 input columns shared by both models, in order."""
    return [c for g in INPUTS for c in meta["feature_groups"][g]]


def _rises_with_upstream(c: str) -> bool:
    """Current upstream levels and recent upstream rises: more of either = more water coming."""
    upstream_rise = c.split("_")[0] in ROUTED_GAUGES and c.endswith(("_d1", "_d6", "_d12"))
    return c in UPSTREAM_LEVELS or upstream_rise


def monotonic_constraints(feats: list[str]) -> dict[str, int]:
    """+1 (forecast may only increase) for upstream levels/rises, rain totals and the
    routing forecast `phys`; Point's own history, season and flags are unconstrained."""
    return {c: 1 for c in feats if _rises_with_upstream(c) or c.startswith("rain_") or c == "phys"}


def perturb_upstream(X: pd.DataFrame, delta: float) -> pd.DataFrame:
    """Every upstream gauge `delta` ft higher at the issue time than it was."""
    Xp = X.copy()
    for c in X.columns:
        if _rises_with_upstream(c):
            Xp[c] = X[c] + delta
    return Xp


def routing_by_fold(f: pd.DataFrame, horizons) -> dict[int, tuple[RoutingModel, dict]]:
    """{held-out water year: (routing model fitted without it, {h: predicted change})}."""
    levels = load_hourly(CLEAN_HOURLY)[list(ROUTED_GAUGES)].reindex(f.index)  # raw upstream levels
    designs = {h: routing_design(levels, h) for h in horizons}
    targets = {h: f[f"y_{h}h"] for h in horizons}
    wy, target_wy = f["water_year"], f["water_year"].shift(-max(horizons))
    out = {}
    for y in sorted(wy.unique()):
        train = ((wy != y) & (target_wy != y) & f["point"].notna()).to_numpy()
        rm = RoutingModel().fit(designs, f["point"], targets, train)
        out[int(y)] = (rm, rm.predict_change(designs, f["point"]))
    return out


def loyo(f: pd.DataFrame, h: int, fit_predict) -> tuple[pd.Series, pd.Series]:
    """Leave-one-water-year-out (predicted level, perturbed predicted level) at t+h,
    indexed by issue time t."""
    target = f"dy_{h}h"
    pred, pert = pd.Series(np.nan, index=f.index), pd.Series(np.nan, index=f.index)
    issue_wy = f["water_year"]
    target_wy = issue_wy.shift(-h)
    has_now = f["point"].notna()
    for wy in sorted(issue_wy.unique()):
        test = has_now & (issue_wy == wy)
        train = f[target].notna() & has_now & (issue_wy != wy) & (target_wy != wy)
        d, dp = fit_predict(train, test, target, int(wy))
        now = f.loc[test, "point"].to_numpy()
        pred[test], pert[test] = now + d, now + dp
    return pred, pert


def run(f: pd.DataFrame, meta: dict, name: str) -> dict[str, pd.Series]:
    """Held-out forecasts of one model: {"pred_6h": ..., "pert_6h": ..., ...}."""
    H = meta["horizons"]
    if name == REFERENCE:
        return {f"{k}_{h}h": f["point"].copy() for h in H for k in ("pred", "pert")}
    assert name in MODELS, f"unknown model {name}; choose from {MODELS + (REFERENCE,)}"

    feats = features_for(meta)
    physics = name == "gbm_AR_mono"
    routing = routing_by_fold(f, H) if physics else None
    params = dict(GBM)
    if physics:
        params["monotonic_cst"] = monotonic_constraints(feats + ["phys"])
    out = {}
    for h in H:
        def fit_predict(train, test, target, wy):
            X = f[feats].assign(phys=routing[wy][1][h]) if physics else f[feats]
            m = HistGradientBoostingRegressor(**params).fit(X[train], f.loc[train, target])
            Xt = X[test]
            Xp = perturb_upstream(Xt, PERTURB_FT)
            if physics:
                Xp["phys"] = Xt["phys"] + PERTURB_FT * routing[wy][0].step_gain(h)
            return m.predict(Xt), m.predict(Xp)
        out[f"pred_{h}h"], out[f"pert_{h}h"] = loyo(f, h, fit_predict)
    return out


def save(name: str, cols: dict[str, pd.Series]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(cols).to_parquet(OUT / f"{name}.parquet")


def load(name: str, horizons, kind: str = "pred") -> dict[int, pd.Series]:
    """Saved forecasts of a model: {h: level at t+h}; kind="pert" for the perturbed ones."""
    d = pd.read_parquet(OUT / f"{name}.parquet")
    return {h: d[f"{kind}_{h}h"] for h in horizons}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", nargs="*", default=[*MODELS, REFERENCE])
    ap.add_argument("--force", action="store_true", help="retrain even if a file exists")
    args = ap.parse_args()

    f, meta = load_features()
    for name in args.run:
        if (OUT / f"{name}.parquet").exists() and not args.force:
            print(f"{name}: kept existing predictions")
            continue
        t0 = time.time()
        save(name, run(f, meta, name))
        print(f"{name}: done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
