"""Train the forecasting models on time-ordered folds and save their held-out predictions.

Main models (scored in fig 1):
  persistence        baseline: level at t+h = level now (no training)
  persistence_trend  rule baseline: level now + damp * (recent rate of change) * h, with the
                     window and damping chosen per horizon on the validation year (tune.py)
  gbm_A              GBM on Point history + upstream gauges + observed rain + season + gap flags
  gbm_AR_mono        physics-informed gbm_A:
                       - the fitted routing model's forecast (physics.py) as one more input
                       - monotonic constraints: the forecast may only rise with upstream levels,
                         upstream rises, rain totals and the routing forecast
Ablations (scored separately, fig 3):
  gbm_A_eliz         gbm_A + Elizabeth's level changes

GBMs: one HistGradientBoostingRegressor per horizon, all with the same hyperparameters
(chosen on the validation year by tune.py; defaults in DEFAULT_GBM if tune has not run).
Each GBM predicts the change dy = level(t+h) - level(t).

Folds (SPLIT):
  walk_forward (default)  expanding window by water year. WY2016-2017 train only,
                          WY2018 is the validation year (for tuning; never tested),
                          WY2019-2026 are tested one at a time, each trained on all
                          EARLIER water years. Training rows whose target time falls in
                          the test year are dropped (the horizon-sized gap).
  loyo                    leave-one-water-year-out: test each year, train on all others
                          (also drops training rows whose target falls in the test year).
The routing model is refitted in every fold on that fold's training hours only.
Hours outside the test years get no prediction (NaN).

Physical-consistency check: every model also predicts with all upstream gauges
PERTURB_FT higher at the issue time (level and recent rises). A physically consistent
model never lowers its forecast when that happens.

Usage (from the repository root, after systems_final_project.features and .tune):
    uv run python -m systems_final_project.models [--run NAME ...] [--split loyo] [--force]
Writes the hourly held-out forecasts (pred_6h, pred_12h, pred_24h, and pert_* for the
perturbed forecasts) to results/predictions/<name>.parquet, plus <name>.json recording
the split and test years. An existing model is kept only if it is newer than the
feature table and the tuning file and was made with the same split; otherwise (or with
--force) it is retrained.
"""

import argparse
import json
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from systems_final_project.data.config import CLEAN_HOURLY, FEATURES_DIR, RESULTS_DIR, PREDICTIONS_DIR as OUT
from systems_final_project.features import load_features, load_hourly
from systems_final_project.physics import ROUTED_GAUGES, RoutingModel, routing_design

# ------------------------------------------------------------------------------ models
INPUTS = ("point", "upstream", "rain_obs", "season", "flags")   # feature groups, in column order
MODEL_INPUTS = {                                                 # GBM -> its feature groups
    "gbm_A": INPUTS,
    "gbm_AR_mono": INPUTS,                       # + routing forecast "phys", monotonic constraints
    "gbm_A_eliz": INPUTS + ("upstream_eliz",),   # ablation: does Elizabeth add anything?
}
REFERENCE = "persistence"
TREND = "persistence_trend"
MAIN = (REFERENCE, TREND, "gbm_A", "gbm_AR_mono")
ABLATIONS = ("gbm_A_eliz",)
ALL_MODELS = MAIN + ABLATIONS

UPSTREAM_LEVELS = ("natrona", "kittanning", "sutersville")      # current upstream levels
UPSTREAM_RISE_GAUGES = (*ROUTED_GAUGES, "elizabeth")            # gauges whose rises mean more water
PERTURB_FT = 0.5

DEFAULT_GBM = dict(learning_rate=0.05, max_iter=400, max_leaf_nodes=31, min_samples_leaf=40,
                   l2_regularization=1.0, early_stopping=False, random_state=0)
DEFAULT_TREND = {"window_h": 6, "damp": 1.0}
TUNING_FILE = RESULTS_DIR / "tuning.json"

# ------------------------------------------------------------------------------ split
SPLIT = "walk_forward"           # or "loyo"
VALIDATION_YEAR = 2018           # walk_forward: reserved for tuning, never a test year
TEST_YEARS = range(2019, 2027)   # walk_forward: WY2027 is a partial year


def fold_years(f: pd.DataFrame, split: str = SPLIT) -> list[int]:
    """Water years that get held-out predictions."""
    years = sorted(int(y) for y in f["water_year"].dropna().unique())
    if split == "loyo":
        return years
    return [y for y in years if y in TEST_YEARS]


def train_mask(issue_wy: pd.Series, target_wy: pd.Series, test_wy: int, split: str = SPLIT) -> pd.Series:
    """Training rows for one fold. Rows whose target falls in the test year are always out."""
    if split == "loyo":
        return (issue_wy != test_wy) & (target_wy != test_wy)
    return (issue_wy < test_wy) & (target_wy < test_wy)


# ------------------------------------------------------------------------------ settings
def load_tuning() -> dict:
    """Settings chosen by tune.py, or the defaults if it has not run."""
    if TUNING_FILE.exists():
        return json.loads(TUNING_FILE.read_text())
    return {"gbm": {}, "trend": {}}


def gbm_params(tuning: dict | None = None) -> dict:
    return {**DEFAULT_GBM, **(tuning or load_tuning())["gbm"]}


def trend_settings(h: int, tuning: dict | None = None) -> dict:
    return (tuning or load_tuning())["trend"].get(str(h), DEFAULT_TREND)


def trend_forecast(point: pd.Series, h: int, window_h: int, damp: float) -> pd.Series:
    """Level now + damp * (change over the last window_h hours / window_h) * h.
    Falls back to persistence when the reading window_h hours ago is missing."""
    rate = (point - point.shift(window_h)) / window_h
    return point + damp * rate.fillna(0.0) * h


def features_for(meta: dict, name: str = "gbm_A") -> list[str]:
    """Input columns of one GBM, in order."""
    return [c for g in MODEL_INPUTS[name] for c in meta["feature_groups"][g]]


# ------------------------------------------------------------------------------ physics
def _rises_with_upstream(c: str) -> bool:
    """Current upstream levels and recent upstream rises: more of either = more water coming."""
    upstream_rise = c.split("_")[0] in UPSTREAM_RISE_GAUGES and c.endswith(("_d1", "_d6", "_d12"))
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


def routing_designs(f: pd.DataFrame, horizons) -> dict[int, np.ndarray]:
    """Routing-model inputs built from the raw upstream levels (only readings at or before t)."""
    levels = load_hourly(CLEAN_HOURLY)[list(ROUTED_GAUGES)].reindex(f.index)
    return {h: routing_design(levels, h) for h in horizons}


def routing_fit(f: pd.DataFrame, designs: dict, train: np.ndarray) -> tuple[RoutingModel, dict]:
    """Routing model fitted on `train` hours, and its predicted change {h: Series} for all hours."""
    targets = {h: f[f"y_{h}h"] for h in designs}
    rm = RoutingModel().fit(designs, f["point"], targets, train)
    return rm, rm.predict_change(designs, f["point"])


def routing_by_fold(f: pd.DataFrame, horizons, split: str = SPLIT) -> dict[int, tuple[RoutingModel, dict]]:
    """{test water year: (routing model fitted on that fold's training hours, {h: predicted change})}."""
    designs = routing_designs(f, horizons)
    wy, target_wy = f["water_year"], f["water_year"].shift(-max(horizons))
    return {y: routing_fit(f, designs, (train_mask(wy, target_wy, y, split) & f["point"].notna()).to_numpy())
            for y in fold_years(f, split)}


# ------------------------------------------------------------------------------ training
def held_out(f: pd.DataFrame, h: int, fit_predict, split: str = SPLIT) -> tuple[pd.Series, pd.Series]:
    """Held-out (predicted level, perturbed predicted level) at t+h, indexed by issue
    time t. NaN outside the test years."""
    target = f"dy_{h}h"
    pred, pert = pd.Series(np.nan, index=f.index), pd.Series(np.nan, index=f.index)
    issue_wy = f["water_year"]
    target_wy = issue_wy.shift(-h)
    has_now = f["point"].notna()
    for wy in fold_years(f, split):
        test = has_now & (issue_wy == wy)
        train = f[target].notna() & has_now & train_mask(issue_wy, target_wy, wy, split)
        d, dp = fit_predict(train, test, target, wy)
        now = f.loc[test, "point"].to_numpy()
        pred[test], pert[test] = now + d, now + dp
    return pred, pert


def run(f: pd.DataFrame, meta: dict, name: str, split: str = SPLIT) -> dict[str, pd.Series]:
    """Held-out forecasts of one model: {"pred_6h": ..., "pert_6h": ..., ...}."""
    H = meta["horizons"]
    in_test = f["water_year"].isin(fold_years(f, split))
    tuning = load_tuning()
    if name == REFERENCE:
        return {f"{k}_{h}h": f["point"].where(in_test) for h in H for k in ("pred", "pert")}
    if name == TREND:   # ignores upstream, so the perturbed forecast is unchanged
        out = {}
        for h in H:
            s = trend_settings(h, tuning)
            out[f"pred_{h}h"] = out[f"pert_{h}h"] = trend_forecast(f["point"], h, **s).where(in_test)
        return out
    assert name in MODEL_INPUTS, f"unknown model {name}; choose from {ALL_MODELS}"

    feats = features_for(meta, name)
    physics = name == "gbm_AR_mono"
    routing = routing_by_fold(f, H, split) if physics else None
    params = gbm_params(tuning)
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
        out[f"pred_{h}h"], out[f"pert_{h}h"] = held_out(f, h, fit_predict, split)
    return out


# ------------------------------------------------------------------------------ files
def save(name: str, cols: dict[str, pd.Series], split: str, test_years: list[int]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(cols).to_parquet(OUT / f"{name}.parquet")
    (OUT / f"{name}.json").write_text(json.dumps({"split": split, "test_years": test_years}))


def load(name: str, horizons, kind: str = "pred") -> dict[int, pd.Series]:
    """Saved forecasts of a model: {h: level at t+h}; kind="pert" for the perturbed ones."""
    d = pd.read_parquet(OUT / f"{name}.parquet")
    return {h: d[f"{kind}_{h}h"] for h in horizons}


def split_info(name: str) -> dict:
    """{"split": ..., "test_years": [...]} that a saved model was made with."""
    path = OUT / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: rerun `python -m systems_final_project.models --force`")
    return json.loads(path.read_text())


def is_current(name: str, split: str) -> bool:
    """Saved predictions exist, are newer than the features and the tuning file, and used this split."""
    pred, info = OUT / f"{name}.parquet", OUT / f"{name}.json"
    if not (pred.exists() and info.exists()):
        return False
    inputs = [FEATURES_DIR / "features.parquet"] + ([TUNING_FILE] if TUNING_FILE.exists() else [])
    newer = all(pred.stat().st_mtime >= p.stat().st_mtime for p in inputs)
    return newer and json.loads(info.read_text())["split"] == split


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", nargs="*", default=list(ALL_MODELS), choices=list(ALL_MODELS))
    ap.add_argument("--split", choices=["walk_forward", "loyo"], default=SPLIT)
    ap.add_argument("--force", action="store_true", help="retrain even if predictions are current")
    args = ap.parse_args()

    f, meta = load_features()
    years = fold_years(f, args.split)
    print(f"split: {args.split}, test water years {years[0]}-{years[-1]}")
    print("GBM settings:", {k: v for k, v in gbm_params().items() if k in ("max_iter", "max_leaf_nodes",
                                                                           "min_samples_leaf", "learning_rate")},
          "(tuned)" if TUNING_FILE.exists() else "(defaults: tune has not run)")
    for name in args.run:
        if is_current(name, args.split) and not args.force:
            print(f"{name}: kept existing predictions (current)")
            continue
        t0 = time.time()
        save(name, run(f, meta, name, args.split), args.split, years)
        print(f"{name}: done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
