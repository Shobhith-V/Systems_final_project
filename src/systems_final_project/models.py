"""
Train the forecasting models on time-ordered folds and save their held-out
predictions.

Main models (scored in fig 1):

  persistence        Baseline: the level at t+h is the level now. No training.

  persistence_trend  Rule-based baseline: level now + damp * (recent rate of
                     change) * h. The window and the damping are chosen for
                     each horizon on the validation year (see tune.py).

  linear             Linear regression on the same inputs as gbm_A. Missing
                     inputs are filled with the median of that fold's training
                     rows, and each input gets an extra "was missing" column.
                     This way it predicts every hour and is scored on the same
                     hours as the other models.

  gbm_A              GBM on Point history + upstream gauges + observed rain +
                     season + gap flags.

  gbm_AR_mono        Physics-informed version of gbm_A:
                       - the fitted routing model's forecast (physics.py) is
                         added as one more input
                       - monotonic constraints: the forecast is only allowed to
                         go up when upstream levels, upstream rises, rain totals
                         or the routing forecast go up

Ablations (scored separately, fig 3):

  gbm_A_eliz         gbm_A + Elizabeth's level changes

GBMs: there is one HistGradientBoostingRegressor per horizon, and they all use
the same hyperparameters. These are chosen on the validation year by tune.py;
if tune.py has not been run, DEFAULT_GBM is used instead. Each GBM predicts the
change dy = level(t+h) - level(t), not the level itself.

Folds (SPLIT):

  walk_forward (default)
      Expanding window by water year.
        - WY2016-2017 are only ever used for training.
        - WY2018 is the validation year. It is used for tuning and is never
          tested.
        - WY2019-2026 are tested one at a time. Each one is trained on all
          EARLIER water years.
      Training rows whose target time falls inside the test year are dropped
      (this leaves a horizon-sized gap before the test year).

  loyo
      Leave-one-water-year-out: test each year and train on all the others.
      This also drops training rows whose target falls in the test year.

The routing model is refitted in every fold, using only that fold's training
hours. Hours outside the test years get no prediction (NaN).

Physical-consistency check: every model also makes a second forecast where all
upstream gauges are PERTURB_FT higher at the issue time (both their levels and
their recent rises). A physically consistent model should never lower its
forecast when this happens.

Usage (from the repository root, after running systems_final_project.features
and systems_final_project.tune):

    uv run python -m systems_final_project.models [--run NAME ...] [--split loyo] [--force]

For each model this writes:
  - results/predictions/<name>.parquet with the hourly held-out forecasts
    (pred_6h, pred_12h, pred_24h, and pert_* for the perturbed forecasts)
  - results/predictions/<name>.json recording the split and the test years

An existing model's predictions are kept only if they are newer than the
feature table and the tuning file, and were made with the same split.
Otherwise (or if --force is given) the model is retrained.
"""

import argparse
import json
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from systems_final_project.data.config import CLEAN_HOURLY, FEATURES_DIR, RESULTS_DIR, PREDICTIONS_DIR
from systems_final_project.features import load_features, load_hourly
from systems_final_project.physics import ROUTED_GAUGES, RoutingModel, routing_design

# Folder where the predictions are saved.
OUT = PREDICTIONS_DIR


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

# Feature groups, in the order their columns are given to the models.
INPUTS = ("point", "upstream", "rain_obs", "season", "flags")

# Which feature groups each trained model uses.
MODEL_INPUTS = {
    "linear": INPUTS,
    "gbm_A": INPUTS,
    # gbm_AR_mono also gets the routing forecast "phys" and monotonic constraints.
    "gbm_AR_mono": INPUTS,
    # Ablation: does adding Elizabeth help at all?
    "gbm_A_eliz": INPUTS + ("upstream_eliz",),
}

REFERENCE = "persistence"
TREND = "persistence_trend"
MAIN = (REFERENCE, TREND, "linear", "gbm_A", "gbm_AR_mono")
ABLATIONS = ("gbm_A_eliz",)
ALL_MODELS = MAIN + ABLATIONS

# Current upstream levels.
UPSTREAM_LEVELS = ("natrona", "kittanning", "sutersville")

# Gauges where a rise means more water is on its way to the Point.
UPSTREAM_RISE_GAUGES = tuple(ROUTED_GAUGES) + ("elizabeth",)

# How much (in feet) we raise the upstream gauges for the consistency check.
PERTURB_FT = 0.5

# GBM settings to use if tune.py has not been run.
DEFAULT_GBM = {
    "learning_rate": 0.05,
    "max_iter": 400,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 40,
    "l2_regularization": 1.0,
    "early_stopping": False,
    "random_state": 0,
}

# Trend baseline settings to use if tune.py has not been run.
DEFAULT_TREND = {"window_h": 6, "damp": 1.0}

TUNING_FILE = RESULTS_DIR / "tuning.json"


# ---------------------------------------------------------------------------
# Splitting into folds
# ---------------------------------------------------------------------------

SPLIT = "walk_forward"           # the other option is "loyo"
VALIDATION_YEAR = 2018           # walk_forward: used for tuning, never a test year
TEST_YEARS = range(2019, 2027)   # walk_forward: WY2027 is only a partial year


def fold_years(f, split=SPLIT):
    """Return the water years that get held-out predictions."""
    all_years = []
    for year in f["water_year"].dropna().unique():
        all_years.append(int(year))
    all_years.sort()

    # Leave-one-year-out tests every year.
    if split == "loyo":
        return all_years

    # Walk-forward only tests the years in TEST_YEARS.
    test_years = []
    for year in all_years:
        if year in TEST_YEARS:
            test_years.append(year)
    return test_years


def train_mask(issue_wy, target_wy, test_wy, split=SPLIT):
    """Return True/False for each row: True if the row can be used for training.

    A row is never used for training if its target time falls in the test year.
    """
    if split == "loyo":
        # Use every other year.
        issue_not_in_test = issue_wy != test_wy
        target_not_in_test = target_wy != test_wy
        return issue_not_in_test & target_not_in_test

    # Walk-forward: only use years before the test year.
    issue_before_test = issue_wy < test_wy
    target_before_test = target_wy < test_wy
    return issue_before_test & target_before_test


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

def load_tuning():
    """Return the settings chosen by tune.py, or empty settings if it hasn't run."""
    if TUNING_FILE.exists():
        text = TUNING_FILE.read_text()
        return json.loads(text)
    return {"gbm": {}, "trend": {}}


def gbm_params(tuning=None):
    """Return the GBM settings: the defaults, overwritten by any tuned values."""
    if not tuning:
        tuning = load_tuning()

    params = DEFAULT_GBM.copy()
    for key, value in tuning["gbm"].items():
        params[key] = value
    return params


def trend_settings(h, tuning=None):
    """Return the trend baseline settings for horizon h (tuned if available)."""
    if not tuning:
        tuning = load_tuning()

    trend_tuning = tuning["trend"]
    key = str(h)
    if key in trend_tuning:
        return trend_tuning[key]
    return DEFAULT_TREND


def trend_forecast(point, h, window_h, damp):
    """Forecast = level now + damp * (rate of change) * h.

    The rate of change is the change over the last window_h hours, divided by
    window_h. If the reading from window_h hours ago is missing, the rate is
    treated as 0, so the forecast falls back to persistence.
    """
    level_before = point.shift(window_h)
    rate = (point - level_before) / window_h
    rate = rate.fillna(0.0)
    return point + damp * rate * h


def make_estimator(name, gbm_settings):
    """Return a new, untrained model.

    Everything the model learns (including the medians used to fill in missing
    values and the scaling for the linear model) comes only from the training
    rows of one fold.
    """
    if name == "linear":
        # Replace missing inputs with the training median (and add a
        # "was missing" column for each input).
        imputer = SimpleImputer(strategy="median", add_indicator=True)
        # Put every input on the same scale.
        scaler = StandardScaler()
        regression = LinearRegression()
        return make_pipeline(imputer, scaler, regression)

    # The GBM can handle missing inputs on its own.
    return HistGradientBoostingRegressor(**gbm_settings)


def features_for(meta, name="gbm_A"):
    """Return the list of input columns for one model, in order."""
    columns = []
    for group in MODEL_INPUTS[name]:
        for column in meta["feature_groups"][group]:
            columns.append(column)
    return columns


# ---------------------------------------------------------------------------
# Physics
# ---------------------------------------------------------------------------

def _rises_with_upstream(column):
    """Return True if this column is an upstream level or a recent upstream rise.

    More of either means more water is coming, so the forecast should go up.
    """
    # Current upstream levels.
    if column in UPSTREAM_LEVELS:
        return True

    # Recent rises, e.g. "natrona_d6" = change in the Natrona level over 6 hours.
    gauge = column.split("_")[0]
    if gauge not in UPSTREAM_RISE_GAUGES:
        return False
    if column.endswith("_d1") or column.endswith("_d6") or column.endswith("_d12"):
        return True
    return False


def monotonic_constraints(feats):
    """Return the monotonic constraints for the GBM.

    A value of +1 means the forecast is only allowed to go up as this input
    goes up. This applies to upstream levels and rises, rain totals, and the
    routing forecast "phys". Point's own history, the season and the flags are
    left unconstrained (they are simply not listed).
    """
    constraints = {}
    for column in feats:
        if _rises_with_upstream(column):
            constraints[column] = 1
        elif column.startswith("rain_"):
            constraints[column] = 1
        elif column == "phys":
            constraints[column] = 1
    return constraints


def perturb_upstream(X, delta):
    """Return a copy of X where every upstream gauge is delta ft higher."""
    X_perturbed = X.copy()
    for column in X.columns:
        if _rises_with_upstream(column):
            X_perturbed[column] = X[column] + delta
    return X_perturbed


def routing_designs(f, horizons):
    """Build the routing model's inputs from the raw upstream levels.

    Only readings at or before the issue time t are used.
    """
    all_levels = load_hourly(CLEAN_HOURLY)
    levels = all_levels[list(ROUTED_GAUGES)]
    levels = levels.reindex(f.index)

    designs = {}
    for h in horizons:
        designs[h] = routing_design(levels, h)
    return designs


def routing_fit(f, designs, train):
    """Fit the routing model on the training hours.

    Returns the fitted model and its predicted change for every hour, as a
    dictionary {horizon: Series}.
    """
    targets = {}
    for h in designs:
        targets[h] = f[f"y_{h}h"]

    model = RoutingModel()
    model = model.fit(designs, f["point"], targets, train)
    predicted_change = model.predict_change(designs, f["point"])
    return model, predicted_change


def routing_by_fold(f, horizons, split=SPLIT):
    """Fit one routing model per fold.

    Returns {test water year: (fitted routing model, {horizon: predicted change})}.
    """
    designs = routing_designs(f, horizons)

    issue_wy = f["water_year"]
    longest_horizon = max(horizons)
    target_wy = f["water_year"].shift(-longest_horizon)
    has_level_now = f["point"].notna()

    routing = {}
    for year in fold_years(f, split):
        train = train_mask(issue_wy, target_wy, year, split) & has_level_now
        train = train.to_numpy()
        routing[year] = routing_fit(f, designs, train)
    return routing


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def held_out(f, h, fit_predict, split=SPLIT):
    """Make held-out forecasts for horizon h, one test year at a time.

    fit_predict(train, test, target, wy) must train a model on the training
    rows and return two arrays for the test rows: the predicted change, and
    the predicted change with upstream raised.

    Returns two Series indexed by issue time t: the predicted level at t+h,
    and the perturbed predicted level at t+h. Both are NaN outside the test
    years.
    """
    target = f"dy_{h}h"

    pred = pd.Series(np.nan, index=f.index)
    pert = pd.Series(np.nan, index=f.index)

    issue_wy = f["water_year"]
    target_wy = issue_wy.shift(-h)
    has_level_now = f["point"].notna()
    has_target = f[target].notna()

    for wy in fold_years(f, split):
        test = has_level_now & (issue_wy == wy)
        train = has_target & has_level_now & train_mask(issue_wy, target_wy, wy, split)

        change, perturbed_change = fit_predict(train, test, target, wy)

        # The models predict a change, so add it to the current level.
        level_now = f.loc[test, "point"].to_numpy()
        pred[test] = level_now + change
        pert[test] = level_now + perturbed_change

    return pred, pert


def run_persistence(f, horizons, in_test):
    """Persistence: the forecast is just the current level.

    It ignores upstream, so the perturbed forecast is the same.
    """
    out = {}
    for h in horizons:
        out[f"pred_{h}h"] = f["point"].where(in_test)
        out[f"pert_{h}h"] = f["point"].where(in_test)
    return out


def run_trend(f, horizons, in_test, tuning):
    """Trend baseline. It ignores upstream, so the perturbed forecast is the same."""
    out = {}
    for h in horizons:
        settings = trend_settings(h, tuning)
        forecast = trend_forecast(f["point"], h, settings["window_h"], settings["damp"])
        forecast = forecast.where(in_test)
        out[f"pred_{h}h"] = forecast
        out[f"pert_{h}h"] = forecast
    return out


def run_trained_model(f, meta, name, split, tuning):
    """Linear model or one of the GBMs, trained separately for each fold."""
    horizons = meta["horizons"]
    feats = features_for(meta, name)
    uses_physics = name == "gbm_AR_mono"

    # The physics model needs a routing model fitted for each fold.
    routing = None
    if uses_physics:
        routing = routing_by_fold(f, horizons, split)

    params = gbm_params(tuning)
    if uses_physics:
        params["monotonic_cst"] = monotonic_constraints(feats + ["phys"])

    out = {}
    for h in horizons:

        # held_out calls this once per test year. It uses h from the loop above.
        def fit_predict(train, test, target, wy):
            # Build the inputs. The physics model gets the routing forecast too.
            if uses_physics:
                routing_model, routing_change = routing[wy]
                X = f[feats].copy()
                X["phys"] = routing_change[h]
            else:
                X = f[feats]

            # Train on this fold's training rows.
            model = make_estimator(name, params)
            model.fit(X[train], f.loc[train, target])

            # Inputs for the test rows, and the same inputs with upstream raised.
            X_test = X[test]
            X_perturbed = perturb_upstream(X_test, PERTURB_FT)
            if uses_physics:
                # Raising upstream also raises the routing forecast.
                gain = routing_model.step_gain(h)
                X_perturbed["phys"] = X_test["phys"] + PERTURB_FT * gain

            change = model.predict(X_test)
            perturbed_change = model.predict(X_perturbed)
            return change, perturbed_change

        pred, pert = held_out(f, h, fit_predict, split)
        out[f"pred_{h}h"] = pred
        out[f"pert_{h}h"] = pert

    return out


def run(f, meta, name, split=SPLIT):
    """Make held-out forecasts for one model.

    Returns a dictionary like {"pred_6h": ..., "pert_6h": ..., ...}.
    """
    horizons = meta["horizons"]
    test_years = fold_years(f, split)
    in_test = f["water_year"].isin(test_years)
    tuning = load_tuning()

    if name == REFERENCE:
        return run_persistence(f, horizons, in_test)

    if name == TREND:
        return run_trend(f, horizons, in_test, tuning)

    assert name in MODEL_INPUTS, f"unknown model {name}; choose from {ALL_MODELS}"
    return run_trained_model(f, meta, name, split, tuning)


# ---------------------------------------------------------------------------
# Saving and loading
# ---------------------------------------------------------------------------

def save(name, cols, split, test_years):
    """Save a model's forecasts and a small file saying how they were made."""
    OUT.mkdir(parents=True, exist_ok=True)

    table = pd.DataFrame(cols)
    table.to_parquet(OUT / f"{name}.parquet")

    info = {"split": split, "test_years": test_years}
    (OUT / f"{name}.json").write_text(json.dumps(info))


def load(name, horizons, kind="pred"):
    """Load a model's saved forecasts as {horizon: level at t+h}.

    Use kind="pert" to get the perturbed forecasts instead.
    """
    table = pd.read_parquet(OUT / f"{name}.parquet")

    forecasts = {}
    for h in horizons:
        forecasts[h] = table[f"{kind}_{h}h"]
    return forecasts


def split_info(name):
    """Return {"split": ..., "test_years": [...]} for a saved model."""
    path = OUT / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: rerun `python -m systems_final_project.models --force`")
    return json.loads(path.read_text())


def is_current(name, split):
    """Check whether a model's saved predictions can be reused.

    They can if they exist, are newer than the feature table and the tuning
    file, and were made with the same split.
    """
    pred_path = OUT / f"{name}.parquet"
    info_path = OUT / f"{name}.json"

    if not pred_path.exists() or not info_path.exists():
        return False

    # The files the predictions depend on.
    input_files = [FEATURES_DIR / "features.parquet"]
    if TUNING_FILE.exists():
        input_files.append(TUNING_FILE)

    # If any of them changed after the predictions were saved, retrain.
    for path in input_files:
        if pred_path.stat().st_mtime < path.stat().st_mtime:
            return False

    info = json.loads(info_path.read_text())
    return info["split"] == split


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", nargs="*", default=list(ALL_MODELS), choices=list(ALL_MODELS))
    parser.add_argument("--split", choices=["walk_forward", "loyo"], default=SPLIT)
    parser.add_argument("--force", action="store_true", help="retrain even if predictions are current")
    args = parser.parse_args()

    f, meta = load_features()
    years = fold_years(f, args.split)
    print(f"split: {args.split}, test water years {years[0]}-{years[-1]}")

    # Show the main GBM settings.
    settings_to_show = ("max_iter", "max_leaf_nodes", "min_samples_leaf", "learning_rate")
    shown = {}
    for key, value in gbm_params().items():
        if key in settings_to_show:
            shown[key] = value

    if TUNING_FILE.exists():
        source = "(tuned)"
    else:
        source = "(defaults: tune has not run)"
    print("GBM settings:", shown, source)

    # Train (or skip) each model.
    for name in args.run:
        if is_current(name, args.split) and not args.force:
            print(f"{name}: kept existing predictions (current)")
            continue

        start_time = time.time()
        predictions = run(f, meta, name, args.split)
        save(name, predictions, args.split, years)
        seconds = time.time() - start_time
        print(f"{name}: done in {seconds:.0f}s", flush=True)


if __name__ == "__main__":
    main()