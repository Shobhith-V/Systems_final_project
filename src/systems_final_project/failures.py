"""Find where each model fails, and tag a likely cause from the data around it.

Written by evaluation.py into results/ (read by the dashboard):
  events.csv        one row per (model, horizon, event): warned early / late / missed
  false_alarms.csv  one row per (model, horizon, alarm period) that matched no event
  worst_errors.csv  the largest forecast errors per (model, horizon), >= 3 days apart

Failure kinds
  missed       the river reached the alarm level and no forecast targeted the event
  late         an alarm targeted the event, but was issued only after the onset
  false alarm  an alarm period (alarm hours less than ALARM_MERGE_H apart) whose target
               times come no closer than EVENT_TOLERANCE_H to any event
  large error  |forecast - observed| among the largest of that model and horizon

Likely causes (simple rules on the data, labelled "likely" in the dashboard):
  gauge offline        an upstream gauge had no reading in the 24 h before the target
  borderline           the event peaked (or the false alarm's river reached) within
                       BORDERLINE_FT of the alarm level: a near miss, not a blind spot
  above training range the observed level exceeded every level in that fold's training years
  rain not yet fallen  most of the rain of the last 48 h fell within the horizon before the
                       onset, i.e. after the last issue time that could have warned in time
"""

import numpy as np
import pandas as pd

UPSTREAM = ("natrona", "kittanning", "sutersville", "braddock")
BORDERLINE_FT = 0.5
OFFLINE_WINDOW_H = 24
RAIN_WINDOW_H = 48
N_WORST = 10
ALARM_MERGE_H = 6        # alarm hours closer than this belong to one alarm period (flicker)
EVENT_TOLERANCE_H = 6    # an alarm aimed this close to an event is early, not false


# ------------------------------------------------------------------------------ context
def offline_gauges(clean: pd.DataFrame, end: pd.Timestamp, hours: int = OFFLINE_WINDOW_H) -> list[str]:
    """Upstream gauges with at least one missing hour in (end - hours, end]."""
    win = clean.loc[end - pd.Timedelta(hours=hours - 1): end, list(UPSTREAM)]
    return [g for g in UPSTREAM if win[g].isna().any()]


def training_max(level: pd.Series, water_year: pd.Series, wy: int, split: str) -> float:
    """Highest Point level in the training years of the fold that tests water year wy."""
    train = (water_year < wy) if split == "walk_forward" else (water_year != wy)
    return float(level[train].max())


def rain_share_within(clean: pd.DataFrame, onset: pd.Timestamp, h: int) -> float:
    """Share of the basin-mean rain of the last RAIN_WINDOW_H hours before onset that fell
    in the last h hours before onset. NaN if no rain."""
    rain = clean["rain_basin_mean"]
    total = rain.loc[onset - pd.Timedelta(hours=RAIN_WINDOW_H - 1): onset].sum()
    late = rain.loc[onset - pd.Timedelta(hours=h - 1): onset].sum()
    return float(late / total) if total > 0 else np.nan


def join(causes: list[str]) -> str:
    return "; ".join(causes) if causes else "no clear data cause"


# ------------------------------------------------------------------------------ tables
def event_table(events: pd.DataFrame, f, clean, thr, model, h, split) -> pd.DataFrame:
    """events: output of evaluation.event_scores for one model and horizon."""
    rows = []
    for ev in events.itertuples():
        outcome = ("warned early" if ev.warned and ev.lead_h > 0
                   else "late" if ev.warned else "missed")
        causes = []
        if outcome != "warned early":
            off = offline_gauges(clean, ev.start)
            if off:
                causes.append(f"gauge offline: {', '.join(off)}")
            if ev.peak_ft < thr + BORDERLINE_FT:
                causes.append(f"borderline: peak {ev.peak_ft:.2f} ft vs {thr:.0f} ft alarm level")
            wy = int(f.at[ev.start, "water_year"])
            top = training_max(f["point"], f["water_year"], wy, split)
            if ev.peak_ft > top:
                causes.append(f"above training range: peak {ev.peak_ft:.1f} ft > {top:.1f} ft")
            share = rain_share_within(clean, ev.start, h)
            if share > 0.5:
                causes.append(f"rain not yet fallen: {share:.0%} of 48 h rain fell in the last {h} h")
        rows.append({"model": model, "horizon_h": h, "start": ev.start, "end": ev.end,
                     "peak_ft": ev.peak_ft, "outcome": outcome, "lead_h": ev.lead_h,
                     "first_alarm_issued": ev.first_alarm_issued, "likely_cause": join(causes)})
    return pd.DataFrame(rows)


def false_alarm_table(pred: pd.Series, events, f, clean, thr, model, h) -> pd.DataFrame:
    """Alarm periods (issue hours with forecast >= thr, merged when < ALARM_MERGE_H apart)
    whose target times stay more than EVENT_TOLERANCE_H away from every event."""
    alarm_t = pred.index[pred.to_numpy() >= thr].to_series()
    if alarm_t.empty:
        return pd.DataFrame()
    period = (alarm_t.diff() > pd.Timedelta(hours=ALARM_MERGE_H)).cumsum()
    lag, tol = pd.Timedelta(hours=h), pd.Timedelta(hours=EVENT_TOLERANCE_H)
    rows = []
    for _, issue in alarm_t.groupby(period):
        t0, t1 = issue.min() + lag, issue.max() + lag            # target times covered
        touches = (((events["start"] - tol) <= t1) & ((events["end"] + tol) >= t0)).any() if len(events) else False
        if touches:
            continue
        obs_max = float(f["point"].loc[t0:t1].max())
        causes = []
        if obs_max >= thr - BORDERLINE_FT:
            causes.append(f"borderline: river reached {obs_max:.2f} ft")
        off = offline_gauges(clean, t0)
        if off:
            causes.append(f"gauge offline: {', '.join(off)}")
        rows.append({"model": model, "horizon_h": h, "first_issued": issue.min(), "last_issued": issue.max(),
                     "target_start": t0, "target_end": t1, "hours": len(issue),
                     "max_forecast_ft": float(pred.loc[issue].max()), "max_observed_ft": obs_max,
                     "likely_cause": join(causes)})
    return pd.DataFrame(rows)


def worst_errors(pred: pd.Series, f, clean, model, h, split, n: int = N_WORST,
                 min_gap_h: int = 72) -> pd.DataFrame:
    """The n largest |forecast - observed| at horizon h, at least min_gap_h apart."""
    y = f[f"y_{h}h"]
    err = (pred - y).dropna()
    chosen = []
    for t in err.abs().sort_values(ascending=False).index:
        if all(abs(t - c) >= pd.Timedelta(hours=min_gap_h) for c in chosen):
            chosen.append(t)
        if len(chosen) == n:
            break
    rows = []
    for t in chosen:
        target = t + pd.Timedelta(hours=h)
        causes = []
        off = offline_gauges(clean, target)
        if off:
            causes.append(f"gauge offline: {', '.join(off)}")
        top = training_max(f["point"], f["water_year"], int(f.at[t, "water_year"]), split)
        if y[t] > top:
            causes.append(f"above training range: {y[t]:.1f} ft > {top:.1f} ft")
        share = rain_share_within(clean, target, h)
        if share > 0.5:
            causes.append(f"rain not yet fallen: {share:.0%} of 48 h rain fell in the last {h} h")
        rows.append({"model": model, "horizon_h": h, "issued": t, "target_time": target,
                     "observed_ft": float(y[t]), "forecast_ft": float(pred[t]), "error_ft": float(err[t]),
                     "likely_cause": join(causes)})
    return pd.DataFrame(rows)
