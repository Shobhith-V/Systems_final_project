"""Evaluate the held-out forecasts and draw the two result figures.

Reads data/features/ and the leave-one-year-out predictions in results/predictions/
(run models.py first) and writes results/figures/:
  fig1_accuracy.png     error, alarm skill, early warning and physical consistency by horizon
                        (2016-2027, 40 events)
  fig2_april_2024.png   forecasts vs observed during the April 2024 floods (held-out year)

Scoring (per horizon 6/12/24 h, all models on the same hours, see eval_rows):
  - level error: RMSE / MAE on all hours, on high-water hours (truth >= 18 ft) and
    on rising hours (truth - now >= 1 ft)
  - hourly alarm skill for "level >= 18 ft at t+h": precision, recall (POD), CSI
  - event skill: a flood event (truth >= 18 ft, runs < 24 h apart merged) is warned
    if an alarm targets a time inside it; lead time = event onset minus the issue
    time of the first such alarm (> 0 = warned before onset)
A forecast is a {horizon h: predicted level at t+h} dict of Series indexed by issue time t.

Usage (from the repository root):
    uv run python -m systems_final_project.evaluation
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from systems_final_project import models  # noqa: E402
from systems_final_project.data.config import CLEAN_HOURLY, FIGURES_DIR  # noqa: E402
from systems_final_project.features import load_features, load_hourly  # noqa: E402
from systems_final_project.physics import ROUTED_GAUGES, RoutingModel, routing_design  # noqa: E402


# ------------------------------------------------------------------------------ scoring
def eval_rows(f: pd.DataFrame, h: int) -> pd.Index:
    """Hours scored at horizon h: current level and target both known."""
    return f.index[f["point"].notna() & f[f"y_{h}h"].notna()]


def flood_events(level: pd.Series, threshold: float, merge_gap_h: int = 24) -> pd.DataFrame:
    """Runs of hours with level >= threshold; runs < merge_gap_h apart are merged."""
    above = level.dropna()
    above = above[above >= threshold]
    if above.empty:
        return pd.DataFrame(columns=["start", "end", "peak_ft"])
    ev_id = (above.index.to_series().diff() > pd.Timedelta(hours=merge_gap_h)).cumsum()
    return (above.groupby(ev_id.values)
            .agg(start=lambda s: s.index.min(), end=lambda s: s.index.max(), peak_ft="max")
            .reset_index(drop=True))


def level_scores(y, pred, now, threshold) -> dict:
    err = pred - y
    high, rising = y >= threshold, (y - now) >= 1.0
    return {"n_hours": int(len(y)), "mae": err.abs().mean(), "rmse": np.sqrt((err ** 2).mean()),
            "mae_high_water": err[high].abs().mean(), "n_high_water": int(high.sum()),
            "mae_rising": err[rising].abs().mean(), "n_rising": int(rising.sum())}


def alarm_scores(y, pred, threshold) -> dict:
    truth, alarm = y >= threshold, pred >= threshold
    tp, fp, fn = int((truth & alarm).sum()), int((~truth & alarm).sum()), int((truth & ~alarm).sum())
    return {"alarm_tp": tp, "alarm_fp": fp, "alarm_fn": fn,
            "precision": tp / (tp + fp) if tp + fp else np.nan,
            "recall_pod": tp / (tp + fn) if tp + fn else np.nan,
            "csi": tp / (tp + fp + fn) if tp + fp + fn else np.nan}


def event_scores(events: pd.DataFrame, pred: pd.Series, h: int, threshold: float) -> pd.DataFrame:
    """pred is indexed by issue time t and predicts the level at t+h."""
    alarm_t = pred.index[pred.to_numpy() >= threshold]
    target_t = alarm_t + pd.Timedelta(hours=h)
    rows = []
    for ev in events.itertuples():
        hit = (target_t >= ev.start) & (target_t <= ev.end)
        first = alarm_t[hit].min() if hit.any() else pd.NaT
        lead = (ev.start - first) / pd.Timedelta(hours=1) if hit.any() else np.nan
        rows.append({"horizon_h": h, "start": ev.start, "end": ev.end, "peak_ft": ev.peak_ft,
                     "warned": bool(hit.any()), "first_alarm_issued": first, "lead_h": lead})
    return pd.DataFrame(rows)


def evaluate(f: pd.DataFrame, preds: dict[int, pd.Series], threshold: float, model: str):
    """Score one model: one summary row per horizon. preds[h] is the predicted level
    at t+h indexed by issue time t, covering eval_rows()."""
    events = flood_events(f["point"], threshold)
    summary = []
    for h, pred in preds.items():
        idx = eval_rows(f, h)
        pred = pred.reindex(idx)
        assert pred.notna().all(), f"{model}: missing predictions at {h} h"
        now, y = f.loc[idx, "point"], f.loc[idx, f"y_{h}h"]
        ev = event_scores(events, pred, h, threshold)
        summary.append({"model": model, "horizon_h": h,
                        **level_scores(y, pred, now, threshold), **alarm_scores(y, pred, threshold),
                        "n_events": len(ev), "events_warned": int(ev["warned"].sum()),
                        "events_warned_early": int((ev["lead_h"] > 0).sum()),
                        "median_lead_h": ev["lead_h"].median()})
    return pd.DataFrame(summary)


# ------------------------------------------------------------------------------ figures
NAMES = ["persistence", "gbm_A", "gbm_AR_mono"]
LABEL = {"persistence": "Persistence (baseline)", "gbm_A": "GBM A",
         "gbm_AR_mono": "GBM AR_mono (physics-informed)"}
COLOR = {"persistence": "#a3a29c", "gbm_A": "#2a78d6", "gbm_AR_mono": "#eb6834"}
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8984"

plt.rcParams.update({
    "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#c9c8c2", "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": "#ebeae6", "grid.linewidth": 0.6, "axes.axisbelow": True,
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "legend.frameon": False,
    "axes.titlesize": 11, "axes.titlelocation": "left",
})


# ------------------------------------------------------------------------------ helpers
def title(fig, text, sub=None):
    """Title (+ subtitle) in a fixed band of inches at the top, plots below it."""
    h = fig.get_size_inches()[1]
    fig.suptitle(text, x=0.01, y=1 - 0.1 / h, ha="left", va="top", fontsize=13, color=INK)
    if sub:
        fig.text(0.01, 1 - 0.42 / h, sub, ha="left", va="top", fontsize=9, color=INK2)
    fig.get_layout_engine().set(rect=(0, 0, 1, 1 - (0.72 if sub else 0.45) / h))


def save(fig, name):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / name, dpi=150)
    plt.close(fig)
    print("saved", FIGURES_DIR / name)


# ------------------------------------------------------------------------------ fig 1
def fig_accuracy(summary, horizons, thr):
    """Seven metrics per horizon for every model, from evaluate() and consistency()."""
    n_ev = int(summary["n_events"].iloc[0])
    panels = [("rmse", "RMSE, all hours (ft)  - lower is better", "%.2f"),
              ("mae_high_water", f"MAE when observed >= {thr:.0f} ft (ft)  - lower is better", "%.2f"),
              ("mae_rising", "MAE while rising >= 1 ft (ft)  - lower is better", "%.2f"),
              ("csi", f"Hourly alarm skill at {thr:.0f} ft (CSI)  - higher is better", "%.2f"),
              ("events_warned_early", f"Events warned before onset (of {n_ev})  - higher is better", "%d"),
              ("median_lead_h", "Median warning lead time (h)  - higher is better", "%.0f"),
              ("violations_pct", f"Forecasts that FALL when upstream is {models.PERTURB_FT} ft higher (%)"
                                 "  - lower is better", "%.1f")]
    fig, axes = plt.subplots(2, 4, figsize=(22, 8.8), constrained_layout=True)
    x, w = np.arange(len(horizons)), 0.26
    for ax, (col, text, fmt) in zip(axes.flat, panels):
        for i, n in enumerate(NAMES):
            vals = [summary.query("model == @n and horizon_h == @h")[col].iloc[0] for h in horizons]
            bars = ax.bar(x + (i - 1) * w, vals, w * 0.92, color=COLOR[n], label=LABEL[n])
            ax.bar_label(bars, fmt=fmt, fontsize=7.5, padding=2, color=INK2)
        if col == "events_warned_early":
            ax.axhline(n_ev, color=MUTED, ls="--", lw=1)
            ax.set_ylim(0, n_ev * 1.12)
        ax.set_xticks(x, [f"{h} h ahead" for h in horizons])
        ax.set_title(text, fontsize=10.5)
        ax.grid(axis="x", visible=False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    axes[1, 3].axis("off")
    axes[1, 3].legend(handles, labels, loc="center", fontsize=11, title="Model", title_fontsize=11)
    title(fig, f"Model accuracy, leave-one-water-year-out over 2016-2027 ({n_ev} flood events >= {thr:.0f} ft)",
          "Every hour is predicted by a model that never saw its water year. Persistence = 'the level "
          "stays as it is now'; it can never warn before a flood starts. Physical-consistency test: "
          "all upstream gauges raised at the issue time.")
    save(fig, "fig1_accuracy.png")


# ------------------------------------------------------------------------------ fig 2
def fig_april_2024(f, preds, horizons, thr, start="2024-03-31", end="2024-04-18"):
    """Observed level and each model's forecast, drawn at the time it predicts."""
    lo, hi = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
    truth = f["point"][lo:hi]
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True, constrained_layout=True)
    style = {"persistence": dict(ls="--", lw=1.5), "gbm_A": dict(lw=1.9), "gbm_AR_mono": dict(lw=1.6)}
    for ax, h in zip(axes, horizons):
        ax.plot(truth.index, truth.values, color=INK, lw=2.2, label="Observed")
        for n in NAMES:
            s = preds[n][h].copy()
            s.index = s.index + pd.Timedelta(hours=h)   # issue time t -> target time t+h
            ax.plot(s[lo:hi].index, s[lo:hi].values, color=COLOR[n], label=LABEL[n], **style[n])
        ax.axhline(thr, color=MUTED, ls=":", lw=1)
        ax.text(lo, thr + 0.2, f"{thr:.0f} ft NWS action stage", color=INK2, fontsize=8.5)
        ax.set_title(f"{h} h ahead (each forecast drawn at the time it predicts)")
        ax.set_ylabel("Point level (ft)")
    axes[0].legend(loc="upper right", fontsize=9)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    title(fig, "April 2024 floods, the highest level in the record (28.3 ft), in a held-out year")
    save(fig, "fig2_april_2024.png")


def consistency(f, name, horizons) -> dict[int, float]:
    """% of held-out hours whose forecast falls by > 0.01 ft when every upstream gauge is
    raised by models.PERTURB_FT ft (persistence ignores upstream, so it never falls)."""
    if name == models.REFERENCE:
        return {h: 0.0 for h in horizons}
    pred, pert = models.load(name, horizons), models.load(name, horizons, kind="pert")
    out = {}
    for h in horizons:
        d = (pert[h] - pred[h]).dropna()
        out[h] = 100 * float((d < -0.01).mean())
    return out


def main() -> None:
    f, meta = load_features()
    horizons, thr = meta["horizons"], meta["flood_ft"]
    preds = {n: models.load(n, horizons) for n in NAMES}
    summary = pd.concat([evaluate(f, preds[n], thr, n) for n in NAMES], ignore_index=True)
    summary["violations_pct"] = [consistency(f, m, horizons)[h] for m, h in zip(summary.model, summary.horizon_h)]
    print(summary[["model", "horizon_h", "rmse", "mae_high_water", "mae_rising", "precision", "recall_pod",
                   "csi", "events_warned_early", "median_lead_h", "violations_pct"]].round(3).to_string(index=False))
    fig_accuracy(summary, horizons, thr)
    fig_april_2024(f, preds, horizons, thr)

    # routing model fitted on all years, for interpreting gbm_AR_mono's physics input
    levels = load_hourly(CLEAN_HOURLY)[list(ROUTED_GAUGES)].reindex(f.index)
    designs = {h: routing_design(levels, h) for h in horizons}
    rm = RoutingModel().fit(designs, f["point"], {h: f[f"y_{h}h"] for h in horizons},
                            f["point"].notna().to_numpy())
    print("\nrouting kernels (all years):")
    print(rm.summary().round(2).to_string(index=False))
    print(f"pool level the Point drains toward: {rm.pool_level(horizons[0]):.2f} ft")


if __name__ == "__main__":
    main()
