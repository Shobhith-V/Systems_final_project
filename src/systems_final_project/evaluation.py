"""
Evaluate the held-out forecasts and draw the result figures.

This reads data/features/ and the held-out predictions in results/predictions/
(so run models.py first), and writes:

  results/figures/fig1_accuracy.png
      Error, alarm skill, early warning and physical consistency, by horizon.
  results/figures/fig2_april_2024.png
      Forecasts vs observed levels during the April 2024 floods (a test year).
  results/figures/fig3_ablation_*.png
      Each ablation variant compared with gbm_A (e.g. + Elizabeth gauge).
  results/summary.csv
      The scores behind figs 1 and 3 (every model, every horizon).
  results/events.csv, results/false_alarms.csv, results/worst_errors.csv
      Where each model fails, with a likely cause (see failures.py).

Only the test water years saved with the predictions are scored. For
walk_forward that is WY2019-2026; the validation year WY2018 is never scored.

How the scoring works (for each horizon: 6, 12 and 24 h). All models are
scored on exactly the same hours (see eval_rows).

  - Level error: RMSE and MAE on all hours, on high-water hours
    (truth >= 18 ft), and on rising hours (truth - now >= 1 ft).
  - Hourly alarm skill for "level >= 18 ft at t+h": precision, recall (POD)
    and CSI.
  - Event skill: a flood event is a run of hours with truth >= 18 ft (runs
    less than 24 h apart are merged into one event). An event counts as
    warned if some alarm targets a time inside it. The lead time is the event
    onset minus the issue time of the first such alarm, so a positive lead
    time means the warning came before the flood started.

Throughout this file, a "forecast" is a dictionary
{horizon h: predicted level at t+h}, where each value is a Series indexed by
the issue time t.

Run from the repository root:
    uv run python -m systems_final_project.evaluation
"""

# matplotlib has to be told to draw to files (no screen) before pyplot is
# imported. That's why the imports below come after this line; the
# "noqa: E402" comments stop the linter from complaining about it.
import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from systems_final_project import failures, models  # noqa: E402
from systems_final_project.data.config import CLEAN_HOURLY, FIGURES_DIR, RESULTS_DIR  # noqa: E402
from systems_final_project.features import load_features, load_hourly  # noqa: E402
from systems_final_project.physics import ROUTED_GAUGES, RoutingModel, routing_design  # noqa: E402


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def eval_rows(f, h, years):
    """Return the hours that are scored at horizon h.

    An hour is scored if it was issued in a test year and both the current
    level and the level h hours later are known.
    """
    in_test_year = f["water_year"].isin(years)
    has_level_now = f["point"].notna()
    has_target = f[f"y_{h}h"].notna()

    keep = in_test_year & has_level_now & has_target
    return f.index[keep]


def flood_events(level, threshold, merge_gap_h=24):
    """Find flood events: runs of hours where the level is at or above threshold.

    If two runs are less than merge_gap_h hours apart, they are merged into
    one event. Returns a table with one row per event and the columns
    start, end and peak_ft.
    """
    # Keep only the hours at or above the threshold.
    above = level.dropna()
    above = above[above >= threshold]

    if above.empty:
        return pd.DataFrame(columns=["start", "end", "peak_ft"])

    times = list(above.index)
    levels = list(above.values)
    max_gap = pd.Timedelta(hours=merge_gap_h)

    events = []

    # Start the first event at the first high-water hour.
    event_start = times[0]
    event_end = times[0]
    event_peak = levels[0]

    for i in range(1, len(times)):
        gap = times[i] - times[i - 1]

        if gap > max_gap:
            # Too long since the last high-water hour: close this event and
            # start a new one.
            events.append({"start": event_start, "end": event_end, "peak_ft": event_peak})
            event_start = times[i]
            event_end = times[i]
            event_peak = levels[i]
        else:
            # Still the same event: extend it and update the peak.
            event_end = times[i]
            if levels[i] > event_peak:
                event_peak = levels[i]

    # Don't forget the last event.
    events.append({"start": event_start, "end": event_end, "peak_ft": event_peak})

    return pd.DataFrame(events, columns=["start", "end", "peak_ft"])


def level_scores(y, pred, now, threshold):
    """Error scores for the predicted levels.

    y is the true level, pred the predicted level and now the level at the
    issue time.
    """
    error = pred - y
    absolute_error = error.abs()
    squared_error = error ** 2

    # High-water hours: the true level is at or above the threshold.
    is_high_water = y >= threshold

    # Rising hours: the river rose at least 1 ft between now and the target.
    is_rising = (y - now) >= 1.0

    scores = {}
    scores["n_hours"] = int(len(y))
    scores["mae"] = absolute_error.mean()
    scores["rmse"] = np.sqrt(squared_error.mean())
    scores["mae_high_water"] = error[is_high_water].abs().mean()
    scores["n_high_water"] = int(is_high_water.sum())
    scores["mae_rising"] = error[is_rising].abs().mean()
    scores["n_rising"] = int(is_rising.sum())
    return scores


def alarm_scores(y, pred, threshold):
    """Hourly alarm skill for "the level will be at or above threshold".

    An alarm is raised whenever the predicted level reaches the threshold.
    """
    truly_high = y >= threshold
    alarm = pred >= threshold

    # Count hits, false alarms and misses.
    true_positives = int((truly_high & alarm).sum())
    false_positives = int((~truly_high & alarm).sum())
    false_negatives = int((truly_high & ~alarm).sum())

    # Precision: of all alarms, how many were right?
    if true_positives + false_positives > 0:
        precision = true_positives / (true_positives + false_positives)
    else:
        precision = np.nan

    # Recall (probability of detection): of all high-water hours, how many
    # got an alarm?
    if true_positives + false_negatives > 0:
        recall = true_positives / (true_positives + false_negatives)
    else:
        recall = np.nan

    # CSI (critical success index): hits divided by hits + false alarms + misses.
    total = true_positives + false_positives + false_negatives
    if total > 0:
        csi = true_positives / total
    else:
        csi = np.nan

    scores = {}
    scores["alarm_tp"] = true_positives
    scores["alarm_fp"] = false_positives
    scores["alarm_fn"] = false_negatives
    scores["precision"] = precision
    scores["recall_pod"] = recall
    scores["csi"] = csi
    return scores


def event_scores(events, pred, h, threshold):
    """Check, for each flood event, whether it was warned and how early.

    pred is indexed by the issue time t and predicts the level at t+h.
    """
    # Issue times of all alarms, and the times those alarms are about.
    is_alarm = pred.to_numpy() >= threshold
    alarm_times = pred.index[is_alarm]
    target_times = alarm_times + pd.Timedelta(hours=h)

    rows = []
    for event in events.itertuples():
        # Alarms whose target time falls inside this event.
        hits = (target_times >= event.start) & (target_times <= event.end)
        was_warned = bool(hits.any())

        if was_warned:
            first_alarm = alarm_times[hits].min()
            lead_hours = (event.start - first_alarm) / pd.Timedelta(hours=1)
        else:
            first_alarm = pd.NaT
            lead_hours = np.nan

        row = {
            "horizon_h": h,
            "start": event.start,
            "end": event.end,
            "peak_ft": event.peak_ft,
            "warned": was_warned,
            "first_alarm_issued": first_alarm,
            "lead_h": lead_hours,
        }
        rows.append(row)

    return pd.DataFrame(rows)


def evaluate(f, preds, threshold, model, years):
    """Score one model over the test years.

    preds[h] is the predicted level at t+h, indexed by issue time t, and must
    cover every hour in eval_rows(). Returns one summary row per horizon.
    """
    # Flood events in the test years (the same for every horizon).
    level_in_test_years = f["point"].where(f["water_year"].isin(years))
    events = flood_events(level_in_test_years, threshold)

    summary = []
    for h in preds:
        hours = eval_rows(f, h, years)
        pred = preds[h].reindex(hours)
        assert pred.notna().all(), f"{model}: missing predictions at {h} h"

        now = f.loc[hours, "point"]
        y = f.loc[hours, f"y_{h}h"]

        level = level_scores(y, pred, now, threshold)
        alarms = alarm_scores(y, pred, threshold)
        event_table = event_scores(events, pred, h, threshold)

        # Build the summary row one column at a time.
        row = {}
        row["model"] = model
        row["horizon_h"] = h
        for key, value in level.items():
            row[key] = value
        for key, value in alarms.items():
            row[key] = value
        row["n_events"] = len(event_table)
        row["events_warned"] = int(event_table["warned"].sum())
        row["events_warned_early"] = int((event_table["lead_h"] > 0).sum())
        row["median_lead_h"] = event_table["lead_h"].median()

        summary.append(row)

    return pd.DataFrame(summary)


def write_failures(f, preds, names, thr, years, split, horizons):
    """Write events.csv, false_alarms.csv and worst_errors.csv for every model and horizon."""
    clean = load_hourly(CLEAN_HOURLY).reindex(f.index)

    level_in_test_years = f["point"].where(f["water_year"].isin(years))
    events = flood_events(level_in_test_years, thr)

    event_tables = []
    false_alarm_tables = []
    worst_error_tables = []

    for name in names:
        for h in horizons:
            pred = preds[name][h].reindex(eval_rows(f, h, years))
            event_table = event_scores(events, pred, h, thr)

            event_tables.append(failures.event_table(event_table, f, clean, thr, name, h, split))
            false_alarm_tables.append(failures.false_alarm_table(pred, events, f, clean, thr, name, h))
            worst_error_tables.append(failures.worst_errors(pred, f, clean, name, h, split))

    outputs = [
        ("events", event_tables),
        ("false_alarms", false_alarm_tables),
        ("worst_errors", worst_error_tables),
    ]
    for file_name, tables in outputs:
        combined = pd.concat(tables, ignore_index=True)
        path = RESULTS_DIR / f"{file_name}.csv"
        combined.to_csv(path, index=False)
        print("saved", path)


# ---------------------------------------------------------------------------
# Figure settings
# ---------------------------------------------------------------------------

# Models shown in fig 1 and fig 2.
NAMES = list(models.MAIN)

# Fig 3: each set is compared with gbm_A.
ABLATION_SETS = {"elizabeth": ["gbm_A", "gbm_A_eliz"]}

# How each model is named in the legends.
LABEL = {
    "persistence": "Persistence (baseline)",
    "persistence_trend": "Persistence + trend (rule)",
    "linear": "Linear regression",
    "gbm_A": "GBM A",
    "gbm_AR_mono": "GBM AR_mono (physics-informed)",
    "gbm_A_eliz": "GBM A + Elizabeth",
}

# Colour for each model.
COLOR = {
    "persistence": "#a3a29c",
    "persistence_trend": "#6b6a64",
    "linear": "#2e9d6f",
    "gbm_A": "#2a78d6",
    "gbm_AR_mono": "#eb6834",
    "gbm_A_eliz": "#7fb0e8",
}

# Text colours: main, secondary and faded.
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8984"

# Overall look of every figure.
plt.rcParams.update({
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#c9c8c2",
    "axes.labelcolor": INK2,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "axes.grid": True,
    "grid.color": "#ebeae6",
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "figure.facecolor": "#fcfcfb",
    "axes.facecolor": "#fcfcfb",
    "legend.frameon": False,
    "axes.titlesize": 11,
    "axes.titlelocation": "left",
})


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------

def title(fig, text, sub=None):
    """Put a title (and optional subtitle) at the top of the figure.

    The title takes up a fixed height in inches, and the plots are fitted
    into the space below it.
    """
    fig_height = fig.get_size_inches()[1]

    fig.suptitle(text, x=0.01, y=1 - 0.1 / fig_height, ha="left", va="top", fontsize=13, color=INK)

    if sub:
        fig.text(0.01, 1 - 0.42 / fig_height, sub, ha="left", va="top", fontsize=9, color=INK2)
        title_height = 0.72
    else:
        title_height = 0.45

    # Leave room at the top for the title.
    plot_top = 1 - title_height / fig_height
    fig.get_layout_engine().set(rect=(0, 0, 1, plot_top))


def save(fig, name):
    """Save a figure to the figures folder and close it."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("saved", path)


# ---------------------------------------------------------------------------
# Fig 1: accuracy
# ---------------------------------------------------------------------------

def split_text(split, years):
    """Short description of the split, e.g. "walk-forward, test WY2019-2026"."""
    first_year = min(years)
    last_year = max(years)
    if split == "loyo":
        return f"leave-one-water-year-out, WY{first_year}-{last_year}"
    return f"walk-forward, test WY{first_year}-{last_year}"


def fig_accuracy(summary, horizons, thr, split, years, names=None, filename="fig1_accuracy.png",
                 heading="Model accuracy"):
    """Bar charts of seven scores per horizon for each model in names.

    The scores come from evaluate() and consistency().
    """
    if not names:
        names = NAMES

    number_of_events = int(summary["n_events"].iloc[0])

    # Each panel: (column in summary, panel title, number format for the bar labels).
    panels = [
        ("rmse",
         "RMSE, all hours (ft)  - lower is better",
         "%.2f"),
        ("mae_high_water",
         f"MAE when observed >= {thr:.0f} ft (ft)  - lower is better",
         "%.2f"),
        ("mae_rising",
         "MAE while rising >= 1 ft (ft)  - lower is better",
         "%.2f"),
        ("csi",
         f"Hourly alarm skill at {thr:.0f} ft (CSI)  - higher is better",
         "%.2f"),
        ("events_warned_early",
         f"Events warned before onset (of {number_of_events})  - higher is better",
         "%d"),
        ("median_lead_h",
         "Median warning lead time (h)  - higher is better",
         "%.0f"),
        ("violations_pct",
         f"Forecasts that FALL when upstream is {models.PERTURB_FT} ft higher (%)  - lower is better",
         "%.1f"),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(22, 8.8), constrained_layout=True)
    all_axes = axes.flatten()

    # One group of bars per horizon, one bar per model inside each group.
    x = np.arange(len(horizons))
    bar_width = 0.8 / len(names)

    for panel_number in range(len(panels)):
        ax = all_axes[panel_number]
        column, panel_title, number_format = panels[panel_number]

        for model_number in range(len(names)):
            model = names[model_number]

            # This model's score at each horizon.
            values = []
            for h in horizons:
                is_this_row = (summary["model"] == model) & (summary["horizon_h"] == h)
                value = summary[is_this_row][column].iloc[0]
                values.append(value)

            # Shift each model's bars sideways so they sit next to each other.
            offset = (model_number - (len(names) - 1) / 2) * bar_width
            bars = ax.bar(x + offset, values, bar_width * 0.92, color=COLOR[model], label=LABEL[model])
            ax.bar_label(bars, fmt=number_format, fontsize=7.5, padding=2, color=INK2)

        # For the events panel, draw a dashed line at the total number of events.
        if column == "events_warned_early":
            ax.axhline(number_of_events, color=MUTED, ls="--", lw=1)
            ax.set_ylim(0, number_of_events * 1.12)

        tick_labels = []
        for h in horizons:
            tick_labels.append(f"{h} h ahead")
        ax.set_xticks(x, tick_labels)

        ax.set_title(panel_title, fontsize=10.5)
        ax.grid(axis="x", visible=False)

    # There are 7 panels in a 2 x 4 grid, so use the 8th spot for the legend.
    handles, labels = axes[0, 0].get_legend_handles_labels()
    axes[1, 3].axis("off")
    axes[1, 3].legend(handles, labels, loc="center", fontsize=11, title="Model", title_fontsize=11)

    if split == "loyo":
        trained_on = "all other water years"
    else:
        trained_on = "only earlier water years"

    main_title = f"{heading}, {split_text(split, years)} ({number_of_events} events >= {thr:.0f} ft)"
    subtitle = (
        f"Every hour is predicted by a model trained on {trained_on}; settings chosen on the "
        "validation year. Persistence = 'the level stays as it is now'; it can never warn before a "
        "flood starts. Physical-consistency test: all upstream gauges raised at the issue time."
    )
    title(fig, main_title, subtitle)
    save(fig, filename)


# ---------------------------------------------------------------------------
# Fig 2: April 2024 floods
# ---------------------------------------------------------------------------

def fig_april_2024(f, preds, horizons, thr, start="2024-03-31", end="2024-04-18"):
    """Plot the observed level and each model's forecast during April 2024.

    Each forecast is drawn at the time it predicts (not the time it was made).
    """
    start_time = pd.Timestamp(start, tz="UTC")
    end_time = pd.Timestamp(end, tz="UTC")
    truth = f["point"][start_time:end_time]

    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True, constrained_layout=True)

    # Line style and width for each model.
    line_styles = {
        "persistence": "--",
        "persistence_trend": "-.",
        "linear": "-",
        "gbm_A": "-",
        "gbm_AR_mono": "-",
    }
    line_widths = {
        "persistence": 1.5,
        "persistence_trend": 1.3,
        "linear": 1.4,
        "gbm_A": 1.9,
        "gbm_AR_mono": 1.6,
    }

    # One panel per horizon.
    for i in range(len(horizons)):
        ax = axes[i]
        h = horizons[i]

        ax.plot(truth.index, truth.values, color=INK, lw=2.2, label="Observed")

        for name in NAMES:
            # Move each forecast from its issue time t to the time it predicts, t+h.
            forecast = preds[name][h].copy()
            forecast.index = forecast.index + pd.Timedelta(hours=h)

            window = forecast[start_time:end_time]
            ax.plot(window.index, window.values, color=COLOR[name], label=LABEL[name],
                    ls=line_styles[name], lw=line_widths[name])

        # Dotted line at the flood threshold.
        ax.axhline(thr, color=MUTED, ls=":", lw=1)
        if thr == 18:
            threshold_name = "NWS action stage"
        else:
            threshold_name = "alarm level"
        ax.text(start_time, thr + 0.2, f"{thr:.0f} ft {threshold_name}", color=INK2, fontsize=8.5)

        ax.set_title(f"{h} h ahead (each forecast drawn at the time it predicts)")
        ax.set_ylabel("Point level (ft)")

    axes[0].legend(loc="upper right", fontsize=9)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

    title(fig, "April 2024 floods, the highest level in the record (28.3 ft), in a test year")
    save(fig, "fig2_april_2024.png")


# ---------------------------------------------------------------------------
# Physical consistency
# ---------------------------------------------------------------------------

def consistency(name, horizons):
    """Percent of held-out hours where the forecast drops when upstream goes up.

    "Drops" means it falls by more than 0.01 ft when every upstream gauge is
    raised by models.PERTURB_FT ft. Persistence ignores upstream, so it never
    drops. Returns {horizon: percent}.
    """
    result = {}

    if name == models.REFERENCE:
        for h in horizons:
            result[h] = 0.0
        return result

    normal = models.load(name, horizons)
    perturbed = models.load(name, horizons, kind="pert")

    for h in horizons:
        difference = perturbed[h] - normal[h]
        difference = difference.dropna()
        dropped = difference < -0.01
        result[h] = 100 * float(dropped.mean())

    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    f, meta = load_features()
    horizons = meta["horizons"]
    thr = meta["flood_ft"]

    # Every main model must have predictions.
    missing = []
    for name in NAMES:
        if not (models.OUT / f"{name}.parquet").exists():
            missing.append(name)
    assert not missing, f"no predictions for {missing}: run `python -m systems_final_project.models` first"

    # Ablation models are optional: include the ones that have predictions.
    ablation_names = []
    for set_names in ABLATION_SETS.values():
        for name in set_names:
            if name not in NAMES:
                ablation_names.append(name)

    names = list(NAMES)
    for name in ablation_names:
        if (models.OUT / f"{name}.parquet").exists():
            names.append(name)

    # All models must have been trained with the same split.
    infos = {}
    for name in names:
        infos[name] = models.split_info(name)

    splits = set()
    for info in infos.values():
        splits.add(info["split"])
    assert len(splits) == 1, f"models were trained with different splits {infos}; rerun models --force"

    split = splits.pop()
    years = infos[NAMES[0]]["test_years"]
    print(f"scoring {split_text(split, years)}")

    # Load the predictions and score every model.
    preds = {}
    for name in names:
        preds[name] = models.load(name, horizons)

    summaries = []
    for name in names:
        summaries.append(evaluate(f, preds[name], thr, name, years))
    summary = pd.concat(summaries, ignore_index=True)

    # Add the physical-consistency score to each row.
    violations = {}
    for name in names:
        violations[name] = consistency(name, horizons)

    violations_column = []
    for i in range(len(summary)):
        model = summary["model"].iloc[i]
        h = summary["horizon_h"].iloc[i]
        violations_column.append(violations[model][h])
    summary["violations_pct"] = violations_column

    summary.insert(0, "split", split)

    # Save and print the summary.
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(RESULTS_DIR / "summary.csv", index=False)

    columns_to_print = ["model", "horizon_h", "rmse", "mae_high_water", "mae_rising", "precision",
                        "recall_pod", "csi", "events_warned_early", "median_lead_h", "violations_pct"]
    print(summary[columns_to_print].round(3).to_string(index=False))
    print("saved", RESULTS_DIR / "summary.csv")

    # Failure tables and figures.
    write_failures(f, preds, names, thr, years, split, horizons)
    fig_accuracy(summary, horizons, thr, split, years)
    fig_april_2024(f, preds, horizons, thr)

    for key, set_names in ABLATION_SETS.items():
        # Only draw an ablation figure if all its models have predictions.
        missing_from_set = []
        for name in set_names:
            if name not in names:
                missing_from_set.append(name)

        if len(missing_from_set) == 0:
            fig_accuracy(summary, horizons, thr, split, years, names=set_names,
                         filename=f"fig3_ablation_{key}.png", heading=f"Ablation: {key} gauge")
        else:
            print(f"ablation '{key}' skipped: no predictions for {missing_from_set}")

    # Fit the routing model on all years, to help interpret gbm_AR_mono's
    # physics input.
    all_levels = load_hourly(CLEAN_HOURLY)
    levels = all_levels[list(ROUTED_GAUGES)].reindex(f.index)

    designs = {}
    targets = {}
    for h in horizons:
        designs[h] = routing_design(levels, h)
        targets[h] = f[f"y_{h}h"]

    has_level = f["point"].notna().to_numpy()
    routing_model = RoutingModel()
    routing_model = routing_model.fit(designs, f["point"], targets, has_level)

    print("\nrouting kernels (all years):")
    print(routing_model.summary().round(2).to_string(index=False))
    print(f"pool level the Point drains toward: {routing_model.pool_level(horizons[0]):.2f} ft")


if __name__ == "__main__":
    main()