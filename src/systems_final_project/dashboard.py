"""Streamlit dashboard: how well the forecasts do, replays of past events, and failures.

Reads only files the pipeline writes (it never trains anything):
  data/features/features.parquet, data/clean/hourly.parquet, results/predictions/*,
  results/summary.csv, events.csv, false_alarms.csv, worst_errors.csv, tuning.json

Run (from the repository root, after the pipeline):
    uv run streamlit run src/systems_final_project/dashboard.py
"""

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from systems_final_project import models
from systems_final_project.data.config import CLEAN_HOURLY, FEATURES_DIR, RESULTS_DIR
from systems_final_project.evaluation import COLOR, LABEL, split_text

st.set_page_config(page_title="Point water-level forecasts", page_icon="🌊", layout="wide")

UPSTREAM = {"kittanning": "Kittanning (Allegheny)", "natrona": "Natrona (Allegheny)",
            "sutersville": "Sutersville (Youghiogheny)", "braddock": "Braddock (Mon)"}
OBSERVED = "#0b0b0b"
METRICS = {  # summary.csv column -> (label, format, higher is better)
    "rmse": ("RMSE, all hours (ft)", "{:.2f}", False),
    "mae_high_water": ("Error at high water (ft)", "{:.2f}", False),
    "mae_rising": ("Error while rising (ft)", "{:.2f}", False),
    "csi": ("Hourly alarm skill (CSI)", "{:.2f}", True),
    "events_warned_early": ("Events warned before onset", "{:.0f}", True),
    "median_lead_h": ("Median warning time (h)", "{:.0f}", True),
    "violations_pct": ("Physics violations (%)", "{:.1f}", False),
}


# ------------------------------------------------------------------------------ data
def mtime(path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


@st.cache_data
def load_tables(_stamp: float) -> dict:
    """Result tables; _stamp (summary.csv mtime) refreshes the cache after a pipeline run."""
    dates = {"events": ["start", "end", "first_alarm_issued"],
             "false_alarms": ["first_issued", "last_issued", "target_start", "target_end"],
             "worst_errors": ["issued", "target_time"]}
    out = {"summary": pd.read_csv(RESULTS_DIR / "summary.csv")}
    for name, cols in dates.items():
        path = RESULTS_DIR / f"{name}.csv"
        out[name] = pd.read_csv(path, parse_dates=cols) if path.exists() else pd.DataFrame()
    tuning = models.TUNING_FILE
    out["tuning"] = json.loads(tuning.read_text()) if tuning.exists() else None
    out["meta"] = json.loads((FEATURES_DIR / "feature_groups.json").read_text())
    return out


@st.cache_data
def load_series(_stamp: float, names: tuple, horizons: tuple):
    """Observed levels, upstream gauges, rain and every model's forecasts (by issue time)."""
    clean = pd.read_parquet(CLEAN_HOURLY)
    preds = {n: models.load(n, horizons) for n in names}
    return clean, preds


def available_models(summary: pd.DataFrame) -> list[str]:
    return [m for m in models.ALL_MODELS if m in set(summary["model"])]


NICE = {"start": "event start", "peak_ft": "peak (ft)", "lead_h": "warning (h)",
        "first_alarm_issued": "first alarm issued", "likely_cause": "likely cause",
        "target_start": "alarm aimed at", "hours": "alarm hours", "max_forecast_ft": "max forecast (ft)",
        "max_observed_ft": "max observed (ft)", "target_time": "time", "observed_ft": "observed (ft)",
        "forecast_ft": "forecast (ft)", "error_ft": "error (ft)"}


def pretty(df: pd.DataFrame) -> pd.DataFrame:
    """Readable table: UTC times as 'YYYY-MM-DD HH:MM', numbers rounded, friendly headers."""
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[c]):
            out[c] = out[c].dt.strftime("%Y-%m-%d %H:%M").fillna("—")
        elif pd.api.types.is_float_dtype(out[c]):
            out[c] = out[c].round(2)
    if "model" in out:
        out["model"] = out["model"].map(LABEL).fillna(out["model"])
    return out.rename(columns=NICE)


# ------------------------------------------------------------------------------ charts
def offline_spans(clean: pd.DataFrame, lo, hi) -> list[tuple]:
    """(start, end, gauges) for runs of hours where any upstream gauge is missing."""
    win = clean.loc[lo:hi, list(UPSTREAM)]
    off = win.isna().any(axis=1)
    spans = []
    for _, run in off[off].index.to_series().groupby((~off).cumsum()[off]):
        gauges = [g for g in UPSTREAM if win.loc[run.min():run.max(), g].isna().any()]
        spans.append((run.min(), run.max(), gauges))
    return spans


def window_chart(clean, preds, names, h, lo, hi, thr, mark=None) -> go.Figure:
    """Level (observed + forecasts at the time they predict), upstream rises, rain."""
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                        row_heights=[0.58, 0.24, 0.18],
                        subplot_titles=(f"Point level: observed vs {h} h forecasts (drawn at the time they predict)",
                                        "Upstream gauges: rise since the start of the window (ft)",
                                        "Rain, basin average (mm/h)"))
    obs = clean["point"].loc[lo:hi]
    fig.add_trace(go.Scatter(x=obs.index, y=obs, name="Observed", line=dict(color=OBSERVED, width=2.6)),
                  row=1, col=1)
    for n in names:
        s = preds[n][h].copy()
        s.index = s.index + pd.Timedelta(hours=h)            # issue time -> target time
        s = s.loc[lo:hi]
        dash = "dash" if n == models.REFERENCE else "dashdot" if n == models.TREND else None
        fig.add_trace(go.Scatter(x=s.index, y=s, name=LABEL[n],
                                 line=dict(color=COLOR[n], width=1.8, dash=dash)), row=1, col=1)
    fig.add_hline(y=thr, line=dict(color="#c0392b", dash="dot", width=1), row=1, col=1,
                  annotation_text=f"{thr:.0f} ft alarm level", annotation_position="top left")

    for g, label in UPSTREAM.items():
        s = clean[g].loc[lo:hi]
        base = s.dropna().iloc[0] if s.notna().any() else np.nan
        fig.add_trace(go.Scatter(x=s.index, y=s - base, name=label, line=dict(width=1.3),
                                 legendgroup="upstream", showlegend=True), row=2, col=1)
    rain = clean["rain_basin_mean"].loc[lo:hi]
    fig.add_trace(go.Bar(x=rain.index, y=rain, name="Rain", marker_color="#5b8fd1", showlegend=False),
                  row=3, col=1)

    for a, b, gauges in offline_spans(clean, lo, hi):
        fig.add_vrect(x0=a - pd.Timedelta(minutes=30), x1=b + pd.Timedelta(minutes=30),
                      fillcolor="#e74c3c", opacity=0.12, line_width=0, row=2, col=1,
                      annotation_text="offline: " + ", ".join(gauges), annotation_position="top left",
                      annotation_font=dict(size=10, color="#a93226"))
    if mark is not None:
        a, b = mark
        fig.add_vrect(x0=a, x1=b, fillcolor="#f1c40f", opacity=0.18, line_width=0, row=1, col=1)
    fig.update_layout(height=760, margin=dict(l=10, r=10, t=40, b=10), hovermode="x unified",
                      legend=dict(orientation="h", y=-0.12, yanchor="top"))
    fig.update_yaxes(title_text="ft", row=1, col=1)
    return fig


def offline_note(clean, lo, hi) -> None:
    spans = offline_spans(clean, lo, hi)
    if spans:
        hours = {}
        for a, b, gauges in spans:
            for g in gauges:
                hours[g] = hours.get(g, 0) + int(clean.loc[a:b, g].isna().sum())
        st.caption("Red shading in the upstream panel: a gauge was offline. "
                   + ", ".join(f"{UPSTREAM[g]} {n} h" for g, n in hours.items()))


# ------------------------------------------------------------------------------ tabs
def tab_performance(t, names, split, years, thr):
    s = t["summary"]
    main = [n for n in models.MAIN if n in names]
    n_ev = int(s["n_events"].iloc[0])
    c = st.columns(4)
    c[0].metric("Evaluation", split_text(split, years).split(",")[0])
    c[1].metric("Test water years", f"{min(years)}–{max(years)}")
    c[2].metric("Alarm level", f"{thr:.0f} ft")
    c[3].metric("Events in test years", n_ev)
    st.caption("Every hour is predicted by a model that never trained on its water year. "
               "Settings were chosen on the validation year only.")

    metric = st.selectbox("Metric", list(METRICS), format_func=lambda m: METRICS[m][0])
    label, fmt, higher = METRICS[metric]
    fig = go.Figure()
    for n in main:
        d = s[s.model == n].sort_values("horizon_h")
        fig.add_trace(go.Bar(x=[f"{h} h ahead" for h in d.horizon_h], y=d[metric], name=LABEL[n],
                             marker_color=COLOR[n], text=[fmt.format(v) for v in d[metric]],
                             textposition="outside"))
    fig.update_layout(barmode="group", height=420, yaxis_title=label,
                      title=f"{label} ({'higher' if higher else 'lower'} is better)",
                      margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(fig, use_container_width=True)

    table = (s[s.model.isin(main)].assign(model=lambda d: d.model.map(LABEL))
             .pivot(index="model", columns="horizon_h", values=list(METRICS)))
    table.columns = [f"{METRICS[m][0]} · {h} h" for m, h in table.columns]
    st.dataframe(table.round(2), use_container_width=True)

    abl = [n for n in models.ABLATIONS if n in names]
    if abl:
        st.subheader("Ablation: does adding the Elizabeth gauge help?")
        d = s[s.model.isin(["gbm_A", *abl])].assign(model=lambda d: d.model.map(LABEL))
        st.dataframe(d.pivot(index="model", columns="horizon_h", values=["rmse", "mae_high_water",
                                                                         "events_warned_early"]).round(2),
                     use_container_width=True)
    if t["tuning"]:
        with st.expander("Settings chosen on the validation year"):
            st.json(t["tuning"])


def tab_replay(t, clean, preds, names, thr):
    ev = (t["events"][["start", "end", "peak_ft"]].drop_duplicates("start")
          .sort_values("peak_ft", ascending=False).reset_index(drop=True))
    if ev.empty:
        st.info("No events in the test years.")
        return
    c1, c2 = st.columns([2, 1])
    i = c1.selectbox("Event", ev.index, format_func=lambda k: f"{ev.start[k]:%Y-%m-%d} · peak {ev.peak_ft[k]:.1f} ft")
    h = c2.radio("Horizon", t["meta"]["horizons"], horizontal=True, format_func=lambda x: f"{x} h", key="replay_h")
    shown = st.multiselect("Models", names, default=[n for n in models.MAIN if n in names],
                           format_func=LABEL.get)
    e = ev.loc[i]
    lo, hi = e.start - pd.Timedelta(days=4), e.end + pd.Timedelta(days=2)
    st.plotly_chart(window_chart(clean, preds, shown, h, lo, hi, thr, mark=(e.start, e.end)),
                    use_container_width=True)
    offline_note(clean, lo, hi)
    out = t["events"][(t["events"].start == e.start) & (t["events"].horizon_h == h)]
    out = out[out.model.isin(shown)]
    st.caption("Times are UTC. Warning = hours between the first alarm and the river reaching the alarm level.")
    st.dataframe(pretty(out[["model", "outcome", "lead_h", "first_alarm_issued", "likely_cause"]]),
                 hide_index=True, use_container_width=True)


def tab_failures(t, clean, preds, names, thr):
    c1, c2, c3 = st.columns([1.3, 1, 1.4])
    n = c1.selectbox("Model", [m for m in names if m != models.REFERENCE],
                     index=[m for m in names if m != models.REFERENCE].index("gbm_AR_mono")
                     if "gbm_AR_mono" in names else 0, format_func=LABEL.get)
    h = c2.radio("Horizon", t["meta"]["horizons"], horizontal=True, format_func=lambda x: f"{x} h", key="fail_h")
    kind = c3.radio("Failure", ["Missed or late events", "False alarms", "Largest errors"], horizontal=True)

    if kind == "Missed or late events":
        d = t["events"]
        d = d[(d.model == n) & (d.horizon_h == h) & (d.outcome != "warned early")].sort_values("peak_ft", ascending=False)
        cols = ["start", "peak_ft", "outcome", "lead_h", "likely_cause"]
        span = lambda r: (r.start - pd.Timedelta(days=3), r.end + pd.Timedelta(days=2), (r.start, r.end))
    elif kind == "False alarms":
        d = t["false_alarms"]
        d = d[(d.model == n) & (d.horizon_h == h)].sort_values("max_forecast_ft", ascending=False)
        cols = ["target_start", "hours", "max_forecast_ft", "max_observed_ft", "likely_cause"]
        span = lambda r: (r.target_start - pd.Timedelta(days=3), r.target_end + pd.Timedelta(days=2),
                          (r.target_start, r.target_end))
    else:
        d = t["worst_errors"]
        d = d[(d.model == n) & (d.horizon_h == h)]
        cols = ["target_time", "observed_ft", "forecast_ft", "error_ft", "likely_cause"]
        span = lambda r: (r.target_time - pd.Timedelta(days=3), r.target_time + pd.Timedelta(days=2),
                          (r.target_time - pd.Timedelta(hours=1), r.target_time + pd.Timedelta(hours=1)))

    if d.empty:
        st.success(f"No {kind.lower()} for {LABEL[n]} at {h} h in the test years.")
        return
    d = d.reset_index(drop=True)
    st.caption(f"{len(d)} found (times UTC). Likely causes come from simple rules on the data "
               "(see failures.py), not proof.")
    st.dataframe(pretty(d[cols]), hide_index=True, use_container_width=True, height=230)
    k = st.selectbox("Show on the chart", d.index,
                     format_func=lambda j: f"{d.loc[j, cols[0]]:%Y-%m-%d %H:%M} · {d.loc[j, 'likely_cause']}")
    lo, hi, mark = span(d.loc[k])
    shown = [m for m in (models.REFERENCE, n) if m in names]
    st.plotly_chart(window_chart(clean, preds, shown, h, lo, hi, thr, mark=mark), use_container_width=True)
    offline_note(clean, lo, hi)


# ------------------------------------------------------------------------------ app
def main() -> None:
    summary_path = RESULTS_DIR / "summary.csv"
    if not summary_path.exists():
        st.error("No results yet. Run `uv run python -m systems_final_project.pipeline` first.")
        st.stop()
    t = load_tables(mtime(summary_path))
    names = available_models(t["summary"])
    horizons = tuple(t["meta"]["horizons"])
    clean, preds = load_series(mtime(summary_path), tuple(names), horizons)
    split = t["summary"]["split"].iloc[0]
    years = models.split_info(names[0])["test_years"]
    thr = t["meta"]["flood_ft"]

    st.title("Water level at the Point, Pittsburgh")
    st.caption(f"Forecasts 6, 12 and 24 h ahead · {split_text(split, years)} · alarm at {thr:.0f} ft")
    perf, replay, fails = st.tabs(["Model performance", "Replay an event", "Where the model fails"])
    with perf:
        tab_performance(t, names, split, years, thr)
    with replay:
        tab_replay(t, clean, preds, names, thr)
    with fails:
        tab_failures(t, clean, preds, names, thr)


main()
