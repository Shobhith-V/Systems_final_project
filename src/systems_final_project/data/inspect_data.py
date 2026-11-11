"""Print summary statistics for all downloaded raw data.

Covers, per dataset: files, disk size, rows, columns/dtypes, date range,
sampling interval, missing values, gaps, value distribution, per-year
completeness and medians (to spot shifts like Braddock's), and flood counts.

Usage:
    uv run python -m systems_final_project.data.inspect_data
    uv run python -m systems_final_project.data.inspect_data > data_summary.txt
"""

from pathlib import Path

import pandas as pd

from .config import RAIN_DIR, RAIN_FCST_DIR, USGS_DIR

FLOOD_FT = 22.0
GAP_REPORT = pd.Timedelta(hours=6)  # report gaps longer than this
pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 20)


def header(title: str) -> None:
    print(f"\n{'=' * 80}\n{title}\n{'=' * 80}")


def disk_size(files: list[Path]) -> str:
    return f"{sum(f.stat().st_size for f in files) / 1e6:.2f} MB"


def basic_info(df: pd.DataFrame, files: list[Path]) -> None:
    print(f"files: {len(files)}   disk: {disk_size(files)}   rows: {len(df):,}")
    print(f"memory: {df.memory_usage(deep=True).sum() / 1e6:.2f} MB")
    print("columns:")
    for c in df.columns:
        n_null = df[c].isna().sum()
        print(f"  {c:<22} {str(df[c].dtype):<22} nulls={n_null:,} ({n_null / max(len(df), 1):.1%})")
    if len(df):
        print(f"range: {df['time'].min()}  ->  {df['time'].max()}")


def sampling(df: pd.DataFrame) -> pd.Timedelta:
    dt = df["time"].diff().dropna()
    if dt.empty:
        return pd.Timedelta(0)
    step = dt.median()
    top = dt.value_counts().head(3)
    print(f"sampling: median step {step}; most common steps: "
          + ", ".join(f"{k} ({v:,})" for k, v in top.items()))
    return step


def gaps(df: pd.DataFrame, col: str) -> None:
    valid = df.loc[df[col].notna(), "time"]
    dt = valid.diff()
    big = dt[dt > GAP_REPORT]
    if big.empty:
        print(f"gaps > {GAP_REPORT}: none")
        return
    total = big.sum()
    print(f"gaps > {GAP_REPORT}: {len(big)} gaps, total {total}, longest 5:")
    for idx in big.sort_values(ascending=False).head(5).index:
        print(f"    {valid[idx] - dt[idx]}  ->  {valid[idx]}   ({dt[idx]})")


def distribution(s: pd.Series) -> None:
    s = s.dropna()
    if s.empty:
        print("distribution: (no valid values)")
        return
    q = s.quantile([0, 0.01, 0.25, 0.5, 0.75, 0.99, 1])
    print("distribution: " + "  ".join(f"p{int(k * 100)}={v:.2f}" for k, v in q.items())
          + f"  mean={s.mean():.2f}  std={s.std():.2f}")


def per_year(df: pd.DataFrame, col: str, step: pd.Timedelta, extra: dict | None = None) -> None:
    """Completeness = valid readings / expected readings for that calendar year."""
    d = df.assign(year=df["time"].dt.year)
    g = d.groupby("year")[col]
    out = pd.DataFrame({"n_valid": g.count(), "median": g.median(), "max": g.max()})
    if step > pd.Timedelta(0):
        span = d.groupby("year")["time"].agg(["min", "max"])
        expected = ((span["max"] - span["min"]) / step).astype(int) + 1
        out["complete_%"] = (100 * out["n_valid"] / expected).round(1)
    for name, fn in (extra or {}).items():
        out[name] = g.apply(fn)
    print("per year:")
    print(out.round(2).to_string())


def load_dir(path: Path) -> tuple[pd.DataFrame, list[Path]]:
    files = sorted(path.glob("*.parquet"))
    if not files:
        return pd.DataFrame(), []
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return df.sort_values("time").reset_index(drop=True), files


def flood_events(df: pd.DataFrame, col: str) -> None:
    """Count distinct episodes above FLOOD_FT (separate if >24 h apart)."""
    above = df.loc[df[col] >= FLOOD_FT, ["time", col]]
    if above.empty:
        print(f"flood (>= {FLOOD_FT} ft): 0 readings")
        return
    new_event = above["time"].diff() > pd.Timedelta(hours=24)
    above = above.assign(event=new_event.cumsum())
    ev = above.groupby("event").agg(start=("time", "min"), end=("time", "max"), peak=(col, "max"))
    print(f"flood (>= {FLOOD_FT} ft): {len(above):,} readings in {len(ev)} events")
    print(ev.to_string(index=False))


def inspect_usgs() -> None:
    header("USGS GAUGES")
    if not USGS_DIR.exists():
        print(f"missing: {USGS_DIR}")
        return
    summary = []
    for gdir in sorted(p for p in USGS_DIR.iterdir() if p.is_dir()):
        df, files = load_dir(gdir)
        print(f"\n--- {gdir.name} " + "-" * (74 - len(gdir.name)))
        if df.empty:
            print("no data")
            continue
        basic_info(df, files)
        step = sampling(df)
        distribution(df["value"])
        gaps(df, "value")
        if "approval_status" in df:
            print("approval_status: " + ", ".join(
                f"{k}={v:,}" for k, v in df["approval_status"].value_counts(dropna=False).items()))
        if "qualifier" in df:
            q = df["qualifier"].value_counts(dropna=False).head(5)
            print("qualifier (top 5): " + ", ".join(f"{k}={v:,}" for k, v in q.items()))
        neg = (df["value"] < 0).sum()
        if neg:
            print(f"WARNING: {neg:,} negative values")
        per_year(df, "value", step)
        if gdir.name == "point":
            flood_events(df, "value")
        summary.append({
            "gauge": gdir.name, "files": len(files), "rows": len(df),
            "start": df["time"].min().date(), "end": df["time"].max().date(),
            "missing_%": round(100 * df["value"].isna().mean(), 2),
            "min": df["value"].min(), "median": df["value"].median(), "max": df["value"].max(),
        })
    if summary:
        print("\nUSGS overview:")
        print(pd.DataFrame(summary).to_string(index=False))


def inspect_rain(path: Path, label: str) -> None:
    header(label)
    files = sorted(path.glob("*.parquet")) if path.exists() else []
    if not files:
        print(f"missing: {path}")
        return
    summary = []
    for f in files:
        df = pd.read_parquet(f).sort_values("time").reset_index(drop=True)
        col = next(c for c in df.columns if c != "time")
        print(f"\n--- {f.stem} " + "-" * (74 - len(f.stem)))
        basic_info(df, [f])
        step = sampling(df)
        first_valid = df.loc[df[col].notna(), "time"].min()
        print(f"first non-null: {first_valid}")
        s = df[col].dropna()
        wet = (s > 0.1).mean() if len(s) else float("nan")
        print(f"{col}: total={s.sum():.0f} mm  max hourly={s.max():.1f} mm  wet hours(>0.1mm)={wet:.1%}")
        distribution(s[s > 0.1])  # distribution of wet hours only
        gaps(df, col)
        per_year(df, col, step, extra={"total_mm": "sum"})
        summary.append({
            "location": f.stem, "rows": len(df), "first_valid": first_valid,
            "end": df["time"].max(), "missing_%": round(100 * df[col].isna().mean(), 2),
            "total_mm": round(s.sum()), "max_hourly": s.max(),
        })
    print(f"\n{label} overview:")
    print(pd.DataFrame(summary).to_string(index=False))


def main() -> None:
    inspect_usgs()
    inspect_rain(RAIN_DIR, "OBSERVED RAIN (Open-Meteo archive)")
    inspect_rain(RAIN_FCST_DIR, "FORECAST RAIN (Open-Meteo previous runs, ~24h ahead)")


if __name__ == "__main__":
    main()