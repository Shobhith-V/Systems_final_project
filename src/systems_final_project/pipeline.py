"""Run the whole project end to end, or any part of it.

Steps, in order (each is also runnable on its own as `python -m <module>`):
  download   data.download_usgs + data.download_rain   raw data -> data/raw/   (opt-in)
  clean      data.clean                                 -> data/clean/hourly.parquet
  features   features                                   -> data/features/
  tune       tune     (settings chosen on WY2018)       -> results/tuning.json
  models     models                                     -> results/predictions/
  evaluate   evaluation                                 -> results/summary.csv, results/figures/

Usage (from the repository root):
    uv run python -m systems_final_project.pipeline                  # clean -> evaluate
    uv run python -m systems_final_project.pipeline --download       # also re-download raw data
    uv run python -m systems_final_project.pipeline --from models    # start at a later step
    uv run python -m systems_final_project.pipeline --only evaluate  # a single step
    uv run python -m systems_final_project.pipeline --split loyo     # leave-one-year-out instead

Downloading needs USGS_API_KEY in the environment (see data/download_usgs.py).
Models are retrained automatically when the feature table or tuning file is newer
than their predictions, so rerunning from `clean` never evaluates stale predictions.
"""

import argparse
import subprocess
import sys
import time

PKG = "systems_final_project"
STEPS = {
    "download": [[f"{PKG}.data.download_usgs"], [f"{PKG}.data.download_rain"]],
    "clean": [[f"{PKG}.data.clean"]],
    "features": [[f"{PKG}.features"]],
    "tune": [[f"{PKG}.tune"]],
    "models": [[f"{PKG}.models"]],
    "evaluate": [[f"{PKG}.evaluation"]],
}
ORDER = list(STEPS)


def run_module(args: list[str]) -> None:
    cmd = [sys.executable, "-m", *args]
    print(f"\n$ {' '.join(cmd[1:])}", flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--download", action="store_true", help="also re-download the raw data first")
    ap.add_argument("--from", dest="start", choices=ORDER, help="first step to run")
    ap.add_argument("--only", choices=ORDER, help="run just this step")
    ap.add_argument("--split", choices=["walk_forward", "loyo"], help="passed to the models step")
    ap.add_argument("--force", action="store_true", help="retrain models even if predictions are current")
    args = ap.parse_args()

    if args.only:
        steps = [args.only]
    else:
        start = args.start or ("download" if args.download else "clean")
        steps = ORDER[ORDER.index(start):]

    t0 = time.time()
    for step in steps:
        print(f"\n=== {step} " + "=" * (60 - len(step)), flush=True)
        for module_args in STEPS[step]:
            extra = []
            if step == "models":
                extra += ["--split", args.split] if args.split else []
                extra += ["--force"] if args.force else []
            run_module(module_args + extra)
    print(f"\npipeline finished in {time.time() - t0:.0f}s: {', '.join(steps)}")


if __name__ == "__main__":
    main()
