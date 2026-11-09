"""
Resumable, per-day replacement for build_multiday_dataset.py's windowing step.

Builds AIS-ACNet's windowed trajectory_dataset for ONE day at a time and caches it
to day_windows/YYYY-MM-DD.pt. Re-running with an overlapping --start/--end skips any
day whose cache file already exists, so an interrupted run (machine sleep, lost
power, etc.) only needs to redo whatever didn't finish -- nothing is lost.

Run this in chunks (e.g. a few days per invocation) rather than all 44 days in one
call, so progress can be checked and reported incrementally instead of waiting on a
single multi-hour call with no visibility until the very end.

Usage:
    python build_day_windows.py --start 2025-04-19 --end 2025-04-21
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
EXTERNAL_REPO = HERE.parents[1] / "external" / "AIS-ACNet" / "AIS-ACNet"
sys.path.insert(0, str(EXTERNAL_REPO))
sys.path.insert(0, str(HERE))

import data as ais_acnet_data  # noqa: E402  (external repo module, imported unmodified)
from resample_to_grid import resample_file  # noqa: E402

# Upstream bug workaround (see build_dataset.py / build_multiday_dataset.py for details):
# get_sequence() returns a 4-tuple but its own caller only unpacks 3.
_orig_get_sequence = ais_acnet_data.trajectory_dataset.get_sequence


def _get_sequence_3tuple(self, frame):
    return _orig_get_sequence(self, frame)[:3]


ais_acnet_data.trajectory_dataset.get_sequence = _get_sequence_3tuple

RAW_BY_DAY = HERE / "raw_by_day"
RESAMPLED_BY_DAY = HERE / "resampled_by_day"
DAY_WINDOWS = HERE / "day_windows"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--sequence_length", type=int, default=15)
    ap.add_argument("--prediction_length", type=int, default=15)
    ap.add_argument("--feature_size", type=int, default=4)
    args = ap.parse_args()

    RESAMPLED_BY_DAY.mkdir(exist_ok=True)
    DAY_WINDOWS.mkdir(exist_ok=True)

    days = pd.date_range(args.start, args.end, freq="D")
    for day in days:
        day_str = day.strftime("%Y-%m-%d")
        out_path = DAY_WINDOWS / f"{day_str}.pt"
        if out_path.exists():
            print(f"[skip, cached] {day_str}")
            continue

        raw_path = RAW_BY_DAY / f"{day_str}.csv"
        if not raw_path.exists():
            print(f"[skip, no raw data] {day_str}")
            continue

        t0 = time.time()
        resampled_path = RESAMPLED_BY_DAY / f"{day_str}.csv"
        if not resampled_path.exists():
            resample_file(raw_path, resampled_path, freq="1min")

        ds = ais_acnet_data.trajectory_dataset(
            str(resampled_path), args.sequence_length, args.prediction_length, args.feature_size
        )
        torch.save(ds, out_path)
        print(f"[done] {day_str}: {len(ds)} windows ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
