"""
Builds AIS-ACNet's windowed train/val/test tensors across many days, keeping per-day
cost bounded (their windowing loop re-scans the whole frame on every timestamp, so
feeding N days as one file costs ~N^2, not ~N -- see conversation notes). Each day is
resampled onto a 1-minute grid (resample_to_grid.resample_file) and windowed
separately via their own trajectory_dataset, then all days' windows are concatenated
IN CHRONOLOGICAL ORDER and split 70% train / 10% val / 20% test by that chronological
order (earliest days -> train, latest days -> test, no shuffling) -- matching the
no-leakage, train-on-earlier/test-on-later-dates approach already used for the
aviation chronological split, and the same ratio order AIS-ACNet's own load_data()
uses internally (just applied globally across days instead of within one file).

Usage:
    python build_multiday_dataset.py --start 2025-04-19 --end 2025-06-01 --out processed_full
"""
import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import ConcatDataset, Subset

HERE = Path(__file__).resolve().parent
EXTERNAL_REPO = HERE.parents[1] / "external" / "AIS-ACNet" / "AIS-ACNet"
sys.path.insert(0, str(EXTERNAL_REPO))
sys.path.insert(0, str(HERE))

import data as ais_acnet_data  # noqa: E402  (external repo module, imported unmodified)
from resample_to_grid import resample_file  # noqa: E402

# Same upstream-bug workaround as build_dataset.py (get_sequence returns a 4-tuple,
# its own caller only unpacks 3).
_orig_get_sequence = ais_acnet_data.trajectory_dataset.get_sequence


def _get_sequence_3tuple(self, frame):
    return _orig_get_sequence(self, frame)[:3]


ais_acnet_data.trajectory_dataset.get_sequence = _get_sequence_3tuple

RAW_BY_DAY = HERE / "raw_by_day"
RESAMPLED_BY_DAY = HERE / "resampled_by_day"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sequence_length", type=int, default=15)
    ap.add_argument("--prediction_length", type=int, default=15)
    ap.add_argument("--feature_size", type=int, default=4)
    ap.add_argument("--train_frac", type=float, default=0.70)
    ap.add_argument("--val_frac", type=float, default=0.10)
    args = ap.parse_args()

    RESAMPLED_BY_DAY.mkdir(exist_ok=True)

    days = pd.date_range(args.start, args.end, freq="D")
    day_datasets = []
    day_sizes = []

    for day in days:
        day_str = day.strftime("%Y-%m-%d")
        raw_path = RAW_BY_DAY / f"{day_str}.csv"
        if not raw_path.exists():
            print(f"[skip] no raw data for {day_str}")
            continue

        resampled_path = RESAMPLED_BY_DAY / f"{day_str}.csv"
        if not resampled_path.exists():
            print(f"[resample] {day_str}")
            resample_file(raw_path, resampled_path, freq="1min")

        print(f"[window] {day_str}")
        ds = ais_acnet_data.trajectory_dataset(
            str(resampled_path), args.sequence_length, args.prediction_length, args.feature_size
        )
        if len(ds) == 0:
            print(f"  -> 0 windows, skipping")
            continue
        day_datasets.append(ds)
        day_sizes.append(len(ds))
        print(f"  -> {len(ds)} windows (running total: {sum(day_sizes)})")

    if not day_datasets:
        raise SystemExit("No windows produced for any day in range.")

    full = ConcatDataset(day_datasets)
    n = len(full)
    n_train = int(n * args.train_frac)
    n_val = int(n * args.val_frac)
    n_test = n - n_train - n_val

    train_ds = Subset(full, range(0, n_train))
    val_ds = Subset(full, range(n_train, n_train + n_val))
    test_ds = Subset(full, range(n_train + n_val, n))

    out_dir = HERE / args.out / "15"
    for split in ("train", "val", "test"):
        os.makedirs(out_dir / split, exist_ok=True)

    torch.save(train_ds, out_dir / "train" / f"{args.sequence_length:02d}_{args.prediction_length:02d}.pt")
    torch.save(val_ds, out_dir / "val" / f"{args.sequence_length:02d}_{args.prediction_length:02d}.pt")
    torch.save(test_ds, out_dir / "test" / f"{args.sequence_length:02d}_{args.prediction_length:02d}.pt")

    print()
    print(f"days used: {len(day_datasets)}, total windows: {n}")
    print(f"train: {len(train_ds)} ({args.train_frac:.0%}), "
          f"val: {len(val_ds)} ({args.val_frac:.0%}), "
          f"test: {len(test_ds)} ({1 - args.train_frac - args.val_frac:.0%})")
    print(f"saved to {out_dir}")


if __name__ == "__main__":
    main()
