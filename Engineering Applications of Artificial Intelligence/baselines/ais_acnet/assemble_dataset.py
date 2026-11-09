"""
Final, fast step: loads every cached per-day windowed dataset from day_windows/
(built by build_day_windows.py) in chronological order, concatenates them, and
applies a chronological 70% train / 10% val / 20% test split (earliest days ->
train, latest days -> test, no shuffling -- same no-leakage approach as the
aviation chronological split, and the same ratio AIS-ACNet's own load_data() uses
internally, just applied globally across days).

Usage:
    python assemble_dataset.py --start 2025-04-19 --end 2025-06-01 --out processed_full
"""
import argparse
import os
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import ConcatDataset, Subset

HERE = Path(__file__).resolve().parent
DAY_WINDOWS = HERE / "day_windows"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sequence_length", type=int, default=15)
    ap.add_argument("--prediction_length", type=int, default=15)
    ap.add_argument("--train_frac", type=float, default=0.70)
    ap.add_argument("--val_frac", type=float, default=0.10)
    args = ap.parse_args()

    days = pd.date_range(args.start, args.end, freq="D")
    day_datasets = []
    missing = []
    for day in days:
        day_str = day.strftime("%Y-%m-%d")
        path = DAY_WINDOWS / f"{day_str}.pt"
        if not path.exists():
            missing.append(day_str)
            continue
        ds = torch.load(path, weights_only=False)
        if len(ds) > 0:
            day_datasets.append(ds)

    if missing:
        print(f"WARNING: {len(missing)} day(s) not yet built, excluded: {missing}")

    if not day_datasets:
        raise SystemExit("No cached day windows found -- run build_day_windows.py first.")

    full = ConcatDataset(day_datasets)
    n = len(full)
    n_train = int(n * args.train_frac)
    n_val = int(n * args.val_frac)

    train_ds = Subset(full, range(0, n_train))
    val_ds = Subset(full, range(n_train, n_train + n_val))
    test_ds = Subset(full, range(n_train + n_val, n))

    out_dir = HERE / args.out / "15"
    for split in ("train", "val", "test"):
        os.makedirs(out_dir / split, exist_ok=True)

    fname = f"{args.sequence_length:02d}_{args.prediction_length:02d}.pt"
    torch.save(train_ds, out_dir / "train" / fname)
    torch.save(val_ds, out_dir / "val" / fname)
    torch.save(test_ds, out_dir / "test" / fname)

    print(f"days used: {len(day_datasets)} (of {len(days)} in range), total windows: {n}")
    print(f"train: {len(train_ds)} ({args.train_frac:.0%}), "
          f"val: {len(val_ds)} ({args.val_frac:.0%}), "
          f"test: {len(test_ds)} ({1 - args.train_frac - args.val_frac:.0%})")
    print(f"saved to {out_dir}")


if __name__ == "__main__":
    main()
