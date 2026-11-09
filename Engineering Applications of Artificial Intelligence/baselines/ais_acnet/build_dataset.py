"""
Builds AIS-ACNet's windowed train/val/test .pt tensors from one of our prepared CSVs
(see prepare_data.py), by calling the external repo's own data.trajectory_dataset /
data.load_data unmodified.

external/AIS-ACNet/AIS-ACNet/data.py::load_data() hardcodes the "15" subfolder name
regardless of --sequence_length, so we keep sequence_length = prediction_length = 15,
matching the paper's own default setup -- also the right call for a faithful baseline
comparison (no reason to deviate from their validated hyperparameters).

Usage:
    python build_dataset.py --csv houston_2025-04-19_smoketest.csv --out processed_smoketest
"""
import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
EXTERNAL_REPO = HERE.parents[1] / "external" / "AIS-ACNet" / "AIS-ACNet"
sys.path.insert(0, str(EXTERNAL_REPO))

import data as ais_acnet_data  # noqa: E402  (external repo module, imported unmodified)

# Upstream bug workaround (not a modification of their file): trajectory_dataset
# .get_sequence() returns a 4-tuple (sequence, mask, vessel_count, vessels) but its
# own caller, get_file_samples(), only unpacks 3 values -- ValueError otherwise.
_orig_get_sequence = ais_acnet_data.trajectory_dataset.get_sequence


def _get_sequence_3tuple(self, frame):
    return _orig_get_sequence(self, frame)[:3]


ais_acnet_data.trajectory_dataset.get_sequence = _get_sequence_3tuple


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="CSV produced by prepare_data.py (in this folder)")
    ap.add_argument("--out", required=True, help="output processed-data folder (created under this script's dir)")
    ap.add_argument("--sequence_length", type=int, default=15)
    ap.add_argument("--prediction_length", type=int, default=15)
    ap.add_argument("--feature_size", type=int, default=4)
    args = ap.parse_args()

    data_dir = str(HERE / args.out) + os.sep
    for split in ("train", "val", "test"):
        os.makedirs(os.path.join(data_dir, "15", split), exist_ok=True)

    # data.load_data() builds the CSV path as `data_dir + filename` (plain string
    # concatenation, see data.py:227), so `filename` must be relative to data_dir.
    csv_abspath = str(HERE / args.csv)
    filename_rel = os.path.relpath(csv_abspath, start=data_dir)

    loader_args = SimpleNamespace(
        data_dir=data_dir,
        filename=filename_rel,
        split_data=True,
        feature_size=args.feature_size,
        sequence_length=args.sequence_length,
        prediction_length=args.prediction_length,
    )

    train_ds, val_ds, test_ds = ais_acnet_data.load_data(loader_args)
    print(f"train: {len(train_ds)} windows, val: {len(val_ds)} windows, test: {len(test_ds)} windows")


if __name__ == "__main__":
    main()
