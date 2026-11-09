"""
Fixes the root cause of the 0-window result from AIS-ACNet's own windowing code
(external/AIS-ACNet/AIS-ACNet/data.py::trajectory_dataset._condition_vessels).

That filter only counts a vessel as "valid" for a window if it has a reading at
EVERY ONE of N consecutive timestamps shared across ALL vessels in the area -- not
N consecutive readings of its own. With real, asynchronous per-vessel AIS reporting
(each vessel reports every ~2-10s, independently of every other vessel), almost no
vessel will ever satisfy that by chance, so the windowing loop finds ~0 samples no
matter how much data you give it.

Fix: snap every vessel onto the SAME regular 1-minute grid before handing data to
their loader, so "15 consecutive shared timestamps" and "a vessel's own 15
consecutive readings" become the same thing (this also satisfies their separate
_condition_time check, which requires gaps between consecutive timestamps of at
most 1 minute). This mirrors the same kind of fixed-grid resampling our own
maritime data_pipeline (MARITIME-SPOOFING/data_pipeline/build_voyages.py) already
does, just at 1-minute instead of 5-minute resolution, since AIS-ACNet's own gap
tolerance requires the finer grid.

- LAT/LON/SOG: linearly interpolated onto the grid, only *inside* each vessel's own
  observed time span (never extrapolated before its first or after its last report).
- COG: nearest-neighbour hold, not interpolated -- COG is a circular quantity
  (0 deg and 359 deg are adjacent), so linear interpolation across the wrap would be
  wrong. Same convention our own pipeline already uses for heading/cog/status.

Usage:
    python resample_to_grid.py --csv houston_2025-04-19_smoketest.csv \
        --out houston_2025-04-19_smoketest_1min.csv --freq 1min
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def resample_vessel(group: pd.DataFrame, freq: str) -> pd.DataFrame:
    group = group.drop_duplicates(subset="BaseDateTime").set_index("BaseDateTime").sort_index()
    if len(group) < 2:
        return group.iloc[0:0]

    grid = pd.date_range(group.index.min().floor(freq), group.index.max().ceil(freq), freq=freq)
    reindexed = group.reindex(group.index.union(grid)).sort_index()

    reindexed["LAT"] = reindexed["LAT"].interpolate(method="time", limit_area="inside")
    reindexed["LON"] = reindexed["LON"].interpolate(method="time", limit_area="inside")
    reindexed["SOG"] = reindexed["SOG"].interpolate(method="time", limit_area="inside")
    reindexed["COG"] = reindexed["COG"].ffill()
    reindexed["MMSI"] = reindexed["MMSI"].ffill().bfill()

    out = reindexed.reindex(grid).dropna(subset=["LAT", "LON", "SOG", "COG"])
    out = out.reset_index().rename(columns={"index": "BaseDateTime"})
    return out


def resample_dataframe(df: pd.DataFrame, freq: str = "1min") -> pd.DataFrame:
    """Snap every vessel in df onto the same regular time grid. See module docstring."""
    out_frames = []
    for mmsi, group in df.groupby("MMSI", sort=False):
        resampled = resample_vessel(group[["BaseDateTime", "MMSI", "LAT", "LON", "SOG", "COG"]], freq)
        if not resampled.empty:
            out_frames.append(resampled)
    if not out_frames:
        return df.iloc[0:0]
    out = pd.concat(out_frames, ignore_index=True)
    return out.sort_values("BaseDateTime")[["BaseDateTime", "MMSI", "LAT", "LON", "SOG", "COG"]]


def resample_file(csv_path, out_path, freq: str = "1min") -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["BaseDateTime"])
    out = resample_dataframe(df, freq)
    out.to_csv(out_path, index=False)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--freq", default="1min", help="pandas offset alias, e.g. 1min")
    args = ap.parse_args()

    df = pd.read_csv(HERE / args.csv, parse_dates=["BaseDateTime"])
    print(f"input: {len(df):,} rows, {df.MMSI.nunique():,} vessels")

    out = resample_dataframe(df, args.freq)
    out_path = HERE / args.out
    out.to_csv(out_path, index=False)
    print(f"output: {len(out):,} rows, {out.MMSI.nunique():,} vessels, "
          f"{out.BaseDateTime.nunique():,} unique timestamps -> {out_path}")

    gaps = out.BaseDateTime.drop_duplicates().sort_values().diff().dropna()
    if len(gaps):
        print(f"max gap between consecutive unique timestamps: {gaps.max()}")


if __name__ == "__main__":
    main()
