"""
Converts our own raw Houston Ship Channel AIS data (MARITIME-SPOOFING/data/HOU-SHIPCHANNEL/ais_raw.csv)
into the exact CSV schema AIS-ACNet's data.py::trajectory_dataset expects:
    BaseDateTime, MMSI, LAT, LON, SOG, COG

AIS-ACNet's own loader (external/AIS-ACNet/AIS-ACNet/data.py) discards any sample window where the
max gap between consecutive unique timestamps exceeds 1 minute, so we keep our data at native AIS
reporting resolution (no resampling) -- unlike MARITIME-SPOOFING's own voyage pipeline, which
resamples to a 5-minute grid and would fail that check.

Usage:
    python prepare_data.py --start 2025-04-19 --end 2025-04-19 --out houston_2025-04-19.csv
"""
import argparse
from pathlib import Path

import pandas as pd

RAW_CSV = (
    Path(__file__).resolve().parents[3]
    / "MARITIME-SPOOFING" / "data" / "HOU-SHIPCHANNEL" / "ais_raw.csv"
)

# Must match external/AIS-ACNet/AIS-ACNet/geographic_utils.py's hardcoded Houston bounding box
MIN_LAT, MAX_LAT, MIN_LON, MAX_LON = 28.8, 29.8, -95.25, -94.25

RAW_COLS = ["mmsi", "base_date_time", "longitude", "latitude", "sog", "cog"]
RENAME = {
    "mmsi": "MMSI",
    "base_date_time": "BaseDateTime",
    "longitude": "LON",
    "latitude": "LAT",
    "sog": "SOG",
    "cog": "COG",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True, help="YYYY-MM-DD, inclusive")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD, inclusive")
    ap.add_argument("--out", required=True, help="output CSV filename (written next to this script)")
    ap.add_argument("--chunksize", type=int, default=2_000_000)
    args = ap.parse_args()

    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end) + pd.Timedelta(days=1)  # exclusive upper bound
    out_path = Path(__file__).resolve().parent / args.out

    kept_chunks = []
    total_rows = 0
    kept_rows = 0

    for chunk in pd.read_csv(
        RAW_CSV,
        usecols=RAW_COLS,
        parse_dates=["base_date_time"],
        chunksize=args.chunksize,
    ):
        total_rows += len(chunk)
        chunk = chunk[(chunk.base_date_time >= start) & (chunk.base_date_time < end)]
        chunk = chunk[
            (chunk.latitude >= MIN_LAT) & (chunk.latitude <= MAX_LAT)
            & (chunk.longitude >= MIN_LON) & (chunk.longitude <= MAX_LON)
        ]
        chunk = chunk.dropna(subset=["sog", "cog"])
        if not chunk.empty:
            kept_chunks.append(chunk)
            kept_rows += len(chunk)
        print(f"scanned {total_rows:,} rows, kept {kept_rows:,} so far", end="\r")

    print()
    if not kept_chunks:
        raise SystemExit("No rows matched the given date range / bounding box.")

    df = pd.concat(kept_chunks, ignore_index=True)
    df = df.rename(columns=RENAME)[["BaseDateTime", "MMSI", "LAT", "LON", "SOG", "COG"]]
    df = df.sort_values("BaseDateTime")
    df.to_csv(out_path, index=False)
    print(f"wrote {len(df):,} rows to {out_path}")
    print(f"date range in output: {df.BaseDateTime.min()} .. {df.BaseDateTime.max()}")
    print(f"unique vessels (MMSI): {df.MMSI.nunique()}")


if __name__ == "__main__":
    main()
