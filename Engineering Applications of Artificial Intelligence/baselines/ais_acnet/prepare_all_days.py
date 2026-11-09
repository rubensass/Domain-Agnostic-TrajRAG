"""
Single-pass extraction of every calendar day in MARITIME-SPOOFING/data/HOU-SHIPCHANNEL/ais_raw.csv
(bbox-filtered, AIS-ACNet's CSV schema) into one file per day under raw_by_day/.

Avoids re-scanning the 2.1GB raw file once per day (44x I/O waste) -- a single streamed pass
buckets rows by date and writes each day out incrementally.

Usage:
    python prepare_all_days.py --start 2025-04-19 --end 2025-06-01
"""
import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd

RAW_CSV = (
    Path(__file__).resolve().parents[3]
    / "MARITIME-SPOOFING" / "data" / "HOU-SHIPCHANNEL" / "ais_raw.csv"
)

MIN_LAT, MAX_LAT, MIN_LON, MAX_LON = 28.8, 29.8, -95.25, -94.25
RAW_COLS = ["mmsi", "base_date_time", "longitude", "latitude", "sog", "cog"]
RENAME = {
    "mmsi": "MMSI", "base_date_time": "BaseDateTime",
    "longitude": "LON", "latitude": "LAT", "sog": "SOG", "cog": "COG",
}
OUT_DIR = Path(__file__).resolve().parent / "raw_by_day"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True, help="YYYY-MM-DD, inclusive")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD, inclusive")
    ap.add_argument("--chunksize", type=int, default=2_000_000)
    args = ap.parse_args()

    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)
    OUT_DIR.mkdir(exist_ok=True)

    day_writers = {}  # date -> output file handle (append mode), header written once
    total_rows = 0
    kept_rows = 0

    for chunk in pd.read_csv(RAW_CSV, usecols=RAW_COLS, parse_dates=["base_date_time"],
                             chunksize=args.chunksize):
        total_rows += len(chunk)
        chunk = chunk[(chunk.base_date_time >= start) & (chunk.base_date_time < end + pd.Timedelta(days=1))]
        chunk = chunk[(chunk.latitude >= MIN_LAT) & (chunk.latitude <= MAX_LAT)
                       & (chunk.longitude >= MIN_LON) & (chunk.longitude <= MAX_LON)]
        chunk = chunk.dropna(subset=["sog", "cog"])
        if chunk.empty:
            print(f"scanned {total_rows:,} rows, kept {kept_rows:,}", end="\r")
            continue

        chunk = chunk.rename(columns=RENAME)[["BaseDateTime", "MMSI", "LAT", "LON", "SOG", "COG"]]
        chunk["_date"] = chunk.BaseDateTime.dt.date

        for date, sub in chunk.groupby("_date", sort=False):
            sub = sub.drop(columns="_date")
            out_path = OUT_DIR / f"{date}.csv"
            write_header = date not in day_writers
            sub.to_csv(out_path, mode="a", header=write_header, index=False)
            day_writers[date] = True
            kept_rows += len(sub)

        print(f"scanned {total_rows:,} rows, kept {kept_rows:,}", end="\r")

    print()
    print(f"wrote {len(day_writers)} day files to {OUT_DIR}")
    for date in sorted(day_writers):
        print(" ", date)


if __name__ == "__main__":
    main()
