"""
Streams NOAA Marine Cadastre daily AIS files (nationwide) and filters rows down
to two corridor bounding boxes, without ever writing the full nationwide CSV to
disk. Output: one growing CSV per corridor under
MARITIME-SPOOFING/data/<CORRIDOR>/ais_raw.csv

Source: https://coast.noaa.gov/htdata/CMSP/AISDataHandler/{year}/ais-{year}-{mm}-{dd}.csv.zst
Schema (lowercase, confirmed by sampling): mmsi, base_date_time, longitude,
latitude, sog, cog, heading, vessel_name, imo, call_sign, vessel_type, status,
length, width, draft, cargo, transceiver
"""

import csv
import io
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import requests
import zstandard

DATA_ROOT = Path(__file__).parent.parent / "data"

CORRIDORS = {
    "HOU-SHIPCHANNEL": {
        "label": "Galveston Bay entrance -> Port of Houston Turning Basin",
        "lat_min": 29.20, "lat_max": 29.85,
        "lon_min": -95.35, "lon_max": -94.65,
    },
    "LA-LONGBEACH": {
        "label": "San Pedro Bay approach -> Port of LA / Long Beach",
        "lat_min": 33.60, "lat_max": 33.80,
        "lon_min": -118.35, "lon_max": -118.05,
    },
}

START_DATE = date(2025, 4, 19)
END_DATE = date(2025, 6, 1)  # inclusive, mirrors the aircraft study's window

BASE_URL = "https://coast.noaa.gov/htdata/CMSP/AISDataHandler/{year}/ais-{year}-{mm}-{dd}.csv.zst"

FIELDNAMES = [
    "mmsi", "base_date_time", "longitude", "latitude", "sog", "cog", "heading",
    "vessel_name", "imo", "call_sign", "vessel_type", "status", "length",
    "width", "draft", "cargo", "transceiver",
]


def daterange(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def process_day(d: date, writers: dict, counts: dict, session: requests.Session):
    url = BASE_URL.format(year=d.year, mm=f"{d.month:02d}", dd=f"{d.day:02d}")
    dctx = zstandard.ZstdDecompressor()

    last_err = None
    for attempt in range(6):
        try:
            with session.get(url, stream=True, timeout=(30, 120)) as resp:
                resp.raise_for_status()
                with dctx.stream_reader(resp.raw) as reader:
                    text_stream = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
                    r = csv.DictReader(text_stream, fieldnames=FIELDNAMES)
                    next(r)  # header row, skip
                    day_matched = {k: 0 for k in CORRIDORS}
                    n_rows = 0
                    for row in r:
                        n_rows += 1
                        try:
                            lat = float(row["latitude"])
                            lon = float(row["longitude"])
                        except (ValueError, TypeError):
                            continue
                        for name, box in CORRIDORS.items():
                            if (box["lat_min"] <= lat <= box["lat_max"]
                                    and box["lon_min"] <= lon <= box["lon_max"]):
                                writers[name].writerow(row)
                                day_matched[name] += 1
                                counts[name] += 1
                    matched_str = ", ".join(f"{k}={v}" for k, v in day_matched.items())
                    print(f"  {d}: scanned {n_rows:,} rows, matched {matched_str}")
            return
        except Exception as e:
            last_err = e
            wait = min(60, 5 * (attempt + 1))
            print(f"  {d}: attempt {attempt+1} failed ({e}), retrying in {wait}s...", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError(f"Failed to fetch {url} after 6 attempts: {last_err}")


def main():
    files = {}
    writers = {}
    counts = {name: 0 for name in CORRIDORS}

    for name in CORRIDORS:
        out_dir = DATA_ROOT / name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "ais_raw.csv"
        f = open(out_path, "w", newline="", encoding="utf-8")
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        files[name] = f
        writers[name] = w

    session = requests.Session()
    failed_days = []
    try:
        for d in daterange(START_DATE, END_DATE):
            t0 = time.time()
            try:
                process_day(d, writers, counts, session)
            except Exception as e:
                print(f"  {d}: GAVE UP after retries ({e}); skipping this day", file=sys.stderr)
                failed_days.append(d)
                continue
            for f in files.values():
                f.flush()
            print(f"    ({time.time() - t0:.0f}s)")
    finally:
        for f in files.values():
            f.close()

    if failed_days:
        print(f"\nWARNING: {len(failed_days)} day(s) could not be fetched: {failed_days}")

    print("\nDone. Total matched rows:")
    for name, c in counts.items():
        print(f"  {name}: {c:,}")


if __name__ == "__main__":
    main()
