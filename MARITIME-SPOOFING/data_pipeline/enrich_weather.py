"""
Enriches each voyage point with marine weather: wave height/period/direction,
swell, ocean current velocity/direction, sea-surface temperature (Open-Meteo
Marine API) and surface wind speed/direction (Open-Meteo historical archive
API). This is the vessel analog of the aircraft study's Meteomatics alignment
(turbulence/wind-aloft indexed by ADS-B timestamp + barometric altitude).

Both Open-Meteo endpoints are free, keyless, and support historical dates.
Since each call returns a full day of hourly values for one lat/lon, we batch
requests by (date, lat/lon rounded to a 0.1 deg grid) rather than calling once
per point -- one call covers every voyage point that day in that grid cell.
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

DATA_ROOT = Path(__file__).parent.parent / "data"
CORRIDORS = ["HOU-SHIPCHANNEL", "LA-LONGBEACH"]

GRID_DEG = 0.1
MARINE_URL = "https://marine-api.open-meteo.com/v1/marine"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

MARINE_VARS = [
    "wave_height", "wave_direction", "wave_period",
    "swell_wave_height", "swell_wave_direction",
    "ocean_current_velocity", "ocean_current_direction",
    "sea_surface_temperature",
]
WIND_VARS = ["wind_speed_10m", "wind_direction_10m"]


def fetch_day(session: requests.Session, url: str, lat: float, lon: float, day: str, hourly_vars: list[str]) -> dict | None:
    params = {
        "latitude": round(lat, 2), "longitude": round(lon, 2),
        "start_date": day, "end_date": day,
        "hourly": ",".join(hourly_vars),
        "timezone": "UTC",
    }
    for attempt in range(4):
        try:
            resp = session.get(url, params=params, timeout=30)
            if resp.status_code == 200:
                return resp.json().get("hourly")
            time.sleep(2 * (attempt + 1))
        except requests.RequestException:
            time.sleep(2 * (attempt + 1))
    print(f"  WARN: failed to fetch {url} for {lat},{lon} on {day}")
    return None


def build_cache_key(lat: float, lon: float, day: str) -> tuple:
    return (round(lat / GRID_DEG) * GRID_DEG, round(lon / GRID_DEG) * GRID_DEG, day)


def enrich_corridor(name: str):
    print(f"\n=== {name} ===")
    path = DATA_ROOT / name / "voyages.json"
    with open(path, "r", encoding="utf-8") as f:
        voyages = json.load(f)

    # Collect every (grid_lat, grid_lon, day) combo we'll need, up front.
    needed = set()
    for v in voyages:
        for p in v["points"]:
            ts = pd.Timestamp(p["timestamp"])
            key = build_cache_key(p["latitude"], p["longitude"], ts.strftime("%Y-%m-%d"))
            needed.add(key)
    print(f"{len(voyages):,} voyages, {sum(len(v['points']) for v in voyages):,} points, "
          f"{len(needed):,} unique (grid-cell, day) weather queries needed")

    marine_cache: dict[tuple, dict] = {}
    wind_cache: dict[tuple, dict] = {}
    session = requests.Session()

    for i, (glat, glon, day) in enumerate(sorted(needed)):
        marine_cache[(glat, glon, day)] = fetch_day(session, MARINE_URL, glat, glon, day, MARINE_VARS)
        wind_cache[(glat, glon, day)] = fetch_day(session, ARCHIVE_URL, glat, glon, day, WIND_VARS)
        if (i + 1) % 50 == 0 or i + 1 == len(needed):
            print(f"  fetched {i + 1}/{len(needed)} weather cells")
        time.sleep(0.2)

    def nearest_hour_value(hourly: dict | None, ts: pd.Timestamp, var: str):
        if not hourly or var not in hourly:
            return None
        times = hourly["time"]
        idx = min(range(len(times)), key=lambda j: abs(pd.Timestamp(times[j]) - ts), default=None)
        if idx is None:
            return None
        val = hourly[var][idx]
        return None if val is None else val

    for v in voyages:
        for p in v["points"]:
            ts = pd.Timestamp(p["timestamp"])
            key = build_cache_key(p["latitude"], p["longitude"], ts.strftime("%Y-%m-%d"))
            marine = marine_cache.get(key)
            wind = wind_cache.get(key)
            for var in MARINE_VARS:
                p[var] = nearest_hour_value(marine, ts, var)
            for var in WIND_VARS:
                p[var] = nearest_hour_value(wind, ts, var)

    out_path = DATA_ROOT / name / "voyages_enriched.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(voyages, f)
    print(f"Wrote {out_path}")


def main():
    for name in CORRIDORS:
        enrich_corridor(name)


if __name__ == "__main__":
    main()
