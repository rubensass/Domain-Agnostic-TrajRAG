"""
Turns the raw bounding-box-filtered AIS rows (data/<CORRIDOR>/ais_raw.csv) into
discrete per-vessel "voyages" through the corridor, resampled to a fixed
interval -- the maritime analog of a single commercial flight in the aircraft
study.

Pipeline:
  1. Load raw rows, keep commercial vessel types only (cargo=70-79, tanker=80-89),
     matching the aircraft study's restriction to scheduled commercial flights.
  2. Group by MMSI, sort by time.
  3. Split each vessel's points into candidate voyages: a new voyage starts on
     a signal gap (> GAP_MINUTES) or a prolonged idle/loiter period (SOG below
     IDLE_SOG_KNOTS for > IDLE_MINUTES) -- this separates real corridor transits
     from vessels sitting at anchor for hours/days.
  4. Keep voyages with enough net displacement, duration, and point count to be
     a real transit (not noise).
  5. Resample each kept voyage to a fixed interval via linear interpolation of
     lat/lon (numeric, safe to interpolate) and nearest-neighbour for
     circular/categorical fields (heading, cog, status) -- interpolating a
     circular quantity like heading naively would be wrong across the 0/360
     wrap, so we deliberately don't.
  6. Segment each voyage into three phases -- departure / transit / arrival --
     using the SOG trend (rises after departure, plateaus in transit, falls on
     arrival), the same role barometric-altitude-trend played for flights.
     NOTE: this SOG-based segmentation is for OFFLINE memory construction only,
     where full ground truth is available. Online (during a blackout), phase
     must be inferred from a GNSS-independent signal -- for vessels that's
     heading/rate-of-turn stability, not SOG (SOG is GPS-derived and would be
     corrupted right alongside position). That online logic belongs to the
     Phase 4 reconstruction engine, not here.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA_ROOT = Path(__file__).parent.parent / "data"

CORRIDORS = ["HOU-SHIPCHANNEL", "LA-LONGBEACH"]

RESAMPLE_MINUTES = 5
GAP_MINUTES = 60          # signal gap => new voyage
IDLE_SOG_KNOTS = 1.0      # below this = "not underway"
IDLE_MINUTES = 120        # idle longer than this => split (anchored/loitering)
MIN_POINTS = 8
MIN_DURATION_MIN = 30
MAX_DURATION_MIN = 12 * 60
MIN_NET_DISPLACEMENT_NM = 5.0

COMMERCIAL_VESSEL_TYPES = set(range(70, 80)) | set(range(80, 90))  # cargo, tanker


def haversine_nm(lat1, lon1, lat2, lon2):
    R_nm = 3440.065
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R_nm * 2 * np.arcsin(np.sqrt(a))


def load_corridor(name: str) -> pd.DataFrame:
    path = DATA_ROOT / name / "ais_raw.csv"
    df = pd.read_csv(
        path,
        dtype={
            "mmsi": "int64", "vessel_name": "string", "imo": "string",
            "call_sign": "string", "status": "string", "transceiver": "string",
        },
        parse_dates=["base_date_time"],
    )
    df["vessel_type"] = pd.to_numeric(df["vessel_type"], errors="coerce")
    df = df[df["vessel_type"].isin(COMMERCIAL_VESSEL_TYPES)].copy()
    for col in ["latitude", "longitude", "sog", "cog", "heading", "length", "width", "draft"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["latitude", "longitude", "base_date_time"])
    df = df.sort_values(["mmsi", "base_date_time"]).reset_index(drop=True)
    return df


def split_voyages(vessel_df: pd.DataFrame) -> list[pd.DataFrame]:
    """Split one vessel's chronological points into candidate voyage segments."""
    t = vessel_df["base_date_time"]
    gap_min = t.diff().dt.total_seconds().fillna(0) / 60.0

    sog = vessel_df["sog"].fillna(0.0)
    is_idle = sog < IDLE_SOG_KNOTS
    # length of the current idle run, in minutes, ending at each row
    idle_run_min = np.zeros(len(vessel_df))
    run = 0.0
    for i in range(len(vessel_df)):
        if is_idle.iloc[i]:
            run += gap_min.iloc[i] if i > 0 else 0.0
        else:
            run = 0.0
        idle_run_min[i] = run

    new_voyage = (gap_min > GAP_MINUTES) | (idle_run_min > IDLE_MINUTES)
    new_voyage.iloc[0] = True
    voyage_id = new_voyage.cumsum()

    segments = []
    for _, seg in vessel_df.groupby(voyage_id):
        segments.append(seg)
    return segments


def resample_voyage(seg: pd.DataFrame) -> pd.DataFrame:
    seg = seg.set_index("base_date_time")
    # Multiple AIS messages can share the same timestamp (sub-second pings
    # collapsed by the source data); keep the last one per timestamp so the
    # index is unique before reindexing onto the resample grid.
    seg = seg[~seg.index.duplicated(keep="last")].sort_index()
    start, end = seg.index.min(), seg.index.max()
    grid = pd.date_range(start, end, freq=f"{RESAMPLE_MINUTES}min")
    if len(grid) < 2:
        return pd.DataFrame()

    numeric_cols = ["latitude", "longitude"]
    interp = seg[numeric_cols].reindex(seg.index.union(grid)).interpolate(method="time").loc[grid]

    nearest_cols = ["sog", "cog", "heading", "status", "vessel_type", "length", "width", "draft"]
    nearest = seg[nearest_cols].reindex(seg.index.union(grid)).ffill().bfill().loc[grid]

    out = pd.concat([interp, nearest], axis=1)
    out.index.name = "timestamp"
    return out.reset_index()


def phase_segment(voyage: pd.DataFrame) -> pd.DataFrame:
    """Offline phase segmentation from the SOG trend (see module docstring)."""
    sog = voyage["sog"].rolling(3, min_periods=1, center=True).mean()
    n = len(voyage)
    peak_idx = int(sog.idxmax())

    # departure: from start until SOG first reaches ~90% of its peak
    threshold = 0.9 * sog.max()
    dep_end = next((i for i in range(n) if sog.iloc[i] >= threshold), n // 3)
    dep_end = max(dep_end, max(1, n // 6))

    # arrival: from when SOG permanently drops back below ~90% of peak, to the end
    arr_start = n - 1
    for i in range(n - 1, -1, -1):
        if sog.iloc[i] >= threshold:
            arr_start = i
            break
    arr_start = min(arr_start + 1, n - max(1, n // 6))

    dep_end = min(dep_end, arr_start - 1) if arr_start > dep_end else max(1, n // 6)

    phases = np.empty(n, dtype=object)
    phases[:dep_end] = "departure"
    phases[dep_end:arr_start] = "transit"
    phases[arr_start:] = "arrival"
    voyage = voyage.copy()
    voyage["phase"] = phases
    return voyage


def build_corridor(name: str):
    print(f"\n=== {name} ===")
    df = load_corridor(name)
    print(f"Loaded {len(df):,} commercial-vessel rows across {df['mmsi'].nunique():,} distinct MMSIs")

    voyages_out = []
    heading_available = 0
    heading_total = 0

    for mmsi, vessel_df in df.groupby("mmsi"):
        for seg in split_voyages(vessel_df.reset_index(drop=True)):
            if len(seg) < MIN_POINTS:
                continue
            duration_min = (seg["base_date_time"].max() - seg["base_date_time"].min()).total_seconds() / 60.0
            if not (MIN_DURATION_MIN <= duration_min <= MAX_DURATION_MIN):
                continue
            net_disp = haversine_nm(
                seg["latitude"].iloc[0], seg["longitude"].iloc[0],
                seg["latitude"].iloc[-1], seg["longitude"].iloc[-1],
            )
            if net_disp < MIN_NET_DISPLACEMENT_NM:
                continue

            resampled = resample_voyage(seg)
            if len(resampled) < MIN_POINTS:
                continue
            resampled = phase_segment(resampled)

            heading_total += len(seg)
            heading_available += seg["heading"].notna().sum()

            t0 = resampled["timestamp"].iloc[0]
            resampled["minutes_since_start"] = (resampled["timestamp"] - t0).dt.total_seconds() / 60.0

            meta_row = seg.iloc[0]

            def _s(val):
                return "" if pd.isna(val) else str(val)

            voyages_out.append({
                "vessel_metadata": {
                    "mmsi": int(mmsi),
                    "vessel_name": _s(meta_row.get("vessel_name")),
                    "imo": _s(meta_row.get("imo")),
                    "call_sign": _s(meta_row.get("call_sign")),
                    "vessel_type": int(meta_row["vessel_type"]) if pd.notna(meta_row["vessel_type"]) else None,
                    "length": float(meta_row["length"]) if pd.notna(meta_row["length"]) else None,
                    "width": float(meta_row["width"]) if pd.notna(meta_row["width"]) else None,
                },
                "start_time": t0.isoformat(),
                "duration_min": round(duration_min, 1),
                "net_displacement_nm": round(float(net_disp), 2),
                "points": json.loads(resampled.to_json(orient="records", date_format="iso")),
            })

    print(f"Reconstructed {len(voyages_out):,} voyages")
    if heading_total:
        print(f"Heading field availability: {heading_available / heading_total:.1%} of raw points")
    if voyages_out:
        durations = [v["duration_min"] for v in voyages_out]
        print(f"Duration (min): min={min(durations):.0f} median={np.median(durations):.0f} max={max(durations):.0f}")
        phase_counts = {}
        for v in voyages_out:
            for p in v["points"]:
                phase_counts[p["phase"]] = phase_counts.get(p["phase"], 0) + 1
        print(f"Phase point counts: {phase_counts}")

    out_path = DATA_ROOT / name / "voyages.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(voyages_out, f)
    print(f"Wrote {out_path}")


def main():
    for name in CORRIDORS:
        build_corridor(name)


if __name__ == "__main__":
    main()
