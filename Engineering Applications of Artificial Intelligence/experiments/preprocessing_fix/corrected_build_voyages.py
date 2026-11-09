"""
Free (no-API-cost) diagnostic fix for a confirmed leakage bug in
MARITIME-SPOOFING/data_pipeline/build_voyages.py's resample_voyage():
latitude/longitude use pandas .interpolate(method="time"), which blends the
nearest raw AIS report BEFORE and AFTER each 5-minute grid timestamp -- i.e.
a grid point can be computed using a report recorded after it. Every other
resampled field (sog/cog/heading/...) already uses .ffill().bfill() (causal).

This script does NOT modify MARITIME-SPOOFING -- it only imports read-only
from build_voyages.py and re-runs the orchestration with a corrected
resampler that treats lat/lon exactly like every other field.

Outputs (this folder):
  voyages_corrected_<CORRIDOR>.json  -- same schema as the original voyages.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MARITIME_ROOT = Path(__file__).resolve().parents[3] / "MARITIME-SPOOFING"
sys.path.insert(0, str(MARITIME_ROOT / "data_pipeline"))

import build_voyages as bv  # noqa: E402  (read-only import of original pipeline)

OUT_DIR = Path(__file__).parent
CORRIDORS = bv.CORRIDORS


def resample_voyage_corrected(seg: pd.DataFrame) -> pd.DataFrame:
    """Identical to build_voyages.resample_voyage, except latitude/longitude
    also use .ffill().bfill() instead of .interpolate(method='time'), so they
    follow the same causal (no-future-information) convention already used
    for every other field in the original function."""
    seg = seg.set_index("base_date_time")
    seg = seg[~seg.index.duplicated(keep="last")].sort_index()
    start, end = seg.index.min(), seg.index.max()
    grid = pd.date_range(start, end, freq=f"{bv.RESAMPLE_MINUTES}min")
    if len(grid) < 2:
        return pd.DataFrame()

    causal_cols = [
        "latitude", "longitude",
        "sog", "cog", "heading", "status", "vessel_type", "length", "width", "draft",
    ]
    causal = seg[causal_cols].reindex(seg.index.union(grid)).ffill().bfill().loc[grid]

    out = causal.copy()
    out.index.name = "timestamp"
    return out.reset_index()


def build_corridor_corrected(name: str) -> list:
    print(f"\n=== {name} (corrected resampling) ===")
    df = bv.load_corridor(name)
    print(f"Loaded {len(df):,} commercial-vessel rows across {df['mmsi'].nunique():,} distinct MMSIs")

    voyages_out = []
    for mmsi, vessel_df in df.groupby("mmsi"):
        for seg in bv.split_voyages(vessel_df.reset_index(drop=True)):
            if len(seg) < bv.MIN_POINTS:
                continue
            duration_min = (seg["base_date_time"].max() - seg["base_date_time"].min()).total_seconds() / 60.0
            if not (bv.MIN_DURATION_MIN <= duration_min <= bv.MAX_DURATION_MIN):
                continue
            net_disp = bv.haversine_nm(
                seg["latitude"].iloc[0], seg["longitude"].iloc[0],
                seg["latitude"].iloc[-1], seg["longitude"].iloc[-1],
            )
            if net_disp < bv.MIN_NET_DISPLACEMENT_NM:
                continue

            resampled = resample_voyage_corrected(seg)
            if len(resampled) < bv.MIN_POINTS:
                continue
            resampled = bv.phase_segment(resampled)

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

    print(f"Reconstructed {len(voyages_out):,} voyages (corrected)")

    out_path = OUT_DIR / f"voyages_corrected_{name}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(voyages_out, f)
    print(f"Wrote {out_path}")
    return voyages_out


def main():
    for name in CORRIDORS:
        build_corridor_corrected(name)


if __name__ == "__main__":
    main()
