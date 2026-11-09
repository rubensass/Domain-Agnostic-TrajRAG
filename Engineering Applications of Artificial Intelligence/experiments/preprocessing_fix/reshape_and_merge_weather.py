"""
Reshapes voyages_corrected_<CORRIDOR>.json into the phase-keyed schema the
free baselines consume (same schema as
MARITIME-SPOOFING/data/<CORRIDOR>/voyage_data_with_minutes_since_start.json),
and copies weather fields positionally from the ORIGINAL enriched data (this
fix only changes how lat/lon are resampled; weather is joined by timestamp
independently of that, and grid timestamps are unchanged by the fix, so
point counts per phase should align 1:1 for the same voyage_id).

Writes: voyage_data_corrected_<CORRIDOR>.json
"""

import json
import sys
from pathlib import Path

MARITIME_ROOT = Path(__file__).resolve().parents[3] / "MARITIME-SPOOFING"
sys.path.insert(0, str(MARITIME_ROOT / "data_pipeline"))

import build_embeddings as be  # noqa: E402  (read-only import; PHASES, reshape_to_phase_keyed)

THIS_DIR = Path(__file__).parent
DATA_ROOT = MARITIME_ROOT / "data"
CORRIDORS = ["HOU-SHIPCHANNEL", "LA-LONGBEACH"]
PHASES = be.PHASES

WEATHER_FIELDS = [
    "wave_height", "wave_direction", "wave_period",
    "swell_wave_height", "swell_wave_direction",
    "ocean_current_velocity", "ocean_current_direction",
    "sea_surface_temperature", "wind_speed_10m", "wind_direction_10m",
]


def merge_corridor(corridor: str):
    print(f"\n=== {corridor} ===")
    with open(THIS_DIR / f"voyages_corrected_{corridor}.json", "r", encoding="utf-8") as f:
        corrected_raw = json.load(f)
    corrected_phase_keyed = be.reshape_to_phase_keyed(corrected_raw)
    corrected_by_id = {r["voyage_id"]: r for r in corrected_phase_keyed}

    with open(DATA_ROOT / corridor / "voyage_data_with_minutes_since_start.json", "r", encoding="utf-8") as f:
        original = json.load(f)
    original_by_id = {r["voyage_id"]: r for r in original}

    n_total = 0
    n_matched_voyage = 0
    n_missing_in_corrected = 0
    n_phase_mismatch = 0
    n_points_merged = 0
    n_points_dropped_no_match = 0

    merged_out = []
    for vid, orig_rec in original_by_id.items():
        n_total += 1
        corr_rec = corrected_by_id.get(vid)
        if corr_rec is None:
            n_missing_in_corrected += 1
            continue
        n_matched_voyage += 1

        out_rec = {
            "vessel_metadata": corr_rec["vessel_metadata"],
            "voyage_id": vid,
        }
        ok = True
        for phase in PHASES:
            corr_pts = corr_rec.get(phase, [])
            orig_pts = orig_rec.get(phase, [])
            if len(corr_pts) != len(orig_pts):
                n_phase_mismatch += 1
                ok = False
                break
            merged_pts = []
            for cp, op in zip(corr_pts, orig_pts):
                mp = dict(cp)
                for wf in WEATHER_FIELDS:
                    mp[wf] = op.get(wf)
                mp["phase"] = phase
                merged_pts.append(mp)
                n_points_merged += 1
            out_rec[phase] = merged_pts
        if not ok:
            continue
        merged_out.append(out_rec)

    print(f"Original voyages: {n_total}")
    print(f"Matched in corrected set: {n_matched_voyage}  (missing: {n_missing_in_corrected})")
    print(f"Phase-length mismatches (voyage dropped): {n_phase_mismatch}")
    print(f"Voyages written: {len(merged_out)}")
    print(f"Points merged with weather: {n_points_merged}")

    out_path = THIS_DIR / f"voyage_data_corrected_{corridor}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(merged_out, f)
    print(f"Wrote {out_path}")

    return {
        "corridor": corridor,
        "n_original_voyages": n_total,
        "n_matched_voyage": n_matched_voyage,
        "n_missing_in_corrected": n_missing_in_corrected,
        "n_phase_mismatch_dropped": n_phase_mismatch,
        "n_voyages_written": len(merged_out),
    }


def main():
    report = [merge_corridor(c) for c in CORRIDORS]
    with open(THIS_DIR / "merge_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
