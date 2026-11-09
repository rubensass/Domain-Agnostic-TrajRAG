"""
Task C: recompute the kinematic-guard calibration table (rho = d_hav(y_{i-1},
y_i) / (v_{i-1} * dt)) on the CORRECTED Houston data, over consecutive
ordinary step pairs within each phase of the non-test training voyages
(near-zero-SOG steps, v_{i-1} < 1.0 knot, excluded -- matching
IDLE_SOG_KNOTS used elsewhere in this project, e.g.
MARITIME-SPOOFING/data_pipeline/build_voyages.py).
"""
import json
import math
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).parent
EXPERIMENTS_DIR = THIS_DIR.parent
EAAI_DIR = EXPERIMENTS_DIR.parent
THESIS_ROOT = EAAI_DIR.parent
PREPROC_DIR = EXPERIMENTS_DIR / "preprocessing_fix"

RESAMPLE_MINUTES = 5
DT_HOURS = RESAMPLE_MINUTES / 60.0
IDLE_SOG_KNOTS = 1.0
PHASES = ["departure", "transit", "arrival"]


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def main():
    with open(PREPROC_DIR / "voyage_data_corrected_HOU-SHIPCHANNEL.json", "r", encoding="utf-8") as f:
        voyages = json.load(f)
    with open(THESIS_ROOT / "MARITIME-SPOOFING" / "evaluation" / "results" / "test_voyage_selection.json",
              "r", encoding="utf-8") as f:
        test_ids = {v["voyage_id"] for v in json.load(f)["HOU-SHIPCHANNEL"]}

    train_voyages = [v for v in voyages if v["voyage_id"] not in test_ids]

    table = {}
    all_ratios = []
    for phase in PHASES:
        ratios = []
        for v in train_voyages:
            seq = v.get(phase, [])
            for i in range(1, len(seq)):
                prev, cur = seq[i - 1], seq[i]
                v_prev = prev.get("sog") or 0.0
                if v_prev < IDLE_SOG_KNOTS:
                    continue
                d_km = haversine_km(prev["latitude"], prev["longitude"], cur["latitude"], cur["longitude"])
                v_kmh = v_prev * 1.852
                denom = v_kmh * DT_HOURS
                if denom <= 0:
                    continue
                ratios.append(d_km / denom)
        arr = np.array(ratios)
        q99 = float(np.percentile(arr, 99)) if len(arr) else None
        share = float((arr > 1.2).mean() * 100) if len(arr) else None
        table[phase] = {"step_pairs": len(arr), "Q99_rho": round(q99, 2) if q99 else None,
                         "share_over_1.2_pct": round(share, 2) if share else None}
        all_ratios.extend(ratios)
        print(phase, table[phase])

    arr_all = np.array(all_ratios)
    table["all_phases"] = {
        "step_pairs": len(arr_all),
        "Q99_rho": round(float(np.percentile(arr_all, 99)), 2),
        "share_over_1.2_pct": round(float((arr_all > 1.2).mean() * 100), 2),
    }
    print("all_phases", table["all_phases"])

    out_path = THIS_DIR / "calibration_corrected.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(table, f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
