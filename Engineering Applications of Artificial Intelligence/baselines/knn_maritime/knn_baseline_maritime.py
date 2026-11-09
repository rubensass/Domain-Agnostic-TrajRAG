"""
Physical (non-parametric) KNN baseline for maritime GNSS-blackout trajectory
reconstruction -- maritime port of GPS-SPOOFING-CORRECTION/knn_baseline.py.

Protocol (identical to TrajRAG's own evaluation):
  * For each test voyage-phase, observe the first N//2 steps (prefix).
  * Find the 5 training voyage-phases (same phase label) whose prefixes are
    most similar by mean point-wise Haversine distance.
  * Predict each blackout step as the coordinate-wise mean of the 5 neighbours'
    continuations at that step.
  * Evaluate with mean Haversine error (ADE, km) + final-step error (FDE, km).

Unlike the 7 TSLib models (which need a small, exact-fit window for the fixed
architecture), KNN naturally handles variable-length phases -- so it is run
against the FULL real second half of each phase (matching TrajRAG/Kalman/
ARIMA/LSTM's own evaluation protocol exactly), not the smaller fixed windows
used for the 7 TSLib forecasters.

Train/test split: same exact 125-voyage HOU-SHIPCHANNEL TrajRAG test set used
throughout this project; remaining voyages split chronologically 85/15 train/val
(val unused by KNN itself, kept only for parity in reporting).
"""
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
VOYAGE_FILE = (
    HERE.parents[2] / "MARITIME-SPOOFING" / "data" / "HOU-SHIPCHANNEL"
    / "voyage_data_with_minutes_since_start.json"
)
TEST_SELECTION_FILE = (
    HERE.parents[2] / "MARITIME-SPOOFING" / "evaluation" / "results"
    / "test_voyage_selection.json"
)
PHASES = ["departure", "transit", "arrival"]
MIN_PHASE_POINTS = 4
K = 5


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    p = math.pi / 180.0
    dlat, dlon = (lat2 - lat1) * p, (lon2 - lon1) * p
    a = (math.sin(dlat / 2) ** 2
         + math.cos(lat1 * p) * math.cos(lat2 * p) * math.sin(dlon / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(max(0.0, min(1.0, a))))


def prefix_dist(q, t):
    overlap = min(len(q), len(t))
    if overlap == 0:
        return float("inf")
    return sum(haversine_km(q[i][0], q[i][1], t[i][0], t[i][1])
              for i in range(overlap)) / overlap


def knn_predict(prefix, pred_len, index, k=K):
    dists = sorted(((prefix_dist(prefix, p), c) for p, c in index), key=lambda x: x[0])
    neighbors = dists[:k]
    predicted = []
    for step in range(pred_len):
        lats = [c[step][0] for _, c in neighbors if len(c) > step]
        lons = [c[step][1] for _, c in neighbors if len(c) > step]
        if lats:
            predicted.append((float(np.mean(lats)), float(np.mean(lons))))
        elif predicted:
            predicted.append(predicted[-1])
        else:
            predicted.append(prefix[-1])
    return predicted


def load_entries():
    with open(VOYAGE_FILE, "r", encoding="utf-8") as f:
        voyages = json.load(f)
    with open(TEST_SELECTION_FILE, "r", encoding="utf-8") as f:
        test_ids = {e["voyage_id"] for e in json.load(f)["HOU-SHIPCHANNEL"]}

    entries = []
    for v in voyages:
        for phase in PHASES:
            pts = v.get(phase)
            if not pts or len(pts) < MIN_PHASE_POINTS:
                continue
            pts_sorted = sorted(pts, key=lambda p: p["timestamp"])
            latlon = [(float(p["latitude"]), float(p["longitude"])) for p in pts_sorted]
            entries.append({
                "voyage_id": v["voyage_id"], "phase": phase,
                "timestamp": pts_sorted[0]["timestamp"],
                "latlon": latlon, "n": len(latlon),
                "is_test": v["voyage_id"] in test_ids,
            })
    return entries


def main():
    entries = load_entries()
    test = [e for e in entries if e["is_test"]]
    remaining = sorted((e for e in entries if not e["is_test"]), key=lambda e: e["timestamp"])
    n_train = int(len(remaining) * 0.85)
    train = remaining[:n_train]

    print(f"Entries: {len(entries)}  train={len(train)}  test={len(test)}")

    train_index = defaultdict(list)
    for e in train:
        split = e["n"] // 2
        train_index[e["phase"]].append((e["latlon"][:split], e["latlon"][split:]))

    phase_ade = defaultdict(list)
    phase_fde = defaultdict(list)
    for e in test:
        split = e["n"] // 2
        prefix, ground_truth = e["latlon"][:split], e["latlon"][split:]
        pred_len = len(ground_truth)
        if pred_len == 0 or not train_index[e["phase"]]:
            continue
        predicted = knn_predict(prefix, pred_len, train_index[e["phase"]])
        errs = [haversine_km(ground_truth[i][0], ground_truth[i][1],
                             predicted[i][0], predicted[i][1])
               for i in range(pred_len)]
        phase_ade[e["phase"]].append(float(np.mean(errs)))
        phase_fde[e["phase"]].append(errs[-1])

    results = {}
    for phase in PHASES:
        ade = phase_ade[phase]
        fde = phase_fde[phase]
        results[phase] = {
            "ade_km": float(np.mean(ade)) if ade else float("nan"),
            "fde_km": float(np.mean(fde)) if fde else float("nan"),
            "n_test": len(ade),
        }
        print(f"[kNN/{phase}] n_test={len(ade)}  "
              f"ADE={results[phase]['ade_km']:.4f} km  FDE={results[phase]['fde_km']:.4f} km")

    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "knn_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {out_dir / 'knn_results.json'}")


if __name__ == "__main__":
    main()
