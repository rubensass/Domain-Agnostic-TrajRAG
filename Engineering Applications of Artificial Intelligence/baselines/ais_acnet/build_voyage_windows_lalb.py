"""
R5 (action plan): AIS-ACNet for the LA/Long Beach corridor.

Parallel to build_voyage_windows.py (Houston), but with LA/Long Beach's own
bounding box and phase-length distribution -- these differ materially from
Houston (LA/Long Beach is a shorter, more open approach; its departure/
transit phases are much shorter, and its arrival phase, while shorter at the
tail than Houston's, has a higher median), so reusing Houston's window/block
sizes would be wrong for this corridor. Values below were computed directly
from the real LA/Long Beach data (see conversation / commit message), not
guessed:
  lat range 33.60-33.78, lon range -118.35--118.05
  departure: p50=4, p95=9   transit: p50=3, p95=7   arrival: p50=15, p95=22
"""
import json
import math
from pathlib import Path

import torch

from voyage_dataset import VoyageWindowDataset

HERE = Path(__file__).resolve().parent
VOYAGE_FILE = (
    HERE.parents[2] / "MARITIME-SPOOFING" / "data" / "LA-LONGBEACH"
    / "voyage_data_with_minutes_since_start.json"
)
TEST_SELECTION_FILE = (
    HERE.parents[2] / "MARITIME-SPOOFING" / "evaluation" / "results"
    / "test_voyage_selection.json"
)
OUT_DIR = HERE / "voyage_processed_lalb"

# LA/Long Beach's own bounding box (with margin), from the real data range above.
MIN_LAT, MAX_LAT, MIN_LON, MAX_LON = 33.5, 33.85, -118.40, -118.00
MIN_LAT_R, MAX_LAT_R = math.radians(MIN_LAT), math.radians(MAX_LAT)
MIN_LON_R, MAX_LON_R = math.radians(MIN_LON), math.radians(MAX_LON)
MAX_SOG_KNOTS = 30.0

# window sized near each phase's own p95 half-length (LA/Long Beach, not
# Houston); blocks sized so the model's receptive field (~1 + 3*blocks)
# covers the window, mirroring the ~3:1 window:blocks ratio used for Houston.
PHASE_WINDOWS = {"departure": 9, "transit": 7, "arrival": 22}
PHASE_BLOCKS = {"departure": 3, "transit": 3, "arrival": 7}
MIN_PHASE_POINTS = 4
FEATURE_SIZE = 4  # lat, lon, sog, cog


def normalize_point(p: dict) -> list:
    lat_r = math.radians(p["latitude"])
    lon_r = math.radians(p["longitude"])
    lat_n = (lat_r - MIN_LAT_R) / (MAX_LAT_R - MIN_LAT_R)
    lon_n = (lon_r - MIN_LON_R) / (MAX_LON_R - MIN_LON_R)
    sog_n = min(max(p.get("sog") or 0.0, 0.0), MAX_SOG_KNOTS) / MAX_SOG_KNOTS
    cog = p.get("cog") or 0.0
    if cog < 0:
        cog += 360
    cog_n = (cog % 360) / 360.0
    return [lat_n, lon_n, sog_n, cog_n]


def build_sample(phase_data: list, window: int):
    n = len(phase_data)
    spoof_index = n // 2
    real_obs_len = min(spoof_index, window)
    real_pred_len = min(n - spoof_index, window)
    if real_obs_len < 1 or real_pred_len < 1:
        return None

    observed_real = phase_data[spoof_index - real_obs_len: spoof_index]
    predicted_real = phase_data[spoof_index: spoof_index + real_pred_len]
    observed_feats = torch.tensor([normalize_point(p) for p in observed_real])
    predicted_feats = torch.tensor([normalize_point(p) for p in predicted_real])

    input_feats = observed_feats[0:1].repeat(window, 1)
    input_feats[window - real_obs_len:] = observed_feats
    output_feats = predicted_feats[-1:].repeat(window, 1)
    output_feats[:real_pred_len] = predicted_feats

    input_mask = torch.zeros(window, dtype=torch.bool)
    output_mask = torch.zeros(window, dtype=torch.bool)
    input_mask[window - real_obs_len:] = True
    output_mask[:real_pred_len] = True

    return {
        "input": input_feats.unsqueeze(0),
        "output": output_feats.unsqueeze(0),
        "input_mask": input_mask.unsqueeze(0),
        "output_mask": output_mask.unsqueeze(0),
        "vessels": torch.tensor(1),
    }


def build_phase_samples(voyages: list, phase: str, window: int) -> list:
    samples = []
    for v in voyages:
        phase_data = v.get(phase)
        if not phase_data or len(phase_data) < MIN_PHASE_POINTS:
            continue
        sample = build_sample(phase_data, window)
        if sample is None:
            continue
        samples.append((phase_data[0]["timestamp"], v["voyage_id"], sample))
    samples.sort(key=lambda t: t[0])
    return samples


def main():
    with open(VOYAGE_FILE, "r", encoding="utf-8") as f:
        voyages = json.load(f)
    print(f"loaded {len(voyages)} voyages from {VOYAGE_FILE}")

    with open(TEST_SELECTION_FILE, "r", encoding="utf-8") as f:
        test_ids = {e["voyage_id"] for e in json.load(f)["LA-LONGBEACH"]}
    print(f"TrajRAG test set: {len(test_ids)} voyage ids")

    OUT_DIR.mkdir(exist_ok=True)

    for phase, window in PHASE_WINDOWS.items():
        samples = build_phase_samples(voyages, phase, window)
        n = len(samples)

        test_samples = [s for s in samples if s[1] in test_ids]
        remaining = [s for s in samples if s[1] not in test_ids]
        n_train = int(len(remaining) * 0.85)
        train_samples = remaining[:n_train]
        val_samples = remaining[n_train:]

        train_ds = VoyageWindowDataset(train_samples)
        val_ds = VoyageWindowDataset(val_samples)
        test_ds = VoyageWindowDataset(test_samples)

        phase_dir = OUT_DIR / phase
        phase_dir.mkdir(parents=True, exist_ok=True)
        torch.save(train_ds, phase_dir / "train.pt")
        torch.save(val_ds, phase_dir / "val.pt")
        torch.save(test_ds, phase_dir / "test.pt")

        print(f"{phase} (window={window}, blocks={PHASE_BLOCKS[phase]}): {n} usable voyages "
              f"-> train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)} "
              f"(of {len(test_ids)} TrajRAG test ids, {len(test_samples)} had a usable {phase} phase)")


if __name__ == "__main__":
    main()
