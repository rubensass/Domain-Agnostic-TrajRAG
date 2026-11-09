"""
Builds AIS-ACNet-compatible windows from our own real, point-to-point voyage data
(MARITIME-SPOOFING/data/HOU-SHIPCHANNEL/voyage_data_with_minutes_since_start.json),
anchored to the exact TrajRAG blackout boundary (spoof_index = N // 2).

Unlike the first version of this script, windows now use PADDING + MASKING (the
mechanism AIS-ACNet's own code already has, just unused by our earlier
fixed-exact-fit design) so a single window size can hold each phase's real
half-length, whatever it is, rather than requiring every voyage to exactly fill a
small fixed slot. The per-phase window size is chosen near that phase's 95th
percentile half-length, so the large majority of real voyages are used in full
(the model predicts their true full second half, matching TrajRAG's own semantics),
with only the longest ~5% truncated to the window ceiling.

AIS-ACNet is a CNN whose fixed-size prediction head only works correctly while the
input length stays within the model's receptive field (16 steps with their default
5 blocks/2 layers/kernel size 2). Departure and transit's window sizes fit under
that default. Arrival's half-lengths run much longer (median 14, p95 32), so its
model needs a deliberately deeper network (more blocks) to cover it -- see
PHASE_BLOCKS below, consumed by train_voyage_model.py when constructing the model.

The test split uses TrajRAG's own existing 125-voyage HOU-SHIPCHANNEL test set
(MARITIME-SPOOFING/evaluation/results/test_voyage_selection.json) so both methods
are compared on the literal same held-out voyages. Train/val is the chronological
85/15 split of whatever voyages remain.

Usage:
    python build_voyage_windows.py
"""
import json
import math
from pathlib import Path

import torch

from voyage_dataset import VoyageWindowDataset

HERE = Path(__file__).resolve().parent
VOYAGE_FILE = (
    HERE.parents[2] / "MARITIME-SPOOFING" / "data" / "HOU-SHIPCHANNEL"
    / "voyage_data_with_minutes_since_start.json"
)
TEST_SELECTION_FILE = (
    HERE.parents[2] / "MARITIME-SPOOFING" / "evaluation" / "results"
    / "test_voyage_selection.json"
)
OUT_DIR = HERE / "voyage_processed"

# Same bounding box AIS-ACNet's own code hardcodes (geographic_utils.py), in degrees.
MIN_LAT, MAX_LAT, MIN_LON, MAX_LON = 28.8, 29.8, -95.25, -94.25
MIN_LAT_R, MAX_LAT_R = math.radians(MIN_LAT), math.radians(MAX_LAT)
MIN_LON_R, MAX_LON_R = math.radians(MIN_LON), math.radians(MAX_LON)
MAX_SOG_KNOTS = 30.0

# window sized near each phase's p95 half-length; blocks sized so the model's
# receptive field (~1 + 3*blocks for the default layers=2/kernel_size=2) covers it.
PHASE_WINDOWS = {"departure": 21, "transit": 15, "arrival": 32}
PHASE_BLOCKS = {"departure": 7, "transit": 5, "arrival": 11}
MIN_PHASE_POINTS = 4  # same minimum-sequence-length floor TrajRAG's own evaluation uses
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
    """Right-aligns the real observed half and left-aligns the real predicted half
    within a fixed-size (window) slot, padding/masking whatever doesn't fit."""
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

    # Pad with the nearest REAL value repeated, not zeros. In normalized space,
    # (0,0,0,0) is a real point (the bounding box's SW corner), and the model's
    # masking is only applied to its final output, never to the input during the
    # convolution stack -- so zero-padding looked, to the model, like the vessel
    # genuinely teleported to the map's corner before reappearing. Repeating the
    # edge value instead reads as "sitting still here", which the model can
    # actually learn to handle, matching the fillarr() convention their own
    # data.py already uses elsewhere for other kinds of gaps.
    input_feats = observed_feats[0:1].repeat(window, 1)
    input_feats[window - real_obs_len:] = observed_feats
    output_feats = predicted_feats[-1:].repeat(window, 1)
    output_feats[:real_pred_len] = predicted_feats

    input_mask = torch.zeros(window, dtype=torch.bool)
    output_mask = torch.zeros(window, dtype=torch.bool)
    input_mask[window - real_obs_len:] = True
    output_mask[:real_pred_len] = True

    return {
        "input": input_feats.unsqueeze(0),        # (vessels=1, window, 4)
        "output": output_feats.unsqueeze(0),
        "input_mask": input_mask.unsqueeze(0),      # (vessels=1, window)
        "output_mask": output_mask.unsqueeze(0),
        "vessels": torch.tensor(1),
    }


def build_phase_samples(voyages: list, phase: str, window: int) -> list:
    """Returns (timestamp, voyage_id, sample) tuples, timestamp-sorted."""
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
        test_ids = {e["voyage_id"] for e in json.load(f)["HOU-SHIPCHANNEL"]}
    print(f"TrajRAG test set: {len(test_ids)} voyage ids")

    OUT_DIR.mkdir(exist_ok=True)

    for phase, window in PHASE_WINDOWS.items():
        samples = build_phase_samples(voyages, phase, window)
        n = len(samples)

        test_samples = [s for s in samples if s[1] in test_ids]
        remaining = [s for s in samples if s[1] not in test_ids]
        n_train = int(len(remaining) * (0.85))
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
