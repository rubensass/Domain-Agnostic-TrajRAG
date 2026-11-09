"""
Builds exact-fit (no padding) windowed samples for the 7 TSLib forecasters
(DLinear, PatchTST, iTransformer, Informer, Autoformer, FEDformer, TimesNet),
from our real voyage data, anchored to TrajRAG's own blackout boundary
(spoof_index = N // 2), on the exact same 125-voyage HOU-SHIPCHANNEL test set
used throughout this project.

Unlike AIS-ACNet's windows, these are NOT padded/masked -- a voyage-phase is only
included if its real half-length exactly fits the chosen window (same "small,
universal, exact-fit" philosophy GPS-SPOOFING-CORRECTION's own aviation harness
uses: seq_len=pred_len chosen small enough that every phase can be used as-is).
This avoids repeating the AIS-ACNet masking bug across seven different, less
thoroughly-understood architectures under time pressure -- a deliberate,
documented trade-off (see this project's README for numbers).

Window sizes (already validated for retention earlier in this project):
    departure=4 (80% of voyages), transit=3 (71%), arrival=4 (77%)

8 features per point: latitude, longitude, sog, cog, heading, wave_height,
wind_speed_10m, ocean_current_velocity. Target = [latitude, longitude].
Normalization: z-score per feature, fit on the TRAIN split only.
"""
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

HERE = Path(__file__).resolve().parent
VOYAGE_FILE = (
    HERE.parents[2] / "MARITIME-SPOOFING" / "data" / "HOU-SHIPCHANNEL"
    / "voyage_data_with_minutes_since_start.json"
)
TEST_SELECTION_FILE = (
    HERE.parents[2] / "MARITIME-SPOOFING" / "evaluation" / "results"
    / "test_voyage_selection.json"
)

PHASE_WINDOWS = {"departure": 4, "transit": 3, "arrival": 4}
FEATURES = ["latitude", "longitude", "sog", "cog", "heading",
           "wave_height", "wind_speed_10m", "ocean_current_velocity"]
TARGET_IDX = [0, 1]  # latitude, longitude
MIN_PHASE_POINTS = 4
EARTH_R_KM = 6371.0


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    p = math.pi / 180.0
    a = (math.sin((lat2 - lat1) * p / 2) ** 2
         + math.cos(lat1 * p) * math.cos(lat2 * p) * math.sin((lon2 - lon1) * p / 2) ** 2)
    return 2 * EARTH_R_KM * math.asin(min(1.0, math.sqrt(max(0.0, a))))


def extract_vector(p: dict) -> list:
    return [float(p.get(f) or 0.0) for f in FEATURES]


@dataclass
class Scaler:
    mean: np.ndarray = None
    std: np.ndarray = None

    def fit(self, rows: np.ndarray):
        self.mean = rows.mean(axis=0)
        self.std = rows.std(axis=0) + 1e-8
        return self

    def transform(self, rows: np.ndarray) -> np.ndarray:
        return (rows - self.mean) / self.std

    def inverse_target(self, rows: np.ndarray) -> np.ndarray:
        return rows * self.std[TARGET_IDX] + self.mean[TARGET_IDX]


def build_phase_windows(voyages: list, phase: str, window: int):
    """Returns (timestamp, voyage_id, x_raw, y_raw) tuples, time-sorted, where
    x_raw/y_raw are (window, 8) float arrays -- exact-fit, no padding."""
    out = []
    for v in voyages:
        phase_data = v.get(phase)
        if not phase_data or len(phase_data) < MIN_PHASE_POINTS:
            continue
        n = len(phase_data)
        mid = n // 2
        if mid < window or (n - mid) < window:
            continue
        observed = phase_data[mid - window: mid]
        predicted = phase_data[mid: mid + window]
        x_raw = np.array([extract_vector(p) for p in observed], dtype=np.float32)
        y_raw = np.array([extract_vector(p) for p in predicted], dtype=np.float32)
        out.append((observed[0]["timestamp"], v["voyage_id"], x_raw, y_raw))
    out.sort(key=lambda t: t[0])
    return out


class WindowDataset(Dataset):
    def __init__(self, entries: list, scaler: Scaler):
        self.entries = entries
        self.scaler = scaler

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        _, _, x_raw, y_raw = self.entries[idx]
        x = torch.from_numpy(self.scaler.transform(x_raw))
        y = torch.from_numpy(self.scaler.transform(y_raw))
        return x, y


def load_phase_data(phase: str):
    """Returns (train_ds, val_ds, test_entries, scaler, window). test_entries kept
    raw (not wrapped in a Dataset) since evaluation needs voyage ids + Haversine
    on the ORIGINAL lat/lon scale, not normalized."""
    window = PHASE_WINDOWS[phase]
    with open(VOYAGE_FILE, "r", encoding="utf-8") as f:
        voyages = json.load(f)
    with open(TEST_SELECTION_FILE, "r", encoding="utf-8") as f:
        test_ids = {e["voyage_id"] for e in json.load(f)["HOU-SHIPCHANNEL"]}

    entries = build_phase_windows(voyages, phase, window)
    test_entries = [e for e in entries if e[1] in test_ids]
    remaining = [e for e in entries if e[1] not in test_ids]
    n_train = int(len(remaining) * 0.85)
    train_entries, val_entries = remaining[:n_train], remaining[n_train:]

    train_x = np.concatenate([e[2] for e in train_entries], axis=0)
    scaler = Scaler().fit(train_x)

    return (
        WindowDataset(train_entries, scaler),
        WindowDataset(val_entries, scaler),
        test_entries,
        scaler,
        window,
    )
