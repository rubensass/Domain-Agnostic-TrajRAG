"""
Learning-based sequence baseline for vessel trajectory reconstruction -- the
maritime analog of methods/LSTM.ipynb (read-only reference; fresh
implementation). Same architecture family as the aircraft study: an
autoregressive LSTM (optionally bidirectional-encoder) trained per phase
across many voyages, then used open-loop (feeding its own predictions back
in) to reconstruct the spoofed half at inference time.

FEATURES mirrors the aircraft study's 8-feature state vector, swapped for the
vessel/marine equivalents:
  aircraft: lat, lon, altitude_ft, gspeed, vspeed, track, wind_speed_10000m_ms, wind_dir_10000m_d
  vessel:   latitude, longitude, sog, cog, heading, wave_height, wind_speed_10m, ocean_current_velocity
"""

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.utils.data import Dataset, DataLoader

from kalman_baseline import haversine_km

FEATURES = [
    "latitude", "longitude", "sog", "cog", "heading",
    "wave_height", "wind_speed_10m", "ocean_current_velocity",
]


def seq_to_matrix(seq: list[dict]) -> np.ndarray:
    return np.array([[p.get(f) or 0.0 for f in FEATURES] for p in seq], dtype=np.float32)


class PhaseDataset(Dataset):
    def __init__(self, sequences: list[np.ndarray], means: np.ndarray, stds: np.ndarray):
        self.sequences = [(seq - means) / stds for seq in sequences]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        return torch.from_numpy(seq[:-1]), torch.from_numpy(seq[1:])


def collate_fn(batch):
    xs, ys = zip(*batch)
    lengths = torch.tensor([x.shape[0] for x in xs])
    xs_pad = pad_sequence(xs, batch_first=True)
    ys_pad = pad_sequence(ys, batch_first=True)
    return xs_pad, ys_pad, lengths


class BaselineLSTM(nn.Module):
    def __init__(self, input_dim, hidden=128, layers=6, bidirectional=False):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, layers, batch_first=True, bidirectional=bidirectional)
        out_dim = hidden * (2 if bidirectional else 1)
        self.fc = nn.Linear(out_dim, input_dim)

    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(out, batch_first=True)
        return self.fc(out)


def compute_norm_stats(sequences: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    stacked = np.concatenate(sequences, axis=0)
    means = stacked.mean(axis=0)
    stds = stacked.std(axis=0)
    stds[stds < 1e-6] = 1.0
    return means, stds


def train_phase_model(train_sequences: list[np.ndarray], means, stds, epochs=50, bidirectional=False) -> BaselineLSTM:
    dataset = PhaseDataset(train_sequences, means, stds)
    loader = DataLoader(dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)

    model = BaselineLSTM(input_dim=len(FEATURES), bidirectional=bidirectional)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = nn.MSELoss()

    model.train()
    for _ in range(epochs):
        for xb, yb, lengths in loader:
            opt.zero_grad()
            pred = model(xb, lengths)
            mask = (torch.arange(yb.shape[1])[None, :] < lengths[:, None]).unsqueeze(-1)
            mask = mask.expand_as(pred)
            loss = loss_fn(pred[mask], yb[mask])
            loss.backward()
            opt.step()
    return model


def lstm_predict_phase(model: BaselineLSTM, seq: list[dict], means, stds) -> tuple[np.ndarray, np.ndarray]:
    """Feed the observed half, then run open-loop (feed predictions back in) over the spoofed half."""
    mat = seq_to_matrix(seq)
    T = len(mat)
    mid = T // 2
    norm = (mat - means) / stds

    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(norm[:mid]).unsqueeze(0)
        lengths = torch.tensor([mid])
        out = model(x, lengths)
        current = out[0, -1].clone()

        preds = []
        for _ in range(T - mid):
            x_step = current.unsqueeze(0).unsqueeze(0)
            out_step = model(x_step, torch.tensor([1]))
            current = out_step[0, -1]
            preds.append(current.numpy())

    preds = np.array(preds) * stds + means
    gt = mat[mid:]
    lat_lon_idx = [FEATURES.index("latitude"), FEATURES.index("longitude")]
    return preds[:, lat_lon_idx], gt[:, lat_lon_idx]


def evaluate_phase(train_sequences: list[list[dict]], test_sequences: list[list[dict]],
                    epochs=50, bidirectional=False) -> list[float]:
    train_mats = [seq_to_matrix(s) for s in train_sequences if len(s) >= 4]
    if not train_mats:
        return []
    means, stds = compute_norm_stats(train_mats)
    model = train_phase_model(train_mats, means, stds, epochs=epochs, bidirectional=bidirectional)

    errors = []
    for seq in test_sequences:
        if len(seq) < 4:
            continue
        try:
            preds, gt = lstm_predict_phase(model, seq, means, stds)
        except Exception:
            continue
        err = np.mean([haversine_km(gt[i][0], gt[i][1], preds[i][0], preds[i][1]) for i in range(len(preds))])
        errors.append(float(err))
    return errors
