"""
Generic trainer/evaluator for one (model, phase) combination out of the 7
TSLib forecasters x 3 maritime phases = 21 runs.

Usage:
    python train_tslib_model.py --model DLinear --phase departure
    python train_tslib_model.py --model TimesNet --phase arrival --epochs 60
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import load_phase_data, haversine_km, TARGET_IDX
from model_wrapper import (MaritimeConfigs, build_model, make_decoder_inputs,
                            make_time_marks, ENCODER_DECODER_MODELS, SUPPORTED)

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
CHECKPOINT_DIR = HERE / "checkpoints"
NM_TO_KM = 1.852


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)


def run_model_forward(model, name, x, y, device):
    if name in ENCODER_DECODER_MODELS:
        label_len = x.shape[1]
        pred_len = y.shape[1]
        dec_inp = make_decoder_inputs(x, label_len, pred_len, device)
        x_mark_enc = make_time_marks(x.shape[0], x.shape[1], device)
        x_mark_dec = make_time_marks(x.shape[0], dec_inp.shape[1], device)
        out = model(x, x_mark_enc, dec_inp, x_mark_dec)
    else:
        out = model(x, None, None, None)
    return out[:, -y.shape[1]:, :]


def train_and_eval(model_name: str, phase: str, epochs: int, patience: int, lr: float):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds, val_ds, test_entries, scaler, window = load_phase_data(phase)
    print(f"[{model_name}/{phase}] window={window}  train={len(train_ds)} "
          f"val={len(val_ds)} test={len(test_entries)}")

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    cfg = MaritimeConfigs(seq_len=window, label_len=window, pred_len=window)
    model = build_model(model_name, cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    best_val = float("inf")
    best_state = None
    no_improve = 0
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = run_model_forward(model, model_name, x, y, device)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = run_model_forward(model, model_name, x, y, device)
                val_losses.append(criterion(pred, y).item())

        val_loss = float(np.mean(val_losses)) if val_losses else float("inf")
        marker = ""
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
            marker = "  *"
        else:
            no_improve += 1

        print(f"epoch {epoch:03d}  train_loss={np.mean(train_losses):.4f}  "
              f"val_loss={val_loss:.4f}{marker}")

        if no_improve >= patience:
            print(f"early stopping at epoch {epoch} (no val improvement for {patience} epochs)")
            break

    print(f"training time: {time.time() - t0:.0f}s")

    if best_state is not None:
        model.load_state_dict(best_state)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, CHECKPOINT_DIR / f"{model_name}_{phase}_best.pt")

    # --- test evaluation (Haversine km, original scale) ---
    model.eval()
    ade_list, fde_list = [], []
    with torch.no_grad():
        for _, _, x_raw, y_raw in test_entries:
            x = torch.from_numpy(scaler.transform(x_raw)).unsqueeze(0).to(device)
            y_norm = torch.from_numpy(scaler.transform(y_raw)).unsqueeze(0).to(device)
            pred_norm = run_model_forward(model, model_name, x, y_norm, device)
            pred_norm = pred_norm.squeeze(0).cpu().numpy()
            pred_latlon = scaler.inverse_target(pred_norm[:, TARGET_IDX])
            true_latlon = y_raw[:, TARGET_IDX]

            step_errs = [haversine_km(true_latlon[i, 0], true_latlon[i, 1],
                                      pred_latlon[i, 0], pred_latlon[i, 1])
                        for i in range(len(true_latlon))]
            ade_list.append(float(np.mean(step_errs)))
            fde_list.append(step_errs[-1])

    ade_km = float(np.mean(ade_list)) if ade_list else float("nan")
    fde_km = float(np.mean(fde_list)) if fde_list else float("nan")
    print(f"[{model_name}/{phase}] TEST  ADE={ade_km:.4f} km ({ade_km/NM_TO_KM:.4f} nm)  "
          f"FDE={fde_km:.4f} km ({fde_km/NM_TO_KM:.4f} nm)")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result = {
        "model": model_name, "phase": phase, "window": window,
        "n_params": n_params,
        "train_size": len(train_ds), "val_size": len(val_ds), "test_size": len(test_entries),
        "ade_km": ade_km, "fde_km": fde_km,
        "ade_nm": ade_km / NM_TO_KM, "fde_nm": fde_km / NM_TO_KM,
        "best_val_loss": best_val,
    }
    with open(RESULTS_DIR / f"{model_name}_{phase}.json", "w") as f:
        json.dump(result, f, indent=2)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=SUPPORTED)
    ap.add_argument("--phase", required=True, choices=["departure", "transit", "arrival"])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()
    train_and_eval(args.model, args.phase, args.epochs, args.patience, args.lr)


if __name__ == "__main__":
    main()
