"""
Trains one AIS-ACNet model on the voyage-anchored dataset built by
build_voyage_windows.py, for ONE phase at a time (departure/transit/arrival each
need their own model, since the CNN's prediction head is a fixed size matching
that phase's window).

Reuses external/AIS-ACNet/AIS-ACNet's model.py and engine.py unmodified, with the
same upstream-bug workaround used elsewhere in this project: engine.trainer
references self.loss2 without ever defining it, so we subclass and add it
(masked_mse, matching the style of the other masked losses already in their
utils.py) rather than editing their file.

Usage:
    python train_voyage_model.py --phase departure --epochs 60
"""
import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
EXTERNAL_REPO = HERE.parents[1] / "external" / "AIS-ACNet" / "AIS-ACNet"
sys.path.insert(0, str(EXTERNAL_REPO))

import torch.optim as optim

from data import collate_function  # noqa: E402  (external repo module, unmodified)
from engine import trainer as _BaseTrainer  # noqa: E402
from model import gwnet  # noqa: E402
import utils as ais_acnet_utils  # noqa: E402
from geographic_utils import min_lat, max_lat, min_lon, max_lon  # noqa: E402

from voyage_dataset import VoyageWindowDataset  # noqa: E402,F401  (needed for unpickling)
from build_voyage_windows import PHASE_WINDOWS, PHASE_BLOCKS  # noqa: E402

NM_TO_KM = 1.852  # utils.haversine_loss uses Earth radius in nautical miles


def _squeeze_vessels(mask: torch.Tensor) -> torch.Tensor:
    """s_pred/h_pred go through .squeeze() inside engine.py's train()/eval(), which
    collapses the vessels dimension whenever vessels == 1 (always true for our
    single-voyage windows, never true in AIS-ACNet's original multi-vessel-scene
    setting -- which is why this mismatch never surfaced in their own code). This
    mirrors that same squeeze on the mask so the two line up for the aux losses."""
    return mask.squeeze(1) if mask.dim() == 3 and mask.size(1) == 1 else mask


class FixedTrainer(_BaseTrainer):
    """Two upstream-bug workarounds, kept local to this file rather than editing
    their repo:
    1. engine.trainer.train/eval call self.loss2 for the SOG/COG auxiliary losses,
       but __init__ never defines it -- added here (masked_mse, matching the style
       of the other masked losses already in their utils.py).
    2. train()/eval() index s_pred/h_pred with the *un-squeezed* op_mask, which only
       works when vessels > 1 (see _squeeze_vessels above). train()/eval() are
       therefore copied here verbatim from engine.py with just that one shape fix,
       rather than patched at the attribute level."""

    def __init__(self, in_dim, seq_length, nhid, dropout, lrate, wdecay, device, blocks=5):
        super().__init__(in_dim, seq_length, nhid, dropout, lrate, wdecay, device)
        self.loss2 = ais_acnet_utils.masked_mse

        if blocks != 5:
            # engine.trainer.__init__ hardcodes blocks=5 (gwnet's own default) and
            # doesn't expose it as a parameter -- rebuild the model (and, critically,
            # the optimizer, which otherwise still points at the old model's
            # parameters) with the receptive field this phase's window actually needs.
            self.model = gwnet(
                device, dropout, in_dim=in_dim, out_dim=seq_length,
                residual_channels=nhid, dilation_channels=nhid,
                skip_channels=nhid * 8, end_channels=nhid * 16, blocks=blocks,
            )
            self.model.to(device)
            self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)

    def train(self, input, real, ip_mask, op_mask, adj=None):
        self.model.train()
        self.optimizer.zero_grad()
        output, s_pred, h_pred = self.model(input, ip_mask, adj=adj)

        output = output.transpose(1, 3).contiguous()
        # Targeted squeeze(1) twice (not a blanket squeeze()) so this is correct
        # regardless of batch size -- a blanket squeeze() also drops the batch
        # dimension itself whenever a batch happens to contain exactly 1 sample
        # (e.g. a dataset size not evenly divisible by batch_size), which silently
        # breaks the op_mask_aux indexing below for that one batch.
        s_pred = s_pred.transpose(1, 3).contiguous().squeeze(1).squeeze(1)
        h_pred = h_pred.transpose(1, 3).contiguous().squeeze(1).squeeze(1)
        op_mask_aux = _squeeze_vessels(op_mask)

        s_pred = s_pred * 51.1
        real[:, 2, :, :] = real[:, 2, :, :] * 51.1
        s_pred[~op_mask_aux] = np.nan

        h_pred = h_pred * torch.pi
        real[:, 3, :, :] = real[:, 3, :, :] * torch.pi
        h_pred[~op_mask_aux] = np.nan

        output[:, 0, :, :] = output[:, 0, :, :] * (max_lat - min_lat) + min_lat
        output[:, 1, :, :] = output[:, 1, :, :] * (max_lon - min_lon) + min_lon
        real[:, 0, :, :] = real[:, 0, :, :] * (max_lat - min_lat) + min_lat
        real[:, 1, :, :] = real[:, 1, :, :] * (max_lon - min_lon) + min_lon

        ADE = self.loss(real[:, :2, :, :], output, mask=op_mask)
        sloss = self.alpha * self.loss2(real[:, 2, :, :], s_pred, null_val=np.nan)
        hloss = self.beta * self.loss2(real[:, 3, :, :], h_pred, null_val=np.nan)
        loss = ADE + sloss + hloss

        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()

        fde = ais_acnet_utils.haversine_loss(
            output[:, :, :, -1:], real[:, :2, :, -1:], mask=op_mask[:, :, -1:]
        ).item()
        return loss.item(), ADE.item(), fde

    def eval(self, input, real, ip_mask, op_mask, adj=None, return_loss_terms=False):
        self.model.eval()
        output, s_pred, h_pred = self.model(input, ip_mask, adj=adj)
        output = output.transpose(1, 3).contiguous()
        # Targeted squeeze(1) twice (not a blanket squeeze()) so this is correct
        # regardless of batch size -- a blanket squeeze() also drops the batch
        # dimension itself whenever a batch happens to contain exactly 1 sample
        # (e.g. a dataset size not evenly divisible by batch_size), which silently
        # breaks the op_mask_aux indexing below for that one batch.
        s_pred = s_pred.transpose(1, 3).contiguous().squeeze(1).squeeze(1)
        h_pred = h_pred.transpose(1, 3).contiguous().squeeze(1).squeeze(1)
        op_mask_aux = _squeeze_vessels(op_mask)

        s_pred = s_pred * 51.1
        real[:, 2, :, :] = real[:, 2, :, :] * 51.1
        s_pred[~op_mask_aux] = np.nan

        h_pred = h_pred * torch.pi
        real[:, 3, :, :] = real[:, 3, :, :] * torch.pi
        h_pred[~op_mask_aux] = np.nan

        output[:, 0, :, :] = output[:, 0, :, :] * (max_lat - min_lat) + min_lat
        output[:, 1, :, :] = output[:, 1, :, :] * (max_lon - min_lon) + min_lon
        real[:, 0, :, :] = real[:, 0, :, :] * (max_lat - min_lat) + min_lat
        real[:, 1, :, :] = real[:, 1, :, :] * (max_lon - min_lon) + min_lon

        ADE = self.loss(real[:, :2, :, :], output, mask=op_mask)
        sloss = self.alpha * self.loss2(real[:, 2, :, :], s_pred, null_val=np.nan)
        hloss = self.beta * self.loss2(real[:, 3, :, :], h_pred, null_val=np.nan)
        loss = ADE + sloss + hloss

        average_displacement = ais_acnet_utils.haversine_loss(output, real, mask=op_mask, mse=True)
        fde = ais_acnet_utils.haversine_loss(
            output[:, :, :, -1:], real[:, :2, :, -1:], mask=op_mask[:, :, -1:]
        ).item()

        if return_loss_terms:
            return loss.item(), ADE.item(), fde, average_displacement.item(), sloss.item(), hloss.item()
        return loss.item(), ADE.item(), fde, average_displacement.item()


def eval_per_voyage(engine, test_ds, device):
    """Per-sample (= per-voyage, since each AIS-ACNet sample is one voyage-phase)
    Haversine ADE in km, reusing FixedTrainer.eval()'s exact de-normalization
    logic (verbatim below) rather than re-deriving tensor indexing/unit
    conventions from scratch. Used for normalized-error reporting and the
    Wilcoxon signed-rank comparison against TrajRAG/Kalman/ARIMA/LSTM/BiLSTM/
    Numeric-kNN, which all report ADE in km."""
    loader = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=collate_function())
    voyage_ids = [s[1] for s in test_ds.samples]
    engine.model.eval()
    per_voyage_km = {}
    with torch.no_grad():
        for i, batch in enumerate(loader):
            x = batch[0].to(device).permute(0, 3, 1, 2)
            real = batch[1].to(device).permute(0, 3, 1, 2).clone()
            ip_mask = torch.ones_like(batch[2]).to(device)
            op_mask = batch[3].to(device)

            output, _, _ = engine.model(x, ip_mask, adj=None)
            output = output.transpose(1, 3).contiguous()

            output[:, 0, :, :] = output[:, 0, :, :] * (max_lat - min_lat) + min_lat
            output[:, 1, :, :] = output[:, 1, :, :] * (max_lon - min_lon) + min_lon
            real[:, 0, :, :] = real[:, 0, :, :] * (max_lat - min_lat) + min_lat
            real[:, 1, :, :] = real[:, 1, :, :] * (max_lon - min_lon) + min_lon

            mask = op_mask.squeeze(0).squeeze(0)       # (window,) bool
            pred_lat = output[0, 0, 0, :][mask]
            pred_lon = output[0, 1, 0, :][mask]
            true_lat = real[0, 0, 0, :][mask]
            true_lon = real[0, 1, 0, :][mask]
            if mask.sum().item() == 0:
                continue

            # NOTE: min_lat/max_lat/min_lon/max_lon (imported from geographic_utils)
            # are already in RADIANS (geographic_utils.py converts them at import
            # time), so pred_lat/true_lat/pred_lon/true_lon here are radians
            # already -- do NOT multiply by pi/180 again (that bug first produced
            # ADE values ~55x too small here).
            dlat = true_lat - pred_lat
            dlon = true_lon - pred_lon
            a = (torch.sin(dlat / 2) ** 2
                + torch.cos(pred_lat) * torch.cos(true_lat) * torch.sin(dlon / 2) ** 2)
            d_km = 2 * 6371.0 * torch.asin(torch.sqrt(torch.clamp(a, 0.0, 1.0)))
            per_voyage_km[voyage_ids[i]] = float(d_km.mean().item())
    return per_voyage_km


def run_epoch(engine, loader, device, train: bool):
    losses, ades, fdes = [], [], []
    for batch in loader:
        x = batch[0].to(device).permute(0, 3, 1, 2)       # (B, 4, vessels, window)
        y = batch[1].to(device).permute(0, 3, 1, 2)
        # NOT batch[2] (the real input mask): inside model.forward(), the output is
        # gated by ip_mask using the SAME time index as the input. Their original
        # design shares one time axis across input/output (multi-vessel presence);
        # ours doesn't (input is right-aligned observed data, output is left-aligned
        # future predictions) -- using the real mask here forces the model's output
        # to exactly zero at any index where that index's INPUT happened to be
        # padding, regardless of whether that index is a real PREDICTION target.
        # op_mask (below) already does the correct masking at the loss level, so
        # this is intentionally all-ones: nothing else in engine.py reads ip_mask.
        ip_mask = torch.ones_like(batch[2]).to(device)
        op_mask = batch[3].to(device)
        if train:
            loss, ade, fde = engine.train(x, y, ip_mask, op_mask)
        else:
            loss, ade, fde, _ = engine.eval(x, y, ip_mask, op_mask)
        losses.append(loss)
        ades.append(ade)
        fdes.append(fde)
    return float(np.mean(losses)), float(np.mean(ades)), float(np.mean(fdes))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=list(PHASE_WINDOWS))
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--nhid", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wdecay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--patience", type=int, default=12, help="early stopping patience")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    window = PHASE_WINDOWS[args.phase]
    phase_dir = HERE / "voyage_processed" / args.phase

    train_ds = torch.load(phase_dir / "train.pt", weights_only=False)
    val_ds = torch.load(phase_dir / "val.pt", weights_only=False)
    test_ds = torch.load(phase_dir / "test.pt", weights_only=False)
    print(f"[{args.phase}] window={window}  train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_function())
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_function())
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_function())

    blocks = PHASE_BLOCKS[args.phase]
    engine = FixedTrainer(4, window, args.nhid, args.dropout, args.lr, args.wdecay, device, blocks=blocks)
    n_params = sum(p.numel() for p in engine.model.parameters() if p.requires_grad)
    print(f"model parameters: {n_params:,}")

    best_val = float("inf")
    best_state = None
    epochs_without_improvement = 0
    ckpt_dir = HERE / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_ade, tr_fde = run_epoch(engine, train_loader, device, train=True)
        val_loss, val_ade, val_fde = run_epoch(engine, val_loader, device, train=False)

        improved = val_loss < best_val
        if improved:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in engine.model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print(f"epoch {epoch:03d}  train_loss={tr_loss:.4f} train_ADE={tr_ade:.4f}nm  "
              f"val_loss={val_loss:.4f} val_ADE={val_ade:.4f}nm{'  *' if improved else ''}")

        if epochs_without_improvement >= args.patience:
            print(f"early stopping at epoch {epoch} (no val improvement for {args.patience} epochs)")
            break

    print(f"training time: {time.time() - t0:.0f}s")

    engine.model.load_state_dict(best_state)
    test_loss, test_ade, test_fde = run_epoch(engine, test_loader, device, train=False)
    print(f"[{args.phase}] TEST  loss={test_loss:.4f}  ADE={test_ade:.4f} nm ({test_ade*NM_TO_KM:.4f} km)  "
          f"FDE={test_fde:.4f} nm ({test_fde*NM_TO_KM:.4f} km)")

    ckpt_path = ckpt_dir / f"{args.phase}_best.pt"
    torch.save({"model_state": best_state, "window": window, "phase": args.phase,
               "test_ade_nm": test_ade, "test_fde_nm": test_fde}, ckpt_path)
    print(f"saved best checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
