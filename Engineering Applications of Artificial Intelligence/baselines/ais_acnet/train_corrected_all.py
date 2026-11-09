"""
Task B: train AIS-ACNet on CORRECTED data (both corridors, all 3 phases),
save full per-step predictions (not just aggregate), and evaluate on three
voyage subsets: the full 125-voyage official test set, the smaller
main_reconstruction_voyage_ids subset this rerun's TrajRAG was actually
evaluated on, and (Houston only) the item4_matched_short_window subset with
step-truncation, for the item-4 matched comparison against the forecasters.

Houston and LA/Long Beach each use their OWN trainer class and bounding box
(train_voyage_model.py's FixedTrainer + geographic_utils bbox for Houston;
train_voyage_model_lalb.py's FixedTrainerLALB + its own hardcoded LA/Long
Beach bbox) -- never mixed, per that file's own documented reason (reusing
Houston's bbox for LA/Long Beach silently de-normalizes into the wrong
coordinate range).

Epochs/patience kept at 60/12 (the scripts' own defaults) rather than the
paper's reported 150/20, to fit this diagnostic's time budget -- flagged in
the final report.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
EAAI_DIR = HERE.parents[1]
FINAL_RERUN_DIR = EAAI_DIR / "experiments" / "final_rerun"
EXTERNAL_REPO = HERE.parents[1] / "external" / "AIS-ACNet" / "AIS-ACNet"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(EXTERNAL_REPO))

from data import collate_function  # noqa: E402
from voyage_dataset import VoyageWindowDataset  # noqa: E402,F401

PHASES = ["departure", "transit", "arrival"]
DEVICE = torch.device("cpu")
OUT_DIR = FINAL_RERUN_DIR / "ais_acnet_corrected"
OUT_DIR.mkdir(exist_ok=True, parents=True)


def eval_per_voyage_detailed(engine, test_ds, device, bbox, max_steps=None):
    min_lat, max_lat, min_lon, max_lon = bbox
    loader = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=collate_function())
    voyage_ids = [s[1] for s in test_ds.samples]
    engine.model.eval()
    per_voyage_km = {}
    per_voyage_points = {}
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

            mask = op_mask.squeeze(0).squeeze(0)
            idx = mask.nonzero(as_tuple=True)[0]
            if max_steps is not None:
                idx = idx[:max_steps]
            if idx.numel() == 0:
                continue
            pred_lat = output[0, 0, 0, idx]
            pred_lon = output[0, 1, 0, idx]
            true_lat = real[0, 0, 0, idx]
            true_lon = real[0, 1, 0, idx]

            dlat = true_lat - pred_lat
            dlon = true_lon - pred_lon
            a = (torch.sin(dlat / 2) ** 2
                + torch.cos(pred_lat) * torch.cos(true_lat) * torch.sin(dlon / 2) ** 2)
            d_km = 2 * 6371.0 * torch.asin(torch.sqrt(torch.clamp(a, 0.0, 1.0)))
            per_voyage_km[voyage_ids[i]] = float(d_km.mean().item())
            per_voyage_points[voyage_ids[i]] = {
                "predicted": list(zip(torch.rad2deg(pred_lat).tolist(), torch.rad2deg(pred_lon).tolist())),
                "ground_truth": list(zip(torch.rad2deg(true_lat).tolist(), torch.rad2deg(true_lon).tolist())),
            }
    return per_voyage_km, per_voyage_points


def train_one(corridor, phase, processed_dir, windows, blocks, trainer_cls, run_epoch_fn):
    phase_dir = processed_dir / phase
    train_ds = torch.load(phase_dir / "train.pt", weights_only=False)
    val_ds = torch.load(phase_dir / "val.pt", weights_only=False)
    test_ds = torch.load(phase_dir / "test.pt", weights_only=False)
    print(f"[{corridor}/{phase}] train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_function())
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=collate_function())

    window = windows[phase]
    engine = trainer_cls(4, window, 32, 0.3, 1e-3, 1e-4, DEVICE, blocks=blocks[phase])

    best_val = float("inf")
    best_state = None
    no_improve = 0
    t0 = time.time()
    for epoch in range(1, 151):
        run_epoch_fn(engine, train_loader, DEVICE, train=True)
        val_loss, _, _ = run_epoch_fn(engine, val_loader, DEVICE, train=False)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in engine.model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= 20:
            print(f"  early stop at epoch {epoch}")
            break
    print(f"  trained in {time.time()-t0:.0f}s, best_val={best_val:.4f}")
    engine.model.load_state_dict(best_state)
    return engine, test_ds


def run_corridor(corridor, processed_dir, windows, blocks, trainer_cls, run_epoch_fn, bbox, manifest, item4=None):
    results = {}
    for phase in PHASES:
        engine, test_ds = train_one(corridor, phase, processed_dir, windows, blocks, trainer_cls, run_epoch_fn)

        full_km, full_pts = eval_per_voyage_detailed(engine, test_ds, DEVICE, bbox)
        main_ids = set(manifest[corridor]["main_reconstruction_voyage_ids"])
        matched_km = {vid: v for vid, v in full_km.items() if vid in main_ids}

        entry = {
            "full125": {"n": len(full_km), "mean_km": round(float(np.mean(list(full_km.values()))), 4)},
            "matched": {"n": len(matched_km), "mean_km": round(float(np.mean(list(matched_km.values()))), 4) if matched_km else None},
        }

        if corridor == "HOU-SHIPCHANNEL" and item4 and phase in item4:
            window_steps = item4[phase]["window"]
            item4_ids = set(item4[phase]["voyage_ids"])
            item4_km, item4_pts = eval_per_voyage_detailed(engine, test_ds, DEVICE, bbox, max_steps=window_steps)
            item4_km_matched = {vid: v for vid, v in item4_km.items() if vid in item4_ids}
            entry["item4"] = {"n": len(item4_km_matched), "mean_km": round(float(np.mean(list(item4_km_matched.values()))), 4) if item4_km_matched else None}
            with open(OUT_DIR / f"item4_points_{phase}.json", "w", encoding="utf-8") as f:
                json.dump({vid: item4_pts[vid] for vid in item4_km_matched}, f)

        results[phase] = entry
        with open(OUT_DIR / f"points_{corridor}_{phase}.json", "w", encoding="utf-8") as f:
            json.dump(full_pts, f)
        with open(OUT_DIR / f"per_voyage_{corridor}_{phase}.json", "w", encoding="utf-8") as f:
            json.dump(full_km, f)
        print(f"[{corridor}/{phase}] {entry}")
    return results


def main():
    with open(FINAL_RERUN_DIR / "voyage_manifest.json", "r", encoding="utf-8") as f:
        manifest = json.load(f)
    item4 = manifest["HOU-SHIPCHANNEL"].get("item4_matched_short_window")

    import train_voyage_model as tvm
    from geographic_utils import min_lat as h_minlat, max_lat as h_maxlat, min_lon as h_minlon, max_lon as h_maxlon
    from build_voyage_windows import PHASE_WINDOWS as HOU_WINDOWS, PHASE_BLOCKS as HOU_BLOCKS

    all_results = {}
    all_results["HOU-SHIPCHANNEL"] = run_corridor(
        "HOU-SHIPCHANNEL", HERE / "voyage_processed_corrected", HOU_WINDOWS, HOU_BLOCKS,
        tvm.FixedTrainer, tvm.run_epoch, (h_minlat, h_maxlat, h_minlon, h_maxlon), manifest, item4=item4,
    )

    import train_voyage_model_lalb as tvml
    from build_voyage_windows_lalb import PHASE_WINDOWS as LALB_WINDOWS, PHASE_BLOCKS as LALB_BLOCKS
    all_results["LA-LONGBEACH"] = run_corridor(
        "LA-LONGBEACH", HERE / "voyage_processed_lalb_corrected", LALB_WINDOWS, LALB_BLOCKS,
        tvml.FixedTrainerLALB, tvml.run_epoch,
        (tvml.min_lat, tvml.max_lat, tvml.min_lon, tvml.max_lon), manifest,
    )

    with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print("\nDone.")
    print(json.dumps(all_results, indent=2))


if __name__ == "__main__":
    main()
