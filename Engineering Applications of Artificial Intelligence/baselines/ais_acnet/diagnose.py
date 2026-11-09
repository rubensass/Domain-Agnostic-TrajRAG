import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
EXTERNAL_REPO = HERE.parents[1] / "external" / "AIS-ACNet" / "AIS-ACNet"
sys.path.insert(0, str(EXTERNAL_REPO))

from data import collate_function  # noqa: E402

from voyage_dataset import VoyageWindowDataset  # noqa: E402,F401
from train_voyage_model import FixedTrainer, PHASE_WINDOWS, PHASE_BLOCKS  # noqa: E402

phase = "departure"
window = PHASE_WINDOWS[phase]
blocks = PHASE_BLOCKS[phase]
device = torch.device("cpu")

ckpt = torch.load(HERE / "checkpoints" / f"{phase}_best.pt", weights_only=False)
engine = FixedTrainer(4, window, 32, 0.3, 1e-3, 1e-4, device, blocks=blocks)
engine.model.load_state_dict(ckpt["model_state"])
engine.model.eval()

test_ds = torch.load(HERE / "voyage_processed" / phase / "test.pt", weights_only=False)
loader = DataLoader(test_ds, batch_size=8, shuffle=False, collate_fn=collate_function())
batch = next(iter(loader))

print("raw collated shapes:", [b.shape for b in batch[:4]])

x_raw = batch[0]           # (batch, vessels=1, window, 4) normalized [0,1]
y_raw = batch[1]
ip_mask = batch[2]
op_mask = batch[3]

x = x_raw.to(device).permute(0, 3, 1, 2)
with torch.no_grad():
    output, s_pred, h_pred = engine.model(x, ip_mask.to(device), adj=None)
output = output.transpose(1, 3).contiguous()  # (batch, 2, vessels=1, window) normalized [0,1]

print(f"\ninput_mask sum per sample:  {ip_mask[:, 0, :].sum(dim=1).tolist()}")
print(f"output_mask sum per sample: {op_mask[:, 0, :].sum(dim=1).tolist()}")

for i in range(min(3, x.size(0))):
    print(f"\n--- sample {i} (normalized [0,1] space) ---")
    print("t   valid?  real_lat  real_lon   pred_lat  pred_lon")
    for t in range(window):
        valid = bool(op_mask[i, 0, t].item())
        real_lat = y_raw[i, 0, t, 0].item()
        real_lon = y_raw[i, 0, t, 1].item()
        pred_lat = output[i, 0, 0, t].item()
        pred_lon = output[i, 1, 0, t].item()
        print(f"{t:2d}  {valid!s:5}  {real_lat:8.4f}  {real_lon:8.4f}   {pred_lat:8.4f}  {pred_lon:8.4f}")
