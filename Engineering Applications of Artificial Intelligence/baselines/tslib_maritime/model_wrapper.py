"""
Builds any of the 7 TSLib forecasters for the maritime baseline suite, reusing
the SAME cloned TSLib checkout GPS-SPOOFING-CORRECTION already points at
(read-only; nothing in GPS-SPOOFING-CORRECTION/ or TSLib/ is modified).

DLinear / PatchTST / iTransformer reuse aviation's own Configs shape as-is.
Informer / Autoformer / FEDformer / TimesNet need 3 extra fields TSLib reads
that aviation's Configs dataclass doesn't define (distil, num_kernels, top_k) --
added here, in our own dataclass, rather than editing the original file.
"""
import os
import sys
from dataclasses import dataclass

import torch

CORR_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..",
    "GPS-SPOOFING-CORRECTION",
)
TSLIB_PATH = os.environ.get("TSLIB_PATH", os.path.join(CORR_DIR, "TSLib"))
if TSLIB_PATH not in sys.path:
    sys.path.insert(0, TSLIB_PATH)

SUPPORTED = ["DLinear", "PatchTST", "iTransformer",
             "Informer", "Autoformer", "FEDformer", "TimesNet"]
ENCODER_DECODER_MODELS = {"Informer", "Autoformer", "FEDformer", "TimesNet"}


@dataclass
class MaritimeConfigs:
    task_name: str = "long_term_forecast"
    seq_len: int = 4
    label_len: int = 4
    pred_len: int = 4
    enc_in: int = 8
    dec_in: int = 8
    c_out: int = 8

    d_model: int = 128
    n_heads: int = 4
    e_layers: int = 2
    d_layers: int = 1
    d_ff: int = 256
    factor: int = 1
    dropout: float = 0.1
    activation: str = "gelu"

    embed: str = "timeF"
    freq: str = "t"

    moving_avg: int = 3
    output_attention: bool = False
    num_class: int = 0

    patch_len: int = 2
    stride: int = 1

    # extra fields needed by Informer / TimesNet only
    distil: bool = True
    num_kernels: int = 6
    top_k: int = 2

    def for_model(self, name: str) -> "MaritimeConfigs":
        if name == "DLinear":
            k = min(self.moving_avg, self.seq_len)
            if k % 2 == 0:
                k -= 1
            self.moving_avg = max(k, 3)
        if name == "PatchTST":
            self.patch_len = min(self.patch_len, self.seq_len)
            self.stride = max(1, self.patch_len // 2)
        return self


def build_model(name: str, configs: MaritimeConfigs):
    configs.for_model(name)
    if name == "iTransformer":
        from models.iTransformer import Model
        return Model(configs)
    elif name == "PatchTST":
        from models.PatchTST import Model
        patch_len = min(configs.patch_len, configs.seq_len)
        stride = max(1, patch_len // 2)
        return Model(configs, patch_len=patch_len, stride=stride)
    elif name == "DLinear":
        from models.DLinear import Model
        return Model(configs)
    elif name == "Informer":
        from models.Informer import Model
        return Model(configs)
    elif name == "Autoformer":
        from models.Autoformer import Model
        return Model(configs)
    elif name == "FEDformer":
        from models.FEDformer import Model
        return Model(configs)
    elif name == "TimesNet":
        from models.TimesNet import Model
        return Model(configs)
    else:
        raise ValueError(f"Unknown model '{name}'. Supported: {SUPPORTED}")


def make_decoder_inputs(x_batch, label_len, pred_len, device):
    """TSLib's own convention (exp/exp_long_term_forecasting.py):
    dec_inp = cat([label_slice, zeros(pred_len)], dim=1), where the label slice
    is the TAIL OF THE OBSERVED INPUT (warm-up context for the decoder) --
    NOT a slice of the target/future. Using the future here would leak the
    evaluation target directly into the model's input. We use label_len ==
    seq_len (the whole observed window) as that tail."""
    label_slice = x_batch[:, -label_len:, :]
    zeros = x_batch.new_zeros((x_batch.shape[0], pred_len, x_batch.shape[2]))
    dec_inp = torch.cat([label_slice, zeros], dim=1).to(device)
    return dec_inp


def make_time_marks(batch_size, seq_len, device, freq_cols: int = 5):
    """freq='t' (minute-level) -> TimeFeatureEmbedding expects 5 columns
    (TSLib/layers/Embed.py freq_map). We have no real calendar features for
    these short relative-time windows, so we pass zeros: a documented
    simplification (constant bias term), not a missing-data bug."""
    import torch
    return torch.zeros(batch_size, seq_len, freq_cols, device=device)
