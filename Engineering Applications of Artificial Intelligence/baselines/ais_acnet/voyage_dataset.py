"""Shared Dataset class used by build_voyage_windows.py and train_voyage_model.py.
Kept in its own module (not __main__) so pickled .pt files can be loaded from
either script without an import error."""
from torch.utils.data import Dataset


class VoyageWindowDataset(Dataset):
    def __init__(self, samples: list):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx][2]
