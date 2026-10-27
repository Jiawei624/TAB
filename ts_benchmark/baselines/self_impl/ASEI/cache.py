"""Entity-local fixed validation evidence, with a bounded device cache and disk spill."""

from pathlib import Path
import tempfile

import numpy as np
import torch


class ValidationCache:
    def __init__(self, device, megabytes):
        self.device = device
        self.budget = int(megabytes * 1024 ** 2)
        self.used = 0
        self.entries = []
        self.directory = tempfile.TemporaryDirectory(prefix="asei_validation_")

    def append(self, features, metadata):
        features = features.detach()
        size = features.numel() * features.element_size()
        if self.used + size <= self.budget:
            location = features
            self.used += size
        else:
            location = Path(self.directory.name) / f"{len(self.entries)}.npy"
            np.save(location, features.cpu().numpy(), allow_pickle=False)
        self.entries.append((location, metadata))

    def get(self, index):
        location, metadata = self.entries[index]
        if isinstance(location, Path):
            features = torch.from_numpy(np.load(location, allow_pickle=False))
            if self.device.type == "cuda":
                features = features.pin_memory()
            features = features.to(self.device, non_blocking=True)
        else:
            features = location
        return features, metadata

    def close(self):
        self.entries.clear()
        self.directory.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
