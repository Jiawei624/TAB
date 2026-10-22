"""Per-segment deterministic observations; no learned states or raw window cube.

Rolling scale m is indexed by its start s. Window w reads s=w+q*m, so every
window retains its own right-aligned block grid. Statistics use direct local
detrending in float64, after the training energy floors have been fixed.
"""

import shutil
import uuid
from pathlib import Path

import numpy as np
import torch


class FeatureCache:
    def __init__(self, values, config, energy_floor, directory, neighborhood=True):
        self.config, self.energy_floor = config, np.asarray(energy_floor)
        self.length, self.channels = values.shape
        self.count = self.length - config.win_size + 1
        self.mode, self.directory = config.feature_cache, Path(directory)
        self.scales = config.scales if neighborhood else ()
        self.arrays, self.specifications, self.paths = {}, {}, {}
        self.owner, self.ready = True, False
        self.token = uuid.uuid4().hex
        # Bound temporary detrending arrays independently of the dataset size.
        working_bytes = max(1, self.channels * max(config.spans) * 8 * 6)
        self.chunk_size = min(config.reference_chunk_size, max(1, 64 * 2**20 // working_bytes))
        shapes = {"coarse_{}".format(m): (self.length - m + 1, self.channels, 2) for m in self.scales}
        shapes["statistics"] = (self.count, self.channels, len(config.spans), 3)
        self.specifications = shapes
        self.size_bytes = sum(int(np.prod(shape)) * 8 for shape in shapes.values())
        self.total_work = sum(shape[0] for shape in shapes.values())
        if self.mode == "disk" and shutil.disk_usage(self.directory).free < self.size_bytes:
            raise OSError("Not enough disk space for {} bytes of observation cache".format(self.size_bytes))

    def _allocate(self, name):
        shape = self.specifications[name]
        if self.mode == "memory":
            array = np.empty(shape, dtype=np.float64)
        elif self.mode == "disk":
            path = self.directory / (".features_{}_{}".format(self.token, name))
            self.paths[name] = str(path)
            array = np.memmap(path, mode="w+", dtype=np.float64, shape=shape)
        else:
            raise ValueError("A cache requires feature_cache=memory or disk")
        self.arrays[name] = array
        return array

    def build(self, values, progress=None):
        completed = 0
        for m in self.scales:
            destination = self._allocate("coarse_{}".format(m))
            denominator = np.abs(np.linspace(-1, 1, m)).sum()
            for start in range(0, len(destination), min(self.chunk_size, 2048)):
                size = min(min(self.chunk_size, 2048), len(destination) - start)
                # Restart prefixes frequently; never subtract sequence-length prefixes.
                segment = values[start:start + size + m - 1]
                anchor = segment[0]
                centered = segment - anchor
                zeros = np.zeros((1, self.channels), dtype=np.float64)
                sums = np.concatenate((zeros, centered.cumsum(axis=0)))
                weighted = np.concatenate((zeros, (centered * np.arange(len(segment))[:, None]).cumsum(axis=0)))
                begins = np.arange(size)
                ends = begins + m
                total = sums[ends] - sums[begins]
                local = weighted[ends] - weighted[begins] - begins[:, None] * total
                destination[start:start + size, :, 0] = anchor + total / m
                destination[start:start + size, :, 1] = (2 * local / (m - 1) - total) / denominator
                if progress is not None:
                    progress(completed + start + size, "scale={} mode={}".format(m, self.mode))
            completed += len(destination)
        destination = self._allocate("statistics")
        blocks = [np.lib.stride_tricks.sliding_window_view(values, h, axis=0)[self.config.win_size - h:]
                  for h in self.config.spans]
        for start in range(0, self.count, self.chunk_size):
            stop = min(start + self.chunk_size, self.count)
            for index, (h, windows) in enumerate(zip(self.config.spans, blocks)):
                block = windows[start:stop]
                u = np.linspace(-1, 1, h)
                slope = np.einsum("ndh,h->nd", block, u) / np.square(u).sum()
                residual = block - block.mean(axis=-1, keepdims=True) - slope[..., None] * u
                energy = np.square(residual).sum(axis=-1)
                floor = self.energy_floor[:, index]
                denominator = energy + (h - 2) * floor
                destination[start:stop, :, index, 0] = np.log(energy / (h - 2) + floor)
                destination[start:stop, :, index, 1] = (residual[..., 1:] * residual[..., :-1]).sum(axis=-1) / denominator
                destination[start:stop, :, index, 2] = np.power(residual, 4).sum(axis=-1) / np.square(denominator)
            if progress is not None:
                progress(completed + stop, "statistics mode={}".format(self.mode))
        for array in self.arrays.values():
            if isinstance(array, np.memmap):
                array.flush()
        self.ready = True

    def gather(self, indices):
        if not self.ready:
            raise RuntimeError("Observation cache is incomplete")
        coarse = tuple(torch.from_numpy(self.arrays["coarse_{}".format(m)][
            indices[:, None] + (np.arange(self.config.win_size // m) * m)[None]].transpose(0, 2, 1, 3))
                       for m in self.scales)
        return {"coarse": coarse, "statistics": torch.from_numpy(self.arrays["statistics"][indices])}

    def __getstate__(self):
        state = self.__dict__.copy()
        state["owner"] = False
        if self.mode == "disk":
            # Pickling a memmap itself can serialize its whole contents on Windows.
            state["arrays"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if self.arrays is None:
            self.arrays = {name: np.memmap(path, mode="r", dtype=np.float64, shape=self.specifications[name])
                           for name, path in self.paths.items()}

    def close(self):
        self.ready = False
        for array in self.arrays.values():
            if isinstance(array, np.memmap):
                array._mmap.close()
        self.arrays.clear()
        if self.owner:
            for path in self.paths.values():
                Path(path).unlink(missing_ok=True)
