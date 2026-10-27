"""Bounded FP32 evidence storage with verified, atomic disk spill."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import numpy as np
import torch
from .auxiliary import objective_to_device


class EvidenceBank:
    def __init__(self, device, device_mb=256, host_mb=1024, cache_dir=None, cache_key=""):
        self.device = torch.device(device)
        self.device_limit, self.host_limit = int(device_mb*1024**2), int(host_mb*1024**2)
        self.device_bytes = self.host_bytes = self.disk_bytes = 0
        self.entries, self.cache_key, self.sealed = [], cache_key, False
        root = Path.cwd() if cache_dir is None else Path(cache_dir)
        root.mkdir(parents=True,exist_ok=True)
        self._temporary = tempfile.TemporaryDirectory(prefix=".asei-bank-",dir=str(root))
        self.directory = Path(self._temporary.name)

    def append(self, evidence, metadata, task):
        if self.sealed:
            raise RuntimeError("Evidence bank is immutable after seal")
        evidence = evidence.detach().to(dtype=torch.float32).contiguous()
        nbytes = evidence.numel()*4
        digest = hashlib.sha256(evidence.cpu().numpy().tobytes()).hexdigest()
        if self.device.type=="cuda" and self.device_bytes+nbytes<=self.device_limit:
            storage = evidence.to(self.device).clone()
            self.device_bytes += nbytes
        elif self.host_bytes+nbytes<=self.host_limit:
            storage = evidence.to("cpu").clone()
            if self.device.type=="cuda":
                storage = storage.pin_memory()
            self.host_bytes += nbytes
        else:
            path = self.directory/f"{len(self.entries):05d}.npy"
            temporary = path.with_suffix(".partial")
            with temporary.open("wb") as output:
                np.save(output,evidence.cpu().numpy(),allow_pickle=False)
            os.replace(str(temporary),str(path))
            storage = path
            self.disk_bytes += nbytes
        self.entries.append(dict(storage=storage,shape=tuple(evidence.shape),sha256=digest,
                                 metadata=objective_to_device([metadata],self.device)[0],task=task))

    def seal(self):
        manifest = dict(cache_key=self.cache_key,complete=True,entries=[
            dict(shape=item["shape"],sha256=item["sha256"]) for item in self.entries])
        temporary = self.directory/"manifest.partial"
        temporary.write_text(json.dumps(manifest,sort_keys=True),encoding="utf-8")
        os.replace(str(temporary),str(self.directory/"manifest.json"))
        self.sealed = True

    def verify_key(self, expected):
        if not self.sealed or expected!=self.cache_key:
            raise ValueError("Incomplete evidence bank or cache key mismatch")

    def get(self,index):
        self.verify_key(self.cache_key)
        item = self.entries[index]
        storage = item["storage"]
        if isinstance(storage,Path):
            try:
                values = np.load(storage,mmap_mode="r",allow_pickle=False)
                if (values.dtype!=np.float32 or tuple(values.shape)!=item["shape"]
                        or hashlib.sha256(values.tobytes()).hexdigest()!=item["sha256"]):
                    raise ValueError("shape, dtype or checksum mismatch")
                storage = torch.from_numpy(np.array(values,copy=True))
            except (OSError,ValueError) as error:
                raise ValueError(f"Corrupt ASEI evidence cache: {storage}") from error
        return storage.to(self.device,non_blocking=True), item["metadata"]

    def batch(self,indices):
        items = [self.get(index) for index in indices]
        return torch.stack([item[0] for item in items]), [item[1] for item in items]

    def batches(self,batch_size,rng=None):
        buckets = {}
        for i,item in enumerate(self.entries):
            buckets.setdefault(item["shape"][1],[]).append(i)
        batches = []
        for length in sorted(buckets):
            indices = buckets[length]
            if rng is not None:
                indices = rng.permutation(indices).tolist()
            batches.extend(indices[i:i+batch_size] for i in range(0,len(indices),batch_size))
        if rng is not None:
            batches = [batches[int(i)] for i in rng.permutation(len(batches))]
        return batches

    def statistics(self):
        return dict(pairs=len(self.entries),device_bytes=self.device_bytes,host_bytes=self.host_bytes,
                    disk_bytes=self.disk_bytes,cache_key=self.cache_key,dtype="float32")

    def close(self):
        self.entries.clear()
        self._temporary.cleanup()