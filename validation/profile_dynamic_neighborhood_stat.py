"""Short, reproducible runtime diagnosis; never a benchmark-quality experiment."""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from ts_benchmark.baselines.self_impl.dynamic_neighborhood_stat.config import Config
from ts_benchmark.baselines.self_impl.dynamic_neighborhood_stat.features import training_references
from ts_benchmark.baselines.self_impl.dynamic_neighborhood_stat.model import DualPathNetwork


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channels", nargs="+", type=int, default=[3, 51])
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--disable-cudnn", action="store_true", help="Compare against TAB's inherited cuDNN-disabled setting")
    parser.add_argument("--cpu-threads", type=int, default=3)
    parser.add_argument("--output", default="result/DynamicNeighborhoodStat/diagnosis/profile.json")
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 1 or args.cpu_threads < 1 or any(channels < 2 for channels in args.channels):
        parser.error("steps, batch size and CPU threads must be positive; channels must be at least two")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    torch.set_num_threads(args.cpu_threads)
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    report = {"torch": torch.__version__, "device": str(device), "cpu_threads": torch.get_num_threads(),
              "batch_size": args.batch_size, "steps": args.steps, "cudnn_available": torch.backends.cudnn.is_available(),
              "cudnn_enabled": not args.disable_cudnn, "measurements": []}
    if device.type == "cuda":
        report["gpu"] = torch.cuda.get_device_name(device)
    print(json.dumps({key: value for key, value in report.items() if key != "measurements"}), flush=True)
    for channels in args.channels:
        for recompute in (True, False):
            torch.manual_seed(42)
            config = Config(batch_size=args.batch_size, gradient_checkpointing=recompute)
            values = np.random.default_rng(42).normal(size=(config.win_size + args.batch_size, channels))
            start = time.perf_counter()
            refs = training_references(values, config)
            reference_seconds = time.perf_counter() - start
            model = DualPathNetwork(channels, config, refs).to(device).train()
            windows = torch.from_numpy(np.stack([values[i:i + config.win_size] for i in range(args.batch_size)])).to(device)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            # Include one warmup, then time full forward + loss + backward steps.
            timings, forward_timings = [], []
            for step in range(args.steps + 1):
                model.zero_grad(set_to_none=True)
                synchronize(device)
                start = time.perf_counter()
                with torch.backends.cudnn.flags(enabled=not args.disable_cudnn, benchmark=False, deterministic=True):
                    output = model(windows)
                    loss = model.losses(windows, output)["loss"]
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError("Nonfinite profile loss")
                    synchronize(device)
                    forward_seconds = time.perf_counter() - start
                    loss.backward()
                synchronize(device)
                elapsed = time.perf_counter() - start
                if step:
                    timings.append(elapsed)
                    forward_timings.append(forward_seconds)
                print("channels={} checkpointing={} step={}/{}{} elapsed={:.3f}s".format(
                    channels, recompute, step, args.steps, " warmup" if step == 0 else "", elapsed), flush=True)
            measurement = {"channels": channels, "gradient_checkpointing": recompute,
                           "reference_seconds": reference_seconds,
                           "step_seconds": timings, "mean_step_seconds": float(np.mean(timings)),
                           "mean_forward_seconds": float(np.mean(forward_timings)),
                           "windows_per_second": args.batch_size / float(np.mean(timings))}
            if device.type == "cuda":
                measurement["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
            # Arithmetic extrapolation for context only; server hardware will differ.
            if channels == 51 and config.win_size == 128:
                measurement["swat_train_batches_per_epoch"] = (396000 - 128 + 1 + args.batch_size - 1) // args.batch_size
                measurement["swat_train_only_seconds_per_epoch_estimate"] = measurement["swat_train_batches_per_epoch"] * measurement["mean_step_seconds"]
            report["measurements"].append(measurement)
            print(json.dumps(measurement), flush=True)
            del loss, output, model, windows
    path = Path(args.output)
    if not path.is_absolute():
        path = ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Profile saved: " + str(path), flush=True)


if __name__ == "__main__":
    main()
