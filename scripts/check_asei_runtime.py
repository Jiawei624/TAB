"""Diagnose a few ASEI predictor batches without loading or training a benchmark dataset."""

import argparse
import json
import logging
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ts_benchmark.baselines.self_impl import ASEI
from ts_benchmark.baselines.self_impl.ASEI.model import Predictor, lag_graph
from ts_benchmark.baselines.self_impl.ASEI.progress import IMPLEMENTATION_VERSION, Progress


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", type=int, nargs="+", default=[1, 0])
    parser.add_argument("--device", choices=["auto", "cpu"], default="auto")
    parser.add_argument("--channels", type=int, default=51)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--win-size", type=int, default=63)
    parser.add_argument("--batches", type=int, default=2)
    parser.add_argument("--block-sizes", type=int, nargs="+", default=[1, 16])
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--log-interval", type=float, default=10)
    args = parser.parse_args()
    if min(args.channels, args.batch_size, args.win_size, args.batches, args.threads, *args.block_sizes) < 1:
        parser.error("Sizes and counts must be positive")
    if not args.gpus or min(args.gpus) < 0 or len(set(args.gpus)) != len(args.gpus):
        parser.error("--gpus must contain unique nonnegative IDs")
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, args.gpus))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    torch.set_num_threads(args.threads)
    logging.info("ASEI diagnostic implementation=%s torch=%s cuda_build=%s visible_devices=%s threads=%d",
                 IMPLEMENTATION_VERSION, torch.__version__, torch.version.cuda,
                 os.environ["CUDA_VISIBLE_DEVICES"], args.threads)
    for block_size in args.block_sizes:
        with Progress(f"runtime setup block_size={block_size}", interval=args.log_interval):
            torch.manual_seed(0)
            use_cuda = args.device != "cpu" and torch.cuda.is_available()
            detector = ASEI(device=args.device, multi_gpu=use_cuda and len(args.gpus) > 1,
                            device_ids=list(range(len(args.gpus))), batch_size=args.batch_size,
                            win_size=args.win_size, predictor_block_size=block_size,
                            log_interval_seconds=args.log_interval)
            length = detector.history + args.batch_size * args.batches
            values = np.random.default_rng(0).normal(size=(length, args.channels)).astype(np.float32)
            detector.columns_ = list(range(args.channels))
            cfg = detector.config
            graph = lag_graph(values, cfg.graph_lags, cfg.graph_neighbors)
            detector.predictor = detector._wrap(Predictor(
                args.channels, *graph, hidden=cfg.hidden_dim, embedding=cfg.embedding_dim,
                kernel=cfg.kernel_size, dilations=cfg.dilations, win_size=cfg.win_size))
            loader = detector._predictor_loader(values)
            optimizer = torch.optim.AdamW(detector.predictor.parameters(), lr=cfg.learning_rate,
                                          weight_decay=cfg.weight_decay)
            if use_cuda:
                logging.info("ASEI diagnostic GPU names=%s",
                             [torch.cuda.get_device_name(index) for index in detector.device_ids or [0]])
        if use_cuda:
            torch.cuda.synchronize(detector.device)
        started = time.monotonic()
        loss = detector._predictor_loss(loader, optimizer, epoch="diagnostic")
        if use_cuda:
            torch.cuda.synchronize(detector.device)
        elapsed = time.monotonic() - started
        result = dict(implementation=IMPLEMENTATION_VERSION, torch=torch.__version__,
                      device=str(detector.device), block_size=block_size, channels=args.channels,
                      W=detector.win_size, batches=len(loader), seconds=elapsed,
                      target_points_per_second=(length - detector.history) / elapsed, loss=loss)
        if use_cuda:
            result["visible_gpus"] = [torch.cuda.get_device_name(index) for index in detector.device_ids or [0]]
        print(json.dumps(result), flush=True)
        del loader, optimizer, detector


if __name__ == "__main__":
    main()
