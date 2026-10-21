"""Two tiny epochs using three real SWAT channels; no detection-quality claims.

Read selected contiguous ranges directly from TAB's variable-major CSV, without
loading the approximately 1 GB long table or changing the registered dataset.
"""

import argparse
from collections import deque
from itertools import islice
import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from ts_benchmark.baselines.self_impl.dynamic_neighborhood_stat import DynamicNeighborhoodStat
from ts_benchmark.utils.random_utils import fix_all_random_seed


def read_swat_sample(path, time_steps, official_train, channels, train_start, train_points, test_points):
    train, test, names = [], [], []
    if train_start < 0 or train_start + train_points > official_train or official_train + test_points > time_steps:
        raise ValueError("Requested ranges must stay within the official train/test segments")
    with path.open("rb") as stream:
        if stream.readline().strip() != b"date,data,cols":
            raise ValueError("This small-data reader expects SWAT's date,data,cols CSV format")
        for channel in range(channels):
            deque(islice(stream, train_start), maxlen=0)
            train_rows = [stream.readline().decode("utf-8").strip().split(",") for _ in range(train_points)]
            deque(islice(stream, official_train - train_start - train_points), maxlen=0)
            test_rows = [stream.readline().decode("utf-8").strip().split(",") for _ in range(test_points)]
            name = train_rows[0][2]
            if name == "label" or any(len(row) != 3 or row[2] != name for row in train_rows + test_rows):
                raise ValueError("Unexpected variable boundary in the SWAT sample")
            np.testing.assert_array_equal([int(row[0]) for row in train_rows],
                                          np.arange(train_start + 1, train_start + train_points + 1))
            np.testing.assert_array_equal([int(row[0]) for row in test_rows],
                                          np.arange(official_train + 1, official_train + test_points + 1))
            train.append([float(row[1]) for row in train_rows])
            test.append([float(row[1]) for row in test_rows])
            names.append(name)
            if channel + 1 < channels:
                deque(islice(stream, time_steps - official_train - test_points), maxlen=0)
    return pd.DataFrame(np.asarray(train).T, columns=names), pd.DataFrame(np.asarray(test).T, columns=names)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--train-start", type=int, default=50000)
    parser.add_argument("--train-points", type=int, default=800)
    parser.add_argument("--test-points", type=int, default=192)
    parser.add_argument("--output", default="result/DynamicNeighborhoodStat/diagnosis/swat_smoke")
    args = parser.parse_args()
    if not 2 <= args.channels <= 51:
        parser.error("SWAT smoke channels must be between 2 and 51")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    torch.set_num_threads(3)
    started = time.perf_counter()
    metadata = pd.read_csv(ROOT / "dataset/anomaly_detect/DETECT_META.csv").set_index("file_name").loc["SWAT.csv"]
    train, test = read_swat_sample(ROOT / "dataset/anomaly_detect/data/SWAT.csv", int(metadata["time_steps"]),
                                   int(metadata["train_lens"]), args.channels, args.train_start,
                                   args.train_points, args.test_points)
    load_seconds = time.perf_counter() - started
    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    # Reproduce the exact global setting used by scripts/run_benchmark.py.
    fix_all_random_seed(42)
    detector = DynamicNeighborhoodStat(num_epochs=2, device=args.device, artifact_dir=str(output),
                                       reuse_checkpoint=False, log_interval_seconds=1.0)
    detector.set_run_context("SWAT_small_real_sample", 42)
    started = time.perf_counter()
    detector.detect_fit(train)
    fit_seconds = time.perf_counter() - started
    started = time.perf_counter()
    scores, _ = detector.detect_score(test)
    inference_seconds = time.perf_counter() - started
    alarms, _ = detector.detect_label(test)
    restored = DynamicNeighborhoodStat.load(detector.checkpoint_path_, device=args.device)
    restored_scores, _ = restored.detect_score(test)
    np.testing.assert_allclose(scores, restored_scores, rtol=1e-12, atol=1e-12, equal_nan=True)
    assert np.isnan(scores[:127]).all()
    assert np.isfinite(scores[127:]).all()
    assert detector.evaluation_valid_mask_.sum() == len(test) - 127
    assert len(detector.training_history_) == 2
    report = {"source": "SWAT.csv, real observations, selected channels only", "columns": list(train.columns),
              "official_train_range": [args.train_start, args.train_start + len(train)],
              "official_test_range": [0, len(test)], "train_shape": list(train.shape), "test_shape": list(test.shape),
              "device": str(detector.device_), "load_seconds": load_seconds, "fit_seconds": fit_seconds,
              "inference_seconds": inference_seconds, "epochs": len(detector.training_history_),
              "valid_points": int(detector.evaluation_valid_mask_.sum()), "threshold": detector.threshold_,
              "alarms": int(next(iter(alarms.values())).sum()), "checkpoint_roundtrip": "passed",
              "inherited_cudnn_restored": not torch.backends.cudnn.enabled,
              "artifacts": str(detector.run_dir_)}
    output.mkdir(parents=True, exist_ok=True)
    (output / "smoke_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
