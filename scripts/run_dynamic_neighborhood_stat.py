"""Run the main method through TAB's existing score/label strategies.

Each entity is loaded once. Score and label runs share the calibrated checkpoint.
Only real returned records enter the summary; missing values are never imputed.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from ts_benchmark.data.data_source import LocalAnomalyDetectDataSource
from ts_benchmark.data.suites.global_storage import GlobalStorageDataServer
from ts_benchmark.evaluation.evaluate_model import eval_model
from ts_benchmark.models import ModelFactory
from ts_benchmark.models.model_loader import get_model_info
from ts_benchmark.recording import save_log
from ts_benchmark.utils.parallel import ParallelBackend


COUNT_FIELDS = ("test_length", "valid_points", "excluded_points", "total_events",
                "complete_events", "excluded_boundary_events")
COVERAGE_FIELDS = COUNT_FIELDS + ("valid_start", "valid_end_exclusive")


def summarize(records, groups):
    """Entity macro-average inside a seed, then mean/std across independent seeds."""
    rows = []
    for record in records:
        frame, mode, seed, metrics = record
        for _, row in frame.iterrows():
            ratio = row.get("typical_anomaly_ratio", "None")
            ratio = "None" if pd.isna(ratio) else str(ratio)
            for metric in metrics:
                rows.append({"entity": row["file_name"], "dataset": groups.get(row["file_name"], Path(row["file_name"]).stem),
                             "mode": mode, "typical_anomaly_ratio": ratio,
                             "seed": seed, "metric": metric, "value": row[metric],
                             "failed": bool(str(row["log_info"] or "")),
                             **{field: row.get(field, np.nan) for field in COVERAGE_FIELDS}})
    long = pd.DataFrame(rows)
    seed_rows = []
    for (dataset, mode, ratio, seed, metric), group in long.groupby(
            ["dataset", "mode", "typical_anomaly_ratio", "seed", "metric"], sort=False):
        finite = np.isfinite(group["value"].to_numpy(dtype=float))
        complete = bool(finite.all() and not group["failed"].any())
        counts = {}
        for field in COUNT_FIELDS:
            counts[field] = float(group[field].sum()) if group[field].notna().all() else np.nan
        seed_rows.append({"dataset": dataset, "mode": mode, "typical_anomaly_ratio": ratio,
                          "seed": seed, "metric": metric,
                          "value": float(group["value"].mean()) if complete else np.nan,
                          "expected_entities": len(group), "finite_entities": int(finite.sum()),
                          "failed_entities": int(group["failed"].sum()), **counts})
    by_seed = pd.DataFrame(seed_rows)
    aggregation = {"mean": ("value", "mean"), "std": ("value", "std"),
                   "valid_seeds": ("value", "count"), "expected_seeds": ("value", "size"),
                   "coverage_seeds": ("valid_points", "count")}
    for field in COUNT_FIELDS:
        aggregation[field + "_min"] = (field, "min")
        aggregation[field + "_max"] = (field, "max")
    summary = by_seed.groupby(["dataset", "mode", "typical_anomaly_ratio", "metric"], sort=False).agg(**aggregation).reset_index()
    return long, by_seed, summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-name-list", nargs="+", default=["SWAT.csv"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--mode", choices=["both", "label", "score"], default="both")
    parser.add_argument("--gpus", nargs="+", type=int, default=[0], help="Visible physical GPUs; sequential TAB also uses this setting")
    parser.add_argument("--device", default="auto", help="auto, cpu, or a logical CUDA device, e.g. cuda:0")
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--win-size", type=int, default=None)
    parser.add_argument("--cpu-threads", type=int, default=3)
    parser.add_argument("--model-hyper-params", default="{}", help="Additional Config fields as a JSON object")
    parser.add_argument("--entity-groups", default="{}", help='Optional file-to-dataset mapping as JSON')
    parser.add_argument("--output-dir", default="result/DynamicNeighborhoodStat")
    parser.add_argument("--retrain", action="store_true", help="Retrain the first mode; the second still reuses that checkpoint")
    args = parser.parse_args(argv)
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.data_name_list)) != len(args.data_name_list):
        parser.error("Seeds and entity filenames must be unique")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    torch.set_num_threads(args.cpu_threads)
    output = Path(args.output_dir)
    if not output.is_absolute():
        output = ROOT / output
    output.mkdir(parents=True, exist_ok=True)
    parameters = json.loads(args.model_hyper_params)
    parameters.update(device=args.device, artifact_dir=str((output / "artifacts").resolve()))
    for key in ("num_epochs", "batch_size", "win_size"):
        if getattr(args, key) is not None:
            parameters[key] = getattr(args, key)
    groups = json.loads(args.entity_groups)
    modes = ["label", "score"] if args.mode == "both" else [args.mode]
    detector = get_model_info({"model_name": "self_impl.DynamicNeighborhoodStat"})
    backend = ParallelBackend()
    backend.init(backend="sequential", n_workers=1, n_cpus=args.cpu_threads,
                 gpu_devices=[] if args.device == "cpu" else args.gpus)
    records, log_files = [], []
    try:
        for name in args.data_name_list:
            source = LocalAnomalyDetectDataSource()
            source.load_series_list([name])
            server = GlobalStorageDataServer(source, backend)
            server.start_async()
            for seed in args.seeds:
                for index, mode in enumerate(modes):
                    config = json.loads((ROOT / "config" / ("dynamic_neighborhood_stat_" + mode + ".json")).read_text(encoding="utf-8"))
                    evaluation = config["evaluation_config"]
                    evaluation["strategy_args"]["seed"] = seed
                    current = dict(parameters, seed=seed)
                    if index > 0:
                        current["reuse_checkpoint"] = True
                    elif args.retrain:
                        current["reuse_checkpoint"] = False
                    factory = ModelFactory("DynamicNeighborhoodStat", detector, current)
                    result = eval_model(factory, [name], evaluation)
                    frame = pd.concat(list(result.collect()), ignore_index=True)
                    metadata = getattr(result.strategy, "evaluation_metadata_", None) or {}
                    for field in COVERAGE_FIELDS:
                        frame[field] = metadata.get(field, np.nan)
                    directory = output / "records" / Path(name).stem / ("seed_" + str(seed)) / mode
                    log_files.append(save_log(frame, str(directory), "DynamicNeighborhoodStat",
                                              compress_method=evaluation.get("record_compression", "gz")))
                    records.append((frame, mode, seed, result.strategy.evaluator.metric_names))
    finally:
        backend.close(force=True)
    long, by_seed, summary = summarize(records, groups)
    long.to_csv(output / "per_entity_metrics.csv", index=False)
    coverage = long[["entity", "dataset", "mode", "typical_anomaly_ratio", "seed", "failed"] + list(COVERAGE_FIELDS)].drop_duplicates(
        subset=["entity", "dataset", "mode", "typical_anomaly_ratio", "seed"])
    coverage.to_csv(output / "evaluation_coverage.csv", index=False)
    by_seed.to_csv(output / "per_seed_metrics.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    pd.concat([frame.assign(mode=mode, seed=seed) for frame, mode, seed, _ in records], ignore_index=True).to_csv(output / "all_records.csv", index=False)
    (output / "run_manifest.json").write_text(json.dumps({"arguments": vars(args), "parameters": parameters,
        "records": log_files, "summary_std_ddof": 1, "entity_aggregation": "equal_weight_within_seed",
        "ratio_aggregation": "separate_metrics_and_coverage_for_each_anomaly_ratio",
        "coverage_aggregation": "entity_counts_summed_within_seed; min_max_across_seeds; unknown_counts_remain_nan",
        "missing_values": "not_imputed; incomplete_entity_groups_have_nan_seed_metric"}, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    if any(frame["log_info"].fillna("").astype(bool).any() for frame, _, _, _ in records):
        raise SystemExit("Some TAB runs failed. See all_records.csv/log_info; completed runs are retained.")


if __name__ == "__main__":
    main()
