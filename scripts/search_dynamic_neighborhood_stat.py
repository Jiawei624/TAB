"""Run a sequential parameter grid through the original TAB benchmark CLI.

Every trial keeps TAB's original train/validation/test and checkpoint behavior.
The summary selects the requested anomaly-ratio row from the final test results.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import itertools
import json
import logging
import math
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
import uuid


ROOT = Path(__file__).resolve().parents[1]
MODEL = "self_impl.DynamicNeighborhoodStat"
logger = logging.getLogger(__name__)
DEFAULT_PARAMETERS = {
    "batch_size": 128, "learning_rate": 0.001, "win_size": 128,
    "num_epochs": 50, "patience": 7,
}
SEARCH_VALUES = {
    "beta": [0.1, 0.3, 1.0, 3.0],
    "alpha": [0.25, 0.5, 0.75],
    "eta_g": [0.01, 0.05, 0.1],
    "zeta_error": [0.9, 0.95, 0.99],
}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config-path", default="dynamic_neighborhood_stat_label.json")
    parser.add_argument("--data-name-list", nargs="+", default=["SWAT.csv"])
    parser.add_argument("--model-hyper-params", default=json.dumps(DEFAULT_PARAMETERS),
                        help="Fixed model parameters; the four grid fields override matching keys")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpus", nargs="+", type=int, default=[1])
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=-1)
    for name, values in SEARCH_VALUES.items():
        parser.add_argument("--" + name.replace("_", "-") + "-values", nargs="+", type=float,
                            default=values, help="Candidate values for " + name)
    parser.add_argument("--target-ratio", type=float, default=0.1,
                        help="Anomaly ratio to summarize; does not alter benchmark predictions")
    parser.add_argument("--save-root", default="label/DynamicNeighborhoodStat_grid",
                        help="Result root relative to TAB/result, or an absolute directory")
    parser.add_argument("--log-dir", default="log/dynamic_neighborhood_stat_grid",
                        help="Log root relative to TAB, or an absolute directory")
    parser.add_argument("--runner", choices=["uv", "python"], default="uv",
                        help="uv run python, or the current Python environment")
    parser.add_argument("--start-index", type=int, default=1, help="First trial, numbered from 1")
    parser.add_argument("--end-index", type=int, default=None, help="Last trial, inclusive")
    parser.add_argument("--resume", action="store_true",
                        help="Skip successful matching trials whose result files still exist")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Export commands without running benchmarks")
    return parser


def number_tag(value):
    return format(value, ".12g").replace("-", "m").replace(".", "p").replace("+", "")


def resolve_path(value, base):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def write_csv(path, rows):
    if not rows:
        path.unlink(missing_ok=True)
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def build_plan(args, parser):
    try:
        fixed = json.loads(args.model_hyper_params)
    except json.JSONDecodeError as error:
        parser.error("Invalid --model-hyper-params JSON: {}".format(error))
    if not isinstance(fixed, dict):
        parser.error("--model-hyper-params must be a JSON object")
    try:
        json.dumps(fixed, allow_nan=False)
    except ValueError:
        parser.error("--model-hyper-params must not contain nonfinite numbers")
    if len(set(args.data_name_list)) != len(args.data_name_list):
        parser.error("Dataset names must be unique")
    if args.num_workers <= 0 or any(gpu < 0 for gpu in args.gpus):
        parser.error("num-workers must be positive and GPU indices must be nonnegative")
    if not math.isfinite(args.target_ratio) or not 0 < args.target_ratio < 100:
        parser.error("target-ratio must be a finite percentage between 0 and 100")
    grid = {}
    for name in SEARCH_VALUES:
        values = getattr(args, name + "_values")
        if len(set(values)) != len(values) or any(not math.isfinite(value) or value <= 0 for value in values):
            parser.error("{} values must be unique, finite, and positive".format(name))
        if name != "beta" and any(value >= 1 for value in values):
            parser.error("{} values must be smaller than 1".format(name))
        grid[name] = values
    config_path = resolve_path(args.config_path, ROOT / "config")
    if not config_path.is_file():
        parser.error("Config file does not exist: {}".format(config_path))
    config_text = config_path.read_text(encoding="utf-8")
    config = json.loads(config_text)
    evaluation = config["evaluation_config"]
    if "label" not in evaluation["strategy_args"]["strategy_name"]:
        parser.error("This grid requires a label evaluation config")
    if evaluation.get("record_compression", "gz") is not None:
        parser.error("The config must use record_compression=null to collect CSV results")
    total = math.prod(len(values) for values in grid.values())
    end = total if args.end_index is None else args.end_index
    if not 1 <= args.start_index <= end <= total:
        parser.error("Require 1 <= start-index <= end-index <= {}".format(total))
    names = "_".join(Path(name).stem for name in args.data_name_list)
    group = re.sub(r"[^\w.\-]+", "_", names)
    if len(group) > 60:
        group = group[:40] + "_" + hashlib.sha256(names.encode("utf-8")).hexdigest()[:12]
    group += "_seed{}".format(args.seed)
    directory = resolve_path(args.save_root, ROOT / "result") / group
    log_directory = resolve_path(args.log_dir, ROOT) / group
    runner = ["uv", "run", "python", "-u"] if args.runner == "uv" else [sys.executable, "-u"]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:6]
    trials = []
    for index, combination in enumerate(itertools.product(*grid.values()), 1):
        parameters = dict(zip(grid, combination))
        tag = "trial_{:03d}_b{}_a{}_g{}_q{}".format(index, *(number_tag(parameters[key]) for key in grid))
        trial_directory = directory / tag
        hyper_params = dict(fixed, **parameters)
        command = runner + [str(ROOT / "scripts" / "run_benchmark.py"), "--config-path", args.config_path,
                            "--model-name", MODEL, "--data-name-list", *args.data_name_list,
                            "--model-hyper-params", json.dumps(hyper_params, separators=(",", ":"), allow_nan=False),
                            "--seed", str(args.seed), "--gpus", *map(str, args.gpus),
                            "--num-workers", str(args.num_workers), "--timeout", str(args.timeout),
                            "--save-path", str(trial_directory)]
        trials.append({"trial_index": index, "parameters": parameters, "hyper_params": hyper_params,
                       "command": command, "result_directory": str(trial_directory),
                       "log_file": str(log_directory / (tag + "_" + stamp + ".log")),
                       "config_sha256": hashlib.sha256(config_text.encode("utf-8")).hexdigest()})
    return directory, log_directory, trials, total


def selected_records(record_files, names, ratio, parameters):
    selected = []
    for filename in record_files:
        with Path(filename).open("r", encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                try:
                    same_ratio = math.isclose(float(row["typical_anomaly_ratio"]), ratio, rel_tol=0, abs_tol=1e-9)
                except (KeyError, TypeError, ValueError):
                    same_ratio = False
                if row.get("file_name") in names and same_ratio:
                    if json.loads(row["model_params"]) != parameters:
                        raise ValueError("Record parameters do not match the trial")
                    selected.append(dict(row, record_file=str(filename)))
    if sorted(row["file_name"] for row in selected) != sorted(names):
        raise ValueError("Expected one ratio={} record for each requested dataset; found {}".format(
            ratio, [row.get("file_name") for row in selected]))
    return selected


def record_error(rows):
    for row in rows:
        if row.get("log_info", "").strip():
            return row["log_info"]
        try:
            if not math.isfinite(float(row["f_score"])):
                return "Nonfinite f_score in benchmark result"
        except (KeyError, TypeError, ValueError):
            return "Missing or invalid f_score in benchmark result"
    return ""


def trial_identity(trial, ratio):
    identity = {key: trial[key] for key in ("hyper_params", "command", "config_sha256")}
    identity["target_ratio"] = ratio
    return identity


def read_saved_trial(trial, args):
    manifest = Path(trial["result_directory"]) / "trial.json"
    if not manifest.is_file():
        return None
    try:
        previous = json.loads(manifest.read_text(encoding="utf-8"))
        if (previous["status"] not in ("completed", "failed", "interrupted")
                or "elapsed_seconds" not in previous
                or previous["identity"] != trial_identity(trial, args.target_ratio)):
            return None
        try:
            rows = selected_records(previous["record_files"], args.data_name_list, args.target_ratio, trial["hyper_params"])
        except (OSError, ValueError, KeyError):
            if previous["status"] == "completed":
                return None
            rows = []
        if previous["status"] == "completed" and record_error(rows):
            return None
        return previous, rows
    except (OSError, ValueError, KeyError):
        return None


def execute_trial(trial, args):
    reused = read_saved_trial(trial, args) if args.resume else None
    if reused is not None and reused[0]["status"] == "completed":
        logger.info("[%d] Reusing completed trial: %s", trial["trial_index"], Path(trial["result_directory"]).name)
        return reused
    directory = Path(trial["result_directory"])
    directory.mkdir(parents=True, exist_ok=True)
    before = set(directory.glob("DynamicNeighborhoodStat.*.csv"))
    manifest = {**trial, "identity": trial_identity(trial, args.target_ratio), "status": "running",
                "started_at": datetime.now(timezone.utc).isoformat(), "record_files": [], "error": ""}
    write_json(directory / "trial.json", manifest)
    logger.info("[%d] Starting %s", trial["trial_index"], directory.name)
    logger.info("Command: %s", shlex.join(trial["command"]))
    logger.info("Log: %s", trial["log_file"])
    started, rows = time.perf_counter(), []
    try:
        with Path(trial["log_file"]).open("w", encoding="utf-8") as stream:
            stream.write("# " + shlex.join(trial["command"]) + "\n")
            stream.flush()
            process = subprocess.run(trial["command"], cwd=str(ROOT), stdout=stream,
                                     stderr=subprocess.STDOUT, check=False)
        manifest["exit_code"] = process.returncode
        manifest["record_files"] = [str(path) for path in sorted(set(directory.glob("DynamicNeighborhoodStat.*.csv")) - before)]
        if process.returncode:
            raise RuntimeError("Benchmark exited with code {}".format(process.returncode))
        rows = selected_records(manifest["record_files"], args.data_name_list, args.target_ratio, trial["hyper_params"])
        error = record_error(rows)
        if error:
            raise RuntimeError(error)
        manifest["status"] = "completed"
    except KeyboardInterrupt:
        manifest["status"], manifest["error"] = "interrupted", "Interrupted by the user"
        raise
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        manifest["status"], manifest["error"] = "failed", str(error)
        logger.error("[%d] Failed: %s; see %s", trial["trial_index"], error, trial["log_file"])
    finally:
        manifest["elapsed_seconds"] = time.perf_counter() - started
        write_json(directory / "trial.json", manifest)
    logger.info("[%d] %s in %.1fs", trial["trial_index"], manifest["status"], manifest["elapsed_seconds"])
    return manifest, rows


def summary_rows(manifest, records, args):
    metadata = {"trial_index": manifest["trial_index"], **manifest["parameters"], "status": manifest["status"],
                "exit_code": manifest.get("exit_code", ""), "elapsed_seconds": manifest["elapsed_seconds"],
                "log_file": manifest["log_file"], "result_directory": manifest["result_directory"],
                "error": manifest["error"]}
    if not records:
        records = [{"file_name": name, "typical_anomaly_ratio": args.target_ratio} for name in args.data_name_list]
    return [{**metadata, **row} for row in records]


def update_summary(directory, rows, ratio):
    suffix = "ratio_" + number_tag(ratio)
    write_csv(directory / ("summary_" + suffix + ".csv"), sorted(rows, key=lambda row: (row["trial_index"], row["file_name"])))
    completed = [row for row in rows if row["status"] == "completed"]

    def optional_metric(row, name, default):
        try:
            value = float(row.get(name, ""))
            return value if math.isfinite(value) else default
        except (TypeError, ValueError):
            return default

    ranked = sorted(completed, key=lambda row: (row["file_name"], -float(row["f_score"]),
                    optional_metric(row, "fpr", float("inf")), optional_metric(row, "add", float("inf")), row["trial_index"]))
    counts, ranking, best = {}, [], {}
    for row in ranked:
        name = row["file_name"]
        counts[name] = counts.get(name, 0) + 1
        ranking.append(dict(rank=counts[name], **row))
        if name not in best:
            best[name] = row
    write_csv(directory / ("ranking_" + suffix + ".csv"), ranking)
    write_json(directory / ("best_" + suffix + ".json"), {
        "selection_source": "official_test_results", "typical_anomaly_ratio": ratio,
        "ranking": "f_score descending; fpr ascending; add ascending; trial_index ascending",
        "best_by_dataset": best})


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    directory, log_directory, all_trials, total = build_plan(args, parser)
    end = total if args.end_index is None else args.end_index
    trials = all_trials[args.start_index - 1:end]
    directory.mkdir(parents=True, exist_ok=True)
    log_directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "commands.json", {"arguments": vars(args), "grid_size": total, "trials": trials})
    shell_lines = ["#!/usr/bin/env bash", "set -euo pipefail", "cd " + shlex.quote(str(ROOT)),
                   "mkdir -p " + shlex.quote(str(log_directory))]
    shell_lines.extend(shlex.join(trial["command"]) + " > " + shlex.quote(trial["log_file"]) + " 2>&1" for trial in trials)
    (directory / "commands.sh").write_text("\n".join(shell_lines) + "\n", encoding="utf-8")
    logger.info("Grid: %d combinations; selected trials: %d; runner: %s", total, len(trials), args.runner)
    logger.info("Commands and summaries: %s", directory)
    if args.dry_run:
        logger.info("Dry run completed; no benchmarks were started")
        return 0
    rows, failures, processed = [], 0, 0
    selected_indices = {trial["trial_index"] for trial in trials}
    # Keep earlier batches in the summary, after checking their exact configuration.
    for trial in all_trials:
        if trial["trial_index"] not in selected_indices:
            saved = read_saved_trial(trial, args)
            if saved is not None:
                rows.extend(summary_rows(saved[0], saved[1], args))
    for trial in trials:
        manifest, records = execute_trial(trial, args)
        processed += 1
        rows.extend(summary_rows(manifest, records, args))
        update_summary(directory, rows, args.target_ratio)
        if manifest["status"] != "completed":
            failures += 1
            if args.stop_on_error:
                break
    logger.info("Finished selected batch: %d successful trials, %d failed trials", processed - failures, failures)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
