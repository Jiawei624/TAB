"""Focused checks for the sequential CLI grid; no benchmark-quality claims."""

import contextlib
import csv
import importlib.util
import io
import json
import logging
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_search():
    specification = importlib.util.spec_from_file_location("dns_grid", ROOT / "scripts/search_dynamic_neighborhood_stat.py")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def read_csv(path):
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


class GridChecks(unittest.TestCase):
    def setUp(self):
        self.runtime = Path(tempfile.mkdtemp(prefix="dns-grid-check-", dir=str(ROOT / "validation"))).resolve()
        self.addCleanup(self.cleanup)
        self.search = load_search()
        root_logger = logging.getLogger()
        previous_handlers, previous_level = list(root_logger.handlers), root_logger.level

        def restore_logging():
            for handler in list(root_logger.handlers):
                if handler not in previous_handlers:
                    root_logger.removeHandler(handler)
                    handler.close()
            root_logger.handlers[:] = previous_handlers
            root_logger.setLevel(previous_level)

        self.addCleanup(restore_logging)
        previous_logging = logging.root.manager.disable
        logging.disable(logging.CRITICAL)
        self.addCleanup(logging.disable, previous_logging)
        self.common = ["--runner", "python", "--save-root", str(self.runtime / "results"),
                       "--log-dir", str(self.runtime / "logs")]

    def cleanup(self):
        if self.runtime.parent != (ROOT / "validation").resolve() or not self.runtime.name.startswith("dns-grid-check-"):
            raise RuntimeError("Unexpected validation cleanup target")
        shutil.rmtree(self.runtime)

    def test_default_grid_preserves_original_command_and_unique_logs(self):
        parser = self.search.build_parser()
        args = parser.parse_args(self.common)
        directory, logs, trials, total = self.search.build_plan(args, parser)
        self.assertEqual(total, 108)
        self.assertEqual(len(trials), 108)
        self.assertEqual(len({trial["log_file"] for trial in trials}), 108)
        self.assertEqual(len({trial["result_directory"] for trial in trials}), 108)
        combinations = set()
        for trial in trials:
            combinations.add(tuple(trial["parameters"].values()))
            command = trial["command"]
            self.assertEqual(command[command.index("--gpus") + 1], "1")
            self.assertEqual(command[command.index("--num-workers") + 1], "1")
            parameters = json.loads(command[command.index("--model-hyper-params") + 1])
            for name, value in self.search.DEFAULT_PARAMETERS.items():
                self.assertEqual(parameters[name], value)
            self.assertNotIn("train_ratio", parameters)
            self.assertNotIn("anomaly_ratio", parameters)  # retain all ten ratio groups
            self.assertNotIn("reuse_checkpoint", parameters)
        self.assertEqual(len(combinations), 108)
        with patch.object(self.search.subprocess, "run", side_effect=AssertionError("dry run started a child")):
            self.assertEqual(self.search.main(self.common + ["--dry-run"]), 0)
        plan = json.loads((directory / "commands.json").read_text())
        self.assertEqual(plan["grid_size"], 108)
        self.assertEqual(len((directory / "commands.sh").read_text().splitlines()), 112)
        self.assertTrue(logs.is_dir())

    def test_failure_with_zero_exit_stale_records_ranking_and_resume(self):
        arguments = self.common + ["--beta-values", "0.1", "0.3", "1", "--alpha-values", "0.5",
                                   "--eta-g-values", "0.05", "--zeta-error-values", "0.95"]
        parser = self.search.build_parser()
        directory, _, trials, _ = self.search.build_plan(parser.parse_args(arguments), parser)
        calls = []

        def run(command, **kwargs):
            trial_directory = Path(command[command.index("--save-path") + 1])
            parameters = json.loads(command[command.index("--model-hyper-params") + 1])
            beta = parameters["beta"]
            calls.append(beta)
            if beta == 1:
                # TAB can catch a fit error, write a failure row, and exit zero.
                rows = [{"file_name": "SWAT.csv", "typical_anomaly_ratio": "", "log_info": "fit failed"}]
            else:
                # Ranking must use raw F1 at 0.1, ignoring PA and other ratios.
                rows = [{"file_name": "SWAT.csv", "model_params": json.dumps(parameters),
                         "typical_anomaly_ratio": ratio, "f_score": f_score, "adjust_f_score": 1 - f_score,
                         "fpr": 0.01, "add": 12, "log_info": ""}
                        for ratio, f_score in ((0.1, 0.2 if beta == 0.1 else 0.4), (0.5, 0.99))]
            self.search.write_csv(trial_directory / "DynamicNeighborhoodStat.new.csv", rows)
            kwargs["stdout"].write("test fixture output\n")
            return subprocess.CompletedProcess(command, 0)

        # A pre-existing successful CSV must not mask a new failed attempt.
        stale = Path(trials[-1]["result_directory"])
        stale.mkdir(parents=True)
        self.search.write_csv(stale / "DynamicNeighborhoodStat.old.csv", [{
            "file_name": "SWAT.csv", "typical_anomaly_ratio": 0.1, "f_score": 0.9,
            "model_params": json.dumps(trials[-1]["hyper_params"]), "log_info": ""}])
        with patch.object(self.search.subprocess, "run", side_effect=run):
            self.assertEqual(self.search.main(arguments), 1)
        self.assertEqual(calls, [0.1, 0.3, 1.0])
        summary = read_csv(directory / "summary_ratio_0p1.csv")
        self.assertEqual([row["status"] for row in summary], ["completed", "completed", "failed"])
        ranking = read_csv(directory / "ranking_ratio_0p1.csv")
        self.assertEqual([float(row["beta"]) for row in ranking], [0.3, 0.1])
        self.assertTrue(all(float(row["typical_anomaly_ratio"]) == 0.1 for row in ranking))
        # Successful trials are skipped; the failed one is attempted again.
        calls.clear()
        with patch.object(self.search.subprocess, "run", side_effect=run):
            self.assertEqual(self.search.main(arguments + ["--resume"]), 1)
        self.assertEqual(calls, [1.0])
        best = json.loads((directory / "best_ratio_0p1.json").read_text())
        self.assertEqual(best["best_by_dataset"]["SWAT.csv"]["beta"], 0.3)
        # Running one index range retains completed and failed rows from earlier batches.
        with patch.object(self.search.subprocess, "run", side_effect=AssertionError("successful trial reran")):
            self.assertEqual(self.search.main(arguments + ["--resume", "--start-index", "2", "--end-index", "2"]), 0)
        self.assertEqual(len(read_csv(directory / "summary_ratio_0p1.csv")), 3)

    def test_two_tiny_trials_through_original_tab_entry(self):
        import numpy as np
        import pandas as pd
        import torch
        from ts_benchmark.baselines.self_impl.dynamic_neighborhood_stat import DynamicNeighborhoodStat
        from ts_benchmark.data.data_pool import DataPool
        from ts_benchmark.data.data_source import LocalDataSource
        from ts_benchmark.pipeline import PREDEFINED_DATASETS

        torch.set_num_threads(2)
        old_pool = DataPool().get_pool()
        self.addCleanup(DataPool().set_pool, old_pool)
        fixture = self.runtime / "dataset"
        (fixture / "data").mkdir(parents=True)
        values = np.random.default_rng(42).normal(size=(530, 3))
        values[442:456, 1] += 5
        data = pd.DataFrame(values, columns=["a", "b", "c"])
        data["label"] = 0
        data.loc[442:455, "label"] = 1
        pd.concat([pd.DataFrame({"date": np.arange(1, len(data) + 1), "data": data[column], "cols": column})
                   for column in data], ignore_index=True).to_csv(fixture / "data/synthetic.csv", index=False)
        pd.DataFrame([{"file_name": "synthetic.csv", "train_lens": 360}]).to_csv(fixture / "META.csv", index=False)
        parameters = dict(win_size=64, num_epochs=1, batch_size=32, target_chunk_size=2,
                          device="cpu", artifact_dir=str(self.runtime / "artifacts"))
        arguments = self.common + ["--data-name-list", "synthetic.csv", "--model-hyper-params", json.dumps(parameters),
                                   "--beta-values", "0.1", "0.3", "--alpha-values", "0.5",
                                   "--eta-g-values", "0.05", "--zeta-error-values", "0.95"]
        train_calls = []
        original_train = DynamicNeighborhoodStat._train

        def train(detector, *args):
            train_calls.append(detector.config.beta)
            return original_train(detector, *args)

        def run(command, **kwargs):
            # Execute exactly the generated benchmark arguments on a tiny source.
            script_argv = command[2:]
            self.assertEqual(Path(script_argv[0]), ROOT / "scripts/run_benchmark.py")
            with patch.object(PREDEFINED_DATASETS["large_detect"], "datasrc_class", lambda: LocalDataSource(str(fixture), "META.csv")), \
                 patch.object(sys, "argv", script_argv), patch.object(logging, "basicConfig"), \
                 contextlib.redirect_stdout(kwargs["stdout"]), contextlib.redirect_stderr(kwargs["stdout"]):
                runpy.run_path(script_argv[0], run_name="__main__")
            return subprocess.CompletedProcess(command, 0)

        with patch.object(self.search.subprocess, "run", side_effect=run), \
             patch.object(DynamicNeighborhoodStat, "_train", train), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(self.search.main(arguments), 0)
        self.assertEqual(train_calls, [0.1, 0.3])
        directory = self.runtime / "results/synthetic_seed42"
        summary = read_csv(directory / "summary_ratio_0p1.csv")
        self.assertEqual(len(summary), 2)
        self.assertTrue(all(row["status"] == "completed" for row in summary))
        self.assertTrue(all(row["log_info"] == "" for row in summary))
        for row in summary:
            self.assertEqual(len(read_csv(Path(row["record_file"]))), 10)
            self.assertEqual(len(list(Path(row["result_directory"]).glob("test_report*.csv"))), 1)
            self.assertTrue(Path(row["log_file"]).is_file())


if __name__ == "__main__":
    unittest.main(verbosity=2)
