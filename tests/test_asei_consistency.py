"""TAB all-metric integration and shared VUS computation regression."""
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import numpy as np
import pandas as pd

from ts_benchmark.evaluation.evaluator import Evaluator
from ts_benchmark.evaluation.metrics import classification_metrics_score as score_metrics
from test_asei_core import SMALL, data


def test_shared_vus_curve_preserves_values_and_call_isolation():
    labels = np.array([0]*5+[1]*3+[0]*12)
    scores = np.linspace(0,1,len(labels))
    expected = [score_metrics.VUS_ROC(labels,scores),score_metrics.VUS_PR(labels,scores)]
    evaluator = Evaluator([{"name":"VUS_ROC"},{"name":"VUS_PR"}])
    with patch.object(score_metrics,"generate_curve",wraps=score_metrics.generate_curve) as curve:
        actual,log = evaluator.evaluate_with_log(labels,scores)
        assert log==""
        assert curve.call_count==1
        np.testing.assert_array_equal(actual,expected)
        evaluator.evaluate(labels,scores[::-1])
        assert curve.call_count==2


def test_tab_original_configs_all_metrics_pipeline_and_csv(tmp_path):
    from ts_benchmark.data.data_pool import DataPool
    from ts_benchmark.pipeline import pipeline
    from ts_benchmark.recording import load_record_data
    from ts_benchmark.report import report_csv
    from ts_benchmark.utils.parallel import ParallelBackend
    previous_pool = DataPool().pool
    try:
        (tmp_path/"data").mkdir()
        values = data(4300)
        values.loc[4160:4190,"a"] += 8
        values["label"] = ((np.arange(len(values))>=4160)&(np.arange(len(values))<=4190)).astype(int)
        raw = pd.concat([pd.DataFrame({"date":np.arange(len(values)),"data":values[column],"cols":column})
                         for column in values],ignore_index=True)
        raw.to_csv(tmp_path/"data/tiny.csv",index=False)
        pd.DataFrame({"file_name":["tiny.csv"],"train_lens":[4000],"size":["large"]}).to_csv(
            tmp_path/"DETECT_META.csv",index=False)
        backend = None
        with patch.dict("os.environ"):
            backend = ParallelBackend()
            backend.init(backend="sequential",n_cpus=1,gpu_devices=[])
            try:
                for mode,count in (("score",6),("label",16)):
                    config = json.loads((Path(__file__).resolve().parents[1]/f"config/unfixed_detect_{mode}_multi_config.json").read_text())
                    model_config = {"models":[{"model_name":"self_impl.ASEI","model_hyper_params":
                                     dict(SMALL,score_epochs=1,device="cpu",checkpoint_dir=str(tmp_path/"checkpoints"))}]}
                    with patch("ts_benchmark.data.data_source.ANOMALY_DETECT_DATASET_PATH",str(tmp_path)):
                        paths = pipeline({"data_name_list":["tiny.csv"]},model_config,config["evaluation_config"],str(tmp_path/mode))
                    records = load_record_data(paths)
                    assert len(records)==1
                    assert records["log_info"].fillna("").eq("").all(),records["log_info"].tolist()
                    metrics = config["report_config"]["report_metrics"]
                    assert len(metrics)==count
                    assert np.isfinite(records[metrics].to_numpy()).all()
                    report_csv.report(dict(config["report_config"],log_files_list=records,
                                           save_path=str(tmp_path/f"{mode}_report"),leaderboard_file_name="metrics.csv"))
                    report = pd.read_csv(tmp_path/f"{mode}_report/metrics.csv")
                    assert set(report["metric_name"])==set(metrics)
                assert len(list((tmp_path/"checkpoints").glob("*.pt")))==1
            finally:
                backend.close()
    finally:
        DataPool().pool = previous_pool
