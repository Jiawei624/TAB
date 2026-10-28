"""Data coverage and independent mathematical oracles for the implementation audit."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd
import pytest
import torch
from torch.nn import functional as F

from ts_benchmark.baselines.self_impl.ASEI import ASEI
from ts_benchmark.baselines.self_impl.ASEI.auxiliary import Task, prepare_objective, objective_to_device, task_objectives
from ts_benchmark.baselines.self_impl.ASEI.data import CausalPreprocessor
from ts_benchmark.baselines.self_impl.ASEI.model import Predictor, lag_graph
from ts_benchmark.baselines.self_impl.ASEI.optimization import index_batches
from ts_benchmark.baselines.self_impl.ASEI.audit import (
    audit_preprocessing, audit_frozen_sequence, compare_calibration, frozen_numeric_probe,
    infer_point_rates, audit_supplied_run)
from test_asei_core import SMALL,data


@pytest.fixture(scope="module")
def frozen_audit_case(tmp_path_factory):
    from ts_benchmark.data.utils import read_data
    root = tmp_path_factory.mktemp("asei_audit_data")
    (root/"data").mkdir()
    frame = data(4300)
    frame["c"] = .3
    frame["label"] = ((np.arange(4300)>=4160)&(np.arange(4300)<4190)).astype(int)
    raw = pd.concat([pd.DataFrame({"date":np.arange(4300),"data":frame[column],"cols":column})
                     for column in frame],ignore_index=True)
    raw.to_csv(root/"data/tiny.csv",index=False)
    pd.DataFrame({"file_name":["tiny.csv"],"train_lens":[4000]}).to_csv(root/"DETECT_META.csv",index=False)
    table = read_data(str(root/"data/tiny.csv")).reset_index(drop=True)
    model = ASEI(**dict(SMALL,score_epochs=1),device="cpu").detect_fit(table.iloc[:4000].drop(columns="label"))
    return model,root,table


def test_decimal_constant_columns_have_unit_scale_and_exact_zero_zscores():
    # The axis-0 FP64 reduction can produce std>0 for genuinely constant decimals.
    values = np.tile([.3,7.2],(316800,1))
    pre = CausalPreprocessor().fit(values)
    np.testing.assert_array_equal(pre.constant,[True,True])
    np.testing.assert_array_equal(pre.scale,[1.,1.])
    np.testing.assert_array_equal(pre.mean,values[0])
    z,_ = pre.transform(values[:17])
    np.testing.assert_array_equal(z,np.zeros_like(z))
    shifted = values[:1].copy()
    shifted[0,0] += 1e-6
    changed,_ = pre.transform(shifted)
    assert 0<changed[0,0]<2e-6


def test_true_small_variance_is_preserved_instead_of_arbitrary_scale_clipping():
    values = np.column_stack((np.full(2000,.3),.3+np.arange(2000)*1e-10))
    pre = CausalPreprocessor().fit(values)
    np.testing.assert_array_equal(pre.constant,[True,False])
    assert pre.scale[0]==1 and 0<pre.scale[1]<1e-6
    np.testing.assert_allclose(pre.scale[1],np.std(values[:,1]),rtol=1e-7)


@pytest.mark.parametrize("points,shuffle,expected_batches",[(316800,True,155),(49500,False,25)])
def test_swat_predictor_and_selection_loaders_cover_every_target_once(points,shuffle,expected_batches):
    model = ASEI(device="cpu")
    model._resolve_capacity(316800)
    values = np.arange(points,dtype=np.float32)[:,None]
    loader = model._predictor_loader(values,shuffle)
    actual = []
    for _,target,mask in loader:
        actual.append(target[...,0][mask].numpy())
    ids = np.concatenate(actual).astype(np.int64)
    assert len(loader)==expected_batches
    np.testing.assert_array_equal(np.sort(ids),np.arange(78,points))


def test_swat_fixed_corrector_budget_covers_five_complete_passes_per_stage():
    batches = index_batches(39518,2048,np.random.default_rng(42))
    for _ in range(2):
        visits = np.zeros(39518,dtype=np.int32)
        processed = 0
        for step in range(100):
            indices = next(batches)
            visits[indices] += 1
            processed += len(indices)
        assert processed==197590
        np.testing.assert_array_equal(visits,np.full(39518,5))


def test_block_prediction_preserves_huber_objective_and_parameter_gradients():
    torch.manual_seed(31)
    values = np.random.default_rng(31).normal(size=(105,3))
    model = Predictor(3,*lag_graph(values,[1,2,4],2),hidden=4,embedding=2,
                      kernel=3,dilations=(1,2),win_size=7).double()
    other = copy.deepcopy(model)
    history,steps = 10,5
    x = torch.tensor(values[:history+steps-1],requires_grad=True)
    y = torch.tensor(values[history:history+steps])
    block = model(x[None],steps)
    # Independent point-window evaluation with shared original input variables.
    point = torch.cat([other(x[index:index+history][None]) for index in range(steps)],1)
    loss = F.huber_loss(block,y[None,...,None].expand_as(block))
    expected_loss = F.huber_loss(point,y[None,...,None].expand_as(point))
    grads = torch.autograd.grad(loss,tuple(model.parameters())+(x,))
    expected_grads = torch.autograd.grad(expected_loss,tuple(other.parameters())+(x,))
    torch.testing.assert_close(block,point,atol=1e-10,rtol=1e-10)
    torch.testing.assert_close(loss,expected_loss,atol=1e-10,rtol=1e-10)
    for a,b in zip(grads,expected_grads):
        torch.testing.assert_close(a,b,atol=1e-9,rtol=1e-9)


def test_detection_and_ranking_losses_match_independent_numpy_formula():
    tasks = [Task(0,50,[(10,11)],0,0,1),Task(0,50,[(10,20),(25,40)],1,1,2)]
    metadata = [prepare_objective(t,0,32,np.random.default_rng(i)) for i,t in enumerate(tasks)]
    scores = np.random.default_rng(8).normal(size=(2,2,50))
    bias,tau = .7,.25
    detection,ranking = task_objectives(torch.tensor(scores),torch.tensor((scores-bias)/tau),
                                        objective_to_device(metadata,"cpu"))
    for j,task in enumerate(tasks):
        z = (scores[j]-bias)/tau
        positive = np.mean([np.logaddexp(0,-z[1,left:right]).mean() for left,right in task.events])
        first = task.events[0][0]
        negative_sum = np.logaddexp(0,z[0,first:]).sum()
        negative_count = len(z[0,first:])
        mask = np.arange(50)>=first
        for left,right in task.events:
            mask[left:right] = False
        negative_sum += 2*np.logaddexp(0,z[1,mask]).sum()
        negative_count += 2*mask.sum()
        expected = .5*positive+.5*negative_sum/negative_count
        pos,neg = metadata[j]["rank_positive"],metadata[j]["rank_negative"]
        rank = np.logaddexp(0,1-scores[j,1,pos]+scores[j].reshape(-1)[neg]).mean()
        # The frozen event weights are FP32, including 1/L for non-power-of-two L.
        np.testing.assert_allclose(detection[j],expected,atol=1e-7,rtol=1e-7)
        np.testing.assert_allclose(ranking[j],rank,atol=1e-12,rtol=1e-12)


def test_frozen_diagnostics_reproduce_scores_without_mutating_models_or_stream(frozen_audit_case):
    model,_,table = frozen_audit_case
    sample = table.iloc[4000:].drop(columns="label")
    expected = model.detect_score(sample)[0]
    model.detect_stream(sample.iloc[:95],reset=True)
    cache,state,last = model._raw_cache.copy(),model._stream_state.clone(),model._imputer_last.copy()
    metadata,thresholds = copy.deepcopy(model.metadata_),dict(model.thresholds_)
    weights = [{k:v.clone() for k,v in module.state_dict().items()} for module in
               (model.predictor,model.corrector,model.scorer)]
    stats,scores = audit_frozen_sequence(model,sample,table["label"].iloc[4000:].values)
    np.testing.assert_array_equal(scores,expected)
    assert stats["ready_scores"]==211
    assert abs(sum(c["maximum_score_dominance"] for c in stats["channels"])-1)<1e-9
    assert stats["empirical_alarms"]["0.01"]["alarms"]==int(np.count_nonzero(scores[89:]>thresholds[.01]))
    assert model.metadata_==metadata and model.thresholds_==thresholds
    np.testing.assert_array_equal(cache,model._raw_cache)
    np.testing.assert_array_equal(last,model._imputer_last)
    torch.testing.assert_close(state,model._stream_state,atol=0,rtol=0)
    for module,original in zip((model.predictor,model.corrector,model.scorer),weights):
        for key,value in module.state_dict().items():
            torch.testing.assert_close(value,original[key],atol=0,rtol=0)


def test_calibration_reproduction_preprocessing_audit_and_full_coverage_counters(frozen_audit_case):
    model,_,table = frozen_audit_case
    _,scores = audit_frozen_sequence(model,table.iloc[3600:4000].drop(columns="label"))
    record = compare_calibration(model,scores)[0]
    assert record["N"]==311 and record["threshold_abs_difference"]==0
    assert record["calibration_scores_max_abs_difference"]==0
    fit = table.iloc[:2560].drop(columns="label").values
    assert audit_preprocessing(model,fit)["constant_scale_violations"]==[]
    legacy = copy.deepcopy(model)
    legacy.preprocessor.scale[2] = 1e-12
    legacy.preprocessor.constant[2] = False
    assert audit_preprocessing(legacy,fit)["constant_scale_violations"][0]["channel"]=="c"
    for row in model.metadata_["predictor"]["history"]:
        assert row["training_targets"]==2482 and row["validation_targets"]==322
    for name in ("corrector_linear","corrector_nonlinear"):
        assert model.metadata_["corrector"][name]["distinct_samples_seen"]==238
        assert model.metadata_["corrector"][name]["minimum_sample_visits"]==3
    row = model.metadata_["selection_history"][0]
    assert row["training_tasks_seen"]==12 and row["validation_components"]["tasks_seen"]==6
    assert model.metadata_["data_coverage"]["independent_reference_points"]==238
    standardized = model.preprocessor.transform(table.iloc[2880:3200].drop(columns="label").values)[0]
    probe = frozen_numeric_probe(model,standardized)
    assert probe["score_scan_reference_max_abs"]<=5e-5
    assert probe["evidence_slice_max_abs"]<=1e-5
    assert probe["evidence_batch_max_abs"]<=1e-5


def test_report_rate_inference_and_log_reconciliation(tmp_path):
    a,p,r = .257773065818514,.1337013966297291,.9332857325936912
    rates = infer_point_rates(a,p,r)
    assert abs(rates["normal_false_positive_rate"]-.8355670911565451)<1e-12
    assert abs(rates["predicted_positive_fraction"]-.84743031523452)<1e-12
    assert abs(rates["true_positive_fraction"]+rates["true_negative_fraction"]-a)<1e-12
    assert infer_point_rates(.9,0,.1) is None
    log = tmp_path/"run.log"
    log.write_text("Start loading 1 series in parallel\n"
                   "ASEI CUDA startup implementation=asei-finite-bank-scan-3.1 torch=2.4.1+cu121 device=cuda:0\n"
                   "ASEI predictor epoch=13 train=0.02 Vsel=0.01931489\n"
                   "ASEI predictor epoch=14 train=0.02 Vsel=0.01914982\n"
                   "ASEI predictor epoch=18 train=0.02 Vsel=0.02223442\n"
                   "ASEI scorer epoch=9 Vsel_loss=50.291997 lambda=0.879836 kappa=0.026114\n"
                   "ASEI scorer epoch=10 Vsel_loss=49.002122 lambda=0.877622 kappa=0.027109\n"
                   "ASEI stage=predictor elapsed_s=70.782\n",encoding="utf-8")
    report = tmp_path/"metrics.csv"
    pd.DataFrame({"strategy_args":["{}"]*3,"metric_name":["accuracy","precision","recall"],"ASEI":[a,p,r]}).to_csv(report,index=False)
    parsed = audit_supplied_run(log,report)
    assert parsed["predictor_best_epoch_from_log"]==14
    assert parsed["loaded_series_count"]==1
    assert parsed["observed_runtime"]["implementation_version"]=="asei-finite-bank-scan-3.1"
    assert parsed["reported_models"]["ASEI"]["inferred_point_rates"] is not None
    assert parsed["predictor_actual_epochs_from_log"]==18
    assert parsed["scorer_best_is_last"] and parsed["scorer_validation_still_decreasing"]
    assert parsed["stage_seconds"]["predictor"]==70.782


def test_audit_cli_uses_frozen_checkpoint_and_verifies_actual_dataset(frozen_audit_case,tmp_path):
    model,root,_ = frozen_audit_case
    checkpoint,report = tmp_path/"model.pt",tmp_path/"audit.json"
    model.save(checkpoint)
    result = subprocess.run([sys.executable,"-B","scripts/audit_asei.py","--checkpoint",str(checkpoint),
                             "--dataset-root",str(root),"--data-name","tiny.csv","--include-test",
                             "--device","cpu","--output",str(report)],
                            cwd=str(Path(__file__).resolve().parents[1]),capture_output=True,text=True,timeout=120)
    assert result.returncode==0,result.stdout+result.stderr
    parsed = json.loads(report.read_text(encoding="utf-8"))
    assert parsed["data_fingerprint_matches"]
    assert parsed["preprocessing"]["constant_scale_violations"]==[]
    assert parsed["sequences"]["Vcal"]["ready_scores"]==311
    assert parsed["sequences"]["test"]["points"]==300
    assert parsed["calibration_reproduction"][0]["threshold_abs_difference"]==0


@pytest.mark.skipif(not torch.cuda.is_available(),reason="Requires one CUDA GPU")
def test_single_gpu_decimal_channels_and_frozen_numerical_consistency():
    frame = data()
    frame["c"] = .3
    model = ASEI(**dict(SMALL,score_epochs=1),device="cuda:0").detect_fit(frame)
    values = model.preprocessor.transform(frame.iloc[2880:3200].values)[0]
    probe = frozen_numeric_probe(model,values)
    assert probe["evidence_slice_max_abs"]<=1e-5,probe
    assert probe["evidence_batch_max_abs"]<=1e-5,probe
    assert probe["score_scan_reference_max_abs"]<=5e-5,probe
