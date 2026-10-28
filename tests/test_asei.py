"""Small complete fits, frozen-bank invariants, persistence and streaming."""
import copy
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
import textwrap
from unittest.mock import patch
import numpy as np
import pytest
import torch
from ts_benchmark.baselines.self_impl.ASEI import ASEI
from ts_benchmark.baselines.self_impl.ASEI.auxiliary import TaskSampler
from ts_benchmark.baselines.self_impl.ASEI.cache import EvidenceBank
from ts_benchmark.baselines.self_impl.ASEI.calibration import error_history
from ts_benchmark.baselines.self_impl.ASEI.data import chronological_split
from ts_benchmark.baselines.self_impl.ASEI.model import cpu_state
from test_asei_core import SMALL, data


@pytest.fixture(scope="module")
def fitted():
    return ASEI(**SMALL,device="cpu").detect_fit(data(),train_label=object())


def test_complete_fit_and_calibration_counts(fitted):
    model = fitted
    assert model.metadata_["splits"]["segment_lengths"]==dict(F=2560,Cfit=320,Rref=320,Vsel=400,Vcal=400)
    assert model.history==78 and model.raw_history==89
    assert len(model.reference.errors)==238 and len(model.validation_scores_)==311
    status = model.metadata_["corrector"]
    for name in ("corrector_linear","corrector_nonlinear"):
        assert status[name]["updates"]==3
        assert status[name]["processed_samples"]==238*3
        assert np.isfinite(status[name]["last_loss"])
        assert "converged" not in status[name]
    assert model.corrector.beta.dtype==torch.float32
    assert model.corrector.beta.abs().sum(-1).max()<=.950001
    assert all(not p.requires_grad for m in (model.predictor,model.corrector,model.scorer) for p in m.parameters())
    record = model.metadata_["calibration"][0]
    assert record["N"]==311 and record["K"]==3
    assert record["false_alarm_count"]==int((model.validation_scores_>record["threshold"]).sum())<=3
    assert model.metadata_["selection"]["scorer_instances"]==1
    assert model.metadata_["parameters"]["scorer"]==8
    assert len(model.metadata_["selection_history"])==3
    assert all(len(row["groups"])==6 for row in model.metadata_["selection_history"])
    assert model.metadata_["counters"]==dict(normal_evidence_passes=2,changed_trajectory_builds=18,
                                            score_epoch_feature_extractor_calls=0)


def test_corrector_scales_and_residual_formula(fitted):
    raw,_ = chronological_split(data().values)
    c = fitted.preprocessor.transform(raw["Cfit"])[0]
    errors = fitted._error_tensor(c)
    d = errors.numpy()
    expected = np.maximum(np.maximum(d.mean(0,dtype=np.float64),np.percentile(d,75,axis=0)-np.percentile(d,25,axis=0)),1e-4)
    np.testing.assert_allclose(fitted.corrector.error_scale,expected,rtol=1e-6)
    y = errors/fitted.corrector.error_scale
    h = torch.tensor(error_history(y.numpy(),4))
    expected_r = fitted.corrector.error_scale*(y[4:]-fitted.corrector(h))
    torch.testing.assert_close(fitted._residual_tensor(errors),expected_r,atol=1e-6,rtol=1e-6)
    # The shared nonlinear network is one P -> 8 -> 1 module for all channels/branches.
    assert fitted.corrector.mlp[0].in_features==4 and fitted.corrector.mlp[-1].out_features==1


def test_cached_evidence_matches_independent_trajectory_and_no_epoch_extractor(fitted,tmp_path):
    raw,_ = chronological_split(data().values)
    fit = fitted.preprocessor.transform(raw["F"])[0]
    sampler = TaskSampler(fit,89,fitted.metadata_["T0"],fitted.config,fitted.protocol_)
    task = copy.deepcopy(fitted.validation_tasks_[0])
    task.start = 19
    task.segment_id = "F"
    task.perturbations = None
    task = sampler.resolve(task)[0]
    normal,changed = sampler.materialize(task)
    full = fitted._evidence_tensor(fit)
    independent = fitted._evidence_tensor(normal)
    n = task.length-82
    torch.testing.assert_close(full[task.start:task.start+n],independent,atol=1e-5,rtol=1e-5)
    fitted._cache_device_bytes = fitted._cache_host_bytes = 0
    bank = fitted._build_bank(sampler,[task],full)
    try:
        pair,metadata = bank.get(0)
        torch.testing.assert_close(pair[1],fitted._evidence_tensor(changed),atol=1e-5,rtol=1e-5)
        torch.testing.assert_close(pair[0],independent,atol=1e-5,rtol=1e-5)
        with patch.object(fitted,"_error_tensor",side_effect=AssertionError("extractor called during score epoch")):
            detection,ranking = fitted._bank_loss(bank,[0])
            assert torch.isfinite(detection+ranking).all()
    finally:
        bank.close()


def test_streaming_missing_history_cold_start_and_future_causality(fitted):
    x = data(220,31)
    x.iloc[:4,0] = np.nan
    x.iloc[92:97,1] = np.inf
    labels,whole = fitted.detect_label(x)
    assert len(whole)==220
    assert not fitted.last_ready_mask_[:89].any() and fitted.last_ready_mask_[89:].all()
    assert not labels[1.][:89].any()
    np.testing.assert_array_equal(whole[:89],np.full(89,-6.2,dtype=np.float32))
    np.testing.assert_array_equal(whole,fitted.detect_score(x)[0])
    pieces,alarms = [],[]
    fitted.reset_state()
    # Includes one-point calls, cold calls shorter than D, and post-warmup arbitrary cuts.
    edges = [0,1,12,77,89,90,91,95,96,103,128,199,220]
    for left,right in zip(edges[:-1],edges[1:]):
        y,s = fitted.detect_stream(x.iloc[left:right])
        pieces.append(s)
        alarms.append(y[1.])
    np.testing.assert_allclose(np.concatenate(pieces),whole,atol=1e-5,rtol=1e-5)
    np.testing.assert_array_equal(np.concatenate(alarms),labels[1.])
    future = x.copy()
    future.iloc[170:] += 100
    np.testing.assert_allclose(fitted.detect_score(future)[0][:170],whole[:170],atol=1e-5,rtol=1e-5)
    empty_labels,empty_scores = fitted.detect_stream(x.iloc[:0])
    assert len(empty_scores)==0 and len(empty_labels[1.])==0


def test_save_reload_and_old_format_rejection(fitted,tmp_path):
    path = tmp_path/"model.pt"
    fitted.save(path)
    restored = ASEI.load(path,device="cpu")
    x = data(210,71)
    np.testing.assert_array_equal(restored.detect_score(x)[0],fitted.detect_score(x)[0])
    assert restored.thresholds_==fitted.thresholds_
    assert restored.metadata_["training_tasks"]==fitted.metadata_["training_tasks"]
    assert restored._processed_points==210
    payload = torch.load(path,weights_only=False)
    payload["format_version"] = 2
    torch.save(payload,path)
    with pytest.raises(ValueError,match="format_version=3"):
        ASEI.load(path,device="cpu")


def test_vcal_does_not_affect_fits_selection_or_reference(fitted):
    x = data()
    x.iloc[3600:] = x.iloc[3600:]*20+100
    other = ASEI(**SMALL,device="cpu").detect_fit(x,train_label=object())
    for name in ("predictor","corrector","scorer"):
        for key,value in cpu_state(getattr(fitted,name)).items():
            torch.testing.assert_close(value,cpu_state(getattr(other,name))[key],atol=0,rtol=0)
    np.testing.assert_array_equal(fitted.reference.errors,other.reference.errors)
    np.testing.assert_array_equal(fitted.reference.center,other.reference.center)
    assert fitted.metadata_["selection"]==other.metadata_["selection"]
    assert fitted.thresholds_!=other.thresholds_


def test_training_amp_setting_does_not_prevent_cpu_checkpoint_inference(fitted,tmp_path):
    # BF16 is a training policy; all persisted weights and evidence remain FP32.
    model = copy.deepcopy(fitted)
    model.config.predictor_amp = "bf16"
    path = tmp_path/"training_amp.pt"
    model.save(path)
    restored = ASEI.load(path,device="cpu")
    np.testing.assert_array_equal(restored.detect_score(data(160))[0],fitted.detect_score(data(160))[0])
    with pytest.raises(ValueError,match="training requires"):
        ASEI(predictor_amp="bf16",device="cpu").detect_fit(data())


def test_external_validation_matches_internal_and_entities_are_independent(fitted):
    x = data()
    other = ASEI(**SMALL,device="cpu").detect_fit(x.iloc[:3200],valid_data=x.iloc[3200:])
    assert other.metadata_["splits"]["external_validation"]
    for name in ("predictor","corrector","scorer"):
        for key,value in cpu_state(getattr(fitted,name)).items():
            torch.testing.assert_close(value,cpu_state(getattr(other,name))[key],atol=0,rtol=0)
    assert fitted.thresholds_==other.thresholds_
    assert fitted.scorer is not other.scorer and fitted.corrector is not other.corrector
    other.thresholds_[.01] = -100
    assert fitted.thresholds_[.01]!=other.thresholds_[.01]


def test_short_input_columns_cpu_and_bounded_api(fitted):
    with pytest.raises(ValueError,match="split"):
        ASEI(device="cpu").detect_fit(data(1000))
    with pytest.raises(ValueError,match="order"):
        fitted.detect_score(data(30)[["b","a","c"]])
    with patch("torch.cuda.is_available",return_value=False):
        assert ASEI().device.type=="cpu"
    model = copy.deepcopy(fitted)
    model.config.score_transform = "bounded"
    raw,_ = chronological_split(data().values)
    model._calibrate(model.preprocessor.transform(raw["Vcal"])[0])
    labels,scores = model.detect_label(data(220))
    assert np.all((scores>=0)&(scores<=1))
    np.testing.assert_array_equal(labels[1.],fitted.detect_label(data(220))[0][1.])


def test_disk_bank_budget_integrity_and_missing_key(tmp_path):
    values = torch.randn(2,20,3,2)
    bank = EvidenceBank("cpu",0,0,tmp_path,"key")
    try:
        bank.append(values,{},None)
        with pytest.raises(ValueError,match="Incomplete"):
            bank.get(0)
        bank.seal()
        assert bank.statistics()["disk_bytes"]==values.numel()*4
        torch.testing.assert_close(bank.get(0)[0],values)
        with pytest.raises(ValueError,match="mismatch"):
            bank.verify_key("other")
        bank.entries[0]["storage"].write_bytes(b"incomplete")
        with pytest.raises(ValueError,match="Corrupt"):
            bank.get(0)
    finally:
        directory = bank.directory
        bank.close()
    assert not directory.exists()


def test_complete_fit_with_disk_spill_and_equal_validation_early_stop(tmp_path):
    groups = {f"{family}:{layout}":2. for family in range(3) for layout in ("single","recurrent")}
    options = dict(SMALL,score_epochs=5,score_min_epochs=2,score_patience=2,
                   evidence_device_cache_mb=0,evidence_host_cache_mb=0,cache_dir=str(tmp_path))
    with patch.object(ASEI,"_validate_bank",return_value=(2.,groups)):
        model = ASEI(**options,device="cpu").detect_fit(data())
    assert model.metadata_["selection"]["epoch"]==1
    assert model.metadata_["selection"]["actual_epochs"]==3
    assert model.metadata_["counters"]["score_epoch_feature_extractor_calls"]==0
    for stats in model.metadata_["evidence_banks"].values():
        assert stats["device_bytes"]==stats["host_bytes"]==0 and stats["disk_bytes"]>0
    assert not list(tmp_path.glob(".asei-bank-*"))


@pytest.mark.parametrize("requested,index",[("cuda:0",0),("cuda:1",1),("cuda",1)])
def test_fit_initializes_selected_cuda_before_resetting_memory_stats(requested,index):
    # Reproduce PyTorch's lazy allocator on CPU: availability/device_count alone
    # do not register allocator devices. reset_peak_memory_stats must come later.
    model = ASEI(device="cpu")
    model.device = torch.device(requested)
    events = []
    initialized = False

    class ReachedInputValidation(Exception):
        pass

    def initialize():
        nonlocal initialized
        initialized = True
        events.append("init")

    def select(device):
        assert initialized
        assert device==torch.device("cuda",index)
        events.append("select")

    def reset(device):
        if not initialized:
            raise RuntimeError("Invalid device argument")
        assert device==torch.device("cuda",index)
        events.append("reset")

    with patch("torch.cuda.init",side_effect=initialize), \
            patch("torch.cuda.current_device",return_value=index), \
            patch("torch.cuda.set_device",side_effect=select), \
            patch("torch.cuda.reset_peak_memory_stats",side_effect=reset), \
            patch("torch.manual_seed"), \
            patch.object(model,"_values",side_effect=ReachedInputValidation):
        with pytest.raises(ReachedInputValidation):
            model.detect_fit(None)
    assert events==["init","select","reset"]
    assert model.device==torch.device("cuda",index)


def test_cpu_fit_does_not_initialize_cuda_or_reset_cuda_statistics():
    model = ASEI(device="cpu")

    class ReachedInputValidation(Exception):
        pass

    with patch("torch.cuda.init",side_effect=AssertionError("CPU initialized CUDA")), \
            patch("torch.cuda.set_device",side_effect=AssertionError("CPU selected CUDA")), \
            patch("torch.cuda.reset_peak_memory_stats",side_effect=AssertionError("CPU reset CUDA stats")), \
            patch.object(model,"_values",side_effect=ReachedInputValidation):
        with pytest.raises(ReachedInputValidation):
            model.detect_fit(None)


def test_cuda_initialization_errors_are_not_silently_ignored():
    model = ASEI(device="cpu")
    model.device = torch.device("cuda:0")
    with patch("torch.cuda.init",side_effect=RuntimeError("CUDA driver initialization failed")), \
            patch("torch.cuda.reset_peak_memory_stats") as reset, \
            patch.object(model,"_values") as values:
        with pytest.raises(RuntimeError,match="CUDA driver initialization failed"):
            model.detect_fit(None)
    reset.assert_not_called()
    values.assert_not_called()


@pytest.mark.skipif(not torch.cuda.is_available(),reason="Requires one CUDA GPU")
def test_fresh_process_single_gpu_fit_initializes_allocator():
    # Other GPU tests can initialize the allocator and conceal the startup bug.
    # Inherit CUDA_VISIBLE_DEVICES so physical GPU1 remains logical cuda:0.
    program = textwrap.dedent('''
        import numpy as np
        import torch
        from ts_benchmark.baselines.self_impl.ASEI import ASEI
        assert not torch.cuda.is_initialized()
        values = np.random.default_rng(42).normal(size=(4000,3))
        model = ASEI(device="auto",seed=42,predictor_epochs=1,
                     corrector_linear_steps=1,corrector_nonlinear_steps=1,
                     score_epochs=1,auxiliary_train_pairs=12,
                     auxiliary_validation_pairs=6)
        assert model.device==torch.device("cuda:0")
        model.detect_fit(values)
        assert torch.cuda.is_initialized() and torch.cuda.current_device()==0
        assert model._fitted
        assert model.metadata_["peak_cuda_allocated_bytes"]>0
        assert model.metadata_["calibration"][0]["N"]==311
        print("fresh-process CUDA fit passed")
    ''')
    result = subprocess.run([sys.executable,"-B","-c",program],
                            cwd=str(Path(__file__).resolve().parents[1]),
                            capture_output=True,text=True,timeout=180)
    assert result.returncode==0,result.stdout+result.stderr
    assert "fresh-process CUDA fit passed" in result.stdout


@pytest.mark.skipif(not torch.cuda.is_available(),reason="Requires one CUDA GPU")
def test_single_gpu_fit_gradients_and_cpu_portability(tmp_path):
    model = ASEI(**dict(SMALL,score_epochs=1),device="cuda:0").detect_fit(data())
    assert all(p.device.index==0 for m in (model.predictor,model.corrector,model.scorer) for p in m.parameters())
    features = torch.randn(2,2,513,3,4,device="cuda:0")
    optimized = copy.deepcopy(model.scorer).requires_grad_(True)
    reference = copy.deepcopy(optimized)
    reference.recurrence_backend = "reference"
    outputs = []
    for head in (optimized,reference):
        s,z = head(features)
        (s.square().mean()+z.mean()).backward()
        outputs.append(s)
    torch.testing.assert_close(*outputs,atol=5e-5,rtol=1e-5)
    for a,b in zip(optimized.parameters(),reference.parameters()):
        torch.testing.assert_close(a.grad,b.grad,atol=5e-4,rtol=1e-4)
    path = tmp_path/"gpu.pt"
    model.save(path)
    restored = ASEI.load(path,device="cpu")
    assert restored.thresholds_==model.thresholds_
