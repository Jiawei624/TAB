"""Capacity/budget selection uses Vsel; extended training preserves ASEI formulas."""
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from ts_benchmark.baselines.self_impl.ASEI import ASEI
from ts_benchmark.baselines.self_impl.ASEI.calibration import error_history
from ts_benchmark.baselines.self_impl.ASEI.data import chronological_split
from ts_benchmark.baselines.self_impl.ASEI.model import Predictor, cpu_state, lag_graph
from ts_benchmark.baselines.self_impl.ASEI.training import make_scheduler, score_optimizer
from test_asei_core import data


def tiny_options():
    return dict(training_profile="extended",hidden_dim_candidates=(4,8),
                predictor_epochs=2,corrector_linear_epochs=2,corrector_nonlinear_epochs=2,
                corrector_batch_size=127,score_epochs=3,auxiliary_train_pairs=12,
                auxiliary_validation_pairs=6,seed=42,device="cpu")


@pytest.fixture(scope="module")
def extended():
    return ASEI(**tiny_options()).detect_fit(data())


def test_profile_is_explicit_and_user_aliases_override_presets():
    original = ASEI(device="cpu").config
    assert original.training_profile=="original" and original.predictor_epochs==20
    assert original.score_epochs==10 and original.corrector_linear_epochs is None
    assert original.predictor_scheduler==original.score_scheduler=="none"
    config = ASEI(training_profile="extended",device="cpu").config
    assert config.hidden_dim_candidates==(32,64,128)
    assert config.predictor_epochs==60 and config.score_epochs==100
    assert config.learning_rate==3e-4 and config.score_bias_learning_rate==3e-2
    explicit = ASEI(training_profile="extended",device="cpu",lr=.002,num_epochs=7,
                    hidden_dim=24,corrector_linear_steps=11).config
    assert explicit.learning_rate==.002 and explicit.predictor_epochs==7
    assert explicit.hidden_dim==24 and explicit.hidden_dim_candidates is None
    assert explicit.corrector_linear_steps==11 and explicit.corrector_linear_epochs is None
    with pytest.raises(ValueError,match="Conflicting"):
        ASEI(training_profile="extended",lr=.001,learning_rate=.002)


def test_capacity_counts_for_swat_and_shared_temporal_topology():
    graph = lag_graph(np.zeros((100,51)),(1,2,4,8,16),4)
    for width,count in ((32,10886),(64,37542),(128,140006)):
        predictor = Predictor(51,*graph,hidden=width,win_size=63)
        assert sum(p.numel() for p in predictor.parameters())==count
        assert predictor.receptive_field==63 and len(predictor.encoder)==5


def test_complete_extended_fit_keeps_all_targets_and_every_fixed_task(extended):
    selection = extended.metadata_["capacity_selection"]
    rows = selection["results"]
    expected = min(enumerate(rows),key=lambda p:(p[1]["best_validation_huber"],p[1]["best_epoch"],p[0]))[1]
    assert extended.config.hidden_dim==selection["selected_hidden_dim"]==expected["hidden_dim"]
    for row in rows:
        assert all(item["training_targets"]==2482 and item["validation_targets"]==322 for item in row["history"])
    for row in extended.metadata_["selection_history"]:
        assert row["training_tasks_seen"]==12 and row["validation_components"]["tasks_seen"]==6
        assert len(row["learning_rates"])==2
    assert extended.metadata_["parameters"]["scorer"]==8
    assert extended.metadata_["counters"]==dict(normal_evidence_passes=2,changed_trajectory_builds=18,
                                               score_epoch_feature_extractor_calls=0)
    assert extended.metadata_["calibration"][0]["N"]==311
    assert extended.metadata_["calibration"][0]["false_alarm_count"]<=3
    assert extended.metadata_["frozen_numerical_preflight"]["passed"]
    assert set(extended.metadata_["normal_evidence_diagnostics"])=={"F","Vsel"}


def test_corrector_epoch_budget_covers_tail_and_reports_comparable_full_loss(extended):
    parts,_ = chronological_split(data().values)
    values = extended.preprocessor.transform(parts["Cfit"])[0]
    errors = extended._error_tensor(values)
    y = errors/extended.corrector.error_scale
    history = torch.from_numpy(error_history(y.numpy(),4))
    with torch.no_grad():
        for nonlinear,name in ((False,"corrector_linear"),(True,"corrector_nonlinear")):
            expected = F.huber_loss(extended.corrector(history,nonlinear=nonlinear),y[4:]).item()
            row = extended.metadata_["corrector"][name]
            assert row["updates"]==4 and row["processed_samples"]==238*2
            assert row["minimum_sample_visits"]==row["maximum_sample_visits"]==2
            assert row["full_training_huber_after"]==pytest.approx(expected,abs=1e-6)
            assert row["budget_unit"]=="complete_data_passes"
    assert extended.corrector.beta.abs().sum(-1).max()<=.950001


def test_epoch_corrector_phase_starts_complete_passes_after_partial_step_phase():
    options = dict(tiny_options(),hidden_dim=4,hidden_dim_candidates=None,predictor_epochs=1,score_epochs=1,
                   corrector_linear_epochs=None,corrector_linear_steps=1,corrector_batch_size=80)
    model = ASEI(**options).detect_fit(data())
    linear = model.metadata_["corrector"]["corrector_linear"]
    nonlinear = model.metadata_["corrector"]["corrector_nonlinear"]
    assert linear["processed_samples"]==80 and linear["minimum_sample_visits"]==0
    assert nonlinear["updates"]==6 and nonlinear["processed_samples"]==238*2
    assert nonlinear["minimum_sample_visits"]==nonlinear["maximum_sample_visits"]==2


def test_capacity_and_model_selection_ignore_vcal_and_reload_is_identical(extended,tmp_path):
    changed = data()
    changed.iloc[3600:] = changed.iloc[3600:]*100+1000
    other = ASEI(**tiny_options()).detect_fit(changed,train_label=object())
    assert extended.metadata_["capacity_selection"]==other.metadata_["capacity_selection"]
    assert extended.metadata_["selection_history"]==other.metadata_["selection_history"]
    for name in ("predictor","corrector","scorer"):
        for key,value in cpu_state(getattr(extended,name)).items():
            torch.testing.assert_close(value,cpu_state(getattr(other,name))[key],rtol=0,atol=0)
    np.testing.assert_array_equal(extended.reference.errors,other.reference.errors)
    assert extended.thresholds_!=other.thresholds_
    path = tmp_path/"extended.pt"
    extended.save(path)
    restored = ASEI.load(path,device="cpu")
    assert restored.config.hidden_dim==extended.config.hidden_dim
    assert restored.config.corrector_linear_epochs==2
    np.testing.assert_array_equal(restored.detect_score(data(190))[0],extended.detect_score(data(190))[0])


def test_capacity_ties_choose_earlier_checkpoint_then_candidate_order():
    model = ASEI(training_profile="extended",device="cpu",hidden_dim_candidates=(8,4),predictor_epochs=6,
                 predictor_min_epochs=1,predictor_patience=3,predictor_lr_patience=1)
    model.columns_ = ["a","b","c"]
    model.metadata_ = {}
    model._resolve_capacity(600)

    def loss(loader,optimizer=None,epoch=0):
        model._last_predictor_pass = dict(targets_seen=loader.dataset.target_count)
        if optimizer is not None:
            with torch.no_grad():
                model.predictor.embedding.weight.fill_(epoch)
            return 1.
        return 2. if model.predictor.input_projection.out_features==8 and epoch==1 else 1.

    with patch.object(model,"_predictor_loss",side_effect=loss):
        model._fit_predictor(data(600).values.astype(np.float32),data(500).values.astype(np.float32))
    assert model.config.hidden_dim==4 and model.metadata_["predictor"]["best_epoch"]==1
    torch.testing.assert_close(model.predictor.embedding.weight,torch.ones_like(model.predictor.embedding.weight))
    for row in model.metadata_["capacity_selection"]["results"]:
        assert row["stop_reason"]=="validation_patience"
        assert any(item["learning_rate"]<3e-4 for item in row["history"])


def test_scheduler_reduces_all_groups_and_bias_lr_does_not_change_inference():
    model = ASEI(training_profile="extended",device="cpu",score_lr_patience=1)
    scorer = model._new_scorer()
    optimizer = score_optimizer(scorer,model.config)
    identities = [id(p) for group in optimizer.param_groups for p in group["params"]]
    assert len(set(identities))==len(list(scorer.parameters()))
    before = [group["lr"] for group in optimizer.param_groups]
    scheduler = make_scheduler(optimizer,model.config,"score")
    for value in (2.,2.,2.):
        scheduler.step(value)
    assert [group["lr"] for group in optimizer.param_groups]==[rate*.5 for rate in before]
    features = torch.randn(1,2,50,3,4)
    scores,logits = scorer(features)
    with torch.no_grad():
        scorer.training_bias.add_(10)
    new_scores,new_logits = scorer(features)
    torch.testing.assert_close(new_scores,scores,atol=0,rtol=0)
    torch.testing.assert_close(new_logits,logits-40)


@pytest.mark.skipif(not torch.cuda.is_available(),reason="Requires one CUDA GPU")
def test_single_gpu_extended_fit_and_wide_frozen_frontend(tmp_path):
    from scripts.check_asei_runtime import check_frozen_evidence
    options = dict(tiny_options(),device="cuda:0",predictor_epochs=1,score_epochs=1)
    model = ASEI(**options).detect_fit(data())
    assert model.metadata_["frozen_numerical_preflight"]["passed"]
    assert all(p.device==torch.device("cuda:0") for p in model.predictor.parameters())
    for width in (64,128):
        assert check_frozen_evidence(model.device,51,hidden_dim=width)["evidence_consistency_passed"]
    model.save(tmp_path/"single_gpu.pt")
    restored = ASEI.load(tmp_path/"single_gpu.pt",device="cpu")
    assert restored.config.hidden_dim==model.config.hidden_dim
