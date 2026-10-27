"""Independent formula and indexing checks for ASEI 1.4."""
import copy
from dataclasses import asdict
import json
import numpy as np
import pandas as pd
import pytest
import torch

from ts_benchmark.baselines.self_impl.ASEI import ASEI
from ts_benchmark.baselines.self_impl.ASEI.asei import Config
from ts_benchmark.baselines.self_impl.ASEI.auxiliary import (
    IneffectivePerturbation, Task, TaskSampler, prepare_objective, objective_to_device, task_objectives)
from ts_benchmark.baselines.self_impl.ASEI.calibration import EvidenceReference, local_features_tensor, empirical_threshold
from ts_benchmark.baselines.self_impl.ASEI.data import BlockWindowDataset, CausalPreprocessor, chronological_split, collate_blocks
from ts_benchmark.baselines.self_impl.ASEI.model import EvidenceScorer, Predictor, lag_graph, project_l1_, temporal_layout
from ts_benchmark.baselines.self_impl.ASEI.protocol import select_protocol
from ts_benchmark.baselines.self_impl.ASEI.recurrence import integrate, max_affine_scan

torch.set_num_threads(2)
SMALL = dict(predictor_epochs=1, corrector_linear_steps=3, corrector_nonlinear_steps=3,
             score_epochs=3, auxiliary_train_pairs=12, auxiliary_validation_pairs=6, seed=42)


def data(length=4000, seed=17):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(length,3))
    for t in range(1,length):
        x[t] += .55*x[t-1]
    return pd.DataFrame(x,columns=["a","b","c"])


@pytest.mark.parametrize("length,chunk",[(1,1),(7,3),(31,8),(257,64)])
def test_scan_forward_and_all_gradients(length,chunk):
    torch.manual_seed(length)
    originals = [torch.randn(2,length,3,dtype=torch.float64),
                 torch.tensor(.9,dtype=torch.float64),torch.rand(2,3,dtype=torch.float64),
                 torch.tensor(.02,dtype=torch.float64)]
    weights = torch.randn(2,length,dtype=torch.float64)
    final_weight = torch.randn(2,3,dtype=torch.float64)
    results = []
    for backend in ("reference","scan"):
        u,decay,initial,kappa = [x.clone().requires_grad_() for x in originals]
        scores,state = integrate(u,decay,initial,kappa,backend,chunk)
        ((scores*weights).sum()+(state*final_weight).sum()).backward()
        results.append((scores,state,[x.grad for x in (u,decay,initial,kappa)]))
    torch.testing.assert_close(results[0][0],results[1][0],atol=1e-10,rtol=1e-10)
    torch.testing.assert_close(results[0][1],results[1][1],atol=1e-10,rtol=1e-10)
    for a,b in zip(results[0][2],results[1][2]):
        torch.testing.assert_close(a,b,atol=1e-10,rtol=1e-10)


def test_scan_zero_ties_and_unused_outputs():
    for loss_kind in ("scores","final","both"):
        results = []
        for backend in ("reference","scan"):
            u = torch.tensor([[[0.,0.],[1.,1.],[-1.,-1.],[.2,.2]]],dtype=torch.float64,requires_grad=True)
            decay = torch.tensor(.5,dtype=torch.float64,requires_grad=True)
            initial = torch.zeros(1,2,dtype=torch.float64,requires_grad=True)
            kappa = torch.zeros((),dtype=torch.float64,requires_grad=True)
            scores,state = integrate(u,decay,initial,kappa,backend,2)
            loss = scores.sum() if loss_kind=="scores" else state.sum() if loss_kind=="final" else scores.sum()+state.sum()
            loss.backward()
            results.append([x.grad for x in (u,decay,initial,kappa)])
        for actual,expected in zip(*results):
            torch.testing.assert_close(actual,expected,atol=1e-12,rtol=1e-12)
    u = torch.empty(1,0,2)
    initial = torch.ones(1,2)
    scores,state = integrate(u,torch.tensor(.9),initial,torch.tensor(.02))
    assert scores.shape==(1,0) and state is initial


def test_scan_gradcheck_and_long_fp32_bound():
    torch.manual_seed(21)
    args = (torch.randn(1,11,2,dtype=torch.float64,requires_grad=True),
            torch.tensor(.9,dtype=torch.float64,requires_grad=True),
            torch.rand(1,2,dtype=torch.float64,requires_grad=True),
            torch.tensor(.02,dtype=torch.float64,requires_grad=True))
    assert torch.autograd.gradcheck(lambda *x:integrate(*x,chunk_size=4),args)
    u = torch.randn(1,10000,3).clamp(-6,6)
    decay,initial,kappa = torch.tensor(.979),torch.zeros(1,3),torch.tensor(.011)
    a,s = integrate(u,decay,initial,kappa,"scan",256)
    b,t = integrate(u,decay,initial,kappa,"reference")
    torch.testing.assert_close(a,b,atol=5e-5,rtol=1e-5)
    torch.testing.assert_close(s,t,atol=5e-5,rtol=1e-5)
    _,_,state = max_affine_scan(torch.full((1,2048,1),6.),torch.tensor(.98),torch.zeros(1,1),torch.tensor(.01))
    assert 0<=state.item()<=299.501


def test_causal_fill_and_f_only_statistics():
    x = np.array([[np.nan,np.nan,7],[2,np.inf,7],[np.inf,np.nan,7],[4,np.nan,7.]])
    pre = CausalPreprocessor().fit(x)
    assert pre.all_missing.tolist()==[False,True,False]
    assert pre.initial.tolist()==[3,0,7]
    assert pre.scale[2]==1
    filled,_ = pre.fill(x)
    np.testing.assert_array_equal(filled[:,0],[3,2,2,4])
    stream = np.array([[1,np.nan,8],[np.nan,4,8],[np.inf,np.nan,np.nan]])
    whole,_ = pre.transform(stream)
    a,last = pre.transform(stream[:1])
    b,_ = pre.transform(stream[1:],last)
    np.testing.assert_array_equal(np.concatenate((a,b)),whole)
    np.testing.assert_allclose(pre.transform(stream[1:])[0][0],
                               [(pre.initial[0]-pre.mean[0])/pre.scale[0],4,1],rtol=1e-7)


def test_precise_splits_and_genesis_layout():
    parts,bounds = chronological_split(np.zeros((3604,3)))
    assert {k:len(v) for k,v in parts.items()}==dict(F=2306,Cfit=288,Rref=289,Vsel=360,Vcal=361)
    protocol,trial = select_protocol(Config(),89,8,len(parts["F"]),len(parts["Vsel"]))
    assert protocol.normal_prefix==33 and protocol.recovery_suffix==121
    assert protocol.length_cap==54 and protocol.maximum_raw_length==359 and trial["feasible"]
    assert protocol.training_lengths==(1,2,8,32,54)
    with pytest.raises(ValueError,match="cannot contain"):
        select_protocol(Config(),89,8,100,100)
    parts,_ = chronological_split(np.zeros((2883,3)),np.zeros((721,3)))
    assert len(parts["Vsel"])==360 and len(parts["Vcal"])==361


def test_capacity_aliases_and_legacy_errors():
    model = ASEI(lr=1e-4,num_epochs=1,win_size=100)
    assert model.history==115 and model.raw_history==126
    assert model.config.learning_rate==1e-4
    for n,expected in ((8191,(16,128,4,96)),(8192,(32,256,8,384))):
        model._resolve_capacity(n)
        assert tuple(getattr(model.config,k) for k in ("hidden_dim","predictor_block_size","batch_size","auxiliary_train_pairs"))==expected
    assert temporal_layout(3,(1,2,4,8,16),1)==((),1)
    assert temporal_layout(3,(1,2,4,8,16),100)[1]==100
    with pytest.raises(ValueError,match="Conflicting"):
        ASEI(lr=1e-3,learning_rate=1e-4)
    for option in ("multi_gpu","kappa_values","ar_max_iter","checkpoint_epochs","auxiliary_per_epoch"):
        with pytest.raises(ValueError,match="Unknown ASEI hyperparameters"):
            ASEI(**{option:1})


def test_network_size_blocks_tail_and_causality():
    for channels,hidden,expected in ((51,32,10886),(18,16,3306)):
        x = np.random.default_rng(2).normal(size=(300,channels)).astype(np.float32)
        network = Predictor(channels,*lag_graph(x,(1,2,4,8,16),4),hidden=hidden)
        assert sum(p.numel() for p in network.parameters())==expected
        assert all(layer.conv.groups==hidden for layer in network.encoder)
    x = np.random.default_rng(4).normal(size=(209,3)).astype(np.float32)
    dataset = BlockWindowDataset(x,78,64)
    assert sum(len(dataset[i][1]) for i in range(len(dataset)))==len(x)-78
    h,t,mask = collate_blocks([dataset[0],dataset[-1]])
    assert mask.sum()==64+3
    assert torch.equal(t[1,:3],torch.tensor(x[206:]))
    network = Predictor(3,*lag_graph(x,(1,2,4,8,16),4)).eval()
    with torch.no_grad():
        block = network(torch.from_numpy(x[:78+12-1])[None],12)
        points = torch.cat([network(torch.from_numpy(x[j:j+78])[None]) for j in range(12)],dim=1)
        torch.testing.assert_close(block,points,atol=2e-6,rtol=2e-6)
        changed = torch.from_numpy(x[:78+12-1].copy())[None]
        changed[:,82:] += 100
        torch.testing.assert_close(block[:,:5],network(changed,12)[:,:5],atol=1e-6,rtol=1e-6)


def test_graph_degenerate_negative_and_common_anchors():
    source = np.random.default_rng(8).normal(size=200)
    x = np.column_stack((source,np.r_[0,-source[:-1]],np.zeros(200)))
    sources,delays,mask = lag_graph(x,[1,2,4],2,37)
    assert sources[1,0]==0 and delays[1,0]==1 and mask[1,0]
    assert not mask[2].any()
    assert not lag_graph(np.ones((100,1)),[1,2],4)[2].any()
    # A repeated deterministic anchor selection cannot depend on the graph RNG.
    for a,b in zip(lag_graph(x,[1,2,4],2,37),lag_graph(x,[4,2,1],2,37)):
        np.testing.assert_array_equal(a,b)


def test_true_l1_projection_and_gate_constraints():
    x = torch.tensor([[3.,-1.,0.],[.1,-.2,.3]])
    project_l1_(x,2)
    torch.testing.assert_close(x,torch.tensor([[2.,0.,0.],[.1,-.2,.3]]))
    values = torch.randn(100,4)*8
    original = values.clone()
    project_l1_(values,.95)
    assert torch.all(values.abs().sum(-1)<=.950001)
    # KKT: all surviving entries lose the same absolute amount.
    for a,b in zip(original,values):
        survived = b!=0
        loss = a.abs()[survived]-b.abs()[survived]
        if len(loss)>1:
            torch.testing.assert_close(loss,loss[:1].expand_as(loss),atol=2e-6,rtol=2e-6)
    model = EvidenceScorer(6,.05,.8,.98,.9)
    assert sum(p.numel() for p in model.parameters())==8
    with torch.no_grad():
        model.gate_weight.fill_(3)
        model.gate_bias.fill_(5)
    model.project_()
    assert model.gate_weight.abs().sum()<=2.000001 and model.gate_bias==1
    model(torch.randn(2,2,100,3,4))[0].sum().backward()
    assert model.kappa_logit.grad is not None and model.kappa_logit.grad.abs()>0
    assert .01<float(model.kappa.detach())<.20


def test_fp32_reference_ties_and_local_rolling():
    d = np.tile(np.array([0,1,1,2],dtype=np.float32)[:,None,None],(1,2,2))
    ref = EvidenceReference(6).fit(d,d)
    query = np.tile(np.array([-1,1,3],dtype=np.float32)[:,None,None],(1,2,2))
    result = ref.transform_tensor(torch.from_numpy(query),torch.from_numpy(query))
    expected = -np.log(np.array([1.,.8,.2],dtype=np.float32))
    np.testing.assert_allclose((result+torch.tensor(ref.center))[:,0,0],expected,atol=1e-7)
    assert ref.errors.dtype==ref.residuals.dtype==ref.center.dtype==np.float32
    assert ref._tensor_errors.is_contiguous()
    np.testing.assert_allclose(ref.transform(d,d).mean(0),0,atol=1e-7)
    e = torch.randn(2,1000,3,2)
    full = local_features_tensor(e,8)
    short = local_features_tensor(e[:,400:600],8)
    torch.testing.assert_close(short,full[:,400:593],atol=1e-7,rtol=1e-7)
    with pytest.raises(ValueError,match="nonfinite"):
        EvidenceReference(6).fit(d*np.nan,d)


def test_threshold_signed_ties_zero_alpha():
    for scores in (np.linspace(-3,-1,1000),np.ones(201),np.array([-3,-2,-2,-2,-1.])):
        for alpha in (0,.01,.25):
            h,count = empirical_threshold(scores,alpha)
            assert h==np.sort(scores)[len(scores)-int(np.floor(alpha*len(scores)))-1]
            assert count==int(np.count_nonzero(scores>h))<=int(np.floor(alpha*len(scores)))
    assert empirical_threshold(-np.arange(200),.01)[0]<0


def test_fixed_tasks_resampling_metadata_and_layout_balance():
    cfg = Config()
    values = np.zeros((1400,3),dtype=np.float32)
    values[700:] = data(700).values
    protocol,_ = select_protocol(cfg,89,8,1400,1400)
    sampler = TaskSampler(values,89,8,cfg,protocol)
    task = Task(0,220,[(100,104)],2,0,17,pair_indices_seed=91)
    resolved,normal,changed = sampler.resolve(task)
    assert resolved.start>0 and resolved.base_attempts>0
    np.testing.assert_array_equal(normal[:100],changed[:100])
    assert np.any(normal[100]!=changed[100]) and np.any(normal[103]!=changed[103])
    reconstructed = Task(**json.loads(json.dumps(asdict(resolved))))
    np.testing.assert_array_equal(sampler.materialize(reconstructed)[1],changed)
    tasks = sampler.validation()
    assert len(tasks)==48
    assert all(sum(t.family==f and t.scenario==s for t in tasks)==8 for f in range(3) for s in ("single","recurrent"))
    assert [asdict(t) for t in tasks]==[asdict(t) for t in sampler.validation()]
    for t in tasks:
        if t.scenario=="recurrent":
            assert t.perturbations[0]["affected_channels"]==t.perturbations[1]["affected_channels"]
    flat = TaskSampler(np.zeros((1400,3),dtype=np.float32),89,8,cfg,protocol)
    with pytest.raises(IneffectivePerturbation,match="effective"):
        flat.resolve(Task(0,220,[(100,104)],2,0,17))


def test_loss_equal_per_task_instead_of_event_or_duration_pooling():
    tasks = [Task(0,50,[(10,11)],0,0,1),Task(0,50,[(10,20),(25,40)],1,1,2)]
    metadata = objective_to_device([prepare_objective(t,0,32,np.random.default_rng(i)) for i,t in enumerate(tasks)],"cpu")
    torch.manual_seed(3)
    scores = torch.randn(2,2,50,requires_grad=True)
    logits = scores/.25
    det,rank = task_objectives(scores,logits,metadata)
    for i in range(2):
        a,b = task_objectives(scores[i:i+1],logits[i:i+1],metadata[i:i+1])
        torch.testing.assert_close(a[0],det[i])
        torch.testing.assert_close(b[0],rank[i])
    # A batch of tasks always averages complete per-task objectives.
    ((det+rank).mean()).backward()
    assert scores.grad is not None and torch.isfinite(scores.grad).all()
