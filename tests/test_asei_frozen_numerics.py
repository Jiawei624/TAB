"""Regression for small shape-dependent roundoff amplified at empirical-tail atoms."""
import copy
import numpy as np
import pytest
import torch
from torch import nn

from ts_benchmark.baselines.self_impl.ASEI import ASEI
from ts_benchmark.baselines.self_impl.ASEI.calibration import EvidenceReference, empirical_threshold
from ts_benchmark.baselines.self_impl.ASEI.calibration import error_history
from ts_benchmark.baselines.self_impl.ASEI.model import ErrorCorrector
from test_asei_audit import frozen_audit_case


class ShapeSensitivePredictor(nn.Module):
    """Emulate a one-ULP GEMM difference without needing CUDA on the test host."""
    def __init__(self,sensitive):
        super().__init__()
        self.sensitive = sensitive
        self.calls = []

    def forward(self,history,output_steps=1):
        self.calls.append((tuple(history.shape),output_steps))
        value = history.new_tensor(.5)
        if self.sensitive and (history.shape[0]!=1 or output_steps!=32):
            value = torch.nextafter(value,history.new_tensor(float("inf")))
        return value.expand(history.shape[0],output_steps,history.shape[2],2).clone()


class ShapeSensitiveCorrector(ErrorCorrector):
    def __init__(self,sensitive):
        super().__init__(1,4,.95)
        self.sensitive = sensitive
        self.calls = []

    def forward(self,history,nonlinear=True):
        self.calls.append(tuple(history.shape))
        value = history.new_tensor(.125)
        if self.sensitive and (history.shape[0]!=1 or history.shape[1]!=32):
            # Two prediction ULPs avoid a midpoint tie when subtracting from .25;
            # the resulting residual differs by one ULP from the reference atom.
            for _ in range(2):
                value = torch.nextafter(value,history.new_tensor(float("-inf")))
        return value.expand(history.shape[:-1]).clone()


def atom_model(backend,cause):
    model = ASEI(device="cpu",evidence_chunk_size=32,frozen_evidence_backend=backend)
    model.columns_ = ["pump"]
    model.metadata_ = {}
    model.predictor = ShapeSensitivePredictor(cause=="predictor")
    model.corrector = ShapeSensitiveCorrector(cause=="corrector")
    # 640 reference points; a query one ULP above the sole atom attains cap 6.
    normal = np.zeros((model.history+model.config.ar_order+640,1),dtype=np.float32)
    if cause=="predictor":
        normal = normal[:-model.config.ar_order]  # Full legacy prediction blocks.
    d = model._error_tensor(normal)
    r = model._residual_tensor(d)
    model.reference = EvidenceReference(6).fit(d[4:].numpy(),r.numpy())
    return model,normal


@pytest.mark.parametrize("cause",["predictor","corrector"])
def test_legacy_roundoff_reproduces_six_unit_evidence_error(cause):
    model,normal = atom_model("legacy",cause)
    with pytest.raises(RuntimeError,match="frozen evidence differs"):
        model._check_frozen_evidence_consistency(normal)
    record = model.metadata_["frozen_numerical_preflight"]
    assert not record["passed"]
    assert max(record["evidence_slice_max_abs"],record["evidence_batch_max_abs"])==6.


@pytest.mark.parametrize("cause",["predictor","corrector"])
def test_canonical_geometry_removes_roundoff_amplification_without_changing_ecdf(cause):
    model,normal = atom_model("canonical",cause)
    model._check_frozen_evidence_consistency(normal)
    record = model.metadata_["frozen_numerical_preflight"]
    assert record["passed"] and record["evidence_slice_max_abs"]==record["evidence_batch_max_abs"]==0
    assert model.reference.errors.dtype==model.reference.residuals.dtype==np.float32
    # Strict upper-tail semantics stay intact: a genuine nextafter query is still above the atom.
    atom = torch.tensor(model.reference.errors[-1:])
    query = torch.nextafter(atom,torch.full_like(atom,float("inf")))
    residual = torch.tensor(model.reference.residuals[-1:])
    assert model.reference.transform_tensor(query,residual)[0,0,1]==6
    assert all(shape[0]==1 and steps==43 for shape,steps in model.predictor.calls)
    assert all(shape==(1,39,1,2,4) for shape in model.corrector.calls)
    assert len(model.reference.errors)==len(normal)-model.history-4


def test_future_computation_padding_does_not_create_history_or_reference_samples():
    model,normal = atom_model("canonical","predictor")
    assert model._error_tensor(normal[:model.history]).shape==(0,1,2)
    short = model._error_tensor(normal[:model.history+3])
    assert short.shape==(3,1,2)
    assert model._residual_tensor(short).shape==(0,1,2)
    values = np.zeros((3,model.history+71,1),dtype=np.float32)
    errors = model._error_tensor(values)
    assert errors.shape==(3,71,1,2)
    assert model._residual_tensor(errors).shape==(3,67,1,2)
    scores = np.arange(71-model.config.ar_order-model.config.rolling_window+1,dtype=np.float32)
    _,count = empirical_threshold(scores,.01)
    assert count<=int(.01*len(scores))


def test_new_checkpoint_persists_canonical_and_historical_checkpoints_keep_legacy(frozen_audit_case,tmp_path):
    model,_,table = frozen_audit_case
    assert model.config.frozen_evidence_backend=="canonical"
    path = tmp_path/"canonical.pt"
    model.save(path)
    restored = ASEI.load(path,device="cpu")
    assert restored.config.frozen_evidence_backend=="canonical"
    sample = table.iloc[4000:].drop(columns="label")
    np.testing.assert_array_equal(restored.detect_score(sample)[0],model.detect_score(sample)[0])
    # A historical payload lacks the backend field; loading must not silently
    # query its old FP32 references with a newly selected numerical policy.
    legacy = copy.deepcopy(model)
    legacy.config.frozen_evidence_backend = "legacy"
    legacy.save(path)
    payload = torch.load(path,weights_only=False)
    payload["config"].pop("frozen_evidence_backend")
    payload["requested_config"].pop("frozen_evidence_backend",None)
    payload["metadata"]["implementation_version"] = "asei-finite-bank-scan-3.1"
    torch.save(payload,path)
    restored = ASEI.load(path,device="cpu")
    assert restored.config.frozen_evidence_backend=="legacy"
    np.testing.assert_array_equal(restored.detect_score(sample)[0],legacy.detect_score(sample)[0])


def test_fixed_future_slots_preserve_real_causal_prediction_and_residual_formulas(frozen_audit_case):
    model,_,table = frozen_audit_case
    values = model.preprocessor.transform(table.iloc[2880:3200].drop(columns="label").values)[0]
    tensor = torch.from_numpy(values)
    with torch.no_grad():
        predicted = model.predictor(tensor[:-1][None],output_steps=len(tensor)-model.history)[0]
        expected_errors = (tensor[model.history:,:,None]-predicted).square()
        actual_errors = model._error_tensor(values)
        torch.testing.assert_close(actual_errors,expected_errors,atol=2e-6,rtol=2e-6)
        y = expected_errors/model.corrector.error_scale
        history = torch.from_numpy(error_history(y.numpy(),model.config.ar_order))
        expected_residual = model.corrector.error_scale*(y[model.config.ar_order:]-model.corrector(history))
        torch.testing.assert_close(model._residual_tensor(actual_errors),expected_residual,atol=2e-6,rtol=2e-6)
