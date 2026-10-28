"""Run two small complete fits, including the 3604-point short-record case."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import numpy as np
import pandas as pd
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ts_benchmark.baselines.self_impl import ASEI
from ts_benchmark.baselines.self_impl.ASEI.asei import Config
from ts_benchmark.baselines.self_impl.ASEI.progress import IMPLEMENTATION_VERSION
from check_asei_runtime import check_recurrence


def normal_series(length,seed):
    x = np.random.default_rng(seed).normal(size=(length,3))
    for t in range(1,length):
        x[t] += .55*x[t-1]
    return pd.DataFrame(x,columns=["a","b","c"])


def validate(output,device="auto",training_profile="original"):
    torch.set_num_threads(2)
    overrides = dict(predictor_epochs=1,corrector_linear_steps=3,corrector_nonlinear_steps=3,score_epochs=2,
                     auxiliary_train_pairs=12,auxiliary_validation_pairs=48,device=device,
                     training_profile=training_profile)
    report = dict(implementation_version=IMPLEMENTATION_VERSION,
                  scope="Synthetic implementation verification only; no complete benchmark or detection-quality claim.",
                  preset_configuration=asdict(ASEI(training_profile=training_profile,device=device).config),
                  official_configuration=asdict(Config()),smoke_overrides=overrides,
                  cuda_available=torch.cuda.is_available(),entities=[])
    previous = None
    for name,length,seed in (("synthetic_A",4000,42),("synthetic_short",3604,43)):
        model = ASEI(**overrides,seed=seed).detect_fit(normal_series(length,seed))
        for record in model.metadata_["calibration"]:
            assert record["false_alarm_count"]<=record["K"]
            assert record["false_alarm_count"]==int((model.validation_scores_>record["threshold"]).sum())
        assert model.metadata_["corrector"]["maximum_beta_l1"]<=.950001
        assert model.metadata_["counters"]["score_epoch_feature_extractor_calls"]==0
        test = normal_series(250,91)
        whole = model.detect_score(test)[0]
        model.reset_state()
        streamed = np.concatenate([model.detect_stream(test.iloc[a:b])[1] for a,b in ((0,31),(31,90),(90,91),(91,250))])
        np.testing.assert_allclose(whole,streamed,atol=1e-5,rtol=1e-5)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"asei.pt"
            model.save(path)
            restored = ASEI.load(path,device=device)
            np.testing.assert_array_equal(restored.detect_score(test)[0],whole)
        if previous is not None:
            assert not np.shares_memory(previous.reference.errors,model.reference.errors)
            assert previous.corrector is not model.corrector and previous.scorer is not model.scorer
        summary = {key:value for key,value in model.metadata_.items() if key not in ("training_tasks","validation_tasks")}
        for key in ("training_tasks","validation_tasks"):
            tasks = model.metadata_[key]
            summary[key] = dict(count=len(tasks),sha256=hashlib.sha256(json.dumps(tasks,sort_keys=True).encode()).hexdigest(),
                                examples=tasks[:1])
        report["entities"].append(dict(entity=name,points=length,actual_configuration=asdict(model.config),
                                       metadata=summary,streaming_matches=True,reload_matches=True))
        print(json.dumps(dict(entity=name,calibration=model.metadata_["calibration"],
                              corrector=model.metadata_["corrector"],counters=model.metadata_["counters"])),flush=True)
        previous = model
    report["recurrence_correctness_and_timing"] = check_recurrence(previous.device,8,1024)
    report["unverified"] = ["Full datasets, benchmark scores and training convergence", "BF16 predictor training"]
    if previous.device.type!="cuda":
        report["unverified"].append("CUDA execution, GPU throughput and peak memory")
    output = Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(f"Validation report: {output}",flush=True)
    return report


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",default="result/asei_validation/summary.json")
    parser.add_argument("--device",default="auto")
    parser.add_argument("--training-profile",choices=["original","extended"],default="original")
    args = parser.parse_args()
    validate(args.output,args.device,args.training_profile)
