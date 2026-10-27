"""Run two small synthetic ASEI fits and record effective protocols and AR status."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ts_benchmark.baselines.self_impl import ASEI
from ts_benchmark.baselines.self_impl.ASEI.asei import Config
from ts_benchmark.baselines.self_impl.ASEI.protocol import PROTOCOL_SCALES, make_protocol
from check_asei_runtime import check_recurrence


def normal_series(length, phase=0):
    time = np.arange(length, dtype=np.float64) + phase
    return pd.DataFrame({"a": np.sin(time / 7) + 0.1 * np.cos(time / 3),
                         "b": np.cos(time / 11) + 0.2 * np.sin(time / 7)})


def validate(output, device="auto"):
    torch.set_num_threads(2)
    defaults = Config()
    table = []
    for factor in PROTOCOL_SCALES:
        protocol = make_protocol(defaults, 89, 8, factor)
        training, scenarios = protocol.required_lengths()
        table.append(dict(protocol=asdict(protocol), training_points_required=training,
                          validation_points_required=scenarios))
    report = dict(
        scope="Synthetic implementation verification only; no full benchmark experiment.",
        official_configuration=asdict(defaults),
        protocol_table_T0_8=table,
        smoke_overrides=dict(predictor_epochs=1, correction_epochs=1, score_epochs=1,
                             checkpoint_epochs=[1], auxiliary_per_epoch=2,
                             auxiliary_batch_size=2, validation_seeds=[1001], device=device),
        cuda_available=torch.cuda.is_available(),
        entities=[],
    )
    previous = None
    for entity, length, seed, phase in [("synthetic_A", 5000, 3, 0), ("synthetic_B", 5500, 4, 1.5)]:
        model = ASEI(**dict(report["smoke_overrides"], seed=seed))
        model.detect_fit(normal_series(length, phase))
        meta = model.metadata_
        ar = {key: value for key, value in meta["corrector"]["ar_optimization"].items() if key != "trace"}
        for calibration in meta["calibration"]:
            assert calibration["false_alarm_count"] <= calibration["K"]
            assert calibration["false_alarm_count"] == int(np.count_nonzero(model.validation_scores_ > calibration["threshold"]))
        assert meta["corrector"]["maximum_beta_l1"] <= 0.95 + 1e-12
        assert ar["converged"] == (ar["gradient_mapping_inf"] <= 1e-5)
        assert len(meta["scoring_candidates"]) == 6
        if previous is not None:
            assert not np.shares_memory(previous.reference.errors, model.reference.errors)
            assert previous.corrector.mlp[0].weight.data_ptr() != model.corrector.mlp[0].weight.data_ptr()
        report["entities"].append(dict(
            entity=entity, points=length, seed=seed, actual_configuration=asdict(model.config),
            data_sha256=meta["data_sha256"], splits=meta["splits"], protocol=meta["auxiliary_protocol"],
            protocol_feasibility=meta["protocol_feasibility"], reference_points=meta["reference_points"],
            calibration=meta["calibration"], linear_ar=ar,
            execution_device=str(model.device), reference_warmup=meta["reference_warmup"],
            score_warmup=meta["score_warmup"], validation_evidence_builds=meta["validation_evidence_builds"],
            validation_cache_resident_bytes=meta["validation_cache_resident_bytes"],
            maximum_beta_l1=meta["corrector"]["maximum_beta_l1"],
            scoring_candidates=meta["scoring_candidates"], selection=meta["selection"],
            saved_scoring_checkpoints=meta["saved_scoring_checkpoints"], versions=meta["versions"],
        ))
        previous = model
        print(json.dumps(dict(entity=entity, s=model.protocol_.s, calibration=meta["calibration"],
                              ar_converged=ar["converged"], ar_iterations=ar["iterations"],
                              ar_mapping_inf=ar["gradient_mapping_inf"]), ensure_ascii=False), flush=True)
    report["recurrence_correctness_and_timing"] = check_recurrence(previous.device, 8, 256)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Validation report: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="result/asei_validation/summary.json")
    parser.add_argument("--device", default="auto", help="auto selects one CUDA device when available; otherwise CPU")
    args = parser.parse_args()
    validate(args.output, args.device)
