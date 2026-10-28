"""Reconcile supplied logs/reports and diagnose a frozen checkpoint without retraining."""
import argparse
import json
import logging
import os
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-log",type=Path)
    parser.add_argument("--metrics-csv",type=Path)
    parser.add_argument("--checkpoint",type=Path)
    parser.add_argument("--data-name",help="TAB long-format dataset filename, e.g. SWAT.csv")
    parser.add_argument("--dataset-root",type=Path)
    parser.add_argument("--include-test",action="store_true",help="Post-hoc frozen test diagnostics; never tune a threshold")
    parser.add_argument("--gpus",type=int,nargs=1,default=[1])
    parser.add_argument("--device",default="auto")
    parser.add_argument("--output",type=Path,default=Path("result/asei_audit/audit.json"))
    args = parser.parse_args()
    if bool(args.training_log)!=bool(args.metrics_csv):
        parser.error("--training-log and --metrics-csv must be supplied together")
    if args.data_name and not args.checkpoint:
        parser.error("--data-name requires --checkpoint")
    if args.include_test and not (args.data_name and args.checkpoint):
        parser.error("--include-test requires --checkpoint and --data-name")
    if not (args.training_log or args.checkpoint) or args.gpus[0]<0:
        parser.error("Supply logs/reports or a checkpoint, and a nonnegative GPU ID")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpus[0])
    import numpy as np
    import pandas as pd
    import torch
    from ts_benchmark.baselines.self_impl.ASEI import ASEI
    from ts_benchmark.baselines.self_impl.ASEI.audit import (
        audit_supplied_run, audit_frozen_sequence, audit_preprocessing,
        frozen_numeric_probe, compare_calibration, training_fingerprint)
    from ts_benchmark.baselines.self_impl.ASEI.data import chronological_split
    from ts_benchmark.baselines.self_impl.ASEI.progress import IMPLEMENTATION_VERSION
    from ts_benchmark.common.constant import ANOMALY_DETECT_DATASET_PATH
    from ts_benchmark.data.utils import read_data
    logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(message)s")
    torch.set_num_threads(2)
    report = dict(audit_runtime_version=IMPLEMENTATION_VERSION,
                  scope="Read-only diagnosis. No training, model selection or threshold update.",
                  cuda_available=torch.cuda.is_available())
    if args.training_log:
        report["supplied_run"] = audit_supplied_run(args.training_log,args.metrics_csv)
    if args.checkpoint:
        model = ASEI.load(args.checkpoint,device=args.device)
        report["checkpoint"] = dict(path=str(args.checkpoint),device=str(model.device),config=vars(model.config),
            fitted_version=model.metadata_.get("implementation_version"),selection=model.metadata_.get("selection"),
            corrector=model.metadata_.get("corrector"),calibration=model.metadata_.get("calibration"),
            data_coverage=model.metadata_.get("data_coverage"),
            learned_scorer=dict(decay=float(model.scorer.decay.cpu()),kappa=float(model.scorer.kappa.cpu()),
                               training_bias=float(model.scorer.training_bias.cpu()),
                               gate_bias=float(model.scorer.gate_bias.cpu()),
                               gate_weight=model.scorer.gate_weight.cpu().tolist()))
        if args.data_name:
            if model.metadata_["splits"]["external_validation"]:
                raise ValueError("This CLI expects the TAB internal validation split; external validation needs its original separate input")
            root = args.dataset_root or Path(ANOMALY_DETECT_DATASET_PATH)
            metadata = pd.read_csv(root/"DETECT_META.csv").set_index("file_name")
            train_length = metadata.loc[args.data_name,"train_lens"]
            if not np.isfinite(train_length) or int(train_length)!=train_length:
                raise ValueError("train_lens must be an integer")
            table = read_data(str(root/"data"/args.data_name))
            train_length = int(train_length)
            if not 0<train_length<len(table):
                raise ValueError("Official train_lens must preserve a nonempty train/test split")
            if isinstance(table.index,pd.DatetimeIndex):
                timestamp_summary = {}
                for name,index in (("training",table.index[:train_length]),("test",table.index[train_length:])):
                    row = dict(missing_timestamps=int(index.isna().sum()),duplicate_timestamps=int(index.duplicated().sum()))
                    if len(index)>1 and not row["missing_timestamps"]:
                        delta = np.diff(index.asi8)
                        unique,counts = np.unique(delta,return_counts=True)
                        mode = unique[np.argmax(counts)]
                        row.update(modal_step_ns=int(mode),irregular_step_count=int(np.count_nonzero(delta!=mode)),
                                   nonpositive_step_count=int(np.count_nonzero(delta<=0)))
                    timestamp_summary[name] = row
                report["source_timestamp_diagnostics"] = timestamp_summary
                report["timestamp_diagnostic_scope"] = "Reports raw timestamp steps; does not infer or alter entity boundaries."
            features = table.drop(columns="label")
            training = features.iloc[:train_length]
            model._values(training)  # Check exact feature names/order before hashing.
            fingerprint = training_fingerprint(training.values,training.columns)
            if fingerprint!=model.metadata_["data_sha256"]:
                raise ValueError("Dataset fingerprint does not match this checkpoint; do not diagnose it as the original run")
            parts,_ = chronological_split(training.values)
            label_parts,_ = chronological_split(table["label"].iloc[:train_length].to_numpy())
            report["posthoc_training_label_counts"] = {
                name:{str(label):int(count) for label,count in zip(*np.unique(part,return_counts=True))}
                for name,part in label_parts.items()}
            report["training_labels_are_audited_only"] = True
            report["data_fingerprint_matches"] = True
            report["preprocessing"] = audit_preprocessing(model,parts["F"])
            report["frozen_numeric_probe"] = frozen_numeric_probe(
                model,model.preprocessor.transform(parts["Rref"])[0])
            report["sequences"] = {}
            for name,part in parts.items():
                logging.info("ASEI audit independent segment=%s points=%d",name,len(part))
                # Restore feature names for the ordinary model input contract.
                stats,scores = audit_frozen_sequence(model,pd.DataFrame(part,columns=features.columns))
                report["sequences"][name] = stats
                if name=="Vcal":
                    report["calibration_reproduction"] = compare_calibration(model,scores)
            if args.include_test:
                logging.info("ASEI audit frozen test; labels are used only for post-hoc confusion counts")
                stats,_ = audit_frozen_sequence(model,features.iloc[train_length:],
                                                table["label"].iloc[train_length:].values)
                report["sequences"]["test"] = stats
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(f"ASEI read-only audit: {args.output}",flush=True)
    if "supplied_run" in report:
        for name,item in report["supplied_run"]["reported_models"].items():
            print(json.dumps(dict(model=name,inferred_point_rates=item.get("inferred_point_rates")),ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
