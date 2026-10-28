"""Read-only run reconciliation and frozen-model diagnostics; never select parameters."""
import ast
import hashlib
import math
import re

import numpy as np
import pandas as pd
import torch

from .asei import _digest_array
from .calibration import local_features_tensor, empirical_threshold
from .recurrence import integrate, max_affine_scan


def infer_point_rates(accuracy, precision, recall):
    """Infer normalized confusion counts for ONE population, not averaged reports."""
    if not (0<=accuracy<=1 and 0<precision<=1 and 0<=recall<=1):
        return None
    denominator = 1-2*recall+recall/precision
    if denominator<=0:
        return None
    prevalence = (1-accuracy)/denominator
    tp = recall*prevalence
    fp = tp*(1/precision-1)
    fn = prevalence-tp
    tn = 1-tp-fp-fn
    if min(tp,fp,fn,tn)<-1e-8 or not 0<prevalence<1:
        return None
    return dict(source="inferred from accuracy, precision, recall; not raw confusion counts",
                assumption="same binary population and conventional pointwise metrics",
                anomaly_prevalence=prevalence,predicted_positive_fraction=tp+fp,
                normal_false_positive_rate=fp/(1-prevalence),
                true_positive_fraction=tp,false_positive_fraction=fp,
                false_negative_fraction=fn,true_negative_fraction=tn)


def audit_supplied_run(log_path,metrics_path):
    text = log_path.read_text(encoding="utf-8-sig")
    predictor = []
    for epoch,train,valid,extra in re.findall(
            r"ASEI predictor epoch=(\d+) train=([\deE.+-]+) Vsel=([\deE.+-]+)([^\n]*)",text):
        row = dict(epoch=int(epoch),training_huber=float(train),validation_huber=float(valid))
        width = re.search(r"hidden=(\d+)",extra)
        if width:
            row["hidden_dim"] = int(width.group(1))
        predictor.append(row)
    scorer = [dict(epoch=int(epoch),validation_loss=float(loss),decay=float(decay),kappa=float(kappa))
              for epoch,loss,decay,kappa in re.findall(
                  r"ASEI scorer epoch=(\d+) Vsel_loss=([\deE.+-]+) lambda=([\deE.+-]+) kappa=([\deE.+-]+)",text)]
    result = dict(log_file=str(log_path),metric_file=str(metrics_path),predictor_history=predictor,
                  scorer_history=scorer,stage_seconds={key:float(value) for key,value in
                      re.findall(r"ASEI stage=(\w+) elapsed_s=([\d.]+)",text)})
    match = re.search(r"ASEI CUDA startup implementation=(\S+) torch=(\S+) device=(\S+)",text)
    if match:
        result["observed_runtime"] = dict(implementation_version=match.group(1),torch=match.group(2),device=match.group(3))
    match = re.search(r"Start loading (\d+) series",text)
    result["loaded_series_count"] = int(match.group(1)) if match else None
    match = re.search(r"splits=(\{[^\n]+?\}) capacity=",text)
    if match:
        result["splits"] = ast.literal_eval(match.group(1))
    match = re.search(r"ASEI selected=(\{[^\n]+?\}) calibration=(\[[^\n]+?\]) counters=(\{[^\n]+?\})",text)
    if match:
        result.update(selection=ast.literal_eval(match.group(1)),calibration=ast.literal_eval(match.group(2)),
                      counters=ast.literal_eval(match.group(3)))
    if predictor:
        result["predictor_best_epoch_from_log"] = min(predictor,key=lambda row:(row["validation_huber"],row["epoch"]))["epoch"]
        result["predictor_actual_epochs_from_log"] = predictor[-1]["epoch"]
        widths = sorted({row["hidden_dim"] for row in predictor if "hidden_dim" in row})
        if widths:
            result["predictor_by_capacity"] = {
                str(width):[row for row in predictor if row.get("hidden_dim")==width] for width in widths}
            result["predictor_total_epochs_all_candidates"] = len(predictor)
            chosen = re.search(r"ASEI predictor selected hidden=(\d+) parameters=\d+ best_epoch=(\d+)",text)
            if chosen:
                width,epoch = map(int,chosen.groups())
                result["predictor_selected_hidden_dim_from_log"] = width
                result["predictor_best_epoch_from_log"] = epoch
                result["predictor_actual_epochs_from_log"] = max(
                    row["epoch"] for row in predictor if row.get("hidden_dim")==width)
    if scorer:
        result["scorer_best_is_last"] = min(scorer,key=lambda row:(row["validation_loss"],row["epoch"]))["epoch"]==scorer[-1]["epoch"]
        result["scorer_validation_still_decreasing"] = len(scorer)>1 and scorer[-1]["validation_loss"]<scorer[-2]["validation_loss"]
    table = pd.read_csv(metrics_path)
    models = {}
    for column in table.columns:
        if column in ("strategy_args","metric_name"):
            continue
        values = {name:float(value) if math.isfinite(float(value)) else None
                  for name,value in zip(table["metric_name"],table[column])}
        item = dict(metrics=values)
        if (result["loaded_series_count"]==1 and
                all(values.get(key) is not None for key in ("accuracy","precision","recall"))):
            item["inferred_point_rates"] = infer_point_rates(values["accuracy"],values["precision"],values["recall"])
        else:
            item["inferred_point_rates"] = None
            item["rate_inference_limit"] = "Needs one confirmed binary population; means over series cannot be inverted safely."
        models[column] = item
    result["reported_models"] = models
    result["scope"] = "Log/configuration arithmetic and supplied measurements; no raw SWAT prediction reconstruction."
    return result


def training_fingerprint(values,columns):
    digest = hashlib.sha256(repr(list(columns)).encode())
    _digest_array(digest,np.asarray(values,dtype=np.float64))
    return digest.hexdigest()


def audit_preprocessing(model,fit_values):
    filled,_ = model.preprocessor.fill(np.asarray(fit_values,dtype=np.float64))
    constant = np.all(filled==filled[:1],axis=0)
    violations = constant & (model.preprocessor.scale!=1)
    return dict(actual_constant_columns=[str(model.columns_[i]) for i in np.flatnonzero(constant)],
                saved_constant_columns=[str(model.columns_[i]) for i in np.flatnonzero(model.preprocessor.constant)],
                constant_scale_violations=[dict(channel=str(model.columns_[i]),saved_scale=float(model.preprocessor.scale[i]),
                                                constant_value=float(filled[0,i]))
                                           for i in np.flatnonzero(violations)],
                action="Refit if constant_scale_violations is nonempty; do not replace a fitted scaler in place.")


@torch.no_grad()
def frozen_numeric_probe(model,standardized):
    """Compare batched/sliced evidence, including ECDF amplification of small roundoff."""
    length = min(len(standardized),model.raw_history+model.config.evidence_chunk_size)
    segment = standardized[:length]
    if length<=model.raw_history+16:
        return dict(skipped="segment too short for independent sliced histories")
    original_calls = getattr(model,"_extractor_calls",None)
    try:
        d = model._error_tensor(segment)
        r = model._residual_tensor(d)
        e = model._evidence_tensor(segment)
        offset = 13
        other_d = model._error_tensor(segment[offset:])
        other_r = model._residual_tensor(other_d)
        other_e = model._evidence_tensor(segment[offset:])
        batch_e = model._evidence_tensor(np.stack((segment,segment)))[0]
        features = local_features_tensor(e,model.config.rolling_window)[None]
        u = model.scorer.fuse(features)
        initial = u.new_zeros((1,u.shape[-1]))
        reference_scores,_ = integrate(u,model.scorer.decay,initial,model.scorer.kappa,"reference")
        scan_scores,_ = integrate(u,model.scorer.decay,initial,model.scorer.kappa,"scan",model.config.scan_chunk_size)
        differences = torch.stack(((d[offset:]-other_d).abs().amax((0,2)),
                                   (r[offset:]-other_r).abs().amax((0,2)),
                                   (e[offset:]-other_e).abs().amax((0,2)),
                                   (e-batch_e).abs().amax((0,2))),-1).cpu().numpy()
        maxima = differences.max(0)
        return dict(frozen_evidence_backend=model.config.frozen_evidence_backend,
                    raw_error_slice_max_abs=float(maxima[0]),residual_slice_max_abs=float(maxima[1]),
                    evidence_slice_max_abs=float(maxima[2]),evidence_batch_max_abs=float(maxima[3]),
                    evidence_consistency_passed=bool(max(maxima[2:])<=1e-5),
                    channels_above_evidence_tolerance=[dict(channel=str(model.columns_[i]),raw_error_slice_max_abs=float(row[0]),
                        residual_slice_max_abs=float(row[1]),evidence_slice_max_abs=float(row[2]),evidence_batch_max_abs=float(row[3]))
                        for i,row in enumerate(differences) if max(row[2:])>1e-5],
                    score_scan_reference_max_abs=float((scan_scores-reference_scores).abs().max().cpu()),
                    warning="An ECDF can amplify tiny raw-value differences at ties; diagnose on the deployment GPU.",
                    dtype="float32",reference_is_frozen=True)
    finally:
        if original_calls is None:
            model.__dict__.pop("_extractor_calls",None)
        else:
            model._extractor_calls = original_calls


@torch.no_grad()
def audit_frozen_sequence(model,data,labels=None):
    """Score an entire independent segment and attribute channels without fitting."""
    values = model._values(data)
    standardized,_ = model.preprocessor.transform(values)
    channels = len(model.columns_)
    cfg = model.config
    ready_count = max(0,len(values)-model.raw_history)
    raw_scores = np.full(len(values),-cfg.evidence_bound-cfg.kappa_max,dtype=np.float32)
    sums = torch.zeros((channels,2),device=model.device,dtype=torch.float64)
    caps = torch.zeros_like(sums)
    errors_above = torch.zeros_like(sums)
    residuals_above = torch.zeros_like(sums)
    dominance = torch.zeros(channels,device=model.device,dtype=torch.float64)
    state = None
    original_calls = getattr(model,"_extractor_calls",None)
    try:
        for begin in range(model.raw_history,len(values),cfg.evidence_chunk_size):
            end = min(len(values),begin+cfg.evidence_chunk_size)
            errors = model._error_tensor(standardized[begin-model.raw_history:end])
            residuals = model._residual_tensor(errors)
            evidence = model.reference.transform_tensor(errors[cfg.ar_order:],residuals)
            features = local_features_tensor(evidence,cfg.rolling_window)
            aligned_e = evidence[cfg.rolling_window-1:]
            aligned_d = errors[cfg.ar_order+cfg.rolling_window-1:]
            aligned_r = residuals[cfg.rolling_window-1:]
            sums += aligned_e.to(torch.float64).sum(0)
            # Recover uncentered g for attribution only; no alarm threshold is changed.
            g = aligned_e+model.reference._tensor_center
            caps += (g>=cfg.evidence_bound-2*torch.finfo(torch.float32).eps).sum(0)
            errors_above += (aligned_d>model.reference._tensor_errors[:,:,-1]).sum(0)
            residuals_above += (aligned_r>model.reference._tensor_residuals[:,:,-1]).sum(0)
            fused = model.scorer.fuse(features[None])
            if state is None:
                state = fused.new_zeros((1,channels))
            block_scores = []
            for block in fused.split(cfg.scan_chunk_size,dim=1):
                if cfg.recurrence_backend=="scan":
                    a,_,state = max_affine_scan(block,model.scorer.decay,state,model.scorer.kappa)
                else:
                    activations = []
                    for current in block.unbind(1):
                        a = model.scorer.decay*state+current-model.scorer.kappa
                        state = a.relu()
                        activations.append(a)
                    a = torch.stack(activations,1)
                maximum = a.amax(-1,keepdim=True)
                ties = (a==maximum).to(torch.float64)
                dominance += (ties/ties.sum(-1,keepdim=True)).sum((0,1))
                block_scores.append(maximum.squeeze(-1))
            raw_scores[begin:end] = torch.cat(block_scores,1)[0].cpu().numpy()
        scores = model._output_scores(raw_scores)
        ready = scores[model.raw_history:]
        denominator = max(1,ready_count)
        stats = dict(points=len(values),ready_scores=ready_count,warmup_points=min(len(values),model.raw_history),
                     score_quantiles=dict(zip(("min","median","p90","p99","p999","max"),
                                              np.quantile(ready,[0,.5,.9,.99,.999,1]).tolist())) if len(ready) else {},
                     empirical_alarms={},channels=[])
        means,cap_rates,d_rates,r_rates,dominance = [value.cpu().numpy()/denominator for value in
                                                    (sums,caps,errors_above,residuals_above,dominance)]
        for i,name in enumerate(model.columns_):
            stats["channels"].append(dict(channel=str(name),mean_corrected_evidence=float(means[i,0]),
                mean_state_evidence=float(means[i,1]),corrected_cap_fraction=float(cap_rates[i,0]),
                state_cap_fraction=float(cap_rates[i,1]),d_above_reference_max=d_rates[i].tolist(),
                r_above_reference_max=r_rates[i].tolist(),maximum_score_dominance=float(dominance[i]),
                large_standardized_value_fraction=float(np.mean(np.abs(standardized[:,i])>6))))
        for alpha,threshold in model.thresholds_.items():
            predictions = np.zeros(len(values),dtype=bool)
            predictions[model.raw_history:] = ready>threshold
            item = dict(alpha=float(alpha),threshold=float(threshold),alarms=int(predictions.sum()),
                        ready_alarm_fraction=float(predictions.sum()/denominator),
                        all_points_alarm_fraction=float(predictions.sum()/max(1,len(values))))
            if labels is not None:
                truth = np.asarray(labels).reshape(-1)
                if len(truth)!=len(values) or not np.isin(truth,[0,1]).all():
                    raise ValueError("Post-hoc audit labels must be binary and aligned")
                positive = truth==1
                tp,fp = int(np.sum(predictions&positive)),int(np.sum(predictions&~positive))
                fn,tn = int(np.sum(~predictions&positive)),int(np.sum(~predictions&~positive))
                item.update(posthoc_confusion=dict(tp=tp,fp=fp,fn=fn,tn=tn),
                            normal_false_positive_rate=fp/max(1,fp+tn),
                            recall=tp/max(1,tp+fn),precision=tp/max(1,tp+fp))
            stats["empirical_alarms"][str(alpha)] = item
        stats["top_dominant_channels"] = sorted(stats["channels"],key=lambda row:-row["maximum_score_dominance"])[:5]
        return stats,scores
    finally:
        if original_calls is None:
            model.__dict__.pop("_extractor_calls",None)
        else:
            model._extractor_calls = original_calls


def compare_calibration(model,scores):
    ready = np.asarray(scores)[model.raw_history:]
    results = []
    for alpha,saved in model.thresholds_.items():
        h,count = empirical_threshold(ready,alpha)
        results.append(dict(alpha=alpha,N=len(ready),recomputed_threshold=h,saved_threshold=saved,
                            threshold_abs_difference=abs(h-saved),recomputed_false_alarms=count,
                            calibration_scores_max_abs_difference=float(np.max(np.abs(ready-model.validation_scores_)))
                                if len(ready)==len(model.validation_scores_) else None))
    return results
