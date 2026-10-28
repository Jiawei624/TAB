"""TAB adapter for the finite-bank ASEI method in In1_1_4.md."""
import copy
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import json
import logging
import math
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .auxiliary import TaskSampler, characteristic_scale, prepare_objective, task_objectives
from .cache import EvidenceBank
from .calibration import EvidenceReference, empirical_threshold, local_features_tensor
from .data import BlockWindowDataset, CausalPreprocessor, chronological_split, collate_blocks
from .model import ErrorCorrector, EvidenceScorer, Predictor, cpu_state, lag_graph, temporal_layout
from .optimization import fit_corrector
from .protocol import AuxiliaryProtocol, select_protocol
from .progress import IMPLEMENTATION_VERSION, Progress

logger = logging.getLogger(__name__)


@dataclass
class Config:
    seed: int = 2026
    device: str = "auto"
    lr: object = None
    num_epochs: object = None
    win_size: int = 63
    hidden_dim: object = None
    embedding_dim: int = 8
    kernel_size: int = 3
    dilations: tuple = (1,2,4,8,16)
    graph_lags: tuple = (1,2,4,8,16)
    graph_neighbors: int = 4
    graph_sample_cap: int = 65536
    predictor_block_size: object = None
    batch_size: object = None
    predictor_epochs: int = 20
    predictor_min_epochs: int = 5
    predictor_patience: int = 4
    learning_rate: float = 1e-3
    predictor_weight_decay: float = 1e-4
    predictor_amp: str = "off"
    num_workers: int = 0
    ar_order: int = 4
    rho: float = .95
    corrector_batch_size: int = 2048
    corrector_linear_steps: int = 100
    corrector_linear_lr: float = 1e-2
    corrector_nonlinear_steps: int = 100
    corrector_nonlinear_lr: float = 1e-3
    error_scale_floor: float = 1e-4
    nonlinear_iqr_fraction: float = .1
    evidence_bound: float = 6.
    rolling_window: int = 8
    omega_min: float = .05
    lambda_min: float = .8
    lambda_max: float = .98
    lambda_init: float = .9
    kappa_min: float = .01
    kappa_max: float = .20
    kappa_init: float = .02
    auxiliary_train_pairs: object = None
    auxiliary_validation_pairs: int = 48
    auxiliary_batch_size: int = 8
    hard_pool_size: int = 16
    perturbation_attempts: int = 128
    rank_pairs_per_event: int = 32
    classification_temperature: float = .25
    ranking_temperature: float = 1.
    ranking_margin: float = 1.
    ranking_weight: float = 1.
    score_epochs: int = 10
    score_min_epochs: int = 3
    score_patience: int = 3
    score_learning_rate: float = 1e-3
    score_gradient_clip: float = 1.
    evidence_chunk_size: int = 1024
    scan_chunk_size: int = 256
    recurrence_backend: str = "scan"
    evidence_device_cache_mb: float = 256.
    evidence_host_cache_mb: float = 1024.
    cache_dir: object = None
    alpha: object = .01
    anomaly_ratio: object = None
    log_interval_seconds: float = 30.
    checkpoint_dir: object = None
    score_transform: str = "raw"


def _digest_array(digest, values):
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    values = np.ascontiguousarray(values)
    digest.update(str((values.shape,str(values.dtype))).encode())
    digest.update(memoryview(values).cast("B"))


class ASEI:
    model_name = "ASEI"

    def __init__(self, **kwargs):
        unknown = set(kwargs)-set(Config.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown ASEI hyperparameters: {sorted(unknown)}. "
                             "ASEI 1.4 uses one trainable-kappa scorer and fixed-budget corrector; "
                             "remove legacy grid, backtracking and multi_gpu options.")
        options = dict(kwargs)
        for alias,target in (("lr","learning_rate"),("num_epochs","predictor_epochs")):
            if options.get(alias) is not None:
                if target in options and options[target]!=options[alias]:
                    raise ValueError(f"Conflicting ASEI hyperparameters: {alias} and {target}")
                options[target] = options[alias]
        self.config = Config(**options)
        self._requested_config = asdict(self.config)
        self._validate_config()
        cfg = self.config
        self.temporal_layout_, self.receptive_field = temporal_layout(cfg.kernel_size,cfg.dilations,cfg.win_size)
        self.win_size = self.receptive_field
        self.history = self.receptive_field+max(cfg.graph_lags)-1
        self.raw_history = self.history+cfg.ar_order+cfg.rolling_window-1
        self.seed = cfg.seed
        device = ("cuda:0" if torch.cuda.is_available() else "cpu") if cfg.device=="auto" else cfg.device
        self.device = torch.device(device)
        if self.device.type not in ("cpu","cuda"):
            raise ValueError("ASEI supports a single CUDA device or CPU")
        if self.device.type=="cuda" and (not torch.cuda.is_available() or
                                         (self.device.index or 0)>=torch.cuda.device_count()):
            raise ValueError("Requested CUDA device unavailable; device='auto' enables CPU fallback")
        if self.device.type=="cuda":
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        self.preprocessor = CausalPreprocessor()
        self._fitted = False

    def _validate_config(self):
        cfg = self.config
        optional = ("hidden_dim","predictor_block_size","batch_size","auxiliary_train_pairs")
        positive = ("win_size","embedding_dim","kernel_size","graph_neighbors","graph_sample_cap",
                    "predictor_epochs","predictor_min_epochs","predictor_patience","ar_order","rolling_window",
                    "corrector_batch_size","corrector_linear_steps","corrector_nonlinear_steps",
                    "auxiliary_validation_pairs","auxiliary_batch_size","hard_pool_size","perturbation_attempts",
                    "rank_pairs_per_event","score_epochs","score_min_epochs","score_patience",
                    "evidence_chunk_size","scan_chunk_size")
        for name in positive+optional:
            value = getattr(cfg,name)
            if name in optional and value is None:
                continue
            if isinstance(value,bool) or not isinstance(value,int) or value<1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("dilations","graph_lags"):
            value = getattr(cfg,name)
            if not value or any(isinstance(v,bool) or not isinstance(v,int) or v<1 for v in value):
                raise ValueError(f"{name} must contain positive integers")
        for name in ("learning_rate","corrector_linear_lr","corrector_nonlinear_lr","error_scale_floor",
                     "evidence_bound","classification_temperature","ranking_temperature","ranking_margin",
                     "score_learning_rate","score_gradient_clip","log_interval_seconds"):
            value = getattr(cfg,name)
            if isinstance(value,bool) or not np.isfinite(value) or value<=0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("evidence_device_cache_mb","evidence_host_cache_mb","predictor_weight_decay","nonlinear_iqr_fraction"):
            value = getattr(cfg,name)
            if not np.isfinite(value) or value<0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not 0<cfg.lambda_min<cfg.lambda_init<cfg.lambda_max<1:
            raise ValueError("Require 0 < lambda_min < lambda_init < lambda_max < 1")
        if not 0<=cfg.kappa_min<cfg.kappa_init<cfg.kappa_max<cfg.evidence_bound:
            raise ValueError("Require 0 <= kappa_min < kappa_init < kappa_max < evidence_bound")
        if not 0<cfg.rho<1 or not 0<cfg.omega_min<.5:
            raise ValueError("Invalid rho or omega_min")
        if cfg.ranking_weight!=1:
            raise ValueError("The ASEI main method fixes ranking_weight=1")
        if isinstance(cfg.seed,bool) or not isinstance(cfg.seed,int) or cfg.seed<0:
            raise ValueError("seed must be a nonnegative integer")
        if isinstance(cfg.num_workers,bool) or not isinstance(cfg.num_workers,int) or cfg.num_workers<0:
            raise ValueError("num_workers must be a nonnegative integer")
        if cfg.recurrence_backend not in ("scan","reference"):
            raise ValueError("recurrence_backend must be scan or reference")
        if cfg.predictor_amp not in ("off","bf16") or cfg.score_transform not in ("raw","bounded"):
            raise ValueError("Invalid predictor_amp or score_transform")
        for name in ("auxiliary_train_pairs","auxiliary_validation_pairs"):
            if getattr(cfg,name) is not None and getattr(cfg,name)%6:
                raise ValueError(f"{name} must be a multiple of 6")
        ratios = cfg.anomaly_ratio
        alphas = cfg.alpha if isinstance(cfg.alpha,(tuple,list)) else [cfg.alpha]
        if ratios is not None:
            ratios = ratios if isinstance(ratios,(tuple,list)) else [ratios]
            alphas = [value/100 for value in ratios]
        if not alphas or any(isinstance(a,bool) or not np.isfinite(a) or not 0<=a<1 for a in alphas):
            raise ValueError("alpha must be in [0,1); anomaly_ratio uses percentages in [0,100)")
        self.alphas = tuple(dict.fromkeys(float(a) for a in alphas))

    @staticmethod
    def required_hyper_params():
        return {}

    def _values(self,data,fitting=False):
        array = np.asarray(data)
        columns = list(data.columns) if isinstance(data,pd.DataFrame) else list(range(array.shape[1])) if array.ndim==2 else []
        if fitting:
            self.columns_ = columns
        elif columns!=self.columns_:
            raise ValueError("ASEI channel names and order must match the fitted input")
        if len(set(columns))!=len(columns):
            raise ValueError("ASEI requires unique feature columns")
        if any(str(col).lower() in ("label","timestamp","fault","fault_id") for col in columns):
            raise ValueError("Remove timestamp, label and fault identifiers from ASEI features")
        values = np.asarray(data,dtype=np.float64)
        if values.ndim!=2 or values.shape[1]<1:
            raise ValueError("ASEI input must have shape [T,C], C >= 1")
        return values

    @staticmethod
    def _freeze(module):
        module.eval().requires_grad_(False)

    @staticmethod
    def _finite(values,name):
        if not bool(torch.isfinite(values).all()):
            raise FloatingPointError(f"ASEI {name} contains nonfinite values")

    def _progress(self,label,total):
        return Progress(label,total,self.config.log_interval_seconds)

    def _stage_seed(self,stage):
        return int(np.random.SeedSequence([self.seed,stage]).generate_state(1)[0])

    def _initialize_cuda_runtime(self):
        if self.device.type!="cuda":
            return
        logger.info("ASEI CUDA startup implementation=%s torch=%s device=%s",
                    IMPLEMENTATION_VERSION,torch.__version__,self.device)
        # Availability checks do not initialize PyTorch's caching allocator.
        # Some releases call native resetPeakMemoryStats without lazy init.
        torch.cuda.init()
        if self.device.index is None:
            self.device = torch.device("cuda",torch.cuda.current_device())
        torch.cuda.set_device(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)

    @contextmanager
    def _stage(self,name):
        if self.device.type=="cuda":
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        with self._progress(name,None):
            yield
        if self.device.type=="cuda":
            torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter()-started
        self.metadata_.setdefault("stage_seconds",{})[name] = elapsed
        logger.info("ASEI stage=%s elapsed_s=%.3f",name,elapsed)

    def _resolve_capacity(self,fit_length):
        small = fit_length<8192
        for name,value in (("hidden_dim",16 if small else 32),("predictor_block_size",128 if small else 256),
                           ("batch_size",4 if small else 8),("auxiliary_train_pairs",96 if small else 384)):
            requested = self._requested_config[name]
            setattr(self.config,name,value if requested is None else requested)

    def _predictor_loader(self,values,shuffle):
        cfg = self.config
        options = dict(batch_size=cfg.batch_size,shuffle=shuffle,drop_last=False,num_workers=cfg.num_workers,
                       pin_memory=self.device.type=="cuda",collate_fn=collate_blocks,
                       generator=torch.Generator().manual_seed(self._stage_seed(20)))
        if cfg.num_workers:
            options.update(multiprocessing_context="spawn",persistent_workers=True)
        return DataLoader(BlockWindowDataset(values,self.history,cfg.predictor_block_size),**options)

    def _predictor_loss(self,loader,optimizer=None,epoch=0):
        training = optimizer is not None
        self.predictor.train(training)
        total = torch.zeros((),device=self.device)
        count = 0
        with torch.set_grad_enabled(training), self._progress(
                f"predictor {'train' if training else 'Vsel'} epoch={epoch}",len(loader)) as progress:
            for step,(history,target,mask) in enumerate(loader,1):
                valid_count = int(mask.sum())
                history,target,mask = [value.to(self.device,non_blocking=True) for value in (history,target,mask)]
                if training:
                    optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=self.device.type,dtype=torch.bfloat16,
                                    enabled=training and self.config.predictor_amp=="bf16"):
                    predicted = self.predictor(history,output_steps=target.shape[1])
                    losses = F.huber_loss(predicted.float(),target[...,None].expand_as(predicted),reduction="none")
                    numerator = (losses*mask[...,None,None]).sum()
                    denominator = valid_count*target.shape[2]*2
                    loss = numerator/denominator
                if training:
                    loss.backward()
                    optimizer.step()
                total += numerator.detach()
                count += denominator
                progress.update(step)
        value = float((total/count).cpu())
        if not math.isfinite(value):
            raise FloatingPointError("ASEI predictor Huber loss is not finite")
        return value

    def _fit_predictor(self,fit,validation):
        cfg = self.config
        torch.manual_seed(self._stage_seed(20))
        with self._stage("graph"):
            graph = lag_graph(fit,cfg.graph_lags,cfg.graph_neighbors,cfg.graph_sample_cap)
        self.predictor = Predictor(len(self.columns_),*graph,hidden=cfg.hidden_dim,embedding=cfg.embedding_dim,
                                   kernel=cfg.kernel_size,dilations=cfg.dilations,win_size=cfg.win_size).to(self.device)
        train_loader,valid_loader = self._predictor_loader(fit,True),self._predictor_loader(validation,False)
        optimizer = torch.optim.AdamW(self.predictor.parameters(),lr=cfg.learning_rate,weight_decay=cfg.predictor_weight_decay)
        best_loss,stale,history = float("inf"),0,[]
        with self._stage("predictor"):
            for epoch in range(1,cfg.predictor_epochs+1):
                train_loss = self._predictor_loss(train_loader,optimizer,epoch)
                val_loss = self._predictor_loss(valid_loader,epoch=epoch)
                history.append(dict(epoch=epoch,train_huber=train_loss,validation_huber=val_loss))
                if val_loss<best_loss:
                    best_loss,best,best_epoch,stale = val_loss,cpu_state(self.predictor),epoch,0
                else:
                    stale += 1
                logger.info("ASEI predictor epoch=%d train=%.7g Vsel=%.7g",epoch,train_loss,val_loss)
                if epoch>=min(cfg.predictor_min_epochs,cfg.predictor_epochs) and stale>=cfg.predictor_patience:
                    break
        self.predictor.load_state_dict(best)
        self._freeze(self.predictor)
        self.metadata_["predictor"] = dict(best_epoch=best_epoch,actual_epochs=epoch,history=history,
                                          targets_per_epoch=len(fit)-self.history,
                                          optimizer_steps=epoch*len(train_loader),blocks_per_batch=cfg.batch_size,
                                          target_block_size=cfg.predictor_block_size,graph_edges=int(graph[2].sum()),
                                          graph_targets=min(len(fit)-max(cfg.graph_lags),cfg.graph_sample_cap))

    @torch.inference_mode()
    def _error_tensor(self,values,progress_label=None):
        self._extractor_calls = getattr(self,"_extractor_calls",0)+1
        values = torch.as_tensor(values,dtype=torch.float32,device=self.device)
        single = values.ndim==2
        if single:
            values = values[None]
        outputs = []
        for begin in range(self.history,values.shape[1],self.config.evidence_chunk_size):
            end = min(values.shape[1],begin+self.config.evidence_chunk_size)
            predicted = self.predictor(values[:,begin-self.history:end-1],output_steps=end-begin)
            outputs.append((values[:,begin:end,:,None]-predicted).square())
        result = torch.cat(outputs,1) if outputs else values.new_empty((values.shape[0],0,len(self.columns_),2))
        return result[0] if single else result

    @torch.no_grad()
    def _residual_tensor(self,errors):
        order = self.config.ar_order
        single = errors.ndim==3
        values = errors[None] if single else errors
        if values.shape[1]<=order:
            result = values.new_empty((values.shape[0],0,values.shape[2],2))
        else:
            y = values/self.corrector.error_scale
            history = y.unfold(1,order+1,1)[...,:order].flip(-1)
            predicted = torch.cat([self.corrector(block.contiguous()) for block in
                                   history.split(self.config.evidence_chunk_size,dim=1)],1)
            result = self.corrector.error_scale*(y[:,order:]-predicted)
        return result[0] if single else result

    @torch.no_grad()
    def _evidence_tensor(self,values):
        errors = self._error_tensor(values)
        single = errors.ndim==3
        if single:
            residuals = self._residual_tensor(errors)
            return self.reference.transform_tensor(errors[self.config.ar_order:],residuals)
        residuals = self._residual_tensor(errors)
        batch,length,channels,_ = residuals.shape
        d = errors[:,self.config.ar_order:].reshape(-1,channels,2)
        r = residuals.reshape(-1,channels,2)
        return self.reference.transform_tensor(d,r).reshape(batch,length,channels,2)

    def _feature_tensor(self,values):
        return local_features_tensor(self._evidence_tensor(values),self.config.rolling_window)

    def _features(self,values):
        return self._feature_tensor(values).cpu().numpy()

    def _fit_corrector(self,values):
        errors = self._error_tensor(values)
        self._finite(errors,"corrector errors")
        torch.manual_seed(self._stage_seed(30))
        self.corrector = ErrorCorrector(len(self.columns_),self.config.ar_order,self.config.rho).to(self.device)
        self.metadata_["corrector"] = fit_corrector(
            self.corrector,errors.cpu().numpy(),self.config,self.seed,self.device,self._stage,self._progress)
        self._freeze(self.corrector)

    def _fit_reference(self,values):
        with self._stage("reference"):
            errors = self._error_tensor(values)
            residuals = self._residual_tensor(errors)
            d,r = errors[self.config.ar_order:].cpu().numpy(),residuals.cpu().numpy()
            self.reference = EvidenceReference(self.config.evidence_bound).fit(d,r,self.device)
        self.metadata_["reference"] = dict(points=len(d),first_raw_index=self.history+self.config.ar_order,
                                           maximum_untruncated_evidence=math.log(len(d)+1),dtype="float32")

    def _new_scorer(self):
        cfg = self.config
        return EvidenceScorer(cfg.evidence_bound,cfg.omega_min,cfg.lambda_min,cfg.lambda_max,cfg.lambda_init,
                              cfg.kappa_init,cfg.classification_temperature,cfg.recurrence_backend,
                              cfg.kappa_min,cfg.kappa_max,cfg.scan_chunk_size).to(self.device)

    def _cache_key(self,tasks):
        digest = hashlib.sha256()
        digest.update(json.dumps(dict(version=IMPLEMENTATION_VERSION,data=self.metadata_["data_sha256"],
                                      splits=self.metadata_["splits"],config=asdict(self.config),
                                      columns=self.columns_,tasks=[asdict(t) for t in tasks],
                                      dtype="float32"),sort_keys=True,default=str).encode())
        for module in (self.predictor,self.corrector):
            for name,value in module.state_dict().items():
                digest.update(name.encode())
                _digest_array(digest,value)
        for value in (self.preprocessor.initial,self.preprocessor.mean,self.preprocessor.scale,
                      self.reference.errors,self.reference.residuals,self.reference.center):
            _digest_array(digest,value)
        return digest.hexdigest()

    def _build_bank(self,sampler,tasks,normal_evidence):
        cfg = self.config
        bank = EvidenceBank(self.device,max(0,cfg.evidence_device_cache_mb-self._cache_device_bytes/1024**2),
                            max(0,cfg.evidence_host_cache_mb-self._cache_host_bytes/1024**2),
                            cfg.cache_dir,self._cache_key(tasks))
        groups = {}
        for task in tasks:
            groups.setdefault(task.length,[]).append(task)
        try:
            with self._progress(f"bank_build {sampler.segment_id}",len(tasks)) as progress:
                completed = 0
                for length,group in sorted(groups.items()):
                    batch_limit = cfg.auxiliary_batch_size
                    begin = 0
                    while begin<len(group):
                        current = group[begin:begin+batch_limit]
                        changed = np.stack([sampler.materialize(task)[1] for task in current])
                        try:
                            evidence = self._evidence_tensor(changed)
                        except torch.cuda.OutOfMemoryError:
                            if self.device.type!="cuda" or batch_limit==1:
                                raise
                            batch_limit = max(1,batch_limit//2)
                            torch.cuda.empty_cache()
                            continue
                        for task,perturbed in zip(current,evidence):
                            count = task.length-self.history-cfg.ar_order
                            normal = normal_evidence[task.start:task.start+count].to(self.device)
                            pair = torch.stack((normal,perturbed))
                            metadata = prepare_objective(task,self.raw_history,cfg.rank_pairs_per_event,
                                                        np.random.default_rng(task.pair_indices_seed))
                            bank.append(pair,metadata,task)
                            self.metadata_["counters"]["changed_trajectory_builds"] += 1
                        begin += len(current)
                        completed += len(current)
                        progress.update(completed)
                        del evidence
            bank.seal()
            self._cache_device_bytes += bank.device_bytes
            self._cache_host_bytes += bank.host_bytes
            return bank
        except Exception:
            bank.close()
            raise

    def _bank_loss(self,bank,indices):
        evidence,metadata = bank.batch(indices)
        features = local_features_tensor(evidence.flatten(0,1),self.config.rolling_window)
        features = features.reshape(len(indices),2,*features.shape[1:])
        scores,logits = self.scorer(features)
        return task_objectives(scores,logits,metadata,self.config.ranking_margin,self.config.ranking_temperature)

    @torch.no_grad()
    def _validate_bank(self,bank):
        groups = {}
        for indices in bank.batches(self.config.auxiliary_batch_size):
            detection,ranking = self._bank_loss(bank,indices)
            losses = (detection+ranking).cpu().tolist()
            for index,loss in zip(indices,losses):
                task = bank.entries[index]["task"]
                groups.setdefault((task.family,task.scenario),[]).append(loss)
        group_means = {f"{key[0]}:{key[1]}":float(np.mean(values)) for key,values in sorted(groups.items())}
        return float(np.mean(list(group_means.values()))),group_means

    def _fit_scorer(self,fit,validation,scale):
        cfg = self.config
        banks = []
        self._cache_device_bytes = self._cache_host_bytes = 0
        try:
            with self._stage("bank_build"):
                normal = self._evidence_tensor(fit)
                self.metadata_["counters"]["normal_evidence_passes"] += 1
                intensity = np.zeros(len(fit),dtype=np.float64)
                intensity[self.history+cfg.ar_order:] = (normal[...,1]+
                    torch.as_tensor(self.reference.center[:,1],device=self.device)).mean(-1).cpu().numpy()
                prefix = np.concatenate(([0.],np.cumsum(intensity)))
                sampler = TaskSampler(fit,self.raw_history,scale,cfg,self.protocol_,self.seed,"F",self.metadata_["data_sha256"])
                sampler.hardness_prefix = prefix
                train_tasks = sampler.fixed_tasks(cfg.auxiliary_train_pairs)
                train_bank = self._build_bank(sampler,train_tasks,normal)
                banks.append(train_bank)
                del normal
                normal_valid = self._evidence_tensor(validation)
                self.metadata_["counters"]["normal_evidence_passes"] += 1
                valid_sampler = TaskSampler(validation,self.raw_history,scale,cfg,self.protocol_,self.seed,"Vsel",self.metadata_["data_sha256"])
                valid_tasks = valid_sampler.validation()
                valid_bank = self._build_bank(valid_sampler,valid_tasks,normal_valid)
                banks.append(valid_bank)
                del normal_valid
            self.metadata_["training_tasks"] = [asdict(task) for task in train_tasks]
            self.metadata_["validation_tasks"] = [asdict(task) for task in valid_tasks]
            self.validation_tasks_ = valid_tasks
            self.metadata_["evidence_banks"] = dict(training=train_bank.statistics(),validation=valid_bank.statistics())
            self.scorer = self._new_scorer()
            optimizer = torch.optim.Adam(self.scorer.parameters(),lr=cfg.score_learning_rate,weight_decay=0)
            best_loss,stale,history,steps = float("inf"),0,[],0
            calls_before = self._extractor_calls
            with self._stage("scorer"):
                for epoch in range(1,cfg.score_epochs+1):
                    self.scorer.train()
                    rng = np.random.default_rng(np.random.SeedSequence([self.seed,60,epoch]))
                    batches = train_bank.batches(cfg.auxiliary_batch_size,rng)
                    total = torch.zeros((),device=self.device)
                    with self._progress(f"scorer epoch={epoch}",len(batches)) as progress:
                        for step,indices in enumerate(batches,1):
                            optimizer.zero_grad(set_to_none=True)
                            detection,ranking = self._bank_loss(train_bank,indices)
                            loss = (detection+ranking).mean()
                            loss.backward()
                            nn.utils.clip_grad_norm_(self.scorer.parameters(),cfg.score_gradient_clip)
                            optimizer.step()
                            self.scorer.project_()
                            total += loss.detach()*len(indices)
                            steps += 1
                            progress.update(step)
                    self.scorer.eval()
                    val_loss,groups = self._validate_bank(valid_bank)
                    if not math.isfinite(val_loss):
                        raise FloatingPointError("ASEI scorer validation loss is nonfinite")
                    history.append(dict(epoch=epoch,training_loss=float((total/len(train_tasks)).cpu()),
                                        validation_loss=val_loss,groups=groups))
                    if val_loss<best_loss:
                        best_loss,best,best_epoch,stale = val_loss,cpu_state(self.scorer),epoch,0
                    else:
                        stale += 1
                    logger.info("ASEI scorer epoch=%d Vsel_loss=%.8g lambda=%.6f kappa=%.6f",
                                epoch,val_loss,float(self.scorer.decay.detach().cpu()),float(self.scorer.kappa.detach().cpu()))
                    if epoch>=min(cfg.score_min_epochs,cfg.score_epochs) and stale>=cfg.score_patience:
                        break
            self.metadata_["counters"]["score_epoch_feature_extractor_calls"] = self._extractor_calls-calls_before
            self.scorer.load_state_dict(best)
            self._freeze(self.scorer)
            self.metadata_["selection"] = dict(epoch=best_epoch,loss=best_loss,actual_epochs=epoch,
                                                optimizer_steps=steps,scorer_instances=1)
            self.metadata_["selection_history"] = history
            self.metadata_["decay"] = float(self.scorer.decay.cpu())
            self.metadata_["kappa"] = float(self.scorer.kappa.cpu())
        finally:
            for bank in banks:
                bank.close()

    @torch.no_grad()
    def _sequence_scores(self,values):
        outputs,state = [],None
        cfg = self.config
        for begin in range(self.raw_history,len(values),cfg.evidence_chunk_size):
            end = min(len(values),begin+cfg.evidence_chunk_size)
            features = self._feature_tensor(values[begin-self.raw_history:end])
            scores,state = self.scorer.integrate(features[None],state)
            outputs.append(scores[0])
        return torch.cat(outputs).cpu().numpy() if outputs else np.empty(0,dtype=np.float32)

    def _output_scores(self,scores):
        if self.config.score_transform=="bounded":
            lower = -self.config.evidence_bound-self.config.kappa_max
            upper = (self.config.evidence_bound-self.config.kappa_min)/(1-self.config.lambda_max)
            return ((scores-lower)/(upper-lower)).astype(np.float32)
        return scores

    def _calibrate(self,validation):
        with self._stage("calibration"):
            self.validation_scores_ = self._output_scores(self._sequence_scores(validation))
            self.thresholds_,records = {},[]
            for alpha in self.alphas:
                threshold,count = empirical_threshold(self.validation_scores_,alpha)
                self.thresholds_[alpha] = threshold
                n = len(self.validation_scores_)
                records.append(dict(alpha=alpha,N=n,K=int(np.floor(alpha*n)),threshold=threshold,
                                    false_alarm_count=count,actual_false_alarm_rate=count/n,resolution=1/n))
        self.metadata_["calibration"] = records

    def detect_fit(self,train_data,train_label=None,valid_data=None):
        self._fitted = False
        self._extractor_calls = 0
        started = time.perf_counter()
        self._initialize_cuda_runtime()
        if self.config.predictor_amp=="bf16":
            if self.device.type!="cuda":
                raise ValueError("predictor_amp='bf16' training requires a supported CUDA GPU")
            with torch.cuda.device(self.device):
                if not torch.cuda.is_bf16_supported():
                    raise ValueError("predictor_amp='bf16' training requires a supported CUDA GPU")
        torch.manual_seed(self.seed)
        if self.device.type=="cuda":
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        values = self._values(train_data,True)
        external = None if valid_data is None else self._values(valid_data)
        parts,bounds = chronological_split(values,external)
        self._resolve_capacity(len(parts["F"]))
        cfg = self.config
        for name,minimum in (("F",self.history+256),("Cfit",self.history+cfg.ar_order+64),
                             ("Rref",self.history+cfg.ar_order+128),("Vcal",self.raw_history+128)):
            if len(parts[name])<minimum:
                raise ValueError(f"ASEI {name} split has {len(parts[name])} points; requires at least {minimum}. "
                                 "No padding, test borrowing or entity concatenation.")
        self.preprocessor.fit(parts["F"])
        parts = {name:self.preprocessor.transform(part)[0] for name,part in parts.items()}
        scale = characteristic_scale(parts["Cfit"])
        self.protocol_,feasibility = select_protocol(cfg,self.raw_history,scale,len(parts["F"]),len(parts["Vsel"]))
        digest = hashlib.sha256(repr(self.columns_).encode())
        _digest_array(digest,values)
        if external is not None:
            _digest_array(digest,external)
        self.metadata_ = dict(implementation_version=IMPLEMENTATION_VERSION,data_sha256=digest.hexdigest(),
                              seed=self.seed,splits=bounds,resolved_config=asdict(cfg),execution="single_device",
                              device=str(self.device),receptive_field=self.receptive_field,history=self.history,
                              raw_history=self.raw_history,reference_warmup=self.history+cfg.ar_order,
                              T0=scale,auxiliary_protocol=asdict(self.protocol_),protocol_feasibility=feasibility,
                              preprocessing=dict(all_missing_columns=np.flatnonzero(self.preprocessor.all_missing).tolist(),
                                                 constant_columns=np.flatnonzero(self.preprocessor.constant).tolist()),
                              counters=dict(normal_evidence_passes=0,changed_trajectory_builds=0,score_epoch_feature_extractor_calls=0),
                              numeric_policy=dict(dtype="float32",evidence_amp="off",tf32=False,
                                                  predictor_amp=cfg.predictor_amp,recurrence_backend=cfg.recurrence_backend,
                                                  scan_chunk_size=cfg.scan_chunk_size),
                              versions=dict(python=platform.python_version(),numpy=np.__version__,torch=str(torch.__version__),pandas=pd.__version__))
        logger.info("ASEI implementation=%s device=%s splits=%s capacity=(d=%d Q=%d Bp=%d MA=%d) protocol=%s",
                    IMPLEMENTATION_VERSION,self.device,bounds,cfg.hidden_dim,cfg.predictor_block_size,cfg.batch_size,
                    cfg.auxiliary_train_pairs,asdict(self.protocol_))
        self._fit_predictor(parts["F"],parts["Vsel"])
        self._fit_corrector(parts["Cfit"])
        self._fit_reference(parts["Rref"])
        self._fit_scorer(parts["F"],parts["Vsel"],scale)
        self._calibrate(parts["Vcal"])
        self.metadata_["parameters"] = {name:sum(p.numel() for p in module.parameters()) for name,module in
                                        (("predictor",self.predictor),("corrector",self.corrector),("scorer",self.scorer))}
        self.metadata_["total_fit_seconds"] = time.perf_counter()-started
        self.metadata_["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated(self.device) if self.device.type=="cuda" else 0
        self._fitted = True
        self.reset_state()
        logger.info("ASEI selected=%s calibration=%s counters=%s",self.metadata_["selection"],
                    self.metadata_["calibration"],self.metadata_["counters"])
        if cfg.checkpoint_dir:
            config_hash = hashlib.sha256(json.dumps(asdict(cfg),sort_keys=True).encode()).hexdigest()[:8]
            path = Path(cfg.checkpoint_dir)/f"asei_seed{self.seed}_{digest.hexdigest()[:16]}_{config_hash}.pt"
            self.save(path)
            logger.info("ASEI checkpoint=%s",path)
        return self

    def reset_state(self):
        self._raw_cache = np.empty((0,len(self.columns_)),dtype=np.float32)
        self._imputer_last = self.preprocessor.initial.copy()
        self._stream_state = None
        self._processed_points = 0

    @torch.no_grad()
    def detect_stream(self,data,reset=False):
        if not self._fitted:
            raise RuntimeError("Call ASEI.detect_fit before detection")
        if reset:
            self.reset_state()
        values,last = self.preprocessor.transform(self._values(data),self._imputer_last)
        cfg = self.config
        lower = -cfg.evidence_bound-cfg.kappa_max
        scores = np.full(len(values),lower,dtype=np.float32)
        self.last_ready_mask_ = np.zeros(len(values),dtype=bool)
        for begin in range(0,len(values),cfg.evidence_chunk_size):
            end = min(len(values),begin+cfg.evidence_chunk_size)
            previous = len(self._raw_cache)
            joined = np.concatenate((self._raw_cache,values[begin:end]))
            first_ready = begin+min(end-begin,max(0,self.raw_history-previous))
            if first_ready<end:
                features = self._feature_tensor(joined)
                current,self._stream_state = self.scorer.integrate(features[None],self._stream_state)
                self._finite(current,"online scores")
                scores[first_ready:end] = current[0].cpu().numpy()
                self.last_ready_mask_[first_ready:end] = True
            self._raw_cache = joined[-self.raw_history:].copy() if self.raw_history else joined[:0]
        self._imputer_last = last
        self._processed_points += len(values)
        scores = self._output_scores(scores)
        labels = {100*alpha:((scores>threshold)&self.last_ready_mask_).astype(np.int64)
                  for alpha,threshold in self.thresholds_.items()}
        return labels,scores

    def detect_score(self,test_data):
        _,scores = self.detect_stream(test_data,reset=True)
        return scores,scores

    def detect_label(self,test_data):
        return self.detect_stream(test_data,reset=True)

    def save(self,path):
        if not self._fitted:
            raise RuntimeError("Cannot save an unfitted ASEI detector")
        payload = dict(format_version=3,config=asdict(self.config),requested_config=self._requested_config,
                       columns=self.columns_,seed=self.seed,protocol=asdict(self.protocol_),
                       predictor=cpu_state(self.predictor),corrector=cpu_state(self.corrector),scorer=cpu_state(self.scorer),
                       preprocessing={key:copy.deepcopy(value) for key,value in vars(self.preprocessor).items()},
                       reference={name:torch.from_numpy(getattr(self.reference,name)) for name in ("errors","residuals","center")},
                       validation_scores=torch.from_numpy(self.validation_scores_),thresholds=self.thresholds_,metadata=self.metadata_)
        path = Path(path)
        path.parent.mkdir(parents=True,exist_ok=True)
        torch.save(payload,path)

    @classmethod
    def load(cls,path,device="auto"):
        payload = torch.load(path,map_location="cpu",weights_only=False)
        if payload.get("format_version")!=3:
            raise ValueError("Unsupported ASEI checkpoint format; ASEI 1.4 requires format_version=3 and refitting older models")
        cfg = dict(payload["config"],device=device)
        detector = cls(**cfg)
        detector._requested_config = payload["requested_config"]
        detector.columns_,detector.seed = payload["columns"],payload["seed"]
        detector.protocol_ = AuxiliaryProtocol(**payload["protocol"])
        state,cfg = payload["predictor"],detector.config
        detector.predictor = Predictor(len(detector.columns_),state["sources"],state["delays"],state["edge_mask"],
                                       cfg.hidden_dim,cfg.embedding_dim,cfg.kernel_size,cfg.dilations,cfg.win_size).to(detector.device)
        detector.predictor.load_state_dict(state)
        detector.corrector = ErrorCorrector(len(detector.columns_),cfg.ar_order,cfg.rho).to(detector.device)
        detector.corrector.load_state_dict(payload["corrector"])
        detector.scorer = detector._new_scorer()
        detector.scorer.load_state_dict(payload["scorer"])
        for module in (detector.predictor,detector.corrector,detector.scorer):
            detector._freeze(module)
        for key,value in payload["preprocessing"].items():
            setattr(detector.preprocessor,key,value)
        detector.reference = EvidenceReference(cfg.evidence_bound)
        for key,value in payload["reference"].items():
            setattr(detector.reference,key,value.numpy())
        detector.reference.to(detector.device)
        detector.validation_scores_ = payload["validation_scores"].numpy()
        detector.thresholds_,detector.metadata_ = payload["thresholds"],payload["metadata"]
        detector._fitted = True
        detector.reset_state()
        return detector
