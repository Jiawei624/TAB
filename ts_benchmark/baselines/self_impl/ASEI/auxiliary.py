"""Deterministic finite perturbation banks and per-task balanced objectives."""
from dataclasses import dataclass, replace
import numpy as np
import torch
from torch.nn import functional as F


def characteristic_scale(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[:, np.std(values, axis=0) > 0]
    if values.shape[1] == 0:
        return 8
    for lag in range(1, min(128, len(values)-2)+1):
        left, right = values[lag:], values[:-lag]
        left, right = left-left.mean(0), right-right.mean(0)
        denominator = np.sqrt((left*left).sum(0)*(right*right).sum(0))
        acf = np.divide((left*right).sum(0), denominator,
                        out=np.zeros(values.shape[1]), where=denominator>0)
        if np.median(np.abs(acf)) < np.exp(-1):
            nearest = min([2**power for power in range(8)], key=lambda x:(abs(x-lag), x))
            return int(np.clip(nearest, 4, 32))
    return 32


@dataclass
class Task:
    start: int
    length: int
    events: list
    family: int
    subtype: int
    seed: int
    scenario: str = "single"
    hard: bool = False
    base_attempts: int = 0
    entity_id: str = ""
    segment_id: str = ""
    perturbations: object = None
    pair_indices_seed: int = 0

    def __post_init__(self):
        self.events = [tuple(event) for event in self.events]


class IneffectivePerturbation(ValueError):
    pass


class TaskSampler:
    def __init__(self, values, raw_history, scale, config, protocol, seed=None, segment_id="F", entity_id=""):
        self.values, self.raw_history, self.config = values, raw_history, config
        self.protocol, self.scale = protocol, scale
        self.lengths, self.event_start = protocol.training_lengths, protocol.event_start
        self.seed = config.seed if seed is None else seed
        self.segment_id, self.entity_id = segment_id, entity_id
        self.hardness_prefix = None
        self._eligible = {}

    def _starts(self, task):
        if task.family != 2:
            return None
        key = (task.length, tuple(task.events))
        if key not in self._eligible:
            n = len(self.values)-task.length+1
            changed = np.zeros(len(self.values), dtype=bool)
            for delay in (1,2,4,8,16):
                changed[delay:] |= np.any(self.values[delay:] != self.values[:-delay], axis=1)
            possible = np.ones(n, dtype=bool)
            for begin, end in task.events:
                possible &= changed[begin:begin+n] & changed[end-1:end-1+n]
            self._eligible[key] = np.flatnonzero(possible)
        return self._eligible[key]

    def _position(self, task, rng):
        eligible = self._starts(task)
        if eligible is not None and not len(eligible):
            raise IneffectivePerturbation("No effective auxiliary base trajectory for time shift; event endpoints are constant")
        upper = len(self.values)-task.length+1
        if upper < 1:
            raise ValueError("Auxiliary task exceeds its continuous segment")
        count = min(self.config.hard_pool_size,upper if eligible is None else len(eligible)) if task.hard else 1
        starts = rng.choice(upper if eligible is None else eligible,size=count,replace=False)
        if task.hard:
            if self.hardness_prefix is None:
                raise RuntimeError("Hard tasks require frozen normal evidence")
            p = self.hardness_prefix
            difficulty = (p[starts+task.length]-p[starts+self.raw_history])/(task.length-self.raw_history)
            starts = starts[np.lexsort((starts,-difficulty))]
            return int(starts[int(rng.integers(max(1,int(np.ceil(count/4)))))])
        return int(starts[0])

    def fixed_tasks(self, count, validation=False):
        if count % 6:
            raise ValueError("Auxiliary pair count must be a multiple of 6 (family x layout)")
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, 50 if validation else 40]))
        tasks = []
        for family in range(3):
            for layout in ("single","recurrent"):
                for index in range(count//6):
                    length = int(self.lengths[index % len(self.lengths)])
                    events = [(self.event_start,self.event_start+length)]
                    if layout == "recurrent":
                        second = events[0][1]+self.protocol.recurrence_gap
                        events.append((second,second+length))
                    task = Task(0, events[-1][1]+self.protocol.recovery_suffix, events, family, index%2,
                                int(rng.integers(2**31)), layout,
                                hard=(not validation and index%2==1),
                                entity_id=self.entity_id, segment_id=self.segment_id,
                                pair_indices_seed=int(rng.integers(2**31)))
                    task.start = self._position(task, rng)
                    tasks.append(self.resolve(task)[0])
        return [tasks[int(i)] for i in rng.permutation(len(tasks))]

    def validation(self):
        return self.fixed_tasks(self.config.auxiliary_validation_pairs, validation=True)

    def resolve(self, task):
        if task.perturbations is not None:
            normal, changed = self._materialize_fixed(task)
            return task, normal, changed
        rng = np.random.default_rng(np.random.SeedSequence([task.seed,104729]))
        for attempt in range(self.config.perturbation_attempts):
            current = task if attempt == 0 else replace(
                task, start=self._position(task,rng), seed=int(rng.integers(2**31)), base_attempts=attempt)
            try:
                normal, changed = self._materialize_fixed(current)
                return current, normal, changed
            except IneffectivePerturbation:
                pass
        raise IneffectivePerturbation(f"Cannot construct effective task after {self.config.perturbation_attempts} attempts: "
                                     f"family={task.family} layout={task.scenario} segment={task.segment_id}")

    def _materialize_fixed(self, task):
        normal = self.values[task.start:task.start+task.length].copy()
        changed = normal.copy()
        rng = np.random.default_rng(task.seed)
        details, channels = [], None
        for event_index, (begin,end) in enumerate(task.events):
            if task.perturbations is not None:
                detail = task.perturbations[event_index]
            else:
                if channels is None:
                    count = int(rng.integers(1,max(1,int(np.ceil(0.1*normal.shape[1])))+1))
                    channels = rng.choice(normal.shape[1], size=count, replace=False).tolist()
                onsets = rng.integers(0,min(16,end-begin-1)+1,size=len(channels))
                onsets[0] = 0
                value = (float(np.exp(rng.uniform(np.log(.2),np.log(3))))*int(rng.choice([-1,1])) if task.family==0
                         else float(np.exp(rng.uniform(np.log(.2),np.log(2)))) if task.family==1
                         else int(rng.choice([1,2,4,8,16])))
                detail = dict(affected_channels=channels, onsets=onsets.tolist(), value=value,
                              noise_seed=int(rng.integers(2**31)))
            details.append(detail)
            noise_rng = np.random.default_rng(detail["noise_seed"])
            shared_noise = noise_rng.normal(size=end-begin)
            for channel,onset in zip(detail["affected_channels"],detail["onsets"]):
                first, value = begin+onset, detail["value"]
                n = end-first
                if task.family==0:
                    change = value if task.subtype==0 else value*np.arange(1,n+1)/n
                    changed[first:end,channel] += change
                elif task.family==1:
                    noise = noise_rng.normal(size=n) if task.subtype==0 else shared_noise[onset:]
                    changed[first:end,channel] += value*noise
                else:
                    changed[first:end,channel] = normal[first-int(value):end-int(value),channel]
        if any(not np.any(changed[begin]!=normal[begin]) or not np.any(changed[end-1]!=normal[end-1])
               for begin,end in task.events):
            raise IneffectivePerturbation("Perturbation must change both endpoints of each event")
        task.perturbations = details
        return normal, changed

    def materialize(self, task):
        _, normal, changed = self.resolve(task)
        return normal, changed


def prepare_objective(task, raw_history, rank_pairs, rng):
    length = task.length-raw_history
    positive = np.zeros(length,dtype=np.float32)
    events = [(begin-raw_history,end-raw_history) for begin,end in task.events]
    for begin,end in events:
        positive[begin:end] = 1/((end-begin)*len(events))
    negative = np.zeros((2,length),dtype=np.float32)
    negative[0,events[0][0]:] = 1
    negative[1,events[0][0]:] = 2
    for begin,end in events:
        negative[1,begin:end] = 0
    probability = negative.reshape(-1).astype(np.float64)
    probability /= probability.sum()
    pos_indices, neg_indices = [], []
    for begin,end in events:
        pos_indices.extend(rng.integers(begin,end,rank_pairs))
        neg_indices.extend(rng.choice(len(probability),rank_pairs,p=probability))
    return dict(positive=positive,negative=negative,event_count=len(events),
                rank_positive=np.asarray(pos_indices),rank_negative=np.asarray(neg_indices))


def task_objectives(scores, logits, metadata, margin=1, rank_temperature=1):
    detection, ranking = [], []
    for index,item in enumerate(metadata):
        positive, negative = item["positive"], item["negative"]
        current, z = scores[index], logits[index]
        detection.append(.5*(F.softplus(-z[1])*positive).sum()
                         + .5*(F.softplus(z)*negative).sum()/negative.sum())
        ranking.append(F.softplus((margin-current[1,item["rank_positive"]]
                                   +current.reshape(-1)[item["rank_negative"]])/rank_temperature).mean())
    return torch.stack(detection), torch.stack(ranking)


def objective(scores, logits, metadata, margin=1, rank_temperature=1):
    detection, ranking = task_objectives(scores, logits, metadata, margin, rank_temperature)
    return detection.mean(), ranking.mean()


def objective_to_device(metadata, device):
    return [{key:torch.as_tensor(value,device=device) if isinstance(value,np.ndarray) else value
             for key,value in item.items()} for item in metadata]
