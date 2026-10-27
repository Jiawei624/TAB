"""Two fixed-budget FP32 corrector fits, with feasible linear parameters."""
import numpy as np
import torch
from torch.nn import functional as F
from .calibration import error_history


def index_batches(count, batch_size, rng):
    while True:
        order = rng.permutation(count)
        for begin in range(0, count, batch_size):
            yield order[begin:begin+batch_size]


def fit_corrector(corrector, errors, config, seed, device, stage, progress_factory):
    errors = np.asarray(errors, dtype=np.float32)
    iqr = np.percentile(errors, 75, axis=0) - np.percentile(errors, 25, axis=0)
    scale = np.maximum(np.maximum(errors.mean(0, dtype=np.float64), iqr), config.error_scale_floor).astype(np.float32)
    y = errors / scale
    history = error_history(y, config.ar_order)
    target = y[config.ar_order:]
    logged = np.log1p(history).astype(np.float64)
    log_scale = logged.std(0)
    log_scale[log_scale == 0] = 1
    with torch.no_grad():
        for name, value in dict(error_scale=scale, log_mean=logged.mean(0), log_scale=log_scale,
                                eta=config.nonlinear_iqr_fraction * (np.percentile(y,75,axis=0)-np.percentile(y,25,axis=0)),
                                bias=np.median(y,axis=0)).items():
            getattr(corrector, name).copy_(torch.as_tensor(value, dtype=torch.float32, device=device))
    history = torch.as_tensor(history, device=device)
    target = torch.as_tensor(target, device=device)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 30]))
    batches = index_batches(len(target), min(config.corrector_batch_size,len(target)), rng)
    result = dict(samples=len(target), dtype="float32", huber_delta=1, weight_decay=0,
                  scale=scale.tolist(), bias_initialization="median_all_normalized_errors")
    for nonlinear, name, steps, lr in (
        (False, "corrector_linear", config.corrector_linear_steps, config.corrector_linear_lr),
        (True, "corrector_nonlinear", config.corrector_nonlinear_steps, config.corrector_nonlinear_lr)):
        corrector.bias.requires_grad_(not nonlinear)
        corrector.beta.requires_grad_(not nonlinear)
        corrector.mlp.requires_grad_(nonlinear)
        parameters = list(corrector.mlp.parameters()) if nonlinear else [corrector.bias, corrector.beta]
        optimizer = torch.optim.Adam(parameters, lr=lr, weight_decay=0)
        losses, processed = [], 0
        with stage(name), progress_factory(name, steps) as progress:
            for step in range(1, steps+1):
                indices = torch.as_tensor(next(batches), device=device)
                optimizer.zero_grad(set_to_none=True)
                loss = F.huber_loss(corrector(history[indices], nonlinear=nonlinear), target[indices], delta=1)
                loss.backward()
                optimizer.step()
                if not nonlinear:
                    corrector.project_()
                losses.append(loss.detach())
                processed += len(indices)
                progress.update(step)
            first, last = torch.stack((losses[0], losses[-1])).cpu().tolist()
        result[name] = dict(optimizer="Adam", updates=steps, processed_samples=processed,
                            first_loss=first, last_loss=last, learning_rate=lr, budget_exhausted=True)
        del optimizer
    result["maximum_beta_l1"] = float(corrector.beta.detach().abs().sum(-1).max().cpu())
    return result