"""Float64 full-batch Huber AR optimization with projected backtracking."""

import logging
import math

import torch
from torch import nn

from .model import project_l1_, unwrap


logger = logging.getLogger(__name__)


class LinearARObjective(nn.Module):
    """Full-batch sufficient sums on one device."""
    def __init__(self, corrector):
        super().__init__()
        self.bias, self.beta = corrector.bias, corrector.beta

    def forward(self, history, target, gradients=True):
        residual = self.bias + (history * self.beta).sum(-1) - target
        absolute = residual.abs()
        loss = torch.where(absolute <= 1, 0.5 * residual.square(), absolute - 0.5).sum()
        if not gradients:
            return loss[None]
        derivative = residual.clamp(-1, 1)
        return (loss[None], derivative.sum(0)[None],
                (derivative[..., None] * history).sum(0)[None])


@torch.no_grad()
def fit_projected_ar(objective, history, target, radius=0.95, max_iter=1000,
                     initial_step=0.01, shrink=0.5, tolerance=1e-5, progress=None):
    """Optimize the global mean Huber_1 loss; never make mini-batch updates."""
    module = unwrap(objective)
    if any(value.dtype != torch.float64 for value in (history, target, module.bias, module.beta)):
        raise ValueError("Projected AR optimization requires float64 inputs and linear parameters")
    if max_iter < 1 or initial_step <= 0 or not 0 < shrink < 1 or tolerance <= 0:
        raise ValueError("Invalid projected AR optimization controls")
    normalizer = target.numel()

    def statistics():
        loss, grad_bias, grad_beta = objective(history, target, gradients=True)
        return (loss.sum() / normalizer, grad_bias.sum(0) / normalizer,
                grad_beta.sum(0) / normalizer)

    def proposal(bias, beta, grad_bias, grad_beta, step):
        candidate_bias = bias - step * grad_bias
        candidate_beta = beta - step * grad_beta
        project_l1_(candidate_beta, radius)
        return candidate_bias, candidate_beta

    trace, total_reductions, converged = [], 0, False
    step = initial_step
    initial_loss = None
    for iteration in range(1, max_iter + 1):
        if progress is not None:
            progress.phase(f"iteration={iteration} full_gradient")
        loss, grad_bias, grad_beta = statistics()
        finite = torch.stack((torch.isfinite(loss).all(), torch.isfinite(grad_bias).all(), torch.isfinite(grad_beta).all())).all()
        if not finite.item():
            raise FloatingPointError("ASEI full-batch AR loss/gradient is not finite")
        loss_value = float(loss.item())
        if initial_loss is None:
            initial_loss = loss_value
        bias, beta = module.bias.clone(), module.beta.clone()
        if iteration > 1:
            step *= 2
        reductions = 0
        while True:
            if progress is not None:
                progress.phase(f"iteration={iteration} backtracking={reductions}")
            candidate_bias, candidate_beta = proposal(bias, beta, grad_bias, grad_beta, step)
            diff_bias, diff_beta = candidate_bias - bias, candidate_beta - beta
            rhs = (loss + (grad_bias * diff_bias).sum() + (grad_beta * diff_beta).sum()
                   + (diff_bias.square().sum() + diff_beta.square().sum()) / (2 * step))
            module.bias.copy_(candidate_bias)
            module.beta.copy_(candidate_beta)
            candidate_loss = objective(history, target, gradients=False).sum() / normalizer
            candidate_value, upper_bound = torch.stack((candidate_loss, rhs)).cpu().tolist()
            slack = 32 * torch.finfo(torch.float64).eps * max(1.0, abs(loss_value), abs(upper_bound))
            if (math.isfinite(candidate_value)
                    and candidate_value <= upper_bound + slack):
                break
            step *= shrink
            reductions += 1
            if reductions >= 128 or step == 0:
                module.bias.copy_(bias)
                module.beta.copy_(beta)
                raise FloatingPointError("ASEI projected AR backtracking failed to find a finite quadratic-upper-bound step")
        mapping = float(torch.maximum(diff_bias.abs().max(), diff_beta.abs().max()).item()) / step
        total_reductions += reductions
        trace.append(dict(iteration=iteration, loss=loss_value, candidate_loss=candidate_value,
                          quadratic_upper_bound=upper_bound, numerical_slack=slack,
                          step=step, backtracking_reductions=reductions, gradient_mapping_inf=mapping))
        if progress is not None:
            progress.update(iteration, f"loss={candidate_value:.8g} mapping_inf={mapping:.5g} step={step:.5g}")
        if mapping <= tolerance:
            # This mapping was measured at (bias, beta); return that verified point.
            module.bias.copy_(bias)
            module.beta.copy_(beta)
            converged = True
            break
        if iteration == 1 or iteration % 100 == 0:
            logger.info("ASEI linear AR iteration=%d loss=%.8g mapping_inf=%.5g step=%.5g",
                        iteration, candidate_value, mapping, step)
    # Report the mapping at the parameters actually returned, including capped runs.
    final_loss, grad_bias, grad_beta = statistics()
    candidate_bias, candidate_beta = proposal(module.bias, module.beta, grad_bias, grad_beta, step)
    final_mapping = float(torch.maximum((module.bias - candidate_bias).abs().max(),
                                       (module.beta - candidate_beta).abs().max()).item()) / step
    converged = final_mapping <= tolerance
    return dict(optimizer="full_batch_projected_gradient", dtype="float64", huber_delta=1,
                weight_decay=0, radius=radius, max_iter=max_iter, initial_step=initial_step,
                backtracking_factor=shrink, tolerance=tolerance, iterations=len(trace),
                converged=converged, stopping_reason="gradient_mapping" if converged else "max_iter",
                initial_loss=initial_loss, final_loss=float(final_loss.item()),
                gradient_mapping_inf=final_mapping, last_step=step,
                backtracking_reductions=total_reductions, trace=trace)
