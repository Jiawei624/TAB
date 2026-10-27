"""Scripted exact recurrence and its first-order adjoint, with reference operators."""

from typing import Tuple

import torch
from torch.nn import functional as F


@torch.jit.script
def _forward(evidence: torch.Tensor, decay: torch.Tensor, initial: torch.Tensor,
             kappa: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    activations = torch.empty_like(evidence)
    state = initial
    for time in range(evidence.size(1)):
        activation = decay * state + evidence[:, time] - kappa
        activations[:, time] = activation
        state = torch.relu(activation)
    return torch.amax(activations, dim=-1), state, activations


@torch.jit.script
def _backward(activations: torch.Tensor, scores: torch.Tensor, decay: torch.Tensor,
              initial: torch.Tensor, grad_scores: torch.Tensor,
              grad_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tied = activations == scores.unsqueeze(-1)
    direct = grad_scores.unsqueeze(-1) * tied.to(activations.dtype) / tied.sum(-1, keepdim=True)
    grad_evidence = torch.empty_like(activations)
    grad_decay = torch.zeros_like(decay)
    adjoint = grad_state
    for time in range(activations.size(1) - 1, -1, -1):
        grad_activation = direct[:, time] + adjoint * (activations[:, time] > 0).to(activations.dtype)
        previous = initial if time == 0 else torch.relu(activations[:, time - 1])
        grad_decay = grad_decay + (grad_activation * previous).sum()
        grad_evidence[:, time] = grad_activation
        adjoint = decay * grad_activation
    return grad_evidence, grad_decay, adjoint


class _Recurrence(torch.autograd.Function):
    @staticmethod
    def forward(ctx, evidence, decay, initial, kappa):
        scores, state, activations = _forward(evidence, decay, initial, kappa)
        ctx.save_for_backward(activations, scores, decay, initial)
        return scores, state

    @staticmethod
    def backward(ctx, grad_scores, grad_state):
        activations, scores, decay, initial = ctx.saved_tensors
        if grad_scores is None:
            grad_scores = torch.zeros_like(scores)
        if grad_state is None:
            grad_state = torch.zeros_like(initial)
        grad_evidence, grad_decay, grad_initial = _backward(
            activations, scores, decay, initial, grad_scores, grad_state)
        return grad_evidence, grad_decay, grad_initial, None


def integrate(evidence, decay, initial, kappa, backend="scripted"):
    if evidence.shape[1] == 0:
        return evidence.new_empty(evidence.shape[:2]), initial
    if backend == "scripted":
        if torch.is_grad_enabled() and any(tensor.requires_grad for tensor in (evidence, decay, initial)):
            return _Recurrence.apply(evidence, decay, initial, float(kappa))
        scores, state, _ = _forward(evidence, decay, initial, float(kappa))
        return scores, state
    if backend != "reference":
        raise ValueError("Unknown recurrence backend")
    state, scores = initial, []
    for current in evidence.unbind(1):
        activation = decay * state + current - kappa
        scores.append(torch.amax(activation, dim=-1))
        state = F.relu(activation)
    return torch.stack(scores, 1), state
