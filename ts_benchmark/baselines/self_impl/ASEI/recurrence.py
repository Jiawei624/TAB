"""Public-PyTorch parallel scans with the sequential recurrence's first derivative."""

import torch


def max_affine_scan(evidence, decay, initial, kappa):
    """Inclusive prefix composition; each doubling round reads the previous round."""
    aa = torch.ones_like(evidence) * decay
    bb = evidence - kappa
    cc = bb.relu()
    offset = 1
    while offset < evidence.shape[1]:
        a0, b0, c0 = aa, bb, cc
        aa = torch.cat((a0[:, :offset], a0[:, offset:] * a0[:, :-offset]), 1)
        bb = torch.cat((b0[:, :offset], a0[:, offset:] * b0[:, :-offset] + b0[:, offset:]), 1)
        cc = torch.cat((c0[:, :offset], torch.maximum(
            a0[:, offset:] * c0[:, :-offset] + b0[:, offset:], c0[:, offset:])), 1)
        offset *= 2
    states = torch.maximum(aa * initial[:, None] + bb, cc)
    previous = torch.cat((initial[:, None], states[:, :-1]), 1)
    activations = decay * previous + evidence - kappa
    return activations, previous, states[:, -1]


def affine_scan(coefficient, intercept):
    """Inclusive affine prefix, zero initial state."""
    aa, bb = coefficient, intercept
    offset = 1
    while offset < bb.shape[1]:
        a0, b0 = aa, bb
        aa = torch.cat((a0[:, :offset], a0[:, offset:] * a0[:, :-offset]), 1)
        bb = torch.cat((b0[:, :offset], a0[:, offset:] * b0[:, :-offset] + b0[:, offset:]), 1)
        offset *= 2
    return bb


class _Scan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, evidence, decay, initial, kappa):
        a, previous, final = max_affine_scan(evidence, decay, initial, kappa)
        scores = a.amax(-1)
        ctx.save_for_backward(a, previous, decay)
        return scores, final

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_scores, grad_final):
        a, previous, decay = ctx.saved_tensors
        if grad_scores is None:
            grad_scores = a.new_zeros(a.shape[:2])
        if grad_final is None:
            grad_final = a.new_zeros((a.shape[0], a.shape[2]))
        ties = (a == a.amax(-1, keepdim=True)).to(a.dtype)
        direct = grad_scores[..., None] * ties / ties.sum(-1, keepdim=True)
        active = (a > 0).to(a.dtype)
        direct[:, -1] += active[:, -1] * grad_final
        adjoint = affine_scan((decay * active).flip(1), direct.flip(1)).flip(1)
        return adjoint, (adjoint * previous).sum(), decay * adjoint[:, 0], -adjoint.sum()


def reference(evidence, decay, initial, kappa):
    state, activations = initial, []
    for current in evidence.unbind(1):
        a = decay * state + current - kappa
        state = a.relu()
        activations.append(a)
    return torch.stack(activations, 1).amax(-1), state


def integrate(evidence, decay, initial, kappa, backend="scan", chunk_size=256):
    """Carry the actual state AND its gradient across all computation chunks."""
    if backend not in ("scan", "reference"):
        raise ValueError("recurrence_backend must be 'scan' or 'reference'")
    if evidence.shape[1] == 0:
        return evidence.new_empty(evidence.shape[:2]), initial
    kappa = torch.as_tensor(kappa, dtype=evidence.dtype, device=evidence.device)
    decay = torch.as_tensor(decay, dtype=evidence.dtype, device=evidence.device)
    if backend == "reference":
        return reference(evidence, decay, initial, kappa)
    pieces, state = [], initial
    for block in evidence.split(chunk_size, dim=1):
        scores, state = _Scan.apply(block, decay, state, kappa)
        pieces.append(scores)
    return torch.cat(pieces, 1), state
