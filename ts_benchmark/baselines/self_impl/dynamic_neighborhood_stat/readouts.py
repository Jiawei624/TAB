"""Static dependency plans for exact finite-window causal TCN readouts.

Indices address each layer's existing states. Negative time positions are that
layer's zero padding, never hidden states obtained by evaluating its biases.
All buffers are derived and nonpersistent, so original checkpoints stay valid.
"""

import torch
from torch import nn
from torch.nn import functional as F


def _ancestors(positions, dilation):
    return sorted({q - tap * dilation for q in positions for tap in range(3) if q - tap * dilation >= 0})


class ConvolutionReadout(nn.Module):
    def __init__(self, inputs, outputs, dilation):
        super().__init__()
        lookup = {position: index for index, position in enumerate(inputs)}
        # Conv1d is cross-correlation: the last kernel tap reads current time.
        positions = [[q - 2 * dilation, q - dilation, q] for q in outputs]
        valid = torch.tensor([[q >= 0 for q in row] for row in positions], dtype=torch.bool)
        indices = torch.tensor([[lookup[q] if q >= 0 else 0 for q in row] for row in positions], dtype=torch.long)
        self.register_buffer("indices", indices.flatten(), persistent=False)
        self.register_buffer("padding_mask", ~valid[None, :, :, None], persistent=False)
        self.output_count = len(outputs)

    def forward(self, x, convolution):
        batch, _, width = x.shape
        taps = x.index_select(1, self.indices).reshape(batch, self.output_count, 3, width)
        taps = taps.masked_fill(self.padding_mask, 0)
        weight = convolution.weight.permute(0, 2, 1).reshape(convolution.out_channels, 3 * width)
        return F.linear(taps.reshape(batch, self.output_count, 3 * width), weight, convolution.bias)


class BlockReadout(nn.Module):
    def __init__(self, inputs, intermediate, outputs, dilation):
        super().__init__()
        self.first = ConvolutionReadout(inputs, intermediate, dilation)
        self.second = ConvolutionReadout(intermediate, outputs, dilation)
        lookup = {position: index for index, position in enumerate(inputs)}
        self.register_buffer("residual_indices", torch.tensor([lookup[q] for q in outputs]), persistent=False)

    def forward(self, x, block):
        residual = x.index_select(1, self.residual_indices)
        for operation, convolution, norm in zip((self.first, self.second), block.convs, block.norms):
            x = operation(x, convolution)
            x = block.dropout(F.gelu(norm(x)))
        return F.gelu(x + residual)


class EncoderReadout(nn.Module):
    def __init__(self, length, positions, blocks):
        super().__init__()
        if not positions or min(positions) < 0 or max(positions) >= length:
            raise ValueError("Readouts must be valid positions within the encoder input")
        requested = sorted(set(positions))
        needed, plans = requested, []
        for block in reversed(blocks):
            dilation = block.convs[0].dilation[0]
            intermediate = _ancestors(needed, dilation)
            previous = _ancestors(intermediate, dilation)
            plans.append(BlockReadout(previous, intermediate, needed, dilation))
            needed = previous
        self.layers = nn.ModuleList(reversed(plans))
        self.register_buffer("input_indices", torch.tensor(needed), persistent=False)
        lookup = {position: index for index, position in enumerate(requested)}
        self.register_buffer("output_indices", torch.tensor([lookup[q] for q in positions]), persistent=False)
        self.dense_positions = length * 2 * len(blocks)
        self.computed_positions = sum(layer.first.output_count + layer.second.output_count for layer in self.layers)

    def forward(self, x, encoder):
        x = encoder.input_projection(x.index_select(1, self.input_indices))
        for operation, block in zip(self.layers, encoder.blocks):
            x = operation(x, block)
        return x.index_select(1, self.output_indices)
