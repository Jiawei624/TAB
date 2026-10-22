"""Independent neighborhood and strictly causal univariate predictors."""

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .features import Observations
from .readouts import EncoderReadout


def mlp(input_dim, hidden, output_dim):
    return nn.Sequential(nn.Linear(input_dim, hidden), nn.GELU(), nn.Linear(hidden, output_dim))


class CausalBlock(nn.Module):
    def __init__(self, width, dilation, dropout):
        super().__init__()
        self.left_padding = 2 * dilation
        self.convs = nn.ModuleList([nn.Conv1d(width, width, 3, dilation=dilation) for _ in range(2)])
        self.norms = nn.ModuleList([nn.LayerNorm(width) for _ in range(2)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        for conv, norm in zip(self.convs, self.norms):
            x = conv(F.pad(x, (self.left_padding, 0)))
            # LayerNorm sees hidden features at ONE position, never time/batch/variables.
            x = norm(x.transpose(1, 2)).transpose(1, 2)
            x = self.dropout(F.gelu(x))
        return F.gelu(x + residual)


class CausalEncoder(nn.Module):
    def __init__(self, input_dim, width, required_receptive_field, dropout):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, width)
        blocks, receptive_field, dilation = [], 1, 1
        while receptive_field < required_receptive_field:
            blocks.append(CausalBlock(width, dilation, dropout))
            receptive_field += 4 * dilation  # two convolutions, kernel=3
            dilation *= 2
        self.blocks = nn.Sequential(*blocks)
        self.receptive_field = receptive_field
        self.readout_plans = nn.ModuleDict()

    @staticmethod
    def _readout_key(length, positions):
        return "{}_{}".format(length, "_".join(map(str, positions)))

    def add_readouts(self, length, positions):
        self.readout_plans[self._readout_key(length, positions)] = EncoderReadout(length, positions, self.blocks)

    def forward(self, x, readouts=None):
        if readouts is not None:
            return self.readout_plans[self._readout_key(x.shape[1], readouts)](x, self)
        x = self.input_projection(x).transpose(1, 2)
        return self.blocks(x).transpose(1, 2)


class DualPathNetwork(nn.Module):
    def __init__(self, channels, config, references):
        super().__init__()
        self.channels, self.config = channels, config
        self.observations = Observations(config.win_size, config.scales, config.spans)
        for name, array in references.items():
            self.register_buffer(name, torch.as_tensor(array).clone())
        n, emb, hidden = channels - 1, config.embedding_dim, config.mlp_hidden
        self.top_r = min(2, n - 1) if n >= 2 else 0
        self.rho_a = (1 + self.top_r / n) / 2 if n >= 2 else 1.0
        off_diagonal = 1 - torch.eye(channels)
        self.register_buffer("context_mask", off_diagonal / n)

        # Neighborhood modules share parameters across variables/scales, not with the self path.
        self.variable_n = nn.Embedding(channels, emb)
        self.scale_n = nn.Embedding(len(config.scales), emb)
        self.encoder_n = CausalEncoder(2 + 2 * emb, config.hidden_n, 32, config.dropout)
        self.context = mlp(len(config.scales) * config.hidden_n, hidden, config.context_dim)
        self.scale_gate = mlp(3 * emb + config.hidden_n + config.context_dim, hidden, 1)
        self.value = mlp(config.hidden_n + 3 * emb, hidden, config.message_dim)
        self.relation = mlp(2 * emb + config.message_dim + config.context_dim, hidden, 1)
        self.decoder_n = mlp(config.message_dim + 2 * emb, hidden, 2)

        self.variable_s = nn.Embedding(channels, emb)
        self.span_s = nn.Embedding(len(config.spans), emb)
        self.encoder_s = CausalEncoder(1 + emb, config.hidden_s, config.win_size - 1, config.dropout)
        self.stat_head = mlp(config.hidden_s + 2 * emb, hidden, 3)
        self.point_head = mlp(config.hidden_s + emb, hidden, 1)
        for scale in config.scales:
            length = config.win_size // scale
            self.encoder_n.add_readouts(length, (length - 1,))
        self.self_readouts = tuple(config.win_size - h - 1 for h in config.spans) + (config.win_size - 2,)
        self.encoder_s.add_readouts(config.win_size - 1, self.self_readouts)

    @staticmethod
    def _project_parts(layer, *inputs):
        weights = layer.weight.split([value.shape[-1] for value in inputs], dim=1)
        return tuple(F.linear(value, weight) for value, weight in zip(inputs, weights))

    def _route_projections(self, encoded, context):
        """Project each source/target once; nonlinear mixing still sees every pair."""
        variable, scale = self.variable_n.weight, self.scale_n.weight
        gi, gj, gm, gh, gc = self._project_parts(self.scale_gate[0], variable, variable, scale, encoded, context)
        scale_source = gh + gj[None, :, None] + gm[None, None] + self.scale_gate[0].bias
        scale_target = gc + gi[None]
        vh, vi, vj, vm = self._project_parts(self.value[0], encoded, variable, variable, scale)
        value_source = vh + vj[None, :, None] + vm[None, None] + self.value[0].bias
        emb, message = self.config.embedding_dim, self.config.message_dim
        wi, wj, _, wc = self.relation[0].weight.split((emb, emb, message, self.config.context_dim), dim=1)
        relation_target = F.linear(context, wc) + F.linear(variable, wi)[None] + self.relation[0].bias
        relation_source = F.linear(variable, wj)
        _, wi, wm = self.decoder_n[0].weight.split((message, emb, emb), dim=1)
        return (scale_source, scale_target, value_source, vi, relation_target, relation_source,
                F.linear(variable, wi), F.linear(scale, wm))

    def _route_projected(self, scale_source, scale_target, value_source, value_target,
                         relation_target, relation_source, decoder_target, decoder_scale, target_ids):
        scale_hidden = scale_source[:, None] + scale_target[:, target_ids, None, None]
        weights_scale = torch.softmax(self.scale_gate[2](self.scale_gate[1](scale_hidden)).squeeze(-1)
                                      / self.config.temperature_scale, dim=-1)
        value_hidden = value_source[:, None] + value_target[target_ids][None, :, None, None]
        messages = (weights_scale[..., None] * self.value[2](self.value[1](value_hidden))).sum(dim=-2)
        emb, message = self.config.embedding_dim, self.config.message_dim
        wm = self.relation[0].weight[:, 2 * emb:2 * emb + message]
        relation_hidden = F.linear(messages, wm) + relation_target[:, target_ids, None] + relation_source[None, None]
        logits = self.relation[2](self.relation[1](relation_hidden)).squeeze(-1) / self.config.temperature_relation
        self_mask = target_ids[:, None] == torch.arange(self.channels, device=messages.device)[None]
        attention = torch.softmax(logits.masked_fill(self_mask[None], -torch.inf), dim=-1)
        aggregate = (attention[..., None] * messages).sum(dim=-2)
        decoded = F.linear(aggregate, self.decoder_n[0].weight[:, :message], self.decoder_n[0].bias)[:, :, None]
        decoded = decoded + decoder_target[target_ids][None, :, None] + decoder_scale[None, None]
        prediction = self.decoder_n[2](self.decoder_n[1](decoded))
        concentration = attention.topk(self.top_r, dim=-1).values.sum(dim=-1) if self.top_r else attention.sum(dim=-1)
        magnitude = messages.abs().amax(dim=-1).masked_fill(self_mask[None], 0)
        diagnostics = torch.stack((magnitude.sum(dim=-1) / (self.channels - 1),
                                   (magnitude * attention).sum(dim=-1) / (self.channels - 1)), dim=-1)
        return prediction, concentration, diagnostics

    def _route(self, encoded, context, target_ids):
        """A target chunk still normalizes over ALL nonself sources and ALL scales."""
        batch, channels, scales, _ = encoded.shape
        count = len(target_ids)
        target = self.variable_n(target_ids)
        source = self.variable_n.weight
        scale = self.scale_n.weight
        shape = (batch, count, channels, scales)
        h = encoded[:, None].expand(*shape, -1)
        ei = target[None, :, None, None].expand(*shape, -1)
        ej = source[None, None, :, None].expand(*shape, -1)
        em = scale[None, None, None].expand(*shape, -1)
        ci = context[:, target_ids, None, None].expand(*shape, -1)
        weights_scale = torch.softmax(
            self.scale_gate(torch.cat((ei, ej, em, h, ci), dim=-1)).squeeze(-1)
            / self.config.temperature_scale, dim=-1)
        messages_scale = self.value(torch.cat((h, ei, ej, em), dim=-1))
        messages = (weights_scale[..., None] * messages_scale).sum(dim=-2)
        rel_input = torch.cat((ei[..., 0, :], ej[..., 0, :], messages, ci[..., 0, :]), dim=-1)
        logits = self.relation(rel_input).squeeze(-1) / self.config.temperature_relation
        self_mask = target_ids[:, None] == torch.arange(channels, device=encoded.device)[None, :]
        attention = torch.softmax(logits.masked_fill(self_mask[None], -torch.inf), dim=-1)
        aggregate = (attention[..., None] * messages).sum(dim=-2)
        decoder_input = torch.cat((
            aggregate[:, :, None].expand(batch, count, scales, -1),
            target[None, :, None].expand(batch, count, scales, -1),
            scale[None, None].expand(batch, count, scales, -1)), dim=-1)
        prediction = self.decoder_n(decoder_input)
        if self.top_r:
            concentration = attention.topk(self.top_r, dim=-1).values.sum(dim=-1)
        else:
            concentration = attention.sum(dim=-1)  # d=2: exactly one source, no penalty
        magnitude = messages.abs().amax(dim=-1).masked_fill(self_mask[None], 0)
        weighted_magnitude = magnitude * attention
        # Small per-row diagnostic outputs; full weights are optional and not retained by the trainer.
        diagnostics = torch.stack((magnitude.sum(dim=-1) / (channels - 1),
                                   weighted_magnitude.sum(dim=-1) / (channels - 1)), dim=-1)
        return prediction, concentration, diagnostics

    def forward(self, windows):
        batch, _, channels = windows.shape
        coarse = self.observations.coarse(windows)
        embeddings = self.variable_n.weight
        encoded = []
        for index, z in enumerate(coarse):
            blocks = z.shape[2]
            sequence = torch.cat((z.float(),
                                  embeddings[None, :, None].expand(batch, channels, blocks, -1),
                                  self.scale_n.weight[index][None, None, None].expand(
                                      batch, channels, blocks, -1)), dim=-1)
            h = self.encoder_n(sequence.reshape(batch * channels, blocks, -1),
                               readouts=(blocks - 1,) if self.config.prune_tcn else None)[:, -1]
            encoded.append(h.reshape(batch, channels, -1))
        encoded = torch.stack(encoded, dim=2)
        variable_context = self.context(encoded.flatten(start_dim=2))
        # A masked sum (rather than total-minus-self) also avoids cancellation by the excluded input.
        context = torch.einsum("ij,bjc->bic", self.context_mask, variable_context)
        predictions, concentrations, diagnostics = [], [], []
        projected = self._route_projections(encoded, context) if self.config.factorized_routing else None
        for start in range(0, channels, self.config.target_chunk_size):
            ids = torch.arange(start, min(start + self.config.target_chunk_size, channels), device=windows.device)
            if projected is not None:
                if self.training and self.config.gradient_checkpointing:
                    chunk = checkpoint(self._route_projected, *projected, ids, use_reentrant=False)
                else:
                    chunk = self._route_projected(*projected, ids)
            elif self.training and self.config.gradient_checkpointing:
                chunk = checkpoint(self._route, encoded, context, ids, use_reentrant=False)
            else:
                chunk = self._route(encoded, context, ids)
            predictions.append(chunk[0])
            concentrations.append(chunk[1])
            diagnostics.append(chunk[2])

        # No current target, trend estimate, or target statistic enters the self encoder.
        history = windows[:, :-1].transpose(1, 2).float()
        own_id = self.variable_s.weight
        own_input = torch.cat((history[..., None], own_id[None, :, None].expand(
            batch, channels, self.config.win_size - 1, -1)), dim=-1)
        own_hidden = self.encoder_s(own_input.reshape(batch * channels, self.config.win_size - 1, -1),
                                    readouts=self.self_readouts if self.config.prune_tcn else None)
        own_hidden = own_hidden.reshape(batch, channels, own_hidden.shape[1], -1)
        if self.config.prune_tcn:
            states = own_hidden[:, :, :-1]
        else:
            states = torch.stack([own_hidden[:, :, self.config.win_size - h - 1] for h in self.config.spans], dim=2)
        raw = self.stat_head(torch.cat((states,
            own_id[None, :, None].expand(batch, channels, len(self.config.spans), -1),
            self.span_s.weight[None, None].expand(batch, channels, len(self.config.spans), -1)), dim=-1))
        stat = torch.stack((self.a_low + (self.a_high - self.a_low) * torch.sigmoid(raw[..., 0]),
                            torch.tanh(raw[..., 1]), torch.sigmoid(raw[..., 2])), dim=-1)
        stat = torch.where(self.fixed_s[None], self.fixed_values_s[None], stat)
        point = self.point_head(torch.cat((own_hidden[:, :, -1], own_id[None].expand(batch, -1, -1)), dim=-1)).squeeze(-1)
        concentration = torch.cat(concentrations, dim=1)
        regularizer = F.relu(concentration - self.rho_a).square().mean() if self.top_r else concentration.sum() * 0
        return {"prediction_n": torch.cat(predictions, dim=1), "prediction_s": stat,
                "prediction_p": point, "target_n": torch.stack([z[:, :, -1] for z in coarse], dim=2),
                "concentration": concentration, "regularizer": regularizer,
                "message_diagnostics": torch.cat(diagnostics, dim=1)}

    def errors(self, windows, output):
        n = (output["target_n"] - output["prediction_n"]).square()
        s = (self.observations.statistics(windows, self.energy_floor) - output["prediction_s"]).square()
        p = (windows[:, -1].to(torch.float64) - output["prediction_p"]).square()
        return n, s, p

    def losses(self, windows, output):
        n, s, p = self.errors(windows, output)
        loss_n = (n / self.variance_n.clamp_min(self.config.epsilon_v)).mean()
        loss_s = self.config.chi * (p / self.variance_p.clamp_min(self.config.epsilon_v)).mean()
        loss_s = loss_s + (1 - self.config.chi) * (s / self.variance_s.clamp_min(self.config.epsilon_v)).mean()
        reconstruction = self.config.lambda_n * loss_n + self.config.lambda_s * loss_s
        return {"loss": reconstruction + self.config.lambda_a * output["regularizer"],
                "reconstruction": reconstruction, "loss_n": loss_n, "loss_s": loss_s,
                "regularizer": output["regularizer"]}
