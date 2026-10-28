"""Explicit training budgets; the default reproduces the supplied method."""
import torch


PROFILES = {
    "original": {},
    "extended": dict(
        hidden_dim_candidates=(32, 64, 128),
        batch_size=4,
        predictor_epochs=60, predictor_min_epochs=10, predictor_patience=12,
        learning_rate=3e-4, predictor_gradient_clip=1.,
        predictor_scheduler="plateau", predictor_lr_patience=3,
        corrector_linear_epochs=20, corrector_nonlinear_epochs=20,
        auxiliary_train_pairs=1536,
        score_epochs=100, score_min_epochs=10, score_patience=15,
        score_learning_rate=3e-3, score_bias_learning_rate=3e-2,
        score_scheduler="plateau", score_lr_patience=5,
    ),
}


def profile_options(kwargs):
    name = kwargs.get("training_profile", "original")
    if name not in PROFILES:
        raise ValueError("training_profile must be original or extended")
    options = dict(PROFILES[name])
    options.update(kwargs)
    # An explicit fixed width or step budget takes precedence over a preset search
    # or epoch budget. Persisted configurations include both fields explicitly.
    if "hidden_dim" in kwargs and "hidden_dim_candidates" not in kwargs:
        options["hidden_dim_candidates"] = None
    for stage in ("linear", "nonlinear"):
        if f"corrector_{stage}_steps" in kwargs and f"corrector_{stage}_epochs" not in kwargs:
            options[f"corrector_{stage}_epochs"] = None
    for alias, target in (("lr", "learning_rate"), ("num_epochs", "predictor_epochs")):
        if kwargs.get(alias) is not None:
            if target in kwargs and kwargs[target] != kwargs[alias]:
                raise ValueError(f"Conflicting ASEI hyperparameters: {alias} and {target}")
            options[target] = kwargs[alias]
    return options


def make_scheduler(optimizer, config, stage):
    if getattr(config, f"{stage}_scheduler") == "none":
        return None
    base = optimizer.param_groups[0]["lr"]
    # Preserve the bias/model LR ratio when reducing both groups.
    minima = [min(group["lr"], config.minimum_learning_rate * group["lr"] / base)
              for group in optimizer.param_groups]
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=config.lr_decay_factor,
        patience=getattr(config, f"{stage}_lr_patience"),
        threshold=0., threshold_mode="abs", min_lr=minima,
    )


def score_optimizer(scorer, config):
    if config.score_bias_learning_rate is None:
        return torch.optim.Adam(scorer.parameters(), lr=config.score_learning_rate, weight_decay=0)
    other = [value for name, value in scorer.named_parameters() if name != "training_bias"]
    return torch.optim.Adam([
        dict(params=other, lr=config.score_learning_rate, name="scoring_parameters"),
        dict(params=[scorer.training_bias], lr=config.score_bias_learning_rate, name="training_bias"),
    ], weight_decay=0)
