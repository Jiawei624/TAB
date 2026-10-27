"""Import the selected model without loading every optional baseline."""

from importlib import import_module

_MODELS = {
    "TimerModel": "ts_benchmark.baselines.pre_train.model.timer",
    "TimesFM": "ts_benchmark.baselines.pre_train.model.timesfm",
    "UniTS": "ts_benchmark.baselines.pre_train.model.units",
    "Moment": "ts_benchmark.baselines.pre_train.model.moment",
    "TinyTimeMixer": "ts_benchmark.baselines.pre_train.model.tinytimemixer",
    "Chronos": "ts_benchmark.baselines.pre_train.model.ChronosModel",
    "DadaModel": "ts_benchmark.baselines.pre_train.model.dada",
}

__all__ = list(_MODELS)


def __getattr__(name):
    if name not in _MODELS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    model = getattr(import_module(_MODELS[name]), name)
    globals()[name] = model
    return model
