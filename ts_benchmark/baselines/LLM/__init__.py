"""Import the selected model without loading every optional baseline."""

from importlib import import_module

_MODELS = {
    "GPT4TSModel": "ts_benchmark.baselines.LLM.model.GPT4TS_model",
    "UniTimeModel": "ts_benchmark.baselines.LLM.model.UniTime_model",
    "CALFModel": "ts_benchmark.baselines.LLM.model.CALF_model",
    "LLMMixerModel": "ts_benchmark.baselines.LLM.model.LLMMixer_model",
}

__all__ = list(_MODELS)


def __getattr__(name):
    if name not in _MODELS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    model = getattr(import_module(_MODELS[name]), name)
    globals()[name] = model
    return model
