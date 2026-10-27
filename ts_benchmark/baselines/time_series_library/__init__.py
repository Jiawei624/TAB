"""Import the selected model without loading every optional baseline."""

from importlib import import_module

_MODELS = {
    "Autoformer": "ts_benchmark.baselines.time_series_library.models.Autoformer",
    "Crossformer": "ts_benchmark.baselines.time_series_library.models.Crossformer",
    "DLinear": "ts_benchmark.baselines.time_series_library.models.DLinear",
    "ETSformer": "ts_benchmark.baselines.time_series_library.models.ETSformer",
    "FEDformer": "ts_benchmark.baselines.time_series_library.models.FEDformer",
    "FiLM": "ts_benchmark.baselines.time_series_library.models.FiLM",
    "Informer": "ts_benchmark.baselines.time_series_library.models.Informer",
    "iTransformer": "ts_benchmark.baselines.time_series_library.models.iTransformer",
    "Koopa": "ts_benchmark.baselines.time_series_library.models.Koopa",
    "LightTS": "ts_benchmark.baselines.time_series_library.models.LightTS",
    "Linear": "ts_benchmark.baselines.time_series_library.patchs.Linear",
    "MICN": "ts_benchmark.baselines.time_series_library.models.MICN",
    "NLinear": "ts_benchmark.baselines.time_series_library.patchs.NLinear",
    "Nonstationary_Transformer": "ts_benchmark.baselines.time_series_library.models.Nonstationary_Transformer",
    "PatchTST": "ts_benchmark.baselines.time_series_library.models.PatchTST",
    "Pyraformer": "ts_benchmark.baselines.time_series_library.models.Pyraformer",
    "Reformer": "ts_benchmark.baselines.time_series_library.models.Reformer",
    "TimesNet": "ts_benchmark.baselines.time_series_library.models.TimesNet",
    "Transformer": "ts_benchmark.baselines.time_series_library.models.Transformer",
    "Triformer": "ts_benchmark.baselines.time_series_library.patchs.Triformer",
}

__all__ = list(_MODELS)


def __getattr__(name):
    if name not in _MODELS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    model = getattr(import_module(_MODELS[name]), name)
    globals()[name] = model
    return model


