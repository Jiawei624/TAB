"""Load only the selected baseline and its optional dependencies."""

from importlib import import_module


_MODELS = {
    "ASEI": "ASEI.asei",
    "VAR_model": "VAR.VAR",
    "LOF": "LOF.lof",
    "DCdetector": "DCdetector.DCdetector",
    "AnomalyTransformer": "Anomaly_trans.AnomalyTransformer",
    "ModernTCN": "ModernTCN.ModernTCN",
    "DualTF": "DualTF.DualTF",
    "TFAD": "TFAD.TFAD",
    "TranAD": "TranAD.TranAD",
    "MatrixProfile": "MatrixProfile.MatrixProfile",
    "LeftSTAMPi": "LeftSTAMPi.LeftSTAMPi",
    "KMeans": "KMeans.KMeans",
    "DWT_MLEAD": "DWT_MLEAD.DWTMLEAD",
    "SAND": "SAND.SAND",
    "Torsk": "torsk.torsk",
    "EIF": "eif.eif",
    "ContraAD": "ContraAD.ContraAD",
    "Series2Graph": "Series2Graph.Series2Graph",
    "CATCH": "CATCH.CATCH",
    "CrossAD": "CrossAD.CrossAD",
}

__all__ = list(_MODELS)


def __getattr__(name):
    if name not in _MODELS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    model = getattr(import_module(f"{__name__}.{_MODELS[name]}"), name)
    globals()[name] = model
    return model
