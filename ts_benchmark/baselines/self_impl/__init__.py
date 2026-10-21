"""Load only the requested baseline and its dependencies."""

from importlib import import_module

__all__ = [
    "VAR_model",
    "LOF",
    "DCdetector",
    "AnomalyTransformer",
    "ModernTCN",
    "DualTF",
    "TFAD",
    "TranAD",
    "MatrixProfile",
    "LeftSTAMPi",
    "KMeans",
    "DWT_MLEAD",
    "SAND",
    "Torsk",
    "EIF",
    "ContraAD",
    "CATCH",
    "CrossAD",
    "DynamicNeighborhoodStat",
]

_MODELS = {
    "LOF": ("LOF.lof", "LOF"),
    "VAR_model": ("VAR.VAR", "VAR_model"),
    "DCdetector": ("DCdetector.DCdetector", "DCdetector"),
    "AnomalyTransformer": ("Anomaly_trans.AnomalyTransformer", "AnomalyTransformer"),
    "ModernTCN": ("ModernTCN.ModernTCN", "ModernTCN"),
    "DualTF": ("DualTF.DualTF", "DualTF"),
    "TFAD": ("TFAD.TFAD", "TFAD"),
    "MatrixProfile": ("MatrixProfile.MatrixProfile", "MatrixProfile"),
    "TranAD": ("TranAD.TranAD", "TranAD"),
    "LeftSTAMPi": ("LeftSTAMPi.LeftSTAMPi", "LeftSTAMPi"),
    "KMeans": ("KMeans.KMeans", "KMeans"),
    "DWT_MLEAD": ("DWT_MLEAD.DWTMLEAD", "DWT_MLEAD"),
    "SAND": ("SAND.SAND", "SAND"),
    "Torsk": ("torsk.torsk", "Torsk"),
    "EIF": ("eif.eif", "EIF"),
    "ContraAD": ("ContraAD.ContraAD", "ContraAD"),
    "Series2Graph": ("Series2Graph.Series2Graph", "Series2Graph"),
    "CATCH": ("CATCH.CATCH", "CATCH"),
    "CrossAD": ("CrossAD.CrossAD", "CrossAD"),
    "DynamicNeighborhoodStat": ("dynamic_neighborhood_stat.DynamicNeighborhoodStat", "DynamicNeighborhoodStat"),
}


def load_baseline(name):
    if name not in _MODELS:
        raise AttributeError("module {!r} has no attribute {!r}".format(__name__, name))
    module, attribute = _MODELS[name]
    value = getattr(import_module("{}.{}".format(__name__, module)), attribute)
    globals()[name] = value
    return value


def __getattr__(name):
    return load_baseline(name)


def __dir__():
    return sorted(set(globals()) | set(_MODELS))
