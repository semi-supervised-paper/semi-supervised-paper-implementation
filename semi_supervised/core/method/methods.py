from ..utils.fun_util import export
from .mean_teacher import MeanTeacher
from .temporal_ensembling import TemporalEnsembling
from .vat import VAT


@export
def mean_teacher(**kwargs):
    return MeanTeacher(**kwargs)


@export
def temporal_ensembling(**kwargs):
    return TemporalEnsembling(**kwargs)


@export
def vat(**kwargs):
    return VAT(**kwargs)
