"""A global amplitude multiplier, preserving family definitions and nominal rho."""
from copy import deepcopy
from contextlib import contextmanager
import seven_shifts as original

BASE_PARAMETERS=deepcopy(original.PARAMETERS)
SCALED=['x_weight_exponent','affine_scale_increment','affine_translation_sd',
        'support_translation_sd','regression_response_drift_sd','max_boundary_flip_fraction']


def candidates():return [{'name':f'amplitude_{x:g}','amplitude_multiplier':x} for x in [.125,.25,.375,.5,.625,.75,1.,.55,.575,.6,.56,.565,.58,.59]]


@contextmanager
def amplitude(value):
    previous=deepcopy(original.PARAMETERS)
    try:
        original.PARAMETERS.update({k:BASE_PARAMETERS[k]*value for k in SCALED})
        yield
    finally:original.PARAMETERS.clear();original.PARAMETERS.update(previous)


def materialize(meta,family,rho,clean,multiplier):
    with amplitude(multiplier):arrays,info=original.materialize(meta,family,rho,clean)
    info.update(amplitude_multiplier=multiplier,nominal_rho=rho,
                calibration_protocol='global amplitude frozen after independent-query calibration')
    return arrays,info
