"""Repair auxiliary assignments to empty query clusters without changing rules."""
import numpy as np
from sklearn.metrics import pairwise_distances
import pooled_shared_contexts_v279 as parent

balanced_rows=parent.balanced_rows
contexts=parent.contexts


def geometry(X,y,Q,C,task):
    model=parent.geometry(X,y,Q,C,task)
    active=np.unique(model['query_groups'])
    missing=~np.isin(model['reference_groups'],active)
    if np.any(missing):
        distance=pairwise_distances(model['U'][missing],model['centers'][active],metric='sqeuclidean')
        model['reference_groups'][missing]=active[np.argmin(distance,axis=1)]
    assert np.isin(model['reference_groups'],active).all()
    model['auxiliary_assignments_repaired']=int(missing.sum())
    return model
