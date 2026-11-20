"""Reindex sparse episodic labels before training-ensemble construction.

Identity on contiguous labels; successful predecessor runs can be retained.
Both context and training-query labels receive the same per-chunk bijection.
No benchmark or final-evaluation label is accessed.
"""
import numpy as np
import shared_finetune_pilot_v300 as parent

source=parent.source
GPU_POLICY=parent.GPU_POLICY
PilotClassifier=parent.PilotClassifier
PilotRegressor=parent.PilotRegressor
class_split=parent.class_split
model_digest=parent.model_digest
episode_data=parent.episode_data
_original_build=episode_data._build_meta_batch


def supported_label_meta_batch(X_chunk,y_chunk,*,classification,**kwargs):
    if classification:
        labels=np.unique(y_chunk)
        if not np.array_equal(labels,np.arange(len(labels))):
            y_chunk=np.searchsorted(labels,y_chunk).astype(np.int64)
    return _original_build(X_chunk,y_chunk,classification=classification,**kwargs)


episode_data._build_meta_batch=supported_label_meta_batch
