"""Reserve labeled target-adaptation examples in every deployed context."""
import numpy as np
import valid_cluster_shared_contexts_v282 as parent


def mix(name,X,y,prediction_X,target_X,target_y,task,quota_y,columns=None):
    n=min(len(y),256-len(target_y))
    assert n>0 and len(target_y)<=64
    if task=='classification':
        rows=parent.balanced_rows(np.arange(len(y),dtype=float),y,quota_y,n)
    else:
        rows=np.arange(n)
    AX=target_X[:,columns] if name=='feature_subspace' and task=='classification' else target_X
    A=np.concatenate([X[rows],AX]);b=np.concatenate([y[rows],target_y])
    assert A.shape[1]==prediction_X.shape[1] and len(b)<=256
    return A,b,{'retained_source_positions_in_candidate':rows.tolist(),
        'source_rows':len(rows),'labeled_target_adaptation_rows':len(target_y)}
