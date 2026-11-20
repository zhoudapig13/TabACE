"""Reconstruct a frozen source-OOF policy without repeating its HPO search."""
import numpy as np
import calibration_head_v45 as h


def replay(cells,m,block,policy,fallback):
    h.risk.row_folds=lambda meta,track,n:h.folds(meta,block,n)
    d=h.risk.prepare(cells,m,block)
    model=policy['reliability']
    if model is None:baseline=np.concatenate([fallback[c['key']] for c in cells])
    else:
        baseline=h.risk.combine(d['P'],d['risks'][d['F'],d['G']],d['prior'][d['F']],
            d['counts'][d['F'],d['G']],d['sigmas'][d['F'],d['G']],model['profile'])
    if policy['head'] is None:prediction=baseline
    else:
        values=[np.empty_like(c['y'],dtype=np.float64) for c in cells];groups={}
        for i,c in enumerate(cells):groups.setdefault(str(c['meta']['task_id']),[]).append(i)
        for indices in groups.values():
            c=cells[indices[0]];cal=h.calibration(c['meta'],block);X=cal['X_validation'];F=h.folds(c['meta'],block,len(c['y']))
            matrices=[h.features(X,cells[i],m) for i in indices]
            for fold in range(4):
                train=F!=fold;valid=F==fold
                A=np.concatenate([M[train] for M in matrices]);y=np.concatenate([cells[i]['y'][train] for i in indices])
                fitted=h.fit_estimator(A,y,len(indices),int(train.sum()),policy['head']['profile'])
                for i,M in zip(indices,matrices):values[i][valid]=h.head_predict(M[valid],fitted)
        a=policy['head']['blend'];prediction=(1-a)*baseline+a*np.concatenate(values)
    assert np.isfinite(prediction).all()
    return {c['key']:prediction[start:start+n] for c,start,n in zip(cells,d['starts'],d['lengths'])},h.risk.score(d,prediction)[0]
