"""Fix classification perturbation polarity, keeping rows/labels/strengths fixed.

The available LimiX helper uses encoded-label >= median(encoded-label) for the
feature direction. With an imbalanced binary source, both labels can satisfy
that inequality; with a negatively associated feature the direction can also
reinforce the original signal. Here the opposite direction is defined from
source class-conditional feature means, as in the recovered historical helper.
No target labels or target scores determine the direction or amplitude.
"""
from __future__ import annotations
import hashlib
import numpy as np
from prepare_data import protocol as original

REVISION='class_conditional_feature_polarity_v2'


def reuse_unchanged_clean(source,output):
    """Reuse identical rho=0 predictions, changing only run provenance."""
    import json,shutil
    from prepare_data import RUN,atomic_json
    if not source.exists() or not source.with_suffix('.json').exists():return False
    old=json.loads(source.with_suffix('.json').read_text())
    if old.get('status')!='COMPLETE':return False
    output.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,output)
    atomic_json(output.with_suffix('.json'),old|{'track':'five_corrected','polarity_revision':REVISION,
        'cached_unchanged_clean_from':str(source.relative_to(RUN)),'source_condition_unchanged':True})
    return True


def manipulate_source_context(task,X_source,y_source,*,family,rho,seed,shift_scale,label_scale=1.):
    if task!='classification':
        return original.manipulate_source_context(task,X_source,y_source,family=family,rho=rho,
            seed=seed,shift_scale=shift_scale,label_scale=label_scale)
    X=np.asarray(X_source,dtype=np.float64);y=np.asarray(y_source)
    classes,encoded=original._encoded_target(y);top=original._top_feature(X,encoded)
    count=0 if family=='original' or rho<=0 else min(len(y),max(1,int(np.ceil(rho*len(y)))))
    rows=original._select_rows(X,encoded,family=family,count=count,top=top,seed=seed) if count else np.empty(0,dtype=int)
    means=np.array([X[y==label,top].mean() for label in classes])
    distances=np.abs(means[:,None]-means[None]);np.fill_diagonal(distances,-np.inf)
    opposite=np.argmax(distances,axis=1) if len(classes)>1 else np.zeros(1,dtype=int)
    polarity=np.sign(means[opposite]-means)[encoded.astype(int)]

    def apply(A,b,selected,kind):
        A=np.asarray(A,dtype=np.float64).copy();b=b.copy()
        if not len(selected):return A,b
        current_classes,current_encoded=original._encoded_target(b)
        loc=A.mean(0);sd=np.where(A.std(0)>1e-12,A.std(0),1.)
        direction=polarity[selected]
        if kind=='shortcut':A[selected,top]=loc[top]+direction*shift_scale*sd[top]
        elif kind in ['boundary','band']:
            changed=selected[::2 if kind=='boundary' else 3]
            b[changed]=current_classes[(current_encoded[changed].astype(int)+1)%len(current_classes)]
            A[selected,top]+=direction*(.35 if kind=='boundary' else .22)*shift_scale*sd[top]
        elif kind=='local':
            changed=selected[::2];b[changed]=current_classes[(current_encoded[changed].astype(int)+1)%len(current_classes)]
            A[selected,top]=loc[top]+direction*.7*shift_scale*sd[top]
        else:raise ValueError(kind)
        return A,b

    modified,labels=X.copy(),y.copy()
    if family=='mixed':
        for kind,chunk in zip(['shortcut','boundary','band','local'],np.array_split(rows,4)):
            modified,labels=apply(modified,labels,np.asarray(chunk),kind)
    elif count:modified,labels=apply(modified,labels,rows,family)
    return original.ManipulationResult(X_context=np.asarray(modified,dtype=np.float32),y_context=labels,
        affected_rows=rows,top_feature=top,metadata={'task':task,'family':family,'rho':rho,'seed':seed,
        'affected_rows':len(rows),'source_rows':len(y),'query_argument_present':False,
        'revision':REVISION,'only_feature_polarity_corrected':True})


def verify_definition():
    X=np.array([[0.,0.],[.1,1.],[.2,2.],[1.,3.]])
    y=np.array([0,0,0,1]);rows=[]
    for family in original.FAMILIES:
        a=manipulate_source_context('classification',X,y,family=family,rho=1.,seed=1829,shift_scale=6.)
        b=manipulate_source_context('classification',X,1-y,family=family,rho=1.,seed=1829,shift_scale=6.)
        legacy=original.manipulate_source_context('classification',X,y,family=family,rho=1.,seed=1829,shift_scale=6.)
        assert np.array_equal(a.X_context,b.X_context),(family,'label-name dependence')
        assert np.array_equal(a.y_context,legacy.y_context),(family,'labels changed by polarity correction')
        assert np.array_equal(a.affected_rows,legacy.affected_rows),(family,'affected rows changed')
        rows.append({'family':family,'label_permutation_equivariance':True,'labels_unchanged_vs_legacy':True,'affected_rows_unchanged':True})
    return rows


if __name__=='__main__':
    from prepare_data import RUN,atomic_json
    result={'status':'PASS','revision':REVISION,'checks':verify_definition(),
        'scope':'same source/target cohorts, same four rho levels, same selected rows, same label edits, same amplitudes; fix only feature polarity'}
    atomic_json(RUN/'results/classification_polarity_definition_audit.json',result)
    print(result)
