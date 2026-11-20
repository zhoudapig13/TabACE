"""Restore available selection pool versus per-member context-row budget.

The canonical Base inputs/prediction remain unchanged. Extra pool rows are
observed under the same registered corruption mechanism. Constructors receive
no held-out query labels. This module supplies data/contexts only; fitting and
query evaluation remain separate source-calibrated stages.
"""
import numpy as np
import tabace_context_reproduction_v28 as reference
import tabace_validation_banks_v28 as validation
from prepare_data import RUN,protocol

OUT=RUN/'results/tabace_full_pool_v34'


def filtered_class_pool(pool,split):
    if split is None:return pool
    return pool|dict(H=pool['H'][split['h_keep']],hy=pool['hy'][split['h_keep']],
        R=pool['R'][split['r_keep']],ry=pool['ry'][split['r_keep']],
        H0=pool['H0'][split['h_keep'][:len(pool['H0'])]],hy0=pool['hy0'][split['h_keep'][:len(pool['H0'])]],
        R0=pool['R0'][split['r_keep'][:len(pool['R0'])]],ry0=pool['ry0'][split['r_keep'][:len(pool['R0'])]])


def class_world(pool,family,rho):
    # Same operators as scaling.class_seven, applied to all available H/R rows.
    from sklearn.linear_model import LogisticRegression
    from seven_shifts import standardize
    cfg=reference.frozen.configuration()['classification_seven']['generator_parameters'];s=rho/100.
    H,hy,R,ry=[pool[k].copy() for k in ['H','hy','R','ry']]
    H0,R0=pool['H0'],pool['R0'];ref=np.vstack([H0,R0]);j=int(np.argmax(ref.var(0)))
    if family in ['covariate','selection']:
        learner=LogisticRegression(max_iter=500,random_state=0).fit(standardize(ref,ref),np.r_[pool['hy0'],pool['ry0']]) if family=='selection' else None
        majority=int(np.mean(np.r_[pool['hy0'],pool['ry0']])>=.5)
        for A,b in [(H,hy),(R,ry)]:
            score=standardize(A,ref)[:,j] if learner is None else learner.predict_proba(standardize(A,ref))[:,majority]
            logits=np.clip(cfg['x_weight_exponent']*s*score,-15,15);prob=np.exp(logits-logits.max());prob/=prob.sum()
            rows=np.random.default_rng(0).choice(len(score),size=len(score),replace=True,p=prob)
            A[:],b[:]=A[rows].copy(),b[rows].copy()
    elif family in ['concept','temporal']:
        j=int(np.argmax(H0.var(0)));order=np.argsort(np.abs(R[:,j]-np.median(H0[:,j])),kind='stable')
        chosen=order[:int(np.ceil(cfg['max_boundary_flip_fraction']*s*len(ry)))];ry[chosen]=1-ry[chosen]
        if family=='temporal':
            mean=H0.mean(0);sd=np.maximum(H0.std(0),1e-6)
            R=mean+(R-mean)*(1+cfg['affine_scale_increment']*s)+cfg['affine_translation_sd']*s*sd
    return H.astype(np.float32),hy,R.astype(np.float32),ry


def materialize(role,task,block,meta,family,rho,budget,pool=None,split=None):
    track='seven' if block=='seven' else 'five'
    if pool is None:pool=reference.scaling.prepare(task,track,meta)
    if role=='validation' and split is None:split=validation.validation_split(task,track,meta,pool)
    if role=='query':canonical,scale,base,origin,provisional=reference.query_input(task,block,meta,family,rho,budget,pool)
    else:
        canonical,scale=validation.materialize(task,block,meta,family,rho,budget,pool,split)
        origin=str(reference.OUT/'banks/validation'/str(budget)/task/block/str(meta['task_id'])/f'{family}_{rho}.npz')
        with np.load(origin) as z:base=z['base']
        provisional=task=='regression' and block=='seven' and budget>256
    core_X,core_y=canonical['X_source'],canonical['y_source'];n=len(core_y)
    if task=='classification' and block=='seven':
        filtered=filtered_class_pool(pool,split);H,hy,R,ry=class_world(filtered,family,rho)
        total=min(budget,len(filtered['hy'])+len(filtered['ry']));nr=total//4;nh=total-nr
        nh=min(nh,len(H));nr=min(nr,len(R));assert nh+nr==n
        # Full-pool resampling/flip cutoffs differ from the canonical sample;
        # preserve exactly the observations the immutable Base received.
        H[:nh]=core_X[:nh];hy[:nh]=core_y[:nh]
        R[:nr]=core_X[nh:];ry[:nr]=core_y[nh:]
        X,y=np.vstack([H,R]),np.r_[hy,ry]
        history=(H,hy);recent=(R,ry)
        assert np.array_equal(H[:nh],core_X[:nh]) and np.array_equal(R[:nr],core_X[nh:])
    else:
        positions=split['train_positions'] if split is not None else np.arange(len(pool['y']))
        if task=='regression' and block=='five':
            import json
            rate=json.loads((reference.repaired.OUT/'frozen_five_query/parameters.json').read_text())['repair_probability']
            env=reference.repaired.environment(pool,meta,block,family,rho,positions)
            X,y,_=reference.repaired.context(env,rate,len(positions))
        elif task=='regression' and block=='seven':
            filtered=pool|dict(X=pool['X'][positions],y=pool['y'][positions])
            Q=reference.repaired.core_arrays(block,meta,family,rho)['X_target']
            X,y=reference.scaling.reg_seven(filtered,family,rho,len(positions),Q)
            X[:n]=core_X;y[:n]=core_y
        else:X,y=pool['X'][positions].copy(),pool['y'][positions].copy()
        assert np.array_equal(X[:n],core_X) and np.array_equal(y[:n],core_y)
        h=min(max(1,3*len(y)//4),len(y)-1);history=(X[:h],y[:h]);recent=(X[h:],y[h:])
    if split is not None:
        assert not np.intersect1d(split['source_original_ids'],split['validation_original_ids']).size
        assert not np.intersect1d(split['validation_original_ids'],pool['query_ids']).size
    assert len(y)>=n and len(y)<=8192 and all(np.isfinite(a).all() for a in [X,y])
    return dict(canonical=canonical,base=base,scale=scale,Base_origin=origin,provisional=provisional,
        X_pool=X,y_pool=y,history=history,reference=recent,
        canonical_input_hashes={k:protocol.array_sha256(v) for k,v in canonical.items()},
        pool_input_hashes=dict(X_pool=protocol.array_sha256(X),y_pool=protocol.array_sha256(y)),
        requested_context=budget,actual_Base_context=n,available_observed_pool=len(y),
        selection_scope='full observed TRAIN pool; per-branch row cap unchanged',
        extra_pre_corruption_labels_given_to_constructors=False)


def cap_view(view,budget,task,stream):
    name,A,b,T=view
    if len(b)>budget:
        rows=reference.specific.balanced(b,budget,seed=stream) if task=='classification' else reference.specific.response_balanced(b,budget,seed=stream)
        A,b=A[rows],b[rows]
        name=name+'_budget_sample'
    assert 0<len(b)<=budget
    return name,np.asarray(A,dtype=np.float32),b,np.asarray(T,dtype=np.float32)


def constructor_input(payload):
    # A deliberate allowlist: final query truths and their hashes stay outside
    # the context-construction interface.
    return {k:payload[k] for k in ['X_pool','y_pool','base','history','reference','actual_Base_context']}|dict(
        canonical={k:payload['canonical'][k] for k in ['X_source','y_source','X_target']})


def contexts(payload,task,block,family):
    assert set(payload['canonical'])=={'X_source','y_source','X_target'},'Pass constructor_input(payload), without query truths'
    X,y=payload['X_pool'],payload['y_pool'];c=payload['canonical'];Q=c['X_target'];n=payload['actual_Base_context']
    if task=='classification':
        common=reference.shared_contexts(X,y,Q,n,task,payload['base'])
        order=reference.specific.importance(X,y.astype(float));cols=np.array([j for j in range(X.shape[1]) if j!=int(order[0])])
        if not len(cols):cols=np.array([0])
        common[3]=('feature_subspace',c['X_source'][:,cols],c['y_source'],Q[:,cols])
    else:common=reference.shared.contexts(X,y,Q,c['X_source'],c['y_source'],n,task,base=payload['base'])
    extra=reference.specific.specific_contexts(X,y,Q,c['X_source'],c['y_source'],n,task,
        family=family,track='seven_reimplemented' if block=='seven' else 'five',
        history=payload['history'],reference=payload['reference'],base=payload['base'])
    assert len(common)==5 and len(extra)==5
    return [cap_view(v,n,task,3400+j) for j,v in enumerate(common)], [cap_view(v,n,task,3450+j) for j,v in enumerate(extra)]
