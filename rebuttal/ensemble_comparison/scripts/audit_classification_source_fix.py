"""Audit scientific invariants of the source-only feature-polarity correction."""
import json
import numpy as np
from prepare_data import RUN,protocol,atomic_json
from shared_reference import specifications,load_condition,geometry
from corrected_source_protocol import manipulate_source_context,REVISION,verify_definition
from specific_reference import repair_labels,importance

rows=[];missing_classes=[]
for meta in specifications('five_corrected'):
    with np.load(RUN/'cache/five/classification'/f"{meta['task_id']}.npz") as f:
        X=f['X_source'];y=f['y_source'];Q=f['X_target'];target=f['y_target']
    for family in protocol.FAMILIES:
        for rho in [20,50,80,100]:
            kw={'family':family,'rho':rho/100.,'seed':1729+rho,'shift_scale':6.,'label_scale':1.}
            old=protocol.manipulate_source_context('classification',X,y,**kw)
            new=manipulate_source_context('classification',X,y,**kw)
            assert np.array_equal(old.y_context,new.y_context)
            assert np.array_equal(old.affected_rows,new.affected_rows)
            assert old.top_feature==new.top_feature
            arrays,info=load_condition(meta,family,rho)
            assert np.array_equal(arrays['X_target'],Q) and np.array_equal(arrays['y_target'],target)
            assert np.array_equal(arrays['X_source'],new.X_context)
            top=new.top_feature
            assert np.array_equal(np.delete(old.X_context,top,axis=1),np.delete(new.X_context,top,axis=1))
            for version in [old,new]:
                if family not in ['boundary','band','mixed']:continue
                Z,_,_,_=geometry(version.X_context,Q)
                order=importance(version.X_context,version.y_context.astype(float))
                columns=np.array([j for j in range(X.shape[1]) if j!=int(order[0])]);columns=columns if len(columns) else np.array([0])
                for A,threshold in [(Z[:,columns],.6),(Z[:,columns],.7),(Z[:,columns],.8),(Z,.7)]:
                    repaired=repair_labels(A,version.y_context,threshold)
                    if set(repaired)!=set(version.y_context):missing_classes.append({'task_id':meta['task_id'],'family':family,'rho':rho,'threshold':threshold})
            rows.append({'task_id':meta['task_id'],'data_id':meta['data_id'],'family':family,'rho':rho,
                         'X_context_sha256':info['X_context_sha256'],'y_context_sha256':info['y_context_sha256'],
                         'labels_unchanged':True,'selected_rows_unchanged':True,'nonselected_columns_unchanged':True,
                         'target_unchanged':True})
    print('SOURCE_FIX_AUDITED',meta['task_id'],flush=True)
result={'status':'PASS' if not missing_classes else 'FAIL','revision':REVISION,'conditions':len(rows),
        'tests':verify_definition(),'rows':rows,'specific_repair_class_coverage_failures':missing_classes,
        'amplitude_or_strength_tuned_on_target_scores':False,'original_main_table_bit_exact_claim':False}
atomic_json(RUN/'results/classification_source_fix_audit.json',result)
print(result['status'],len(rows),'missingclass',len(missing_classes),flush=True)
if missing_classes:raise SystemExit(1)
