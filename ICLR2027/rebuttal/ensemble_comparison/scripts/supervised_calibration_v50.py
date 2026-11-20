"""REJECTED exploratory TRAIN-augmentation alternative; not the main protocol.

Original hypothesis was disproved by benchmark256_v22.load: BOTH frozen query
generation and existing TRAIN calibration generation use pseudo-labels from
the same protected-core ExtraTrees teacher. Switching only calibration to true
TRAIN labels would create a mismatch. Retained for provenance; do not deploy.
"""
import json
import numpy as np
import recovery_contexts_v44 as source
from prepare_data import RUN,atomic_json,protocol

OUT=RUN/'results/tabace_supervised_calibration_v50'


def shifted_features(meta,family,rho,X,y,Q,TRAIN_labels):
    from scaled_historical_feature_protocol import apply_feature_scaled_shift
    cfg=source.adaptation.raw.frozen.configuration()['classification_five']['family_settings'][family]
    kind='aucband' if family=='band' else family
    op=f'publicshift_d{meta["data_id"]}_{kind}_rho{rho}_beta{int(cfg["beta"]*100)}'
    shifted,_=apply_feature_scaled_shift(op,X,y,Q,TRAIN_labels,0,feature_multiplier=cfg['feature_multiplier'])
    labels,counts=np.unique(y,return_counts=True);positive=labels[1] if len(labels)==2 else labels[int(np.argmax(counts))]
    direction=X[y==positive].mean(0)-X[y!=positive].mean(0)
    return shifted+cfg.get('location_delta',0.)*(rho/100.)*direction


def diagnostic():
    rows=[]
    cohort=source.adaptation.raw.frozen.cohort('classification','five')
    chosen=[next(m for m in cohort if str(m['task_id'])==str(t)) for t in [167124,167121]]
    chosen.append(next(m for m in cohort if m['task_id'] not in [167124,167121]))
    for meta in chosen:
        pool=source.adaptation.raw.scaling.prepare('classification','five',meta)
        split=source.full.validation.validation_split('classification','five',meta,pool)
        X,y=pool['X'][:256],pool['y'][:256];Q,t=split['X_validation'],split['y_validation']
        for family,rho in [('shortcut',100),('boundary',100),('band',100),('local',100),('mixed',100)]:
            old=source.full.validation.class_five_shift(meta,family,rho,X,y,Q)
            new=shifted_features(meta,family,rho,X,y,Q,t)
            changed=np.any(old!=new,axis=1)
            rows.append(dict(task_id=meta['task_id'],dataset=meta['dataset'],family=family,rho=rho,
                calibration_rows=len(t),fraction_changed_rows=float(changed.mean()),
                max_abs_feature_difference=float(np.max(np.abs(old-new))),old_Q_hash=protocol.array_sha256(old),new_Q_hash=protocol.array_sha256(new)))
    atomic_json(OUT/'generator_diagnosis.json',dict(status='CONFIRMED' if any(r['fraction_changed_rows']>0 for r in rows) else 'NO_DIFFERENCE',
        comparison='pseudo-label TRAIN augmentation versus known-label TRAIN augmentation; fixed query generator unchanged',
        final_query_data_loaded=False,rows=rows))
    print('CALIBRATION_GENERATOR_DIAGNOSIS',json.dumps(rows),flush=True)


def plan():
    raise RuntimeError('Rejected hypothesis: frozen queries also use pseudo-label generation. Preserve the existing matched calibration protocol.')
    # Retained unreachable declaration documents the superseded hypothesis.
    diagnosis=json.loads((OUT/'generator_diagnosis.json').read_text());assert diagnosis['status']=='CONFIRMED'
    assert not (OUT/'plan.json').exists()
    atomic_json(OUT/'plan.json',dict(status='DECLARED_BEFORE_REGENERATING_CLASSIFICATION_CALIBRATION',
        source_bank=str(source.OUT),change='Use known held-out TRAIN labels for the label-dependent calibration corruption law',
        unchanged=['all final query inputs and labels','all Base query predictions','calibration row membership','all strengths','all context-row limits'],
        regenerated='classification five validation only',reuse='all queries and all other validation groups',
        datasets=72,strengths=[.2,.5,.8,1.],budgets=[256,1024,4096,8192],seed=0,
        shared_members=10,specific_members=21,extra_clean_TRAIN_supervision=True,
        TRAIN_labels_used_by_benchmark_generator=True,final_query_labels_used=False,
        calibration_truth_given_to_context_constructors=False,development_only=True,
        original_paper_replication=False,independent_test_claim=False,query_feedback_used_for_benchmark_design=True,Markdown_written=False))
    (OUT/'paper_targets.json').write_text((source.OUT/'paper_targets.json').read_text())
    for role in ['query','validation']:
        if role=='query':
            dest=OUT/'banks/query';dest.parent.mkdir(parents=True,exist_ok=True);dest.symlink_to(source.OUT/'banks/query',target_is_directory=True)
        else:
            for budget in [256,1024,4096,8192]:
                for task,blocks in [('classification',['clean','seven']),('regression',['clean','five','seven'])]:
                    for block in blocks:
                        dest=OUT/'banks/validation'/str(budget)/task/block;dest.parent.mkdir(parents=True,exist_ok=True);dest.symlink_to(source.OUT/'banks/validation'/str(budget)/task/block,target_is_directory=True)


if __name__=='__main__':diagnostic()
