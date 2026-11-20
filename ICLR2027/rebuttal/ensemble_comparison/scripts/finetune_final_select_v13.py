"""Freeze the strongest source-validation checkpoint in all 175 task tracks."""
import json
import os
from pathlib import Path
import shutil
import time
import numpy as np
import finetune_tabace_main_v1 as main
from prepare_data import RUN,atomic_json

OUT=RUN/'results/finetune_final_selected_v13'
REF=RUN/'results/finetune_final_reference_v14'
STRONG=RUN/'results/finetune_remaining_strong_v12'


def candidates(job):
    if job['task']=='classification' and job['track']=='five':return [RUN/'results/finetune_source_selected_v9']
    return [RUN/('results/finetune_tabace_main_v2' if job['task']=='classification' else 'results/finetune_tabace_main_v1'),STRONG]


def select():
    OUT.mkdir(parents=True,exist_ok=True);choices=[]
    for job in main.jobs():
        rel=Path('jobs')/job['task']/job['track']/str(job['meta']['task_id'])
        paths=[r/rel for r in candidates(job)]
        assert all((p/'completion.json').exists() for p in paths),paths
        records=[json.loads((p/'training_result.json').read_text()) for p in paths]
        if len(paths)==2:
            with np.load(paths[0]/'split.npz') as a,np.load(paths[1]/'split.npz') as b:
                assert set(a.files)==set(b.files) and all(np.array_equal(a[k],b[k]) for k in a.files),(job,'split mismatch')
        losses=[r['selected_checkpoint']['source_validation_loss'] for r in records]
        selected=int(len(paths)>1 and losses[1]<losses[0]-1e-9)
        train_seconds=sum(r['training_seconds'] for r in records)
        if len(paths)==1:
            provenance=json.loads((paths[0]/'selection_provenance.json').read_text());train_seconds=provenance['total_candidate_training_seconds']
        choices.append(dict(task=job['task'],track=job['track'],task_id=job['meta']['task_id'],dataset=job['meta']['dataset'],
            relative=str(rel),candidate_parents=[str(p) for p in paths],source_validation_losses=losses,
            selected_parent=str(paths[selected]),selected_index=selected,total_candidate_training_seconds=train_seconds,
            selected_training_seconds=records[selected]['training_seconds'],checkpoint_sha256=records[selected]['checkpoint_file_sha256']))
    atomic_json(OUT/'selection.json',dict(status='ALL_CHECKPOINTS_FROZEN_BEFORE_COPYING_QUERY_EVIDENCE',choices=choices,
        selection_source_validation_only=True,final_query_labels_or_scores_for_selection=False,
        stronger_candidates_selected=sum(c['selected_index']==1 for c in choices),
        classification_five_preselected_on_source_validation=True,
        time_UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
    for c in choices:
        dest=OUT/c['relative']
        if not dest.exists():shutil.copytree(c['selected_parent'],dest)
        atomic_json(dest/'final_selection_provenance.json',c)
        selected=Path(c['selected_parent']);possible=[]
        if c['task']=='classification' and c['track']=='five':possible=[RUN/'results/finetune_source_selected_reference_v10'/c['relative']]
        elif c['selected_index']==0:possible=[RUN/'results/finetune_tabace_reference_v3'/c['relative']]
        for previous in possible:
            if (previous/'completion.json').exists():
                target=REF/c['relative']
                if not target.exists():shutil.copytree(previous,target)
                atomic_json(target/'reused_evidence.json',dict(previous=str(previous),selected_checkpoint_sha256=c['checkpoint_sha256'],
                    identical_finetuned_weights_and_inputs=True,query_scores_used_for_selection=False))
    atomic_json(OUT/'plan.json',dict(status='FROZEN_DEVELOPMENT_COMPARISON',seed=0,context_cap=256,
        classification_clean_five_datasets=72,classification_seven_datasets=35,regression_clean_five_datasets=35,regression_seven_datasets=33,
        classification_five_source_context_corruption_only=True,classification_five_query_unchanged=True,
        other_blocks='v22 frozen development suite',regression_clean_five_semisynthetic=True,
        source_only_classification_benchmark_sha256=main.sha(RUN/'results/finetune_source_benchmark_v5/frozen.json'),
        selection='best actual weight-update checkpoint on identical source TRAIN validation; all candidate costs retained',
        Shared_Specific_unfinetuned_not_run=True,paper_Shared_Specific='historical references only',
        Markdown_written=False))
    atomic_json(OUT/'completion.json',dict(status='COMPLETE',jobs=175,source_validation_selection=True,query_labels_used_for_selection=False))
    print('FINAL_CHECKPOINT_SELECTION_COMPLETE',len(choices),sum(c['selected_index']==1 for c in choices),'stronger candidates',flush=True)


if __name__=='__main__':
    while True:
        if (STRONG/'workers.json').exists():
            waiting=json.loads((STRONG/'workers.json').read_text())
            if len(waiting)==8:break
        time.sleep(2)
    while waiting:
        rest=[]
        for w in waiting:
            try:os.kill(w['pid'],0);rest.append(w)
            except ProcessLookupError:pass
        waiting=rest
        if waiting:time.sleep(2)
    select()
