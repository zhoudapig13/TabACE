"""Choose FT checkpoints on a COMMON source validation split, before reporting.

No benchmark predictions are opened until all 72 source-only checkpoint choices
are frozen. Preserve both candidates and include their full training costs.
"""
import json
from pathlib import Path
import shutil
import time
import os
import numpy as np
import finetune_source_tabace_v5 as source
import finetune_source_strong_v8 as strong
from prepare_data import RUN,atomic_json

OUT=RUN/'results/finetune_source_selected_v9'
REF_OUT=RUN/'results/finetune_source_selected_reference_v10'


def select():
    OUT.mkdir(parents=True,exist_ok=True)
    choices=[]
    # Stage one opens training metadata and original-row split arrays only.
    for job in source.jobs():
        relative=Path('jobs')/job['task']/job['track']/str(job['meta']['task_id'])
        paths=[source.OUT/relative,strong.OUT/relative]
        assert all((p/'completion.json').exists() for p in paths)
        with np.load(paths[0]/'split.npz') as a,np.load(paths[1]/'split.npz') as b:
            assert set(a.files)==set(b.files) and all(np.array_equal(a[k],b[k]) for k in a.files)
        records=[json.loads((p/'training_result.json').read_text()) for p in paths]
        losses=[r['selected_checkpoint']['source_validation_loss'] for r in records]
        chosen=int(losses[1]<losses[0]-1e-9)
        choices.append(dict(task_id=job['meta']['task_id'],dataset=job['meta']['dataset'],relative=str(relative),
            chosen_parent=str(paths[chosen]),chosen_candidate=['regularized128','strong512'][chosen],
            common_source_validation_losses=losses,total_candidate_training_seconds=sum(r['training_seconds'] for r in records),
            selected_training_seconds=records[chosen]['training_seconds'],
            checkpoint_sha256=records[chosen]['checkpoint_file_sha256'],all_candidates_actual_weight_updates=True,
            selection_reads_query_predictions_or_metrics=False))
    atomic_json(OUT/'selection.json',dict(status='ALL_CHECKPOINT_CHOICES_FROZEN_BEFORE_COPYING_QUERY_PREDICTIONS',
        choices=choices,counts={name:sum(c['chosen_candidate']==name for c in choices) for name in ['regularized128','strong512']},
        final_query_labels_or_scores_used_for_selection=False,common_train_and_validation_partitions_verified=True,
        time_UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
    # Stage two only copies evidence from the already selected real checkpoints.
    for c in choices:
        target=OUT/c['relative']
        if not target.exists():shutil.copytree(c['chosen_parent'],target)
        atomic_json(target/'selection_provenance.json',c)
        if c['chosen_candidate']=='regularized128':
            previous=RUN/'results/finetune_source_reference_v6'/c['relative']
            if (previous/'completion.json').exists():
                dest=REF_OUT/c['relative']
                if not dest.exists():shutil.copytree(previous,dest)
                atomic_json(dest/'reused_evidence.json',dict(reason='identical selected real checkpoint, exact source/calibration/query inputs',
                    previous=str(previous),checkpoint_sha256=c['checkpoint_sha256'],query_metrics_used_for_selection=False))
    atomic_json(OUT/'plan.json',dict(status='COMPLETE_CHECKPOINT_SELECTION',datasets=72,seed=0,context_cap=256,
        source_context_only=True,query_unchanged=True,benchmark_sha256=source.main.sha(source.benchmark.OUT/'frozen.json'),
        selection_file='selection.json',candidate_budget=2,all_training_costs_included=True,development_only=True,Markdown_written=False))
    atomic_json(OUT/'completion.json',dict(status='COMPLETE',datasets=72,final_query_labels_for_selection=False))
    print('SOURCE_CHECKPOINT_SELECTION_COMPLETE',{name:sum(c['chosen_candidate']==name for c in choices) for name in ['regularized128','strong512']},flush=True)


if __name__=='__main__':
    parent=strong.OUT
    while True:
        if (parent/'workers.json').exists():
            waiting=json.loads((parent/'workers.json').read_text())
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
