"""Restore the registered TRAIN-selected Clean generic role before head fitting."""
import argparse
import json
import hashlib
import calibration_head_v45 as head
from prepare_data import RUN,atomic_json

OUT=RUN/'results/tabace_clean_role_head_v46'
original_load=head.risk.load


def load(source,role,budget,block):
    cells=original_load(source,role,budget,block)
    if block!='clean':return cells
    choice=head.source.full.OUT/'policies'/str(budget)/'regression/clean/generic_scale_selection.json'
    selected=json.loads(choice.read_text());assert selected['query_labels_used'] is False
    order=selected['permutation'];assert sorted(order)==list(range(21))
    for c in cells:
        c['bank']=c['bank'][order]
        names=c['meta']['shared_roles']+c['meta']['specific_additional_roles'];names=[names[j] for j in order]
        c['meta']=c['meta']|dict(shared_roles=names[:10],specific_additional_roles=names[10:],role_permutation=order)
    path=OUT/'policies'/str(budget)/'regression/clean/role_permutation.json'
    if role=='validation':
        atomic_json(path,dict(status='FROZEN_BEFORE_QUERY',source=str(choice),source_sha256=hashlib.sha256(choice.read_bytes()).hexdigest(),
            permutation=order,selection='previous immutable inner-TRAIN dataset selection; no query scores',query_labels_used=False))
    else:assert json.loads(path.read_text())['permutation']==order
    return cells


def plan():
    assert not (OUT/'plan.json').exists()
    atomic_json(OUT/'plan.json',dict(status='DECLARED_BEFORE_CORRECTED_FITS',full_goal_checks=48,
        change='Apply the registered Clean generic-scale permutation before calibration-head fitting; v45 had used raw role order',
        unchanged=['Base inputs/predictions','TFM banks','row budgets','TRAIN calibration rows','head candidate grid'],
        refit='regression Clean at all four budgets',reuse='v45 regression five/seven and v44 classification',
        no_additional_TFM_calls=True,query_labels_used_for_role_choice=False,
        extra_clean_TRAIN_supervision=True,adaptation_marker='dagger',development_only=True,
        original_paper_replication=False,independent_test_claim=False,Markdown_written=False))
    (OUT/'paper_targets.json').write_text((head.OUT/'paper_targets.json').read_text())


def fit(budget):
    head.OUT=OUT;head.risk.load=load;head.run(budget,'clean')
    role=json.loads((OUT/'policies'/str(budget)/'regression/clean/role_permutation.json').read_text())
    for path in (OUT/'predictions'/str(budget)/'regression/clean').glob('*/*.json'):
        if '.tmp.' in path.name:continue
        row=json.loads(path.read_text());row.update(adaptation_version='v46',role_permutation=role['permutation'],role_selection_source=role['source']);atomic_json(path,row)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['plan','fit']);p.add_argument('--budget',type=int);a=p.parse_args();plan() if a.phase=='plan' else fit(a.budget)
