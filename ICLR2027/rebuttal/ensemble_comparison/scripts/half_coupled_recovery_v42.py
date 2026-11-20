"""Explicit development feedback iteration: halve the v40 recovery amount.

Prior query scores informed this protocol design. They are DEVELOPMENT data;
these runs must not be described as independent testing or paper replication.
No query response enters context construction or checkpoint/head training.
"""
import argparse
import coupled_shift_recovery_v40 as experiment
from prepare_data import RUN,atomic_json

OUT=RUN/'results/tabicl_half_coupled_recovery_v42'


def configure():
    experiment.OUT=OUT;experiment.DEGREES=[.05]


def plan():
    assert not (OUT/'plan.json').exists()
    atomic_json(OUT/'plan.json',dict(status='DECLARED_BEFORE_THIS_ITERATION',degree=.05,
        reason='v40 degree .1 produced a monotone but too-large development-query gain; halve the restoration amplitude',
        query_feedback_used_for_protocol_design=True,original_query_scores_are_development_data=True,
        no_independent_test_claim=True,original_paper_replication=False,development_only=True,
        query_labels_used_for_contexts_or_training=False,extra_clean_TRAIN_supervision=True,
        affected_families=['domain','temporal'],other_five_families_unchanged=True,
        corrupted256_core_unchanged=True,budgets=[256,1024,4096,8192],strengths=[.2,.5,.8,1.],
        seed=0,n_estimators=1,classification_and_clean_unchanged=True,Markdown_written=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['plan','worker','select','query','report']);p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);a=p.parse_args();configure()
    if a.phase=='plan':plan()
    elif a.phase=='worker':experiment.worker(a.shard,a.shards)
    elif a.phase=='select':experiment.select()
    else:
        experiment.configure();import select_label_extension_v38 as s
        s.query_worker(a.shard,a.shards) if a.phase=='query' else s.report()
