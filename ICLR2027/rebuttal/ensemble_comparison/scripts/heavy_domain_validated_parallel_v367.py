"""Partition only the active endpoint across six CPU readout workers.

The v366 fitting code and data are unchanged. Each child receives a disjoint
dataset list and writes its own execution status; condition artifacts retain
the original v366 paths and provenance. No GPU is used by this scheduler.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
import subprocess
import sys
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(key,'1')
import heavy_domain_validated_probe_v366 as run
from prepare_data import atomic_json


def worker(shard,shards):
    original=run.features.schedule.endpoint.endpoint_cases
    selected=[c for i,c in enumerate(original()) if i%shards==shard]
    assert all((c['track'],c['task'])==('perturbation','classification') for c in selected)
    # Process-local scheduling injection; neither features nor fitting change.
    run.features.schedule.endpoint.endpoint_cases=lambda:selected
    original_write=run.atomic_json
    def scoped_write(path,value):
        if path==run.OUT/'execution.json':path=run.OUT/f'shard_{shard}_execution.json'
        return original_write(path,value)
    run.atomic_json=scoped_write
    run.main(pilot=False,require_complete=True)


def launch():
    run.prepare()
    assert json.loads((run.features.OUT/'full_execution.json').read_text())['status']=='COMPLETE'
    plan={'active_endpoint':['perturbation','classification','cell_heavy_tail'],'CPU_workers':6,'GPU_workers':0,
        'parallelism':'disjoint datasets within the one active endpoint only','fitting_implementation_unchanged':True,
        'case_partition':'original64 case order modulo6','fitter_sha256':run.source.sha(run.__file__),
        'driver_sha256':run.source.sha(__file__),'other_endpoints_not_started':True,'frozen_registry_sha256':run.features.schedule.verify_frozen()}
    p=run.OUT/'parallel_plan.json'
    if p.exists():assert json.loads(p.read_text())==plan
    else:atomic_json(p,plan)
    def execute(i):
        env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='';log=run.OUT/f'CPU_worker_{i}.log'
        with log.open('a') as f:p=subprocess.run([sys.executable,__file__,'worker','--shard',str(i),'--shards','6'],env=env,stdout=f,stderr=subprocess.STDOUT)
        return {'shard':i,'returncode':p.returncode,'log':str(log)}
    atomic_json(run.OUT/'execution.json',{'status':'RUNNING','active_endpoint':plan['active_endpoint'],'CPU_workers':6,'GPU_workers':0,'expected_conditions':256})
    with ThreadPoolExecutor(max_workers=6) as pool:results=list(pool.map(execute,range(6)))
    count=len(list((run.OUT/'predictions').glob('*/*.json')))
    ok=all(r['returncode']==0 for r in results) and count==256
    atomic_json(run.OUT/'execution.json',{'status':'COMPLETE' if ok else 'FAILED','conditions':count,'workers':results,
        'active_endpoint':plan['active_endpoint'],'CPU_workers':6,'GPU_workers':0})
    run.features.schedule.verify_frozen();assert ok


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['launch','worker']);p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=6);a=p.parse_args()
    launch() if a.phase=='launch' else worker(a.shard,a.shards)
