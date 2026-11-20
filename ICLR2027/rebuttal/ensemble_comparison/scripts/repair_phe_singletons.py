"""Complete PHE on singleton-class source contexts using disjoint safe holdout."""
import argparse
import json
import logging
from pathlib import Path
import shutil
import warnings
import numpy as np
import torch
from run_tabpfn import RUN,cell


def main():
    p=argparse.ArgumentParser();p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);args=p.parse_args()
    torch.set_num_threads(1);logging.disable(logging.INFO);warnings.filterwarnings('ignore')
    manifest=[]
    for source in (RUN/'results/tabpfn').rglob('*.json'):
        if '.tmp' in source.name:continue
        m=json.loads(source.read_text())
        if m.get('PHE_status')=='NOT_APPLICABLE_SINGLETON_SOURCE_CLASS':manifest.append(source)
    failures=0
    for path in sorted(manifest)[args.shard::args.shards]:
        m=json.loads(path.read_text());track=m['track'];task=m['task'];task_id=m['task_id']
        if track=='five':
            data_path=RUN/'cache/five'/task/f'{task_id}.npz';meta=json.loads(data_path.with_suffix('.json').read_text())
        else:
            data_path=RUN/'cache/seven_reimplemented'/task/str(task_id)/f"{m['family']}_{m['rho']}.npz"
            meta=json.loads(data_path.with_suffix('.json').read_text())
        archive=RUN/'results/invalidated_phe_singleton_na'/track/task/str(task_id)/path.name
        archive.parent.mkdir(parents=True,exist_ok=True)
        if not archive.exists():
            shutil.copy2(path,archive);shutil.copy2(path.with_suffix('.npz'),archive.with_suffix('.npz'))
        try:
            with np.load(data_path) as data:cell(task,data,meta,m['family'],m['rho'],path.with_suffix('.npz'),'cuda:0',True)
            j=json.loads(path.read_text());assert 'PHE_official_custom6_native_source_calibration' in j['methods']
            print('REPAIRED',track,task_id,m['family'],m['rho'],j['PHE_status'],flush=True)
        except Exception as exc:
            failures+=1;print('FAILED',str(path),repr(exc),flush=True)
    print('REPAIR_FINISHED',args.shard,'failures',failures,flush=True)
    if failures:raise SystemExit(1)


if __name__=='__main__':main()
