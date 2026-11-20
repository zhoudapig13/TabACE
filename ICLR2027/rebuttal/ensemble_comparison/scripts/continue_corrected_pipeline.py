"""Complete corrected inference and held-out calibration, retaining all exits.

Each backbone waits for its own raw bank; GPU stages are sequential within a
backbone. All source-side choices are fixed before reading held-out scores.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
RUN=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--backbone',choices=['tabicl','tabpfn'],required=True);args=p.parse_args()
prefix='continue_corrected_'+args.backbone
history=RUN/'results'/f'{prefix}.json'
state=json.loads(history.read_text()) if history.exists() else {'backbone':args.backbone,'track':'five_corrected','stages':[]}
state['status']='RUNNING'
def save():
    path=RUN/'results'/f'{prefix}.json';tmp=path.with_suffix('.tmp.json');tmp.write_text(json.dumps(state,indent=2));tmp.replace(path)
def call(script,extra):
    if any(row['script']==script and row['args']==extra and row.get('status')=='COMPLETE' for row in state['stages']):
        print('STAGE_REUSED',args.backbone,script,extra,flush=True);return
    command=[str(RUN/'.venv/bin/python'),'-u',str(RUN/'scripts'/script)]+extra
    stream=(RUN/'logs'/(prefix+'_'+script+'.log')).open('a',buffering=1)
    process=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,env=os.environ.copy())
    row={'script':script,'args':extra,'pid':process.pid,'status':'RUNNING'};state['stages'].append(row);save()
    while process.poll() is None:time.sleep(5)
    stream.close();row.update(exit_code=process.returncode,status='COMPLETE' if process.returncode==0 else 'FAILED');save()
    if process.returncode:state['status']='FAILED';save();raise SystemExit(process.returncode)
    print('STAGE_COMPLETE',args.backbone,script,extra,flush=True)
root=RUN/'results'/args.backbone/'five_corrected/classification'
save()
while len(list(root.glob('*/*.json')))<1512:time.sleep(5)
if args.backbone=='tabicl':
    call('fit_combiners.py',['--track','five_corrected'])
    call('fit_convex_combiners.py',['--track','five_corrected'])
    call('dense_full_context.py',['--track','five_corrected'])
call('launch_corrected_stage.py',['--stage','reference','--backbone',args.backbone])
call('fit_shared_reference.py',['--track','five_corrected','--backbone',args.backbone])
call('launch_corrected_stage.py',['--stage','specific','--backbone',args.backbone])
call('fit_specific_reference.py',['--track','five_corrected','--backbone',args.backbone])
if args.backbone=='tabicl':
    call('launch_corrected_stage.py',['--stage','features','--backbone',args.backbone])
    call('fit_feature_nnls.py',['--track','five_corrected'])
state['status']='COMPLETE';save();print('PIPELINE_COMPLETE',args.backbone,flush=True)
