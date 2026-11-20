"""Wait for scheduled work, require complete audits, then publish one report."""
import json
import os
from pathlib import Path
import subprocess
import time
RUN=Path(__file__).resolve().parents[1]
state={'status':'WAITING_FOR_INFERENCE_AND_CALIBRATION','stages':[]}
def save():
    p=RUN/'results/finalization.json';t=p.with_suffix('.tmp.json');t.write_text(json.dumps(state,indent=2));t.replace(p)
save()
while True:
    paths=[RUN/'results'/f'continue_corrected_{b}.json' for b in ['tabicl','tabpfn']]
    states=[json.loads(p.read_text())['status'] if p.exists() else 'WAITING_FOR_RAW_BANK' for p in paths]
    if 'FAILED' in states:state['status']='FAILED_UPSTREAM';save();raise SystemExit(1)
    legacy=list((RUN/'results/combiners/five').rglob('fold_*_tabace_specific_*_reimplemented.json'))
    if states==['COMPLETE','COMPLETE'] and len(legacy)==100:break
    time.sleep(10)
state['status']='AUDITING';save()
for script,extra in [('verify_raw_results.py',['--require-complete']),('audit_oof_results.py',[]),('report.py',[])]:
    with (RUN/'logs'/('final_'+script+'.log')).open('w',buffering=1) as stream:
        command=[str(RUN/'.venv/bin/python'),'-u',str(RUN/'scripts'/script)]+extra
        row={'script':script,'status':'RUNNING'};state['stages'].append(row);save()
        process=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,env=os.environ.copy())
        while process.poll() is None:time.sleep(5)
        row.update(exit_code=process.returncode,status='COMPLETE' if process.returncode==0 else 'FAILED');save()
        if process.returncode:state['status']='FAILED_AUDIT';save();raise SystemExit(process.returncode)
    print('FINAL_STAGE_COMPLETE',script,flush=True)
state['status']='COMPLETE_CURRENT_PROTOCOL';save();print('FINALIZATION_COMPLETE',flush=True)
