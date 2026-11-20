"""Finalize revised TabM, reusing the unchanged, complete OOF audit."""
import json,os,subprocess,time
from pathlib import Path
RUN=Path(__file__).resolve().parents[1]
state={'status':'WAITING_FOR_TABM_SCALER_V2','stages':[]}
def save():
    p=RUN/'results/finalization.json';t=p.with_suffix('.tmp.json');t.write_text(json.dumps(state,indent=2));t.replace(p)
save()
while True:
    p=RUN/'results/tabm_scaler_v2_workers.json'
    workers=json.loads(p.read_text()) if p.exists() else []
    if any(w['status']=='FAILED' for w in workers):state['status']='FAILED_TABM_WORKER';save();raise SystemExit(1)
    if len(workers)==8 and all(w['status']=='COMPLETE' for w in workers):break
    time.sleep(5)
audit=RUN/'results/oof_result_audit.json';previous=json.loads(audit.read_text())
files=list((RUN/'results/combiners').rglob('fold_*.json'))
assert previous['status']=='PASS' and len(files)==previous['files_checked'] and all(p.stat().st_mtime<=audit.stat().st_mtime for p in files)
state['unchanged_oof_audit_reused']=True;state['status']='AUDITING_SCALER_V2';save()
for script,extra in [('verify_raw_results.py',['--require-complete']),('report.py',[])]:
    with (RUN/'logs'/('scaler_v2_final_'+script+'.log')).open('w',buffering=1) as stream:
        row={'script':script,'status':'RUNNING'};state['stages'].append(row);save()
        process=subprocess.Popen([str(RUN/'.venv/bin/python'),'-u',str(RUN/'scripts'/script)]+extra,stdout=stream,stderr=subprocess.STDOUT,env=os.environ.copy())
        while process.poll() is None:time.sleep(5)
        row.update(exit_code=process.returncode,status='COMPLETE' if process.returncode==0 else 'FAILED');save()
        if process.returncode:state['status']='FAILED_AUDIT';save();raise SystemExit(process.returncode)
    print('FINAL_STAGE_COMPLETE',script,flush=True)
state['status']='COMPLETE_CURRENT_PROTOCOL';save();print('SCALER_V2_FINALIZATION_COMPLETE',flush=True)
