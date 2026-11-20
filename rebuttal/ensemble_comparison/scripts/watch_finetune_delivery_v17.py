"""Wait non-interactively for authorized experiment dependencies, then audit."""
import json
from pathlib import Path
import subprocess
import sys
import time

RUN=Path('/mnt/public/zhouyibo/ICLR2027/rebuttal/ensemble_comparison')
OUT=RUN/'results/finetune_delivery_v17'
OUT.mkdir(parents=True,exist_ok=True)

while True:
    counts={}
    for name in ['finetune_final_selected_v13','finetune_final_reference_v14','finetune_final_hierarchical_v15','finetune_calibration_control_v18']:
        counts[name]=len(list((RUN/'results'/name/'jobs').glob('*/*/*/completion.json')))
    counts['finetune_final_profile_v16']=len(list((RUN/'results/finetune_final_profile_v16/profiles').glob('*/*/*.json')))
    (OUT/'live_progress.json').write_text(json.dumps(dict(counts=counts,time_UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())),indent=2)+'\n')
    if all(n==175 for n in counts.values()):break
    time.sleep(3)

with (OUT/'collection.log').open('w',buffering=1) as log:
    proc=subprocess.run([sys.executable,'-u',str(RUN/'scripts/finetune_delivery_v17.py'),'collect'],stdout=log,stderr=subprocess.STDOUT)
(OUT/'collector_exit.json').write_text(json.dumps(dict(exit_code=proc.returncode,completed_UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())),indent=2)+'\n')
print('FINAL_DELIVERY_COLLECTOR_EXIT',proc.returncode,flush=True)
