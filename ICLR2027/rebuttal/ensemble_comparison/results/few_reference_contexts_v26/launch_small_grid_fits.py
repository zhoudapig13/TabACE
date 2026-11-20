from pathlib import Path
from collections import Counter
import subprocess,os,json,time
root=Path('/mnt/public/zhouyibo/ICLR2027/rebuttal/ensemble_comparison')
remaining={(budget,task):(total,gpu) for budget,offset in [(16,0),(32,4)] for task,total,gpu in [('regression',68,offset),('classification',107,offset+1)]}
while remaining:
 for (budget,task),(total,gpu) in list(remaining.items()):
  r=root/f'results/few_reference_contexts_v27_ref{budget}'
  marker=r/f'fit_{task}_launched.json'
  if marker.exists():del remaining[(budget,task)];continue
  count=len(list((r/'jobs'/task).glob('*/*/inference_complete.json')))
  if count==total:
   with (r/'logs'/f'fit_{task}.log').open('a') as out:
    p=subprocess.Popen(['/mnt/public/zhouyibo/TabSwift/.venv/bin/python','-u',str(root/'scripts/few_reference_contexts_v26.py'),'fit','--task',task],env=os.environ|{'CUDA_VISIBLE_DEVICES':str(gpu),'TABACE_REFERENCE_BUDGET':str(budget)},stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
   d={'pid':p.pid,'gpu':gpu,'task':task,'reference_budget':budget};marker.write_text(json.dumps(d));print(d,flush=True);del remaining[(budget,task)]
 if remaining:time.sleep(5)
