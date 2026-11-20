from pathlib import Path
from collections import Counter
import subprocess,os,json,time
root=Path('/mnt/public/zhouyibo/ICLR2027/rebuttal/ensemble_comparison')
r=root/'results/few_reference_contexts_v26'
remaining={'regression':(68,0),'classification':(107,1)}
while remaining:
 counts=Counter(p.parts[-4] for p in r.glob('jobs/*/*/*/inference_complete.json'))
 for task,(total,gpu) in list(remaining.items()):
  marker=r/f'fit_{task}_launched.json'
  if marker.exists(): del remaining[task]; continue
  if counts[task]==total:
   workers=[]
   for rawscale,slot in ([(False,gpu),(True,2)] if task=='regression' else [(False,gpu)]):
    tag=task+('_rawscale' if rawscale else '')
    with (r/'logs'/f'fit_{tag}.log').open('a') as out:
     args=['/mnt/public/zhouyibo/TabSwift/.venv/bin/python','-u',str(root/'scripts/few_reference_contexts_v26.py'),'fit','--task',task]+(['--raw-scale'] if rawscale else [])
     p=subprocess.Popen(args,env=os.environ|{'CUDA_VISIBLE_DEVICES':str(slot)},stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
    workers.append({'pid':p.pid,'gpu':slot,'task':task,'raw_scale':rawscale});print(workers[-1],flush=True)
   marker.write_text(json.dumps(workers));del remaining[task]
 if remaining:time.sleep(5)
