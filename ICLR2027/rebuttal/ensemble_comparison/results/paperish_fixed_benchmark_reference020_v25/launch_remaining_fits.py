from pathlib import Path
from collections import Counter
import subprocess,os,json,time
root=Path('/mnt/public/zhouyibo/ICLR2027/rebuttal/ensemble_comparison')
r=root/'results/paperish_fixed_benchmark_reference020_v25'
remaining={'regression':(68,2),'classification':(107,3)}
while remaining:
 counts=Counter(p.parts[-4] for p in r.glob('jobs/*/*/*/inference_complete.json'))
 for task,(total,gpu) in list(remaining.items()):
  marker=r/f'fit_{task}_launched.json'
  if marker.exists(): del remaining[task]; continue
  if counts[task]==total:
   log=r/'logs'/f'fit_reference020_{task}.log'
   with log.open('a') as out:
    p=subprocess.Popen(['/mnt/public/zhouyibo/TabSwift/.venv/bin/python','-u',str(root/'scripts/paperish_fixed_benchmark_reference020_v25.py'),'fit','--task',task,'--variant','reference020'],env=os.environ|{'CUDA_VISIBLE_DEVICES':str(gpu)},stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
   marker.write_text(json.dumps({'pid':p.pid,'gpu':gpu,'task':task}));print(task,p.pid,flush=True);del remaining[task]
 if remaining:time.sleep(5)
