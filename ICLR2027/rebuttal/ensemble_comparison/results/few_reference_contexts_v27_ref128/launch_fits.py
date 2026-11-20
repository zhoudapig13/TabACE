from pathlib import Path
import subprocess,os,json,time
root=Path('/mnt/public/zhouyibo/ICLR2027/rebuttal/ensemble_comparison');r=root/'results/few_reference_contexts_v27_ref128'
remaining={'regression':(68,3),'classification':(107,7)}
while remaining:
 for task,(total,gpu) in list(remaining.items()):
  if len(list((r/'jobs'/task).glob('*/*/inference_complete.json')))==total:
   with (r/'logs'/f'fit_{task}.log').open('a') as out:
    p=subprocess.Popen(['/mnt/public/zhouyibo/TabSwift/.venv/bin/python','-u',str(root/'scripts/few_reference_contexts_v26.py'),'fit','--task',task],env=os.environ|{'CUDA_VISIBLE_DEVICES':str(gpu),'TABACE_REFERENCE_BUDGET':'128'},stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
   d={'pid':p.pid,'gpu':gpu,'task':task};(r/f'fit_{task}_launched.json').write_text(json.dumps(d));print(d,flush=True);del remaining[task]
 if remaining:time.sleep(5)
