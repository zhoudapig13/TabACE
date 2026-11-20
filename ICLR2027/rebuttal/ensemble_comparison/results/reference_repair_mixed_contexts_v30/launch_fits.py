from pathlib import Path
import subprocess,os,json,time
root=Path('/mnt/public/zhouyibo/ICLR2027/rebuttal/ensemble_comparison');r=root/'results/reference_repair_mixed_contexts_v30'
remaining={'regression':(68,0),'classification':(107,1)}
while remaining:
 for task,(total,gpu) in list(remaining.items()):
  if len(list((r/'jobs'/task).glob('*/*/inference_complete.json')))==total:
   workers=[]
   for canonical,slot in [(False,gpu),(True,gpu+2)]:
    with (r/'logs'/(f'fit_{task}'+('_canonical128.log' if canonical else '.log'))).open('a') as out:
     args=['/mnt/public/zhouyibo/TabSwift/.venv/bin/python','-u',str(root/'scripts/reference_repair_contexts_v28.py'),'fit','--task',task]+(['--canonical128'] if canonical else [])
     p=subprocess.Popen(args,env=os.environ|{'CUDA_VISIBLE_DEVICES':str(slot),'TABACE_REFERENCE_BUDGET':'64','TABACE_REPAIR_MIX_RATE':'0.05'},stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
    workers.append({'pid':p.pid,'gpu':slot,'task':task,'canonical128':canonical});print(workers[-1],flush=True)
   (r/f'fit_{task}_launched.json').write_text(json.dumps(workers));del remaining[task]
 if remaining:time.sleep(5)
