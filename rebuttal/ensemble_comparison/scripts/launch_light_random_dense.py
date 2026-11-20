import os
import subprocess
import sys
from prepare_data import RUN

jobs=[]
for shard in range(16):
    env=os.environ | {'CUDA_VISIBLE_DEVICES':str(shard%8),'OMP_NUM_THREADS':'1',
        'MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
    stream=(RUN/'logs'/f'light_random_dense_{shard}.log').open('a')
    process=subprocess.Popen([sys.executable,str(RUN/'scripts/run_light_random_dense.py'),
        '--shard',str(shard),'--shards','16'],env=env,stdout=stream,stderr=subprocess.STDOUT)
    jobs.append((shard,process,stream))
    print('LAUNCHED',shard,process.pid,flush=True)
failed=[]
for shard,process,stream in jobs:
    code=process.wait();stream.close();print('FINISHED',shard,code,flush=True)
    if code:failed.append(shard)
if failed:raise SystemExit('Failed shards '+str(failed))
