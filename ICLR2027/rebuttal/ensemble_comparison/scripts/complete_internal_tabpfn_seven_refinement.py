"""Resume after bounded family calibration; preserve valid five-condition banks."""
import subprocess
from launch_internal_tabpfn_regression import RUN,PY,stage

if __name__=='__main__':
    script=str(RUN/'scripts/refine_internal_tabpfn_seven.py')
    stage('seven_label_probes',[[script,'run','--shard',str(i),'--shards','8'] for i in range(8)],True)
    stage('seven_family_select',[[script,'select']])
    stage('seven_family_materialize',[[script,'materialize']])
    subprocess.run([str(PY),'-u',str(RUN/'scripts/launch_internal_tabpfn_regression.py')],cwd=RUN,check=True)
