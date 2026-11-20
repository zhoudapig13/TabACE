"""Refresh the single experiment document and resume failed seven-shift cells."""
from __future__ import annotations
import json
from pathlib import Path
import subprocess
import sys
import time
RUN=Path(__file__).resolve().parents[1]


def report():
    subprocess.run([str(RUN/'.venv/bin/python'),str(RUN/'scripts/report.py')],check=True)


def main():
    status_path=RUN/'results/seven_tabpfn_workers.json'
    while True:
        report()
        status=json.loads(status_path.read_text())
        live=[]
        for j in status:
            command=Path(f"/proc/{j['pid']}/cmdline")
            try:live.append('scripts/run_seven_tabpfn.py' in command.read_text())
            except FileNotFoundError:live.append(False)
        if not any(live):break
        time.sleep(60)
    # The initial workers loaded the pre-fix singleton holdout path. One bounded
    # resume evaluates only missing cells with the corrected explicit-N/A path.
    command=[sys.executable,str(RUN/'scripts/launch_seven_tabpfn.py')]
    result=subprocess.run(command,check=False)
    report()
    record={'status':'COMPLETE' if result.returncode==0 else 'INCOMPLETE_WITH_RECORDED_FAILURES',
            'repair_returncode':result.returncode,'original_main_table_comparison_ready':False,
            'reason':'Seven operators are reconstructed; five original operator mismatch remains unresolved. No matched TabACE run exists on the reconstructed seven protocol.'}
    (RUN/'results/seven_execution_status.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record),flush=True)


if __name__=='__main__':main()
