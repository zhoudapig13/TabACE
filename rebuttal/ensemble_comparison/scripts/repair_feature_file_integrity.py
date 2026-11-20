"""Validate feature-view archives; quarantine only incomplete/corrupt files."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import numpy as np
from prepare_data import RUN,atomic_json

root=RUN/'results/feature_views/tabicl/five_corrected/classification'
def check(path):
    try:
        meta=json.loads(path.with_suffix('.json').read_text())
        with np.load(path) as f:
            bank=f['bank'];mean=f['prediction__Feature_view_mean6_adapted'];y=f['y_target'];mu=f['source_mean'];scale=f['source_scale']
            assert bank.shape[0]==6 and bank.shape[1]==len(y)
            assert np.isfinite(bank).all() and np.isfinite(mu) and np.isfinite(scale)
            assert np.allclose(bank.sum(2),1,atol=1e-7)
            assert np.allclose(mean,bank.mean(0),atol=1e-12,rtol=1e-12)
        original=RUN/'results/tabicl/five_corrected/classification'/path.parent.name/path.name
        with np.load(original) as f:assert np.array_equal(bank[0],f['prediction__base_protocol_check'])
        return None
    except Exception as exc:return {'path':str(path.relative_to(RUN)),'error':repr(exc)}
paths=sorted(root.glob('*/*.npz'))
with ThreadPoolExecutor(max_workers=12) as pool:failures=[x for x in pool.map(check,paths) if x]
for failure in failures:
    path=RUN/failure['path'];out=RUN/'results/invalidated_concurrent_feature_files'/path.relative_to(RUN/'results')
    out.parent.mkdir(parents=True,exist_ok=True)
    for source in [path,path.with_suffix('.json')]:
        if source.exists():shutil.copy2(source,out if source==path else out.with_suffix('.json'));source.unlink()
atomic_json(RUN/'results/feature_file_integrity_audit.json',{'status':'PASS' if not failures else 'QUARANTINED_FOR_REPAIR','files_checked':len(paths),'failures':failures,'concurrent_stage_lock_now_enforced':True})
print('FEATURE_FILES',len(paths),'QUARANTINED',len(failures),flush=True)
