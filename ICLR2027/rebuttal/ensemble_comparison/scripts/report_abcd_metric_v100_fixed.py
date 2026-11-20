"""Reporting-only fix: use the actual seven_reimplemented track identifier."""
import json
import shutil
import abcd_metric_aligned_v100 as study
from prepare_data import atomic_json

if __name__=='__main__':
    backup=study.OUT/'initial_classification_only_report';backup.mkdir(exist_ok=True)
    for name in ['development_tables.csv','development_contrasts.csv','development_status.json']:
        p=study.OUT/name
        if p.exists() and not (backup/name).exists():shutil.copy2(p,backup/name)
    original=study.regression_cells
    def load(matched):
        cells=original(matched)
        for c in cells:
            if c['meta']['track']==study.origin.TARGETS['regression'][0]:c['meta']['track']='seven'
        return cells
    study.regression_cells=load
    atomic_json(study.OUT/'reporting_amendment.json',{'reason':'literal seven did not match seven_reimplemented; only reporting filter corrected',
        'model_fitting_and_data_unchanged':True,'reporter_sha256':study.origin.sha(__file__)})
    study.report()
