"""Keep every declared clinical protocol in one aggregate table, without Markdown."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from run_public_shift import ROOT, array_hash, write_json
from summarize_public_shift import holm


def main():
    rows, all_tests = [], []
    protocols = ['public_shift_v1', 'public_casecontrol_v1', 'public_casecontrol_crossfit_v1']
    for protocol in protocols:
        tests = []
        location = ROOT/'results'/protocol
        for metrics_file in sorted(location.glob('*/metrics.json')):
            result = json.loads(metrics_file.read_text())
            task = result['task']
            frozen = json.loads((metrics_file.parent/'frozen_policies.json').read_text())
            assert not frozen['target_labels_used_for_fitting']
            for domain, values in result['domains'].items():
                for method, score in values['methods'].items():
                    rows.append(dict(protocol=protocol, task=task, domain=domain, method=method,
                                     n=values['n'], positives=values['positives'], **score))
                if domain == 'ood':
                    for name, test in values['adjacent_comparisons'].items():
                        tests.append((task+'/'+name, test.copy()))
        holm(tests)
        write_json(location/'paired_primary_tests.json', dict(tests))
        all_tests.extend((protocol+'/'+name, test.copy()) for name, test in tests)
    # Additional conservative correction across ALL primary OOD comparisons
    # reported during this task; do not select the best protocol by test result.
    holm(all_tests)
    write_json(ROOT/'results/medical_shift_all_primary_tests.json', dict(all_tests))
    pd.DataFrame(rows).to_csv(ROOT/'results/medical_shift_all_protocols.csv', index=False)
    # Audit hashes, disjointness, balance and prediction array dimensions.
    checks = []
    for protocol in ['public_shift_v1', 'public_casecontrol_v1']:
        for meta_path in sorted((ROOT/'data/processed'/protocol).glob('*/metadata.json')):
            meta = json.loads(meta_path.read_text())
            with np.load(meta_path.parent/'inputs.npz') as z:
                assert {k: array_hash(z[k]) for k in z.files} == meta['input_hashes']
            member = json.loads((meta_path.parent/'split_membership_hashes.json').read_text())
            keys = list(member)
            for i, a in enumerate(keys):
                assert len(member[a]) == len(set(member[a]))
                for b in keys[i+1:]:
                    assert not set(member[a]).intersection(member[b])
            with np.load(meta_path.parent/'member_predictions.npz') as z:
                assert np.all(z['context_rows'] <= 256)
                assert np.isfinite(z['bank_ood']).all()
            with np.load(meta_path.parent/'test_truth.npz') as z:
                if protocol == 'public_casecontrol_v1':
                    assert all(int(z[k].sum())*2 == len(z[k]) for k in z.files)
            checks.append(dict(protocol=protocol, task=meta['task'], passed=True))
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in (ROOT/'scripts').glob('*.py')}
    write_json(ROOT/'results/medical_shift_verification.json', dict(
        verified_at_utc=datetime.now(timezone.utc).isoformat(), checks=checks,
        all_primary_comparisons=len(all_tests), script_hashes=hashes,
        single_seed=0, paper_exact_replication=False, target_label_tuning=False,
        method='Medical-task reconstruction/transfer; frozen TabICL-v2',
        every_attempt_retained=True, markdown_written=False))
    table = pd.DataFrame(rows)
    print(table[table.domain == 'ood'].pivot(index=['protocol', 'task', 'n'], columns='method', values='accuracy').to_string())
    print('All data-integrity checks passed; no patient-level records in aggregate table.')


if __name__ == '__main__':
    main()
