"""Reconstruct every new latent context using observed source labels only."""
import json
import os

for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(key, '1')

import numpy as np

import shared_supervised_projection_banks_v215 as bank
from run_tabicl import CHECKPOINTS


def main():
    assert json.loads((bank.OUT/'execution.json').read_text())['status'] == 'COMPLETE'
    plan = json.loads((bank.OUT/'plan.json').read_text())
    for path, key in [(bank.__file__, 'script_sha256'),
        (bank.constructor.__file__, 'constructor_sha256'),
        (bank.constructor.encode.__code__.co_filename, 'encoding_implementation_sha256'),
        (bank.parent.OUT/'plan.json', 'parent_bank_plan_sha256'),
        (bank.source.OUT/'plan.json', 'original_plan_sha256')]:
        assert bank.source.sha(path) == plan[key]
    original_plan = json.loads((bank.source.OUT/'plan.json').read_text())
    for path, digest in original_plan['checkpoints'].items():
        assert bank.source.sha(CHECKPOINTS/path) == digest
    cases = json.loads((bank.source.OUT/'case_manifest.json').read_text())
    count, contexts, warnings, fallback = 0, 0, 0, 0
    for c in cases:
        if c['track'] != 'perturbation':
            continue
        assert bank.source.sha(c['input_file']) == c['input_sha256']
        with np.load(c['input_file']) as data:
            assert 'y_target' not in data.files
            X0, y0, Q0, numeric = (data[k] for k in ('X_source', 'y_source', 'X_target', 'numeric_columns'))
        for family, rho in [('clean', 0)]+[(f, r) for f in c['families'] for r in bank.source.novel.STRENGTHS]:
            path = bank.OUT/'banks'/c['task']/str(c['task_id'])/f'{family}_{rho}.npz'
            meta = json.loads(path.with_suffix('.json').read_text())
            assert bank.source.sha(path) == meta['output_sha256']
            assert bank.source.sha(meta['parent_bank']) == meta['parent_bank_sha256']
            assert meta['original_input_sha256'] == c['input_sha256']
            assert meta['new_TFM_calls'] == 2 and not meta['final_query_outcome_files_opened']
            with np.load(path) as data, np.load(meta['parent_bank']) as parent:
                np.testing.assert_array_equal(data['Base'], parent['Base'])
                np.testing.assert_array_equal(data['members'][:3], parent['members'][:3])
                np.testing.assert_array_equal(data['Canonical_native8'], parent['Canonical_native8'])
                assert data['members'].shape[0] == 5 and np.isfinite(data['members']).all()
                if c['task'] == 'classification':
                    np.testing.assert_allclose(data['members'].sum(axis=2), 1., atol=1e-6)
            if rho:
                X, y, Q, _ = bank.source.novel.generate(X0, y0, Q0, numeric, c['task'], family, rho, c['task_id'])
            else:
                X, y, Q = X0, y0, Q0
            np.testing.assert_array_equal(y, y0)
            assert bank.protocol.array_sha256(y) == meta['source_labels_sha256']
            for context, record in zip(bank.constructor.construct(X, y, Q, numeric, c['task']), meta['supervised_contexts']):
                assert context['info'] == {k: record[k] for k in context['info']}
                for key, digest in [('X', 'context_X_sha256'), ('y', 'context_y_sha256'), ('Q', 'projected_query_X_sha256')]:
                    assert bank.protocol.array_sha256(context[key]) == record[digest]
                np.testing.assert_array_equal(context['y'], y0)
                assert len(context['y']) == c['source_rows'] <= 256
                assert not record['query_labels_used'] and not record['source_labels_modified']
                contexts += 1
                warnings += bool(record['fit_warnings'])
                fallback += record['fallback'] is not None
            count += 1
    assert count == 1327 and contexts == 2654
    assert (bank.source.ROOT/'REBUTTAL_EXPERIMENTS.md').stat().st_size == 0
    result = {'status': 'PASS', 'conditions': count, 'latent_contexts_reconstructed': contexts,
        'contexts_with_recorded_PLS_warnings': warnings, 'contexts_using_declared_fallback': fallback,
        'Base_and_first_three_alternatives_bitwise_unchanged': True,
        'same_source_rows_and_labels': True, 'extra_labeled_rows': 0,
        'query_outcomes_opened': False, 'checkpoint_hashes_unchanged': True,
        'constructor_extension_not_original_Shared': True, 'Markdown_bytes': 0}
    bank.atomic_json(bank.OUT/'independent_membership_audit.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
