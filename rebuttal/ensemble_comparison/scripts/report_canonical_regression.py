"""Fit dataset-OOF combiners and report the separate canonical-input control."""
from report_io import read_markdown,write_markdown
import argparse
import json
import math
from pathlib import Path
import numpy as np

from fit_combiners import fold_mapping
from fit_convex_combiners import design, fit_convex, fit_ges, score
from prepare_data import RUN, ROOT, atomic_json
from run_tabicl import metrics


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--regime',choices=['canonical_input','canonical_partial_010'],default='canonical_input')
    args=parser.parse_args();regime=args.regime
    root = RUN/'results'/regime/'tabicl/seven_calibrated_v2/regression'
    mapping, mapping_source = fold_mapping('regression', 'seven_calibrated_v2')
    cells, records = [], []
    paths = sorted(root.rglob('*.npz'))
    assert len(paths) == 33*29, len(paths)
    for path in paths:
        meta = json.loads(path.with_suffix('.json').read_text())
        assert meta['status'] == 'COMPLETE' and meta['context_rows'] == 256
        assert meta['candidate_count'] == 6 and not meta['target_labels_used_for_inference_or_selection']
        with np.load(path) as f:
            bank, y = f['bootstrap_bank'], f['y_target']
            assert bank.shape == (6, len(y)) and np.isfinite(bank).all()
            mu, sigma = float(f['source_y_mean']), float(f['source_y_std'])
            for method, value in meta['methods'].items():
                pred = f['prediction__'+method]
                fresh = metrics(y, pred, 'regression', max(float(y.std()), 1e-12))
                for k in fresh:
                    assert abs(fresh[k]-value[k]) < 1e-10
                records.append(meta | {'method': method} | fresh)
            selected = int(f['bootstrap_iqr'].mean(axis=1).argmin())
            assert np.array_equal(bank[selected], f['prediction__oligoicp_iqr_6_keep1_adapted'])
            assert np.array_equal(bank.mean(0), f['prediction__bootstrap_mean_6'])
        cells.append({'path': path, 'meta': meta, 'fold': mapping[meta['data_id']],
                      'bank': bank, 'y': y, 'mu': mu, 'sigma': sigma,
                      'clean_weight': 7. if meta['rho'] == 0 else 1.})
    for fold in range(5):
        out = RUN/'results'/regime/'combiners/seven_calibrated_v2/regression'/f'fold_{fold}.json'
        train = [c for c in cells if c['fold'] != fold]
        test = [c for c in cells if c['fold'] == fold]
        train_ids = {c['meta']['data_id'] for c in train}
        test_ids = {c['meta']['data_id'] for c in test}
        assert train_ids.isdisjoint(test_ids)
        inner = min(c['fold'] for c in train)
        if not out.exists():
            A, y, w = design(train, 'regression')
            ges, history = fit_ges(A, y, w, 'regression')
            tr = design([c for c in train if c['fold'] != inner], 'regression')
            val = design([c for c in train if c['fold'] == inner], 'regression')
            candidates = []
            for reg in [0., .001, .01, .1]:
                b = fit_convex(*tr, 'regression', reg)
                candidates.append((score(*val, b, 'regression'), reg))
            regularization = min(candidates)[1]
            convex = fit_convex(A, y, w, 'regression', regularization)
            atomic_json(out, {'status': 'COMPLETE', 'seed': 0, 'fold': fold,
                'access_regime': 'canonical_source_only', 'train_data_ids': sorted(train_ids),
                'test_data_ids': sorted(test_ids), 'fold_map_source': mapping_source,
                'inner_validation_fold': inner, 'inner_validation_scores': candidates,
                'regularization': regularization, 'ges': history,
                'ges_weights': ges.tolist(), 'convex_weights': convex.tolist(),
                'outer_test_labels_used_for_fit_or_selection': False,
                'Shared_or_paper_scores_used_for_selection': False, 'Shared_run': False})
        policy = json.loads(out.read_text())
        for method, key in [('GES-6', 'ges_weights'), ('Convex Stacking-6', 'convex_weights')]:
            beta = np.asarray(policy[key])
            assert np.all(beta >= 0) and abs(beta.sum()-1) < 1e-9
            for c in test:
                pred = beta @ c['bank']
                records.append(c['meta'] | {'method': method, 'outer_fold': fold} |
                               metrics(c['y'], pred, 'regression', max(float(c['y'].std()), 1e-12)))
        print('CALIBRATED', fold, flush=True)
    methods = sorted({r['method'] for r in records})
    means = []
    for method in methods:
        item = {'method': method}
        for name, nonzero, count in [('clean', False, 33), ('seven', True, 924)]:
            rows = [r for r in records if r['method'] == method and (r['rho'] > 0) == nonzero]
            assert len(rows) == count
            assert len({(r['task_id'],r['family'],r['rho']) for r in rows}) == count
            item[name] = {k: math.fsum(r[k] for r in rows)/len(rows) for k in ['r2','nrmse']}
        means.append(item)
    atomic_json(RUN/'results'/f'{regime}_regression_summary.json', {
        'status':'COMPLETE', 'seed':0, 'datasets':33, 'conditions':957, 'source_rows':256,
        'candidate_budget':6, 'Shared_run':False, 'original_full_pool_results_retained':True,
        'protocol_changed':False, 'access_regime':'canonical Base source rows only',
        'historical_Shared_not_same_regime':True, 'averages':means,
        'resampling_policy':'ordinary bootstrap' if regime=='canonical_input' else 'fixed 10% source-row replacement',
        'audit':{'all_raw_metrics_recomputed':True, 'all_IQR_selections_checked':True,
                 'OOF_folds_disjoint':True, 'all_complete_grids':True}})
    names = {'base_protocol_check':'Base', 'native_6':'Native-6', 'bootstrap_mean_6':'Random-6',
             'oligoicp_iqr_6_keep1_adapted':'OligoICP 6→1（适配）'}
    marker='CANONICAL_INPUT_REGRESSION' if regime=='canonical_input' else 'CANONICAL_PARTIAL_010_REGRESSION'
    lines = ['<!-- '+marker+' -->', '',
        '七偏移回归的输入匹配控制：与Base使用完全相同的256行源样本，成员上限6，种子0；33数据集×28非零条件完整宏平均。',
        '重采样方式：'+('标准Bootstrap。' if regime=='canonical_input' else '每个成员固定替换10%的行，其余源行原样保留；没有特征或标签适配。'),
        '与先前可使用额外历史/参考样本的完整样本池组分开报告；此处没有修改偏移生成器，Shared未运行。','',
        '| 方法 | 七偏移 R²↑ / nRMSE↓ |', '|---|---:|']
    for m in means:
        v = m['seven']
        lines.append(f"| {names.get(m['method'],m['method'])} | {v['r2']:.3f} / {v['nrmse']:.3f} |")
    lines += ['', '该控制只隔离额外训练样本访问带来的收益；不能把它与历史Shared分数视为同输入实测比较。',
              '<!-- END_'+marker+' -->', '']
    doc = ROOT/'REBUTTAL_EXPERIMENTS.md'
    text = read_markdown(doc)
    start, end = '<!-- '+marker+' -->', '<!-- END_'+marker+' -->'
    section = '\n'.join(lines)
    if start in text:
        before, rest = text.split(start, 1)
        _, after = rest.split(end, 1)
        text = before+section+after
    else:
        title, rest = text.split('\n',1)
        text = title+'\n\n'+section+'\n'+rest
    write_markdown(text,doc)
    print(json.dumps(means), flush=True)


if __name__ == '__main__':
    main()
