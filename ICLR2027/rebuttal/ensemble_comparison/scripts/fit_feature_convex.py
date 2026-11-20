"""Shift-equivariant convex fusion of the same six feature-view predictions.

The previous unconstrained NNLS diagnostic is preserved. This adaptation fits
simplex weights with the same nested dataset-OOF calibration as bootstrap
stacking. It cannot obtain information from the held-out dataset labels.
"""
import json
import numpy as np
from fit_feature_nnls import load_cells
from fit_convex_combiners import design,fit_convex,score
from prepare_data import RUN,atomic_json
from run_tabicl import metrics


def main():
    for track,expected in [('five',735),('seven_calibrated_v2',957)]:
        cells,source=load_cells(track,'regression');assert len(cells)==expected
        for fold in range(5):
            out=RUN/'results/combiners'/track/'regression'/f'fold_{fold}_feature_convex.json'
            if out.exists():continue
            train=[c for c in cells if c['fold']!=fold];test=[c for c in cells if c['fold']==fold]
            train_ids={c['meta']['data_id'] for c in train};test_ids={c['meta']['data_id'] for c in test}
            assert train_ids.isdisjoint(test_ids)
            inner=min(c['fold'] for c in train)
            inner_train=design([c for c in train if c['fold']!=inner],'regression')
            inner_validation=design([c for c in train if c['fold']==inner],'regression')
            scores=[]
            for regularization in [0.,.001,.01,.1]:
                weights=fit_convex(*inner_train,'regression',regularization)
                scores.append((score(*inner_validation,weights,'regression'),regularization))
            best=min(scores)[1];weights=fit_convex(*design(train,'regression'),'regression',best);rows=[]
            for c in test:
                prediction=np.einsum('m,mn->n',weights,c['bank'])
                scale=max(float(c['y'].std()),1e-12) if track.startswith('seven') else c['meta']['scale']
                observed=metrics(c['y'],prediction,'regression',scale)
                rows.append(c['meta']|observed|{'method':'Feature_views6_convex_OOF_adapted','backbone':'tabicl','outer_fold':fold})
            atomic_json(out,{'status':'COMPLETE','track':track,'task':'regression','backbone':'tabicl','seed':0,
                'method':'Feature_views6_convex_OOF_adapted','weights':weights.tolist(),'inner_validation':scores,
                'regularization':best,'train_data_ids':sorted(train_ids),'test_data_ids':sorted(test_ids),
                'held_out_labels_used_for_fit':False,'fold_map_source':source,
                'adaptation':'same six source-fitted feature views; simplex fusion rather than unconstrained regression NNLS',
                'old_unconstrained_NNLS_results_preserved':True,'rows':rows})
            print('FEATURE_CONVEX_DONE',track,fold,weights,flush=True)


if __name__=='__main__':main()
