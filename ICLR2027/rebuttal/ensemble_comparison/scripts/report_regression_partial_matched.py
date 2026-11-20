"""Verify and summarize paired resampling controls to JSON/CSV only."""
import csv
import json
import math
from collections import Counter
import numpy as np
from prepare_data import RUN,atomic_json
from run_canonical_regression import fingerprint
from run_tabicl import metrics


def main():
    summaries=[];paired={};checked=0;hashes={}
    for backbone in ['tabicl','tabpfn']:
        for track,n,families in [('five',35,5),('seven_calibrated_v2',33,7)]:
            root=(RUN/'results/canonical_partial_010/tabicl'/track/'regression'
                  if backbone=='tabicl' and track.startswith('seven') else
                  RUN/'results/regression_partial_matched'/backbone/track/'regression')
            paths=sorted(root.glob('*/*.npz'))
            assert len(paths)==n*(1+4*families)
            rows=[];sizes=[]
            for path in paths:
                meta=json.loads(path.with_suffix('.json').read_text())
                count=meta.get('rows_per_context',meta.get('context_rows'))
                assert meta['resampling_fraction']==.1 and count<=2048
                assert meta.get('context_count',meta.get('candidate_count'))==6
                if track.startswith('seven'):assert count==256
                sizes.append(count)
                if count not in hashes:
                    hashes[count]=[]
                    for i in range(6):
                        rng=np.random.default_rng(1009*(i+1));ids=np.arange(count)
                        pos=rng.choice(count,size=int(np.ceil(.1*count)),replace=False)
                        ids[pos]=rng.choice(count,size=len(pos),replace=True)
                        hashes[count].append(fingerprint(ids))
                assert meta['bootstrap_indices_sha256']==hashes[count]
                with np.load(path) as f:
                    bank,y=f['bootstrap_bank'],f['y_target']
                    base=f['prediction__base_protocol_check']
                    assert bank.shape==(6,len(y)) and np.isfinite(bank).all()
                    mean=bank.mean(0)
                    key='prediction__bootstrap_mean_6' if 'prediction__bootstrap_mean_6' in f else 'prediction__bootstrap_mean_6_partial010'
                    assert np.array_equal(mean,f[key])
                source_hashes=tuple(meta.get('current_'+key,meta.get(key)) for key in ['source_X_sha256','source_y_sha256','query_X_sha256'])
                assert all(source_hashes)
                pair_key=(track,meta['task_id'],meta['family'],meta['rho'])
                if pair_key in paired:
                    assert paired[pair_key][0]==source_hashes
                    assert np.array_equal(paired[pair_key][1],y)
                else:paired[pair_key]=(source_hashes,y)
                scale=max(float(y.std()),1e-12) if track.startswith('seven') else meta['scale']
                observed={k:metrics(y,p,'regression',scale) for k,p in [('Base',base),('Random_partial010',mean)]}
                rows.append({'task_id':meta['task_id'],'family':meta['family'],'rho':meta['rho'],'methods':observed})
                checked+=1
            record={'backbone':backbone,'track':track,'seed':0,'conditions':len(paths),
                'datasets':n,'context_count':6,'rows_per_context_distribution':dict(Counter(sizes)),
                'canonical_source_only':True,'replacement_fraction':.1,'Shared_run':False,
                'source':str(root.relative_to(RUN))}
            for block,nonzero,expected in [('clean',False,n),('shift',True,n*4*families)]:
                selected=[r for r in rows if (r['rho']>0)==nonzero]
                assert len(selected)==expected
                assert len({(r['task_id'],r['family'],r['rho']) for r in selected})==expected
                record[block]={name:{k:math.fsum(r['methods'][name][k] for r in selected)/expected
                                     for k in ['r2','nrmse']} for name in ['Base','Random_partial010']}
            a,b=record['clean']['Random_partial010'],record['clean']['Base']
            record['clean_within_005']=a['r2']>=b['r2']-.005 and a['nrmse']<=b['nrmse']+.005
            record['shift_difference_vs_Base']={k:record['shift']['Random_partial010'][k]-record['shift']['Base'][k] for k in ['r2','nrmse']}
            record['shift_improves_both_metrics']=all(record['shift_difference_vs_Base'][k]*(1 if k=='r2' else -1)>=0 for k in ['r2','nrmse'])
            summaries.append(record)
    assert checked==3384 and len(paired)==1692
    out={'status':'COMPLETE','seed':0,'context_count':6,'resampling_fraction':.1,
        'comparison_role':'Additional Random-6 partial-resampling control; not a published method default',
        'current_controls':summaries,'verified_condition_files':checked,
        'paired_backbone_inputs_verified':len(paired),'normalization':'source std in five; evaluation-only target std in seven',
        'original_and_all_previous_variants_retained':True,'score_ordering_bounds_used_for_fitting':False,
        'Shared_run':False,'Markdown_written':False}
    atomic_json(RUN/'results/regression_partial_matched_summary.json',out)
    flat=[]
    for r in summaries:
        for block in ['clean','shift']:
            for method in ['Base','Random_partial010']:
                flat.append({'backbone':r['backbone'],'track':r['track'],'block':block,'method':method}|r[block][method])
    with (RUN/'results/regression_partial_matched_averages.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    print(json.dumps({'status':'COMPLETE','audit_condition_files':checked,'paired_conditions':len(paired),
                      'current_controls':summaries,'Markdown_written':False},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
