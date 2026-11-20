"""Recover row construction by archived input hashes, never by model scores."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from prepare_data import ROOT, RUN, LOCAL_CC18, protocol, atomic_json


def old_sha(a):
    a=np.ascontiguousarray(a)
    h=hashlib.sha256();h.update(str(a.dtype).encode())
    h.update(np.asarray(a.shape,dtype=np.int64).tobytes());h.update(a.tobytes())
    return h.hexdigest()


def main():
    folder=LOCAL_CC18/'3918'
    frame=pd.read_parquet(folder/'X.parquet');labels=pd.read_parquet(folder/'y.parquet').iloc[:,0]
    valid=~labels.isna().to_numpy();X,_=protocol._numeric_matrix(frame.loc[valid]);y=LabelEncoder().fit_transform(labels.loc[valid])
    record=json.loads((ROOT/'fair_budget_experiments/limix_2m_ovr/results/paper_run_seven/cells/classification/dataset_1068_covariate_s0.json').read_text())
    expected=record['target_features_sha256'];n=len(y);found=[];tested=0

    def check(ids, description):
        nonlocal tested
        ids=np.asarray(ids,dtype=int)
        if len(ids)!=128:return
        tested+=1
        if old_sha(np.asarray(X[ids],dtype=np.float32))==expected:
            found.append({'constructor':description,'target_ids':ids.tolist(),
                          'feature_hash_exact':True,'target_label_hash_exact':old_sha(y[ids])==record['evaluation_target_sha256']})
            print('MATCH',description,flush=True)

    ids=np.arange(n)
    seeds=sorted(set(range(101))|{123,448,640,1024,1068,1234,1729,2024,2025,2026,2718,3918,31415})
    for seed in seeds:
        for mode in ['default_permutation','legacy_permutation','default_choice','legacy_choice']:
            if mode=='default_permutation':order=np.random.default_rng(seed).permutation(n)
            elif mode=='legacy_permutation':order=np.random.RandomState(seed).permutation(n)
            elif mode=='default_choice':order=np.random.default_rng(seed).choice(n,448,replace=False)
            else:order=np.random.RandomState(seed).choice(n,448,replace=False)
            for start in [0,64,128,192,256,320,len(order)-128]:
                check(order[start:start+128],{'mode':mode,'seed':seed,'start':start})
            for stratify in [True,False]:
                for ordering in ['draw_order','sorted_rows']:
                    subset=order[:448] if ordering=='draw_order' else np.sort(order[:448])
                    for train_n,test_n in [(320,128),(256,192)]:
                        a,b=train_test_split(subset,train_size=train_n,test_size=test_n,
                            random_state=seed,stratify=y[subset] if stratify else None)
                        if test_n==192:
                            _,b=train_test_split(b,train_size=64,test_size=128,random_state=seed,
                                                stratify=y[b] if stratify else None)
                        check(b,{'mode':'random_cap_then_split','cap_mode':mode,'ordering':ordering,
                                 'seed':seed,'stratify':stratify,'first_train_n':train_n})
        for stratify in [True,False]:
            for train_n in [None,256,320]:
                a,b=train_test_split(ids,train_size=train_n,test_size=128,random_state=seed,stratify=y if stratify else None)
                check(b,{'mode':'full_train_test_split','seed':seed,'stratify':stratify,'train_n':train_n})
            for cap in [448,640,1024]:
                if cap>=n:continue
                capped,_=train_test_split(ids,train_size=cap,stratify=y if stratify else None,random_state=seed)
                a,b=train_test_split(capped,train_size=320,test_size=128,stratify=y[capped] if stratify else None,random_state=seed)
                check(b,{'mode':'capped_train_test_split','seed':seed,'stratify':stratify,'cap':cap})
            for train_n,test_n in [(256,192),(None,192),(64,384)]:
                a,b=train_test_split(ids,train_size=train_n,test_size=test_n,
                                    random_state=seed,stratify=y if stratify else None)
                _,q=train_test_split(b,train_size=test_n-128,test_size=128,
                                    random_state=seed,stratify=y[b] if stratify else None)
                check(q,{'mode':'full_two_stage_split','seed':seed,'stratify':stratify,
                         'first_train_n':train_n,'first_test_n':test_n})
    out={'dataset':'pc1','data_id':1068,'task_id':3918,'selection_criterion':'exact archived input SHA256',
         'target_features_sha256_expected':expected,'candidate_row_orders_checked':tested,'seeds_checked':seeds,
         'model_metrics_used':False,'matches':found,'exact_split_recovered':bool(found)}
    atomic_json(RUN/'results/seven_prior_preserving_v1/historical_split_hash_probe.json',out)
    print(json.dumps({k:v for k,v in out.items() if k!='matches'}),flush=True)


if __name__=='__main__':main()
